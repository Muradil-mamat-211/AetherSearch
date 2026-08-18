"""
IGPO Vectorized Ground Truth Log Probability Computation

Optimizes T compute_log_prob calls into a single batched call.

How it works:
- Original: Call compute_log_prob in each turn, resulting in T calls total
- Vectorized: Collect data from all turns, batch call compute_log_prob once after the loop

Efficiency improvements:
- Reduce Ray remote call count (from T to 1)
- Reduce data transfer overhead
- Higher GPU computation efficiency with batching

Enable in train.sh:
    +algorithm.use_vectorized_gt_logprob=true
"""

import os
import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
import copy
import math

# ============================================================================
# Global Switch
# ============================================================================
_VECTORIZED_ENABLED = None  # None = not initialized, True/False = set


def is_vectorized_enabled() -> bool:
    """Check if vectorized computation is enabled"""
    global _VECTORIZED_ENABLED
    if _VECTORIZED_ENABLED is None:
        # Read from environment variable
        env_val = os.environ.get('IGPO_USE_VECTORIZED_GT_LOGPROB', '').lower()
        _VECTORIZED_ENABLED = env_val in ('true', '1', 'yes')
        # Print status on first initialization
        print(f"[IGPO] Vectorized GT LogProb: {'ENABLED' if _VECTORIZED_ENABLED else 'DISABLED (default)'}")
    return _VECTORIZED_ENABLED


def set_vectorized_enabled(enabled: bool):
    """Set whether vectorized computation is enabled"""
    global _VECTORIZED_ENABLED
    _VECTORIZED_ENABLED = enabled
    print(f"[IGPO] Vectorized GT LogProb: {'ENABLED' if enabled else 'DISABLED'}")


@dataclass
class VectorizedGTConfig:
    """Configuration for vectorized GT log prob computation."""
    pad_token_id: int
    eos_token_id: int
    # Whether to validate results against sequential computation (slower but safe)
    validate_results: bool = False
    # Tolerance for validation comparison
    validation_rtol: float = 1e-4
    validation_atol: float = 1e-6


class VectorizedGTLogProbComputer:
    """
    Computes ground truth log probabilities for all turns in a single forward pass.
    
    The key insight is that we can append all GT copies to the sequence end and use
    an extended attention mask where GT_t can only attend to tokens up to turn_t.
    
    Extended Sequence Structure:
    [Original Sequence | GT_0 | GT_1 | GT_2 | ... | GT_{T-1}]
    
    Where:
    - GT_0: Can only see tokens up to prompt_end (computes P(GT | Prompt))
    - GT_1: Can only see tokens up to turn1_end (computes P(GT | Prompt + Turn1 + Obs1))
    - GT_t: Can only see tokens up to turn_t_end
    """
    
    def __init__(self, tokenizer, config: VectorizedGTConfig):
        """
        Args:
            tokenizer: HuggingFace tokenizer
            config: VectorizedGTConfig instance
        """
        self.tokenizer = tokenizer
        self.config = config
        self.pad_token_id = config.pad_token_id
        self.eos_token_id = config.eos_token_id
    
    def tokenize_ground_truth(self, ground_truth_text: str) -> torch.Tensor:
        """
        Tokenize ground truth text with proper prefix and suffix.
        
        Args:
            ground_truth_text: The ground truth answer text
            
        Returns:
            gt_tokens: Tensor of shape (gt_len,)
        """
        PREFIX = "\nNow there's enough information to answer\n</think>\n<answer>\n"
        SUFFIX = "\n</answer><|im_end|>"
        
        full_text = f"{PREFIX}{ground_truth_text}{SUFFIX}"
        encoding = self.tokenizer(full_text, return_tensors="pt", add_special_tokens=False)
        gt_tokens = encoding['input_ids'].squeeze(0)  # (gt_len,)
        
        return gt_tokens
    
    def get_gt_answer_token_range(self, ground_truth_text: str) -> Tuple[int, int]:
        """
        Get the token range of the actual answer within the GT sequence.
        Uses offset_mapping for precise boundary detection.
        
        Args:
            ground_truth_text: The ground truth answer text
            
        Returns:
            (start_idx, end_idx): Token indices of the answer portion
        """
        PREFIX = "\nNow there's enough information to answer\n</think>\n<answer>\n"
        SUFFIX = "\n</answer><|im_end|>"
        
        full_text = f"{PREFIX}{ground_truth_text}{SUFFIX}"
        encoding = self.tokenizer(full_text, return_tensors="pt", return_offsets_mapping=True)
        offset_mapping = encoding['offset_mapping'].squeeze(0).tolist()
        
        gt_char_start = len(PREFIX)
        gt_char_end = len(PREFIX) + len(ground_truth_text)
        
        gt_token_start = None
        gt_token_end = None
        
        for token_idx, (char_start, char_end) in enumerate(offset_mapping):
            if gt_token_start is None and char_end > gt_char_start:
                gt_token_start = token_idx
            if char_start < gt_char_end and char_end > 0:
                gt_token_end = token_idx + 1
        
        if gt_token_start is None:
            gt_token_start = len(offset_mapping)
        if gt_token_end is None:
            gt_token_end = len(offset_mapping)
        
        return gt_token_start, gt_token_end
    
    def build_extended_sequence(
        self,
        original_input_ids: torch.Tensor,
        gt_tokens: torch.Tensor,
        num_gt_copies: int
    ) -> Tuple[torch.Tensor, List[int]]:
        """
        Build extended sequence by appending GT copies to the original sequence.
        
        Args:
            original_input_ids: (seq_len,) - Original sequence tokens
            gt_tokens: (gt_len,) - Ground truth tokens
            num_gt_copies: Number of GT copies to append (= number of turns)
            
        Returns:
            extended_input_ids: (extended_len,) - Extended sequence
            gt_start_positions: List of starting positions for each GT copy
        """
        seq_len = original_input_ids.shape[0]
        gt_len = gt_tokens.shape[0]
        extended_len = seq_len + num_gt_copies * gt_len
        
        device = original_input_ids.device
        extended_input_ids = torch.full((extended_len,), self.pad_token_id, 
                                         dtype=original_input_ids.dtype, device=device)
        
        # Copy original sequence
        extended_input_ids[:seq_len] = original_input_ids
        
        # Append GT copies
        gt_start_positions = []
        for t in range(num_gt_copies):
            start_pos = seq_len + t * gt_len
            extended_input_ids[start_pos:start_pos + gt_len] = gt_tokens.to(device)
            gt_start_positions.append(start_pos)
        
        return extended_input_ids, gt_start_positions
    
    def build_extended_attention_mask(
        self,
        original_attention_mask: torch.Tensor,
        turn_end_positions: List[int],
        gt_len: int,
        num_gt_copies: int
    ) -> torch.Tensor:
        """
        Build extended 4D attention mask for the vectorized computation.
        
        CRITICAL: This mask ensures that GT_t can ONLY attend to:
        1. Original sequence tokens from position 0 to turn_end_positions[t] - 1
        2. Its own GT tokens (causally, i.e., token i can see tokens 0 to i)
        3. GT_t CANNOT see any other GT_s (s != t)
        
        Args:
            original_attention_mask: (seq_len,) - Original 1D attention mask
            turn_end_positions: List of positions where each turn ends
                                [prompt_end, turn1_end, turn2_end, ...]
            gt_len: Length of each GT copy
            num_gt_copies: Number of GT copies (should equal len(turn_end_positions))
            
        Returns:
            extended_mask: (1, 1, extended_len, extended_len) - 4D attention mask
                          Values: 0 = attend, -inf = mask out (for additive mask)
                          Or: 1 = attend, 0 = mask out (for multiplicative mask)
        """
        assert num_gt_copies == len(turn_end_positions), \
            f"num_gt_copies ({num_gt_copies}) must equal len(turn_end_positions) ({len(turn_end_positions)})"
        
        seq_len = original_attention_mask.shape[0]
        extended_len = seq_len + num_gt_copies * gt_len
        device = original_attention_mask.device
        
        # Initialize mask as all zeros (will use 1 = attend, 0 = mask)
        mask = torch.zeros(extended_len, extended_len, dtype=torch.float32, device=device)
        
        # Part 1: Original sequence - standard causal mask
        # Token at position q can attend to positions 0 to q (inclusive)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        # Also apply original attention mask (handle padding)
        orig_mask_2d = original_attention_mask.unsqueeze(0).expand(seq_len, -1)  # (seq_len, seq_len)
        mask[:seq_len, :seq_len] = causal_mask * orig_mask_2d
        
        # Part 2: GT attention masks
        for t in range(num_gt_copies):
            gt_start = seq_len + t * gt_len
            gt_end = gt_start + gt_len
            turn_end = turn_end_positions[t]
            
            # GT_t can attend to original sequence positions [0, turn_end)
            # Apply original attention mask for these positions
            for i in range(gt_len):
                query_pos = gt_start + i
                # Can see original tokens up to turn_end
                mask[query_pos, :turn_end] = original_attention_mask[:turn_end].float()
                
                # Can see its own GT tokens causally (positions gt_start to gt_start + i)
                mask[query_pos, gt_start:gt_start + i + 1] = 1.0
        
        # Reshape to 4D: (1, 1, extended_len, extended_len)
        mask = mask.unsqueeze(0).unsqueeze(0)
        
        return mask
    
    def build_extended_position_ids(
        self,
        original_position_ids: torch.Tensor,
        turn_end_positions: List[int],
        gt_len: int,
        num_gt_copies: int
    ) -> torch.Tensor:
        """
        Build extended position IDs for the vectorized computation.
        
        CRITICAL: Each GT_t's position IDs should continue from where turn_t ends.
        This ensures correct rotary position embeddings (RoPE).
        
        Args:
            original_position_ids: (seq_len,) - Original position IDs
            turn_end_positions: List of positions where each turn ends
            gt_len: Length of each GT copy
            num_gt_copies: Number of GT copies
            
        Returns:
            extended_position_ids: (extended_len,)
        """
        seq_len = original_position_ids.shape[0]
        extended_len = seq_len + num_gt_copies * gt_len
        device = original_position_ids.device
        
        extended_position_ids = torch.zeros(extended_len, dtype=torch.long, device=device)
        
        # Copy original position IDs
        extended_position_ids[:seq_len] = original_position_ids
        
        # Set position IDs for each GT copy
        for t in range(num_gt_copies):
            gt_start = seq_len + t * gt_len
            turn_end = turn_end_positions[t]
            
            # GT_t's positions continue from turn_end
            # Get the position value at turn_end - 1 and continue from there
            if turn_end > 0:
                base_position = original_position_ids[turn_end - 1].item() + 1
            else:
                base_position = 0
            
            extended_position_ids[gt_start:gt_start + gt_len] = torch.arange(
                base_position, base_position + gt_len, dtype=torch.long, device=device
            )
        
        return extended_position_ids
    
    def compute_log_probs_single_sequence(
        self,
        model,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        response_start: int,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Compute log probabilities for a single sequence using the model.
        
        Args:
            model: The language model
            input_ids: (1, seq_len) - Input token IDs
            attention_mask: (1, 1, seq_len, seq_len) or (1, seq_len) - Attention mask
            position_ids: (1, seq_len) - Position IDs
            response_start: Starting position of the response (GT part)
            temperature: Temperature for scaling logits
            
        Returns:
            log_probs: (response_len,) - Log probabilities for each response token
        """
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                # Forward pass
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=False
                )
                
                logits = outputs.logits  # (1, seq_len, vocab_size)
                logits = logits / temperature
                
                # Get log probabilities for the response part
                # For position i, we predict token at position i+1
                # So logits[:, response_start-1:-1] predicts tokens at positions response_start to end
                response_len = input_ids.shape[1] - response_start
                pred_logits = logits[:, response_start-1:-1, :]  # (1, response_len, vocab_size)
                target_tokens = input_ids[:, response_start:]  # (1, response_len)
                
                # Compute log probabilities
                log_probs = F.log_softmax(pred_logits, dim=-1)  # (1, response_len, vocab_size)
                log_probs = log_probs.gather(dim=-1, index=target_tokens.unsqueeze(-1)).squeeze(-1)
                
                return log_probs.squeeze(0)  # (response_len,)
    
    def _check_4d_attention_support(self, model) -> bool:
        """
        Check if the model supports 4D attention mask.
        
        FlashAttention 2 does not support arbitrary 4D attention masks,
        only standard causal masks or simple padding masks.
        
        Returns:
            True if 4D attention mask is supported, False otherwise.
        """
        # Check for FlashAttention 2 indicators
        model_config = getattr(model, 'config', None)
        if model_config is not None:
            # Check _attn_implementation attribute (HuggingFace transformers >= 4.36)
            attn_impl = getattr(model_config, '_attn_implementation', None)
            if attn_impl == 'flash_attention_2':
                return False
            
            # Check use_flash_attention_2 flag (older versions)
            if getattr(model_config, 'use_flash_attention_2', False):
                return False
        
        # Check environment variable for VLLM backend
        vllm_backend = os.environ.get('VLLM_ATTENTION_BACKEND', '').upper()
        if vllm_backend == 'FLASH_ATTN':
            return False
        
        return True
    
    def compute_all_turns_vectorized(
        self,
        model,
        original_input_ids: torch.Tensor,
        original_attention_mask: torch.Tensor,
        original_position_ids: torch.Tensor,
        ground_truth_text: str,
        turn_end_positions: List[int],
        temperature: float = 1.0
    ) -> Tuple[List[torch.Tensor], List[Tuple[int, int]]]:
        """
        Compute GT log probabilities for all turns in a SINGLE forward pass.
        
        This is the main vectorized computation method.
        
        Note: If the model uses FlashAttention 2 (which doesn't support 4D masks),
        this method will automatically fall back to sequential computation.
        
        Args:
            model: The language model
            original_input_ids: (seq_len,) - Original sequence token IDs
            original_attention_mask: (seq_len,) - Original 1D attention mask
            original_position_ids: (seq_len,) - Original position IDs
            ground_truth_text: The ground truth answer text
            turn_end_positions: List of positions where each turn ends
                               [prompt_end, turn1_end, turn2_end, ...]
            temperature: Temperature for scaling logits
            
        Returns:
            gt_log_probs_per_turn: List of log prob tensors, one per turn
            gt_answer_ranges: List of (start, end) tuples for the answer portion in each GT
        """
        num_turns = len(turn_end_positions)
        if num_turns == 0:
            return [], []
        
        # Check if model supports 4D attention mask
        if not self._check_4d_attention_support(model):
            print("[IGPO] FlashAttention 2 detected, falling back to sequential GT logprob computation")
            return self.compute_all_turns_sequential(
                model, original_input_ids, original_attention_mask, original_position_ids,
                ground_truth_text, turn_end_positions, temperature
            )
        
        device = original_input_ids.device
        seq_len = original_input_ids.shape[0]
        
        # 1. Tokenize ground truth
        gt_tokens = self.tokenize_ground_truth(ground_truth_text)
        gt_len = gt_tokens.shape[0]
        gt_answer_start, gt_answer_end = self.get_gt_answer_token_range(ground_truth_text)
        
        # 2. Build extended sequence
        extended_input_ids, gt_start_positions = self.build_extended_sequence(
            original_input_ids, gt_tokens, num_turns
        )
        
        # 3. Build extended attention mask (4D)
        extended_attention_mask = self.build_extended_attention_mask(
            original_attention_mask, turn_end_positions, gt_len, num_turns
        )
        
        # 4. Build extended position IDs
        extended_position_ids = self.build_extended_position_ids(
            original_position_ids, turn_end_positions, gt_len, num_turns
        )
        
        # 5. Single forward pass
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                # Add batch dimension
                input_ids_batch = extended_input_ids.unsqueeze(0)  # (1, extended_len)
                position_ids_batch = extended_position_ids.unsqueeze(0)  # (1, extended_len)
                
                # For 4D attention mask, convert to additive mask if needed
                # Most HuggingFace models expect: 0 = attend, large negative = mask
                # But some use multiplicative: 1 = attend, 0 = mask
                # We'll convert our multiplicative mask to additive format
                attn_mask_additive = torch.where(
                    extended_attention_mask == 1,
                    torch.tensor(0.0, device=device),
                    torch.tensor(-10000.0, device=device)
                )
                
                outputs = model(
                    input_ids=input_ids_batch,
                    attention_mask=attn_mask_additive,
                    position_ids=position_ids_batch,
                    use_cache=False
                )
                
                logits = outputs.logits / temperature  # (1, extended_len, vocab_size)
        
        # 6. Extract log probabilities for each GT copy
        gt_log_probs_per_turn = []
        gt_answer_ranges = []
        
        for t in range(num_turns):
            gt_start = gt_start_positions[t]
            
            # Get logits for predicting GT tokens
            # logits[:, gt_start-1:gt_start+gt_len-1] predicts tokens at positions gt_start to gt_start+gt_len-1
            pred_logits = logits[:, gt_start-1:gt_start+gt_len-1, :]  # (1, gt_len, vocab_size)
            target_tokens = extended_input_ids[gt_start:gt_start+gt_len].unsqueeze(0)  # (1, gt_len)
            
            # Compute log probabilities
            log_probs = F.log_softmax(pred_logits, dim=-1)
            log_probs = log_probs.gather(dim=-1, index=target_tokens.unsqueeze(-1)).squeeze(-1).squeeze(0)
            # log_probs shape: (gt_len,)
            
            gt_log_probs_per_turn.append(log_probs)
            gt_answer_ranges.append((gt_answer_start, gt_answer_end))
        
        return gt_log_probs_per_turn, gt_answer_ranges
    
    def compute_all_turns_sequential(
        self,
        model,
        original_input_ids: torch.Tensor,
        original_attention_mask: torch.Tensor,
        original_position_ids: torch.Tensor,
        ground_truth_text: str,
        turn_end_positions: List[int],
        temperature: float = 1.0
    ) -> Tuple[List[torch.Tensor], List[Tuple[int, int]]]:
        """
        Compute GT log probabilities sequentially (original method).
        Used for validation against the vectorized version.
        
        Args:
            Same as compute_all_turns_vectorized
            
        Returns:
            Same as compute_all_turns_vectorized
        """
        num_turns = len(turn_end_positions)
        if num_turns == 0:
            return [], []
        
        device = original_input_ids.device
        
        # Tokenize ground truth
        gt_tokens = self.tokenize_ground_truth(ground_truth_text)
        gt_len = gt_tokens.shape[0]
        gt_answer_start, gt_answer_end = self.get_gt_answer_token_range(ground_truth_text)
        
        gt_log_probs_per_turn = []
        gt_answer_ranges = []
        
        for t, turn_end in enumerate(turn_end_positions):
            # Build sequence for this turn: [original[:turn_end] | GT]
            context = original_input_ids[:turn_end]
            seq_input_ids = torch.cat([context, gt_tokens.to(device)], dim=0)
            
            # Build attention mask
            context_mask = original_attention_mask[:turn_end]
            gt_mask = torch.ones(gt_len, dtype=context_mask.dtype, device=device)
            seq_attention_mask = torch.cat([context_mask, gt_mask], dim=0)
            
            # Build position IDs
            context_pos = original_position_ids[:turn_end]
            if turn_end > 0:
                base_pos = context_pos[-1].item() + 1
            else:
                base_pos = 0
            gt_pos = torch.arange(base_pos, base_pos + gt_len, dtype=torch.long, device=device)
            seq_position_ids = torch.cat([context_pos, gt_pos], dim=0)
            
            # Compute log probs for this turn
            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    outputs = model(
                        input_ids=seq_input_ids.unsqueeze(0),
                        attention_mask=seq_attention_mask.unsqueeze(0),
                        position_ids=seq_position_ids.unsqueeze(0),
                        use_cache=False
                    )
                    
                    logits = outputs.logits / temperature
                    
                    # Get log probs for GT part
                    gt_start_in_seq = turn_end
                    pred_logits = logits[:, gt_start_in_seq-1:gt_start_in_seq+gt_len-1, :]
                    target_tokens = seq_input_ids[gt_start_in_seq:gt_start_in_seq+gt_len].unsqueeze(0)
                    
                    log_probs = F.log_softmax(pred_logits, dim=-1)
                    log_probs = log_probs.gather(dim=-1, index=target_tokens.unsqueeze(-1)).squeeze(-1).squeeze(0)
            
            gt_log_probs_per_turn.append(log_probs)
            gt_answer_ranges.append((gt_answer_start, gt_answer_end))
        
        return gt_log_probs_per_turn, gt_answer_ranges
    
    def validate_vectorized_vs_sequential(
        self,
        model,
        original_input_ids: torch.Tensor,
        original_attention_mask: torch.Tensor,
        original_position_ids: torch.Tensor,
        ground_truth_text: str,
        turn_end_positions: List[int],
        temperature: float = 1.0
    ) -> Tuple[bool, Dict]:
        """
        Validate that vectorized and sequential implementations produce the same results.
        
        Returns:
            is_valid: True if results match within tolerance
            details: Dictionary with comparison details
        """
        # Compute using both methods
        vec_results, vec_ranges = self.compute_all_turns_vectorized(
            model, original_input_ids, original_attention_mask, original_position_ids,
            ground_truth_text, turn_end_positions, temperature
        )
        
        seq_results, seq_ranges = self.compute_all_turns_sequential(
            model, original_input_ids, original_attention_mask, original_position_ids,
            ground_truth_text, turn_end_positions, temperature
        )
        
        # Compare results
        is_valid = True
        details = {
            'num_turns': len(turn_end_positions),
            'turn_comparisons': []
        }
        
        for t in range(len(turn_end_positions)):
            vec_lp = vec_results[t]
            seq_lp = seq_results[t]
            
            # Compare
            max_abs_diff = torch.max(torch.abs(vec_lp - seq_lp)).item()
            mean_abs_diff = torch.mean(torch.abs(vec_lp - seq_lp)).item()
            is_close = torch.allclose(vec_lp, seq_lp, rtol=self.config.validation_rtol, 
                                       atol=self.config.validation_atol)
            
            turn_detail = {
                'turn': t,
                'turn_end_position': turn_end_positions[t],
                'max_abs_diff': max_abs_diff,
                'mean_abs_diff': mean_abs_diff,
                'is_close': is_close,
                'vec_mean_logprob': vec_lp.mean().item(),
                'seq_mean_logprob': seq_lp.mean().item()
            }
            details['turn_comparisons'].append(turn_detail)
            
            if not is_close:
                is_valid = False
        
        details['is_valid'] = is_valid
        return is_valid, details


class VectorizedGTLogProbWrapper:
    """
    A drop-in wrapper that can replace the original sequential GT log prob computation
    in LLMGenerationManager without modifying the original code.
    
    Usage:
        # In your training script, after creating LLMGenerationManager:
        from scrl.llm_agent.vectorized_gt_logprob import VectorizedGTLogProbWrapper
        
        wrapper = VectorizedGTLogProbWrapper(
            generation_manager,
            enable_vectorized=True,
            validate_first_batch=True
        )
        
        # Use wrapper.compute_gt_log_probs instead of the original method
    """
    
    def __init__(
        self,
        tokenizer,
        actor_rollout_wg,
        enable_vectorized: bool = True,
        validate_first_batch: bool = True
    ):
        """
        Args:
            tokenizer: HuggingFace tokenizer
            actor_rollout_wg: Actor rollout worker group (for model access)
            enable_vectorized: Whether to use vectorized computation
            validate_first_batch: Whether to validate against sequential on first batch
        """
        self.tokenizer = tokenizer
        self.actor_rollout_wg = actor_rollout_wg
        self.enable_vectorized = enable_vectorized
        self.validate_first_batch = validate_first_batch
        self.validated = False
        
        config = VectorizedGTConfig(
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            validate_results=validate_first_batch
        )
        self.computer = VectorizedGTLogProbComputer(tokenizer, config)
    
    def compute_info_gain_rewards(
        self,
        messages_list: List[List[Dict]],
        ground_truths: List[Dict],
        model,
        info_gain_type: str = "prob_diff"
    ) -> List[List[float]]:
        """
        Compute information gain rewards for all samples using vectorized GT log prob computation.
        
        This method is designed to replace the per-turn computation in run_llm_loop.
        
        Args:
            messages_list: List of message histories for each sample
            ground_truths: List of ground truth dictionaries
            model: The language model
            info_gain_type: "prob_diff" or "log_prob_diff"
            
        Returns:
            info_gain_rewards: List of List of float, one inner list per sample
        """
        batch_size = len(messages_list)
        info_gain_rewards = [[] for _ in range(batch_size)]
        
        for i in range(batch_size):
            messages = messages_list[i]
            gt_text = ground_truths[i].get('ground_truth', '')
            
            # Collect turn end positions
            turn_end_positions = self._collect_turn_end_positions(messages)
            
            if len(turn_end_positions) == 0:
                continue
            
            # Build the full sequence
            full_text = self.tokenizer.apply_chat_template(messages, tokenize=False)
            encoding = self.tokenizer(full_text, return_tensors='pt')
            input_ids = encoding['input_ids'].squeeze(0)
            attention_mask = encoding['attention_mask'].squeeze(0)
            position_ids = torch.arange(len(input_ids), dtype=torch.long)
            
            # Compute GT log probs for all turns
            if self.enable_vectorized:
                gt_log_probs, gt_ranges = self.computer.compute_all_turns_vectorized(
                    model, input_ids, attention_mask, position_ids,
                    gt_text, turn_end_positions
                )
            else:
                gt_log_probs, gt_ranges = self.computer.compute_all_turns_sequential(
                    model, input_ids, attention_mask, position_ids,
                    gt_text, turn_end_positions
                )
            
            # Validate on first batch if requested
            if self.validate_first_batch and not self.validated and self.enable_vectorized:
                is_valid, details = self.computer.validate_vectorized_vs_sequential(
                    model, input_ids, attention_mask, position_ids,
                    gt_text, turn_end_positions
                )
                if not is_valid:
                    print(f"WARNING: Vectorized validation failed! Details: {details}")
                    print("Falling back to sequential computation.")
                    self.enable_vectorized = False
                else:
                    print(f"Vectorized validation passed! Max diff: {max(d['max_abs_diff'] for d in details['turn_comparisons']):.2e}")
                self.validated = True
            
            # Compute information gain rewards
            prev_value = None
            for t, (log_probs, (ans_start, ans_end)) in enumerate(zip(gt_log_probs, gt_ranges)):
                # Mean log prob of the answer portion
                answer_log_probs = log_probs[ans_start:ans_end]
                mean_log_prob = answer_log_probs.mean().item()
                
                if info_gain_type == "log_prob_diff":
                    cur_value = mean_log_prob
                else:  # prob_diff
                    cur_value = math.exp(mean_log_prob)
                
                if prev_value is not None:
                    info_gain = cur_value - prev_value
                    info_gain_rewards[i].append(info_gain)
                
                prev_value = cur_value
        
        return info_gain_rewards
    
    def _collect_turn_end_positions(self, messages: List[Dict]) -> List[int]:
        """
        Collect the token positions where each turn ends.
        
        Returns positions for:
        - prompt_end: After system + user message
        - turn1_end: After first assistant response
        - turn2_end: After second assistant response + tool response
        - etc.
        """
        turn_end_positions = []
        
        # Find prompt end (after system + first user message)
        prompt_messages = []
        for msg in messages:
            prompt_messages.append(msg)
            if msg['role'] == 'user' and len(prompt_messages) >= 2:
                break
        
        if prompt_messages:
            prompt_text = self.tokenizer.apply_chat_template(
                prompt_messages, add_generation_prompt=True, tokenize=False
            )
            prompt_tokens = self.tokenizer(prompt_text, return_tensors='pt')['input_ids']
            turn_end_positions.append(prompt_tokens.shape[1])
        
        # Find subsequent turn ends (after each assistant message)
        current_messages = prompt_messages.copy()
        for msg in messages[len(prompt_messages):]:
            current_messages.append(msg)
            
            # A turn ends after assistant response (and possibly tool response)
            if msg['role'] == 'assistant' or msg['role'] == 'tool':
                # Check if next message exists and is also tool (multi-turn tool call)
                # For simplicity, we mark turn end after each assistant or tool message
                turn_text = self.tokenizer.apply_chat_template(
                    current_messages, add_generation_prompt=True, tokenize=False
                )
                turn_tokens = self.tokenizer(turn_text, return_tensors='pt')['input_ids']
                
                # Only add if this is after assistant (not just tool)
                if msg['role'] == 'assistant' or (msg['role'] == 'tool' and 
                    len(current_messages) >= 2 and current_messages[-2].get('role') == 'assistant'):
                    turn_end_positions.append(turn_tokens.shape[1])
        
        return turn_end_positions


def create_vectorized_gt_computer(tokenizer) -> VectorizedGTLogProbComputer:
    """
    Factory function to create a VectorizedGTLogProbComputer.
    
    Args:
        tokenizer: HuggingFace tokenizer
        
    Returns:
        VectorizedGTLogProbComputer instance
    """
    config = VectorizedGTConfig(
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        validate_results=True
    )
    return VectorizedGTLogProbComputer(tokenizer, config)


# ============================================================================
# Config Initialization (called at the start of ray_trainer.py fit())
# ============================================================================

def init_from_config(config) -> bool:
    """
    Initialize vectorized switch from Hydra config.
    
    Call this function at the start of ray_trainer.py fit() method.
    
    Args:
        config: Hydra OmegaConf config object
        
    Returns:
        bool: Whether vectorization is enabled
    
    Config example (train.sh):
        +algorithm.use_vectorized_gt_logprob=true
    """
    enabled = getattr(config.algorithm, 'use_vectorized_gt_logprob', False)
    set_vectorized_enabled(enabled)
    return enabled


def compute_gt_logprob_with_switch(
    computer: VectorizedGTLogProbComputer,
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    ground_truth_text: str,
    turn_end_positions: List[int],
    temperature: float = 1.0
) -> Tuple[List[torch.Tensor], List[Tuple[int, int]]]:
    """
    Select computation method based on global switch.
    
    Args:
        computer: VectorizedGTLogProbComputer instance
        Other args same as compute_all_turns_vectorized
        
    Returns:
        Same as compute_all_turns_vectorized
    """
    if is_vectorized_enabled():
        return computer.compute_all_turns_vectorized(
            model, input_ids, attention_mask, position_ids,
            ground_truth_text, turn_end_positions, temperature
        )
    else:
        return computer.compute_all_turns_sequential(
            model, input_ids, attention_mask, position_ids,
            ground_truth_text, turn_end_positions, temperature
        )