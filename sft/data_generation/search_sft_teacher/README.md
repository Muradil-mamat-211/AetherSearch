# Adaptive DeepSeek Search-SFT teacher rollout

This directory is the new-data generation implementation, not the frozen
2,000-row release or its training launcher. Run commands from the AetherSearch
repository root. Set `AETHERSEARCH_SFT_WORKSPACE` to the absolute directory
holding `data/`, `models/`, `envs/`, and `code/Search-R1`; it defaults to the
repository root. The runtime directory may be a different volume from the
Git checkout. Set `AETHERSEARCH_DENSE_INDEX_PATH` if the released FlatIP index
is stored elsewhere, and `AETHERSEARCH_STUDENT_TOKENIZER_PATH` if the Qwen
student tokenizer is stored elsewhere. No runtime assets or credentials are
committed here.

## Production entry point

Use `deepseek_rollout.py` for DeepSeek. It calls the official fixed endpoint
`https://api.deepseek.com/beta/chat/completions` and reads the private key file
described below. No OpenAI account, SDK installation, or environment-variable
inheritance is required. HTTP calls use Python standard-library modules.
Student token counting uses the local `tokenizers` library; real retrieval
additionally requires the existing retriever environment.

The production client always requests `function.strict=true`, with an anchored
action-format pattern and no extra properties. The strict Beta endpoint is
required by the provider's tool-calling protocol. A rejected schema or endpoint
fails the request; there is no retry against a non-strict endpoint or fallback
to another tool/model. Strict mode constrains function arguments, not network
permissions or answer truth. Both thinking settings retain these restrictions.
References: https://api-docs.deepseek.com/guides/tool_calls/ and
https://api-docs.deepseek.com/api/create-chat-completion/ .

Default model: `deepseek-flash`, with thinking explicitly disabled. You may
select `deepseek-v4-pro` or enable thinking explicitly; there is no silent
model substitution. Requested model, returned model, response IDs, usage,
backend fingerprint, request timings and HTTP retry counts are recorded.
An API alias is not an immutable model-version pin.

Maximum thinking uses `thinking={"type":"enabled"}` with
`reasoning_effort="max"`; `type="max"` is invalid. The production CLI accepts
`--thinking enabled --reasoning-effort max`. Thinking defaults to `high` when
enabled without an explicit effort. Conflicting settings fail locally.

The old 1600-token test default has been removed. The client now defaults to
500 total API output tokens without thinking and 16384 with thinking. An
explicit `--max-tokens` overrides that default, up to our bounded 32768-token
limit. The API budget includes generated reasoning and visible output; it is
not the length of the short `<answer>`. A teacher's API tokenizer also differs
from the student's tokenizer. The project's 500-token student action budget
is enforced by the controller and independent validator; setting
the thinking teacher's total output cap to 500 could truncate its reasoning.
References: https://api-docs.deepseek.com/guides/thinking_mode/ and
https://api-docs.deepseek.com/api/create-chat-completion/ .

For enabled thinking, the client also charges each turn's provider-reported
`usage.completion_tokens_details.reasoning_tokens` against a per-candidate
reasoning budget (default 32768, configurable with
`--max-cumulative-reasoning-tokens`). If the provider omits that count, the
UTF-8 byte length of returned reasoning is charged conservatively. Exceeding
the limit rejects the candidate before another retrieval. There is **no
cumulative input-prompt token cap**: the provider's `usage.prompt_tokens` is
recorded for cost/context auditing, including repeated reasoning replay, but
never rejects a candidate. If usage is absent, request UTF-8 bytes are recorded
as accounting units, not claimed as exact API tokens. A 1000000-byte cap on
each serialized HTTP request remains a transport safety guard, not a cumulative
prompt-token budget. The reasoning and request-size limits do not truncate
reasoning and are distinct from the student's 500-token action/observation
limits. Receipts and checkpoint configuration bind the actual limits; audits
store usage sources/counts, number/length of replayed reasoning messages and
request sizes, never private reasoning text.

This is a real rollout controller, not the earlier rule-built trajectories:

```text
Local QA input (gold answers stay in the controller)
  -> DeepSeek sees the question and the retrieve tool schema
  -> DeepSeek chooses a direct answer or search
  -> a direct answer is audited but skipped for search-required SFT; it never loads/calls the retriever
  -> a search is a model-written <think>...</think><search>...</search> action
  -> controller validates the action and dispatches retrieve, nothing else
  -> wiki18 BM25 top-20 + E5-base-v2/FAISS FlatIP top-20 + RRF(k=60)
  -> reserve information tags and clip the real top-3 body to 500 student tokens
  -> reject observations that lose a document header or all of its text
  -> DeepSeek receives its previous action plus the exact tool result
  -> DeepSeek chooses another search or a final <think>...Turn N Doc M...</think><answer>...</answer>
  -> verify the raw final action and evidence, then normalize a searched final think for training
  -> independent structural/answer/provenance checks
  -> one-question SQLite commit, pending semantic review
  -> explicit human approval, then final training export
```

The model writes all action rationales, queries and final answers. The
controller never fills in a missing action, replaces a wrong answer with gold,
or rewrites a wrong citation into a correct one. Search extraction selects a
complete literal substring without trimming or whitespace normalization of its
think/query contents. Answer canonicalization also retains its original text.
Each emitted search action is checked strictly byte-for-byte against the
selected span of the raw API receipt. Both the original response and extraction
details remain in the audit. FINAL receipts
also undergo exact parsing before the documented answer canonicalization;
outer/inter-tag whitespace is not cleaned up. Missing receipts block audited export.
The short `<think>` segments are action rationales, not a dump of the model's
private internal reasoning. Thinking-mode `reasoning_content` is replayed to
DeepSeek when required for tool conversations, but is not inserted into SFT.

### Verified Chat Continuation

`deepseek_chat_history_v2_budgeted_reasoning` retains the entire ordered history for each candidate:
system, question, then each normalized assistant search and its paired tool
observation. The next request appends to that history; it does not replace it
with only the latest search. All enabled-thinking assistant messages carry
their original, unmodified `reasoning_content`, including an over-budget final
answer when asking the model to repair it. The repair request retains the exact
prior wire messages, including the search-budget-exhausted instruction.

`append_tool_result` verifies the received action, response identity, parent
history, call ID and observation format/budget before appending both messages
atomically. It cannot append the same response twice. Before another HTTP call,
the client rejects missing/reordered/changed history, missing reasoning,
orphaned or mismatched tool results, duplicate call IDs, unauthorized roles,
extra message fields and incomplete pairs. Search text remains the exact
selected API substring; only its transport envelope is normalized.

Each receipt records the continuation policy, SHA-256 request fingerprint and
ordered message manifests. Manifests hash public message content separately
from reasoning and record reasoning length, not private reasoning text. The
independent validator rebuilds the complete expected history from action
receipts and actual visible SFT observations, including discarded answer
repairs, and checks every replayed reasoning digest. These are application-side
integrity checks, not provider-signed attestations or proof of model attention.

`DeepSeekClient` is serial and not thread-safe. Parallel generation uses one
client per active trajectory slot, never one client concurrently from two
threads. `rollout` calls
`begin_trajectory()` for each new question; it clears only transient history
state, not request limits or audit receipts. Mid-candidate history truncation
is forbidden. A fresh client may import a structurally valid historical prefix
for off-policy diagnostics, but missing original reasoning cannot be recovered
from SFT text. Those diagnostics are not native on-policy production rollouts.
Process restart resumes at candidate boundaries; an interrupted candidate may
be retried from its question. Private reasoning is not persisted for turn-level
crash recovery, and no missing reasoning is synthesized.

A bounded paid diagnostic exercises the production adapter in non-thinking,
max-thinking and explicitly fault-injected content-only modes. Each case uses
two fixture observations and a final answer that must recover facts from both
turns. It checks complete history prefixes and exact reasoning replay. The
fixture tool is confined to this diagnostic, is not wiki18, and never creates
training records:

```bash
python3 -B sft/data_generation/search_sft_teacher/probe_chat_continuation.py
```

This runs at most four HTTP attempts per case (three cases), uses no HTTP
retries, prints only public actions and verification summaries, and omits
private reasoning and keys. The live model may fail a protocol check; such a
case is reported as failed rather than repaired into a fictitious success.

The current Chat Beta service has accepted normalized content-only search
envelopes in diagnostics. This is an observed compatibility path, not a claim
that the service officially guarantees arbitrary inserted calls. The provider
documents inserted-call support separately for Responses/Anthropic:
https://api-docs.deepseek.com/guides/tool_calls/ . No silent API migration,
raw-history duplication or alternate tool is used on failure. Replay rules:
https://api-docs.deepseek.com/guides/thinking_mode/ .

Every turn uses `tool_choice=auto` until the search budget is spent, including
the first turn. At the search limit, `tool_choice=none` prohibits another search.
A supported, reliable FINAL can still pass; otherwise the controller rejects
the candidate. It does not construct a refusal tag or a synthetic answer.
The controller never decides that a question "needs search"
from its gold answer or replaces a direct answer with a search. Unextractable or ambiguous actions,
unknown/parallel tools, duplicate queries and unsupported answers
reject that candidate, rather than synthesizing a replacement trajectory.

## Adaptive Prompt and Evidence Policy

The system prompt permits reliable prior knowledge and discourages unnecessary
search. It defines exactly two mutually exclusive branches, SEARCH and FINAL,
the sole tool, short decision summaries and the visible-evidence boundary.
The user prompt contains only the optional-search reminder, actual per-question
search budget and question. It does not repeat the system's tool-format rules.
No special chat-template tokens are inserted into API messages. Neither prompt
interpolates hidden reference-answer fields. An answer string may occur naturally
in the original question, model-written actions or visible retrieved evidence.
The complete versioned prompts are `INSTRUCTIONS` and `PROMPT` in
`controlled_rollout.py`; the only function schema is `TOOL` in that file.

Teacher-facing prompts request behavior, not student-token estimates:

```text
Keep the think summary brief and the search query concise and focused.
Tool observations may be truncated. Use only the evidence actually visible in them and do not infer omitted text.
```

The controller and validator enforce the 500-token complete-action budget.
It covers the think summary, search/answer content and all action tags, with
one reserved Qwen end-of-turn token. The old 160-character answer cap has been
removed. The answer should be the shortest complete answer to the actual
question, including all requested items and a concise sentence/list when needed.
A who-is question calls for an identifying role/definition. There is no
special decline action or tag; output outside the search/answer protocol is
rejected, and unsupported answers still fail the ordinary evidence/gold checks.

The controller's authoritative search-action source is
`controller.extracted_model_search_action`. Its preferred raw source is
`tool_calls[].function.arguments.action`; content-only complete search actions
are also extractable under the audited policy below. The teacher prompt and
strict schema still request the exact native-call format; the v11 prompt removes
candidate-answer wording without changing that tag grammar or query acceptance.
The SEARCH rules in the system prompt are:

```text
SEARCH: Emit exactly one native retrieve function call. Assistant content must be empty. Put the complete search action only in the JSON string field retrieve.arguments.action.
The value of retrieve.arguments.action must be exactly:
<think>brief decision summary</think><search>concise focused query</search>
This format is literal and exact:
- The string must begin with the literal <think> tag.
- The string must end with the literal </search> tag.
- </think> must be immediately followed by <search>.
- Do not add whitespace, newlines, text, or any other tags before <think>, between </think> and <search>, or after </search>.
- The JSON field name action is not a markup tag. Never output <action> or </action>.
- Emit exactly one <think>...</think> block followed by exactly one <search>...</search> block.
- The query must be a single line with 1-300 characters.
```

Queries must not contain URLs or repeat a previous query. The current SEARCH
prompt is neutral about whether a query contains a possible answer:

```text
Use the question, prior knowledge, and visible evidence to target the missing fact. Seek evidence for the requested relationship and consider contradictory evidence. Do not include URLs.
```

Continue searching only for a specific unresolved factual gap that a new
focused query is likely to resolve. After the call, stop and wait for its tool
result. The complete system prompt also requires brief summaries and queries,
and forbids inferring omitted text from truncated observations.

### Query Audit

`model_generated_candidate_queries_v1` audits provenance and executable
boundaries, not overlap with hidden reference strings. An answer/entity already
in the question, a historical search or a new model-written query is allowed.
The controller never injects a hidden label into a query or rewrites the
model's query toward the label. Reference answers stay in controller-side final
correctness checks; they do not determine query acceptance or input selection.

Every accepted query must pass these checks:

- Origin: a complete literal action extracted from the raw public API return.
  Receipts retain raw arguments/content, source, selected span and excluded text.
  The independent validator reconstructs extraction and requires exact equality
  between the receipt action, training event and parsed query metadata.
- Grammar: one contiguous think/search block; a nonblank single-line query of
  1..300 characters, without nested markup or URL schemes. Inner query text is
  not stripped, normalized or assembled before acceptance/backend dispatch.
- Budget: the complete action fits 500 actual student tokens including the EOS
  reserve; another search cannot run after the configured 1..5-call limit.
- Repetition: normalized queries within the same trajectory must be distinct.
  This is lexical duplicate detection, not semantic paraphrase detection.
- Execution: only the fixed local Hybrid-RAG retrieve path may execute. The
  audited retrieval trace query must equal the parsed model query exactly.
- Policy: prompt, generator, tool/query policy and tokenizer identity are bound
  to checkpoint/config/metadata; receipts also record the query-policy version.

There is no query/reference overlap rejection or automatic label-based query
correction. Candidate verification does not establish that the candidate is true:
after retrieval, FINAL still requires a supporting visible passage (and a
valid supporting citation if the teacher wrote one), normalized reference
agreement in production, and explicit semantic review before export.
String overlap with evidence alone does not prove the relationship asked by the
question. A malformed/wrong reference is a data-quality issue, not a reason to
force the teacher toward it. These audits establish application-visible origin
and dispatch, not what knowledge the hosted model may have memorized.

The function description contains only the purpose of retrieval:

```text
Search local wiki18 with BM25 + E5 FAISS FlatIP + RRF and return the top-3 passages. Not live web search.
```

The `action` parameter description contains only the exact action format:

```text
Exactly <think>brief decision summary</think><search>concise focused query</search>. No leading, trailing, or inter-tag whitespace.
```

`SEARCH_ACTION_TEMPLATE` is shared by the system and parameter description,
not repeated in the user prompt. The field/tag distinction and empty-content
requirement are defined once in the system. Spaces within the think summary
or query remain valid; forbidden positions are outside the action and between
its blocks. The controller extracts a complete contiguous literal block, then
uses strict full-match checks of that block. It never assembles tags or rewrites
the summary/query. Teacher-facing prompts expose no student
tokenizer, EOS reserve or numeric student-token budget details.

The user message is exactly this template:

```text
Search is optional. Search budget: at most {max_searches} calls.
Question: {question}
```

The SEARCH branch emits one native function call, then waits for its tool
result. `action` is a JSON key containing a string, not an XML parameter that
needs a closing tag. A valid argument-format illustration is:

```json
{"action":"<think>I need evidence for a missing fact.</think><search>question entities and missing fact</search>"}
```

This documentation example is not inserted into the teacher prompt, retrieved
evidence, or a query to reuse. The
requested action value ends immediately at `</search>`; the teacher should not
emit wrappers, invocation markup or suffixes. The extraction policy records
and excludes such surrounding text if a complete valid action is present.
The FINAL branch emits
no tool call, and assistant content must be exactly:

```text
<think>brief basis</think><answer>minimal answer</answer>
```

No whitespace, text or extra tags may occur before `<think>`, between
`</think><answer>`, or after `</answer>`. The answer contains only what the
question requests, without explanations, citations, unrelated background,
biographies or follow-up offers. Lists are appropriate for questions that
ask for multiple items. Without retrieval, the final think summary briefly
states that reliable prior knowledge was sufficient and cites no Doc.
After retrieval, only actually visible evidence can support the answer. The
teacher prompt is unchanged: it asks for a `Turn N Doc M` citation in the raw
final think summary. The controller validates any citation the teacher writes;
if there is none, it checks the answer against all visible retrieved passages.
Only after these checks does it replace the final think summary in the training
event. A searched answer uses exactly
`<think>The retrieved evidence now supports the answer.</think>`; a direct answer
is normalized to `<think>Reliable prior knowledge is sufficient to answer.</think>`
for internal auditing only, then excluded from production SFT. The latter does not falsely claim retrieval. The raw teacher action remains in
the checkpoint audit, and independent validation checks the replacement.
Search turns retain the model's original think and query text; no fixed wording
is imposed on either. The parser rejects document citations inside answer,
even if normalization or canonicalization might otherwise obscure them.
Semantic support still requires review.
Action think summaries are separate from the API's private `reasoning_content`.

The previous 20-question v5 run showed five rejected responses: four invalid
argument suffixes and one mixed content/tool response (also with an invalid
suffix). This establishes the observed failure modes, not their internal cause.
An intermediate v6 trial removed duplicated user-side action templates and
added an inline JSON illustration. It had zero mixed content/tool responses,
but still had three invalid argument strings, one query rejected by the old answer-overlap filter and
one JSON search imitation in ordinary content. These are single-run observations,
not a causal proof or a production success rate. The v7 trial kept
the explicit native-tool/final channel distinction and short user message,
removed the inline JSON illustration, and forbade textual JSON imitations.
The historical v8 prompt added the explicit field-not-tag prohibition and a
mechanical prefix/suffix/adjacency/block-count contract in system and schema
descriptions, with a field-scoped user reminder. Its effectiveness must be measured with unchanged
questions, model/thinking settings and hard checks, not assumed from wording.
The provider documents strict argument patterns, but its standard tool-call
examples allow nonempty assistant content alongside a call. Our empty-content
rule is an additional application protocol; strict mode does not enforce it.
References: https://api-docs.deepseek.com/guides/tool_calls/ and
https://api-docs.deepseek.com/guides/thinking_mode/ .

### Prompt Investigation: 2026-09-27

The same 20 questions, official endpoint, `deepseek-flash`, enabled/max thinking,
`auto` tool choice and unchanged strict pattern/hard checks were used once per
prompt version. No failed response was cleaned or retried for a better result.
Counts below are raw defects and may overlap; they are not training acceptance.

| Prompt | Raw tool turns | Mixed content/tool | Invalid argument strings | Rejected questions |
| --- | ---: | ---: | ---: | ---: |
| v5 | 7 | 1 | 5 | 5 |
| v6 | 7 | 0 | 3 | 5 |
| v7 | 7 | 0 | 4 | 5 |
| v8 | 8 | 0 | 1 | 2 |

The v7 run had 12 direct-answer actions, 3 accepted search actions and 5
rejections: 4 invalid tool strings and 1 search imitation in ordinary content.
It took 53.703 seconds; all accepted actions were within the student budget
(maximum 125 tokens including EOS reserve). No special decline tag appeared.
Only 3 of 11 direct answers with references matched a normalized gold alias
exactly, so protocol-valid does not mean a correct, minimal training answer.
Full retrieval and semantic support were not tested because RAG assets were
unavailable. No formal training data was emitted.

The v8 literal-field trial used the same questions and settings: 12 direct
actions, 6 accepted search actions and 2 rejections in 59.967 seconds, with
exactly 20 HTTP requests and no retries or cleanup. One query included a gold
alias (Angela Robinson) despite gold never being sent to the model; one action
still appended an invalid parameter-closing tag and literal escaped newline.
There were no mixed content/tool responses, special decline tags or student
budget violations; the maximum accepted action was 120 tokens including EOS
reserve. Five of 11 referenced direct answers matched a normalized gold alias
exactly. The 18 protocol-valid actions are not 18 final training samples: real
retrieval, evidence support and semantic acceptance remain untested. The
observed rejection count is lower in this one run, not a guarantee or proof
that all future model responses will follow the mechanical contract.

Two isolated diagnostic requests added a single fixed enum value to the
existing pattern, one with enabled/max thinking and one without thinking.
Both returned that exact value with empty content. They did not execute RAG.
This demonstrates only those fixed-value cases; it does not prove that the
provider's free-string pattern enforcement always works or identify whether
the remaining defect comes from generation or response serialization.
The enum is not in production: freezing an action would defeat real teacher
authorship and adaptive query generation.

The v5-v7 trials showed no improvement in total rejection count; the subsequent
v8 trial had fewer rejections, but still had failures.
These historical prompts clarified SEARCH and FINAL, but are not certified as
a complete fix or a production acceptance result. Those trials used a historical
whole-receipt rejection policy for parameter suffixes, quoted wrappers, textual
JSON imitations and mixed content. The current controller instead uses the
literal extraction policy below. Historical results and checkpoints remain untouched.

Audit artifacts under `logs/search_sft_teacher/`:

- `max_thinking_20_20260927T063307Z_e354ff/`: v5 baseline.
- `max_thinking_20_20260927T065102Z_7c94f6/`: intermediate v6 trial.
- `max_thinking_20_20260927T065539Z_792b1c/`: v7 trial, full requests,
  public responses, review packet, summary and `prompt_research_comparison.json`.
- `max_thinking_20_20260927T070912Z_499b69/`: historical v8 literal-field trial,
  exact request/response examples, full journal, review packet, summary,
  `protocol_audit.json` and same-question `prompt_comparison.json`.
- `max_thinking_100_20260927T072856Z_da8e2b/`: historical v8 additional-100 trial,
  78 direct actions, 16 accepted search requests and 6 rejections. Three tool
  actions had illegal suffixes, one search appeared in content without a native
  call, and two queries contained answer aliases. No mixed content/tool output
  or action-wrapper closing tag appeared in that batch. These results do not
  certify the new v9 protocol; no real RAG or training data was produced.
- `strict_enum_probe_20260927T065313Z_c8bd57/`: isolated fixed-enum probes.

Public diagnostic artifacts omit private `reasoning_content` and credentials;
the production client still replays private reasoning when required by the API.

The strict tool schema uses exactly:

```text
^<think>[^<>]+</think><search>[^<>\r\n]{1,300}</search>$
```

## Controller Extraction Policy

`literal_search_substring_v1` selects an already model-written contiguous
`<think>...</think><search>...</search>` block. The local SEARCH parser still
full-matches the selected block against the strict language above. Outer
whitespace, wrappers and trailing markup can be excluded, but inter-tag
whitespace, missing tags and invalid query contents are never repaired.
The query must be nonblank, single-line, 1..300 characters, and contain no URL.
Real student-token checks and duplicate-query checks remain. Candidate-answer
queries are permitted; hidden reference answers are not used to reject queries.

Native `retrieve.arguments.action` takes priority. An identical search repeated
in assistant content is recorded but contributes no extra SFT event. Other
surrounding text is retained in the receipt, not in training or subsequent
conversation history. Different complete actions in either source are ambiguous
and reject the candidate; mixed search/final output also rejects it. Identical
repeated blocks select their first occurrence and record the match count.
Unknown/parallel tools and malformed argument JSON remain hard failures.

If there is no native tool call but assistant content contains a complete
search action, the controller extracts it and creates an explicitly
controller-owned normalized `retrieve` envelope with a deterministic call ID.
Its receipt records `native_tool_call=false` and
`raw_action_source=assistant.content`. This does not claim the provider emitted
a native call, invent action text or invent evidence. It dispatches the same
real Hybrid-RAG backend and consumes the same search budget. `tool_choice=none`
rejects another search from either raw source before backend dispatch.

Receipts preserve raw tool calls, original arguments, assistant content,
selected character offsets, discarded prefix/suffix, normalized envelope and
raw source. The independent validator reconstructs extraction from these raw
fields, checks every extraction field, and compares the SFT event exactly with
the selected literal action. Extraction adds no API retry. FINAL format,
answer repair, evidence/provenance checks and semantic approval are unchanged.

`token_budget.py` loads the local Qwen tokenizer once, offline, using
`tokenizers==0.22.2` from `envs/teacher_tokenizer_deps`. Its path and tokenizer
file hashes are bound to checkpoints and recorded in metadata. Missing
tokenizer files or library fail before an API request. This dependency is
isolated from the existing conda environments. It can be provisioned with:

```bash
python -m pip install --no-deps --target envs/teacher_tokenizer_deps tokenizers==0.22.2
```

Every trajectory permits zero to five searches (CLI budget defaults to five),
followed by a final action. Search and answer actions are checked against 500
student tokens. The teacher API output budget is independent of this check.

Observations use the same reserve-tags, concatenate-body, prefix-clip algorithm
as AetherSearch's RL loop. The entire top-3 observation is bounded by 500 tokens,
including both information tags. Original passages remain in the audit; only
the visible clipped text is sent to the teacher, saved in training events, and
used to check answer support. If clipping removes Doc 3 or all its text, the
sample is rejected to retain the required three-document training format.
Both injected token IDs and the rendered-text token count are bounded.

The API conversation grows by replaying each normalized assistant `tool_calls`
message with empty content and the extracted action in its arguments,
then appending `{"role":"tool","tool_call_id":...,"content":"<information>..."}`
with the matching call ID. Each subsequent request carries the original
question and all preceding assistant/tool pairs. The SFT representation maps
those tool messages to environment events with loss mask false; assistant
actions have loss mask true. No ordinary user message impersonates the tool.
Provider `reasoning_content` is replayed in API history as required, but is
excluded from SFT. This is token-budget alignment, not a guarantee of identical
provider and student chat templates or tokenizer IDs.

If a well-formed final action exceeds the 500-token student action budget, the DeepSeek client permits
exactly one model-written correction. It replays the original assistant action
and the same conversation/evidence, followed by a generic instruction that
the previous final action exceeded the training action budget. That instruction
asks for a more concise rewrite without exposing a student-token number.
Gold answers are never added to this instruction. The correction uses
`tool_choice=none`; another tool call is rejected before backend dispatch.
Thinking-mode reasoning is replayed as required by the provider, without being
included in training events. A correction that remains too long or malformed
rejects the candidate. Wrong answers, unsupported evidence and invalid citations
still fail the ordinary checks; shortening does not relax them.

The overlong model action is retained in `audit.api_calls` with
`discarded_reason=answer_action_token_budget_exceeded`. Its revision has a
`repair_of_response_id` link and a recorded `tool_choice=none`. Only the accepted
revision becomes the final training event. The independent audit validator
checks that at most one overlong final action was discarded and that it is
immediately followed by the matching final correction. API usage for both
requests remains in the audit and both count against the HTTP attempt budget.
Structural correction alone does not establish factual correctness.

Direct answers contain one assistant event, zero environment events, empty
search/evidence lists, `answer_source=prior_knowledge`, `retrieval_used=false`,
`answer_in_evidence=null`, and `evidence_support_checked=false`. They are checked
against normalized gold aliases but do not claim retrieved evidence or an
evidence-support warning. Production runs skip them with
`zero_search_not_training_eligible`, rather than approving or exporting them.
Calling an answer "prior knowledge" records that no external tool was used;
it is not a measurement of confidence or proof of how the model recalled it.
Their final think is normalized to the fixed, truthful direct-answer sentence
above; no synthetic `<information>` is inserted.

Searched answers contain one real environment event per search. They retain
`answer_source=retrieved_evidence`, `answer_in_evidence=true`, and
`evidence_support_checked=true`. The raw teacher final may cite `Turn N Doc M`
(one-based retrieval turn N, document M in 1..3); bare `Doc M` means the latest
retrieval turn. If cited, at least one cited visible document must contain the
normalized answer; otherwise at least one visible document from any turn must.
The training final think is the fixed published-dataset sentence above.
Actual semantic support and multi-hop relations still need review. Direct
answers cannot cite non-existent retrieved documents.

The fixed wiki18 corpus is not live web access; it cannot establish current
facts that are absent from its historical evidence. No model claim changes
that limitation. Tool schemas/prompts guide the model; the executable whitelist,
argument checks, budget, loss masks and provenance checks enforce the boundary.

## Boundaries and prerequisites

The only advertised function is `retrieve(action)`. There is no web, shell,
filesystem or arbitrary-URL tool and no `eval`-style dispatch. The controller
owns the real retriever and constructs the environment message. Extra QA
fields such as context/evidence are discarded before the model session.

This is application-level tool isolation, not an OS network sandbox. The
controller itself needs HTTPS to DeepSeek and localhost retrieval. It cannot
erase a model's pretrained knowledge or prove that no memorized facts influenced
a decision. Evidence overlap is a necessary check, not semantic proof; an
independent reviewer must check the relationship asked by the question.

Production tool policy `hybrid_only_strict_v10_faiss_flat` (transport restrictions unchanged):

- Only text conversation messages are accepted; caller-injected system messages
  and multimodal URL/file content are rejected before a model request.
- DeepSeek requests use the fixed HTTPS Beta endpoint with TLS verification,
  no redirects, no implicit environment proxies, and a 2 MiB response limit.
  Proxy-dependent installations fail explicitly instead of silently rerouting.
- The sole tool dispatch is the fixed `retrieve(query, topk=3)` path. Function
  names never select arbitrary Python functions, shell commands or URLs.
- Dense requests use exactly `http://127.0.0.1:8000/retrieve` and 20 candidates,
  without environment proxies or redirects. HTTP status must be 200; bodies
  must be uncompressed JSON and no larger than 2 MiB, including chunked bodies.
- A 120-second wall-clock deadline covers each Hybrid-RAG query, RRF and
  provenance checks. It uses Linux SIGALRM in the single main-thread runner;
  an existing timer or another thread is rejected instead of disabling the
  deadline. Initial index loading/preflight is outside this query deadline.
- API receipts and checkpoint configuration record the endpoint and strict
  policy. These are controller audit records, not provider attestations.

Our records establish what external tools this application executed. They do
not independently audit the hosted provider's internal network or computation.
`strict=true`, an absent tool call, or a model's claim that it did not browse
is not proof of provider-side network isolation. `--doctor` reports this trust
boundary separately from resource readiness. No host firewall, VS Code/SSH
connectivity, container settings or existing retrieval server are changed.
OS-enforced isolation would be a separate deployment step with non-root
processes, read-only assets and explicit network rules; provider-side isolation
requires provider assurances or inference in an environment you control.

These transport restrictions are enabled by the DeepSeek teacher only. The
older rule-based generation scripts retain their existing retriever defaults.

Required assets for the teacher's FlatIP variant of Hybrid-RAG V1 are
uncompressed wiki18, existing BM25 SQLite index, E5-base-v2,
`e5_Flat.index`, retriever environment, and a running official dense server
on `127.0.0.1:8000/retrieve`. No index is auto-built and no resources are
installed or downloaded. The preflight verifies the listening process's
explicit corpus/index/model arguments and `--faiss_gpu`, not only that TCP port 8000 is open.
The default index path is `${AETHERSEARCH_SFT_WORKSPACE}/data/wiki18_faiss/e5_Flat.index`; set
`AETHERSEARCH_DENSE_INDEX_PATH` before starting the teacher to reuse the RL
artifact at another absolute path. Preflight requires the RL release artifact's
64,559,075,373-byte size and `IndexFlatIP` (`IxFI`) header. It does not replace
the RL release's SHA-256 verification. The checkpoint binds the resolved index
path, file signature, prompt, tool policy and generator version. A HNSW index
or old checkpoint cannot silently satisfy this contract.
It expects the official `search_r1/search/retrieval_server.py` started with:

```text
--index_path ${AETHERSEARCH_SFT_WORKSPACE}/data/wiki18_faiss/e5_Flat.index
--corpus_path ${AETHERSEARCH_SFT_WORKSPACE}/data/wiki18_corpus/wiki-18.jsonl
--retriever_name e5
--retriever_model ${AETHERSEARCH_SFT_WORKSPACE}/models/e5-base-v2
--faiss_gpu
```

The official Search-R1 server can read this FlatIP index and move it to GPU in
FP16; unlike the AetherSearch RL server, it does not use its Flat-index
streaming path. The startup memory profile differs, so verify host and GPU
capacity before launch. Do not substitute the RL *hybrid* server here: it
returns already-fused results under a different fusion algorithm.
Backend launch configuration is trusted local-operator configuration, not a
cryptographic attestation. Dense candidates and fused documents must exactly
match the corpus-backed BM25 document table; both branches must return 20
distinct candidates. RRF scores are independently recomputed. Any backend
failure stops the run; there is no dense-only or test-evidence fallback.
When all assets are present, `--doctor` also performs one real local
`United States` query through BM25 top-20, the official dense server top-20,
RRF top-3 and corpus-identity checks. `hybrid_retrieval_probe=false` or a
`retriever_probe_error` means `--run` stops before any paid teacher request.
The probe reads existing assets and makes one local retrieval request; it
does not write training data or rebuild an index. A passing probe establishes
backend operability for that query, not that every future query will succeed.

Read-only preflight (no paid API requests):

```bash
python3 -B sft/data_generation/search_sft_teacher/deepseek_rollout.py --doctor
```

Twenty-question first-decision diagnostic using maximum thinking:

```bash
python3 -B sft/data_generation/search_sft_teacher/test_api_20_questions.py
```

This explicitly runs the initial-decision test only: Obama plus 19 questions
sampled reproducibly from the existing 500-file (7 nq, 6 triviaqa, 6
web_questions). Only the question is sent to the model, never previous
trajectories/evidence or gold answers. A search action is recorded as awaiting
real RAG, without inventing a tool result. This diagnostic can run when the RAG
assets are unavailable; the production generator still refuses that state.
Every accepted first action is counted with the student tokenizer, including
its EOS reserve, and the diagnostic records the 500-token budget specification.
Passing first-action checks does not establish answer correctness or complete
retrieval-rollout validity. Results, request schemas, response content, reference comparisons and usage are
saved per question in a fresh `logs/search_sft_teacher/max_thinking_20_*`
directory. Provider reasoning is replayed when needed but omitted from the
review files. No training file is created and no existing output is overwritten.

Additional questions without reusing any previous diagnostic questions:

```bash
python3 -B sft/data_generation/search_sft_teacher/test_api_20_questions.py \
  --num-questions 100 --seed 44 --exclude-history
```

The diagnostic runner keeps its historical filename and default 20-question
batch, but accepts a bounded question count (1..1000). `--exclude-history`
reads prior `max_thinking_*/input_manifest.json` batches and single-question
`exact_search_live_*/summary.json` tests, normalizes their question keys and
excludes them before sampling. It fails before API calls if there are not
enough unique unseen questions; malformed history is not silently ignored.
Sampling balances the three QA sources subject to available unseen questions,
and never sends source contexts or gold answers to the teacher. The seed,
history files, excluded-key count and verified zero overlap are saved in the
input manifest. Each run gets a fresh `max_thinking_{count}_*` directory.
No production prompt, strict schema, controller budget or acceptance rule is
changed by these diagnostic options. API HTTP retries are disabled; the existing
bounded final-action budget repair remains enabled and separately recorded.

A bounded API-only smoke check makes one request through the actual new
client and validates a search action. It does not call RAG, return invented
passages, or write a dataset/checkpoint:

```bash
python3 -B sft/data_generation/search_sft_teacher/deepseek_rollout.py --api-smoke \
  --model deepseek-flash --thinking disabled
```

## Input and Run

### Offline Saved-Response Comparison

For a paid continuation test using historical prefixes from the published
`muradil211/AetherSearch_SFT` dataset, use:

```bash
python3 -B sft/data_generation/search_sft_teacher/test_multiturn_context_api.py \
  --num-contexts 100 --seed 42 --workers 4
```

The diagnostic pins the published revision and verifies its JSONL checksum.
It selects distinct questions before search turns 2, 3 and 4, excluding all
later actions, later observations and the final answer from API messages.
The current production prompt/client/extractor and student budgets are reused.
Historical actions are mapped to tool calls and historical observations use
the existing 500-token truncation policy; states losing top-3 structure fail
selection. Historical private reasoning is unavailable, so the API history
has empty reasoning strings, not invented reasoning or claimed DeepSeek turns.
The original dataset final answer is a held-out reference, not an independently
verified gold label or something sent to the teacher.

This is an off-policy next-action diagnostic, not a new real-RAG rollout:
the model may request another search or answer earlier than the old trajectory.
New search requests are recorded but not executed, and no synthetic result is
returned. Cited visible-answer overlap is checked, not certified semantic support.
Historical doc labels are diagnostic identifiers, not verified corpus doc IDs.
Public requests/responses, extraction receipts and per-target-turn outcomes are
saved in a new log directory; credentials and private reasoning are not saved.
There is no formal SFT export. Maximum thinking is enabled, with four bounded
workers, no HTTP retries, and the existing one-time final-budget repair.
`--prepare-only` makes no DeepSeek calls. `--prepared-dir DIR --resume` appends
only unattempted cases; completed responses are never regenerated on resume.

Replay the same saved first-decision responses through the current controller,
without a key, new API calls, retriever execution or training data generation:

```bash
python3 -B sft/data_generation/search_sft_teacher/replay_api_responses.py \
  --input logs/search_sft_teacher/max_thinking_100_20260927T074941Z_1c1be0/results.jsonl
```

The input must contain 100 unique questions with one saved response each.
Each replay writes a fresh `controller_replay_100_*` directory with
`summary.json`, `replay_results.jsonl` and `comparison.txt`; the original input
and existing outputs are never overwritten. Extraction is shared with the
production adapter, not a separate permissive test parser. Raw-protocol defect
counts in API diagnostics remain separate from controller acceptance counts.

Reclassify saved multi-turn responses with the current query rules, retaining
their original questions, historical prefixes and actual public API returns:

```bash
python3 -B sft/data_generation/search_sft_teacher/test_multiturn_context_api.py \
  --prepared-dir logs/search_sft_teacher/multiturn_context_100_20260927T085400022687Z \
  --reclassify-only
```

This mode needs no key, performs no downloads/API/retrieval, does not resume or
rewrite the old run, and writes a fresh `candidate_query_replay_100_*` directory.
The saved source checksum, input/result identity and student-budget specification
are checked. Extraction and next-action validation share the live diagnostic
implementation. The report identifies the original prompt and marks
`fresh_prompt_test=false`; private reasoning was omitted from the saved logs
and cannot be audited again.

The completed 100-context reclassification changed 69 search requests / 31
rejections to 93 search requests / 7 rejections. All 24 former query/reference
overlap rejections were recovered, with no regressions. The remaining failures
were 2 unsupported cited answers, 3 invalid citations, 1 incomplete output and
1 ambiguous search action. No new API requests or RAG calls were made, and no
training data was created. This is not a fresh prompt benchmark or an end-to-end
SFT success rate: accepted searches still need real retrieval and a valid FINAL.

For the saved first-decision batch, historical counts were 76 direct answers, 11 search requests
and 13 rejected responses. The historical v10 extraction-only controller yielded
76 direct answers, 21 search requests and 3 query/reference-overlap rejections.
Those overlaps are no longer rejected by the candidate-query policy. Offline
reclassification is not a fresh API test of the updated prompt and does not
relabel old checkpoints as current-policy samples. These are first-decision
protocol results, not completed or
semantically approved SFT trajectories. Search requests still require actual
Hybrid-RAG observations and a supported final answer.

Prepare a normalized JSONL from the intended training QA split, independently
of the model. Required fields: `id`, `question`, `golden_answers` (nonempty
list of strings), `data_source`, `split`. No contexts/evidence are consumed.
IDs are namespaced as `data_source:split:id`; duplicate IDs/questions fail
before any model request. `split` other than `train` is rejected without
sending it to DeepSeek. This split check does not detect contamination in a
mislabelled file: separately exclude eval question IDs and duplicates.
An answer string already in the original question is allowed; the question is
not rewritten or filtered using its hidden label.

The retrieval environment is a separate long-running service plus persistent
in-process BM25/RRF: the official Search-R1 E5/FAISS FlatIP dense server receives
HTTP on localhost port 8000, while the teacher process holds the wiki18 BM25
index and fuses 20 candidates from each branch with RRF(k=60). The teacher
sends only the chosen query to the port, verifies returned document identity,
and passes top-3 real passages back to DeepSeek. It is not live web search.
This follows AetherSearch RL's service/port *topology*, not its released
min-max weighted fusion algorithm; substituting that RL hybrid server would
change the SFT evidence distribution. See `DATASET_PROVENANCE.md` for wording
to accompany **new** exports. Do not apply this provenance retroactively to
the frozen 2,000-row published dataset.

After all corpus, index, model, and environment assets exist, start the dense
server in its own persistent terminal session. This reads existing assets; it
does not build or download them. The official server binds port 8000 on all
interfaces, so ensure host/container networking does not expose it publicly.

```bash
cd /path/to/AetherSearch
export AETHERSEARCH_SFT_WORKSPACE=/path/to/runtime-assets
tmux new-session -d -s search-sft-dense \
  'cd "$AETHERSEARCH_SFT_WORKSPACE/code/Search-R1" && exec "$AETHERSEARCH_SFT_WORKSPACE/envs/retriever/bin/python" search_r1/search/retrieval_server.py --index_path "$AETHERSEARCH_SFT_WORKSPACE/data/wiki18_faiss/e5_Flat.index" --corpus_path "$AETHERSEARCH_SFT_WORKSPACE/data/wiki18_corpus/wiki-18.jsonl" --topk 20 --retriever_name e5 --retriever_model "$AETHERSEARCH_SFT_WORKSPACE/models/e5-base-v2" --faiss_gpu'
python3 -B sft/data_generation/search_sft_teacher/deepseek_rollout.py --doctor
```

`--doctor` must report `ready=true`, including verified ownership/configuration
of the dense process. An existing listener on port 8000 is not accepted merely
because it responds. The 5-field public training rows do not contain service
metadata; the versioned checkpoint and provenance note carry it separately.

Run only when preflight is ready. Replace the input placeholder with your
actual training QA JSONL. Each line must contain `id`, `question`, a nonempty
`golden_answers` list, `data_source`, and `split=train`. The frozen public
`muradil211/AetherSearch_SFT` file has no `golden_answers` column; this runner
does not download it or infer a ground-truth label from its old trajectory.
Prepare and verify labels independently before `--run`:

```bash
conda run --no-capture-output -p "${AETHERSEARCH_SFT_WORKSPACE}/envs/retriever" python sft/data_generation/search_sft_teacher/deepseek_rollout.py \
  --run --model deepseek-flash --thinking disabled \
  --questions PATH_TO_TRAIN_QA.jsonl --db logs/search_sft_teacher/deepseek_pilot.sqlite \
  --max-examples 10 --max-searches 5 --max-api-requests 100 \
  --concurrency 8 --retrieval-batch-queries 8 --retrieval-batch-wait-ms 5 \
  --public-id-start 500001
```

`--max-examples` is the total input window, including checkpointed rows; rerunning
the same command does not secretly add another batch. To extend the window,
explicitly increase that limit while keeping the same input file.

Public IDs are assigned by input position, so extending the window leaves
existing IDs unchanged. `--public-id-start` is bound to the checkpoint; use
a distinct six-digit range if separately generated files will be merged.
The default range begins at `500001`, outside the released 2,000-row range.
The internal checkpoint/review ID remains `source:split:source_id`.

API budget counts HTTP attempts, including failures/retries. Authentication, balance and
other non-transient failures are not retried. Transient errors use bounded
backoff, default at most two retries. A timed-out POST may have already been
billed by the provider; retries are not guaranteed exactly-once billing.

`--concurrency` advances up to 8 independent questions in a thread pool by
default; set it to 1 for the previous serial path. Each trajectory still
waits for its own tool observation before its next model turn. A future-based
gateway coalesces searches arriving within the short wait window, then sends
up to 8 queries in one ordered dense HTTP request. BM25 and RRF remain
per-query and the main thread owns the SQLite connection and retriever backend.
This is not Ray and does not silently switch to the AetherSearch RL fusion.
`--max-api-requests` is one atomic process-wide HTTP-attempt budget, including
retries across all worker threads. Batching metrics are printed at completion.

The preflight opens and closes one retriever instance for its real probe.
During rollout, a separate retriever instance is initialized once at the first
validated search and reused across questions. Full real-RAG readiness
is still required by production preflight: optional search is not an excuse to
offer an unavailable tool or silently fall back when it fails. The
database is locked against concurrent writers and commits one candidate at a
time. Ctrl-C/restarts preserve completed candidates. With concurrent generation,
up to `--concurrency` in-flight questions may be repeated after interruption;
successful/pending/approved rows are never regenerated
by ordinary resume. Infrastructure-error rows are retried on resume.
Even a no-work resume still performs the read-only preflight probe.
An append-only attempt-history table preserves failed audits when a later
retry succeeds; the latest-outcome table is not the only copy of earlier work.

Only explicitly retry rejected candidates, without touching passing rows:

```bash
conda run --no-capture-output -p "${AETHERSEARCH_SFT_WORKSPACE}/envs/retriever" python sft/data_generation/search_sft_teacher/deepseek_rollout.py \
  --run --questions PATH_TO_TRAIN_QA.jsonl --db logs/search_sft_teacher/deepseek_pilot.sqlite \
  --max-examples 10 --max-searches 5 --retry-ids 'nq:train:FAILED_ID'
```

Input hash, provider/model, thinking mode, generation limits, prompt/tool-schema
hash, adaptive policy version and asset
signatures are checkpoint-bound. Changed configuration requires a new
checkpoint, not silently mixing incompatible samples. HTTP budgets/timeouts
can be adjusted without changing the trajectory-generating configuration.
The strict endpoint, transport policy and retrieval deadline are also bound.
Pre-strict checkpoints are left untouched and cannot be resumed under the new
policy. They are not regenerated or relabelled as strict-mode runs. Use a new
checkpoint for new strict-mode samples; per-ID retries within that checkpoint
still preserve every passing row.
The current generator is `controlled_deepseek_teacher_v22_faiss_flat`, using
`adaptive_search_v12_faiss_flat` prompts, tool policy
`hybrid_only_strict_v10_faiss_flat`, query policy
`model_generated_candidate_queries_v1`, extraction policy
`literal_search_substring_v1`, continuation policy `deepseek_chat_history_v2_budgeted_reasoning`
and final-summary policy `all_final_think_by_source_v2`, with checkpoint schema
`deepseek_teacher_checkpoint_v22_faiss_flat`. Existing v1..v21 files are
not rewritten or migrated. Start a new checkpoint/output path for this policy;
older checkpoints cannot be resumed or exported through the new
validator as if they were new-policy data. Existing old checkpoints
remain readable by `--stats`.
The compatibility OpenAI generator sharing these prompts is now
`controlled_teacher_v10_faiss_flat`; its checkpoint binds the changed prompt,
generator, tool policy and protocol hash rather than silently reusing old rows.

```bash
python3 -B sft/data_generation/search_sft_teacher/deepseek_rollout.py --stats \
  --db logs/search_sft_teacher/deepseek_pilot.sqlite
```

Stats include checkpoint outcomes, candidate attempts, source distribution,
global and source-level skip reasons, answer-source/search-count distributions,
evidence branches and average searches
and information characters. Historical attempt-level skip counts are reported
separately from latest-outcome counts. Full retrieval traces and API receipts remain in
SQLite, including partial evidence for rejected/error candidates.

## Validate, Review, Export

Checkpoint and candidate-review output retain `id`, `data_source`, `split`, `trajectory_type`, `question`,
`golden_answers`, `prompt`, `messages`, `response`, `events`, `metadata`.
Trajectory type is `teacher_hybrid_v1_real_rollout`, distinguished by adaptive
policy/prompt/generator metadata. Zero-search answers are parseable for audit
but rejected from production generation with `zero_search_not_training_eligible`.
Each search is followed by
an environment event, then another model action. Assistant tokens are trained;
environment tokens are masked. Actual training code must honor these masks.

Checks include exact extracted SEARCH/FINAL format, raw receipt reconstruction,
literal extraction/event equality, empty normalized tool-turn assistant content,
explicit raw-source and controller-envelope provenance, student-token budgets,
normalized final-answer matching, canonical short answers, query provenance and
duplicate detection, evidence in cited retrieval turns (or all visible turns if
uncited), raw-final-to-normalized-final equality, exact messages/events
concatenation, top-3 information structure and audit equality/provenance.
Document citations inside `<answer>` are rejected in both the event and the
unmodified API receipt. No tagged decline protocol is accepted.
For searched records, missing question entities set `evidence_support_warning`; this is a heuristic
warning, neither proof of support nor proof that a record is wrong.

Candidates have `semantic_review_status=pending`, not final training status.

### Published SFT Training Format

`--export-approved` writes the five public fields of
https://huggingface.co/datasets/muradil211/AetherSearch_SFT in order:
`id`, `question`, `trajectory_type`, `search_count`, `full_trajectory_text`.
The current public-format policy is `aethersearch_full_trajectory_v2_numeric_ids`.
The full text uses the published Qwen system/user prompt exactly, followed by
one assistant message containing every model search, every real masked
`<information>` observation and the final answer. It ends with exactly one
assistant `<|im_end|>` token. The published 2000-row snapshot's chat prefix
matches this renderer for every row. Production exports contain only
`search_count>=1` and the published `single_search` / `multi_search` types.
Direct answers are recorded as source-level skips and cannot be approved or
exported. Public IDs are six-digit strings such as `500001`, stored as
`metadata.public_id` in the detailed checkpoint and unique-indexed there.
The source-based internal `teacher_` ID is retained only for review/audit.
The released SFT-2000 trainer's fixed row-count and SHA-256 defaults still
target the frozen 2,000-row snapshot. Supply the actual new row count and
file SHA-256 to that trainer, then run its `--check_data_only` before training;
an ID-format fix alone does not constitute a passed trainer integration test.
For retrieved answers, the final think in this new export is normalized by the
controller to the published dataset's common sentence. Both raw teacher finals and API receipts
stay in the checkpoint, not in this five-field file.

The public artifact has no token-level mask field, matching the published
schema. The trainer must mask `<information>` and system/user text while
supervising assistant action text and its final `<|im_end|>`. The detailed
checkpoint retains event masks and retrieval provenance for audit. Each full
trajectory must fit 4096 actual student tokenizer tokens; all 2000 published
reference rows fit (maximum 2893). Overlong new candidates are rejected before
approval/export, never silently clipped. The standalone public validator checks
the historical five-field chat/action layout, not the new teacher's stricter
per-observation 500-token, top-3, or raw Doc-citation audit rules: historical
records can contain markup from corpus passages and uncited final summaries.
For **new** records, `--db --require-approved` additionally reconstructs the
exact public text from the approved detailed record and verifies those stricter
budget, optional raw-citation, retrieval-provenance and semantic-review requirements. The
published dataset does not carry the detailed receipts needed for that audit.

```bash
python3 -B sft/data_generation/search_sft_teacher/deepseek_rollout.py --export-candidates \
  --db logs/search_sft_teacher/deepseek_pilot.sqlite \
  --output data/search_sft_teacher/deepseek_pilot_needs_review.jsonl

python3 -B sft/data_generation/search_sft_teacher/validate_teacher_rollout.py \
  --input data/search_sft_teacher/deepseek_pilot_needs_review.jsonl \
  --db logs/search_sft_teacher/deepseek_pilot.sqlite

python3 -B sft/data_generation/search_sft_teacher/deepseek_rollout.py --review-packet \
  --db logs/search_sft_teacher/deepseek_pilot.sqlite \
  --output data/search_sft_teacher/deepseek_pilot_review.txt --max-examples 10
```

The review packet shows both the normalized training FINAL and the raw teacher
FINAL from its audit. Review the raw rationale and any citation before approving;
only the normalized action enters the public training text.

After an independent reviewer verifies the actual asked relationship, every
action rationale, and the supporting passage, explicitly approve selected IDs:

```bash
python3 -B sft/data_generation/search_sft_teacher/deepseek_rollout.py --approve \
  --db logs/search_sft_teacher/deepseek_pilot.sqlite --ids 'nq:train:REVIEWED_ID' \
  --reviewer REVIEWER_NAME --review-note 'Verified relationship and visible evidence'
```

If a support warning exists, approval also requires `--ack-evidence-warning`.
Approval cannot make structurally invalid data pass; it is not an automatic
semantic judge. There is no approve-all shortcut.

```bash
python3 -B sft/data_generation/search_sft_teacher/deepseek_rollout.py --export-approved \
  --db logs/search_sft_teacher/deepseek_pilot.sqlite \
  --output data/search_sft_teacher/deepseek_pilot_approved.jsonl

python3 -B sft/data_generation/search_sft_teacher/validate_teacher_rollout.py \
  --input data/search_sft_teacher/deepseek_pilot_approved.jsonl \
  --db logs/search_sft_teacher/deepseek_pilot.sqlite --require-approved
```

Exports prevalidate all selected rows and use a consistent database snapshot.
Every output path must be new; existing datasets are never overwritten.
Passing the independent validator is not a guarantee that training loaders
handle environment loss masking correctly; check that separately.

## Tests and Limitations

```bash
AETHERSEARCH_SFT_WORKSPACE=/path/to/runtime-assets PYTHONPATH=sft/data_generation/search_sft_teacher \
  python3 -B -m unittest discover -s sft/data_generation/search_sft_teacher -p 'test_*.py' -v
```

Mocks exist only in offline unit tests in temporary directories. There is no
mock-RAG CLI option and no test fixture can be selected by a production run.
API smoke success does not imply full-RAG rollout success. Until real assets
and the correctly configured dense server are available, `--run` fails before
paid model requests and before dataset/checkpoint creation.

Current primary API references:
- https://api-docs.deepseek.com/api/create-chat-completion/
- https://api-docs.deepseek.com/guides/thinking_mode/

The old OpenAI `controlled_rollout.py --run` entry is retained for compatibility.
Use the DeepSeek entry above for its stricter provenance, validation and export
gates. The original Search-R1 repository and existing generated data are not
modified by this implementation.

## Private DeepSeek key file

Use `deepseek_key.py` to create `~/.config/search-r1/deepseek_api_key`, outside
the workspace. Its directory must have permissions `700` and its file `600`.
The file contains only the key, not shell commands. The reader rejects symlinks,
wrong ownership, unsafe permissions and malformed keys. This is plaintext with
filesystem access controls, not encryption; root processes can still read it.
Revoke any key previously posted in a conversation before configuring a new one.

Run this from your own terminal. It saves that terminal's `DEEPSEEK_API_KEY`
if present; otherwise it prompts without echo. Existing key files are never
overwritten. The key is never printed:

```bash
python sft/data_generation/search_sft_teacher/deepseek_key.py --configure
```

Check file access from any terminal; this does not call the API:

```bash
python sft/data_generation/search_sft_teacher/deepseek_key.py --check
```

Run a bounded tool-protocol test (two small paid API calls, no real RAG or
training data). The controller returns a random test-only marker after the
first tool call and verifies the model's second response against it:

```bash
python sft/data_generation/search_sft_teacher/deepseek_key.py --probe --model deepseek-flash
```

The key helper's older two-call probe uses a random test-only marker, not corpus
evidence. The production DeepSeek runner never calls that helper probe. Both
clients read the key file without environment inheritance and refuse HTTP
redirects. No key is saved in the workspace, checkpoint, or request body.
