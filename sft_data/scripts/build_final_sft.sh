#!/usr/bin/env bash
set -euo pipefail

base="${1:-${AETHERSEARCH_SFT_WORKDIR:-/root/CLEAN_SFT_FINAL}}"
queryrewrite="$base/clean_queryrewrite_audited_keep_dedup.jsonl"
v31="$base/clean_v31.jsonl"
out="$base/final_sft.jsonl"
report="$base/final_sft.md"

tmp_dir="$(mktemp -d /tmp/build_final_sft.XXXXXX)"
trap 'rm -rf "$tmp_dir"' EXIT

# Keep only V3.1's final repair records. Search-retention records are excluded.
awk '/"sample_type"[[:space:]]*:[[:space:]]*"final_answer_repair"/ { print }' "$v31" > "$tmp_dir/v31_final.jsonl"

# Concatenate the two already-filtered datasets in a stable order.
awk '1' "$queryrewrite" "$tmp_dir/v31_final.jsonl" > "$tmp_dir/final.jsonl"

qr_rows=$(wc -l < "$queryrewrite")
v31_rows=$(wc -l < "$tmp_dir/v31_final.jsonl")
out_rows=$(wc -l < "$tmp_dir/final.jsonl")
qr_ids=$(rg -o '"original_id"[[:space:]]*:[[:space:]]*"[^"]+"' "$queryrewrite" | sed -E 's/.*"([^"]+)"$/\1/' | sort -u | wc -l)
v31_ids=$(rg -o '"original_id"[[:space:]]*:[[:space:]]*"[^"]+"' "$tmp_dir/v31_final.jsonl" | sed -E 's/.*"([^"]+)"$/\1/' | sort -u | wc -l)
out_ids=$(rg -o '"original_id"[[:space:]]*:[[:space:]]*"[^"]+"' "$tmp_dir/final.jsonl" | sed -E 's/.*"([^"]+)"$/\1/' | sort -u | wc -l)
cross_overlap=$(comm -12 \
  <(rg -o '"original_id"[[:space:]]*:[[:space:]]*"[^"]+"' "$queryrewrite" | sed -E 's/.*"([^"]+)"$/\1/' | sort -u) \
  <(rg -o '"original_id"[[:space:]]*:[[:space:]]*"[^"]+"' "$tmp_dir/v31_final.jsonl" | sed -E 's/.*"([^"]+)"$/\1/' | sort -u) | wc -l)

if [ "$out_rows" -ne 2825 ] || [ "$out_ids" -ne "$out_rows" ] || [ "$cross_overlap" -ne 0 ]; then
  echo "Validation failed: qr=$qr_rows v31=$v31_rows output=$out_rows unique=$out_ids cross_overlap=$cross_overlap" >&2
  exit 1
fi

cp "$tmp_dir/final.jsonl" "$out"

printf '%s\n' \
  '# final_sft dataset' \
  '' \
  'This is the combined SFT dataset built from the currently selected and validated subsets.' \
  '' \
  '## Composition' \
  '' \
  "- QueryRewrite: $qr_rows records, KEEP-only, deduplicated by original_id, final_answer only." \
  "- V3.1: $v31_rows records, final_answer_repair only." \
  '- QueryRewrite final target think blocks use: <think>The retrieved evidence now supports the answer.</think>.' \
  "- Total: $out_rows records." \
  "- Unique original_id values: $out_ids." \
  "- Cross-dataset original_id overlap: $cross_overlap." \
  '' \
  '## Excluded' \
  '' \
  '- QueryRewrite REJECT records are excluded.' \
  '- QueryRewrite records without a persisted audit decision are excluded.' \
  '- V3.1 search_retention records are excluded.' \
  '' \
  '## Files' \
  '' \
  '- final_sft.jsonl: uploadable JSONL dataset.' \
  '- clean_queryrewrite_audited_keep_dedup.jsonl: QueryRewrite component.' \
  '- clean_v31.jsonl: source V3.1 file; only final_answer_repair records were selected.' \
  > "$report"

echo "Wrote $out ($out_rows records)"
echo "Wrote $report"
