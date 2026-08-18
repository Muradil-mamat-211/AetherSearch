#!/usr/bin/env bash
set -euo pipefail

base='/root/CLEAN_SFT_FINAL'
raw="$base/clean_queryrewrite.jsonl"
old_log="$base/manual_audit_decisions.tsv"
new_log="$base/manual_audit_remaining_queryrewrite.tsv"
out="$base/clean_queryrewrite_audited_keep_dedup.jsonl"
audit_out="$base/clean_queryrewrite_audited_keep_dedup_audit.tsv"
report="$base/clean_queryrewrite_audited_keep_dedup.md"

tmp_dir="$(mktemp -d /tmp/build_audited_queryrewrite.XXXXXX)"
trap 'rm -rf "$tmp_dir"' EXIT

# Merge decisions by original_id. The newer log is read second and therefore
# supersedes the older decision when the two logs disagree.
awk -F '\t' '
  FNR == NR { if (FNR > 1) decision[$2] = $3; next }
  FNR > 1 { decision[$2] = $3 }
  END {
    for (id in decision) print id "\t" decision[id]
  }
' "$old_log" "$new_log" | sort -t $'\t' -k1,1 > "$tmp_dir/decisions.tsv"

awk -F '\t' '$2 == "KEEP" { print $1 }' "$tmp_dir/decisions.tsv" > "$tmp_dir/keep_ids.txt"

# Retain exactly one final_answer record per kept original_id.
awk -v keep_file="$tmp_dir/keep_ids.txt" '
  BEGIN {
    while ((getline id < keep_file) > 0) keep[id] = 1
    close(keep_file)
  }
  /"sample_type"[[:space:]]*:[[:space:]]*"final_answer"/ {
    if (match($0, /"original_id"[[:space:]]*:[[:space:]]*"[^"]+"/)) {
      token = substr($0, RSTART, RLENGTH)
      sub(/^"original_id"[[:space:]]*:[[:space:]]*"/, "", token)
      sub(/"$/, "", token)
      if (keep[token] && !emitted[token]++) print $0
    }
  }
' "$raw" > "$tmp_dir/data.jsonl"

 perl "$base/standardize_queryrewrite_think.pl" "$tmp_dir/data.jsonl" "$tmp_dir/standardized.jsonl"
 cp "$tmp_dir/standardized.jsonl" "$out"
cp "$tmp_dir/decisions.tsv" "$audit_out"

raw_rows=$(wc -l < "$raw")
raw_ids=$(rg -o '"original_id"[[:space:]]*:[[:space:]]*"[^"]+"' "$raw" | sed -E 's/.*"([^"]+)"$/\1/' | sort -u | wc -l)
old_rows=$(awk -F '\t' 'NR > 1 {n++} END {print n+0}' "$old_log")
new_rows=$(awk -F '\t' 'NR > 1 {n++} END {print n+0}' "$new_log")
unique_decisions=$(wc -l < "$audit_out")
keep_count=$(awk -F '\t' '$2 == "KEEP" {n++} END {print n+0}' "$audit_out")
reject_count=$(awk -F '\t' '$2 == "REJECT" {n++} END {print n+0}' "$audit_out")
out_rows=$(wc -l < "$out")
out_ids=$(rg -o '"original_id"[[:space:]]*:[[:space:]]*"[^"]+"' "$out" | sed -E 's/.*"([^"]+)"$/\1/' | sort -u | wc -l)
unreviewed=$((raw_ids - unique_decisions))

if [ "$keep_count" -ne "$out_rows" ] || [ "$keep_count" -ne "$out_ids" ]; then
  echo "Validation failed: KEEP=$keep_count output_rows=$out_rows output_ids=$out_ids" >&2
  exit 1
fi

printf '%s\n' \
  '# Audited deduplicated QueryRewrite KEEP dataset' \
  '' \
  '> This artifact contains only decisions that were already persisted in the two audit TSV files. It does not include judgments that were only made in conversation and never written to disk.' \
  '' \
  '## Construction' \
  '' \
  "- Source: clean_queryrewrite.jsonl ($raw_rows rows, $raw_ids unique original_id values)." \
  "- Audit logs: $old_rows rows in the earlier log and $new_rows rows in the newer log." \
  '- When the same `original_id` appears in both logs, the newer `manual_audit_remaining_queryrewrite.tsv` decision takes precedence.' \
  '- The output is deduplicated by `original_id`.' \
  '- Only `sample_type=final_answer` is retained; `search_retention` is intentionally excluded.' \
  '- QueryRewrite final target think blocks are standardized to: <think>The retrieved evidence now supports the answer.</think>.' \
  '' \
  '## Result' \
  '' \
  "- Unique logged decisions: $unique_decisions" \
  "- KEEP decisions: $keep_count" \
  "- REJECT decisions: $reject_count" \
  "- Output records: $out_rows" \
  "- Unreviewed source IDs not included: $unreviewed" \
  '' \
  '## Files' \
  '' \
  '- `clean_queryrewrite_audited_keep_dedup.jsonl`: final dataset.' \
  '- `clean_queryrewrite_audited_keep_dedup_audit.tsv`: deduplicated decision map used to build it.' \
  '- `manual_audit_decisions.tsv` and `manual_audit_remaining_queryrewrite.tsv` remain unchanged.' \
  '' \
  'The dataset is therefore the clean, deduplicated KEEP subset of the currently persisted audit decisions; it is not a claim that all 4211 source questions have already been audited.' \
  > "$report"

echo "Wrote $out ($out_rows records)"
echo "Wrote $audit_out ($unique_decisions decisions)"
echo "Wrote $report"
