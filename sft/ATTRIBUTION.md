# ATTRIBUTION.md

## Release status

`PUBLIC_LICENSE_GATE = UNRESOLVED`.

No blanket license is asserted for the combined JSONL. No `LICENSE` file is
included because the local project materials do not establish that one license
can authorize redistribution of every question, answer, and retrieved
information span in this release.

This is an attribution and rights-status record, not legal advice.

## Components and upstream references

### QueryRewrite sources

- **TriviaQA**: 996 records in this release. The official TriviaQA page states
  that the University of Washington does not own the copyright of the
  questions and documents included in TriviaQA. The release therefore does
  not infer a blanket redistribution license from the dataset name alone:
  <https://nlp.cs.washington.edu/triviaqa/>
- **WebQuestions**: 29 records in this release. The local materials identify
  the source but do not include a complete redistribution license notice. The
  official Microsoft WebQuestionsSP download page is retained as a reference:
  <https://www.microsoft.com/en-us/download/details.aspx?id=52763>

### V3.1 sources

- **MuSiQue**: 720 records in this release. The upstream repository states
  that MuSiQue is distributed under CC BY 4.0:
  <https://github.com/StonyBrookNLP/musique>
- **2WikiMultihopQA**: 255 records in this release. The upstream repository is
  marked Apache-2.0:
  <https://github.com/Alab-NII/2wikimultihop>

### Retrieved information corpus

The `<information>...</information>` spans are part of the released full
trajectories. Local provenance does not pin a single corpus snapshot, URL for
every document, document revision, or per-document license. Some content is
consistent with Wikipedia-style retrieval, but that observation is not enough
to grant rights for every released span. Wikimedia states that most Wikimedia
text is under CC BY-SA 4.0 and/or GFDL and that reuse requires attribution and
compliance with the applicable license:
<https://foundation.wikimedia.org/wiki/Legal%3AWikimedia_Developer_App_Guidelines>

The retrieval-corpus redistribution gate therefore remains unresolved.

### Other references

- Search-R1 references were used for filtering and are not included as release
  records:
  <https://github.com/PeterGriffinJin/Search-R1>
- Qwen2.5-3B-Instruct is the base/model-format reference. The historical SFT
  checkpoint has a separate release repository at
  <https://huggingface.co/muradil211/AetherSearch-SFT>; no model license is
  asserted by this data attribution record.

## Required user action before redistribution

Before publishing the JSONL publicly or applying a downstream license, verify
the terms for TriviaQA questions/evidence, WebQuestions, each V3.1 source, and
the exact retrieval corpus snapshot. Preserve the upstream attribution and any
share-alike or other conditions that apply.
