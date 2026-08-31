# Attribution and rights status

## Release status

`PUBLIC_LICENSE_GATE = UNRESOLVED`.

No blanket license is asserted for the combined JSONL. No `LICENSE` file is
included because the available project materials do not establish that one
license authorizes redistribution of every question, answer, preference
continuation, and retrieved information span in this release.

This is an attribution and rights-status record, not legal advice.

## Components and upstream references

- **TriviaQA** — 1,445 preference pairs. The official TriviaQA page states
  that the University of Washington does not own the copyright of the
  questions and documents included in TriviaQA:
  <https://nlp.cs.washington.edu/triviaqa/>
- **MuSiQue** — 410 preference pairs. The upstream repository states that
  MuSiQue is distributed under CC BY 4.0:
  <https://github.com/StonyBrookNLP/musique>
- **Natural Questions** — 130 preference pairs. Consult the official dataset
  page and applicable source-document terms before redistribution:
  <https://ai.google.com/research/NaturalQuestions>
- **WebQuestions** — 87 preference pairs. The official Microsoft
  WebQuestionsSP download page is retained as the source reference:
  <https://www.microsoft.com/en-us/download/details.aspx?id=52763>
- **2WikiMultiHopQA** — 54 preference pairs. The upstream repository is marked
  Apache-2.0:
  <https://github.com/Alab-NII/2wikimultihop>

## Retrieved information

Some preference continuations contain `<information>...</information>` spans
derived from retrieval. The available provenance does not pin a public URL,
revision, and per-document license for every retrieved passage. Some passages
are consistent with Wikipedia-style retrieval, but that observation alone is
not sufficient to grant redistribution rights for every span.

Wikimedia states that most Wikimedia text is under CC BY-SA 4.0 and/or GFDL,
with attribution and reuse requirements:
<https://foundation.wikimedia.org/wiki/Legal%3AWikimedia_Developer_App_Guidelines>

The retrieval-corpus redistribution gate therefore remains unresolved.

## Other references

- Search-R1 question sets were used for overlap filtering and are not included
  as overlapping release records:
  <https://github.com/PeterGriffinJin/Search-R1>
- Qwen2.5-3B-Instruct is a base/model-format reference. This attribution record
  does not assert a model license for the preference data.

## Required user action before redistribution

Before redistributing the JSONL or applying a downstream license, verify the
terms for every question source, answer alias, retrieved passage, and generated
continuation. Preserve upstream attribution and all applicable share-alike or
other conditions.
