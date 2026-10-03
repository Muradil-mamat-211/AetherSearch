# Documentation Assets

This directory contains repository-owned static media used by the public
documentation.

| Asset | Purpose |
|---|---|
| `aethersearch-mark.svg` | AetherSearch monogram displayed in the root README |
| `aethersearch-method.svg` | Static overview of the Agentic RL training flow |
| `figures/search-credit-same-depth.png` and `.svg` | Four recorded second-Search cases for one prompt; immediate, return, and mixed credit |
| `figures/search-credit-delayed-return.png` and `.svg` | Recorded continuation gains and the mixed credit assigned to an earlier repeated Search |
| `figures/search-credit-boundaries.png` and `.svg` | All-negative terminal peer gains and consecutive singleton outcome fallback |

Assets remain local to the repository so the project overview does not depend
on an external image host.

The Search-credit figures use the committed evidence in `docs/data/` and can
be verified and rebuilt with
[`docs/scripts/build_search_credit_figures.py`](../docs/scripts/build_search_credit_figures.py).
Their scope and interpretation are documented in the
[case study](../docs/search-credit-case-study.md).
