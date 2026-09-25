# SuperNeo v1.2 — sectioned source

Source: user-supplied `superneo_v1_2.pdf.md`. The version label follows the supplied filename.

Authors: Wilson Nguyen and Srinath Setty, Microsoft Research.

The numbered files preserve the source bytes in order, including equations,
footnotes, author notices and fenced code. Only file boundaries were added.
The sections follow the existing v1.1 folder structure.

Source size: 271,042 bytes. Source SHA-256:

`e7ac49cd4e2b45d96443c69bd654498a002c057446c95ea9e83c31d2f2f2cf82`

| File | Section | Source lines |
| --- | --- | --- |
| [00_front_matter.md](00_front_matter.md) | Title, authors, abstract and table of contents | 1–55 |
| [01_introduction.md](01_introduction.md) | 1 Introduction | 56–257 |
| [02_technical_overview.md](02_technical_overview.md) | 2 Technical overview | 258–351 |
| [03_overview_of_following_sections.md](03_overview_of_following_sections.md) | 3 Overview of the following sections | 352–357 |
| [04_preliminaries.md](04_preliminaries.md) | 4 Preliminaries | 358–518 |
| [05_embeddings_and_evaluation_homomorphism.md](05_embeddings_and_evaluation_homomorphism.md) | 5 Embeddings and Evaluation Homomorphism | 519–718 |
| [06_strong_and_weak_interactive_reductions.md](06_strong_and_weak_interactive_reductions.md) | 6 Strong and weak interactive reductions | 719–778 |
| [07_superneo_folding_scheme_for_ccs.md](07_superneo_folding_scheme_for_ccs.md) | 7 SuperNeo's folding scheme for CCS | 779–921 |
| [08_concrete_parameters.md](08_concrete_parameters.md) | 8 Concrete parameters | 922–953 |
| [09_references.md](09_references.md) | References | 954–1047 |
| [10_supplementary_appendix_A.md](10_supplementary_appendix_A.md) | Supplementary material and Appendix A | 1048–1053 |
| [11_appendix_B_deferred_theorems_and_proofs.md](11_appendix_B_deferred_theorems_and_proofs.md) | Appendix B: deferred theorems, proofs and scripts | 1054–1902 |

From this directory, verify the section checksums and the reconstructed source:

```sh
sha256sum --check SHA256SUMS
cat [0-9][0-9]_*.md | sha256sum
```

The second command must return the source SHA-256 above.
