# Contract source registry

Status: **mechanically locked; semantic normalization open**.

This file is generated from `src/sources/lock.toml`. Each source ID
names one exact reviewed byte string. SHA-256 identifies artifacts
only. It is not protocol authority.

Repository commit: `8c0b45bdd7d45421ef528d60feb3c9881cbd16c5`.

| Source ID | Path | Reviewed SHA-256 | Base SHA-256 | Role |
|---|---|---|---|---|
| SRC-PAPER-00 | `protocol-contract/paper-sources/00-front-matter.md` | `a87c04df841d749d5832adf28f30d64a4950be204c4751b25c194b92cde7972d` | `a87c04df841d749d5832adf28f30d64a4950be204c4751b25c194b92cde7972d` | Front matter |
| SRC-PAPER-01 | `protocol-contract/paper-sources/01-1-introduction.md` | `5aac1d36ba5a5f7cc998a5ed510a2d6a6dca9a598e66da70bac4d920dea591b9` | `6d6a04e1ac5e7d4d8d31a85206b04e41f81818ee061631a79ef78bc73caf6325` | Introduction and challenge-field statement |
| SRC-PAPER-02 | `protocol-contract/paper-sources/02-2-technical-overview.md` | `7119aba5e9ee194584cece91aaa4ff4280037a5ac784cf8fc1dd960549edb6c5` | `7119aba5e9ee194584cece91aaa4ff4280037a5ac784cf8fc1dd960549edb6c5` | Technical overview |
| SRC-PAPER-03 | `protocol-contract/paper-sources/03-3-overview-of-the-following-sections.md` | `bf9f24240669c23d81ba6b4c531741ccfa30ee57b456026f67fcf1ce87c4f152` | `bf9f24240669c23d81ba6b4c531741ccfa30ee57b456026f67fcf1ce87c4f152` | Section overview |
| SRC-PAPER-04 | `protocol-contract/paper-sources/04-4-preliminaries.md` | `b6b3b5ad0dfb4646bdcba814c3912c6efc2994aafef6ae64302e0ade6f4cfd25` | `94ca3c26d99c737b1834982048f5533f84fbe2712ae28c4b59cc2f5905c6b64b` | Fields, dimensions, norm, split, commitment, and reduction definitions |
| SRC-PAPER-05 | `protocol-contract/paper-sources/05-5-embedding-products-with-evaluation-homomorphism.md` | `946b4464700f730572c7b93d7249ee76088dea052222c2f0e3f93f8d8e2df855` | `946b4464700f730572c7b93d7249ee76088dea052222c2f0e3f93f8d8e2df855` | Coefficient embedding and evaluation homomorphism |
| SRC-PAPER-06 | `protocol-contract/paper-sources/06-6-strong-and-weak-interactive-reductions.md` | `69e3eb15db27c41a180d0f6fbe95aaf54d60276cb385aec7e959a67c3c560abc` | `69e3eb15db27c41a180d0f6fbe95aaf54d60276cb385aec7e959a67c3c560abc` | Strong and weak composition theorem |
| SRC-PAPER-07 | `protocol-contract/paper-sources/07-7-neo-s-folding-scheme-for-ccs.md` | `14ca44334a21a59b414c7bee9a56c5cde44cce7302c5eaa632e1a1327c784136` | `2ed776426ad25c37e9dfe8ee8970dc465605364d9557661ebb9eee6c75de0aed` | CCS, CE, PiCCS, PiRLC, and PiDEC |
| SRC-PAPER-08 | `protocol-contract/paper-sources/08-references.md` | `76829bc29328d816deb0647cb35baf7cdd20c6d86680335df226c9f7687c34c2` | `76829bc29328d816deb0647cb35baf7cdd20c6d86680335df226c9f7687c34c2` | References |
| SRC-PAPER-09 | `protocol-contract/paper-sources/09-supplementary-material.md` | `4377333f3baaeaae7f3f1553a2e15dec73086b3ef93cbdbaa4e4219e9e62d902` | `22ff49e20b2681f49310370057e42749039fc83e3ec6e2899abf1a092d30b6e7` | Reviewed errata and security changelog |
| SRC-PAPER-10 | `protocol-contract/paper-sources/10-a-ai-disclaimer.md` | `5906b91ed3a5c3ad4e8f63ca712604dd46a54caf7f1f7be8fb0ac10758bfa04a` | `5906b91ed3a5c3ad4e8f63ca712604dd46a54caf7f1f7be8fb0ac10758bfa04a` | Paper disclaimer |
| SRC-PAPER-11 | `protocol-contract/paper-sources/11-b-concrete-parameters.md` | `1505d203b2a07dc339c8c3c652b0cd1fa5f64bb89deb0e34923cdee659bd11ad` | `7628750679e2a765f2682e98992c51adf109701f5edfad1c695190d1fde4a0c1` | Concrete parameter profiles |
| SRC-PAPER-12 | `protocol-contract/paper-sources/12-c-additional-background.md` | `c448c85a43d187d457bfb2cbbd09291721555b72986243ab51f8e4b95f48c406` | `23be1f2fbff571b2959d0c1d6a2f81445a8719176d276e9697defedced645f16` | Sampling, extraction, and Module-SIS definitions |
| SRC-PAPER-13 | `protocol-contract/paper-sources/13-d-deferred-theorems-and-proofs.md` | `727c471a324b6ea5d1b22a894119a91b4f5ed6e5b4bee87233d5166cd5d55c08` | `bb542f19749b44c037af2a430ed72460fc8fbb07c8a748ff8ac454a0f2a3c734` | Deferred proofs and concrete losses |
| SRC-PAPER-INDEX | `protocol-contract/paper-sources/INDEX.md` | `ac7e5d671fa7a2519f238bed2118bd56d49bd5450914f05938dfe9a3c68ae6cf` | `f828403728c0d40ec9bad122a73311b1a0bdd443219592c46025b0d18cb9c3b8` | Paper index and source inventory |
| SRC-ERRATA-V4 | `protocol-contract/paper-sources/superneo-paper-errata-v4.patch` | `de684304c5be073ab5b3f50eabfd39f98f4d8900840a77de6d8d9dc2d766a5c9` | `not-applicable` | Exact base-to-reviewed patch |

## Derivation rule

The checker reverse-applies `SRC-ERRATA-V4` without fuzz
or path substitution. It requires every reconstructed base hash. It then
applies the patch forward and requires byte equality with the reviewed
files. An unchanged file has equal base and reviewed hashes.

The external v3 archive has SHA-256
`d2d7f0864d2fa717ee5ded0898b46921f3c817089463700d7d7df097dd5e8636`.
It reconstructs the same base snapshot. It is not normative because this
repository uses the later reviewed v4 patch and source bytes.

Nightstream decisions are not paper sources. Their authority is in
`src/decisions/decisions.jsonl`; `deviations.md` is a generated view.
