# Contract source registry

Status: **mechanically locked; semantic normalization open**.

This file is generated from `src/sources/lock.toml`. Each source ID
names one exact reviewed byte string. SHA-256 identifies artifacts
only. It is not protocol authority.

Repository commit: `26df9e74f54286996c3565394dd18a900c9d4480`.

| Source ID | Path | Reviewed SHA-256 | Base SHA-256 | Role |
|---|---|---|---|---|
| SRC-PAPER-00 | `protocol-contract/paper-sources/00-front-matter.md` | `5ff28783c59736b8a3ff6b8a4669ca2cd532115c63752780d29d3c606bebc99b` | `a87c04df841d749d5832adf28f30d64a4950be204c4751b25c194b92cde7972d` | Front matter |
| SRC-PAPER-01 | `protocol-contract/paper-sources/01-1-introduction.md` | `606d543cdc6dbf8f2ec080c4450df08180c7f9be65b169d153061d01e979e900` | `6d6a04e1ac5e7d4d8d31a85206b04e41f81818ee061631a79ef78bc73caf6325` | Introduction and challenge-field statement |
| SRC-PAPER-02 | `protocol-contract/paper-sources/02-2-technical-overview.md` | `5158c3db404298e88bf52b7b2e0585f7c9b8fc8d03c134f33a06e5b0e6f27559` | `7119aba5e9ee194584cece91aaa4ff4280037a5ac784cf8fc1dd960549edb6c5` | Technical overview |
| SRC-PAPER-03 | `protocol-contract/paper-sources/03-3-overview-of-the-following-sections.md` | `bf9f24240669c23d81ba6b4c531741ccfa30ee57b456026f67fcf1ce87c4f152` | `bf9f24240669c23d81ba6b4c531741ccfa30ee57b456026f67fcf1ce87c4f152` | Section overview |
| SRC-PAPER-04 | `protocol-contract/paper-sources/04-4-preliminaries.md` | `c45f74c944e2e0efe68f57271bcbc11af8975cbc99137003045ed9455347db3d` | `94ca3c26d99c737b1834982048f5533f84fbe2712ae28c4b59cc2f5905c6b64b` | Fields, dimensions, norm, split, commitment, and reduction definitions |
| SRC-PAPER-05 | `protocol-contract/paper-sources/05-5-embedding-products-with-evaluation-homomorphism.md` | `43d084a85ae746b74485c1132e181000dd8b14218f3d8c0536b4d730e2eaeca5` | `946b4464700f730572c7b93d7249ee76088dea052222c2f0e3f93f8d8e2df855` | Coefficient embedding and evaluation homomorphism |
| SRC-PAPER-06 | `protocol-contract/paper-sources/06-6-strong-and-weak-interactive-reductions.md` | `b04cb17d133d1fd81680c3f3f5a84e36564e71a12cfd9276519b87228482abb2` | `69e3eb15db27c41a180d0f6fbe95aaf54d60276cb385aec7e959a67c3c560abc` | Strong and weak composition theorem |
| SRC-PAPER-07 | `protocol-contract/paper-sources/07-7-neo-s-folding-scheme-for-ccs.md` | `46a68727c2abfb5b856517a831bfa8b6f625bf508ae9dd9694a9b33e2e49fbde` | `2ed776426ad25c37e9dfe8ee8970dc465605364d9557661ebb9eee6c75de0aed` | CCS, CE, PiCCS, PiRLC, and PiDEC |
| SRC-PAPER-08 | `protocol-contract/paper-sources/08-references.md` | `4695fdf5e770dec2c5fda6abb59ddbf2854df867e29b22d661bc58a2ee290c46` | `76829bc29328d816deb0647cb35baf7cdd20c6d86680335df226c9f7687c34c2` | References |
| SRC-PAPER-09 | `protocol-contract/paper-sources/09-supplementary-material.md` | `88a418556269efdd4ea83a267fc8080358517110b0bc8b027b3539dabe883854` | `22ff49e20b2681f49310370057e42749039fc83e3ec6e2899abf1a092d30b6e7` | Reviewed errata and security changelog |
| SRC-PAPER-10 | `protocol-contract/paper-sources/10-a-ai-disclaimer.md` | `5906b91ed3a5c3ad4e8f63ca712604dd46a54caf7f1f7be8fb0ac10758bfa04a` | `5906b91ed3a5c3ad4e8f63ca712604dd46a54caf7f1f7be8fb0ac10758bfa04a` | Paper disclaimer |
| SRC-PAPER-11 | `protocol-contract/paper-sources/11-b-concrete-parameters.md` | `8d4f3dc3ab252bf7ee17bf383c1f23679124c356a8729f0592af140b703e3bf5` | `7628750679e2a765f2682e98992c51adf109701f5edfad1c695190d1fde4a0c1` | Concrete parameter profiles |
| SRC-PAPER-12 | `protocol-contract/paper-sources/12-c-additional-background.md` | `f7164435ed2d16fc1d9a3fd6dbe1163009425f9128edaebb1fc2e1768b791ac3` | `23be1f2fbff571b2959d0c1d6a2f81445a8719176d276e9697defedced645f16` | Sampling, extraction, and Module-SIS definitions |
| SRC-PAPER-13 | `protocol-contract/paper-sources/13-d-deferred-theorems-and-proofs.md` | `37131dd724623d9599ff222c7f143182e04fc73e8dae69cb954c7cb253dd24cf` | `bb542f19749b44c037af2a430ed72460fc8fbb07c8a748ff8ac454a0f2a3c734` | Deferred proofs and concrete losses |
| SRC-PAPER-INDEX | `protocol-contract/paper-sources/INDEX.md` | `dd6ac6832d8f933928806357a8157d26033448ad594e5838665275205fd3b128` | `f828403728c0d40ec9bad122a73311b1a0bdd443219592c46025b0d18cb9c3b8` | Paper index and source inventory |
| SRC-ERRATA-V5-PART1 | `protocol-contract/paper-sources/superneo-paper-errata-v5-part1.patch` | `246eac3715133274279f41199cf63145b4b6184ad2700959c4f582a764384d82` | `not-applicable` | Exact base-to-reviewed patch, part 1 of 2 |
| SRC-ERRATA-V5-PART2 | `protocol-contract/paper-sources/superneo-paper-errata-v5-part2.patch` | `7f8eaf68a7b8015a1a9c5057e1c4d593a192c5dd57c7a659b4dbb52c7d127384` | `not-applicable` | Exact base-to-reviewed patch, part 2 of 2 |

## Derivation rule

The checker concatenates and reverse-applies `SRC-ERRATA-V5-PART1`, `SRC-ERRATA-V5-PART2` without fuzz
or path substitution. It requires every reconstructed base hash. It then
applies the patch forward and requires byte equality with the reviewed
files. An unchanged file has equal base and reviewed hashes.

The external v3 archive has SHA-256
`d2d7f0864d2fa717ee5ded0898b46921f3c817089463700d7d7df097dd5e8636`.
It reconstructs the same base snapshot. It is not normative because this
repository uses the later reviewed v5 patch and source bytes.

Nightstream decisions are not paper sources. Their authority is in
`src/decisions/decisions.jsonl`; `deviations.md` is a generated view.
