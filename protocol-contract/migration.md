# Contract migration and ID reconciliation

Status: **provenance map only**. Old and external IDs are not normative aliases.

This file joins the first local draft with the supplied external extension. The
canonical IDs are the headings in `superneo-v1.md` and the decision rows in
the typed decision source `src/decisions/decisions.jsonl`. The generated
`deviations.md` file is a reader view only.

## Source migration

| Input | Result |
|---|---|
| First local draft | Kept the reviewed repository bytes, typed Nightstream carrier review, code-level correspondence findings, and stronger Rust-origin rule. |
| External archive `d2d7f0864d2fa717ee5ded0898b46921f3c817089463700d7d7df097dd5e8636` | Accepted the exact source-derivation method, errata granularity, obligation DAG, beta-coin rule, encoding profile, circuit directions, and evidence formats. |
| External errata v3 layer | Replaced by repository errata v4. Exact reverse application of v4 reconstructs the same base paper bytes. |
| External `claimed-unverified` repository state | Replaced by the local code review. Open edges remain open because source presence is not conformance proof. |

## Important ID migration

| Earlier or external ID | Canonical local ID | Meaning |
|---|---|---|
| `SN-AMB-001` in the old deviation ledger | `NSD-SPLIT-001`, `SN-SPLIT`, `NS-SPLIT-BINARY` | Deterministic decomposition choice |
| `SN-AMB-001` in the old normative table | `SN-NORM` | Universal ambient bound |
| `SN-NORM-002` | `SN-NORM` | Reviewed strict ambient bound |
| `SN-PICCS-DEG-001` | `SN-PICCS-DEGREE` | Joint individual-degree and error terms |
| `SN-PICCS-SEC-001` | `SN-PICCS-EXTRACTION` | Success-gated strong extraction |
| `SN-PIRLC-SEC-001` | `SN-PIRLC-EXTRACTION` | Coordinate-fork and binding boundary |
| `NS-PICCS-SPLIT-001..002` | `NS-PICCS-RECT` | Rectangular FE and NC protocol |
| `NS-NORM-CARRIER-001` | `NS-REL-CE-EXT`, `NS-PICCS-COLUMN-REPLAY` | Typed authoritative carrier and replay |
| `NS-PICCS-COINS-001` | `NS-PICCS-COINS` | `beta_a`, `beta_r`, and `beta_m` obligations |
| `NS-PROFILE-003` | `NS-PROFILE-ENCODING` | Algebra, encoding, and framing profile |
| `NS-FS-001..003` | `NS-FS-POSEIDON2` | Ordered transcript and Fiat--Shamir boundary |
| `NS-CIRCUIT-001..002` | `NS-CIRCUIT-CORRESPONDENCE` | Completeness, soundness, and public-input injectivity |
| `NS-SEC-001..002` | `NS-SEC-CONCRETE` | Concrete end-to-end security experiment |

## Rule-ID policy

Every source, paper-extraction item, erratum, Nightstream decision, normative
rule, and release obligation must have one globally unique identifier. A
mapping row does not make two IDs interchangeable. A proof or artifact must use
the canonical ID for the contract version that it binds.

## Maintainability migration

The current architecture migration keeps the submitted protocol meaning but
changes file ownership.

The immutable import record is
`src/migration/legacy-v1-baseline.json`. The completed gate receipt is
`src/migration/legacy-v1-receipt.json`. They record:

- every normative rule block digest;
- every literal paper-item digest;
- every decision row;
- every coverage row;
- every assurance node, edge, evidence list, blocker, and old status;
- every old assurance artifact and all nine expanded scope lists;
- every release gate;
- every protocol event and challenge blocker.

The completed receipt records that the imported source model matched this
baseline at the migration boundary. The assembled normative contract, literal
paper model, and coverage map were byte-identical to the pre-migration files.

Use `check_contract.py --package-only --verify-import` to audit the sealed
baseline structure, hashes, census, and receipt. The command does not compare
the current protocol semantics with the superseded draft. This rule lets later
decisions and evidence change without invalidating completed migration evidence.

The old assurance status had four values in one field. The import maps it as
follows:

| Old value | New leaf evidence state | Other information |
|---|---|---|
| `closed` | `complete` | closure is derived |
| `partial` | `partial` | dependency and blockers are derived |
| `open` | `none` | old display value stays in the baseline |
| `blocked` | `none` | old display value stays in the baseline |

The active model does not preserve the ambiguous `open` versus `blocked`
choice as an authored fact. It derives a display value from dependencies and
stable blockers.

The old rollups copied 229 rule references. The new rollups use typed scope
selectors. The lossless gate expands each selector and compares it with the
exact old list.

The 42 imported normative IDs are still coarse rule groups. This step is
intentional. `NSD-NORM-BINDING-001` changes the PiCCS rule set. The protocol
owner must select that branch before the affected groups become atomic
requirements. The independent G0B review occurs after that split.
