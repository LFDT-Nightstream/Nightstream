# SingleRound — Sum-Check Proof-System Facade

## Purpose

- **What it is**: Proof-system namespace that re-exports `SuperNeo.SumCheck` types and predicates under `ProofSystem.Sumcheck.SingleRound`, plus forwarding theorems for `Accepted` → structural properties (rounds, challenges, fold steps, initial round, round-sum step) and rejection lemmas.
- **Key property**: `Accepted inst tr` implies \(|\text{roundPolys}| = \text{rounds}\), \(|\text{challenges}| = |\text{roundPolys}|\), and round-consistency identities; rejection theorems give \(\neg\text{Accepted}\) from bad shapes or missing final-oracle witnesses.
- **Protocol role**: Single-round view of sum-check used by Π_CCS (Section 7.3); provides the theorem surface for proof-system composition.

## Target Formulas

- \(\text{Accepted}(\text{inst}, \text{tr}) \to |\text{roundPolys}| = \text{rounds}\).
- \(\text{Accepted}(\text{inst}, \text{tr}) \to |\text{challenges}| = |\text{roundPolys}|\).
- \(\text{Accepted}(\text{inst}, \text{tr}) \wedge i+1 < |\text{roundPolys}| \to \text{eval}(p_i, r_i) = p_{i+1}(0)\) (fold step).
- \(\text{Accepted}(\text{inst}, \text{tr}) \to \text{InitialRoundConsistent}(\text{inst}, \text{tr})\).
- \(\text{Accepted}(\text{inst}, \text{tr}) \wedge i+1 < |\text{roundPolys}| \to p_{i+1}(0) + p_{i+1}(1) = \text{eval}(p_i, r_i)\) (round-sum step).
- \(|\text{challenges}| \neq \text{rounds} \to \neg\text{Accepted}\); analogous for bad round shape, no final-oracle witness, bad initial round.

## Paper Anchors

- **Source**: `./formal/superneo-lean/SuperNeo.pdf.md`
- Definition 6 (The sum-check protocol), lines 352–355.
- Section 7.3 (Π_CCS), lines 481–548: sum-check invocation.

## Module Mapping

| Paper concept | Lean symbol | Role |
|---------------|-------------|--------|
| Instance / transcript | `Instance`, `Transcript` | Definitional (abbrev) |
| Round consistency | `RoundConsistent`, `InitialRoundConsistent` | Definitional (abbrev) |
| Acceptance / claim | `Accepted`, `ClaimTrue` | Definitional (abbrev) |
| Extraction | `accepted_rounds_eq`, `accepted_challenges_eq`, etc. | Theorem-Target |
| Rejection | `not_accepted_of_*` | Theorem-Target |

## Contract Surface

| Group | Symbol | Guarantee | Role |
|-------|--------|-----------|--------|
| Abbrevs | `Instance`, `Transcript`, `RoundConsistent`, `InitialRoundConsistent`, `Accepted`, `ClaimTrue` | Forward to `SuperNeo.SumCheck` | Definitional |
| Extraction | `accepted_rounds_eq`, `accepted_challenges_eq`, `accepted_fold_step`, `accepted_initial_round`, `accepted_round_sum_step` | Structural implications of `Accepted` | Theorem-Target |
| Rejection | `not_accepted_of_challenge_size_ne`, `not_accepted_of_roundpoly_size_ne`, `not_accepted_of_bad_round_shape`, `not_accepted_of_no_final_oracle_witness`, `not_accepted_of_bad_initial_round` | Negation from bad structure / endpoint-witness failure | Theorem-Target |

## Proof Obligations and Closure Plan

All extraction and rejection theorems proved by forwarding to `SuperNeo.SumCheck`.

## Assumption Ledger

No open boundary assumptions in this module.

## Dependency and Consumer Map

- **Dependencies**: imports `SuperNeo.SumCheck`.
- **Consumers**:
  - `SuperNeo.ProofSystem.SumCheck.General`: uses SingleRound theorems for forwarding and theorem package.
  - `SuperNeo.ProofSystem.SumCheck`: imports General (which imports SingleRound) for barrel.

## Implementation Plan

Keep as thin forwarding layer; no new proof work beyond forwarding.

## Quality Expectations

All theorems forward to core `SumCheck`; no duplication of proof logic.

## Acceptance Criteria

- `lake build` succeeds.
- Spec contains explicit paper anchors with line ranges.
- All extraction and rejection theorems proved.

## Out of Scope

- Soundness/completeness; those live in `General` and core `SumCheck`.
