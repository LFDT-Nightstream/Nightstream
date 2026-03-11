# CCS — Section 7.1 Relation Objects

## Purpose

- **What it is**: The proof-system realization of Section 7.1 relation objects: Definition 11 structures, Definition 12 norm-bounded CCS, Definition 13 norm-bounded CE, a relaxed CE carrier used by the folding stack, and the Definition 14 global-parameter package that instantiates those relations.
- **Key property**: The module exposes explicit statement/witness shapes and membership predicates for CCS and CE, together with a proved CE-to-relaxed-CE lift.
- **Protocol role**: This is the paper-facing relation layer that Section 7 reductions are supposed to consume before any compact protocol-specific specialization.

## Target Formulas

- **Definition 11 (`s`)**:
  - `CCSStructure` packages the matrix family together with semantic image/evaluation families and the semantic constraint predicate corresponding to
    `f(\bar(M_1 z), ..., \bar(M_t z)) ∈ Z^{log m}`.
- **Definition 12 (`CCS(b, L)`)**:
  - `CCS.Statement` represents `(c, x)`.
  - `CCS.Witness` represents `w`.
  - `CCS.fullVector stmt wit = [x, w]`.
  - `CCS.Holds ccs stmt wit` means:
    - `c = L(z)`
    - `||z||_∞ < b`
    - the structure-side constraint predicate holds on the encoded matrix images of `z`.
- **Definition 13 (`CE(b, L)`)**:
  - `CE.Statement` represents `(c, x, r, {y_j})`.
  - `CE.Witness` represents `z`.
  - `CE.Holds ce stmt wit` means:
    - `c = L(z)`
    - `x = L_in(z)`
    - `||z||_∞ < b`
    - `{y_j}` matches the structure-side evaluation family at point `r`.
- **Definition 14 (Global Reduction Parameters)**:
  - `GlobalParams` packages the challenge set, commitment map, input projector, and structure.
  - `GlobalParams.ccs`, `GlobalParams.ce`, and `GlobalParams.ceRelaxed` instantiate the Section 7.1 relation carriers from those shared parameters.
- **Coherent theorem-instance packaging**:
  - `Section71Objects` packages one shared Definition-14 parameter package, one norm bound, and one coherent CCS/CE tuple pair.
  - `Section71Instance` extends that package with concrete `CCS.Holds` / `CE.Holds` proofs.

## Paper Anchors

- **Source**: `./formal/superneo-lean/SuperNeo.pdf.md`
- Definition 11 (Structure), lines 449–455.
- Definition 12 (Norm-bounded CCS), lines 457–459.
- Definition 13 (Norm-bounded CCS Evaluation Relation), lines 461–465.
- Definition 14 (Global Reduction Parameters), lines 467–475.

## Module Mapping

| Paper concept | Lean symbol | Role |
|---|---|---|
| Structure `s` | `CCSStructure` | Definitional |
| Commitment map `L` | `CommitmentMap` | Definitional |
| Input projector `L_in` | `InputProjector` | Definitional |
| Global reduction parameters | `GlobalParams` | Definitional |
| Coherent Section 7.1 tuple package | `Section71Objects` | Boundary |
| Concrete Section 7.1 theorem instance | `Section71Instance` | Boundary |
| Norm-bounded CCS relation | `CCS`, `CCS.Statement`, `CCS.Witness`, `CCS.Holds` | Definitional/Theorem-target |
| Norm-bounded CE relation | `CE`, `CE.Statement`, `CE.Witness`, `CE.Holds` | Definitional/Theorem-target |
| Relaxed CE carrier | `CERelaxed`, `CERelaxed.Witness`, `CERelaxed.Holds` | Definitional/Theorem-target |
| CE → relaxed CE lift | `CERelaxed.ofCE`, `CERelaxed.holds_of_ce` | Theorem-target |

## Contract Surface

| Group | Symbol | Guarantee | Role |
|---|---|---|---|
| Structure | `CCSStructure` | Carries matrices, encoded image family, encoded evaluation family, and semantic constraint predicate | Definitional |
| Parameters | `GlobalParams` | Carries the shared challenge-set / commitment-map / projector / structure data from Definition 14 | Definitional |
| Instance data | `Section71Objects` | Carries one shared Definition-14 package, one norm bound, and one coherent CCS/CE tuple pair | Boundary |
| Instance data | `Section71Instance` | Adds concrete `CCS.Holds` / `CE.Holds` proofs to one coherent Section 7.1 tuple package | Boundary |
| CCS | `CCS.fullVector` | Reconstructs `z := [x, w]` | Definitional |
| CCS | `CCS.Holds` | Exact Section 7.1 CCS membership predicate over `(c, x; w)` | Theorem-Target |
| CE | `CE.Holds` | Exact Section 7.1 CE membership predicate over `(c, x, r, {y_j}; z)` | Theorem-Target |
| Relaxation | `CERelaxed.ofCE` | Promotes CE to relaxed CE with the same base norm bound as slack bound | Theorem-Target |
| Relaxation | `CERelaxed.holds_of_ce` | Any CE witness yields a relaxed CE witness using zero slack | Theorem-Target |
| Constructors | `GlobalParams.ccs`, `GlobalParams.ce`, `GlobalParams.ceRelaxed` | Canonical Section 7 relation carriers from Definition 14 data | Theorem-Target |

## Proof Obligations

- `CCS.Holds` and `CE.Holds` must expose the exact tuple-level membership conditions stated in Definitions 12 and 13.
- `CERelaxed.holds_of_ce` must prove that any CE witness can be reused as a relaxed CE witness with zero slack.
- `GlobalParams` constructors must instantiate the shared relation carriers without adding extra assumptions.
- `Section71Objects` must enforce that the CCS and CE tuples share the same commitment, public input, and witness vector data.

## Assumption Ledger

- This module introduces no theorem-level boundary assumptions.
- The semantic matrix-image, evaluation, and constraint families are carried explicitly as data of `CCSStructure`; specialization of those semantics to a concrete protocol context belongs downstream.
- `Section71Objects` / `Section71Instance` are paper-facing packaging boundaries, not compact-protocol specialization boundaries.

## Dependency and Consumer Map

- **Dependencies**:
  - `SuperNeo/Norm.lean`: `normInfCoeffs`.
- **Consumers**:
  - `SuperNeo.ProofSystem.ConstraintSystem`: barrel re-export.
  - `SuperNeo.ProtocolRelations`: specializes compact protocol contexts to these Section 7.1 objects.
  - `SuperNeo.ProofSystem.Folding`: intended proof-system consumer of CCS/CE relation objects.

## Quality Expectations

- Relation predicates remain small and explicit.
- Statement/witness shapes match the paper tuples directly.
- Specialization to compact protocol glue happens outside this module.

## Acceptance Criteria

- `lake build` succeeds.
- No `sorry`.
- Definition 11–14 surfaces are present as typed Lean objects.
- CE-to-relaxed-CE lift is proved.

## Out of Scope

- Protocol-specific realization of these relation objects from `ProtocolTargetContext`.
- Π_CCS / Π_RLC / Π_DEC soundness theorems.
