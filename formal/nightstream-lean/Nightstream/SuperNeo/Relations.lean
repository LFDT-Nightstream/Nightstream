/-!
Paper-level SuperNeo relation ownership.

This file owns the semantic shape of norm-bounded CCS and CE membership. The
algebra, commitment, projection, norm, constraint, and evaluation operations are
parameters until their concrete Goldilocks/Ajtai implementations are verified.
It does not own folding, recursion, transcript security, or Rust encodings.
-/

namespace Nightstream.SuperNeo

universe uStructure uAssignment uPublicInput uPoint uEvaluation uCommitment

/--
Verifier-owned global reduction parameters (SuperNeo Definition 14).

The verifier fixes `q`, `b`, `k`, `B = b^k`, the maximum fold arity, and the
challenge-set expansion factor `T` once per deployment; no statement may carry
its own bound. The Definition 14 inequality is stated at the maximum arity so
every smaller fold inherits it. Rust owns the same discipline at runtime
through `Params::q`, `Params::max_fresh_count`, and the Π_RLC bound check.
-/
structure GlobalParams where
  /-- Base-field modulus `q`; the ambient extraction bound is `q / 2`. -/
  q : Nat
  /-- Fresh-relation witness ∞-norm bound `b`. -/
  b : Nat
  /-- Decomposition arity `k` — the running-accumulator width `CE(b)^k`. -/
  k : Nat
  /-- Maximum fresh instances `K_max` folded in one step. -/
  maxFresh : Nat
  /-- Challenge-set expansion factor `T` (Theorem 9). -/
  expansionT : Nat
  /-- Definition 14: `(K + k) · T · (b − 1) < B = b^k` at maximum arity. -/
  rlc_bound : (maxFresh + k) * expansionT * (b - 1) < b ^ k

namespace GlobalParams

/-- `B = b^k`, the Π_RLC output bound. -/
def bigB (p : GlobalParams) : Nat := p.b ^ p.k

/--
The Module-SIS ∞-norm at which the binding assumption must be taken.

Definition 14 requires the commitment scheme to be `(2B, C)`-relaxed binding.
Theorem 2 gives `(B', C)`-relaxed binding from `MSIS` at ∞-norm `4·T·B'`;
substituting `B' = 2B` yields `MSIS` at `8TB` — the chain Appendix B states
explicitly. A `4TB` MSIS bound is NOT sufficient for the global regime.
-/
def msisNormBound (p : GlobalParams) : Nat := 8 * p.expansionT * p.bigB

end GlobalParams

/--
The norm stage a statement lives at. The pipeline moves through distinct
bounds and conflating them mis-states every fold theorem:

- `fresh`     — `CE(b)^(K+k)`: honest Π_CCS outputs and Π_DEC children;
- `combined`  — `CE(B)`: the Π_RLC output;
- `ambient`   — `CE(q/2)`: where Π_RLC's rewinding extraction lands (D.5);
  the post-decomposition relation does NOT stay at this weaker bound.

Bounds are derived from verifier-owned `GlobalParams`, never carried as free
per-statement data.
-/
inductive NormStage where
  | fresh
  | combined
  | ambient
deriving Repr, DecidableEq

/-- Resolve a stage to its verifier-owned bound. -/
def NormStage.bound (p : GlobalParams) : NormStage → Nat
  | .fresh => p.b
  | .combined => p.bigB
  | .ambient => p.q / 2

/-- Operations needed to state CCS and CE membership without hiding obligations. -/
structure RelationSemantics
    (Structure : Type uStructure)
    (Assignment : Type uAssignment)
    (PublicInput : Type uPublicInput)
    (Point : Type uPoint)
    (Evaluation : Type uEvaluation)
    (Commitment : Type uCommitment) where
  commit : Assignment → Commitment
  projectPublicInput : Assignment → PublicInput
  normBounded : Nat → Assignment → Prop
  ccsSatisfied : Structure → Assignment → Prop
  /-- Verifier-owned domain/length check for the CE evaluation point. A total
  evaluator alone is insufficient: otherwise a malformed point could name a
  different domain while still supplying its matching output array. -/
  evaluationPointValid : Structure → Point → Prop
  evaluations : Structure → Assignment → Point → Array Evaluation

namespace Opening

/-- The authority-bearing commitment, public-input, and norm obligations. -/
def Holds
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    (semantics : RelationSemantics
      Structure Assignment PublicInput Point Evaluation Commitment)
    (normBound : Nat)
    (commitment : Commitment)
    (publicInput : PublicInput)
    (assignment : Assignment) : Prop :=
  semantics.commit assignment = commitment ∧
  semantics.projectPublicInput assignment = publicInput ∧
  semantics.normBounded normBound assignment

/-- Two distinct bounded openings of one commitment. This is the common
model-level event that concrete commitment schemes must reduce to their
binding assumption; it does not itself assert computational hardness. -/
structure BindingCollision
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    (semantics : RelationSemantics
      Structure Assignment PublicInput Point Evaluation Commitment)
    (normBound : Nat)
    (commitment : Commitment) where
  leftOpening : Assignment
  rightOpening : Assignment
  leftCommits : semantics.commit leftOpening = commitment
  rightCommits : semantics.commit rightOpening = commitment
  leftNorm : semantics.normBounded normBound leftOpening
  rightNorm : semantics.normBounded normBound rightOpening
  different : leftOpening ≠ rightOpening

end Opening

namespace CCS

/-- Public norm-bounded CCS instance from SuperNeo Definition 12. The norm
bound is not instance data: statements name a `NormStage` and the bound is
resolved against verifier-owned `GlobalParams`. -/
structure Instance
    (Structure : Type uStructure)
    (PublicInput : Type uPublicInput)
    (Commitment : Type uCommitment) where
  constraintSystem : Structure
  commitment : Commitment
  publicInput : PublicInput
  stage : NormStage

/-- Actual CCS membership: opening authority plus satisfaction of the CCS relation. -/
def Holds
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    (semantics : RelationSemantics
      Structure Assignment PublicInput Point Evaluation Commitment)
    (params : GlobalParams)
    (statement : Instance Structure PublicInput Commitment)
    (assignment : Assignment) : Prop :=
  Opening.Holds semantics (statement.stage.bound params) statement.commitment
      statement.publicInput assignment ∧
    semantics.ccsSatisfied statement.constraintSystem assignment

end CCS

namespace CE

/-- Public norm-bounded CCS evaluation instance from SuperNeo Definition 13.
Norm bound via `NormStage` against verifier-owned `GlobalParams`, as in
`CCS.Instance`. -/
structure Instance
    (Structure : Type uStructure)
    (PublicInput : Type uPublicInput)
    (Point : Type uPoint)
    (Evaluation : Type uEvaluation)
    (Commitment : Type uCommitment) where
  constraintSystem : Structure
  commitment : Commitment
  publicInput : PublicInput
  point : Point
  evaluations : Array Evaluation
  stage : NormStage

/-- Actual CE membership: opening authority plus all claimed matrix evaluations. -/
def Holds
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    (semantics : RelationSemantics
      Structure Assignment PublicInput Point Evaluation Commitment)
    (params : GlobalParams)
    (statement : Instance Structure PublicInput Point Evaluation Commitment)
    (assignment : Assignment) : Prop :=
  Opening.Holds semantics (statement.stage.bound params) statement.commitment
      statement.publicInput assignment ∧
    semantics.evaluationPointValid statement.constraintSystem statement.point ∧
    semantics.evaluations statement.constraintSystem assignment statement.point =
      statement.evaluations

end CE

end Nightstream.SuperNeo
