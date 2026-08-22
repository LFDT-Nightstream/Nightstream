/-!
Owns the semantic shape of norm-bounded CCS and CE membership (SuperNeo
Definitions 12–14) and the verifier-owned global parameters. The algebra,
commitment, projection, norm, and evaluation operations are the
`RelationSemantics` record; `Spec.Profile` instantiates it. Does not own
folding, transcripts, or circuits.

Provenance: copied from
`formal/nightstream-lean/Nightstream/SuperNeo/Relations.lean` at commit
`fb7a8a99aefbb8ebb5474681ecf80f1b95a1b7a2`; doc strings shortened, definitions unchanged.
-/

namespace NightstreamFPrime.Spec

universe uStructure uAssignment uPublicInput uPoint uEvaluation uCommitment

/-- Verifier-owned global reduction parameters (SuperNeo Definition 14).
The inequality is stated at maximum arity so every smaller fold inherits it. -/
structure GlobalParams where
  q : Nat
  b : Nat
  k : Nat
  maxFresh : Nat
  expansionT : Nat
  rlc_bound : (maxFresh + k) * expansionT * (b - 1) < b ^ k

namespace GlobalParams

/-- `B = b^k`, the Π_RLC output bound. -/
def bigB (p : GlobalParams) : Nat := p.b ^ p.k

/-- Least strict bound containing every centered residue of odd `q`. -/
def ambientBound (p : GlobalParams) : Nat := p.q / 2 + 1

/-- MSIS ∞-norm at which (2B, C)-relaxed binding must be assumed (Appendix B). -/
def msisNormBound (p : GlobalParams) : Nat := 8 * p.expansionT * p.bigB

theorem rlc_bound_for (p : GlobalParams) {fresh : Nat}
    (hFresh : fresh ≤ p.maxFresh) :
    (fresh + p.k) * p.expansionT * (p.b - 1) < p.bigB := by
  have hsum : fresh + p.k ≤ p.maxFresh + p.k :=
    Nat.add_le_add_right hFresh p.k
  have hscaled : (fresh + p.k) * p.expansionT ≤
      (p.maxFresh + p.k) * p.expansionT :=
    Nat.mul_le_mul_right p.expansionT hsum
  have hfull : (fresh + p.k) * p.expansionT * (p.b - 1) ≤
      (p.maxFresh + p.k) * p.expansionT * (p.b - 1) :=
    Nat.mul_le_mul_right (p.b - 1) hscaled
  exact Nat.lt_of_le_of_lt hfull p.rlc_bound

end GlobalParams

/-- Norm stage of a statement: `fresh` = `CE(b)`, `combined` = `CE(B)`,
`ambient` = `CE(⌊q/2⌋+1)` where rewinding extraction lands. -/
inductive NormStage where
  | fresh
  | combined
  | ambient
deriving Repr, DecidableEq

def NormStage.bound (p : GlobalParams) : NormStage → Nat
  | .fresh => p.b
  | .combined => p.bigB
  | .ambient => p.ambientBound

/-- Operations needed to state CCS and CE membership. -/
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
  /-- Verifier-owned domain check for the CE evaluation point. -/
  evaluationPointValid : Structure → Point → Prop
  evaluations : Structure → Assignment → Point → Array Evaluation

variable {Structure : Type uStructure} {Assignment : Type uAssignment}
  {PublicInput : Type uPublicInput} {Point : Type uPoint}
  {Evaluation : Type uEvaluation} {Commitment : Type uCommitment}

namespace Opening

/-- Commitment, public-input, and norm obligations of one opening. -/
def Holds
    (semantics : RelationSemantics
      Structure Assignment PublicInput Point Evaluation Commitment)
    (normBound : Nat) (commitment : Commitment) (publicInput : PublicInput)
    (assignment : Assignment) : Prop :=
  semantics.commit assignment = commitment ∧
  semantics.projectPublicInput assignment = publicInput ∧
  semantics.normBounded normBound assignment

/-- Two distinct bounded openings of one commitment: the model-level event a
concrete scheme must reduce to its binding assumption. -/
structure BindingCollision
    (semantics : RelationSemantics
      Structure Assignment PublicInput Point Evaluation Commitment)
    (normBound : Nat) (commitment : Commitment) where
  leftOpening : Assignment
  rightOpening : Assignment
  leftCommits : semantics.commit leftOpening = commitment
  rightCommits : semantics.commit rightOpening = commitment
  leftNorm : semantics.normBounded normBound leftOpening
  rightNorm : semantics.normBounded normBound rightOpening
  different : leftOpening ≠ rightOpening

end Opening

namespace CCS

/-- Norm-bounded CCS instance (Definition 12). -/
structure Instance
    (Structure : Type uStructure) (PublicInput : Type uPublicInput)
    (Commitment : Type uCommitment) where
  constraintSystem : Structure
  commitment : Commitment
  publicInput : PublicInput
  stage : NormStage

def Holds
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

/-- Norm-bounded CCS evaluation instance (Definition 13). -/
structure Instance
    (Structure : Type uStructure) (PublicInput : Type uPublicInput)
    (Point : Type uPoint) (Evaluation : Type uEvaluation)
    (Commitment : Type uCommitment) where
  constraintSystem : Structure
  commitment : Commitment
  publicInput : PublicInput
  point : Point
  evaluations : Array Evaluation
  stage : NormStage

def Holds
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

end NightstreamFPrime.Spec
