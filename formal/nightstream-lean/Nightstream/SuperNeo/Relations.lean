/-!
Paper-level SuperNeo relation ownership.

This file owns the semantic shape of norm-bounded CCS and CE membership. The
algebra, commitment, projection, norm, constraint, and evaluation operations are
parameters until their concrete Goldilocks/Ajtai implementations are verified.
It does not own folding, recursion, transcript security, or Rust encodings.
-/

namespace Nightstream.SuperNeo

universe uStructure uAssignment uPublicInput uPoint uEvaluation uCommitment

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

end Opening

namespace CCS

/-- Public norm-bounded CCS instance from SuperNeo Definition 12. -/
structure Instance
    (Structure : Type uStructure)
    (PublicInput : Type uPublicInput)
    (Commitment : Type uCommitment) where
  constraintSystem : Structure
  commitment : Commitment
  publicInput : PublicInput
  normBound : Nat

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
    (statement : Instance Structure PublicInput Commitment)
    (assignment : Assignment) : Prop :=
  Opening.Holds semantics statement.normBound statement.commitment
      statement.publicInput assignment ∧
    semantics.ccsSatisfied statement.constraintSystem assignment

end CCS

namespace CE

/-- Public norm-bounded CCS evaluation instance from SuperNeo Definition 13. -/
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
  normBound : Nat

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
    (statement : Instance Structure PublicInput Point Evaluation Commitment)
    (assignment : Assignment) : Prop :=
  Opening.Holds semantics statement.normBound statement.commitment
      statement.publicInput assignment ∧
    semantics.evaluations statement.constraintSystem assignment statement.point =
      statement.evaluations

end CE

end Nightstream.SuperNeo
