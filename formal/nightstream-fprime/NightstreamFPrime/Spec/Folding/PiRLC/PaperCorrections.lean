import NightstreamFPrime.Spec.Profile
import NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.Norm.Centered

/-!
Selected ambient norm boundary and historical obstruction for SuperNeo v1.2
Appendix B.3.

Owns: the selected `B_amb = floor(q / 2) + 1` bound, the concrete obstruction
to the older strict `q / 2` bound, and the smallest strict natural bound that
contains every centered Goldilocks residue.

The corrected Appendix B.3 selects `B_amb = floor(q / 2) + 1`.
-/

namespace NightstreamFPrime.Spec.Folding.PiRLC.PaperCorrections

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.Norm

universe uStructure uAssignment uPublicInput uPoint uEvaluation uCommitment

def correctedAmbientBoundFor (params : NightstreamFPrime.Spec.GlobalParams) : Nat :=
  params.ambientBound

theorem ambientStageBound_eq_correctedAmbientBoundFor
    (params : NightstreamFPrime.Spec.GlobalParams) :
    NightstreamFPrime.Spec.NormStage.bound params .ambient =
      correctedAmbientBoundFor params := by
  rfl

def CorrectedAmbientHolds
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    (semantics : NightstreamFPrime.Spec.RelationSemantics
      Structure Assignment PublicInput Point Evaluation Commitment)
    (params : NightstreamFPrime.Spec.GlobalParams)
    (statement : NightstreamFPrime.Spec.CE.Instance
      Structure PublicInput Point Evaluation Commitment)
    (assignment : Assignment) : Prop :=
  NightstreamFPrime.Spec.Opening.Holds semantics
      (correctedAmbientBoundFor params)
      statement.commitment statement.publicInput assignment ∧
    semantics.evaluationPointValid statement.constraintSystem statement.point ∧
    semantics.evaluations statement.constraintSystem assignment statement.point =
      statement.evaluations

def literalAmbientBound : Nat := goldilocksModulus / 2

def correctedAmbientBound : Nat := literalAmbientBound + 1

theorem production_correctedAmbientBoundFor_eq :
    correctedAmbientBoundFor
      NightstreamFPrime.Spec.productionGlobalParams =
      correctedAmbientBound := by
  rfl

def midpointResidue : F :=
  ⟨Centered.halfModulus, by
    unfold Centered.halfModulus goldilocksModulus
    decide⟩

theorem centeredMagnitude_midpointResidue :
    centeredMagnitude midpointResidue = literalAmbientBound := by
  rw [Centered.centeredMagnitude_eq_distance]
  simp [midpointResidue, literalAmbientBound, Centered.distance,
    Centered.halfModulus]

theorem midpointResidue_not_literalAmbientBounded :
    ¬ centeredMagnitude midpointResidue < literalAmbientBound := by
  rw [centeredMagnitude_midpointResidue]
  exact Nat.lt_irrefl literalAmbientBound

theorem all_centeredMagnitude_lt_correctedAmbientBound (value : F) :
    centeredMagnitude value < correctedAmbientBound := by
  rw [Centered.centeredMagnitude_eq_distance]
  change Centered.distance value.val < Centered.halfModulus + 1
  unfold Centered.distance
  by_cases low : value.val ≤ Centered.halfModulus
  · rw [if_pos low]
    omega
  · rw [if_neg low]
    have modulusIdentity := Centered.modulus_eq_two_half_add_one
    have valueBound := value.isLt
    omega

end NightstreamFPrime.Spec.Folding.PiRLC.PaperCorrections
