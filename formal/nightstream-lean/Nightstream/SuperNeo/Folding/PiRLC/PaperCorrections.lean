import Nightstream.SuperNeo.Concrete.Parameters
import Nightstream.SuperNeo.Concrete.Phi81Relation.PiRLCAlgebra.Norm.Centered

/-!
Frozen correction for the ambient norm boundary in SuperNeo Appendix D.5.

Owns: the concrete obstruction to the literal strict `q / 2` ambient bound
and the smallest strict natural bound containing every centered Goldilocks
residue.

Does not own: the Pi_RLC extractor, a commitment reduction, probability,
Fiat--Shamir, Rust, R1CS, or costs.

Emits constraints: no.

Appendix D.5 says that every field element satisfies the ambient `q / 2`
bound. Definition 12 uses a strict norm inequality, however, and the
Goldilocks modulus is odd. The midpoint residues have centered magnitude
`floor(q / 2)`, so the literal claim is false. Keeping the strict relation,
the least corrected natural bound is `floor(q / 2) + 1`.
-/

namespace Nightstream.SuperNeo.Folding.PiRLC.PaperCorrections

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation.PiRLCAlgebra.Norm

universe uStructure uAssignment uPublicInput uPoint uEvaluation uCommitment

/-- Corrected strict ambient bound for arbitrary paper parameters. -/
def correctedAmbientBoundFor (params : Nightstream.SuperNeo.GlobalParams) : Nat :=
  params.q / 2 + 1

/-- The corrected ambient CE relation shared literally by Pi_CCS's relaxed
target and Pi_RLC's relaxed source. The statement's stage is intentionally
irrelevant: this relation owns its verifier-derived corrected bound directly. -/
def CorrectedAmbientHolds
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    (semantics : Nightstream.SuperNeo.RelationSemantics
      Structure Assignment PublicInput Point Evaluation Commitment)
    (params : Nightstream.SuperNeo.GlobalParams)
    (statement : Nightstream.SuperNeo.CE.Instance
      Structure PublicInput Point Evaluation Commitment)
    (assignment : Assignment) : Prop :=
  Nightstream.SuperNeo.Opening.Holds semantics
      (correctedAmbientBoundFor params)
      statement.commitment statement.publicInput assignment ∧
    semantics.evaluationPointValid statement.constraintSystem statement.point ∧
    semantics.evaluations statement.constraintSystem assignment statement.point =
      statement.evaluations

/-- Appendix D.5's literal strict ambient bound. -/
def literalAmbientBound : Nat := goldilocksModulus / 2

/-- Smallest strict natural bound containing every centered residue. -/
def correctedAmbientBound : Nat := literalAmbientBound + 1

/-- The generic corrected paper bound specializes exactly to the production
Goldilocks correction. -/
theorem production_correctedAmbientBoundFor_eq :
    correctedAmbientBoundFor
      Nightstream.SuperNeo.Concrete.productionGlobalParams =
      correctedAmbientBound := by
  rfl

/-- One canonical residue at the lower centered midpoint. -/
def midpointResidue : F :=
  ⟨Centered.halfModulus, by
    unfold Centered.halfModulus goldilocksModulus
    decide⟩

/-- The midpoint has exactly the magnitude excluded by the literal strict
ambient relation. -/
theorem centeredMagnitude_midpointResidue :
    centeredMagnitude midpointResidue = literalAmbientBound := by
  rw [Centered.centeredMagnitude_eq_distance]
  simp [midpointResidue, literalAmbientBound, Centered.distance,
    Centered.halfModulus]

/-- Kernel-checked obstruction to Appendix D.5's universal-coverage claim. -/
theorem midpointResidue_not_literalAmbientBounded :
    ¬ centeredMagnitude midpointResidue < literalAmbientBound := by
  rw [centeredMagnitude_midpointResidue]
  exact Nat.lt_irrefl literalAmbientBound

/-- Every Goldilocks residue satisfies the corrected strict bound. -/
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

end Nightstream.SuperNeo.Folding.PiRLC.PaperCorrections
