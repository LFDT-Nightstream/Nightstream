import NightstreamFPrime.Spec.Profile
import NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.Norm.Centered

/-! Provenance: copied from `formal/nightstream-lean/Nightstream/SuperNeo/Folding/PiRLC/PaperCorrections.lean`
at commit `fb7a8a99aefbb8ebb5474681ecf80f1b95a1b7a2`; namespaces renamed, otherwise unchanged. -/

/-!
Selected ambient norm boundary and historical obstruction for SuperNeo
Appendix D.5.

Owns: the selected `B_amb = floor(q / 2) + 1` bound, the concrete obstruction
to the older strict `q / 2` bound, and the smallest strict natural bound that
contains every centered Goldilocks residue.

Does not own: the Pi_RLC extractor, a commitment reduction, probability,
Fiat--Shamir, Rust, R1CS, or costs.

Emits constraints: no.

The corrected Appendix D.5 selects `B_amb = floor(q / 2) + 1`. The older
strict `q / 2` form cannot contain the midpoint residues because the
Goldilocks modulus is odd. This module records both the selected bound and the
kernel-checked historical obstruction.
-/

namespace NightstreamFPrime.Spec.Folding.PiRLC.PaperCorrections

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.Norm

universe uStructure uAssignment uPublicInput uPoint uEvaluation uCommitment

/-- Selected strict ambient bound for arbitrary paper parameters. -/
def correctedAmbientBoundFor (params : NightstreamFPrime.Spec.GlobalParams) : Nat :=
  params.ambientBound

/-- The shared ambient stage uses the corrected strict bound. -/
theorem ambientStageBound_eq_correctedAmbientBoundFor
    (params : NightstreamFPrime.Spec.GlobalParams) :
    NightstreamFPrime.Spec.NormStage.bound params .ambient =
      correctedAmbientBoundFor params := by
  rfl

/-- The selected ambient CE relation shared literally by Pi_CCS's relaxed
target and Pi_RLC's relaxed source. The statement's stage is intentionally
irrelevant: this relation owns its verifier-derived bound directly. -/
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

/-- On an ambient-stage statement, the explicit relaxed relation and the
shared `CE.Holds` relation are the same relation. -/
theorem correctedAmbientHolds_iff_ceHolds_of_ambient
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
    (assignment : Assignment)
    (ambient : statement.stage = .ambient) :
    CorrectedAmbientHolds semantics params statement assignment ↔
      NightstreamFPrime.Spec.CE.Holds semantics params statement assignment := by
  simp [CorrectedAmbientHolds, NightstreamFPrime.Spec.CE.Holds,
    correctedAmbientBoundFor, ambient, NightstreamFPrime.Spec.NormStage.bound,
    NightstreamFPrime.Spec.GlobalParams.ambientBound]

/-- Historical uncorrected strict ambient bound. -/
def literalAmbientBound : Nat := goldilocksModulus / 2

/-- Smallest strict natural bound containing every centered residue. -/
def correctedAmbientBound : Nat := literalAmbientBound + 1

/-- The generic corrected-paper bound specializes exactly to the selected
production Goldilocks bound. -/
theorem production_correctedAmbientBoundFor_eq :
    correctedAmbientBoundFor
      NightstreamFPrime.Spec.productionGlobalParams =
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

/-- Kernel-checked obstruction to the historical strict-bound coverage claim. -/
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

end NightstreamFPrime.Spec.Folding.PiRLC.PaperCorrections
