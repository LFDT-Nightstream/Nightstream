import Nightstream.SuperNeo.CheckPlan
import Nightstream.SuperNeo.Concrete.Phi81Relation.Necessity
import Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.CanonicalParentVerifier

/-!
Inclusion-minimal semantic plan for the typed Phi81 canonical-parent opening.

Assurance tier: model-level obligation minimality.

Owns: the exact two-leaf plan for a deterministically materialized combined
parent; a concrete wrong-commitment acceptance when the commitment leaf is
removed; a concrete magnitude-`B` acceptance when the norm leaf is removed;
and inclusion-minimal soundness relative to the independent CE relation.

Does not own: production Ajtai binding, an R1CS gate lower bound, raw decoding,
Poseidon2, Rust/R1CS refinement, costs, or row removal.

Emits constraints: no.

Authority boundary: the small non-production Phi81 fixture witnesses semantic
need for each family uniformly over the shape-generic verifier. Its Boolean
commitment is not cryptographic. The norm witness uses the exact production
combined bound `B = 16_384`. Public inputs, evaluations, stage, point shape,
and PiDEC recomposition are absent from the plan because the canonical-parent
verifier computes or types them.

| Stage path | Retained family | Concrete removal witness | Lean owner |
|---|---|---|---|
| `nifs.pi_dec.canonical_opening.phi81.commitment` | opening equals carried parent commitment | carried `true`, computed Boolean commitment `false` | `commitment_necessary` |
| `nifs.pi_dec.canonical_opening.phi81.norm` | every carrier coefficient has magnitude `< B` | one coefficient has magnitude exactly `16_384` | `combinedNorm_necessary` |
| `nifs.pi_dec.canonical_opening.phi81.plan` | both checks and only both checks imply parent CE membership | exactness plus both witnesses | `plan_inclusionMinimalSound` |
-/

namespace Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.CanonicalParentVerifier.Minimality

open Nightstream.SuperNeo
open Nightstream.SuperNeo.CheckPlan
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Concrete.Phi81Relation.Necessity
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.PiDEC.CanonicalChildren.OpeningVerifier

inductive Family where
  | commitment
  | combinedNorm
deriving DecidableEq

abbrev Carrier := Input (Point witnessShape) Bool
abbrev PlanInput := Carrier × Assignment witnessShape

def familySemantics : Family → PlanInput → Prop
  | .commitment, input =>
      booleanCommitment input.2 = input.1.commitment
  | .combinedNorm, input =>
      assignmentNormBounded productionGlobalParams.bigB input.2

def target (input : PlanInput) : Prop :=
  CE.Holds (relationSemantics booleanCommitment) productionGlobalParams
    (parent (relationSemantics booleanCommitment)
      witnessSystem input.1 input.2) input.2

def plan : List Family := [.commitment, .combinedNorm]

/-- Every typed point is valid, so the two-leaf plan is exactly parent CE
membership for the computed statement. -/
theorem plan_exact : Exact familySemantics target plan := by
  intro input
  constructor
  · intro accepted
    apply parentHolds
    exact {
      commitment := accepted .commitment (by simp [plan])
      combinedNorm := accepted .combinedNorm (by simp [plan])
      pointValid := evaluationPointValid_holds witnessSystem input.1.point
    }
  · intro holds family _member
    have accepted := accepted_of_parentHolds holds
    cases family with
    | commitment => exact accepted.commitment
    | combinedNorm => exact accepted.combinedNorm

theorem plan_sound : Sound familySemantics target plan :=
  (exact_iff_sound_and_complete.mp plan_exact).1

/-! ## Commitment removal witness -/

def wrongCommitmentCarrier : Carrier where
  point := witnessPoint
  commitment := true

theorem witnessAssignment_combinedNorm :
    assignmentNormBounded productionGlobalParams.bigB witnessAssignment := by
  intro column
  have fresh := witnessAssignment_fresh_norm column
  change centeredMagnitude (witnessAssignment column) < 2 at fresh
  change centeredMagnitude (witnessAssignment column) < 16384
  omega

theorem wrongCommitment_not_target :
    ¬ target (wrongCommitmentCarrier, witnessAssignment) := by
  intro holds
  have accepted := accepted_of_parentHolds holds
  have impossible : (false : Bool) = true := by
    simpa [relationSemantics, booleanCommitment, wrongCommitmentCarrier] using
      accepted.commitment
  exact Bool.noConfusion impossible

theorem commitment_necessary :
    NecessaryForSoundness familySemantics target plan .commitment := by
  refine ⟨(wrongCommitmentCarrier, witnessAssignment), ?_,
    wrongCommitment_not_target⟩
  intro family member
  cases family with
  | commitment => simp [plan, without] at member
  | combinedNorm => exact witnessAssignment_combinedNorm

/-! ## Combined-norm removal witness -/

def combinedHighNormAssignment : Assignment witnessShape := fun column =>
  if column = carrierColumnZero then (16384 : F) else 0

theorem combinedHighNormAssignment_not_bounded :
    ¬ assignmentNormBounded productionGlobalParams.bigB
      combinedHighNormAssignment := by
  intro bounded
  have atZero := bounded carrierColumnZero
  change centeredMagnitude (16384 : F) < 16384 at atZero
  exact (by decide : ¬ centeredMagnitude (16384 : F) < 16384) atZero

def highNormCarrier : Carrier where
  point := witnessPoint
  commitment := false

theorem highNorm_not_target :
    ¬ target (highNormCarrier, combinedHighNormAssignment) := by
  intro holds
  have accepted := accepted_of_parentHolds holds
  exact combinedHighNormAssignment_not_bounded accepted.combinedNorm

theorem combinedNorm_necessary :
    NecessaryForSoundness familySemantics target plan .combinedNorm := by
  refine ⟨(highNormCarrier, combinedHighNormAssignment), ?_,
    highNorm_not_target⟩
  intro family member
  cases family with
  | commitment => rfl
  | combinedNorm => simp [plan, without] at member

/-- Both retained families have concrete invalid acceptances when removed. -/
theorem plan_inclusionMinimalSound :
    InclusionMinimalSound familySemantics target plan := by
  apply inclusionMinimalSound_of_witnesses plan_sound
  intro family _member
  cases family with
  | commitment => exact commitment_necessary
  | combinedNorm => exact combinedNorm_necessary

end Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.CanonicalParentVerifier.Minimality
