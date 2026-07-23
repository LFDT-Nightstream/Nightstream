import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Metadata
import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Provenance
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.Carrier270.PhysicalPublicAssignment
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.Carrier270.Selectors

/-!
Physical steady-arm selector contract for the production fixed-point profile.

Owns: interpretation of the Rust-generated combined-NC source-arm index as
the exact three-coordinate unit selector; preservation of the public prefix;
the exact generated selector equations; and the steady selector-one fact used
by selective compiler refinement.

Does not own: proof that the active production call site chose this source
arm, private suffix decoding beyond the selector interval, matrix semantics,
commitment binding, or row removal.

Emits constraints: no.

| Stable stage path | Mathematical obligation | Authority class |
|---|---|---|
| `f_prime.carrier270.selector.steady` | generated selector rows set exactly the production steady arm while preserving the 270-coordinate public prefix | checked/refinement |
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.PhysicalSelectorAssignment

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.PhysicalPublicAssignment
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.SelectorRefinement
open Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.SelectorComposition.Semantics

/-- Exact generated steady-recursive arm. -/
def steadyArm : Fin 3 :=
  ⟨Provenance.sourceArm, by decide⟩

theorem steadyArm_eq_two : steadyArm = (2 : Fin 3) := by
  rfl

/-- Replay exactly the selected arm into physical columns `270..272`. -/
def replaySteadySelector (assignment : Fin 11437038 → F) :
    Fin 11437038 → F :=
  withUnitSelectors steadyArm assignment

theorem replaySteadySelector_preserves_public
    (assignment : Fin 11437038 → F)
    (column : Fin 11437038)
    (publicColumn : column.val < 270) :
    replaySteadySelector assignment column = assignment column := by
  unfold replaySteadySelector withUnitSelectors
  rw [dif_neg]
  omega

/-- Exact generated combined-NC gate column is the selected arm's physical
selector coordinate. -/
theorem generatedSteadySelectorColumn_exact :
    Metadata.steadySelectorColumn =
      (selectorColumn steadyArm).val := by
  rfl

theorem replaySteadySelector_one
    (assignment : Fin 11437038 → F) :
    replaySteadySelector assignment (selectorColumn steadyArm) = 1 := by
  rw [replaySteadySelector, withUnitSelectors_at_selector]
  simp [steadyArm, unitWeights]

/-- The generated numeric column used by the combined-NC selective rows is
one under the exact steady-arm replay. -/
theorem replaySteadySelector_generatedColumn_one
    (assignment : Fin 11437038 → F) :
    replaySteadySelector assignment
        ⟨Metadata.steadySelectorColumn, by decide⟩ = 1 := by
  have columnEq :
      (⟨Metadata.steadySelectorColumn, by decide⟩ :
        Fin 11437038) = selectorColumn steadyArm := by
    apply Fin.ext
    exact generatedSteadySelectorColumn_exact
  rw [columnEq]
  exact replaySteadySelector_one assignment

/-- The exact generated Boolean and total selector rows hold for the physical
steady-arm replay. -/
theorem replaySteadySelector_rowsSatisfied
    (prime : EuclidPrime goldilocksP)
    (assignment : Fin 11437038 → F)
    (constantOne : assignment SelectorRefinement.constantColumn = 1) :
    GeneratedRowsSatisfied (replaySteadySelector assignment) := by
  exact withUnitSelectors_satisfies prime steadyArm assignment constantOne

/-- Public replay followed by the generated steady selector keeps the exact
typed public carrier and satisfies all four selector rows. -/
theorem replayPublicThenSteadySelector
    (prime : EuclidPrime goldilocksP)
    (dimensions : Dimensions)
    (legacy : LegacyAssignment dimensions)
    (suffix : Fin 11437038 → F)
    (constantOne : PublicAssignment.SourceConstantOne dimensions legacy) :
    projectPhysical270 dimensions
        (replaySteadySelector
          (replayPhysicalPublicAssignment dimensions legacy suffix)) =
        Phi81Relation.projectPublicInput (assignment dimensions legacy) ∧
      GeneratedRowsSatisfied
        (replaySteadySelector
          (replayPhysicalPublicAssignment dimensions legacy suffix)) := by
  constructor
  · funext column
    rw [projectPhysical270]
    rw [replaySteadySelector_preserves_public]
    · exact congrFun
        (projectPhysical270_replay_eq_projectPublicInput dimensions legacy
          suffix constantOne) column
    · have columnBound := column.isLt
      simpa [physicalPublicColumn, Dimensions.shape_publicWidth] using
        columnBound
  · exact replaySteadySelector_rowsSatisfied prime _
      (replayPhysicalPublicAssignment_constantOne dimensions legacy suffix
        constantOne)

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.PhysicalSelectorAssignment
