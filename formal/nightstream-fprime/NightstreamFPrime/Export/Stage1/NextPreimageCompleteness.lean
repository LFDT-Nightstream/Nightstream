import NightstreamFPrime.Export.Stage1.PerApplicationSourceAssignment
import NightstreamFPrime.Export.Stage1.NextPreimageDirectPlan
import NightstreamFPrime.Layout.R1CS.Completeness

/-!
Owns the five next-preimage rows of the completed canonical assignment.
The zero-allocation wiring and its lowering preserve the same environment.
Only the prior and output preimage inputs are copied through the retained view.
-/

namespace NightstreamFPrime.Export.Stage1.NextPreimageCompleteness

open NightstreamFPrime.Circuit
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.Stage1
open NightstreamFPrime.Spec
open PerApplicationAssignmentTransportExecution

private theorem sourceRows_of_spec (env : Env)
    (specification : NextPreimage.SpecHolds NextPreimageInputs.spartanInterface
      NextPreimagePackage.privateStart env) :
    R1CS.RowsHold env NextPreimagePackage.sourceRows := by
  obtain ⟨completed, agrees, logical⟩ := NextPreimage.completeness
    NextPreimageInputs.spartanInterface env NextPreimagePackage.privateStart specification
  have same : completed = env := by
    funext index
    apply agrees index
    rw [NextPreimage.localLength_eq]
    omega
  subst completed
  have constraints : ConstraintsHold env NextPreimagePackage.constraints := logical
  obtain ⟨lowered, unchanged, rows⟩ := R1CS.lowerConstraints_complete env
    NextPreimagePackage.constraints NextPreimagePackage.privateStart
    (NextPreimage.flatConstraints_varsBelow _ _ env
      (NextPreimageInputs.spartanAssumptions _ env (Nat.le_refl _))) constraints
  have noFresh : R1CS.totalFreshCount NextPreimagePackage.constraints = 0 := by rfl
  have same : lowered = env := by
    funext index
    apply unchanged index
    rw [noFresh]
    omega
  subst lowered
  rw [NextPreimagePackage.sourceRows_eq]
  exact rows

private theorem inputs_before_pilot (env : Env) :
    NextPreimage.Assumptions NextPreimageInputs.sourceInterface
      PilotProduction.witnessOffset env := by
  refine ⟨?_, ?_, fun index => ?_, fun index => ?_⟩
  all_goals simp only [NextPreimageInputs.sourceInterface, Expr.VarsBelow]
  all_goals
    simp only [NextPreimageInputs.priorIterationSource,
      NextPreimageInputs.outputIterationSource, NextPreimageInputs.priorInitialStateSource,
      NextPreimageInputs.outputInitialStateSource,
      RunningTransitionInputs.iterationWordIndex, RunningTransitionInputs.initialStateWordStart,
      PilotProduction.outputPreimageStart, PilotProduction.priorPublicInputStart,
      PilotProduction.priorPreimageStart, PilotProduction.stateHashWords_eq,
      PriorStateHash.publicWidth_eq, PilotProduction.witnessOffset_eq]
  · norm_num
  · norm_num
  · have bounded := index.isLt
    change index.val < 4 at bounded
    omega
  · have bounded := index.isLt
    change index.val < 4 at bounded
    omega

private theorem copied_input
    (application : Lifecycle.Stage1.Application.Program) (target : Env)
    (suffix : Fin (PerApplicationPackage.addedPrivateColumnCount application) → F)
    (column : Nat) (below : column < PilotProduction.witnessOffset) :
    Spartan.pullback (RunningTransitionDirectPlan.transitionEnv application
      (PerApplicationSourceAssignment.ofCompleted application target suffix)) column =
      Spartan.pullback target column := by
  have sourceBound : column < Spartan.SourceColumnCount := by
    rw [PilotProduction.witnessOffset_eq] at below
    rw [Spartan.sourceColumnCount_eq]
    omega
  have beforeC : column < PiCCSInputs.phaseOffset := by
    rw [PilotProduction.witnessOffset_eq] at below
    rw [PiCCSInputs.phaseOffset_eq]
    omega
  exact (RunningTransitionDirectPlan.transitionEnv_of_outside application _ column sourceBound
    (Or.inl beforeC)).trans
    (PerApplicationSourceAssignment.source_ofCompleted application target suffix column sourceBound)

/-- Actual next-preimage wiring makes all five retained next-preimage rows
zero on the existing completed assignment. No separate framing equality or
source-copy premise is supplied by the caller. -/
theorem rowsZero_of_completed
    (application : Lifecycle.Stage1.Application.Program) (target : Env)
    (suffix : Fin (PerApplicationPackage.addedPrivateColumnCount application) → F)
    (nextRows : holdsFlat (Spartan.pullback target) (NextPreimage.opsAt
      NextPreimageInputs.sourceInterface RunningTransitionInputs.phaseOffset)) :
    let raw := canonicalRawValues application
      (PerApplicationSourceAssignment.ofCompleted application target suffix)
    (NextPreimageDirectPlan.plan
      (PerApplicationCanonicalEncodes.piCcsOrdinaryGeometry application)).RowsZero raw.assignment := by
  intro raw
  have sourceSpec := NextPreimage.soundness NextPreimageInputs.sourceInterface
    (Spartan.pullback target) RunningTransitionInputs.phaseOffset
    (holdsFlat_implies_holds _ _ nextRows)
  have bounds := inputs_before_pilot (Spartan.pullback target)
  have copiedSpec : NextPreimage.SpecHolds NextPreimageInputs.sourceInterface
      NextPreimagePackage.privateStart
      (Spartan.pullback (RunningTransitionDirectPlan.transitionEnv application raw.base)) := by
    apply NextPreimage.SpecHolds.of_cross_values_eq _ _
      RunningTransitionInputs.phaseOffset NextPreimagePackage.privateStart
      (Spartan.pullback target) _ _ _ _ _ sourceSpec
    · exact (copied_input application target suffix _ bounds.priorIteration).symm
    · exact (copied_input application target suffix _ bounds.outputIteration).symm
    · intro index
      exact (copied_input application target suffix _ (bounds.priorInitialState index)).symm
    · intro index
      exact (copied_input application target suffix _ (bounds.outputInitialState index)).symm
  apply (NextPreimageDirectPlan.rowsZero_iff_rowsHold _ raw.assignment
    raw.base raw.groupValue raw.products
    (PerApplicationCanonicalEncodes.samplerPrefixEncodes raw).prior.pilotOrdinary.prior
    (PerApplicationCanonicalAssignment.assignment_one raw)).mpr
  exact sourceRows_of_spec _
    ((NextPreimageInputs.spartanSpec_iff_sourceSpec _ _).mpr copiedSpec)

end NightstreamFPrime.Export.Stage1.NextPreimageCompleteness
