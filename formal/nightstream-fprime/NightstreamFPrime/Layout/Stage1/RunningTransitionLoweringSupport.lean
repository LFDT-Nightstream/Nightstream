import NightstreamFPrime.Layout.R1CS.Support
import NightstreamFPrime.Layout.Stage1.RunningTransitionInputsSupport
import NightstreamFPrime.Layout.Stage1.RunningTransitionPreservation
import NightstreamFPrime.Layout.Stage1.SpartanBounds

/-! Owns compact support preservation through lowering and Spartan remapping. -/

namespace NightstreamFPrime.Layout.Stage1.RunningTransitionSourceSupport

open NightstreamFPrime.Layout
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.Stage1
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open RunningTransitionInputs

theorem physicalRows_varsSatisfy
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    ∀ row ∈ RunningTransitionLayout.physicalRows logicalWidth publicFits,
      row.VarsSatisfy Source := by
  have lowered := R1CS.LoweringPlan.rows_varsSatisfy
    (RunningTransitionLayout.plan logicalWidth publicFits) Logical
    (logicalConstraints_varsSatisfy logicalWidth publicFits)
  have endEq : (RunningTransitionLayout.plan logicalWidth publicFits).next =
      physicalEnd := by
    change RunningTransitionLayout.physicalColumnCount logicalWidth publicFits =
      physicalEnd
    rw [RunningTransitionLayout.physicalColumnCount_eq relation]
    rfl
  intro row member
  have support := lowered row member
  rw [endEq] at support
  apply support.mono row
  intro column columnSupport
  rcases columnSupport with logical | fresh
  · rcases logical with external | inverse
    · exact Or.inl external
    · subst column
      exact Or.inr (by
        norm_num [phaseOffset, physicalEnd])
  · exact Or.inr (by
      change RunningTransitionLayout.logicalColumnCount ≤ column ∧
        column < physicalEnd at fresh
      norm_num [RunningTransitionLayout.logicalColumnCount, phaseOffset,
        physicalEnd, RunningTransition.exactPrivateCount] at fresh ⊢
      omega)

theorem remappedRows_varsSatisfy
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    ∀ row ∈ Spartan.remapRows
        (RunningTransitionLayout.physicalRows logicalWidth publicFits),
      row.VarsSatisfy Target := by
  apply Spartan.remapRows_varsSatisfy Source Target _
    (physicalRows_varsSatisfy relation)
  intro column support
  exact ⟨column, support, rfl⟩

end NightstreamFPrime.Layout.Stage1.RunningTransitionSourceSupport
