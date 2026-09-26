import NightstreamFPrime.Export.Stage1.Wide.PiRLCSourceInputs
import NightstreamFPrime.Export.Stage1.Wide.PiDECWitnessInputs
import NightstreamFPrime.Layout.PiRLC.Wide.Preservation

/-! The direct compact PiRLC outputs equal the four physical combination
families consumed by PiDEC. Their source values and challenges come from the
same checked PiCCS prefix. -/

namespace NightstreamFPrime.Export.Stage1.Wide.PiRLCSourceOutput

open NightstreamFPrime.Circuit NightstreamFPrime.Spec NightstreamFPrime.Layout
open NightstreamFPrime.Lifecycle NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.PiRLC.v1_1
open ProductionRelation
open Spec.Folding.PiCCS.PaperJoint
open Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

abbrev Family := PiRLCOutput.Family

local instance (family : Family) : NeZero family.cellCount := ⟨by cases family <;> decide⟩

variable {width : Nat}
  {fits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth width}

private abbrev parentInterface := Layout.Stage1.Wide.PiRLCInputs.interface
  (logicalWidth := width) (publicFits := fits)
private abbrev parentStart := Layout.Stage1.Wide.PiRLCInputs.phaseOffset

/-- Read the existing child interfaces, without changing their operations. -/
def familyInterface (family : Family) : CombinationFamily.Interface family.blockCount family.cellCount :=
  let shared := PiRLC.Wide.Formal.atOffset
    (parentInterface (width := width) (fits := fits)) parentStart
  match family with
  | .commitment => CommitmentCombination.familyInterface (PiRLC.Wide.Formal.commitmentInterface shared)
  | .publicInput => PublicInputCombination.familyInterface (PiRLC.Wide.Formal.publicInputInterface shared)
  | .evalK => RingKCombination.familyInterface
      (EvalKCombination.ringInterface (PiRLC.Wide.Formal.evalKInterface shared))
  | .evalA => RingKCombination.familyInterface
      (EvalACombination.ringInterface (PiRLC.Wide.Formal.evalAInterface shared))

def familyStart : Family → Nat
  | .commitment => Layout.Stage1.Wide.PiRLCStarts.commitmentLogicalStart
  | .publicInput => Layout.Stage1.Wide.PiRLCStarts.publicInputLogicalStart
  | .evalK => Layout.Stage1.Wide.PiRLCStarts.evalKLogicalStart
  | .evalA => Layout.Stage1.Wide.PiRLCStarts.evalALogicalStart

theorem canonical (relation : ProductionKey.LogicalRelation width fits) (env : Env)
    (specification : PiRLC.Wide.Formal.SpecHolds relation
      (parentInterface (width := width) (fits := fits)) parentStart env) (family : Family) :
    CombinationFamily.CanonicalHolds (familyInterface (width := width) (fits := fits) family)
      (familyStart family) env := by
  cases family
  · exact specification.commitment
  · exact specification.publicInput
  · exact specification.eval_K
  · exact specification.eval_A

theorem parent_value (env : Env) (family : Family) (block : Fin family.blockCount)
    (cell : Fin family.cellCount) (lane : Fin ringDegree) :
    CombinationFamily.evalOutput (familyInterface (width := width) (fits := fits) family)
      (familyStart family) env block cell lane =
      PiDECSource.value env (PiDECOutput.parentView family block cell lane) := by
  have blockBound := block.isLt
  have cellBound := cell.isLt
  cases family <;>
    simp only [CombinationFamily.evalOutput, CombinationFamily.output, CombinationStep.output,
      Expr.eval, PiDECSource.value, PiDECSource.column, PiDECOutput.parentView,
      PiRLCCombinationInvocations.indexOf_val, PiRLCCombinationInvocations.logicalIndex] <;>
    apply congrArg env
  · change 19587528 + (block.val * (54 * 1) + lane.val * 1 + cell.val) =
      19587528 + (block.val * 54 + lane.val)
    change cell.val < 1 at cellBound
    omega
  · change 19593036 + (block.val * (54 * 1) + lane.val * 1 + cell.val) =
      19593036 + (block.val * 54 + lane.val)
    change cell.val < 1 at cellBound
    omega
  · change 19595034 + (block.val * (54 * 2) + lane.val * 2 + cell.val) =
      19595034 + (lane.val * 2 + cell.val)
    change block.val < 1 at blockBound
    omega
  · change 19619334 + (block.val * (54 * 2) + lane.val * 2 + cell.val) =
      19619334 + (block.val * 108 + lane.val * 2 + cell.val)
    omega

theorem input_value (env : Env) (family : Family) (source : Fin 17)
    (block : Fin family.blockCount) (cell : Fin family.cellCount) (lane : Fin ringDegree) :
    CombinationFamily.inputValue (familyInterface (width := width) (fits := fits) family)
      (familyStart family) env source block cell lane =
      env (({family, source, block, lane, cell} : PiRLCProductSchedule.Descriptor).valueColumn lane) := by
  cases family with
  | commitment =>
    have cellZero : cell.val = 0 := by have bound : cell.val < 1 := cell.isLt; omega
    change ((Layout.Stage1.PiRLCInputs.sourceInput (logicalWidth := width) (publicFits := fits) source).commitment
      block lane).eval env = _
    rw [PiRLCCombinationInvocations.productionCommitmentValue_eq]
    simp only [PiRLCCombinationInvocations.sourceValue, Expr.eval,
      PiRLCProductSchedule.Descriptor.valueColumn, cellZero, Nat.mul_one]
  | publicInput =>
    have cellZero : cell.val = 0 := by have bound : cell.val < 1 := cell.isLt; omega
    change ((Layout.Stage1.PiRLCInputs.sourceInput (logicalWidth := width) (publicFits := fits) source).publicInput
      (PublicInputCombination.publicColumn block lane)).eval env = _
    rw [PiRLCCombinationInvocations.productionPublicInputValue_eq]
    simp only [PiRLCCombinationInvocations.sourceValue, Expr.eval,
      PiRLCProductSchedule.Descriptor.valueColumn, cellZero, Nat.mul_one]
  | evalK =>
    change (RingKCombination.expressionCell cell
      ((Layout.Stage1.PiRLCInputs.sourceInput (logicalWidth := width) (publicFits := fits) source).evaluation.eval_K
        (EvalKCombination.coefficient lane))).eval env = _
    rw [PiRLCCombinationInvocations.productionEvalKValue_eq source block lane cell]
    rfl
  | evalA =>
    change (RingKCombination.expressionCell cell
      ((Layout.Stage1.PiRLCInputs.sourceInput (logicalWidth := width) (publicFits := fits) source).evaluation.eval_A
        block (EvalKCombination.coefficient lane))).eval env = _
    rw [PiRLCCombinationInvocations.productionEvalAValue_eq source block lane cell]
    rfl

theorem challenge_value (env : Env) (family : Family) (source : Fin 17) (lane : Fin ringDegree) :
    CombinationFamily.challengeValue (familyInterface (width := width) (fits := fits) family)
      (familyStart family) env source lane =
      PiRLC.Wide.Semantics.evalChallenges
        (parentInterface (width := width) (fits := fits)) parentStart env source lane := by
  cases family <;> rfl

/-- The direct ordered sum equals the physical parent value supplied to PiDEC. -/
theorem ordered_value (program : RetainedLayout.Program) (env : Env)
    (application : Fin (PerApplicationPackage.addedPrivateColumnCount program) → F)
    (relation : ProductionKey.LogicalRelation width fits)
    (ajtai : AjtaiKey (logicalWidth := width) (publicFits := fits)) (template : Proof 9)
    (cAssumptions : PiCCS.v1_1.Formal.Assumptions relation
      (Layout.Stage1.PiCCSInputs.interface width fits) Layout.Stage1.PiCCSInputs.phaseOffset env)
    (cRows : Layout.PiCCS.v1_1.PhysicalHolds relation
      (Layout.Stage1.PiCCSInputs.interface width fits) Layout.Stage1.PiCCSInputs.phaseOffset env)
    (rAssumptions : PiRLC.Wide.Formal.Assumptions relation
      (parentInterface (width := width) (fits := fits)) parentStart env)
    (rRows : Layout.PiRLC.Wide.PhysicalHolds relation
      (parentInterface (width := width) (fits := fits)) parentStart env)
    (family : Family) (block : Fin family.blockCount) (cell : Fin family.cellCount) (lane : Fin ringDegree) :
    PiRLCOutput.ordered
      (PiRLCWitness.initial (Stage1Plan.piRlcInterface program)
        (AssignmentProjection.seed program (SourceAssignment.raw program env application).assignment))
      (PiRLCWitness.inputValues (Stage1Plan.piRlcInterface program)
        (AssignmentProjection.seed program (SourceAssignment.raw program env application).assignment))
      family block cell lane = PiDECSource.value env (PiDECOutput.parentView family block cell lane) := by
  let base := AssignmentProjection.seed program (SourceAssignment.raw program env application).assignment
  let initial := PiRLCWitness.initial (Stage1Plan.piRlcInterface program) base
  let values := PiRLCWitness.inputValues (Stage1Plan.piRlcInterface program) base
  let inputs := familyInterface (width := width) (fits := fits) family
  have phase := Layout.PiRLC.Wide.physical_implies_phaseHolds relation ajtai
    (parentInterface (width := width) (fits := fits)) parentStart env rAssumptions rRows
  have specification := Layout.PiRLC.Wide.physical_implies_specHolds relation
    (parentInterface (width := width) (fits := fits)) parentStart env rAssumptions rRows
  have initialEq : List.ofFn initial = PiRLC.Wide.Scalar.evalState env
      ((parentInterface (width := width) (fits := fits)).initialState parentStart) := by
    apply congrArg List.ofFn
    funext lane
    exact PiRLCSourceInputs.initial_value program env application relation ajtai template cAssumptions cRows lane
  have challengeEq (source : Fin 17) : PiRLCValues.challenge initial source.val =
      CombinationFamily.challengeValue inputs (familyStart family) env source := by
    funext lane
    rw [challenge_value]
    unfold PiRLCValues.challenge
    rw [initialEq]
    exact congrFun (congrFun phase.response source) lane
  have valueEq (source : Fin 17) :
      values ({family, source, block, cell} : PiRLCProductRingSchedule.Descriptor).invocation =
      CombinationFamily.inputValue inputs (familyStart family) env source block cell := by
    funext lane
    have copied := PiRLCSourceInputs.input_value program env application
      ({family, source, block, cell} : PiRLCProductRingSchedule.Descriptor).invocation lane
    rw [PiRLCProductRingSchedule.descriptor_laneInvocation,
      PiRLCProductRingSchedule.descriptor_invocation] at copied
    exact copied.trans (input_value (width := width) (fits := fits) env family source block cell lane).symm
  have sumEq : PiRLCOutput.ordered initial values family block cell =
      CombinationFamily.orderedCombination inputs (familyStart family) env block cell := by
    unfold PiRLCOutput.ordered CombinationFamily.orderedCombination
    apply congrArg CombinationFamily.rightCombination
    funext source
    exact congrArg₂ ringFMul (challengeEq source) (valueEq source)
  change PiRLCOutput.ordered initial values family block cell lane = _
  rw [sumEq, ← canonical relation env specification family block cell]
  exact parent_value (width := width) (fits := fits) env family block cell lane

/-- The compact witness's parent forms read the physical PiDEC parent values. -/
theorem retained_parent (program : RetainedLayout.Program) (env : Env)
    (application : Fin (PerApplicationPackage.addedPrivateColumnCount program) → F)
    (relation : ProductionKey.LogicalRelation width fits)
    (ajtai : AjtaiKey (logicalWidth := width) (publicFits := fits)) (template : Proof 9)
    (cAssumptions : PiCCS.v1_1.Formal.Assumptions relation
      (Layout.Stage1.PiCCSInputs.interface width fits) Layout.Stage1.PiCCSInputs.phaseOffset env)
    (cRows : Layout.PiCCS.v1_1.PhysicalHolds relation
      (Layout.Stage1.PiCCSInputs.interface width fits) Layout.Stage1.PiCCSInputs.phaseOffset env)
    (rAssumptions : PiRLC.Wide.Formal.Assumptions relation
      (parentInterface (width := width) (fits := fits)) parentStart env)
    (rRows : Layout.PiRLC.Wide.PhysicalHolds relation
      (parentInterface (width := width) (fits := fits)) parentStart env)
    (family : Family) (block : Fin family.blockCount) (cell : Fin family.cellCount) (lane : Fin ringDegree) :
    (RetainedLayout.renameForm program
      ((PiDECOutput.parentView family block cell lane).form (Stage1Plan.piDecGeometry program))
      (ReadSupport.piDec_location program _ _)).eval (SourceAssignment.assignment program env application) =
      PiDECSource.value env (PiDECOutput.parentView family block cell lane) := by
  rw [PiDECWitnessInputs.parent_value]
  exact ordered_value program env application relation ajtai template cAssumptions cRows
    rAssumptions rRows family block cell lane

end NightstreamFPrime.Export.Stage1.Wide.PiRLCSourceOutput
