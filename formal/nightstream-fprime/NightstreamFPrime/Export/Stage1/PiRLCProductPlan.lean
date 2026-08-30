import NightstreamFPrime.Export.Stage1.PerApplicationPackage
import NightstreamFPrime.Export.Stage1.Package
import NightstreamFPrime.Export.Stage1.PiRLCProductSchedule
import NightstreamFPrime.Layout.ProductionRelation.Phi81ProductFamilyPlan
import NightstreamFPrime.Layout.ProductionRelation.ProductRetainedBlock
import NightstreamFPrime.Layout.ProductionRelation.SourceCompiler

/-!
Owns the direct 14-matrix plan for the canonical PiRLC Phi81 product
invocations of one Lean-authored application package.

The plan is parameterized only by a proved source map into the final low-norm
assignment. It does not select the complete retained set or close the final
application fixed point.
-/

namespace NightstreamFPrime.Export.Stage1.PiRLCProductPlan

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Lifecycle.PiRLC.v1_1
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

def basePackage := PerApplicationPackage.basePackage

def baseSourceWidth (program : Lifecycle.Stage1.Application.Program) : Nat :=
  (PerApplicationPackage.package program).layout.totalColumnCount

def sourceWidth (program : Lifecycle.Stage1.Application.Program) : Nat :=
  ProductRetainedBlock.sourceWidth (baseSourceWidth program)
    PiRLCProductSchedule.invocationCount

private theorem basePackage_constantColumn :
    basePackage.layout.constantColumn = 27695710 := by
  exact NightstreamFPrime.Export.Stage1.Package.circuitPackage_layout_values.2.2.1

private theorem commitmentLogicalStart_eq :
    PiRLCStarts.commitmentLogicalStart = 19266319 := by
  rfl

private theorem publicInputLogicalStart_eq :
    PiRLCStarts.publicInputLogicalStart = 19282843 := by
  rfl

private theorem evalKLogicalStart_eq :
    PiRLCStarts.evalKLogicalStart = 19287433 := by
  rfl

private theorem evalALogicalStart_eq :
    PiRLCStarts.evalALogicalStart = 19289269 := by
  rfl

theorem basePackage_fits (program : Lifecycle.Stage1.Application.Program) :
    basePackage.layout.constantColumn ≤ baseSourceWidth program := by
  rw [baseSourceWidth, PerApplicationPackage.package_totalColumnCount]
  change PerApplicationPackage.basePackage.layout.constantColumn ≤
    PerApplicationPackage.basePackage.layout.totalColumnCount +
      PerApplicationPackage.addedPrivateColumnCount program
  have constant :
      PerApplicationPackage.basePackage.layout.constantColumn = 27695710 :=
    NightstreamFPrime.Export.Stage1.Package.circuitPackage_layout_values.2.2.1
  have total :
      PerApplicationPackage.basePackage.layout.totalColumnCount = 27695989 :=
    NightstreamFPrime.Export.Stage1.Package.circuitPackage_layout_values.2.2.2.2
  rw [constant, total]
  omega

/-- Every shifted column of the complete base package stays in the package
prefix. Derived product values start strictly after this domain. -/
theorem shiftColumn_lt_baseSourceWidth
    (program : Lifecycle.Stage1.Application.Program) (column : Nat)
    (bound : column < basePackage.layout.totalColumnCount) :
    PerApplicationPackage.shiftColumn program column < baseSourceWidth program := by
  rw [baseSourceWidth, PerApplicationPackage.package_totalColumnCount]
  change column < PerApplicationPackage.basePackage.layout.totalColumnCount at bound
  have total :
      PerApplicationPackage.basePackage.layout.totalColumnCount = 27695989 :=
    NightstreamFPrime.Export.Stage1.Package.circuitPackage_layout_values.2.2.2.2
  rw [total] at bound ⊢
  unfold PerApplicationPackage.shiftColumn
  split
  all_goals omega

/-- Canonical source-domain column for one complete base-package column. -/
def shiftedPackageColumn (program : Lifecycle.Stage1.Application.Program)
    (column : Nat) (bound : column < basePackage.layout.totalColumnCount) :
    Fin (baseSourceWidth program) :=
  ⟨PerApplicationPackage.shiftColumn program column,
    shiftColumn_lt_baseSourceWidth program column bound⟩

theorem sourceToSpartan_lt_basePackage (column : Nat)
    (bound : column < basePackage.layout.constantColumn) :
    Spartan.sourceToSpartan column < basePackage.layout.totalColumnCount := by
  have sourceBound : column < Spartan.SourceColumnCount := by
    rw [basePackage_constantColumn] at bound
    rw [Spartan.sourceColumnCount_eq]
    omega
  have mapped := Spartan.sourceToSpartan_lt column sourceBound
  have total : basePackage.layout.totalColumnCount = 27695989 := by
    exact NightstreamFPrime.Export.Stage1.Package.circuitPackage_layout_values.2.2.2.2
  simpa only [total, Spartan.spartanColumnCount_eq] using mapped

/-- One logical source column mapped through the canonical Spartan layout and
the verifier-selected per-application package shift. -/
def mappedPackageColumn (program : Lifecycle.Stage1.Application.Program)
    (column : Nat) (bound : column < basePackage.layout.constantColumn) :
    Fin (baseSourceWidth program) :=
  shiftedPackageColumn program (Spartan.sourceToSpartan column)
    (sourceToSpartan_lt_basePackage column bound)

private theorem challengeColumn_lt_basePackage
    (descriptor : PiRLCProductSchedule.Descriptor)
    (lane : Fin ringDegree) :
    descriptor.challengeColumn lane < basePackage.layout.constantColumn := by
  rw [basePackage_constantColumn]
  rcases descriptor with ⟨family, source, block, productLane, cell⟩
  simp only [PiRLCProductSchedule.Descriptor.challengeColumn]
  rw [PiRLCCombinationInvocations.challengeSourceStart,
    PiRLCStarts.challengeWordStart_eq, PiRLCStarts.phaseLogicalStart_eq]
  have sourceBound := source.isLt
  have laneBound := lane.isLt
  norm_num [PiRLCCombinationInvocations.sourceCount, ringDegree] at sourceBound laneBound ⊢
  omega

private theorem valueColumn_lt_basePackage
    (descriptor : PiRLCProductSchedule.Descriptor)
    (lane : Fin ringDegree) :
    descriptor.valueColumn lane < basePackage.layout.constantColumn := by
  rw [basePackage_constantColumn]
  rcases descriptor with ⟨family, source, block, productLane, cell⟩
  cases family <;>
    simp only [PiRLCProductSchedule.Descriptor.valueColumn,
      PiRLCProductSchedule.Family.blockCount,
      PiRLCProductSchedule.Family.cellCount] at *
  all_goals
    have sourceBound := source.isLt
    have blockBound := block.isLt
    have cellBound := cell.isLt
    have laneBound := lane.isLt
    norm_num [PiRLCCombinationInvocations.sourceCount, ringDegree,
      PiRLCCombinationInvocations.commitmentValueSourceStart,
      PiRLCCombinationInvocations.publicInputValueSourceStart,
      PiRLCCombinationInvocations.evalKValueSourceStart,
      PiRLCCombinationInvocations.evalAValueSourceStart,
      PiCCSInputs.freshCommitmentStart, PiCCSInputs.runningCommitmentStart,
      PiCCSInputs.runningPublicStart, PiCCSInputs.runningGroupStart,
      PiCCSInputs.runningGroupsStart, PiCCSInputs.priorRunningStart,
      PiCCSInputs.runningGroupWords, PiCCSInputs.outputEvaluationStart,
      PiCCSInputs.roundMessageStart, PiCCSInputs.freshCommitmentWords,
      PiCCSInputs.proofInputStart, PiCCSInputs.expectedContextStart,
      PiCCSInputs.expectedContextWords, PiCCSInputs.roundMessageWords,
      PilotProduction.priorPublicInputStart, PilotProduction.priorPreimageStart,
      PilotProduction.stateHashWords_eq] at sourceBound blockBound cellBound laneBound ⊢
  all_goals
    (try split) <;> omega

private theorem outputColumn_lt_basePackage
    (descriptor : PiRLCProductSchedule.Descriptor) :
    descriptor.outputColumn < basePackage.layout.constantColumn := by
  rw [basePackage_constantColumn]
  rcases descriptor with ⟨family, source, block, lane, cell⟩
  cases family <;>
    simp only [PiRLCProductSchedule.Descriptor.outputColumn,
      PiRLCProductSchedule.Descriptor.logicalIndex,
      PiRLCProductSchedule.Family.logicalStart,
      PiRLCProductSchedule.Family.blockCount,
      PiRLCProductSchedule.Family.cellCount, commitmentLogicalStart_eq,
      publicInputLogicalStart_eq, evalKLogicalStart_eq,
      evalALogicalStart_eq] at *
  all_goals
    have sourceBound := source.isLt
    have blockBound := block.isLt
    have laneBound := lane.isLt
    have cellBound := cell.isLt
    norm_num [PiRLCCombinationInvocations.sourceCount,
      PiRLCCombinationInvocations.stepSize,
      PiRLCCombinationInvocations.logicalIndex, ringDegree,
      PiRLCStarts.phaseLogicalStart, PiRLCInputs.phaseOffset] at sourceBound blockBound laneBound cellBound ⊢
    omega

private theorem priorColumn_lt_basePackage
    (descriptor : PiRLCProductSchedule.Descriptor)
    (notFirst : descriptor.source.val ≠ 0) :
    descriptor.priorColumn < basePackage.layout.constantColumn := by
  rw [basePackage_constantColumn]
  rcases descriptor with ⟨family, source, block, lane, cell⟩
  cases family <;>
    simp only [PiRLCProductSchedule.Descriptor.priorColumn,
      PiRLCProductSchedule.Descriptor.logicalIndex,
      PiRLCProductSchedule.Family.logicalStart,
      PiRLCProductSchedule.Family.blockCount,
      PiRLCProductSchedule.Family.cellCount, commitmentLogicalStart_eq,
      publicInputLogicalStart_eq, evalKLogicalStart_eq,
      evalALogicalStart_eq] at *
  all_goals
    have sourceBound := source.isLt
    have blockBound := block.isLt
    have laneBound := lane.isLt
    have cellBound := cell.isLt
    norm_num [PiRLCCombinationInvocations.sourceCount,
      PiRLCCombinationInvocations.stepSize,
      PiRLCCombinationInvocations.logicalIndex, ringDegree,
      PiRLCStarts.phaseLogicalStart, PiRLCInputs.phaseOffset] at sourceBound blockBound laneBound cellBound ⊢
    omega

def baseColumn (program : Lifecycle.Stage1.Application.Program)
    (column : Nat) (bound : column < basePackage.layout.constantColumn) :
    Fin (sourceWidth program) :=
  ProductRetainedBlock.baseColumn (baseSourceWidth program)
    PiRLCProductSchedule.invocationCount
      (mappedPackageColumn program column bound)

def challengeColumn (program : Lifecycle.Stage1.Application.Program)
    (descriptor : PiRLCProductSchedule.Descriptor) (lane : Fin ringDegree) :
    Fin (sourceWidth program) :=
  baseColumn program (descriptor.challengeColumn lane)
    (challengeColumn_lt_basePackage descriptor lane)

def valueColumn (program : Lifecycle.Stage1.Application.Program)
    (descriptor : PiRLCProductSchedule.Descriptor) (lane : Fin ringDegree) :
    Fin (sourceWidth program) :=
  baseColumn program (descriptor.valueColumn lane)
    (valueColumn_lt_basePackage descriptor lane)

def outputColumn (program : Lifecycle.Stage1.Application.Program)
    (descriptor : PiRLCProductSchedule.Descriptor) :
    Fin (sourceWidth program) :=
  baseColumn program descriptor.outputColumn
    (outputColumn_lt_basePackage descriptor)

def priorColumn (program : Lifecycle.Stage1.Application.Program)
    (descriptor : PiRLCProductSchedule.Descriptor)
    (notFirst : descriptor.source.val ≠ 0) : Fin (sourceWidth program) :=
  baseColumn program descriptor.priorColumn
    (priorColumn_lt_basePackage descriptor notFirst)

def groupColumn (program : Lifecycle.Stage1.Application.Program)
    (invocation : Fin PiRLCProductSchedule.invocationCount)
    (group : Fin 33) :
    Fin (sourceWidth program) :=
  ProductRetainedBlock.groupColumn (baseSourceWidth program)
    PiRLCProductSchedule.invocationCount invocation group

/-- Complete shifted package columns and derived group columns are disjoint. -/
theorem shiftedPackageColumn_lt_groupColumn
    (program : Lifecycle.Stage1.Application.Program)
    (column : Nat) (bound : column < basePackage.layout.totalColumnCount)
    (invocation : Fin PiRLCProductSchedule.invocationCount)
    (group : Fin 33) :
    (shiftedPackageColumn program column bound).val <
      (groupColumn program invocation group).val := by
  have packageBound := shiftColumn_lt_baseSourceWidth program column bound
  unfold shiftedPackageColumn groupColumn ProductRetainedBlock.groupColumn
  change PerApplicationPackage.shiftColumn program column <
    baseSourceWidth program + (Fin.encodeProd (invocation, group)).val
  omega

/-- The sole inputs needed to construct this matrix family over a final
logical assignment. -/
structure Inputs (program : Lifecycle.Stage1.Application.Program)
    (logicalWidth : Nat) where
  oneColumn : Fin logicalWidth
  challenge : Fin PiRLCProductSchedule.invocationCount →
    Fin ringDegree → SparseForm logicalWidth
  value : Fin PiRLCProductSchedule.invocationCount →
    Fin ringDegree → SparseForm logicalWidth
  prior : Fin PiRLCProductSchedule.invocationCount → SparseForm logicalWidth
  output : Fin PiRLCProductSchedule.invocationCount → SparseForm logicalWidth
  group : Fin PiRLCProductSchedule.invocationCount →
    Fin 33 → SparseForm logicalWidth

def sourceAssignment (program : Lifecycle.Stage1.Application.Program)
    (base : Fin (baseSourceWidth program) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 33 → F) :
    Fin (sourceWidth program) → F :=
  ProductRetainedBlock.sourceAssignment (baseSourceWidth program)
    PiRLCProductSchedule.invocationCount base groupValue

def baseEnv (program : Lifecycle.Stage1.Application.Program)
    (base : Fin (baseSourceWidth program) → F) : Env :=
  Spartan.pullback <|
    PerApplicationPackage.baseEnv program (SourceCompiler.sourceEnv base)

theorem baseEnv_eq_mappedPackageColumn
    (program : Lifecycle.Stage1.Application.Program)
    (base : Fin (baseSourceWidth program) → F)
    (column : Nat) (bound : column < basePackage.layout.constantColumn) :
    baseEnv program base column = base (mappedPackageColumn program column bound) := by
  have mappedBound := shiftColumn_lt_baseSourceWidth program
    (Spartan.sourceToSpartan column)
    (sourceToSpartan_lt_basePackage column bound)
  unfold baseEnv Spartan.pullback PerApplicationPackage.baseEnv
    SourceCompiler.sourceEnv mappedPackageColumn shiftedPackageColumn
  rw [dif_pos mappedBound]

theorem baseEnv_valueColumn
    (program : Lifecycle.Stage1.Application.Program)
    (base : Fin (baseSourceWidth program) → F)
    (descriptor : PiRLCProductSchedule.Descriptor) (lane : Fin ringDegree) :
    baseEnv program base (descriptor.valueColumn lane) =
      base (mappedPackageColumn program (descriptor.valueColumn lane)
        (valueColumn_lt_basePackage descriptor lane)) := by
  exact baseEnv_eq_mappedPackageColumn program base _
    (valueColumn_lt_basePackage descriptor lane)

theorem baseEnv_outputColumn
    (program : Lifecycle.Stage1.Application.Program)
    (base : Fin (baseSourceWidth program) → F)
    (descriptor : PiRLCProductSchedule.Descriptor) :
    baseEnv program base descriptor.outputColumn =
      base (mappedPackageColumn program descriptor.outputColumn
        (outputColumn_lt_basePackage descriptor)) := by
  exact baseEnv_eq_mappedPackageColumn program base _
    (outputColumn_lt_basePackage descriptor)

def challengeForm {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (inputs : Inputs program logicalWidth)
    (invocation : Fin PiRLCProductSchedule.invocationCount)
    (lane : Fin ringDegree) :
    SparseForm logicalWidth :=
  inputs.challenge invocation lane

def valueForm {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (inputs : Inputs program logicalWidth)
    (invocation : Fin PiRLCProductSchedule.invocationCount)
    (lane : Fin ringDegree) : SparseForm logicalWidth :=
  inputs.value invocation lane

def challengeState {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (inputs : Inputs program logicalWidth)
    (invocation : Fin PiRLCProductSchedule.invocationCount) :
    Phi81ProductPlan.State logicalWidth :=
  fun lane =>
    SparseForm.add
      (challengeForm inputs invocation lane)
      (SparseForm.singleton inputs.oneColumn (-2))

def valueState {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (inputs : Inputs program logicalWidth)
    (invocation : Fin PiRLCProductSchedule.invocationCount) :
    Phi81ProductPlan.State logicalWidth :=
  fun lane => valueForm inputs invocation lane

def priorForm {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (inputs : Inputs program logicalWidth)
    (invocation : Fin PiRLCProductSchedule.invocationCount) : SparseForm logicalWidth :=
  let descriptor := PiRLCProductSchedule.descriptor invocation
  if first : descriptor.source.val = 0 then
    .empty
  else
    inputs.prior invocation

def outputForm {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (inputs : Inputs program logicalWidth)
    (invocation : Fin PiRLCProductSchedule.invocationCount) : SparseForm logicalWidth :=
  inputs.output invocation

def groupForm {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (inputs : Inputs program logicalWidth)
    (invocation : Fin PiRLCProductSchedule.invocationCount)
    (group : Fin 33) : SparseForm logicalWidth :=
  inputs.group invocation group

/-- Exact invocation-major interface for all four PiRLC product families. -/
def interface {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (inputs : Inputs program logicalWidth) :
    Phi81ProductFamilyPlan.Interface logicalWidth
      PiRLCProductSchedule.invocationCount :=
  { oneColumn := inputs.oneColumn
    lane := fun invocation => (PiRLCProductSchedule.descriptor invocation).lane
    left := challengeState inputs
    right := valueState inputs
    groupOutput := groupForm inputs
    prior := priorForm inputs
    output := outputForm inputs }

theorem rowCount_le : PiRLCProductSchedule.invocationCount * 34 ≤
    2 ^ NightstreamFPrime.Lifecycle.cubeVariables := by
  rw [PiRLCProductSchedule.invocationCount_eq]
  norm_num [NightstreamFPrime.Lifecycle.cubeVariables]

def plan {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (inputs : Inputs program logicalWidth) :
    ProductionRelation.Plan logicalWidth :=
  Phi81ProductFamilyPlan.plan (interface inputs) rowCount_le

@[simp] theorem plan_rowCount {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (inputs : Inputs program logicalWidth) :
    (plan inputs).rowCount = 1654236 := by
  rw [plan, Phi81ProductFamilyPlan.plan_rowCount,
    PiRLCProductSchedule.invocationCount_eq]

def challengeRing (program : Lifecycle.Stage1.Application.Program)
    (base : Fin (baseSourceWidth program) → F)
    (descriptor : PiRLCProductSchedule.Descriptor) : RingF :=
  fun lane => baseEnv program base (descriptor.challengeColumn lane) - 2

def valueRing (program : Lifecycle.Stage1.Application.Program)
    (base : Fin (baseSourceWidth program) → F)
    (descriptor : PiRLCProductSchedule.Descriptor) : RingF :=
  fun lane => baseEnv program base (descriptor.valueColumn lane)

def priorValue (program : Lifecycle.Stage1.Application.Program)
    (base : Fin (baseSourceWidth program) → F)
    (descriptor : PiRLCProductSchedule.Descriptor) : F :=
  if descriptor.source.val = 0 then 0
  else baseEnv program base descriptor.priorColumn

def outputValue (program : Lifecycle.Stage1.Application.Program)
    (base : Fin (baseSourceWidth program) → F)
    (descriptor : PiRLCProductSchedule.Descriptor) : F :=
  baseEnv program base descriptor.outputColumn

/-- Exact local source-form preservation needed by this plan. Unused package
columns need no form and no preservation premise. -/
structure Preserves {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (inputs : Inputs program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (base : Fin (baseSourceWidth program) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 33 → F) :
    Prop where
  challenge : ∀ invocation lane,
    (challengeForm inputs invocation lane).eval assignment =
      baseEnv program base
        ((PiRLCProductSchedule.descriptor invocation).challengeColumn lane)
  value : ∀ invocation lane,
    (valueForm inputs invocation lane).eval assignment =
      baseEnv program base
        ((PiRLCProductSchedule.descriptor invocation).valueColumn lane)
  prior : ∀ invocation,
    (priorForm inputs invocation).eval assignment =
      priorValue program base (PiRLCProductSchedule.descriptor invocation)
  output : ∀ invocation,
    (outputForm inputs invocation).eval assignment =
      outputValue program base (PiRLCProductSchedule.descriptor invocation)
  group : ∀ invocation group,
    (groupForm inputs invocation group).eval assignment =
      groupValue invocation group

private theorem challengeState_eval
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (inputs : Inputs program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (base : Fin (baseSourceWidth program) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 33 → F)
    (one : assignment inputs.oneColumn = 1)
    (preserves : Preserves inputs assignment base groupValue)
    (invocation : Fin PiRLCProductSchedule.invocationCount) :
    Phi81ProductPlan.evalState assignment (challengeState inputs invocation) =
      challengeRing program base (PiRLCProductSchedule.descriptor invocation) := by
  funext lane
  simp [Phi81ProductPlan.evalState, challengeState, challengeRing,
    preserves.challenge invocation lane, one,
    sub_eq_add_neg]

private theorem valueState_eval
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (inputs : Inputs program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (base : Fin (baseSourceWidth program) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 33 → F)
    (preserves : Preserves inputs assignment base groupValue)
    (invocation : Fin PiRLCProductSchedule.invocationCount) :
    Phi81ProductPlan.evalState assignment (valueState inputs invocation) =
      valueRing program base (PiRLCProductSchedule.descriptor invocation) := by
  funext lane
  exact preserves.value invocation lane

private theorem priorForm_eval
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (inputs : Inputs program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (base : Fin (baseSourceWidth program) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 33 → F)
    (preserves : Preserves inputs assignment base groupValue)
    (invocation : Fin PiRLCProductSchedule.invocationCount) :
    (priorForm inputs invocation).eval assignment =
      priorValue program base (PiRLCProductSchedule.descriptor invocation) := by
  exact preserves.prior invocation

private theorem outputForm_eval
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (inputs : Inputs program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (base : Fin (baseSourceWidth program) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 33 → F)
    (preserves : Preserves inputs assignment base groupValue)
    (invocation : Fin PiRLCProductSchedule.invocationCount) :
    (outputForm inputs invocation).eval assignment =
      outputValue program base (PiRLCProductSchedule.descriptor invocation) := by
  exact preserves.output invocation

theorem rowsZero_implies_equation
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (inputs : Inputs program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (base : Fin (baseSourceWidth program) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 33 → F)
    (one : assignment inputs.oneColumn = 1)
    (preserves : Preserves inputs assignment base groupValue)
    (rowsZero : (plan inputs).RowsZero assignment)
    (invocation : Fin PiRLCProductSchedule.invocationCount) :
    outputValue program base (PiRLCProductSchedule.descriptor invocation) =
      priorValue program base (PiRLCProductSchedule.descriptor invocation) +
        ringFMul
          (challengeRing program base
            (PiRLCProductSchedule.descriptor invocation))
          (valueRing program base
            (PiRLCProductSchedule.descriptor invocation))
          (PiRLCProductSchedule.descriptor invocation).lane := by
  have equation := Phi81ProductFamilyPlan.planRowsZero_implies_ringProduct
    (interface inputs) rowCount_le assignment one rowsZero invocation
  simp only [interface] at equation
  rw [outputForm_eval inputs assignment base groupValue preserves,
    priorForm_eval inputs assignment base groupValue preserves,
    challengeState_eval inputs assignment base groupValue one preserves,
    valueState_eval inputs assignment base groupValue preserves] at equation
  exact equation

private theorem outputExpr_eval
    (program : Lifecycle.Stage1.Application.Program)
    (base : Fin (baseSourceWidth program) → F)
    (descriptor : PiRLCProductSchedule.Descriptor) :
    descriptor.outputExpr.eval (baseEnv program base) =
      outputValue program base descriptor := by
  rfl

private theorem priorExpr_eval
    (program : Lifecycle.Stage1.Application.Program)
    (base : Fin (baseSourceWidth program) → F)
    (descriptor : PiRLCProductSchedule.Descriptor) :
    descriptor.priorExpr.eval (baseEnv program base) =
      priorValue program base descriptor := by
  unfold PiRLCProductSchedule.Descriptor.priorExpr priorValue
  split <;> rfl

private theorem challengeExpr_evalRing
    (program : Lifecycle.Stage1.Application.Program)
    (base : Fin (baseSourceWidth program) → F)
    (descriptor : PiRLCProductSchedule.Descriptor) :
    CombinationStep.evalRing (baseEnv program base) descriptor.challengeExpr =
      challengeRing program base descriptor := by
  funext lane
  simp [CombinationStep.evalRing,
    PiRLCProductSchedule.Descriptor.challengeExpr, challengeRing]

private theorem valueExpr_evalRing
    (program : Lifecycle.Stage1.Application.Program)
    (base : Fin (baseSourceWidth program) → F)
    (descriptor : PiRLCProductSchedule.Descriptor) :
    CombinationStep.evalRing (baseEnv program base) descriptor.valueExpr =
      valueRing program base descriptor := by
  funext lane
  rfl

/-- The direct semantic recurrence is exactly the current canonical source
constraint expression. -/
theorem sourceConstraint_eval
    (program : Lifecycle.Stage1.Application.Program)
    (base : Fin (baseSourceWidth program) → F)
    (descriptor : PiRLCProductSchedule.Descriptor) :
    descriptor.sourceConstraint.eval (baseEnv program base) =
      outputValue program base descriptor -
        (priorValue program base descriptor +
          ringFMul (challengeRing program base descriptor)
            (valueRing program base descriptor) descriptor.lane) := by
  rw [PiRLCProductSchedule.Descriptor.sourceConstraint_eq_direct]
  rw [Expr.eval_sub, Expr.eval_hadd, CombinationStep.mulExpr_eval]
  rw [outputExpr_eval, priorExpr_eval, challengeExpr_evalRing,
    valueExpr_evalRing]

/-- Every zero row of the direct product family forces the exact canonical
PiRLC source constraint selected by the same invocation descriptor. -/
theorem rowsZero_implies_sourceConstraint
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (inputs : Inputs program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (base : Fin (baseSourceWidth program) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 33 → F)
    (one : assignment inputs.oneColumn = 1)
    (preserves : Preserves inputs assignment base groupValue)
    (rowsZero : (plan inputs).RowsZero assignment)
    (invocation : Fin PiRLCProductSchedule.invocationCount) :
    (PiRLCProductSchedule.descriptor invocation).sourceConstraint.eval
        (baseEnv program base) = 0 := by
  rw [sourceConstraint_eval]
  rw [rowsZero_implies_equation inputs assignment base groupValue one preserves
    rowsZero invocation]
  exact Lean.Grind.AddCommGroup.sub_self _

def groupIndex
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (inputs : Inputs program logicalWidth)
    (invocation : Fin PiRLCProductSchedule.invocationCount)
    (group : Fin 33) :
    Fin (ProductSumPlan.groups
      (Phi81ProductFamilyPlan.laneInterface
        (interface inputs) invocation).terms).length :=
  ⟨group.val, by
    simpa [Phi81ProductFamilyPlan.laneInterface] using group.isLt⟩

/-- Canonical retained values for the 33 five-product groups of each
invocation. -/
def honestGroupValue
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (inputs : Inputs program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (invocation : Fin PiRLCProductSchedule.invocationCount)
    (group : Fin 33) : F :=
  ProductSumPlan.groupTotal assignment <|
    ProductSumPlan.groupAt
      (Phi81ProductFamilyPlan.laneInterface (interface inputs) invocation)
      (groupIndex inputs invocation group)

/-- Canonical source constraints and the exact honest retained group values
are sufficient for every direct product-family row to vanish. -/
theorem sourceConstraints_imply_rowsZero
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (inputs : Inputs program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (base : Fin (baseSourceWidth program) → F)
    (one : assignment inputs.oneColumn = 1)
    (preserves : Preserves inputs assignment base
      (honestGroupValue inputs assignment))
    (constraints : ∀ invocation,
      (PiRLCProductSchedule.descriptor invocation).sourceConstraint.eval
        (baseEnv program base) = 0) :
    (plan inputs).RowsZero assignment := by
  apply (Phi81ProductFamilyPlan.planRowsZero_iff
    (interface inputs) rowCount_le assignment one).mpr
  intro invocation
  let laneInterface := Phi81ProductFamilyPlan.laneInterface
    (interface inputs) invocation
  refine
    { groups := ?_
      final := ?_ }
  · intro group
    let group33 : Fin 33 := ⟨group.val, by
      simpa [laneInterface, Phi81ProductFamilyPlan.laneInterface] using
        group.isLt⟩
    have groupEqual : groupIndex inputs invocation group33 = group := by
      apply Fin.ext
      rfl
    change
      (groupForm inputs invocation group33).eval assignment =
        ProductSumPlan.groupTotal assignment
          (ProductSumPlan.groupAt laneInterface group)
    rw [preserves.group invocation group33]
    unfold honestGroupValue
    rw [groupEqual]
  · have constraint := constraints invocation
    rw [sourceConstraint_eval] at constraint
    have recurrence := Lean.Grind.AddCommGroup.sub_eq_zero_iff.mp constraint
    change
      (outputForm inputs invocation).eval assignment =
        (priorForm inputs invocation).eval assignment +
          ProductSumPlan.total assignment laneInterface.terms
    rw [outputForm_eval inputs assignment base
      (honestGroupValue inputs assignment) preserves]
    rw [priorForm_eval inputs assignment base
      (honestGroupValue inputs assignment) preserves]
    dsimp only [laneInterface, Phi81ProductFamilyPlan.laneInterface,
      interface]
    rw [Phi81ProductPlan.terms_total]
    rw [challengeState_eval inputs assignment base
      (honestGroupValue inputs assignment) one preserves]
    rw [valueState_eval inputs assignment base
      (honestGroupValue inputs assignment) preserves]
    exact recurrence

end NightstreamFPrime.Export.Stage1.PiRLCProductPlan
