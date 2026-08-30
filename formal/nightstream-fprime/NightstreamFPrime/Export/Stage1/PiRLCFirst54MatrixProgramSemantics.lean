import NightstreamFPrime.Export.Stage1.PiRLCFirst54MatrixProgram

/-!
Proves row-by-row equality between the compact affine-grid First54 program
and the canonical Lean `PiRLCFirst54DirectPlan` families.
-/

namespace NightstreamFPrime.Export.Stage1.PiRLCFirst54MatrixProgram

open NightstreamFPrime.Export.MatrixProgram
open NightstreamFPrime.Export.MatrixProgram.AffineGrid
open NightstreamFPrime.Gadgets.Sampling
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

private theorem negOne_ne_one : (-1 : F) ≠ 1 := by
  decide

private theorem empty_add {logicalWidth : Nat}
    (form : SparseForm logicalWidth) :
    SparseForm.add SparseForm.empty form = form := by
  cases form
  rfl

private theorem add_scaled_empty {logicalWidth : Nat}
    (form : SparseForm logicalWidth) :
    SparseForm.add form (SparseForm.scale (-1) SparseForm.empty) = form := by
  cases form
  unfold SparseForm.add SparseForm.scale SparseForm.empty
  simp

def candidate (source : Fin sourceCount) (round : Fin roundCount) :
    PiRLCFirst54DirectSchedule.Candidate := ⟨source, round⟩

def previousRound (round : Fin roundCount) (notFirst : round.val ≠ 0) :
    Fin roundCount :=
  ⟨round.val - 1, by
    have roundBound := round.isLt
    omega⟩

def laterRoundOffset (round : Fin roundCount) (notFirst : round.val ≠ 0) :
    Fin 63 :=
  ⟨round.val - 1, by
    have roundBound := round.isLt
    change round.val < 64 at roundBound
    omega⟩

@[simp] theorem previousCandidate_eq (source : Fin sourceCount)
    (round : Fin roundCount) (notFirst : round.val ≠ 0) :
    PiRLCFirst54DirectPlan.previousCandidate (candidate source round) notFirst =
      candidate source (previousRound round notFirst) := by
  rfl

def positionDescriptor (source : Fin sourceCount) (round : Fin roundCount)
    (slot : Fin positionSlotCount) : PiRLCFirst54DirectSchedule.Position :=
  ⟨candidate source round, slot⟩

def valueDescriptor (source : Fin sourceCount) (round : Fin roundCount)
    (slot : Fin valueSlotCount) : PiRLCFirst54DirectSchedule.Value :=
  ⟨candidate source round, slot⟩

@[simp] theorem candidateIndex_val (source : Fin sourceCount)
    (round : Fin roundCount) :
    (PiRLCFirst54DirectSchedule.candidateIndex
      (candidate source round)).val = source.val * 64 + round.val := by
  simp [candidate, PiRLCFirst54DirectSchedule.candidateIndex,
    Fin.encodeProd, Nat.mul_comm, roundCount]

@[simp] theorem positionIndex_val (source : Fin sourceCount)
    (round : Fin roundCount) (slot : Fin positionSlotCount) :
    (PiRLCFirst54DirectSchedule.positionIndex
      (positionDescriptor source round slot)).val =
        source.val * 3520 + round.val * 55 + slot.val := by
  simp [positionDescriptor, candidate,
    PiRLCFirst54DirectSchedule.positionIndex,
    PiRLCFirst54DirectSchedule.candidateIndex,
    Fin.encodeProd, Nat.mul_comm, roundCount, positionSlotCount,
    First54Step.slotCount]
  ring

@[simp] theorem valueIndex_val (source : Fin sourceCount)
    (round : Fin roundCount) (slot : Fin valueSlotCount) :
    (PiRLCFirst54DirectSchedule.valueIndex
      (valueDescriptor source round slot)).val =
        source.val * 3456 + round.val * 54 + slot.val := by
  simp [valueDescriptor, candidate,
    PiRLCFirst54DirectSchedule.valueIndex,
    PiRLCFirst54DirectSchedule.candidateIndex,
    Fin.encodeProd, Nat.mul_comm, roundCount, valueSlotCount,
    First54ValueStep.outputCount]
  ring

theorem rejectWire_form?
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiRLCRetainedGeometry.Geometry program logicalWidth)
    (source : Fin sourceCount) (round : Fin roundCount) :
    (rejectWire program).form? logicalWidth
        (source.val * 64 + round.val) =
      some ((PiRLCRetainedInputs.first54Inputs geometry).reject
        (candidate source round)) := by
  have direct := RetainedBlock.form?_ofSemantic
    (PiRLCFirst54RetainedBlocks.rejectBlock program)
    (PiRLCRetainedGeometry.rejectStart program)
    (PiRLCRetainedGeometry.rejectFits geometry)
    (PiRLCFirst54DirectSchedule.candidateIndex (candidate source round))
  simpa [rejectWire, PiRLCRetainedInputs.first54Inputs] using direct

theorem symbolWire_form?
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiRLCRetainedGeometry.Geometry program logicalWidth)
    (source : Fin sourceCount) (round : Fin roundCount) :
    (symbolWire program).form? logicalWidth
        (source.val * 64 + round.val) =
      some ((PiRLCRetainedInputs.first54Inputs geometry).symbol
        (candidate source round)) := by
  have direct := RetainedBlock.form?_ofSemantic
    (PiRLCFirst54RetainedBlocks.symbolBlock program)
    (PiRLCRetainedGeometry.symbolStart program)
    (PiRLCRetainedGeometry.symbolFits geometry)
    (PiRLCFirst54DirectSchedule.candidateIndex (candidate source round))
  simpa [symbolWire, PiRLCRetainedInputs.first54Inputs] using direct

theorem positionWire_form?
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiRLCRetainedGeometry.Geometry program logicalWidth)
    (source : Fin sourceCount) (round : Fin roundCount)
    (slot : Fin positionSlotCount) :
    (positionWire program).form? logicalWidth
        (source.val * 3520 + round.val * 55 + slot.val) =
      some ((PiRLCRetainedInputs.first54Inputs geometry).position
        (positionDescriptor source round slot)) := by
  have direct := RetainedBlock.form?_ofSemantic
    (PiRLCFirst54RetainedBlocks.positionBlock program)
    (PiRLCRetainedGeometry.positionStart program)
    (PiRLCRetainedGeometry.positionFits geometry)
    (PiRLCFirst54DirectSchedule.positionIndex
      (positionDescriptor source round slot))
  simpa [positionWire, PiRLCRetainedInputs.first54Inputs] using direct

theorem valueWire_form?
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiRLCRetainedGeometry.Geometry program logicalWidth)
    (source : Fin sourceCount) (round : Fin roundCount)
    (slot : Fin valueSlotCount) :
    (valueWire program).form? logicalWidth
        (source.val * 3456 + round.val * 54 + slot.val) =
      some ((PiRLCRetainedInputs.first54Inputs geometry).value
        (valueDescriptor source round slot)) := by
  have direct := RetainedBlock.form?_ofSemantic
    (PiRLCFirst54RetainedBlocks.valueBlock program)
    (PiRLCRetainedGeometry.valueStart program)
    (PiRLCRetainedGeometry.valueFits geometry)
    (PiRLCFirst54DirectSchedule.valueIndex
      (valueDescriptor source round slot))
  simpa [valueWire, PiRLCRetainedInputs.first54Inputs] using direct

theorem productWire_form?
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiRLCRetainedGeometry.Geometry program logicalWidth)
    (source : Fin sourceCount) (round : Fin roundCount) :
    (productWire program).form? logicalWidth
        (source.val * 64 + round.val) =
      some ((PiRLCRetainedInputs.first54Inputs geometry).product
        (PiRLCFirst54DirectSchedule.candidateIndex
          (candidate source round))) := by
  have direct := RetainedBlock.form?_ofSemantic
    (PiRLCFirst54RetainedBlocks.productBlock program)
    (PiRLCRetainedGeometry.first54ProductStart program)
    (PiRLCRetainedGeometry.first54ProductFits geometry)
    (PiRLCFirst54DirectSchedule.candidateIndex (candidate source round))
  simpa [productWire, PiRLCRetainedInputs.first54Inputs] using direct

theorem constantRule_form?
    {logicalWidth : Nat} (selected : Region)
    (major : Fin selected.majorCount) (middle : Fin selected.middleCount)
    (minor : Fin selected.minorCount) (oneColumn : Fin logicalWidth)
    (coefficient : F) :
    (constantRule selected coefficient).form? logicalWidth oneColumn.val {
        major := selected.majorStart + major.val
        middle := selected.middleStart + middle.val
        minor := selected.minorStart + minor.val } =
      some (some (SparseForm.singleton oneColumn coefficient)) := by
  simpa [constantRule] using
    Rule.constant_form? selected major middle minor oneColumn coefficient

theorem retainedRule_form?_of_loaded
    {logicalWidth : Nat} (selected : Region)
    (coordinate offsets : Coordinate) (block : RetainedBlock) (oneColumn : Nat)
    (slotBase majorStride middleStride minorStride : Nat)
    (coefficient : F) (form : SparseForm logicalWidth)
    (inside : selected.offsets? coordinate = some offsets)
    (loaded : block.form? logicalWidth
      (slotBase + offsets.major * majorStride + offsets.middle * middleStride +
        offsets.minor * minorStride) = some form) :
    (retainedRule selected block slotBase majorStride middleStride minorStride
      coefficient).form? logicalWidth oneColumn coordinate =
      some (some (applyCoefficient coefficient form)) := by
  have termLoaded :
      (Term.retained block slotBase majorStride middleStride minorStride
        coefficient.val).form? logicalWidth oneColumn
          offsets =
        some (applyCoefficient coefficient form) := by
    change (do
      let retained ← block.form? logicalWidth
        (slotBase + offsets.major * majorStride +
          offsets.middle * middleStride + offsets.minor * minorStride)
      if coefficientBound : coefficient.val < goldilocksModulus then
        some (applyCoefficient ⟨coefficient.val, coefficientBound⟩ retained)
      else none) = _
    rw [loaded]
    change (if coefficientBound : coefficient.val < goldilocksModulus then
        some (applyCoefficient ⟨coefficient.val, coefficientBound⟩ form)
      else none) = _
    rw [dif_pos coefficient.isLt]
  unfold retainedRule Rule.form?
  rw [inside]
  simp only
  rw [termLoaded]
  rfl

theorem rule_form?_outside {logicalWidth : Nat} (rule : Rule)
    (oneColumn : Nat) (coordinate : Coordinate)
    (outside : ¬(
      rule.region.majorStart ≤ coordinate.major ∧
      coordinate.major < rule.region.majorStart + rule.region.majorCount ∧
      rule.region.middleStart ≤ coordinate.middle ∧
      coordinate.middle < rule.region.middleStart + rule.region.middleCount ∧
      rule.region.minorStart ≤ coordinate.minor ∧
      coordinate.minor < rule.region.minorStart + rule.region.minorCount)) :
    rule.form? logicalWidth oneColumn coordinate = some none := by
  exact Rule.form?_eq_some_none rule logicalWidth oneColumn coordinate
    (Region.offsets?_eq_none_of_outside rule.region coordinate outside)

theorem acceptedLeftProgram_form?
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiRLCRetainedGeometry.Geometry program logicalWidth)
    (source : Fin sourceCount) (round : Fin roundCount) :
    (acceptedLeftProgram program).form? logicalWidth
        (PiRLCRetainedGeometry.oneColumn geometry).val
        { major := source.val, middle := round.val, minor := 0 } =
      some (PiRLCFirst54DirectPlan.acceptedForm
        (PiRLCRetainedInputs.first54Inputs geometry)
        (candidate source round)) := by
  let inputs := PiRLCRetainedInputs.first54Inputs geometry
  let oneForm := PiRLCFirst54DirectPlan.oneForm inputs
  let rejectForm := PiRLCFirst54DirectPlan.rejectForm inputs
    (candidate source round)
  have oneLoaded :
      (constantRule allAccepted 1).form? logicalWidth
        (PiRLCRetainedGeometry.oneColumn geometry).val
        { major := source.val, middle := round.val, minor := 0 } =
      some (some oneForm) := by
    simpa [allAccepted, region, sourceCount, roundCount, oneForm,
      PiRLCFirst54DirectPlan.oneForm, inputs] using
        constantRule_form? allAccepted source round (0 : Fin 1)
          (PiRLCRetainedGeometry.oneColumn geometry) (1 : F)
  have rejectLoaded :
      (retainedRule allAccepted (rejectWire program) 0 64 1 0 (-1)).form?
        logicalWidth (PiRLCRetainedGeometry.oneColumn geometry).val
        { major := source.val, middle := round.val, minor := 0 } =
      some (some (SparseForm.scale (-1) rejectForm)) := by
    have raw := rejectWire_form? geometry source round
    have inside : allAccepted.offsets?
        { major := source.val, middle := round.val, minor := 0 } =
      some { major := source.val, middle := round.val, minor := 0 } := by
      simpa [allAccepted, region] using
        Region.offsets?_of_offsets allAccepted source round (0 : Fin 1)
    have selected := retainedRule_form?_of_loaded allAccepted
      { major := source.val, middle := round.val, minor := 0 }
      { major := source.val, middle := round.val, minor := 0 }
      (rejectWire program) (PiRLCRetainedGeometry.oneColumn geometry).val
      0 64 1 0 (-1) rejectForm inside
      (by simpa [rejectForm, PiRLCFirst54DirectPlan.rejectForm, inputs]
        using raw)
    simpa [applyCoefficient, negOne_ne_one] using selected
  have folded := Program.form?_of_results
    (acceptedLeftProgram program)
    (PiRLCRetainedGeometry.oneColumn geometry).val
    { major := source.val, middle := round.val, minor := 0 }
    [some oneForm, some (SparseForm.scale (-1) rejectForm)] (by
      exact .cons oneLoaded (.cons rejectLoaded .nil))
  simpa [acceptedLeftProgram, acceptedProgram, combine, addSelected,
    PiRLCFirst54DirectPlan.acceptedForm, PiRLCFirst54DirectPlan.subtract,
    oneForm, rejectForm, inputs, SparseForm.add, SparseForm.empty] using folded

theorem acceptedRightProgram_form?
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiRLCRetainedGeometry.Geometry program logicalWidth)
    (source : Fin sourceCount) (round : Fin roundCount) :
    (acceptedRightProgram program).form? logicalWidth
        (PiRLCRetainedGeometry.oneColumn geometry).val
        { major := source.val, middle := round.val, minor := 0 } =
      some (PiRLCFirst54DirectPlan.symbolForm
        (PiRLCRetainedInputs.first54Inputs geometry)
        (candidate source round)) := by
  let inputs := PiRLCRetainedInputs.first54Inputs geometry
  let symbolForm := PiRLCFirst54DirectPlan.symbolForm inputs
    (candidate source round)
  have raw := symbolWire_form? geometry source round
  have inside : allAccepted.offsets?
      { major := source.val, middle := round.val, minor := 0 } =
    some { major := source.val, middle := round.val, minor := 0 } := by
    simpa [allAccepted, region] using
      Region.offsets?_of_offsets allAccepted source round (0 : Fin 1)
  have loaded :
      (retainedRule allAccepted (symbolWire program) 0 64 1 0 1).form?
        logicalWidth (PiRLCRetainedGeometry.oneColumn geometry).val
        { major := source.val, middle := round.val, minor := 0 } =
      some (some symbolForm) := by
    have selected := retainedRule_form?_of_loaded allAccepted
      { major := source.val, middle := round.val, minor := 0 }
      { major := source.val, middle := round.val, minor := 0 }
      (symbolWire program) (PiRLCRetainedGeometry.oneColumn geometry).val
      0 64 1 0 1 symbolForm inside
      (by simpa [symbolForm, PiRLCFirst54DirectPlan.symbolForm, inputs]
        using raw)
    simpa [applyCoefficient] using selected
  change (Program.mk [retainedRule allAccepted (symbolWire program)
      0 64 1 0 1]).form? logicalWidth
      (PiRLCRetainedGeometry.oneColumn geometry).val
      { major := source.val, middle := round.val, minor := 0 } =
    some symbolForm
  unfold Program.form?
  change (do
    let selected ← (retainedRule allAccepted (symbolWire program)
      0 64 1 0 1).form? logicalWidth
        (PiRLCRetainedGeometry.oneColumn geometry).val
        { major := source.val, middle := round.val, minor := 0 }
    some (addSelected SparseForm.empty selected)) = some symbolForm
  rw [loaded]
  rfl

theorem acceptedOutputProgram_form?
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiRLCRetainedGeometry.Geometry program logicalWidth)
    (source : Fin sourceCount) (round : Fin roundCount) :
    (acceptedOutputProgram program).form? logicalWidth
        (PiRLCRetainedGeometry.oneColumn geometry).val
        { major := source.val, middle := round.val, minor := 0 } =
      some (PiRLCFirst54DirectPlan.productForm
        (PiRLCRetainedInputs.first54Inputs geometry)
        (PiRLCFirst54DirectSchedule.candidateIndex
          (candidate source round))) := by
  let inputs := PiRLCRetainedInputs.first54Inputs geometry
  let productForm := PiRLCFirst54DirectPlan.productForm inputs
    (PiRLCFirst54DirectSchedule.candidateIndex (candidate source round))
  have raw := productWire_form? geometry source round
  have inside : allAccepted.offsets?
      { major := source.val, middle := round.val, minor := 0 } =
    some { major := source.val, middle := round.val, minor := 0 } := by
    simpa [allAccepted, region] using
      Region.offsets?_of_offsets allAccepted source round (0 : Fin 1)
  have loaded :
      (retainedRule allAccepted (productWire program) 0 64 1 0 1).form?
        logicalWidth (PiRLCRetainedGeometry.oneColumn geometry).val
        { major := source.val, middle := round.val, minor := 0 } =
      some (some productForm) := by
    have selected := retainedRule_form?_of_loaded allAccepted
      { major := source.val, middle := round.val, minor := 0 }
      { major := source.val, middle := round.val, minor := 0 }
      (productWire program) (PiRLCRetainedGeometry.oneColumn geometry).val
      0 64 1 0 1 productForm inside
      (by simpa [productForm, PiRLCFirst54DirectPlan.productForm, inputs]
        using raw)
    simpa [applyCoefficient] using selected
  change (Program.mk [retainedRule allAccepted (productWire program)
      0 64 1 0 1]).form? logicalWidth
      (PiRLCRetainedGeometry.oneColumn geometry).val
      { major := source.val, middle := round.val, minor := 0 } =
    some productForm
  unfold Program.form?
  change (do
    let selected ← (retainedRule allAccepted (productWire program)
      0 64 1 0 1).form? logicalWidth
        (PiRLCRetainedGeometry.oneColumn geometry).val
        { major := source.val, middle := round.val, minor := 0 }
    some (addSelected SparseForm.empty selected)) = some productForm
  rw [loaded]
  rfl

theorem acceptedGrid_row?
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiRLCRetainedGeometry.Geometry program logicalWidth)
    (source : Fin sourceCount) (round : Fin roundCount) :
    (acceptedGrid geometry).row? logicalWidth
        (Fin.encodeProd (source, Fin.encodeProd (round, (0 : Fin 1)))).val =
      some (MultiplicationFamilyPlan.forms
        (PiRLCFirst54DirectPlan.acceptedProductInterface
          (PiRLCRetainedInputs.first54Inputs geometry))
        (PiRLCFirst54DirectSchedule.candidateIndex
          (candidate source round))) := by
  let inputs := PiRLCRetainedInputs.first54Inputs geometry
  let accepted := PiRLCFirst54DirectPlan.acceptedForm inputs
    (candidate source round)
  let symbol := PiRLCFirst54DirectPlan.symbolForm inputs
    (candidate source round)
  let product := PiRLCFirst54DirectPlan.productForm inputs
    (PiRLCFirst54DirectSchedule.candidateIndex (candidate source round))
  have direct := MultiplicationGrid.Block.row?_of_results
    (acceptedGrid geometry) (PiRLCRetainedGeometry.oneColumn geometry) rfl
    source round (0 : Fin 1) accepted symbol product
    (by simpa [accepted, inputs] using
      acceptedLeftProgram_form? geometry source round)
    (by simpa [symbol, inputs] using
      acceptedRightProgram_form? geometry source round)
    (by simpa [product, inputs] using
      acceptedOutputProgram_form? geometry source round)
  simpa [MultiplicationFamilyPlan.forms,
    PiRLCFirst54DirectPlan.acceptedProductInterface,
    PiRLCFirst54DirectSchedule.candidate_candidateIndex,
    accepted, symbol, product, inputs] using direct

theorem valueRightProgram_form?
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiRLCRetainedGeometry.Geometry program logicalWidth)
    (source : Fin sourceCount) (round : Fin roundCount)
    (slot : Fin valueSlotCount) :
    (valueRightProgram program).form? logicalWidth
        (PiRLCRetainedGeometry.oneColumn geometry).val
        { major := source.val, middle := round.val, minor := slot.val } =
      some (PiRLCFirst54DirectPlan.productForm
        (PiRLCRetainedInputs.first54Inputs geometry)
        (PiRLCFirst54DirectSchedule.candidateIndex
          (candidate source round))) := by
  let inputs := PiRLCRetainedInputs.first54Inputs geometry
  let productForm := PiRLCFirst54DirectPlan.productForm inputs
    (PiRLCFirst54DirectSchedule.candidateIndex (candidate source round))
  have inside : allValue.offsets?
      { major := source.val, middle := round.val, minor := slot.val } =
    some { major := source.val, middle := round.val, minor := slot.val } := by
    simpa [allValue, region] using
      Region.offsets?_of_offsets allValue source round slot
  have raw := productWire_form? geometry source round
  have loaded :
      (retainedRule allValue (productWire program) 0 64 1 0 1).form?
        logicalWidth (PiRLCRetainedGeometry.oneColumn geometry).val
        { major := source.val, middle := round.val, minor := slot.val } =
      some (some productForm) := by
    have selected := retainedRule_form?_of_loaded allValue
      { major := source.val, middle := round.val, minor := slot.val }
      { major := source.val, middle := round.val, minor := slot.val }
      (productWire program) (PiRLCRetainedGeometry.oneColumn geometry).val
      0 64 1 0 1 productForm inside
      (by simpa [productForm, PiRLCFirst54DirectPlan.productForm, inputs]
        using raw)
    simpa [applyCoefficient] using selected
  change (Program.mk [retainedRule allValue (productWire program)
      0 64 1 0 1]).form? logicalWidth
      (PiRLCRetainedGeometry.oneColumn geometry).val
      { major := source.val, middle := round.val, minor := slot.val } =
    some productForm
  unfold Program.form?
  change (do
    let selected ← (retainedRule allValue (productWire program)
      0 64 1 0 1).form? logicalWidth
        (PiRLCRetainedGeometry.oneColumn geometry).val
        { major := source.val, middle := round.val, minor := slot.val }
    some (addSelected SparseForm.empty selected)) = some productForm
  rw [loaded]
  rfl

theorem valueOutputProgram_form?
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiRLCRetainedGeometry.Geometry program logicalWidth)
    (source : Fin sourceCount) (round : Fin roundCount)
    (slot : Fin valueSlotCount) :
    (valueOutputProgram program).form? logicalWidth
        (PiRLCRetainedGeometry.oneColumn geometry).val
        { major := source.val, middle := round.val, minor := slot.val } =
      some (PiRLCFirst54DirectPlan.valueDifferenceForm
        (PiRLCRetainedInputs.first54Inputs geometry)
        (valueDescriptor source round slot)) := by
  let inputs := PiRLCRetainedInputs.first54Inputs geometry
  let current := PiRLCFirst54DirectPlan.valueOutputForm inputs
    (valueDescriptor source round slot)
  have allInside : allValue.offsets?
      { major := source.val, middle := round.val, minor := slot.val } =
    some { major := source.val, middle := round.val, minor := slot.val } := by
    simpa [allValue, region] using
      Region.offsets?_of_offsets allValue source round slot
  have currentRaw := valueWire_form? geometry source round slot
  have currentLoaded :
      (retainedRule allValue (valueWire program) 0 3456 54 1 1).form?
        logicalWidth (PiRLCRetainedGeometry.oneColumn geometry).val
        { major := source.val, middle := round.val, minor := slot.val } =
      some (some current) := by
    have selected := retainedRule_form?_of_loaded allValue
      { major := source.val, middle := round.val, minor := slot.val }
      { major := source.val, middle := round.val, minor := slot.val }
      (valueWire program) (PiRLCRetainedGeometry.oneColumn geometry).val
      0 3456 54 1 1 current allInside
      (by simpa [current, PiRLCFirst54DirectPlan.valueOutputForm, inputs]
        using currentRaw)
    simpa [applyCoefficient] using selected
  by_cases first : round.val = 0
  · have previousOutside :
        (retainedRule laterValueAll (valueWire program)
          0 3456 54 1 (-1)).form? logicalWidth
          (PiRLCRetainedGeometry.oneColumn geometry).val
          { major := source.val, middle := round.val, minor := slot.val } =
        some none := by
      apply rule_form?_outside
      change ¬(0 ≤ source.val ∧ source.val < 17 ∧
        1 ≤ round.val ∧ round.val < 64 ∧
        0 ≤ slot.val ∧ slot.val < 54)
      omega
    change (Program.mk [
        retainedRule allValue (valueWire program) 0 3456 54 1 1,
        retainedRule laterValueAll (valueWire program) 0 3456 54 1 (-1)]).form?
      logicalWidth (PiRLCRetainedGeometry.oneColumn geometry).val
        { major := source.val, middle := round.val, minor := slot.val } =
      some (PiRLCFirst54DirectPlan.valueDifferenceForm inputs
        (valueDescriptor source round slot))
    have folded := Program.two_form?_of_results
      (retainedRule allValue (valueWire program) 0 3456 54 1 1)
      (retainedRule laterValueAll (valueWire program) 0 3456 54 1 (-1))
      (PiRLCRetainedGeometry.oneColumn geometry).val
      { major := source.val, middle := round.val, minor := slot.val }
      (some current) none currentLoaded previousOutside
    have priorEmpty : PiRLCFirst54DirectPlan.priorValueForm inputs
        (valueDescriptor source round slot) = SparseForm.empty := by
      unfold PiRLCFirst54DirectPlan.priorValueForm
      rw [dif_pos]
      simpa [valueDescriptor, candidate] using first
    have difference : PiRLCFirst54DirectPlan.valueDifferenceForm inputs
        (valueDescriptor source round slot) = current := by
      unfold PiRLCFirst54DirectPlan.valueDifferenceForm
        PiRLCFirst54DirectPlan.subtract
      rw [priorEmpty]
      change SparseForm.add current
        (SparseForm.scale (-1) SparseForm.empty) = current
      exact add_scaled_empty current
    rw [difference]
    have foldResult :
        addSelected (addSelected SparseForm.empty (some current)) none =
          current := by
      unfold addSelected
      exact empty_add current
    rw [foldResult] at folded
    exact folded
  · let priorRound := previousRound round first
    let middle : Fin 63 := laterRoundOffset round first
    let prior := PiRLCFirst54DirectPlan.priorValueForm inputs
      (valueDescriptor source round slot)
    have laterInside : laterValueAll.offsets?
        { major := source.val, middle := round.val, minor := slot.val } =
      some { major := source.val, middle := middle.val, minor := slot.val } := by
      have selected := Region.offsets?_of_offsets laterValueAll source middle slot
      have roundEq : 1 + middle.val = round.val := by
        dsimp only [middle, laterRoundOffset]
        omega
      change laterValueAll.offsets? {
          major := 0 + source.val
          middle := 1 + middle.val
          minor := 0 + slot.val } =
        some { major := source.val, middle := middle.val, minor := slot.val }
        at selected
      simp only [Nat.zero_add, roundEq] at selected
      exact selected
    have priorRaw := valueWire_form? geometry source priorRound slot
    have priorLoaded :
        (retainedRule laterValueAll (valueWire program)
          0 3456 54 1 (-1)).form? logicalWidth
          (PiRLCRetainedGeometry.oneColumn geometry).val
          { major := source.val, middle := round.val, minor := slot.val } =
        some (some (SparseForm.scale (-1) prior)) := by
      have selected := retainedRule_form?_of_loaded laterValueAll
        { major := source.val, middle := round.val, minor := slot.val }
        { major := source.val, middle := middle.val, minor := slot.val }
        (valueWire program) (PiRLCRetainedGeometry.oneColumn geometry).val
        0 3456 54 1 (-1) prior laterInside
        (by simpa [prior, PiRLCFirst54DirectPlan.priorValueForm,
          valueDescriptor, candidate, first, previousCandidate_eq,
          priorRound, inputs] using priorRaw)
      simpa [applyCoefficient, negOne_ne_one] using selected
    change (Program.mk [
        retainedRule allValue (valueWire program) 0 3456 54 1 1,
        retainedRule laterValueAll (valueWire program) 0 3456 54 1 (-1)]).form?
      logicalWidth (PiRLCRetainedGeometry.oneColumn geometry).val
        { major := source.val, middle := round.val, minor := slot.val } =
      some (PiRLCFirst54DirectPlan.valueDifferenceForm inputs
        (valueDescriptor source round slot))
    have folded := Program.two_form?_of_results
      (retainedRule allValue (valueWire program) 0 3456 54 1 1)
      (retainedRule laterValueAll (valueWire program) 0 3456 54 1 (-1))
      (PiRLCRetainedGeometry.oneColumn geometry).val
      { major := source.val, middle := round.val, minor := slot.val }
      (some current) (some (SparseForm.scale (-1) prior))
      currentLoaded priorLoaded
    have difference : PiRLCFirst54DirectPlan.valueDifferenceForm inputs
        (valueDescriptor source round slot) =
      SparseForm.add current (SparseForm.scale (-1) prior) := by
      rfl
    rw [difference]
    have foldResult : addSelected
        (addSelected SparseForm.empty (some current))
        (some (SparseForm.scale (-1) prior)) =
      SparseForm.add current (SparseForm.scale (-1) prior) := by
      change SparseForm.add (SparseForm.add SparseForm.empty current)
        (SparseForm.scale (-1) prior) =
          SparseForm.add current (SparseForm.scale (-1) prior)
      rw [empty_add]
    rw [foldResult] at folded
    exact folded

theorem valueLeftProgram_form?
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiRLCRetainedGeometry.Geometry program logicalWidth)
    (source : Fin sourceCount) (round : Fin roundCount)
    (slot : Fin valueSlotCount) :
    (valueLeftProgram program).form? logicalWidth
        (PiRLCRetainedGeometry.oneColumn geometry).val
        { major := source.val, middle := round.val, minor := slot.val } =
      some (PiRLCFirst54DirectPlan.priorPositionForm
        (PiRLCRetainedInputs.first54Inputs geometry)
        (candidate source round) (First54ValueStep.positionSlot slot)) := by
  let inputs := PiRLCRetainedInputs.first54Inputs geometry
  let oneForm := PiRLCFirst54DirectPlan.oneForm inputs
  let prior := PiRLCFirst54DirectPlan.priorPositionForm inputs
    (candidate source round) (First54ValueStep.positionSlot slot)
  by_cases first : round.val = 0
  · have laterOutside :
        (retainedRule laterValueAll (positionWire program)
          0 3520 55 1 1).form? logicalWidth
          (PiRLCRetainedGeometry.oneColumn geometry).val
          { major := source.val, middle := round.val, minor := slot.val } =
        some none := by
      apply rule_form?_outside
      change ¬(0 ≤ source.val ∧ source.val < 17 ∧
        1 ≤ round.val ∧ round.val < 64 ∧
        0 ≤ slot.val ∧ slot.val < 54)
      omega
    by_cases firstSlot : slot.val = 0
    · have oneLoaded :
          (constantRule firstSlotZero 1).form? logicalWidth
            (PiRLCRetainedGeometry.oneColumn geometry).val
            { major := source.val, middle := round.val, minor := slot.val } =
          some (some oneForm) := by
        have selected := constantRule_form? firstSlotZero source
          (0 : Fin 1) (0 : Fin 1)
          (PiRLCRetainedGeometry.oneColumn geometry) (1 : F)
        simpa [firstSlotZero, region, first, firstSlot, oneForm,
          PiRLCFirst54DirectPlan.oneForm, inputs] using selected
      change (Program.mk [
          constantRule firstSlotZero 1,
          retainedRule laterValueAll (positionWire program) 0 3520 55 1 1]).form?
        logicalWidth (PiRLCRetainedGeometry.oneColumn geometry).val
          { major := source.val, middle := round.val, minor := slot.val } =
        some prior
      have folded := Program.two_form?_of_results
        (constantRule firstSlotZero 1)
        (retainedRule laterValueAll (positionWire program) 0 3520 55 1 1)
        (PiRLCRetainedGeometry.oneColumn geometry).val
        { major := source.val, middle := round.val, minor := slot.val }
        (some oneForm) none oneLoaded laterOutside
      have priorEq : prior = oneForm := by
        unfold prior PiRLCFirst54DirectPlan.priorPositionForm
          PiRLCFirst54DirectPlan.initialPositionForm
        rw [dif_pos]
        · rw [if_pos]
          simpa [First54ValueStep.positionSlot] using firstSlot
        · simpa [candidate] using first
      rw [priorEq]
      have foldResult :
          addSelected (addSelected SparseForm.empty (some oneForm)) none =
            oneForm := by
        change SparseForm.add SparseForm.empty oneForm = oneForm
        exact empty_add oneForm
      rw [foldResult] at folded
      exact folded
    · have constantOutside :
          (constantRule firstSlotZero 1).form? logicalWidth
            (PiRLCRetainedGeometry.oneColumn geometry).val
            { major := source.val, middle := round.val, minor := slot.val } =
          some none := by
        apply rule_form?_outside
        change ¬(0 ≤ source.val ∧ source.val < 17 ∧
          0 ≤ round.val ∧ round.val < 1 ∧
          0 ≤ slot.val ∧ slot.val < 1)
        omega
      change (Program.mk [
          constantRule firstSlotZero 1,
          retainedRule laterValueAll (positionWire program) 0 3520 55 1 1]).form?
        logicalWidth (PiRLCRetainedGeometry.oneColumn geometry).val
          { major := source.val, middle := round.val, minor := slot.val } =
        some prior
      have folded := Program.two_form?_of_results
        (constantRule firstSlotZero 1)
        (retainedRule laterValueAll (positionWire program) 0 3520 55 1 1)
        (PiRLCRetainedGeometry.oneColumn geometry).val
        { major := source.val, middle := round.val, minor := slot.val }
        none none constantOutside laterOutside
      have priorEq : prior = SparseForm.empty := by
        unfold prior PiRLCFirst54DirectPlan.priorPositionForm
          PiRLCFirst54DirectPlan.initialPositionForm
        rw [dif_pos]
        · rw [if_neg]
          simpa [First54ValueStep.positionSlot] using firstSlot
        · simpa [candidate] using first
      rw [priorEq]
      exact folded
  · let priorRound := previousRound round first
    let middle : Fin 63 := laterRoundOffset round first
    let positionSlot : Fin positionSlotCount :=
      First54ValueStep.positionSlot slot
    have constantOutside :
        (constantRule firstSlotZero 1).form? logicalWidth
          (PiRLCRetainedGeometry.oneColumn geometry).val
          { major := source.val, middle := round.val, minor := slot.val } =
        some none := by
      apply rule_form?_outside
      change ¬(0 ≤ source.val ∧ source.val < 17 ∧
        0 ≤ round.val ∧ round.val < 1 ∧
        0 ≤ slot.val ∧ slot.val < 1)
      omega
    have laterInside : laterValueAll.offsets?
        { major := source.val, middle := round.val, minor := slot.val } =
      some { major := source.val, middle := middle.val, minor := slot.val } := by
      have selected := Region.offsets?_of_offsets laterValueAll source middle slot
      have roundEq : 1 + middle.val = round.val := by
        dsimp only [middle, laterRoundOffset]
        omega
      change laterValueAll.offsets? {
          major := 0 + source.val
          middle := 1 + middle.val
          minor := 0 + slot.val } =
        some { major := source.val, middle := middle.val, minor := slot.val }
        at selected
      simp only [Nat.zero_add, roundEq] at selected
      exact selected
    have raw := positionWire_form? geometry source priorRound positionSlot
    have priorLoaded :
        (retainedRule laterValueAll (positionWire program)
          0 3520 55 1 1).form? logicalWidth
          (PiRLCRetainedGeometry.oneColumn geometry).val
          { major := source.val, middle := round.val, minor := slot.val } =
        some (some prior) := by
      have selected := retainedRule_form?_of_loaded laterValueAll
        { major := source.val, middle := round.val, minor := slot.val }
        { major := source.val, middle := middle.val, minor := slot.val }
        (positionWire program) (PiRLCRetainedGeometry.oneColumn geometry).val
        0 3520 55 1 1 prior laterInside
        (by simpa [prior, PiRLCFirst54DirectPlan.priorPositionForm,
          candidate, first, previousCandidate_eq, priorRound, positionSlot,
          positionDescriptor, inputs] using raw)
      simpa [applyCoefficient] using selected
    change (Program.mk [
        constantRule firstSlotZero 1,
        retainedRule laterValueAll (positionWire program) 0 3520 55 1 1]).form?
      logicalWidth (PiRLCRetainedGeometry.oneColumn geometry).val
        { major := source.val, middle := round.val, minor := slot.val } =
      some prior
    have folded := Program.two_form?_of_results
      (constantRule firstSlotZero 1)
      (retainedRule laterValueAll (positionWire program) 0 3520 55 1 1)
      (PiRLCRetainedGeometry.oneColumn geometry).val
      { major := source.val, middle := round.val, minor := slot.val }
      none (some prior) constantOutside priorLoaded
    have foldResult :
        addSelected (addSelected SparseForm.empty none) (some prior) = prior := by
      change SparseForm.add SparseForm.empty prior = prior
      exact empty_add prior
    rw [foldResult] at folded
    exact folded

theorem valueGrid_row?
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiRLCRetainedGeometry.Geometry program logicalWidth)
    (source : Fin sourceCount) (round : Fin roundCount)
    (slot : Fin valueSlotCount) :
    (valueGrid geometry).row? logicalWidth
        (Fin.encodeProd (source, Fin.encodeProd (round, slot))).val =
      some (MultiplicationFamilyPlan.forms
        (PiRLCFirst54DirectPlan.valueInterface
          (PiRLCRetainedInputs.first54Inputs geometry))
        (PiRLCFirst54DirectSchedule.valueIndex
          (valueDescriptor source round slot))) := by
  let inputs := PiRLCRetainedInputs.first54Inputs geometry
  let left := PiRLCFirst54DirectPlan.priorPositionForm inputs
    (candidate source round) (First54ValueStep.positionSlot slot)
  let right := PiRLCFirst54DirectPlan.productForm inputs
    (PiRLCFirst54DirectSchedule.candidateIndex (candidate source round))
  let output := PiRLCFirst54DirectPlan.valueDifferenceForm inputs
    (valueDescriptor source round slot)
  have direct := MultiplicationGrid.Block.row?_of_results
    (valueGrid geometry) (PiRLCRetainedGeometry.oneColumn geometry) rfl
    source round slot left right output
    (by simpa [left, inputs] using
      valueLeftProgram_form? geometry source round slot)
    (by simpa [right, inputs] using
      valueRightProgram_form? geometry source round slot)
    (by simpa [output, inputs] using
      valueOutputProgram_form? geometry source round slot)
  simpa [MultiplicationFamilyPlan.forms,
    PiRLCFirst54DirectPlan.valueInterface,
    PiRLCFirst54DirectSchedule.value_valueIndex,
    left, right, output, inputs] using direct

theorem positionLeftProgram_form?
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiRLCRetainedGeometry.Geometry program logicalWidth)
    (source : Fin sourceCount) (round : Fin roundCount)
    (slot : Fin positionSlotCount) :
    (positionLeftProgram program).form? logicalWidth
        (PiRLCRetainedGeometry.oneColumn geometry).val
        { major := source.val, middle := round.val, minor := slot.val } =
      some (PiRLCFirst54DirectPlan.acceptedForm
        (PiRLCRetainedInputs.first54Inputs geometry)
        (candidate source round)) := by
  let inputs := PiRLCRetainedInputs.first54Inputs geometry
  let oneForm := PiRLCFirst54DirectPlan.oneForm inputs
  let rejectForm := PiRLCFirst54DirectPlan.rejectForm inputs
    (candidate source round)
  have allInside : allPosition.offsets?
      { major := source.val, middle := round.val, minor := slot.val } =
    some { major := source.val, middle := round.val, minor := slot.val } := by
    simpa [allPosition, region] using
      Region.offsets?_of_offsets allPosition source round slot
  have oneLoaded :
      (constantRule allPosition 1).form? logicalWidth
        (PiRLCRetainedGeometry.oneColumn geometry).val
        { major := source.val, middle := round.val, minor := slot.val } =
      some (some oneForm) := by
    have selected := constantRule_form? allPosition source round slot
      (PiRLCRetainedGeometry.oneColumn geometry) (1 : F)
    simpa [allPosition, region, oneForm,
      PiRLCFirst54DirectPlan.oneForm, inputs] using selected
  have raw := rejectWire_form? geometry source round
  have rejectLoaded :
      (retainedRule allPosition (rejectWire program) 0 64 1 0 (-1)).form?
        logicalWidth (PiRLCRetainedGeometry.oneColumn geometry).val
        { major := source.val, middle := round.val, minor := slot.val } =
      some (some (SparseForm.scale (-1) rejectForm)) := by
    have selected := retainedRule_form?_of_loaded allPosition
      { major := source.val, middle := round.val, minor := slot.val }
      { major := source.val, middle := round.val, minor := slot.val }
      (rejectWire program) (PiRLCRetainedGeometry.oneColumn geometry).val
      0 64 1 0 (-1) rejectForm allInside
      (by simpa [rejectForm, PiRLCFirst54DirectPlan.rejectForm, inputs]
        using raw)
    simpa [applyCoefficient, negOne_ne_one] using selected
  change (Program.mk [
      constantRule allPosition 1,
      retainedRule allPosition (rejectWire program) 0 64 1 0 (-1)]).form?
    logicalWidth (PiRLCRetainedGeometry.oneColumn geometry).val
      { major := source.val, middle := round.val, minor := slot.val } =
    some (PiRLCFirst54DirectPlan.acceptedForm inputs (candidate source round))
  have folded := Program.two_form?_of_results
    (constantRule allPosition 1)
    (retainedRule allPosition (rejectWire program) 0 64 1 0 (-1))
    (PiRLCRetainedGeometry.oneColumn geometry).val
    { major := source.val, middle := round.val, minor := slot.val }
    (some oneForm) (some (SparseForm.scale (-1) rejectForm))
    oneLoaded rejectLoaded
  have acceptedEq : PiRLCFirst54DirectPlan.acceptedForm inputs
      (candidate source round) =
    SparseForm.add oneForm (SparseForm.scale (-1) rejectForm) := by
    rfl
  rw [acceptedEq]
  have foldResult : addSelected
      (addSelected SparseForm.empty (some oneForm))
      (some (SparseForm.scale (-1) rejectForm)) =
    SparseForm.add oneForm (SparseForm.scale (-1) rejectForm) := by
    change SparseForm.add (SparseForm.add SparseForm.empty oneForm)
      (SparseForm.scale (-1) rejectForm) = _
    rw [empty_add]
  rw [foldResult] at folded
  exact folded

theorem positionOutputProgram_form?
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiRLCRetainedGeometry.Geometry program logicalWidth)
    (source : Fin sourceCount) (round : Fin roundCount)
    (slot : Fin positionSlotCount) :
    (positionOutputProgram program).form? logicalWidth
        (PiRLCRetainedGeometry.oneColumn geometry).val
        { major := source.val, middle := round.val, minor := slot.val } =
      some (PiRLCFirst54DirectPlan.positionDifferenceForm
        (PiRLCRetainedInputs.first54Inputs geometry)
        (positionDescriptor source round slot)) := by
  let inputs := PiRLCRetainedInputs.first54Inputs geometry
  let current := PiRLCFirst54DirectPlan.positionOutputForm inputs
    (positionDescriptor source round slot)
  have allInside : allPosition.offsets?
      { major := source.val, middle := round.val, minor := slot.val } =
    some { major := source.val, middle := round.val, minor := slot.val } := by
    simpa [allPosition, region] using
      Region.offsets?_of_offsets allPosition source round slot
  have currentRaw := positionWire_form? geometry source round slot
  have currentLoaded :
      (retainedRule allPosition (positionWire program) 0 3520 55 1 1).form?
        logicalWidth (PiRLCRetainedGeometry.oneColumn geometry).val
        { major := source.val, middle := round.val, minor := slot.val } =
      some (some current) := by
    have selected := retainedRule_form?_of_loaded allPosition
      { major := source.val, middle := round.val, minor := slot.val }
      { major := source.val, middle := round.val, minor := slot.val }
      (positionWire program) (PiRLCRetainedGeometry.oneColumn geometry).val
      0 3520 55 1 1 current allInside
      (by simpa [current, PiRLCFirst54DirectPlan.positionOutputForm, inputs]
        using currentRaw)
    simpa [applyCoefficient] using selected
  by_cases first : round.val = 0
  · have previousOutside :
        (retainedRule laterPositionAll (positionWire program)
          0 3520 55 1 (-1)).form? logicalWidth
          (PiRLCRetainedGeometry.oneColumn geometry).val
          { major := source.val, middle := round.val, minor := slot.val } =
        some none := by
      apply rule_form?_outside
      change ¬(0 ≤ source.val ∧ source.val < 17 ∧
        1 ≤ round.val ∧ round.val < 64 ∧
        0 ≤ slot.val ∧ slot.val < 55)
      omega
    by_cases firstSlot : slot.val = 0
    · let negativeOne := SparseForm.singleton
        (PiRLCRetainedGeometry.oneColumn geometry) (-1)
      have constantLoaded :
          (constantRule firstSlotZero (-1)).form? logicalWidth
            (PiRLCRetainedGeometry.oneColumn geometry).val
            { major := source.val, middle := round.val, minor := slot.val } =
          some (some negativeOne) := by
        have selected := constantRule_form? firstSlotZero source
          (0 : Fin 1) (0 : Fin 1)
          (PiRLCRetainedGeometry.oneColumn geometry) (-1 : F)
        simpa [firstSlotZero, region, first, firstSlot, negativeOne]
          using selected
      change (Program.mk [
          retainedRule allPosition (positionWire program) 0 3520 55 1 1,
          constantRule firstSlotZero (-1),
          retainedRule laterPositionAll (positionWire program)
            0 3520 55 1 (-1)]).form? logicalWidth
        (PiRLCRetainedGeometry.oneColumn geometry).val
          { major := source.val, middle := round.val, minor := slot.val } =
        some (PiRLCFirst54DirectPlan.positionDifferenceForm inputs
          (positionDescriptor source round slot))
      have folded := Program.three_form?_of_results
        (retainedRule allPosition (positionWire program) 0 3520 55 1 1)
        (constantRule firstSlotZero (-1))
        (retainedRule laterPositionAll (positionWire program)
          0 3520 55 1 (-1))
        (PiRLCRetainedGeometry.oneColumn geometry).val
        { major := source.val, middle := round.val, minor := slot.val }
        (some current) (some negativeOne) none
        currentLoaded constantLoaded previousOutside
      have priorEq : PiRLCFirst54DirectPlan.priorPositionForm inputs
          (candidate source round) slot =
        PiRLCFirst54DirectPlan.oneForm inputs := by
        unfold PiRLCFirst54DirectPlan.priorPositionForm
          PiRLCFirst54DirectPlan.initialPositionForm
        rw [dif_pos]
        · rw [if_pos firstSlot]
        · simpa [candidate] using first
      have negativeEq : negativeOne = SparseForm.scale (-1)
          (PiRLCFirst54DirectPlan.oneForm inputs) := by
        unfold negativeOne PiRLCFirst54DirectPlan.oneForm inputs
        unfold PiRLCRetainedInputs.first54Inputs
        apply congrArg SparseForm.mk
        simp [SparseForm.singleton]
      have difference : PiRLCFirst54DirectPlan.positionDifferenceForm inputs
          (positionDescriptor source round slot) =
        SparseForm.add current negativeOne := by
        change PiRLCFirst54DirectPlan.subtract current
          (PiRLCFirst54DirectPlan.priorPositionForm inputs
            (candidate source round) slot) = _
        rw [priorEq]
        unfold PiRLCFirst54DirectPlan.subtract
        rw [← negativeEq]
      rw [difference]
      have foldResult : addSelected
          (addSelected (addSelected SparseForm.empty (some current))
            (some negativeOne)) none = SparseForm.add current negativeOne := by
        change SparseForm.add (SparseForm.add SparseForm.empty current)
          negativeOne = SparseForm.add current negativeOne
        rw [empty_add]
      rw [foldResult] at folded
      exact folded
    · have constantOutside :
          (constantRule firstSlotZero (-1)).form? logicalWidth
            (PiRLCRetainedGeometry.oneColumn geometry).val
            { major := source.val, middle := round.val, minor := slot.val } =
          some none := by
        apply rule_form?_outside
        change ¬(0 ≤ source.val ∧ source.val < 17 ∧
          0 ≤ round.val ∧ round.val < 1 ∧
          0 ≤ slot.val ∧ slot.val < 1)
        omega
      change (Program.mk [
          retainedRule allPosition (positionWire program) 0 3520 55 1 1,
          constantRule firstSlotZero (-1),
          retainedRule laterPositionAll (positionWire program)
            0 3520 55 1 (-1)]).form? logicalWidth
        (PiRLCRetainedGeometry.oneColumn geometry).val
          { major := source.val, middle := round.val, minor := slot.val } =
        some (PiRLCFirst54DirectPlan.positionDifferenceForm inputs
          (positionDescriptor source round slot))
      have folded := Program.three_form?_of_results
        (retainedRule allPosition (positionWire program) 0 3520 55 1 1)
        (constantRule firstSlotZero (-1))
        (retainedRule laterPositionAll (positionWire program)
          0 3520 55 1 (-1))
        (PiRLCRetainedGeometry.oneColumn geometry).val
        { major := source.val, middle := round.val, minor := slot.val }
        (some current) none none currentLoaded constantOutside previousOutside
      have priorEmpty : PiRLCFirst54DirectPlan.priorPositionForm inputs
          (candidate source round) slot = SparseForm.empty := by
        unfold PiRLCFirst54DirectPlan.priorPositionForm
          PiRLCFirst54DirectPlan.initialPositionForm
        rw [dif_pos]
        · rw [if_neg firstSlot]
        · simpa [candidate] using first
      have difference : PiRLCFirst54DirectPlan.positionDifferenceForm inputs
          (positionDescriptor source round slot) = current := by
        change PiRLCFirst54DirectPlan.subtract current
          (PiRLCFirst54DirectPlan.priorPositionForm inputs
            (candidate source round) slot) = _
        rw [priorEmpty]
        exact add_scaled_empty current
      rw [difference]
      have foldResult : addSelected
          (addSelected (addSelected SparseForm.empty (some current)) none)
          none = current := by
        change SparseForm.add SparseForm.empty current = current
        exact empty_add current
      rw [foldResult] at folded
      exact folded
  · let priorRound := previousRound round first
    let middle : Fin 63 := laterRoundOffset round first
    let prior := PiRLCFirst54DirectPlan.priorPositionForm inputs
      (candidate source round) slot
    have constantOutside :
        (constantRule firstSlotZero (-1)).form? logicalWidth
          (PiRLCRetainedGeometry.oneColumn geometry).val
          { major := source.val, middle := round.val, minor := slot.val } =
        some none := by
      apply rule_form?_outside
      change ¬(0 ≤ source.val ∧ source.val < 17 ∧
        0 ≤ round.val ∧ round.val < 1 ∧
        0 ≤ slot.val ∧ slot.val < 1)
      omega
    have laterInside : laterPositionAll.offsets?
        { major := source.val, middle := round.val, minor := slot.val } =
      some { major := source.val, middle := middle.val, minor := slot.val } := by
      have selected := Region.offsets?_of_offsets
        laterPositionAll source middle slot
      have roundEq : 1 + middle.val = round.val := by
        dsimp only [middle, laterRoundOffset]
        omega
      change laterPositionAll.offsets? {
          major := 0 + source.val
          middle := 1 + middle.val
          minor := 0 + slot.val } =
        some { major := source.val, middle := middle.val, minor := slot.val }
        at selected
      simp only [Nat.zero_add, roundEq] at selected
      exact selected
    have priorRaw := positionWire_form? geometry source priorRound slot
    have priorLoaded :
        (retainedRule laterPositionAll (positionWire program)
          0 3520 55 1 (-1)).form? logicalWidth
          (PiRLCRetainedGeometry.oneColumn geometry).val
          { major := source.val, middle := round.val, minor := slot.val } =
        some (some (SparseForm.scale (-1) prior)) := by
      have selected := retainedRule_form?_of_loaded laterPositionAll
        { major := source.val, middle := round.val, minor := slot.val }
        { major := source.val, middle := middle.val, minor := slot.val }
        (positionWire program) (PiRLCRetainedGeometry.oneColumn geometry).val
        0 3520 55 1 (-1) prior laterInside
        (by simpa [prior, PiRLCFirst54DirectPlan.priorPositionForm,
          candidate, first, previousCandidate_eq, priorRound,
          positionDescriptor, inputs] using priorRaw)
      simpa [applyCoefficient, negOne_ne_one] using selected
    change (Program.mk [
        retainedRule allPosition (positionWire program) 0 3520 55 1 1,
        constantRule firstSlotZero (-1),
        retainedRule laterPositionAll (positionWire program)
          0 3520 55 1 (-1)]).form? logicalWidth
      (PiRLCRetainedGeometry.oneColumn geometry).val
        { major := source.val, middle := round.val, minor := slot.val } =
      some (PiRLCFirst54DirectPlan.positionDifferenceForm inputs
        (positionDescriptor source round slot))
    have folded := Program.three_form?_of_results
      (retainedRule allPosition (positionWire program) 0 3520 55 1 1)
      (constantRule firstSlotZero (-1))
      (retainedRule laterPositionAll (positionWire program)
        0 3520 55 1 (-1))
      (PiRLCRetainedGeometry.oneColumn geometry).val
      { major := source.val, middle := round.val, minor := slot.val }
      (some current) none (some (SparseForm.scale (-1) prior))
      currentLoaded constantOutside priorLoaded
    have difference : PiRLCFirst54DirectPlan.positionDifferenceForm inputs
        (positionDescriptor source round slot) =
      SparseForm.add current (SparseForm.scale (-1) prior) := by
      rfl
    rw [difference]
    have foldResult : addSelected
        (addSelected (addSelected SparseForm.empty (some current)) none)
        (some (SparseForm.scale (-1) prior)) =
      SparseForm.add current (SparseForm.scale (-1) prior) := by
      change SparseForm.add (SparseForm.add SparseForm.empty current)
        (SparseForm.scale (-1) prior) = _
      rw [empty_add]
    rw [foldResult] at folded
    exact folded

end NightstreamFPrime.Export.Stage1.PiRLCFirst54MatrixProgram
