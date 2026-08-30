import NightstreamFPrime.Export.Stage1.PiRLCFirst54MatrixProgramSemantics

/-!
Owns the six-case position-delta bridge for the compact PiRLC First54 matrix
program. The split follows the canonical initial and later-round position
semantics.

This module does not own the accepted-symbol, value, or final-pin families.
-/

namespace NightstreamFPrime.Export.Stage1.PiRLCFirst54MatrixProgram

open NightstreamFPrime.Export.MatrixProgram
open NightstreamFPrime.Export.MatrixProgram.AffineGrid
open NightstreamFPrime.Gadgets.Sampling
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Spec

private theorem negOne_ne_one_position : (-1 : F) ≠ 1 := by
  decide

private theorem empty_add_position {logicalWidth : Nat}
    (form : SparseForm logicalWidth) :
    SparseForm.add SparseForm.empty form = form := by
  cases form
  rfl

private theorem add_scaled_empty_position {logicalWidth : Nat}
    (form : SparseForm logicalWidth) :
    SparseForm.add form (SparseForm.scale (-1) SparseForm.empty) = form := by
  cases form
  unfold SparseForm.add SparseForm.scale SparseForm.empty
  simp

private theorem initial_delta_zero
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (inputs : PiRLCFirst54DirectPlan.Inputs program logicalWidth)
    (source : Fin sourceCount) (round : Fin roundCount)
    (slot : Fin positionSlotCount)
    (first : round.val = 0) (zero : slot.val = 0) :
    PiRLCFirst54DirectPlan.positionDeltaForm inputs
        (positionDescriptor source round slot) =
      SparseForm.scale (-1) (PiRLCFirst54DirectPlan.oneForm inputs) := by
  unfold PiRLCFirst54DirectPlan.positionDeltaForm
    PiRLCFirst54DirectPlan.previousPositionForm
    PiRLCFirst54DirectPlan.priorPositionForm
    PiRLCFirst54DirectPlan.initialPositionForm
    PiRLCFirst54DirectPlan.subtract
  simp only [positionDescriptor, candidate, zero, first,
    First54Step.fullSlot, ↓reduceIte, ↓reduceDIte]
  exact empty_add_position _

private theorem initial_delta_one
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (inputs : PiRLCFirst54DirectPlan.Inputs program logicalWidth)
    (source : Fin sourceCount) (round : Fin roundCount)
    (slot : Fin positionSlotCount)
    (first : round.val = 0) (one : slot.val = 1) :
    PiRLCFirst54DirectPlan.positionDeltaForm inputs
        (positionDescriptor source round slot) =
      PiRLCFirst54DirectPlan.oneForm inputs := by
  unfold PiRLCFirst54DirectPlan.positionDeltaForm
    PiRLCFirst54DirectPlan.previousPositionForm
    PiRLCFirst54DirectPlan.priorPositionForm
    PiRLCFirst54DirectPlan.initialPositionForm
    PiRLCFirst54DirectPlan.subtract
  simp only [positionDescriptor, candidate, one, first,
    First54Step.fullSlot, First54Step.previousSlot,
    ↓reduceIte, ↓reduceDIte]
  exact add_scaled_empty_position _

private theorem initial_delta_other
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (inputs : PiRLCFirst54DirectPlan.Inputs program logicalWidth)
    (source : Fin sourceCount) (round : Fin roundCount)
    (slot : Fin positionSlotCount)
    (first : round.val = 0) (notZero : slot.val ≠ 0)
    (notOne : slot.val ≠ 1) :
    PiRLCFirst54DirectPlan.positionDeltaForm inputs
      (positionDescriptor source round slot) = SparseForm.empty := by
  unfold PiRLCFirst54DirectPlan.positionDeltaForm
  simp only [positionDescriptor]
  by_cases full : slot.val = First54Step.fullSlot
  · rw [if_pos full]
    unfold PiRLCFirst54DirectPlan.previousPositionForm
      PiRLCFirst54DirectPlan.priorPositionForm
      PiRLCFirst54DirectPlan.initialPositionForm
    rw [dif_neg notZero, dif_pos]
    · rw [if_neg]
      unfold First54Step.previousSlot
      simp only [First54Step.fullSlot, First54Step.slotCount,
        positionSlotCount] at *
      omega
    · simpa [positionDescriptor, candidate] using first
  · rw [if_neg full]
    unfold PiRLCFirst54DirectPlan.previousPositionForm
      PiRLCFirst54DirectPlan.subtract
    rw [dif_neg notZero]
    unfold PiRLCFirst54DirectPlan.priorPositionForm
      PiRLCFirst54DirectPlan.initialPositionForm
    rw [dif_pos]
    · rw [if_neg, dif_pos]
      · rw [if_neg notZero]
        exact add_scaled_empty_position _
      · simpa [positionDescriptor, candidate] using first
      · unfold First54Step.previousSlot
        simp only [First54Step.fullSlot, First54Step.slotCount,
          positionSlotCount] at *
        omega
    · simpa [positionDescriptor, candidate] using first

private theorem later_delta_zero
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (inputs : PiRLCFirst54DirectPlan.Inputs program logicalWidth)
    (source : Fin sourceCount) (round : Fin roundCount)
    (slot : Fin positionSlotCount) (zero : slot.val = 0) :
    PiRLCFirst54DirectPlan.positionDeltaForm inputs
        (positionDescriptor source round slot) =
      SparseForm.scale (-1)
        (PiRLCFirst54DirectPlan.priorPositionForm inputs
          (candidate source round) slot) := by
  unfold PiRLCFirst54DirectPlan.positionDeltaForm
    PiRLCFirst54DirectPlan.previousPositionForm
    PiRLCFirst54DirectPlan.subtract
  simp only [positionDescriptor, zero, First54Step.fullSlot, ↓reduceIte,
    ↓reduceDIte]
  exact empty_add_position _

private theorem later_delta_middle
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (inputs : PiRLCFirst54DirectPlan.Inputs program logicalWidth)
    (source : Fin sourceCount) (round : Fin roundCount)
    (slot : Fin positionSlotCount)
    (notZero : slot.val ≠ 0)
    (notFull : slot.val ≠ First54Step.fullSlot) :
    PiRLCFirst54DirectPlan.positionDeltaForm inputs
        (positionDescriptor source round slot) =
      SparseForm.add
        (PiRLCFirst54DirectPlan.previousPositionForm inputs
          (positionDescriptor source round slot))
        (SparseForm.scale (-1)
          (PiRLCFirst54DirectPlan.priorPositionForm inputs
            (candidate source round) slot)) := by
  unfold PiRLCFirst54DirectPlan.positionDeltaForm
  simp only [positionDescriptor, notFull, ↓reduceIte]
  rfl

private theorem later_delta_full
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (inputs : PiRLCFirst54DirectPlan.Inputs program logicalWidth)
    (source : Fin sourceCount) (round : Fin roundCount)
    (slot : Fin positionSlotCount)
    (full : slot.val = First54Step.fullSlot) :
    PiRLCFirst54DirectPlan.positionDeltaForm inputs
        (positionDescriptor source round slot) =
      PiRLCFirst54DirectPlan.previousPositionForm inputs
        (positionDescriptor source round slot) := by
  unfold PiRLCFirst54DirectPlan.positionDeltaForm
  simp only [positionDescriptor, full, ↓reduceIte]

private theorem negativeOne_eq
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiRLCRetainedGeometry.Geometry program logicalWidth) :
    SparseForm.singleton (PiRLCRetainedGeometry.oneColumn geometry) (-1) =
      SparseForm.scale (-1) (PiRLCFirst54DirectPlan.oneForm
        (PiRLCRetainedInputs.first54Inputs geometry)) := by
  unfold PiRLCFirst54DirectPlan.oneForm
    PiRLCRetainedInputs.first54Inputs
  apply congrArg SparseForm.mk
  simp [SparseForm.singleton]

private theorem positionRight_initial_zero
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiRLCRetainedGeometry.Geometry program logicalWidth)
    (source : Fin sourceCount) (round : Fin roundCount)
    (slot : Fin positionSlotCount)
    (first : round.val = 0) (zero : slot.val = 0) :
    (positionRightProgram program).form? logicalWidth
        (PiRLCRetainedGeometry.oneColumn geometry).val
        { major := source.val, middle := round.val, minor := slot.val } =
      some (PiRLCFirst54DirectPlan.positionDeltaForm
        (PiRLCRetainedInputs.first54Inputs geometry)
        (positionDescriptor source round slot)) := by
  let negativeOne := SparseForm.singleton
    (PiRLCRetainedGeometry.oneColumn geometry) (-1)
  have rule0 :
      (constantRule firstSlotZero (-1)).form? logicalWidth
        (PiRLCRetainedGeometry.oneColumn geometry).val
        { major := source.val, middle := round.val, minor := slot.val } =
      some (some negativeOne) := by
    have selected := constantRule_form? firstSlotZero source
      (0 : Fin 1) (0 : Fin 1)
      (PiRLCRetainedGeometry.oneColumn geometry) (-1 : F)
    simpa [firstSlotZero, region, first, zero, negativeOne] using selected
  have rule1 :
      (constantRule firstSlotOne 1).form? logicalWidth
        (PiRLCRetainedGeometry.oneColumn geometry).val
        { major := source.val, middle := round.val, minor := slot.val } =
      some none := by
    apply rule_form?_outside
    change ¬(0 ≤ source.val ∧ source.val < 17 ∧
      0 ≤ round.val ∧ round.val < 1 ∧
      1 ≤ slot.val ∧ slot.val < 2)
    omega
  have rule2 :
      (retainedRule laterPositionZero (positionWire program)
        0 3520 55 0 (-1)).form? logicalWidth
        (PiRLCRetainedGeometry.oneColumn geometry).val
        { major := source.val, middle := round.val, minor := slot.val } =
      some none := by
    apply rule_form?_outside
    change ¬(0 ≤ source.val ∧ source.val < 17 ∧
      1 ≤ round.val ∧ round.val < 64 ∧
      0 ≤ slot.val ∧ slot.val < 1)
    omega
  have rule3 :
      (retainedRule laterPositionMiddle (positionWire program)
        0 3520 55 1 1).form? logicalWidth
        (PiRLCRetainedGeometry.oneColumn geometry).val
        { major := source.val, middle := round.val, minor := slot.val } =
      some none := by
    apply rule_form?_outside
    change ¬(0 ≤ source.val ∧ source.val < 17 ∧
      1 ≤ round.val ∧ round.val < 64 ∧
      1 ≤ slot.val ∧ slot.val < 54)
    omega
  have rule4 :
      (retainedRule laterPositionMiddle (positionWire program)
        1 3520 55 1 (-1)).form? logicalWidth
        (PiRLCRetainedGeometry.oneColumn geometry).val
        { major := source.val, middle := round.val, minor := slot.val } =
      some none := by
    apply rule_form?_outside
    change ¬(0 ≤ source.val ∧ source.val < 17 ∧
      1 ≤ round.val ∧ round.val < 64 ∧
      1 ≤ slot.val ∧ slot.val < 54)
    omega
  have rule5 :
      (retainedRule laterPositionFull (positionWire program)
        53 3520 55 0 1).form? logicalWidth
        (PiRLCRetainedGeometry.oneColumn geometry).val
        { major := source.val, middle := round.val, minor := slot.val } =
      some none := by
    apply rule_form?_outside
    change ¬(0 ≤ source.val ∧ source.val < 17 ∧
      1 ≤ round.val ∧ round.val < 64 ∧
      54 ≤ slot.val ∧ slot.val < 55)
    omega
  have folded := Program.six_form?_of_results
    (constantRule firstSlotZero (-1))
    (constantRule firstSlotOne 1)
    (retainedRule laterPositionZero (positionWire program)
      0 3520 55 0 (-1))
    (retainedRule laterPositionMiddle (positionWire program)
      0 3520 55 1 1)
    (retainedRule laterPositionMiddle (positionWire program)
      1 3520 55 1 (-1))
    (retainedRule laterPositionFull (positionWire program)
      53 3520 55 0 1)
    (PiRLCRetainedGeometry.oneColumn geometry).val
    { major := source.val, middle := round.val, minor := slot.val }
    (some negativeOne) none none none none none
    rule0 rule1 rule2 rule3 rule4 rule5
  rw [initial_delta_zero _ source round slot first zero,
    ← negativeOne_eq geometry]
  simpa [positionRightProgram, addSelected, SparseForm.empty,
    SparseForm.add] using folded

private theorem positionRight_initial_one
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiRLCRetainedGeometry.Geometry program logicalWidth)
    (source : Fin sourceCount) (round : Fin roundCount)
    (slot : Fin positionSlotCount)
    (first : round.val = 0) (one : slot.val = 1) :
    (positionRightProgram program).form? logicalWidth
        (PiRLCRetainedGeometry.oneColumn geometry).val
        { major := source.val, middle := round.val, minor := slot.val } =
      some (PiRLCFirst54DirectPlan.positionDeltaForm
        (PiRLCRetainedInputs.first54Inputs geometry)
        (positionDescriptor source round slot)) := by
  let oneForm := PiRLCFirst54DirectPlan.oneForm
    (PiRLCRetainedInputs.first54Inputs geometry)
  have rule0 :
      (constantRule firstSlotZero (-1)).form? logicalWidth
        (PiRLCRetainedGeometry.oneColumn geometry).val
        { major := source.val, middle := round.val, minor := slot.val } =
      some none := by
    apply rule_form?_outside
    change ¬(0 ≤ source.val ∧ source.val < 17 ∧
      0 ≤ round.val ∧ round.val < 1 ∧
      0 ≤ slot.val ∧ slot.val < 1)
    omega
  have rule1 :
      (constantRule firstSlotOne 1).form? logicalWidth
        (PiRLCRetainedGeometry.oneColumn geometry).val
        { major := source.val, middle := round.val, minor := slot.val } =
      some (some oneForm) := by
    have selected := constantRule_form? firstSlotOne source
      (0 : Fin 1) (0 : Fin 1)
      (PiRLCRetainedGeometry.oneColumn geometry) (1 : F)
    simpa [firstSlotOne, region, first, one, oneForm,
      PiRLCFirst54DirectPlan.oneForm, PiRLCRetainedInputs.first54Inputs]
      using selected
  have rule2 :
      (retainedRule laterPositionZero (positionWire program)
        0 3520 55 0 (-1)).form? logicalWidth
        (PiRLCRetainedGeometry.oneColumn geometry).val
        { major := source.val, middle := round.val, minor := slot.val } =
      some none := by
    apply rule_form?_outside
    change ¬(0 ≤ source.val ∧ source.val < 17 ∧
      1 ≤ round.val ∧ round.val < 64 ∧
      0 ≤ slot.val ∧ slot.val < 1)
    omega
  have rule3 :
      (retainedRule laterPositionMiddle (positionWire program)
        0 3520 55 1 1).form? logicalWidth
        (PiRLCRetainedGeometry.oneColumn geometry).val
        { major := source.val, middle := round.val, minor := slot.val } =
      some none := by
    apply rule_form?_outside
    change ¬(0 ≤ source.val ∧ source.val < 17 ∧
      1 ≤ round.val ∧ round.val < 64 ∧
      1 ≤ slot.val ∧ slot.val < 54)
    omega
  have rule4 :
      (retainedRule laterPositionMiddle (positionWire program)
        1 3520 55 1 (-1)).form? logicalWidth
        (PiRLCRetainedGeometry.oneColumn geometry).val
        { major := source.val, middle := round.val, minor := slot.val } =
      some none := by
    apply rule_form?_outside
    change ¬(0 ≤ source.val ∧ source.val < 17 ∧
      1 ≤ round.val ∧ round.val < 64 ∧
      1 ≤ slot.val ∧ slot.val < 54)
    omega
  have rule5 :
      (retainedRule laterPositionFull (positionWire program)
        53 3520 55 0 1).form? logicalWidth
        (PiRLCRetainedGeometry.oneColumn geometry).val
        { major := source.val, middle := round.val, minor := slot.val } =
      some none := by
    apply rule_form?_outside
    change ¬(0 ≤ source.val ∧ source.val < 17 ∧
      1 ≤ round.val ∧ round.val < 64 ∧
      54 ≤ slot.val ∧ slot.val < 55)
    omega
  have folded := Program.six_form?_of_results
    (constantRule firstSlotZero (-1))
    (constantRule firstSlotOne 1)
    (retainedRule laterPositionZero (positionWire program)
      0 3520 55 0 (-1))
    (retainedRule laterPositionMiddle (positionWire program)
      0 3520 55 1 1)
    (retainedRule laterPositionMiddle (positionWire program)
      1 3520 55 1 (-1))
    (retainedRule laterPositionFull (positionWire program)
      53 3520 55 0 1)
    (PiRLCRetainedGeometry.oneColumn geometry).val
    { major := source.val, middle := round.val, minor := slot.val }
    none (some oneForm) none none none none
    rule0 rule1 rule2 rule3 rule4 rule5
  rw [initial_delta_one _ source round slot first one]
  simpa [positionRightProgram, addSelected, SparseForm.empty,
    SparseForm.add] using folded

private theorem positionRight_initial_other
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiRLCRetainedGeometry.Geometry program logicalWidth)
    (source : Fin sourceCount) (round : Fin roundCount)
    (slot : Fin positionSlotCount)
    (first : round.val = 0) (notZero : slot.val ≠ 0)
    (notOne : slot.val ≠ 1) :
    (positionRightProgram program).form? logicalWidth
        (PiRLCRetainedGeometry.oneColumn geometry).val
        { major := source.val, middle := round.val, minor := slot.val } =
      some (PiRLCFirst54DirectPlan.positionDeltaForm
        (PiRLCRetainedInputs.first54Inputs geometry)
        (positionDescriptor source round slot)) := by
  have rule0 :
      (constantRule firstSlotZero (-1)).form? logicalWidth
        (PiRLCRetainedGeometry.oneColumn geometry).val
        { major := source.val, middle := round.val, minor := slot.val } =
      some none := by
    apply rule_form?_outside
    change ¬(0 ≤ source.val ∧ source.val < 17 ∧
      0 ≤ round.val ∧ round.val < 1 ∧
      0 ≤ slot.val ∧ slot.val < 1)
    omega
  have rule1 :
      (constantRule firstSlotOne 1).form? logicalWidth
        (PiRLCRetainedGeometry.oneColumn geometry).val
        { major := source.val, middle := round.val, minor := slot.val } =
      some none := by
    apply rule_form?_outside
    change ¬(0 ≤ source.val ∧ source.val < 17 ∧
      0 ≤ round.val ∧ round.val < 1 ∧
      1 ≤ slot.val ∧ slot.val < 2)
    omega
  have rule2 :
      (retainedRule laterPositionZero (positionWire program)
        0 3520 55 0 (-1)).form? logicalWidth
        (PiRLCRetainedGeometry.oneColumn geometry).val
        { major := source.val, middle := round.val, minor := slot.val } =
      some none := by
    apply rule_form?_outside
    change ¬(0 ≤ source.val ∧ source.val < 17 ∧
      1 ≤ round.val ∧ round.val < 64 ∧
      0 ≤ slot.val ∧ slot.val < 1)
    omega
  have rule3 :
      (retainedRule laterPositionMiddle (positionWire program)
        0 3520 55 1 1).form? logicalWidth
        (PiRLCRetainedGeometry.oneColumn geometry).val
        { major := source.val, middle := round.val, minor := slot.val } =
      some none := by
    apply rule_form?_outside
    change ¬(0 ≤ source.val ∧ source.val < 17 ∧
      1 ≤ round.val ∧ round.val < 64 ∧
      1 ≤ slot.val ∧ slot.val < 54)
    omega
  have rule4 :
      (retainedRule laterPositionMiddle (positionWire program)
        1 3520 55 1 (-1)).form? logicalWidth
        (PiRLCRetainedGeometry.oneColumn geometry).val
        { major := source.val, middle := round.val, minor := slot.val } =
      some none := by
    apply rule_form?_outside
    change ¬(0 ≤ source.val ∧ source.val < 17 ∧
      1 ≤ round.val ∧ round.val < 64 ∧
      1 ≤ slot.val ∧ slot.val < 54)
    omega
  have rule5 :
      (retainedRule laterPositionFull (positionWire program)
        53 3520 55 0 1).form? logicalWidth
        (PiRLCRetainedGeometry.oneColumn geometry).val
        { major := source.val, middle := round.val, minor := slot.val } =
      some none := by
    apply rule_form?_outside
    change ¬(0 ≤ source.val ∧ source.val < 17 ∧
      1 ≤ round.val ∧ round.val < 64 ∧
      54 ≤ slot.val ∧ slot.val < 55)
    omega
  have folded := Program.six_form?_of_results
    (constantRule firstSlotZero (-1))
    (constantRule firstSlotOne 1)
    (retainedRule laterPositionZero (positionWire program)
      0 3520 55 0 (-1))
    (retainedRule laterPositionMiddle (positionWire program)
      0 3520 55 1 1)
    (retainedRule laterPositionMiddle (positionWire program)
      1 3520 55 1 (-1))
    (retainedRule laterPositionFull (positionWire program)
      53 3520 55 0 1)
    (PiRLCRetainedGeometry.oneColumn geometry).val
    { major := source.val, middle := round.val, minor := slot.val }
    none none none none none none rule0 rule1 rule2 rule3 rule4 rule5
  rw [initial_delta_other _ source round slot first notZero notOne]
  simpa [positionRightProgram, addSelected, SparseForm.empty,
    SparseForm.add] using folded

private theorem firstZeroRule_outside_later
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiRLCRetainedGeometry.Geometry program logicalWidth)
    (source : Fin sourceCount) (round : Fin roundCount)
    (slot : Fin positionSlotCount) (notFirst : round.val ≠ 0) :
    (constantRule firstSlotZero (-1)).form? logicalWidth
        (PiRLCRetainedGeometry.oneColumn geometry).val
        { major := source.val, middle := round.val, minor := slot.val } =
      some none := by
  apply rule_form?_outside
  change ¬(0 ≤ source.val ∧ source.val < 17 ∧
    0 ≤ round.val ∧ round.val < 1 ∧
    0 ≤ slot.val ∧ slot.val < 1)
  omega

private theorem firstOneRule_outside_later
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiRLCRetainedGeometry.Geometry program logicalWidth)
    (source : Fin sourceCount) (round : Fin roundCount)
    (slot : Fin positionSlotCount) (notFirst : round.val ≠ 0) :
    (constantRule firstSlotOne 1).form? logicalWidth
        (PiRLCRetainedGeometry.oneColumn geometry).val
        { major := source.val, middle := round.val, minor := slot.val } =
      some none := by
  apply rule_form?_outside
  change ¬(0 ≤ source.val ∧ source.val < 17 ∧
    0 ≤ round.val ∧ round.val < 1 ∧
    1 ≤ slot.val ∧ slot.val < 2)
  omega

private theorem positionRight_later_zero
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiRLCRetainedGeometry.Geometry program logicalWidth)
    (source : Fin sourceCount) (round : Fin roundCount)
    (slot : Fin positionSlotCount)
    (notFirst : round.val ≠ 0) (zero : slot.val = 0) :
    (positionRightProgram program).form? logicalWidth
        (PiRLCRetainedGeometry.oneColumn geometry).val
        { major := source.val, middle := round.val, minor := slot.val } =
      some (PiRLCFirst54DirectPlan.positionDeltaForm
        (PiRLCRetainedInputs.first54Inputs geometry)
        (positionDescriptor source round slot)) := by
  let inputs := PiRLCRetainedInputs.first54Inputs geometry
  let priorRound := previousRound round notFirst
  let middle : Fin 63 := laterRoundOffset round notFirst
  let prior := PiRLCFirst54DirectPlan.priorPositionForm inputs
    (candidate source round) slot
  have rule0 := firstZeroRule_outside_later geometry source round slot notFirst
  have rule1 := firstOneRule_outside_later geometry source round slot notFirst
  have inside : laterPositionZero.offsets?
      { major := source.val, middle := round.val, minor := slot.val } =
    some { major := source.val, middle := middle.val, minor := 0 } := by
    have selected := Region.offsets?_of_offsets laterPositionZero source middle
      (0 : Fin 1)
    have roundEq : 1 + middle.val = round.val := by
      dsimp only [middle, laterRoundOffset]
      omega
    change laterPositionZero.offsets? {
        major := 0 + source.val
        middle := 1 + middle.val
        minor := 0 + (0 : Fin 1).val } =
      some { major := source.val, middle := middle.val, minor := 0 }
      at selected
    simp only [Nat.zero_add, roundEq, zero] at selected ⊢
    exact selected
  have raw := positionWire_form? geometry source priorRound slot
  have rule2 :
      (retainedRule laterPositionZero (positionWire program)
        0 3520 55 0 (-1)).form? logicalWidth
        (PiRLCRetainedGeometry.oneColumn geometry).val
        { major := source.val, middle := round.val, minor := slot.val } =
      some (some (SparseForm.scale (-1) prior)) := by
    have selected := retainedRule_form?_of_loaded laterPositionZero
      { major := source.val, middle := round.val, minor := slot.val }
      { major := source.val, middle := middle.val, minor := 0 }
      (positionWire program) (PiRLCRetainedGeometry.oneColumn geometry).val
      0 3520 55 0 (-1) prior inside
      (by simpa [prior, PiRLCFirst54DirectPlan.priorPositionForm,
        candidate, notFirst, previousCandidate_eq, priorRound,
        positionDescriptor, inputs, zero] using raw)
    simpa [applyCoefficient, negOne_ne_one_position] using selected
  have rule3 :
      (retainedRule laterPositionMiddle (positionWire program)
        0 3520 55 1 1).form? logicalWidth
        (PiRLCRetainedGeometry.oneColumn geometry).val
        { major := source.val, middle := round.val, minor := slot.val } =
      some none := by
    apply rule_form?_outside
    change ¬(0 ≤ source.val ∧ source.val < 17 ∧
      1 ≤ round.val ∧ round.val < 64 ∧
      1 ≤ slot.val ∧ slot.val < 54)
    omega
  have rule4 :
      (retainedRule laterPositionMiddle (positionWire program)
        1 3520 55 1 (-1)).form? logicalWidth
        (PiRLCRetainedGeometry.oneColumn geometry).val
        { major := source.val, middle := round.val, minor := slot.val } =
      some none := by
    apply rule_form?_outside
    change ¬(0 ≤ source.val ∧ source.val < 17 ∧
      1 ≤ round.val ∧ round.val < 64 ∧
      1 ≤ slot.val ∧ slot.val < 54)
    omega
  have rule5 :
      (retainedRule laterPositionFull (positionWire program)
        53 3520 55 0 1).form? logicalWidth
        (PiRLCRetainedGeometry.oneColumn geometry).val
        { major := source.val, middle := round.val, minor := slot.val } =
      some none := by
    apply rule_form?_outside
    change ¬(0 ≤ source.val ∧ source.val < 17 ∧
      1 ≤ round.val ∧ round.val < 64 ∧
      54 ≤ slot.val ∧ slot.val < 55)
    omega
  have folded := Program.six_form?_of_results
    (constantRule firstSlotZero (-1))
    (constantRule firstSlotOne 1)
    (retainedRule laterPositionZero (positionWire program)
      0 3520 55 0 (-1))
    (retainedRule laterPositionMiddle (positionWire program)
      0 3520 55 1 1)
    (retainedRule laterPositionMiddle (positionWire program)
      1 3520 55 1 (-1))
    (retainedRule laterPositionFull (positionWire program)
      53 3520 55 0 1)
    (PiRLCRetainedGeometry.oneColumn geometry).val
    { major := source.val, middle := round.val, minor := slot.val }
    none none (some (SparseForm.scale (-1) prior)) none none none
    rule0 rule1 rule2 rule3 rule4 rule5
  rw [later_delta_zero inputs source round slot zero]
  simpa [positionRightProgram, addSelected, SparseForm.empty,
    SparseForm.add] using folded

private theorem positionRight_later_middle
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiRLCRetainedGeometry.Geometry program logicalWidth)
    (source : Fin sourceCount) (round : Fin roundCount)
    (slot : Fin positionSlotCount)
    (notFirst : round.val ≠ 0) (positive : 0 < slot.val)
    (belowFull : slot.val < First54Step.fullSlot) :
    (positionRightProgram program).form? logicalWidth
        (PiRLCRetainedGeometry.oneColumn geometry).val
        { major := source.val, middle := round.val, minor := slot.val } =
      some (PiRLCFirst54DirectPlan.positionDeltaForm
        (PiRLCRetainedInputs.first54Inputs geometry)
        (positionDescriptor source round slot)) := by
  let inputs := PiRLCRetainedInputs.first54Inputs geometry
  let priorRound := previousRound round notFirst
  let middle : Fin 63 := laterRoundOffset round notFirst
  let minor : Fin 53 := ⟨slot.val - 1, by
    change slot.val - 1 < 53
    simp only [First54Step.fullSlot] at belowFull
    omega⟩
  let previousSlot := First54Step.previousSlot slot positive
  let previous := PiRLCFirst54DirectPlan.previousPositionForm inputs
    (positionDescriptor source round slot)
  let prior := PiRLCFirst54DirectPlan.priorPositionForm inputs
    (candidate source round) slot
  have notZero : slot.val ≠ 0 := by omega
  have rule0 := firstZeroRule_outside_later geometry source round slot notFirst
  have rule1 := firstOneRule_outside_later geometry source round slot notFirst
  have rule2 :
      (retainedRule laterPositionZero (positionWire program)
        0 3520 55 0 (-1)).form? logicalWidth
        (PiRLCRetainedGeometry.oneColumn geometry).val
        { major := source.val, middle := round.val, minor := slot.val } =
      some none := by
    apply rule_form?_outside
    change ¬(0 ≤ source.val ∧ source.val < 17 ∧
      1 ≤ round.val ∧ round.val < 64 ∧
      0 ≤ slot.val ∧ slot.val < 1)
    omega
  have inside : laterPositionMiddle.offsets?
      { major := source.val, middle := round.val, minor := slot.val } =
    some { major := source.val, middle := middle.val, minor := minor.val } := by
    have selected := Region.offsets?_of_offsets laterPositionMiddle source
      middle minor
    have roundEq : 1 + middle.val = round.val := by
      dsimp only [middle, laterRoundOffset]
      omega
    have slotEq : 1 + minor.val = slot.val := by
      dsimp only [minor]
      omega
    change laterPositionMiddle.offsets? {
        major := 0 + source.val
        middle := 1 + middle.val
        minor := 1 + minor.val } =
      some { major := source.val, middle := middle.val, minor := minor.val }
      at selected
    simp only [Nat.zero_add, roundEq, slotEq] at selected ⊢
    exact selected
  have previousRaw := positionWire_form? geometry source priorRound previousSlot
  have rule3 :
      (retainedRule laterPositionMiddle (positionWire program)
        0 3520 55 1 1).form? logicalWidth
        (PiRLCRetainedGeometry.oneColumn geometry).val
        { major := source.val, middle := round.val, minor := slot.val } =
      some (some previous) := by
    have indexEq :
        0 + source.val * 3520 + middle.val * 55 + minor.val * 1 =
          source.val * 3520 + priorRound.val * 55 + previousSlot.val := by
      dsimp only [middle, laterRoundOffset, priorRound, previousRound,
        minor, previousSlot, First54Step.previousSlot]
      omega
    have loaded :
        (positionWire program).form? logicalWidth
            (0 + source.val * 3520 + middle.val * 55 + minor.val * 1) =
          some previous := by
      rw [indexEq, previousRaw]
      unfold previous PiRLCFirst54DirectPlan.previousPositionForm
      simp only [positionDescriptor]
      rw [dif_neg]
      · unfold PiRLCFirst54DirectPlan.priorPositionForm
        rw [dif_neg]
        · rw [previousCandidate_eq source round notFirst]
        · simpa [positionDescriptor, candidate] using notFirst
      · simpa [positionDescriptor, previousSlot,
          First54Step.previousSlot] using notZero
    have selected := retainedRule_form?_of_loaded laterPositionMiddle
      { major := source.val, middle := round.val, minor := slot.val }
      { major := source.val, middle := middle.val, minor := minor.val }
      (positionWire program) (PiRLCRetainedGeometry.oneColumn geometry).val
      0 3520 55 1 1 previous inside
      loaded
    simpa [applyCoefficient] using selected
  have priorRaw := positionWire_form? geometry source priorRound slot
  have rule4 :
      (retainedRule laterPositionMiddle (positionWire program)
        1 3520 55 1 (-1)).form? logicalWidth
        (PiRLCRetainedGeometry.oneColumn geometry).val
        { major := source.val, middle := round.val, minor := slot.val } =
      some (some (SparseForm.scale (-1) prior)) := by
    have indexEq :
        1 + source.val * 3520 + middle.val * 55 + minor.val * 1 =
          source.val * 3520 + priorRound.val * 55 + slot.val := by
      dsimp only [middle, laterRoundOffset, priorRound, previousRound, minor]
      omega
    have loaded :
        (positionWire program).form? logicalWidth
            (1 + source.val * 3520 + middle.val * 55 + minor.val * 1) =
          some prior := by
      rw [indexEq, priorRaw]
      unfold prior PiRLCFirst54DirectPlan.priorPositionForm
      rw [dif_neg]
      · rw [previousCandidate_eq source round notFirst]
        rfl
      · simpa [candidate] using notFirst
    have selected := retainedRule_form?_of_loaded laterPositionMiddle
      { major := source.val, middle := round.val, minor := slot.val }
      { major := source.val, middle := middle.val, minor := minor.val }
      (positionWire program) (PiRLCRetainedGeometry.oneColumn geometry).val
      1 3520 55 1 (-1) prior inside
      loaded
    simpa [applyCoefficient, negOne_ne_one_position] using selected
  have rule5 :
      (retainedRule laterPositionFull (positionWire program)
        53 3520 55 0 1).form? logicalWidth
        (PiRLCRetainedGeometry.oneColumn geometry).val
        { major := source.val, middle := round.val, minor := slot.val } =
      some none := by
    apply rule_form?_outside
    change ¬(0 ≤ source.val ∧ source.val < 17 ∧
      1 ≤ round.val ∧ round.val < 64 ∧
      54 ≤ slot.val ∧ slot.val < 55)
    simp only [First54Step.fullSlot] at belowFull
    omega
  have folded := Program.six_form?_of_results
    (constantRule firstSlotZero (-1))
    (constantRule firstSlotOne 1)
    (retainedRule laterPositionZero (positionWire program)
      0 3520 55 0 (-1))
    (retainedRule laterPositionMiddle (positionWire program)
      0 3520 55 1 1)
    (retainedRule laterPositionMiddle (positionWire program)
      1 3520 55 1 (-1))
    (retainedRule laterPositionFull (positionWire program)
      53 3520 55 0 1)
    (PiRLCRetainedGeometry.oneColumn geometry).val
    { major := source.val, middle := round.val, minor := slot.val }
    none none none (some previous) (some (SparseForm.scale (-1) prior)) none
    rule0 rule1 rule2 rule3 rule4 rule5
  have notFull : slot.val ≠ First54Step.fullSlot := by omega
  rw [later_delta_middle inputs source round slot (by omega) notFull]
  simpa [positionRightProgram, addSelected, SparseForm.empty,
    SparseForm.add] using folded

private theorem positionRight_later_full
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiRLCRetainedGeometry.Geometry program logicalWidth)
    (source : Fin sourceCount) (round : Fin roundCount)
    (slot : Fin positionSlotCount)
    (notFirst : round.val ≠ 0)
    (full : slot.val = First54Step.fullSlot) :
    (positionRightProgram program).form? logicalWidth
        (PiRLCRetainedGeometry.oneColumn geometry).val
        { major := source.val, middle := round.val, minor := slot.val } =
      some (PiRLCFirst54DirectPlan.positionDeltaForm
        (PiRLCRetainedInputs.first54Inputs geometry)
        (positionDescriptor source round slot)) := by
  let inputs := PiRLCRetainedInputs.first54Inputs geometry
  let priorRound := previousRound round notFirst
  let middle : Fin 63 := laterRoundOffset round notFirst
  let priorSlot : Fin positionSlotCount := ⟨53, by
    simp [positionSlotCount]⟩
  let previous := PiRLCFirst54DirectPlan.previousPositionForm inputs
    (positionDescriptor source round slot)
  have notZero : slot.val ≠ 0 := by
    simp only [First54Step.fullSlot] at full
    omega
  have rule0 := firstZeroRule_outside_later geometry source round slot notFirst
  have rule1 := firstOneRule_outside_later geometry source round slot notFirst
  have rule2 :
      (retainedRule laterPositionZero (positionWire program)
        0 3520 55 0 (-1)).form? logicalWidth
        (PiRLCRetainedGeometry.oneColumn geometry).val
        { major := source.val, middle := round.val, minor := slot.val } =
      some none := by
    apply rule_form?_outside
    change ¬(0 ≤ source.val ∧ source.val < 17 ∧
      1 ≤ round.val ∧ round.val < 64 ∧
      0 ≤ slot.val ∧ slot.val < 1)
    omega
  have rule3 :
      (retainedRule laterPositionMiddle (positionWire program)
        0 3520 55 1 1).form? logicalWidth
        (PiRLCRetainedGeometry.oneColumn geometry).val
        { major := source.val, middle := round.val, minor := slot.val } =
      some none := by
    apply rule_form?_outside
    change ¬(0 ≤ source.val ∧ source.val < 17 ∧
      1 ≤ round.val ∧ round.val < 64 ∧
      1 ≤ slot.val ∧ slot.val < 54)
    simp only [First54Step.fullSlot] at full
    omega
  have rule4 :
      (retainedRule laterPositionMiddle (positionWire program)
        1 3520 55 1 (-1)).form? logicalWidth
        (PiRLCRetainedGeometry.oneColumn geometry).val
        { major := source.val, middle := round.val, minor := slot.val } =
      some none := by
    apply rule_form?_outside
    change ¬(0 ≤ source.val ∧ source.val < 17 ∧
      1 ≤ round.val ∧ round.val < 64 ∧
      1 ≤ slot.val ∧ slot.val < 54)
    simp only [First54Step.fullSlot] at full
    omega
  have inside : laterPositionFull.offsets?
      { major := source.val, middle := round.val, minor := slot.val } =
    some { major := source.val, middle := middle.val, minor := 0 } := by
    have selected := Region.offsets?_of_offsets laterPositionFull source middle
      (0 : Fin 1)
    have roundEq : 1 + middle.val = round.val := by
      dsimp only [middle, laterRoundOffset]
      omega
    change laterPositionFull.offsets? {
        major := 0 + source.val
        middle := 1 + middle.val
        minor := 54 + (0 : Fin 1).val } =
      some { major := source.val, middle := middle.val, minor := 0 }
      at selected
    simp only [Nat.zero_add, roundEq, First54Step.fullSlot] at full
    simp only [Nat.zero_add, roundEq, full] at selected ⊢
    exact selected
  have raw := positionWire_form? geometry source priorRound priorSlot
  have rule5 :
      (retainedRule laterPositionFull (positionWire program)
        53 3520 55 0 1).form? logicalWidth
        (PiRLCRetainedGeometry.oneColumn geometry).val
        { major := source.val, middle := round.val, minor := slot.val } =
      some (some previous) := by
    have indexEq :
        53 + source.val * 3520 + middle.val * 55 + 0 * 0 =
          source.val * 3520 + priorRound.val * 55 + priorSlot.val := by
      dsimp only [middle, laterRoundOffset, priorRound, previousRound,
        priorSlot]
      omega
    have loaded :
        (positionWire program).form? logicalWidth
            (53 + source.val * 3520 + middle.val * 55 + 0 * 0) =
          some previous := by
      rw [indexEq, raw]
      unfold previous PiRLCFirst54DirectPlan.previousPositionForm
      simp only [positionDescriptor]
      rw [dif_neg]
      · unfold PiRLCFirst54DirectPlan.priorPositionForm
        rw [dif_neg]
        · rw [previousCandidate_eq source round notFirst]
          congr 3
          apply Fin.ext
          simp [priorSlot, First54Step.previousSlot,
            First54Step.fullSlot, full]
        · simpa [candidate] using notFirst
      · exact notZero
    have selected := retainedRule_form?_of_loaded laterPositionFull
      { major := source.val, middle := round.val, minor := slot.val }
      { major := source.val, middle := middle.val, minor := 0 }
      (positionWire program) (PiRLCRetainedGeometry.oneColumn geometry).val
      53 3520 55 0 1 previous inside loaded
    simpa [applyCoefficient] using selected
  have folded := Program.six_form?_of_results
    (constantRule firstSlotZero (-1))
    (constantRule firstSlotOne 1)
    (retainedRule laterPositionZero (positionWire program)
      0 3520 55 0 (-1))
    (retainedRule laterPositionMiddle (positionWire program)
      0 3520 55 1 1)
    (retainedRule laterPositionMiddle (positionWire program)
      1 3520 55 1 (-1))
    (retainedRule laterPositionFull (positionWire program)
      53 3520 55 0 1)
    (PiRLCRetainedGeometry.oneColumn geometry).val
    { major := source.val, middle := round.val, minor := slot.val }
    none none none none none (some previous)
    rule0 rule1 rule2 rule3 rule4 rule5
  rw [later_delta_full inputs source round slot full]
  simpa [positionRightProgram, addSelected, SparseForm.empty,
    SparseForm.add] using folded

theorem positionRightProgram_form?
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiRLCRetainedGeometry.Geometry program logicalWidth)
    (source : Fin sourceCount) (round : Fin roundCount)
    (slot : Fin positionSlotCount) :
    (positionRightProgram program).form? logicalWidth
        (PiRLCRetainedGeometry.oneColumn geometry).val
        { major := source.val, middle := round.val, minor := slot.val } =
      some (PiRLCFirst54DirectPlan.positionDeltaForm
        (PiRLCRetainedInputs.first54Inputs geometry)
        (positionDescriptor source round slot)) := by
  by_cases first : round.val = 0
  · by_cases zero : slot.val = 0
    · exact positionRight_initial_zero geometry source round slot first zero
    · by_cases one : slot.val = 1
      · exact positionRight_initial_one geometry source round slot first one
      · exact positionRight_initial_other geometry source round slot first
          zero one
  · by_cases zero : slot.val = 0
    · exact positionRight_later_zero geometry source round slot first zero
    · by_cases full : slot.val = First54Step.fullSlot
      · exact positionRight_later_full geometry source round slot first full
      · have positive : 0 < slot.val := by omega
        have belowFull : slot.val < First54Step.fullSlot := by
          have bound := slot.isLt
          simp only [positionSlotCount] at bound
          simp only [First54Step.fullSlot] at full ⊢
          omega
        exact positionRight_later_middle geometry source round slot first
          positive belowFull

theorem positionGrid_row?
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiRLCRetainedGeometry.Geometry program logicalWidth)
    (source : Fin sourceCount) (round : Fin roundCount)
    (slot : Fin positionSlotCount) :
    (positionGrid geometry).row? logicalWidth
        (Fin.encodeProd (source, Fin.encodeProd (round, slot))).val =
      some (MultiplicationFamilyPlan.forms
        (PiRLCFirst54DirectPlan.positionInterface
          (PiRLCRetainedInputs.first54Inputs geometry))
        (PiRLCFirst54DirectSchedule.positionIndex
          (positionDescriptor source round slot))) := by
  let inputs := PiRLCRetainedInputs.first54Inputs geometry
  let left := PiRLCFirst54DirectPlan.acceptedForm inputs
    (candidate source round)
  let right := PiRLCFirst54DirectPlan.positionDeltaForm inputs
    (positionDescriptor source round slot)
  let output := PiRLCFirst54DirectPlan.positionDifferenceForm inputs
    (positionDescriptor source round slot)
  have direct := MultiplicationGrid.Block.row?_of_results
    (positionGrid geometry) (PiRLCRetainedGeometry.oneColumn geometry) rfl
    source round slot left right output
    (by simpa [left, inputs] using
      positionLeftProgram_form? geometry source round slot)
    (by simpa [right, inputs] using
      positionRightProgram_form? geometry source round slot)
    (by simpa [output, inputs] using
      positionOutputProgram_form? geometry source round slot)
  simpa [MultiplicationFamilyPlan.forms,
    PiRLCFirst54DirectPlan.positionInterface,
    PiRLCFirst54DirectSchedule.position_positionIndex,
    left, right, output, inputs] using direct

end NightstreamFPrime.Export.Stage1.PiRLCFirst54MatrixProgram
