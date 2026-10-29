import NightstreamFPrime.Layout.LowNormAssignment
import NightstreamFPrime.Lifecycle.XOut

namespace NightstreamFPrime.Layout.ProductionAssignment

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Phi81Relation
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra

def publicWidth : Nat := ringDegree * publicRingColumns

@[simp] theorem publicWidth_eq : publicWidth = 270 := by
  rfl

def logicalWidth {sourceWidth : Nat}
    (slots : List (LowNormAssignment.Slot sourceWidth)) : Nat :=
  publicWidth + LowNormAssignment.logicalWidth slots

theorem publicFits {sourceWidth : Nat}
    (slots : List (LowNormAssignment.Slot sourceWidth)) :
    ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth (logicalWidth slots) := by
  change publicWidth ≤ Phi81CarrierLayout.carrierWidth (logicalWidth slots)
  apply Nat.le_trans (show publicWidth ≤ logicalWidth slots by
    simp [logicalWidth])
  exact Phi81CarrierLayout.logicalWidth_le_carrierWidth _

private def privateCoordinate {sourceWidth : Nat}
    (slots : List (LowNormAssignment.Slot sourceWidth))
    (source : Fin sourceWidth → F)
    (index : Fin (LowNormAssignment.logicalWidth slots)) : F :=
  LowNormAssignment.coordinateAt source slots index.val

def logicalAssignment {sourceWidth : Nat}
    (publicInput : Fin publicWidth → F)
    (slots : List (LowNormAssignment.Slot sourceWidth))
    (source : Fin sourceWidth → F) :
    Fin (logicalWidth slots) → F :=
  fun column =>
    if publicRegion : column.val < publicWidth then
      publicInput ⟨column.val, publicRegion⟩
    else
      privateCoordinate slots source
        ⟨column.val - publicWidth, by
          have columnBound := column.isLt
          simp only [logicalWidth] at columnBound
          omega⟩

def publicColumn {sourceWidth : Nat}
    {slots : List (LowNormAssignment.Slot sourceWidth)}
    (column : Fin publicWidth) : Fin (logicalWidth slots) :=
  ⟨column.val, by
    simp only [logicalWidth]
    omega⟩

def privateColumn {sourceWidth : Nat}
    {slots : List (LowNormAssignment.Slot sourceWidth)}
    (column : Fin (LowNormAssignment.logicalWidth slots)) :
    Fin (logicalWidth slots) :=
  ⟨publicWidth + column.val, by
    simp only [logicalWidth]
    omega⟩

@[simp] theorem logicalAssignment_publicColumn {sourceWidth : Nat}
    (publicInput : Fin publicWidth → F)
    (slots : List (LowNormAssignment.Slot sourceWidth))
    (source : Fin sourceWidth → F) (column : Fin publicWidth) :
    logicalAssignment publicInput slots source (publicColumn column) =
      publicInput column := by
  unfold logicalAssignment publicColumn
  rw [dite_eq_left column.isLt]

@[simp] theorem logicalAssignment_privateColumn {sourceWidth : Nat}
    (publicInput : Fin publicWidth → F)
    (slots : List (LowNormAssignment.Slot sourceWidth))
    (source : Fin sourceWidth → F)
    (column : Fin (LowNormAssignment.logicalWidth slots)) :
    logicalAssignment publicInput slots source (privateColumn column) =
      privateCoordinate slots source column := by
  unfold logicalAssignment privateColumn
  rw [dite_eq_right (by
    change ¬(publicWidth + column.val < publicWidth)
    omega)]
  apply congrArg (privateCoordinate slots source)
  apply Fin.ext
  simp

theorem logicalAssignment_slotCoordinate {sourceWidth : Nat}
    (publicInput : Fin publicWidth → F)
    (slots : List (LowNormAssignment.Slot sourceWidth))
    (source : Fin sourceWidth → F) (slot : Fin slots.length)
    (coordinate : Fin (slots.get slot).width) :
    logicalAssignment publicInput slots source
        (privateColumn (LowNormAssignment.coordinateIndex slots slot coordinate)) =
      ((slots.get slot).encode source).getD coordinate.val 0 := by
  rw [logicalAssignment_privateColumn]
  unfold privateCoordinate
  exact LowNormAssignment.coordinateAt_coordinateIndex
    slots source slot coordinate

def completeAssignment {sourceWidth : Nat}
    (publicInput : Fin publicWidth → F)
    (slots : List (LowNormAssignment.Slot sourceWidth))
    (source : Fin sourceWidth → F) :
    PaperAlgebra.Assignment
      (logicalWidth := logicalWidth slots) (publicFits := publicFits slots) :=
  fun column =>
    if logicalRegion : column.val < logicalWidth slots then
      logicalAssignment publicInput slots source ⟨column.val, logicalRegion⟩
    else
      0

theorem projectPublicInput_completeAssignment {sourceWidth : Nat}
    (publicInput : Fin publicWidth → F)
    (slots : List (LowNormAssignment.Slot sourceWidth))
    (source : Fin sourceWidth → F) :
    Phi81Relation.projectPublicInput
        (completeAssignment publicInput slots source) = publicInput := by
  funext column
  unfold Phi81Relation.projectPublicInput completeAssignment
    Phi81Relation.Shape.publicColumn
  have logicalRegion : column.val < logicalWidth slots := by
    have columnBound : column.val < publicWidth := by
      have rawBound := column.isLt
      change column.val < publicWidth at rawBound
      exact rawBound
    simp only [logicalWidth]
    omega
  rw [dite_eq_left logicalRegion]
  exact logicalAssignment_publicColumn publicInput slots source column

theorem logicalAssignment_norm {sourceWidth : Nat}
    (publicInput : Fin publicWidth → F)
    (slots : List (LowNormAssignment.Slot sourceWidth))
    (source : Fin sourceWidth → F)
    (publicNorm : ∀ column, centeredMagnitude (publicInput column) < 2)
    (privateValid : ∀ slot ∈ slots, slot.Valid source)
    (column : Fin (logicalWidth slots)) :
    centeredMagnitude (logicalAssignment publicInput slots source column) < 2 := by
  unfold logicalAssignment
  split
  next publicRegion => exact publicNorm _
  next privateRegion =>
    apply LowNormAssignment.coordinates_norm slots source privateValid
    unfold privateCoordinate
    rw [LowNormAssignment.coordinateAt_eq_getD]
    have privateBound :
        column.val - publicWidth <
          (LowNormAssignment.coordinates slots source).length := by
      rw [LowNormAssignment.coordinates_length]
      have columnBound := column.isLt
      simp only [logicalWidth] at columnBound
      omega
    rw [List.getD_eq_getElem
      (l := LowNormAssignment.coordinates slots source) (d := 0) privateBound]
    exact List.getElem_mem privateBound

theorem completeAssignment_norm {sourceWidth : Nat}
    (publicInput : Fin publicWidth → F)
    (slots : List (LowNormAssignment.Slot sourceWidth))
    (source : Fin sourceWidth → F)
    (publicNorm : ∀ column, centeredMagnitude (publicInput column) < 2)
    (privateValid : ∀ slot ∈ slots, slot.Valid source)
    (column : Fin
      (Phi81CarrierLayout.carrierWidth (logicalWidth slots))) :
    centeredMagnitude (completeAssignment publicInput slots source column) < 2 := by
  unfold completeAssignment
  split
  next logicalRegion =>
    exact logicalAssignment_norm publicInput slots source publicNorm privateValid _
  next paddingRegion => decide

theorem encHash_completeAssignment_norm {sourceWidth : Nat}
    (digest : Digest)
    (slots : List (LowNormAssignment.Slot sourceWidth))
    (source : Fin sourceWidth → F)
    (privateValid : ∀ slot ∈ slots, slot.Valid source)
    (column : Fin
      (Phi81CarrierLayout.carrierWidth (logicalWidth slots))) :
    centeredMagnitude
      (completeAssignment
        (encHash (publicFits := publicFits slots) digest) slots source column) < 2 :=
  completeAssignment_norm _ slots source
    (encHash_norm (publicFits := publicFits slots) digest) privateValid column

end NightstreamFPrime.Layout.ProductionAssignment
