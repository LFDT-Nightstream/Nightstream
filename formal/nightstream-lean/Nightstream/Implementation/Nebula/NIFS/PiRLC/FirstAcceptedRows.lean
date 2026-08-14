import Nightstream.Implementation.R1CS.Canonical.LinCombNormal
import Nightstream.Implementation.Lowering.Typed.Cost

/-!
Contract: exact fail-closed first-accepted selector for one V2 PiRLC
coefficient.

The caller supplies three already-constrained accept bits and three
modulo-five residues. The occurrence selects the first accepted residue and
requires one of the three attempts to be accepted. Thus three rejections make
the row program unsatisfiable.

This module owns construction, placement, and exact cost. It does not own the
candidate classifiers, semantic proof, or honest witness.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Nebula.ProductPiRlcFirstAcceptedRows

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal

def attemptCount : Nat := 3
def auxiliaryCount : Nat := 8

structure Layout where
  base : Nat
  accept : Fin attemptCount -> Nat
  residue : Fin attemptCount -> Nat

def first : Fin attemptCount := ⟨0, by decide⟩
def second : Fin attemptCount := ⟨1, by decide⟩
def third : Fin attemptCount := ⟨2, by decide⟩

def selectFirstColumn (layout : Layout) : Nat := layout.base
def selectSecondColumn (layout : Layout) : Nat := layout.base + 1
def rejectFirstTwoColumn (layout : Layout) : Nat := layout.base + 2
def selectThirdColumn (layout : Layout) : Nat := layout.base + 3
def productColumn (layout : Layout) (attempt : Nat) : Nat :=
  layout.base + 4 + attempt
def outputColumn (layout : Layout) : Nat := layout.base + 7

def allocation (layout : Layout) : List Nat :=
  (List.range auxiliaryCount).map fun offset => layout.base + offset

theorem allocation_length (layout : Layout) :
    (allocation layout).length = auxiliaryCount := by
  simp [allocation]

theorem allocation_nodup (layout : Layout) :
    (allocation layout).Nodup := by
  unfold allocation
  exact nodup_map _ _ (fun _ _ equal => by omega)
    List.nodup_range

theorem allocation_mem_iff (layout : Layout) (column : Nat) :
    column ∈ allocation layout ↔
      layout.base ≤ column ∧ column < layout.base + auxiliaryCount := by
  unfold allocation
  constructor
  · intro member
    rcases List.mem_map.mp member with ⟨offset, inRange, rfl⟩
    have offsetLt := List.mem_range.mp inRange
    omega
  · rintro ⟨lower, upper⟩
    exact List.mem_map.mpr
      ⟨column - layout.base, List.mem_range.mpr (by omega), by omega⟩

structure InputsBelowBase (layout : Layout) : Prop where
  accept : forall attempt, layout.accept attempt < layout.base
  residue : forall attempt, layout.residue attempt < layout.base

def oneMinus (column : Nat) : LinComb :=
  [(column, goldilocksP - 1), (0, 1)]

def selectFirstRow (layout : Layout) : Row :=
  ⟨[(selectFirstColumn layout, 1)], [(0, 1)],
    [(layout.accept first, 1)]⟩

def selectSecondRow (layout : Layout) : Row :=
  ⟨oneMinus (layout.accept first), [(layout.accept second, 1)],
    [(selectSecondColumn layout, 1)]⟩

def rejectFirstTwoRow (layout : Layout) : Row :=
  ⟨oneMinus (layout.accept first), oneMinus (layout.accept second),
    [(rejectFirstTwoColumn layout, 1)]⟩

def selectThirdRow (layout : Layout) : Row :=
  ⟨[(rejectFirstTwoColumn layout, 1)], [(layout.accept third, 1)],
    [(selectThirdColumn layout, 1)]⟩

/-- Exact fail-closed equation: one and only one selection equals one. -/
def successRow (layout : Layout) : Row :=
  ⟨[(selectFirstColumn layout, 1), (selectSecondColumn layout, 1),
      (selectThirdColumn layout, 1)], [(0, 1)], [(0, 1)]⟩

def selectedColumns (layout : Layout) (attempt : Fin attemptCount) : Nat :=
  if attempt.val = 0 then selectFirstColumn layout
  else if attempt.val = 1 then selectSecondColumn layout
  else selectThirdColumn layout

def productRow (layout : Layout) (attempt : Fin attemptCount) : Row :=
  ⟨[(selectedColumns layout attempt, 1)],
    [(layout.residue attempt, 1)],
    [(productColumn layout attempt.val, 1)]⟩

def productRows (layout : Layout) : List Row :=
  (List.finRange attemptCount).map (productRow layout)

def outputRow (layout : Layout) : Row :=
  ⟨[(outputColumn layout, 1)], [(0, 1)],
    [(productColumn layout 0, 1), (productColumn layout 1, 1),
      (productColumn layout 2, 1)]⟩

def rows (layout : Layout) : List Row :=
  [ selectFirstRow layout,
    selectSecondRow layout,
    rejectFirstTwoRow layout,
    selectThirdRow layout,
    successRow layout ] ++
    productRows layout ++ [outputRow layout]

theorem productRows_length (layout : Layout) :
    (productRows layout).length = 3 := by
  simp [productRows, attemptCount]

theorem rows_length (layout : Layout) :
    (rows layout).length = 9 := by
  simp [rows, productRows, attemptCount]

def cost : Nightstream.Implementation.Lowering.Typed.Cost where
  recurringRows := 9
  committedColumns := 0
  publicColumns := 0
  auxiliaryColumns := auxiliaryCount

theorem cost_rows (layout : Layout) :
    (rows layout).length = cost.recurringRows := rows_length layout

theorem cost_columns (layout : Layout) :
    (allocation layout).length = cost.auxiliaryColumns :=
  allocation_length layout

end Nightstream.Implementation.Nebula.ProductPiRlcFirstAcceptedRows
