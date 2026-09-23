import NightstreamFPrime.Layout.PiRlcWideSampler.StateSemantics
import NightstreamFPrime.Layout.ProductionRelation.PoseidonCompactWitness

/-! Constructive source witnesses for the candidate CCS sampler. Poseidon
uses its existing proved executor. The range source uses the same canonical
children and arithmetic completion whose hinted execution is proved by
WideReduction.Program. Committed coordinates contain checked values only. -/

namespace NightstreamFPrime.Layout.PiRlcWideSampler.Witness

open NightstreamFPrime.Spec NightstreamFPrime.Circuit
open NightstreamFPrime.Gadgets.Sampling
open ProductionRelation BatchPlan
open Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

abbrev FieldState := Fin 8 → F

def domainInput (before : FieldState) (step : Nat) : FieldState :=
  fun lane => if step % 2 = 0 then before lane + entryWord (step / 2) lane else before lane

def stateAt (initial : FieldState) : Nat → FieldState
  | 0 => initial
  | step + 1 => PoseidonCompactWitness.output (domainInput (stateAt initial step) step)

def inputAt (initial : FieldState) (step : Nat) : FieldState := domainInput (stateAt initial step) step

def sboxValues (initial : FieldState) (slot : Fin (34 * 86)) : F :=
  PoseidonCompactWitness.retained (inputAt initial (slot.val / 86))
    ⟨slot.val % 86, by rw [PoseidonRetainedSlots.rows_length]; omega⟩

def drawAt (initial : FieldState) (source : Fin 17) : Fin 4 → F :=
  fun lane => stateAt initial (source.val * 2 + 1) ⟨lane.val, by omega⟩

def inputEnv (draw : Fin 4 → F) : Env :=
  fun column => if below : column < 4 then draw ⟨column, below⟩ else 0

/-- Reconstruct unused helper columns as zero after executing the exact
proved hint program. Row acceptance is invariant under this replacement. -/
def rangeValues (draw : Fin 4 → F) : Env :=
  WideReduction.replaceTemporary (WideReduction.Program.completeEnv RangePlan.interface (inputEnv draw) 4)
    4 WideReduction.HintProgram.helperCount (fun _ => 0)

theorem rangeValues_input (draw : Fin 4 → F) (lane : Fin 4) :
    rangeValues draw lane.val = draw lane := by
  have agreement := (WideReduction.Program.completeEnv_correct RangePlan.interface (inputEnv draw) 4
    (fun index => index.isLt)).1
  unfold rangeValues WideReduction.replaceTemporary
  rw [if_neg (by omega), agreement lane.val (Or.inl lane.isLt)]
  simp only [inputEnv, dif_pos lane.isLt]

theorem rangeValues_helper (draw : Fin 4 → F) (column : Nat) (lower : 4 ≤ column) (upper : column < 1408) :
    rangeValues draw column = 0 := by
  unfold rangeValues WideReduction.replaceTemporary
  rw [if_pos (by change 4 ≤ column ∧ column < 4 + 1404; omega)]

theorem rangeValues_rows (draw : Fin 4 → F) : ConstraintsHold (rangeValues draw) RangePlan.constraints := by
  exact (WideReduction.Program.temporary_read_exclusion RangePlan.interface 4 (fun lane => lane.isLt)
    (WideReduction.Program.completeEnv RangePlan.interface (inputEnv draw) 4) (fun _ => 0)).mpr
      (WideReduction.Program.completeEnv_correct RangePlan.interface (inputEnv draw) 4 (fun lane => lane.isLt)).2

def rangeSources (initial : FieldState) (source : Fin 17) : Env := rangeValues (drawAt initial source)

def rangeCoordinate (env : Env) (position : Fin 937) : F :=
  if bit : position.val < 256 then
    LowNormSlot.coordinate .bit (env (Retained.canonicalBits.source ⟨position.val, bit⟩).val) ⟨0, by decide⟩
  else if field : position.val < 584 then
    LowNormSlot.coordinate .field
      (env (Retained.canonicalFields.source ⟨(position.val - 256) / 41, by change _ < 8; omega⟩).val)
      ⟨(position.val - 256) % 41, Nat.mod_lt _ (by decide)⟩
  else
    LowNormSlot.coordinate .bit
      (env (Retained.resultBits.source ⟨position.val - 584, by change _ < 353; omega⟩).val) ⟨0, by decide⟩

/-- Decode an owned coordinate by block; no helper values have a case. -/
def coordinate (initial : FieldState) (index : Fin 135813) : F :=
  if poseidon : index.val < 119884 then
    LowNormSlot.coordinate .field (sboxValues initial ⟨index.val / 41, by omega⟩)
      ⟨index.val % 41, Nat.mod_lt _ (by decide)⟩
  else
    rangeCoordinate (rangeSources initial ⟨(index.val - 119884) / 937, by omega⟩)
      ⟨(index.val - 119884) % 937, Nat.mod_lt _ (by decide)⟩

def assignment {columns : Nat} (interface : Interface columns) (base : Assignment F columns)
    (initial : FieldState) : Assignment F columns :=
  fun column => if owned : interface.start ≤ column.val ∧ column.val < interface.start + 135813 then
    coordinate initial ⟨column.val - interface.start, by omega⟩
  else base column

theorem assignment_outside {columns : Nat} (interface : Interface columns) (base : Assignment F columns)
    (initial : FieldState) (column : Fin columns)
    (outside : column.val < interface.start ∨ interface.start + 135813 ≤ column.val) :
    assignment interface base initial column = base column := by
  rw [assignment, dif_neg (by omega)]

theorem assignment_at {columns : Nat} (interface : Interface columns) (base : Assignment F columns)
    (initial : FieldState) (index : Fin 135813) :
    assignment interface base initial
        ⟨interface.start + index.val, by have fits := interface.fits; rw [BatchPlan.coordinateCount_eq] at fits; omega⟩ =
      coordinate initial index := by
  unfold assignment
  rw [dif_pos (by simp only; constructor <;> omega)]
  congr 1
  apply Fin.ext
  simp only [Nat.add_sub_cancel_left]

end NightstreamFPrime.Layout.PiRlcWideSampler.Witness
