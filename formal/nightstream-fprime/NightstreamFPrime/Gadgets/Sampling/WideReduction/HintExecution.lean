import NightstreamFPrime.Circuit.StraightLine

/-! Materialized execution of existing hints. The array stores each value
once; the refinement theorem connects it to the unchanged DSL semantics. -/

namespace NightstreamFPrime.Gadgets.Sampling.WideReduction.HintExecution

open NightstreamFPrime.Spec NightstreamFPrime.Circuit

def read (base : Env) (start : Nat) (values : Array F) (index : Nat) : F :=
  if inside : start ≤ index ∧ index < start + values.size then
    values[index - start]'(by omega)
  else base index

theorem read_empty (base : Env) (start : Nat) : read base start #[] = base := by
  funext index
  simp [read]

theorem read_push (base : Env) (start : Nat) (values : Array F) (value : F) :
    read base start (values.push value) = Env.set (read base start values) (start + values.size) value := by
  funext index
  by_cases last : index = start + values.size
  · subst index
    simp [read, Env.set]
  · by_cases inside : start ≤ index ∧ index < start + values.size
    · have after : start ≤ index ∧ index < start + (values.push value).size := by
        simp only [Array.size_push]
        omega
      have before : index - start < values.size := by omega
      simp only [read, dif_pos inside, dif_pos after, Env.set, if_neg last]
      simp [Array.getElem_push, before]
    · have outside : ¬ (start ≤ index ∧ index < start + (values.push value).size) := by
        simp only [Array.size_push]
        omega
      simp only [read, dif_neg inside, dif_neg outside, Env.set, if_neg last]

def run (base : Env) (start : Nat) : Array F → List Hint → Array F
  | values, [] => values
  | values, hint :: rest =>
      run base start (values.push (hint.eval (read base start values))) rest

theorem run_correct (base : Env) (start : Nat) (values : Array F) (hints : List Hint) :
    read base start (run base start values hints) =
      executeHints (read base start values) (start + values.size) hints := by
  induction hints generalizing values with
  | nil => rfl
  | cons hint rest ih =>
      rw [run, ih, read_push, Array.size_push, executeHints]
      congr 1

theorem run_empty_correct (base : Env) (start : Nat) (hints : List Hint) :
    read base start (run base start #[] hints) = executeHints base start hints := by
  simp only [run_correct, read_empty, Array.size_empty, Nat.add_zero]

end NightstreamFPrime.Gadgets.Sampling.WideReduction.HintExecution
