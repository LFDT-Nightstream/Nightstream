import Init.Data.Array.Lemmas
import Init.Data.Nat.Div.Lemmas
import Lean.Elab.Tactic.Omega

/-!
Immutable strided Array.all checks. Worker w visits w, w + workers, and so on.
Only worker indices are materialized. The caller owns task scheduling and the
positive runtime worker count; this module proves the pure result equality.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.StridedArrayAll

universe u

def fuel (size workers : Nat) : Nat :=
  (size + workers - 1) / workers

/-- Check one stride in tail position. Bounds are checked before each read,
and the first failed predicate stops that worker. -/
@[specialize] def run {Alpha : Type u} (values : Array Alpha)
    (predicate : Alpha → Bool) (stride : Nat) : Nat → Nat → Bool
  | 0, _ => true
  | remaining + 1, index =>
      if bounded : index < values.size then
        if predicate values[index] then
          run values predicate stride remaining (index + stride)
        else false
      else true

/-- Exact bounded-offset meaning of the structural helper. -/
theorem run_iff {Alpha : Type u} (values : Array Alpha)
    (predicate : Alpha → Bool) (stride remaining first : Nat) :
    run values predicate stride remaining first = true ↔
      ∀ offset, offset < remaining →
        ∀ live : first + stride * offset < values.size,
          predicate values[first + stride * offset] = true := by
  induction remaining generalizing first with
  | zero =>
      simp [run]
  | succ remaining inductionHypothesis =>
      by_cases live : first < values.size
      · have step :
          run values predicate stride (remaining + 1) first = true ↔
            predicate values[first] = true ∧
              run values predicate stride remaining (first + stride) = true := by
          simp only [run, dif_pos live]
          cases checked : predicate values[first] <;> simp
        rw [step, inductionHypothesis]
        constructor
        · rintro ⟨head, tail⟩ offset offsetBound indexLive
          cases offset with
          | zero => simpa only [Nat.mul_zero, Nat.add_zero] using head
          | succ offset =>
              have shift : first + stride * (offset + 1) =
                  first + stride + stride * offset := by
                rw [Nat.mul_succ]
                omega
              have tailLive : first + stride + stride * offset < values.size := by
                rw [← shift]
                exact indexLive
              simpa only [← shift] using
                tail offset (by omega) tailLive
        · intro checked
          constructor
          · simpa only [Nat.mul_zero, Nat.add_zero] using
              checked 0 (by omega) (by simpa only [Nat.mul_zero, Nat.add_zero] using live)
          · intro offset offsetBound tailLive
            have shift : first + stride * (offset + 1) =
                first + stride + stride * offset := by
              rw [Nat.mul_succ]
              omega
            have indexLive : first + stride * (offset + 1) < values.size := by
              rw [shift]
              exact tailLive
            simpa only [shift] using checked (offset + 1) (by omega) indexLive
      · simp only [run, dif_neg live]
        constructor
        · intro _checked offset _offsetBound indexLive
          omega
        · intro _checked
          trivial

/-- One worker receives ceiling(size / workers) fuel. -/
@[specialize] def worker {Alpha : Type u} (values : Array Alpha)
    (predicate : Alpha → Bool) (workers index : Nat) : Bool :=
  run values predicate workers (fuel values.size workers) index

private theorem size_le_capacity (size workers : Nat) (positive : 0 < workers) :
    size ≤ workers * fuel size workers := by
  exact (Nat.le_mul_iff_le_right positive).2 (Nat.le_refl (fuel size workers))

/-- The fuel covers every live offset, even for an empty array, an excess
worker, or a final stride shorter than the others. -/
theorem worker_iff {Alpha : Type u} (values : Array Alpha)
    (predicate : Alpha → Bool) (workers index : Nat) (positive : 0 < workers) :
    worker values predicate workers index = true ↔
      ∀ offset, ∀ live : index + workers * offset < values.size,
        predicate values[index + workers * offset] = true := by
  unfold worker
  rw [run_iff]
  constructor
  · intro checked offset live
    have capacity := size_le_capacity values.size workers positive
    have offsetBound : offset < fuel values.size workers := by
      by_cases bounded : offset < fuel values.size workers
      · exact bounded
      · have scaled := Nat.mul_le_mul_left workers (Nat.le_of_not_gt bounded)
        omega
    exact checked offset offsetBound live
  · intro checked offset _offsetBound live
    exact checked offset live

/-- Pure conjunction of worker results; the runtime may compute those immutable
results concurrently before taking this same conjunction. -/
def all {Alpha : Type u} (values : Array Alpha) (predicate : Alpha → Bool)
    (workers : Nat) : Bool :=
  (Array.range workers).all fun index => worker values predicate workers index

/-- Division and remainder assign every array index to a worker and offset.
The result is exactly the original Array.all for every positive worker count. -/
theorem all_eq {Alpha : Type u} (values : Array Alpha)
    (predicate : Alpha → Bool) (workers : Nat) (positive : 0 < workers) :
    all values predicate workers = values.all predicate := by
  rw [Bool.eq_iff_iff]
  unfold all
  rw [Array.all_eq_true, Array.all_eq_true]
  simp only [Array.size_range, Array.getElem_range]
  constructor
  · intro checked index live
    have workerBound : index % workers < workers := Nat.mod_lt index positive
    have selected := (worker_iff values predicate workers (index % workers) positive).mp
      (checked (index % workers) workerBound)
    have atIndex := selected (index / workers) (by
      rw [Nat.mod_add_div]
      exact live)
    simpa only [Nat.mod_add_div] using atIndex
  · intro checked index _workerBound
    apply (worker_iff values predicate workers index positive).mpr
    intro offset live
    exact checked (index + workers * offset) live

end NightstreamFPrime.Export.Stage1.StridedArrayAll
