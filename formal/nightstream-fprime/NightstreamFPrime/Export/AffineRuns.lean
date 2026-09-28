import NightstreamFPrime.Export.Codec
import Mathlib.Data.List.Basic
import Mathlib.Data.List.OfFn
import Mathlib.Data.List.GetD
import Mathlib.Tactic.Ring

/-!
Owns a small canonical affine-run codec for verifier-owned index streams.
The reference compressor is proof-oriented. Production emitters may use a
proved tail-recursive equivalent after measurement shows that it is needed.
-/

namespace NightstreamFPrime.Export.AffineRuns

open NightstreamFPrime.Export.Codec

structure Run where
  first : Nat
  step : Nat
  count : Nat
deriving Repr, DecidableEq

def values : Nat → Nat → Nat → List Nat
  | _, _, 0 => []
  | first, step, count + 1 =>
      first :: values (first + step) step count

def Run.expand (run : Run) : List Nat :=
  values run.first run.step run.count

def Run.single (value : Nat) : Run :=
  ⟨value, 0, 1⟩

def expand (runs : List Run) : List Nat :=
  runs.flatMap Run.expand

@[simp] theorem values_length (first step count : Nat) :
    (values first step count).length = count := by
  induction count generalizing first with
  | zero => rfl
  | succ count inductionHypothesis =>
      simp [values, inductionHypothesis]

@[simp] theorem Run.expand_length (run : Run) :
    run.expand.length = run.count := by
  exact values_length run.first run.step run.count

theorem expand_length (runs : List Run) :
    (expand runs).length = (runs.map Run.count).sum := by
  simp [expand]

/-- Add one value to the front of an already compressed suffix. -/
def prepend (value : Nat) : List Run → List Run
  | [] => [Run.single value]
  | run :: runs =>
      if run.count = 1 then
        if value ≤ run.first then
          ⟨value, run.first - value, 2⟩ :: runs
        else
          Run.single value :: run :: runs
      else if value + run.step = run.first then
        ⟨value, run.step, run.count + 1⟩ :: runs
      else
        Run.single value :: run :: runs

/-- Greedy canonical compression from right to left. A decreasing boundary
starts a new run because the encoded step is a natural number. -/
def compress : List Nat → List Run
  | [] => []
  | value :: rest => prepend value (compress rest)

theorem compress_eq_foldr_prepend (indices : List Nat) :
    compress indices = indices.foldr prepend [] := by
  induction indices with
  | nil => rfl
  | cons value rest inductionHypothesis =>
      simp [compress, inductionHypothesis]

/-- Allocation-bounded executable compression for an indexed source. It
visits indices from right to left and retains only the compressed runs. -/
@[inline] def compressIndexedTR {count : Nat}
    (source : Fin count → Nat) : List Run :=
  go count (Nat.le_refl count) []
where
  go : (remaining : Nat) → remaining ≤ count → List Run → List Run
    | 0, _, runs => runs
    | next + 1, hle, runs =>
        go next (Nat.le_trans (Nat.le_succ next) hle)
          (prepend
            (source ⟨next,
              Nat.lt_of_lt_of_le (Nat.lt_succ_self next) hle⟩)
            runs)

private def indexedPrefix {count : Nat} (source : Fin count → Nat)
    {remaining : Nat} (hle : remaining ≤ count) : Fin remaining → Nat :=
  fun index =>
    source ⟨index.val, Nat.lt_of_lt_of_le index.isLt hle⟩

private theorem compressIndexedTR_go_eq {count : Nat}
    (source : Fin count → Nat) :
    ∀ (remaining : Nat) (hle : remaining ≤ count) (runs : List Run),
      compressIndexedTR.go source remaining hle runs =
        (List.ofFn (indexedPrefix source hle)).foldr prepend runs := by
  intro remaining
  induction remaining with
  | zero =>
      intro hle runs
      rfl
  | succ next inductionHypothesis =>
      intro hle runs
      rw [compressIndexedTR.go,
        inductionHypothesis
          (Nat.le_trans (Nat.le_succ next) hle)]
      rw [List.ofFn_succ', List.concat_eq_append, List.foldr_concat]
      rfl

/-- The bounded executable emits the exact canonical run list, not only an
extensionally equivalent source stream. -/
theorem compressIndexedTR_eq_compress_ofFn {count : Nat}
    (source : Fin count → Nat) :
    compressIndexedTR source = compress (List.ofFn source) := by
  rw [compressIndexedTR,
    compressIndexedTR_go_eq source count (Nat.le_refl count) []]
  rw [compress_eq_foldr_prepend]
  rfl

@[simp] theorem Run.expand_single (value : Nat) :
    (Run.single value).expand = [value] := by
  rfl

private theorem expand_pair (value : Nat) (run : Run)
    (one : run.count = 1) (ordered : value ≤ run.first) :
    (Run.mk value (run.first - value) 2).expand =
      value :: run.expand := by
  cases run with
  | mk first step count =>
      simp only at one ordered ⊢
      subst count
      simp [Run.expand, values, Nat.add_sub_of_le ordered]

private theorem expand_prepend (value : Nat) (run : Run)
    (next : value + run.step = run.first) :
    (Run.mk value run.step (run.count + 1)).expand =
      value :: run.expand := by
  simp [Run.expand, values, next]

/-- Compressing and then expanding preserves every index and its order. -/
theorem expand_compress (indices : List Nat) :
    expand (compress indices) = indices := by
  induction indices with
  | nil => rfl
  | cons value rest inductionHypothesis =>
      rw [compress]
      cases compressedEq : compress rest with
      | nil =>
          rw [prepend]
          have restEmpty : rest = [] := by
            rw [compressedEq] at inductionHypothesis
            simpa [expand] using inductionHypothesis.symm
          subst rest
          rfl
      | cons run runs =>
          rw [prepend]
          have tail : run.expand ++ List.flatMap Run.expand runs = rest := by
            rw [compressedEq] at inductionHypothesis
            simpa [expand] using inductionHypothesis
          by_cases one : run.count = 1
          · rw [if_pos one]
            by_cases ordered : value ≤ run.first
            · rw [if_pos ordered]
              simp only [expand, List.flatMap_cons]
              rw [expand_pair value run one ordered, List.cons_append, tail]
            · rw [if_neg ordered]
              simp only [expand, List.flatMap_cons, Run.expand_single,
                List.cons_append, List.nil_append, tail]
          · rw [if_neg one]
            by_cases next : value + run.step = run.first
            · rw [if_pos next]
              simp only [expand, List.flatMap_cons]
              rw [expand_prepend value run next, List.cons_append, tail]
            · rw [if_neg next]
              simp only [expand, List.flatMap_cons, Run.expand_single,
                List.cons_append, List.nil_append, tail]

def Run.format : Format Run where
  encode := fun run => .array [.atom run.first, .atom run.step, .atom run.count]
  decode
    | .array [.atom first, .atom step, .atom count] =>
        .ok ⟨first, step, count⟩
    | _ => .error "invalid affine run"
  decode_encode := by
    intro run
    cases run
    rfl

def format : Format (List Run) := Codec.list Run.format

theorem decode_encode (runs : List Run) :
    format.decode (format.encode runs) = .ok runs :=
  format.decode_encode runs

/-- Direct point lookup in a serialized affine-run stream. -/
def sourceAt : List AffineRuns.Run → Nat → Nat
  | [], _ => 0
  | run :: rest, slot =>
      if slot < run.count then
        run.first + run.step * slot
      else
        sourceAt rest (slot - run.count)

private theorem affineValues_getD_of_lt (first step count slot : Nat)
    (inside : slot < count) :
    (AffineRuns.values first step count).getD slot 0 =
      first + step * slot := by
  induction count generalizing first slot with
  | zero => omega
  | succ count inductionHypothesis =>
      cases slot with
      | zero => simp [AffineRuns.values]
      | succ slot =>
          simp only [AffineRuns.values, List.getD_cons_succ]
          rw [inductionHypothesis (first := first + step)
            (slot := slot) (by omega)]
          ring

/-- Direct run lookup is the pointwise meaning of affine-run expansion. -/
theorem sourceAt_eq_expand_getD
    (runs : List AffineRuns.Run) (slot : Nat) :
    sourceAt runs slot = (AffineRuns.expand runs).getD slot 0 := by
  induction runs generalizing slot with
  | nil => rfl
  | cons run rest inductionHypothesis =>
      change
        (if slot < run.count then run.first + run.step * slot
          else sourceAt rest (slot - run.count)) =
        (run.expand ++ AffineRuns.expand rest).getD slot 0
      by_cases inside : slot < run.count
      · rw [if_pos inside, List.getD_append]
        · simpa [AffineRuns.Run.expand] using
            (affineValues_getD_of_lt run.first run.step run.count slot inside).symm
        · simpa using inside
      · have after : run.expand.length ≤ slot := by
          simpa using Nat.le_of_not_gt inside
        rw [if_neg inside,
          List.getD_append_right _ _ _ _ after]
        simpa using inductionHypothesis (slot := slot - run.count)


end NightstreamFPrime.Export.AffineRuns
