import NightstreamFPrime.Export.Codec

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

/-- Greedy canonical compression from right to left. A decreasing boundary
starts a new run because the encoded step is a natural number. -/
def compress : List Nat → List Run
  | [] => []
  | value :: rest =>
      match compress rest with
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
          have restEmpty : rest = [] := by
            rw [compressedEq] at inductionHypothesis
            simpa [expand] using inductionHypothesis.symm
          subst rest
          rfl
      | cons run runs =>
          have tail : run.expand ++ List.flatMap Run.expand runs = rest := by
            rw [compressedEq] at inductionHypothesis
            simpa [expand] using inductionHypothesis
          simp only
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

end NightstreamFPrime.Export.AffineRuns
