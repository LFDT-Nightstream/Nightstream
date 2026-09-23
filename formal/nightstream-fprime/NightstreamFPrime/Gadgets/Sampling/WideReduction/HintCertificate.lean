import NightstreamFPrime.Circuit.StraightLine

/-! A certificate for a sequential hint program. Each source reads only
earlier variables, and evaluating it in the proposed final environment
gives the value at its destination. Execution then constructs that
environment on the whole allocated prefix. -/

namespace NightstreamFPrime.Gadgets.Sampling.WideReduction.HintCertificate

open NightstreamFPrime.Spec NightstreamFPrime.Circuit

def Valid (target : Env) : Nat → List Hint → Prop
  | _, [] => True
  | start, hint :: rest =>
      hint.source.VarsBelow start ∧ hint.eval target = target start ∧
        Valid target (start + 1) rest

theorem append (target : Env) (start : Nat) (left right : List Hint)
    (hl : Valid target start left)
    (hr : Valid target (start + left.length) right) :
    Valid target start (left ++ right) := by
  induction left generalizing start with
  | nil => simpa using hr
  | cons hint rest ih =>
      refine ⟨hl.1, hl.2.1, ih (start + 1) hl.2.2 ?_⟩
      simpa [Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using hr

theorem range_map (target : Env) (start count : Nat) (hint : Nat → Hint)
    (causal : ∀ index, index < count → (hint index).source.VarsBelow (start + index))
    (values : ∀ index, index < count → (hint index).eval target = target (start + index)) :
    Valid target start ((List.range count).map hint) := by
  induction count with
  | zero => trivial
  | succ count ih =>
      rw [List.range_succ, List.map_append]
      apply append
      · exact ih (fun i hi => causal i (by omega)) (fun i hi => values i (by omega))
      · simp only [List.length_map, List.length_range, List.map_cons, List.map_nil, Valid]
        exact ⟨causal count (by omega), values count (by omega), trivial⟩

theorem range_flatMap (target : Env) (start count stride : Nat) (block : Nat → List Hint)
    (lengths : ∀ index, index < count → (block index).length = stride)
    (blocks : ∀ index, index < count → Valid target (start + index * stride) (block index)) :
    Valid target start ((List.range count).flatMap block) := by
  induction count with
  | zero => trivial
  | succ count ih =>
      rw [List.range_succ, List.flatMap_append, List.flatMap_cons, List.flatMap_nil,
        List.append_nil]
      apply append
      · exact ih (fun i hi => lengths i (by omega)) (fun i hi => blocks i (by omega))
      · have length : ((List.range count).flatMap block).length = count * stride := by
          rw [List.length_flatMap]
          calc
            _ = ((List.range count).map (fun _ => stride)).sum := by
              congr 1
              apply List.map_congr_left
              intro i hi
              exact lengths i (by have := List.mem_range.mp hi; omega)
            _ = count * stride := by simp [List.map_const', List.sum_replicate, smul_eq_mul]
        rw [length]
        exact blocks count (by omega)

theorem finRange_flatMap (target : Env) (start count stride : Nat) (block : Fin count → List Hint)
    (lengths : ∀ index, (block index).length = stride)
    (blocks : ∀ index, Valid target (start + index.val * stride) (block index)) :
    Valid target start ((List.finRange count).flatMap block) := by
  induction count generalizing start with
  | zero => trivial
  | succ count ih =>
      rw [List.finRange_succ, List.flatMap_cons, List.flatMap_map]
      apply append
      · simpa using blocks 0
      · rw [lengths 0]
        apply ih (start + stride) (fun index => block index.succ)
        · exact fun index => lengths index.succ
        · intro index
          simpa [Fin.val_succ, Nat.add_mul, Nat.add_assoc, Nat.add_comm, Nat.add_left_comm]
            using blocks index.succ

theorem execute (target env : Env) (start : Nat) (hints : List Hint)
    (valid : Valid target start hints)
    (before : ∀ index, index < start → env index = target index) :
    ∀ index, index < start + hints.length → executeHints env start hints index = target index := by
  induction hints generalizing env start with
  | nil => simpa using before
  | cons hint rest ih =>
      have same := Hint.eval_eq_of_agree_below hint start env target valid.1 before
      have agreement : ∀ index, index < start + 1 →
          Env.set env start (hint.eval env) index = target index := by
        intro index below
        by_cases atStart : index = start
        · subst index
          rw [Env.set_self, same, valid.2.1]
        · rw [Env.set_of_ne env start index _ (by omega)]
          exact before index (by omega)
      intro index below
      exact ih _ _ valid.2.2 agreement index (by simp only [List.length_cons] at below; omega)

end NightstreamFPrime.Gadgets.Sampling.WideReduction.HintCertificate
