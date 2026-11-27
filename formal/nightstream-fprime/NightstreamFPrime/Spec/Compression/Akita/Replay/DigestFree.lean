import NightstreamFPrime.Spec.Compression.Akita.Replay.JoinShape

/-! The replay reads the hash `Public.digest` of the two last-block digests only at the terminal
operation (Akita revision `d50f700`, `akita-verifier/src/terminal/direct.rs`; ePrint 2026/1983,
Lemma 10.47; `docs/akita-work/w20/design.md`, section 3).

- `step_withDigest`, `runFrom_withDigest`: the step of an operation that is not the terminal, and
  the run of such operations, are the same for each hash.
- `stateBefore_withDigest`: the state before the operation `c` is the same for each hash when no
  operation before `c` is the terminal. `script20_free`: each script of `layout20` has its only
  terminal at the end, so `stateBefore_withDigest20` has no such hypothesis. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

open NightstreamFPrime.Spec.Compression.Akita

open Replayer

@[simp] theorem Public.withDigest_digest (pub : Public) (digest : ByteArray → ByteArray) :
    (pub.withDigest digest).digest = digest := rfl

@[simp] theorem Public.sealed_digest (pub : Public) : pub.sealed.digest = pub.digest := rfl

theorem Public.withDigest_self (pub : Public) : pub.withDigest pub.digest = pub := rfl

theorem Public.withDigest_sealed (pub : Public) (digest : ByteArray → ByteArray) :
    (pub.withDigest digest).sealed = pub.sealed.withDigest digest := rfl

theorem Public.withDigest_shaped (pub : Public) (digest : ByteArray → ByteArray) :
    (pub.withDigest digest).shaped = pub.shaped := rfl

theorem Public.withDigest_ops (pub : Public) (digest : ByteArray → ByteArray) :
    (pub.withDigest digest).ops = pub.ops := rfl

/-- The operation is the terminal. -/
def Op.isTerminal : Op → Bool
  | .terminal _ => true
  | _ => false

namespace Replayer

variable (pub : Public) (digest : ByteArray → ByteArray) (proof : ByteArray)
  (messages : Array Message)

/-- The step of an operation that is not the terminal does not read the hash. -/
theorem step_withDigest (i : ℕ) {op : Op} (free : op.isTerminal = false) :
    step (pub.withDigest digest) proof messages i op = step pub proof messages i op := by
  cases op <;> first | rfl | cases free

/-- The run of operations that are not the terminal does not read the hash. -/
theorem runFrom_withDigest : ∀ (l : List (Op × ℕ)) (s : Replayer),
    (∀ x ∈ l, x.1.isTerminal = false) →
      runFrom (pub.withDigest digest) proof messages l s = runFrom pub proof messages l s
  | [], _, _ => rfl
  | x :: l, s, free => by
      rw [runFrom_cons, runFrom_cons,
        step_withDigest pub digest proof messages x.2 (free x List.mem_cons_self)]
      rcases ((step pub proof messages x.2 x.1).run).run s with ⟨_ | _, s₁⟩
      · rfl
      · exact runFrom_withDigest l s₁ fun y member => free y (List.mem_cons_of_mem _ member)

/-- The state before the operation `c` does not depend on the hash when no operation before `c`
is the terminal. -/
theorem stateBefore_withDigest (ops : Array Op) (c : ℕ)
    (free : ∀ j < c, ∀ level, ops[j]? ≠ some (.terminal level)) :
    stateBefore (pub.withDigest digest) proof messages ops c =
      stateBefore pub proof messages ops c := by
  unfold stateBefore
  show (runFrom (pub.withDigest digest) proof messages _ (channelStart pub)).2 = _
  rw [runFrom_withDigest pub digest proof messages]
  intro x member
  obtain ⟨j, inside, at_⟩ := List.getElem_of_mem member
  have short : j < c ∧ j < ops.zipIdx.toList.length := by
    rw [List.length_take] at inside
    omega
  rw [List.getElem_take, zipIdx_getElem ops short.2] at at_
  subst at_
  cases found : ops[j]'(by simpa using short.2) with
  | terminal level =>
      exact absurd ((Array.getElem?_eq_getElem _).trans (congrArg some found))
        (free j short.1 level)
  | _ => rfl

end Replayer

theorem skeleton20_free :
    ((skeleton (layout20 0 0)).take 1521).all (fun op => !op.isTerminal) = true := by
  decide +kernel

/-- No operation before the last one of a script of `layout20` is the terminal. -/
theorem script20_free {zMax rice : ℕ} {plan : Array Run} {ops : Array Op}
    (scripted : script (layout20 zMax rice) plan = .ok ops) {j : ℕ} (before : j < ops.size - 1)
    (level : ℕ) : ops[j]? ≠ some (.terminal level) := by
  intro at_
  have erased : ops.toList.map Op.erase = skeleton (layout20 0 0) := by
    rw [script_erase scripted, skeleton20]
  have size : ops.size - 1 = 1521 := by rw [size_skeleton erased, skeleton20_length]
  have found : ((skeleton (layout20 0 0)).take 1521)[j]? = some (.terminal level) := by
    rw [List.getElem?_take_of_lt (by omega), ← erased, List.getElem?_map, Array.getElem?_toList,
      at_]
    rfl
  exact Bool.false_ne_true
    (List.all_eq_true.mp skeleton20_free _ (List.mem_of_getElem? found))

/-- For `layout20`, the state of the replay before each operation up to the terminal is the same
for each hash of the two last-block digests: the replay reads the hash only at the terminal. -/
theorem stateBefore_withDigest20 {zMax rice : ℕ} (pub : Public)
    (layout : pub.layout = layout20 zMax rice) (digest : ByteArray → ByteArray)
    (proof : ByteArray) (messages : Array Message) {c : ℕ} (before : c ≤ pub.ops.size - 1) :
    stateBefore (pub.withDigest digest) proof messages pub.ops c =
      stateBefore pub proof messages pub.ops c := by
  refine stateBefore_withDigest pub digest proof messages pub.ops c fun j below level => ?_
  have inside : j < pub.ops.size - 1 := by omega
  revert inside
  unfold Public.ops
  rw [layout]
  cases scripted : script (layout20 zMax rice) pub.plan with
  | error _ => exact fun inside => absurd inside (by simp)
  | ok ops => exact fun inside => script20_free scripted inside level

end NightstreamFPrime.Spec.Compression.Akita.Replay
