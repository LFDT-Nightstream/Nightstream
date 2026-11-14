import NightstreamFPrime.Lifecycle.Nebula.RowList

namespace NightstreamFPrime.Lifecycle.Nebula.PaddingZero

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Nebula
open NightstreamFPrime.Circuit

private theorem natSum_le_length (bits : List F) :
    (bits.map (fun x => (toBool x).toNat)).sum ≤ bits.length := by
  induction bits with
  | nil => simp
  | cons x bits ih =>
    have head := Bool.toNat_le (toBool x)
    simp only [List.map_cons, List.sum_cons, List.length_cons]
    omega

private theorem natSum_zero_iff (bits : List F) :
    (bits.map (fun x => (toBool x).toNat)).sum = 0 ↔
      ∀ x ∈ bits, toBool x = false := by
  induction bits with
  | nil => simp
  | cons x bits ih =>
    cases value : toBool x <;> simp [value, ih]

private theorem sum_eq_natWord {bits : List F} (allBits : ∀ x ∈ bits, IsBit x) :
    bits.sum = natWord (bits.map (fun x => (toBool x).toNat)).sum := by
  induction bits with
  | nil => rfl
  | cons x bits ih =>
    rw [List.sum_cons, List.map_cons, List.sum_cons, natWord_add,
      natWord_toBool (allBits x (by simp)),
      ih (fun x member => allBits x (by simp [member]))]

theorem sum_zero_iff {bits : List F} (allBits : ∀ x ∈ bits, IsBit x)
    (short : bits.length < fieldModulus) :
    bits.sum = 0 ↔ ∀ x ∈ bits, x = 0 := by
  have below := lt_of_le_of_lt (natSum_le_length bits) short
  have wordZero : natWord (bits.map (fun x => (toBool x).toNat)).sum = 0 ↔
      (bits.map (fun x => (toBool x).toNat)).sum = 0 := by
    constructor
    · intro same
      have value := congrArg Fin.val same
      simpa only [natWord_val below, Fin.val_zero] using value
    · intro same
      rw [same, natWord_zero]
  rw [sum_eq_natWord allBits, wordZero, natSum_zero_iff]
  apply forall_congr'
  intro x
  apply forall_congr'
  intro member
  rcases allBits x member with rfl | rfl <;>
    simp [toBool, show (0 : F) ≠ 1 by decide, show (1 : F) ≠ 0 by decide]

theorem gated_sum_zero_iff (gate : F) {bits : List F}
    (allBits : ∀ x ∈ bits, IsBit x) (short : bits.length < fieldModulus) :
    gate * bits.sum = 0 ↔ ∀ x ∈ bits, gate * x = 0 := by
  have product (a b : F) : a * b = 0 ↔ a = 0 ∨ b = 0 := by
    let : Fact (Nat.Prime fieldModulus) := ⟨FieldPrime.fieldModulus_natPrime⟩
    have key (x y : ZMod fieldModulus) : x * y = 0 ↔ x = 0 ∨ y = 0 := mul_eq_zero
    exact key a b
  by_cases gateZero : gate = 0
  · simp [gateZero]
  · have cancel (x : F) : gate * x = 0 ↔ x = 0 := by simpa [gateZero] using product gate x
    simpa only [cancel] using sum_zero_iff allBits short

theorem rows_hold_iff (env : Env) (gate : Circuit.Expr) (bits : List Circuit.Expr)
    (allBits : ∀ x ∈ bits, IsBit (x.eval env)) (short : bits.length < fieldModulus) :
    ConstraintsHold env [gate * Expr.sum bits] ↔
      ConstraintsHold env (bits.map (fun x => gate * x)) := by
  rw [hold_single, hold_map]
  simp only [Circuit.Expr.eval_hmul, Expr.eval_sum]
  have checked := gated_sum_zero_iff (gate.eval env) (bits := bits.map (Circuit.Expr.eval env))
    (by simpa only [List.forall_mem_map] using allBits) (by simpa using short)
  simpa only [List.forall_mem_map] using checked

theorem paddingRows_hold_iff (env : Env) (gate : Circuit.Expr) (bits : List Circuit.Expr)
    (allBits : ∀ x ∈ bits, IsBit (x.eval env)) :
    ConstraintsHold env (Rows.paddingRows gate bits) ↔
      ConstraintsHold env (bits.map (fun x => gate * x)) := by
  unfold Rows.paddingRows
  split_ifs with short
  · exact rows_hold_iff env gate bits allBits short
  · rfl

theorem paddingRows_of_valid {p : Geometry} (valid : p.Valid) (j : ℕ) :
    Rows.paddingRows (Sym.pad p j) (Sym.opLane p j).tail =
      [Sym.pad p j * Expr.sum (Sym.opLane p j).tail] := by
  have widthBound : p.opWidth < fieldModulus :=
    lt_of_le_of_lt (Nat.le_mul_of_pos_left _ valid.positive.1) valid.laneBits.1
  have short : (Sym.opLane p j).tail.length < fieldModulus := by
    simp only [Sym.opLane, List.length_tail, List.length_append, List.length_cons,
      List.length_nil, List.length_ofFn]
    simp only [Geometry.opWidth] at widthBound
    omega
  exact ite_eq_left short

end NightstreamFPrime.Lifecycle.Nebula.PaddingZero
