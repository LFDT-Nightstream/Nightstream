import Std

namespace SuperNeo

def q : Nat := 18446744069414584321

theorem q_pos : 0 < q := by
  decide

structure F where
  val : Nat
  deriving Repr, DecidableEq, BEq, Inhabited

namespace F

def ofNat (x : Nat) : F := ⟨x % q⟩

def ofInt (x : Int) : F :=
  let qi := Int.ofNat q
  let xr := ((x % qi) + qi) % qi
  ofNat xr.toNat

def zero : F := ofNat 0

def one : F := ofNat 1

instance : OfNat F n where
  ofNat := ofNat n

instance : Add F where
  add a b := ofNat (a.val + b.val)

instance : Sub F where
  sub a b := ofNat (a.val + q - b.val)

instance : Mul F where
  mul a b := ofNat (a.val * b.val)

instance : Neg F where
  neg a :=
    if a.val = 0 then
      zero
    else
      ⟨q - a.val⟩

partial def egcd (a b : Int) : Int × Int × Int :=
  if b = 0 then
    (a, 1, 0)
  else
    let (g, x1, y1) := egcd b (a % b)
    (g, y1, x1 - (a / b) * y1)

/-- Multiplicative inverse in Goldilocks (returns 0 for 0). -/
def inv (a : F) : F :=
  if a.val = 0 then
    zero
  else
    let ai := Int.ofNat a.val
    let qi := Int.ofNat q
    let (g, x, _) := egcd ai qi
    if g = 1 ∨ g = -1 then
      let x' := ((x % qi) + qi) % qi
      ofNat x'.toNat
    else
      zero

instance : Inv F where
  inv := inv

def ofNatArray (xs : Array Nat) : Array F :=
  xs.map ofNat

def Canonical (a : F) : Prop := a.val < q

instance canonical_decidable (a : F) : Decidable (Canonical a) := by
  unfold Canonical
  infer_instance

theorem ofNat_val_mod (x : Nat) : (ofNat x).val = x % q := rfl

theorem ofNat_val_lt_q (x : Nat) : (ofNat x).val < q := by
  unfold ofNat
  exact Nat.mod_lt _ q_pos

theorem canonical_ofNat (x : Nat) : Canonical (ofNat x) := by
  unfold Canonical
  exact ofNat_val_lt_q x

theorem canonical_zero : Canonical zero := by
  unfold zero
  exact canonical_ofNat 0

theorem canonical_one : Canonical one := by
  unfold one
  exact canonical_ofNat 1

theorem canonical_default : Canonical (default : F) := by
  change (default : F).val < q
  change 0 < q
  exact q_pos

theorem ofNat_val_eq_of_canonical {a : F} (ha : Canonical a) : ofNat a.val = a := by
  cases a with
  | mk v =>
      unfold Canonical at ha
      simp [ofNat, Nat.mod_eq_of_lt ha]

theorem canonical_add (a b : F) : Canonical (a + b) := by
  unfold Canonical
  change (ofNat (a.val + b.val)).val < q
  exact ofNat_val_lt_q (a.val + b.val)

theorem canonical_sub (a b : F) : Canonical (a - b) := by
  unfold Canonical
  change (ofNat (a.val + q - b.val)).val < q
  exact ofNat_val_lt_q (a.val + q - b.val)

theorem canonical_mul (a b : F) : Canonical (a * b) := by
  unfold Canonical
  change (ofNat (a.val * b.val)).val < q
  exact ofNat_val_lt_q (a.val * b.val)

theorem add_comm (a b : F) : a + b = b + a := by
  cases a with
  | mk av =>
      cases b with
      | mk bv =>
          show ofNat (av + bv) = ofNat (bv + av)
          rw [Nat.add_comm]

theorem mul_comm (a b : F) : a * b = b * a := by
  cases a with
  | mk av =>
      cases b with
      | mk bv =>
          show ofNat (av * bv) = ofNat (bv * av)
          rw [Nat.mul_comm]

theorem zero_val : zero.val = 0 := by
  unfold zero ofNat
  simp [q]

theorem one_val : one.val = 1 := by
  unfold one ofNat
  have hq : 1 < q := by decide
  exact Nat.mod_eq_of_lt hq

theorem zero_add_of_canonical {a : F} (ha : Canonical a) : zero + a = a := by
  change ofNat (0 + a.val) = a
  simpa using (ofNat_val_eq_of_canonical (a := a) ha)

theorem add_zero_of_canonical {a : F} (ha : Canonical a) : a + zero = a := by
  change ofNat (a.val + 0) = a
  simpa using (ofNat_val_eq_of_canonical (a := a) ha)

theorem one_mul_of_canonical {a : F} (ha : Canonical a) : one * a = a := by
  change ofNat (1 * a.val) = a
  simpa using (ofNat_val_eq_of_canonical (a := a) ha)

theorem mul_one_of_canonical {a : F} (ha : Canonical a) : a * one = a := by
  change ofNat (a.val * 1) = a
  simpa using (ofNat_val_eq_of_canonical (a := a) ha)

theorem canonical_getElem!_of_all
  {arr : Array F}
  (hArr : arr.all (fun x => decide (Canonical x)) = true)
  (i : Nat) :
  Canonical (arr[i]!) := by
  by_cases hi : i < arr.size
  · have hIDec : decide (Canonical arr[i]) = true := (Array.all_eq_true.mp hArr) i hi
    have hI : Canonical arr[i] := decide_eq_true_eq.mp hIDec
    simpa [hi] using hI
  · simpa [hi] using canonical_default

end F

end SuperNeo
