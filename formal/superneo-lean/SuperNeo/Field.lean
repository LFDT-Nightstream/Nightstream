import SuperNeo.Goldilocks

namespace SuperNeo

/-- Base field carrier for SuperNeo (`F_q`). -/
abbrev F : Type := Fin Goldilocks.q

namespace F

instance : Inhabited F := ⟨⟨0, Goldilocks.q_pos⟩⟩

def ofNat (n : Nat) : F :=
  ⟨n % Goldilocks.q, Nat.mod_lt _ Goldilocks.q_pos⟩

def zero : F := ofNat 0
def one : F := ofNat 1

instance : Zero F := ⟨zero⟩
instance : One F := ⟨one⟩
instance : Add F := ⟨fun a b => ofNat (a.val + b.val)⟩
instance : Sub F := ⟨fun a b => ofNat (a.val + Goldilocks.q - b.val)⟩
instance : Mul F := ⟨fun a b => ofNat (a.val * b.val)⟩
instance : Neg F := ⟨fun a => ofNat (Goldilocks.q - a.val)⟩

def pow (a : F) (n : Nat) : F :=
  Id.run do
    let mut acc : F := 1
    let mut base := a
    let mut exp := n
    while exp > 0 do
      if exp % 2 = 1 then
        acc := acc * base
      base := base * base
      exp := exp / 2
    return acc

def inv (a : F) : F :=
  if a.val = 0 then
    0
  else
    pow a (Goldilocks.q - 2)

/-- Canonical representative in `[0, q)`. -/
def canonicalRep (a : F) : Nat := a.val

/-- Canonicality predicate (always true for the `Fin` encoding). -/
def isCanonical (a : F) : Prop := canonicalRep a < Goldilocks.q

instance (a : F) : Decidable (isCanonical a) := by
  unfold isCanonical canonicalRep
  infer_instance

theorem canonical (a : F) : isCanonical a :=
  a.isLt

def canonicalCheck (a : F) : Bool :=
  decide (isCanonical a)

theorem canonicalCheck_true (a : F) : canonicalCheck a = true := by
  unfold canonicalCheck
  exact decide_eq_true (canonical a)

/-- Centered integer representative in `[-q/2, q/2]` shape. -/
def centeredRep (a : F) : Int :=
  if _h : a.val ≤ Goldilocks.halfQ then
    Int.ofNat a.val
  else
    Int.ofNat a.val - Int.ofNat Goldilocks.q

def centeredAbs (a : F) : Nat :=
  Int.natAbs (centeredRep a)

theorem centeredRep_eq_of_le_halfQ {a : F} (h : a.val ≤ Goldilocks.halfQ) :
    centeredRep a = Int.ofNat a.val := by
  simp [centeredRep, h]

theorem centeredRep_eq_sub_q_of_halfQ_lt {a : F} (h : Goldilocks.halfQ < a.val) :
    centeredRep a = Int.ofNat a.val - Int.ofNat Goldilocks.q := by
  have hNot : ¬ a.val ≤ Goldilocks.halfQ := Nat.not_le_of_lt h
  simp [centeredRep, hNot]

@[simp] theorem ofNat_val (a : F) : ofNat a.val = a := by
  apply Fin.ext
  simp [ofNat, Nat.mod_eq_of_lt a.isLt]

@[simp] theorem canonicalRep_ofNat (n : Nat) :
    canonicalRep (ofNat n) = n % Goldilocks.q := rfl

@[simp] theorem val_lt_q (a : F) : a.val < Goldilocks.q :=
  a.isLt

end F

end SuperNeo
