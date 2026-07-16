import Nightstream.SuperNeo.Concrete.Algebra

/-!
Contract: define the dimension and extension-arithmetic vocabulary shared by
the Π_CCS norm-check semantic phases.

Owns: verifier-selected Boolean-domain widths, their derived domain sizes,
and small total folds used to state `eq`, `chi`, and mixed NC polynomials.

Does not own: fixed production dimensions, generated manifests, transcript
challenges, assignment authority, SumCheck rows, or R1CS refinement.

Emits constraints: no.

Authority boundary: `Shape` is an explicit semantic input. This module does
not handwrite the current production values of `ell_m`, `ell_d`, or the output
count; a later artifact refinement must supply and authenticate those values.

| Definition | Mathematical obligation | Assumptions | Rust owner / status | Permits row removal? |
|---|---|---|---|---|
| `Shape` | derive `2^ellM` column and `2^ellD` lane domains | widths supplied by the verifier | fixed-profile artifact bridge open | no |
| `sumRange` | finite extension-field sum with explicit cardinality | concrete `K` arithmetic | semantic helper only | no |
| `productRange` | finite extension-field product with explicit cardinality | concrete `K` arithmetic | semantic helper only | no |
| `powK` | ordered powers, including `gamma^(i+1)` | concrete `K` arithmetic | semantic helper only | no |
-/

namespace Nightstream.Implementation.R1CS.PiCcsNc

open Nightstream.SuperNeo.Concrete

/-- Semantic domain widths for the split-NC polynomial.

No fixed profile values are embedded here. -/
structure Shape where
  ellM : Nat
  ellD : Nat
deriving DecidableEq, Repr

/-- Padded column-domain cardinality. -/
def Shape.columnDomain (shape : Shape) : Nat :=
  2 ^ shape.ellM

/-- Padded packed-lane-domain cardinality. -/
def Shape.laneDomain (shape : Shape) : Nat :=
  2 ^ shape.ellD

/-- Total finite sum in the concrete quadratic extension. -/
def sumRange : Nat → (Nat → K) → K
  | 0, _ => K.zero
  | count + 1, term => K.add (sumRange count term) (term count)

/-- Total finite product in the concrete quadratic extension. -/
def productRange : Nat → (Nat → K) → K
  | 0, _ => K.one
  | count + 1, term => K.mul (productRange count term) (term count)

/-- Concrete quadratic-extension exponentiation. -/
def powK (base : K) : Nat → K
  | 0 => K.one
  | exponent + 1 => K.mul (powK base exponent) base

private theorem k_add_zero (value : K) :
    K.add value K.zero = value := by
  rcases value with ⟨c0, c1⟩
  simp [K.add, K.zero]

private theorem k_mul_zero (value : K) :
    K.mul value K.zero = K.zero := by
  rcases value with ⟨c0, c1⟩
  simp only [K.mul, K.zero, Fin.mul_zero, Fin.add_zero]

private theorem k_zero_add (value : K) :
    K.add K.zero value = value := by
  rcases value with ⟨c0, c1⟩
  simp [K.add, K.zero]

private theorem k_zero_mul (value : K) :
    K.mul K.zero value = K.zero := by
  rcases value with ⟨c0, c1⟩
  simp only [K.mul, K.zero, Fin.zero_mul, Fin.mul_zero, Fin.add_zero]

private theorem k_mul_one (value : K) :
    K.mul value K.one = value := by
  rcases value with ⟨c0, c1⟩
  simp only [K.mul, K.one, Fin.mul_one, Fin.mul_zero, Fin.add_zero,
    Fin.zero_add]

private theorem k_one_mul (value : K) :
    K.mul K.one value = value := by
  rcases value with ⟨c0, c1⟩
  simp only [K.mul, K.one, Fin.one_mul, Fin.zero_mul, Fin.mul_zero,
    Fin.add_zero]

/-- A finite sum vanishes when every indexed term vanishes. -/
theorem sumRange_eq_zero
    (count : Nat) (term : Nat → K)
    (zero : ∀ index, index < count → term index = K.zero) :
    sumRange count term = K.zero := by
  induction count with
  | zero => rfl
  | succ count inductionHypothesis =>
      rw [sumRange, inductionHypothesis]
      · rw [zero count (Nat.lt_succ_self count)]
        rfl
      · intro index indexLt
        exact zero index (Nat.lt_trans indexLt (Nat.lt_succ_self count))

/-- Pointwise-equal summands have equal finite sums. -/
theorem sumRange_congr
    (count : Nat) (left right : Nat → K)
    (equal : ∀ index, index < count → left index = right index) :
    sumRange count left = sumRange count right := by
  induction count with
  | zero => rfl
  | succ count inductionHypothesis =>
      rw [sumRange, sumRange, inductionHypothesis]
      · rw [equal count (Nat.lt_succ_self count)]
      · intro index indexLt
        exact equal index (Nat.lt_trans indexLt (Nat.lt_succ_self count))

/-- A finite product is one when every indexed factor is one. -/
theorem productRange_eq_one
    (count : Nat) (term : Nat → K)
    (one : ∀ index, index < count → term index = K.one) :
    productRange count term = K.one := by
  induction count with
  | zero => rfl
  | succ count inductionHypothesis =>
      rw [productRange, inductionHypothesis]
      · rw [one count (Nat.lt_succ_self count)]
        rfl
      · intro index indexLt
        exact one index (Nat.lt_trans indexLt (Nat.lt_succ_self count))

/-- Pointwise-equal factors have equal finite products. -/
theorem productRange_congr
    (count : Nat) (left right : Nat → K)
    (equal : ∀ index, index < count → left index = right index) :
    productRange count left = productRange count right := by
  induction count with
  | zero => rfl
  | succ count inductionHypothesis =>
      rw [productRange, productRange, inductionHypothesis]
      · rw [equal count (Nat.lt_succ_self count)]
      · intro index indexLt
        exact equal index (Nat.lt_trans indexLt (Nat.lt_succ_self count))

/-- A finite product vanishes when any indexed factor vanishes. -/
theorem productRange_eq_zero_of_exists
    (count : Nat) (term : Nat → K)
    (zero : ∃ index, index < count ∧ term index = K.zero) :
    productRange count term = K.zero := by
  induction count with
  | zero =>
      rcases zero with ⟨index, indexLt, _⟩
      omega
  | succ count inductionHypothesis =>
      rcases zero with ⟨index, indexLt, indexZero⟩
      rw [productRange]
      by_cases last : index = count
      · subst index
        rw [indexZero]
        exact k_mul_zero _
      · have earlier : index < count := by omega
        rw [inductionHypothesis ⟨index, earlier, indexZero⟩]
        exact k_zero_mul _

/-- A finite sum with exactly one selected index returns that term. -/
theorem sumRange_select
    (count selected : Nat) (term : Nat → K)
    (selectedLt : selected < count) :
    sumRange count (fun index =>
      if index = selected then term index else K.zero) =
      term selected := by
  induction count with
  | zero => omega
  | succ count inductionHypothesis =>
      rw [sumRange]
      by_cases last : selected = count
      · subst selected
        have prefixZero :
            sumRange count (fun index =>
              if index = count then term index else K.zero) = K.zero := by
          apply sumRange_eq_zero
          intro index indexLt
          simp [Nat.ne_of_lt indexLt]
        rw [prefixZero, if_pos rfl]
        exact k_zero_add _
      · have selectedEarlier : selected < count := by omega
        rw [inductionHypothesis selectedEarlier, if_neg]
        · exact k_add_zero _
        · exact Ne.symm last

/-- Multiplication by a zero right operand vanishes. -/
theorem mul_zero (value : K) :
    K.mul value K.zero = K.zero :=
  k_mul_zero value

/-- Adding a zero right operand is neutral. -/
theorem add_zero (value : K) :
    K.add value K.zero = value :=
  k_add_zero value

/-- Multiplication by a zero left operand vanishes. -/
theorem zero_mul (value : K) :
    K.mul K.zero value = K.zero :=
  k_zero_mul value

/-- Multiplication by one on the right is neutral. -/
theorem mul_one (value : K) :
    K.mul value K.one = value :=
  k_mul_one value

/-- Multiplication by one on the left is neutral. -/
theorem one_mul (value : K) :
    K.mul K.one value = value :=
  k_one_mul value

/-- Adding a zero left operand is neutral. -/
theorem zero_add (value : K) :
    K.add K.zero value = value :=
  k_zero_add value

end Nightstream.Implementation.R1CS.PiCcsNc
