import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier.Algebra
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.NormRange

/-! Provenance: copied from `formal/nightstream-lean/Nightstream/SuperNeo/Folding/PiCCS/PaperJoint/ConcreteCarrier/NoZeroDivisors.lean`
at commit `fb7a8a99aefbb8ebb5474681ecf80f1b95a1b7a2`; namespaces renamed, otherwise unchanged. -/

/-!
Concrete quadratic-extension cancellation boundary for paper-level `Pi_CCS`.

Owns: the quadratic norm and conjugate of the concrete carrier
`K = F[u]/(u² - 7)`, the exact irreducibility premise still missing from the
active dependency-light Lean project, and the derivation of extension-field
no-zero-divisors from that premise plus base-field no-zero-divisors.

Does not own: a proof that the Goldilocks modulus is prime, a proof that seven
is a quadratic nonresidue modulo that modulus, Fiat--Shamir probabilities,
Rust/R1CS refinement, row emission, row removal, or constraint counts.

Emits constraints: no.

Authority boundary: `SevenProjectiveNonresidue` is deliberately visible. The
deprecated Mathlib-backed SuperNeo project proves the corresponding concrete
number-theoretic fact, but importing that theorem here would add a second field
carrier and a large dependency without a proved carrier equivalence. Until an
active arithmetic certificate instantiates this premise, downstream results
remain model-level and conditional.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `pi_ccs.carrier.extension.norm` | `N(a + bu) = a² - 7b²` | computed | `quadraticNorm` |
| `pi_ccs.carrier.extension.conjugate` | `(a - bu)(a + bu) = N(a + bu)` | derived | `conjugate_mul_self` |
| `pi_ccs.carrier.extension.irreducible` | `a² - 7b² = 0` only at `(0,0)` | security boundary | `SevenProjectiveNonresidue` |
| `pi_ccs.carrier.extension.no_zero_divisors` | `xy = 0` implies `x = 0` or `y = 0` | derived | `extensionNoZeroDivisors_of_base_and_seven` |
-/

namespace NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

/-- Norm of `a + bu` for the concrete relation `u² = 7`. -/
def quadraticNorm (value : K) : F :=
  value.c0 * value.c0 - 7 * value.c1 * value.c1

/-- Coefficient conjugation `a + bu ↦ a - bu`. -/
def conjugate (value : K) : K :=
  ⟨value.c0, -value.c1⟩

/-- Minimal concrete irreducibility boundary for `u² - 7`: its homogeneous
norm form has no nontrivial zero over the base carrier. -/
def SevenProjectiveNonresidue : Prop :=
  ∀ real imaginary : F,
    real * real - 7 * imaginary * imaginary = 0 →
      real = 0 ∧ imaginary = 0

/-- The concrete quadratic-extension multiplication has no zero divisors. -/
def ExtensionNoZeroDivisors : Prop :=
  ∀ left right : K,
    K.mul left right = K.zero → left = K.zero ∨ right = K.zero

/-- Multiplying by the conjugate embeds the exact quadratic norm. -/
theorem conjugate_mul_self (value : K) :
    K.mul (conjugate value) value = K.embed (quadraticNorm value) := by
  rcases value with ⟨real, imaginary⟩
  simp only [conjugate, quadraticNorm, K.mul, K.embed, K.mk.injEq]
  constructor
  · rw [Fin.sub_eq_add_neg]
    congr 1
    calc
      7 * -imaginary * imaginary =
          (-imaginary) * (7 * imaginary) := by ac_rfl
      _ = -(imaginary * (7 * imaginary)) :=
        ConcreteCarrier.baseLaws.neg_mul _ _
      _ = -(7 * imaginary * imaginary) := by congr 1 <;> ac_rfl
  · have negMul : -imaginary * real = -(imaginary * real) := by
      simpa [ConcreteCarrier.baseOps] using
        (ConcreteCarrier.baseLaws.neg_mul imaginary real)
    have commute : imaginary * real = real * imaginary := by
      simpa [ConcreteCarrier.baseOps] using
        (ConcreteCarrier.baseLaws.mul_comm imaginary real)
    calc
      real * imaginary + -imaginary * real =
          real * imaginary + -(imaginary * real) := by
            rw [negMul]
      _ = real * imaginary + -(real * imaginary) := by
        rw [commute]
      _ = 0 := ConcreteCarrier.baseLaws.add_neg _

/-- Projective nonresiduosity makes a zero norm equivalent to the zero
extension element. -/
theorem quadraticNorm_eq_zero_iff
    (sevenNonresidue : SevenProjectiveNonresidue)
    (value : K) :
    quadraticNorm value = 0 ↔ value = K.zero := by
  constructor
  · intro normZero
    have coefficients := sevenNonresidue value.c0 value.c1 normZero
    rcases value with ⟨real, imaginary⟩
    simpa [K.zero] using coefficients
  · rintro rfl
    rfl

/-- Base-field no-zero-divisors plus irreducibility of `u² - 7` derive the
exact cancellation property needed by the concrete extension carrier. -/
theorem extensionNoZeroDivisors_of_base_and_seven
    (baseNoZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    (sevenNonresidue : SevenProjectiveNonresidue) :
    ExtensionNoZeroDivisors := by
  intro left right productZero
  by_cases leftZero : left = K.zero
  · exact Or.inl leftZero
  · apply Or.inr
    have normNonzero : quadraticNorm left ≠ 0 := by
      intro normZero
      exact leftZero
        ((quadraticNorm_eq_zero_iff sevenNonresidue left).mp normZero)
    have scaledRightZero :
        K.mul (K.embed (quadraticNorm left)) right = K.zero := by
      calc
        K.mul (K.embed (quadraticNorm left)) right =
            K.mul (K.mul (conjugate left) left) right := by
              rw [conjugate_mul_self]
        _ = K.mul (conjugate left) (K.mul left right) :=
          ConcreteCarrier.extensionLaws.mul_assoc _ _ _
        _ = K.mul (conjugate left) K.zero := by rw [productZero]
        _ = K.zero := ConcreteCarrier.extensionLaws.mul_zero _
    have realProductZero : quadraticNorm left * right.c0 = 0 := by
      have coefficients := congrArg K.c0 scaledRightZero
      simpa [K.mul, K.embed, K.zero, Fin.mul_zero, Fin.zero_mul,
        Fin.add_zero] using coefficients
    have imaginaryProductZero : quadraticNorm left * right.c1 = 0 := by
      have coefficients := congrArg K.c1 scaledRightZero
      simpa [K.mul, K.embed, K.zero, Fin.zero_mul] using coefficients
    have realZero : right.c0 = 0 := by
      rcases baseNoZeroDivisors _ _ realProductZero with impossible | zero
      · exact False.elim (normNonzero impossible)
      · exact zero
    have imaginaryZero : right.c1 = 0 := by
      rcases baseNoZeroDivisors _ _ imaginaryProductZero with impossible | zero
      · exact False.elim (normNonzero impossible)
      · exact zero
    rcases right with ⟨real, imaginary⟩
    simpa [K.zero] using And.intro realZero imaginaryZero

end NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
