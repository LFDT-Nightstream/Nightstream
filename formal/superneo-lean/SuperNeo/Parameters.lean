import SuperNeo.Field
import SuperNeo.Dimensions

namespace SuperNeo
namespace Parameters
namespace Goldilocks

def modulus : Nat := q
def eta : Nat := SuperNeo.eta
def d : Nat := SuperNeo.d

def kappa : Nat := 18
def nF : Nat := 2 ^ 30
def b : Nat := 2
def k : Nat := 14
def Kmax : Nat := 61
def B : Nat := b ^ k
def T : Nat := 216

def cCoeffMin : Int := -2
def cCoeffMax : Int := 2
def extDegreeK : Nat := 2

/-- Appendix B.2 concrete-parameter sanity checks. -/
def sanity : Bool :=
  decide
    (eta = 81 ∧ d = 54 ∧ nF = 1073741824 ∧ b = 2 ∧ k = 14 ∧ Kmax = 61 ∧
      B = 16384 ∧ T = 216 ∧ b < modulus / 2 ∧ extDegreeK = 2)

def sanityProp : Prop :=
  eta = 81 ∧ d = 54 ∧ nF = 1073741824 ∧ b = 2 ∧ k = 14 ∧ Kmax = 61 ∧
    B = 16384 ∧ T = 216 ∧ b < modulus / 2 ∧ extDegreeK = 2

theorem sanity_sound (hOk : sanity = true) : sanityProp := by
  unfold sanity at hOk
  simpa [sanityProp] using (decide_eq_true_eq.mp hOk)

theorem concreteParameters : sanityProp := by
  unfold sanityProp eta d nF b k Kmax B T modulus extDegreeK SuperNeo.eta SuperNeo.d q
  decide

theorem b_lt_modulus_half : b < modulus / 2 := by
  rcases concreteParameters with ⟨_, _, _, _, _, _, _, _, hb, _⟩
  exact hb

theorem B_eq_16384 : B = 16384 := by
  rcases concreteParameters with ⟨_, _, _, _, _, _, hB, _, _, _⟩
  exact hB

theorem B_def : B = b ^ k := rfl

theorem modulus_def : modulus = q := rfl

end Goldilocks
end Parameters
end SuperNeo
