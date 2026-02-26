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

theorem eta_eq_81 : eta = 81 := by
  rcases concreteParameters with ⟨hEta, _, _, _, _, _, _, _, _, _⟩
  exact hEta

theorem d_eq_54 : d = 54 := by
  rcases concreteParameters with ⟨_, hD, _, _, _, _, _, _, _, _⟩
  exact hD

theorem nF_eq_1073741824 : nF = 1073741824 := by
  rcases concreteParameters with ⟨_, _, hNF, _, _, _, _, _, _, _⟩
  exact hNF

theorem b_eq_2 : b = 2 := by
  rcases concreteParameters with ⟨_, _, _, hB, _, _, _, _, _, _⟩
  exact hB

theorem k_eq_14 : k = 14 := by
  rcases concreteParameters with ⟨_, _, _, _, hK, _, _, _, _, _⟩
  exact hK

theorem Kmax_eq_61 : Kmax = 61 := by
  rcases concreteParameters with ⟨_, _, _, _, _, hKmax, _, _, _, _⟩
  exact hKmax

theorem b_lt_modulus_half : b < modulus / 2 := by
  rcases concreteParameters with ⟨_, _, _, _, _, _, _, _, hb, _⟩
  exact hb

theorem B_eq_16384 : B = 16384 := by
  rcases concreteParameters with ⟨_, _, _, _, _, _, hB, _, _, _⟩
  exact hB

theorem T_eq_216 : T = 216 := by
  rcases concreteParameters with ⟨_, _, _, _, _, _, _, hT, _, _⟩
  exact hT

theorem extDegreeK_eq_2 : extDegreeK = 2 := by
  rcases concreteParameters with ⟨_, _, _, _, _, _, _, _, _, hExt⟩
  exact hExt

theorem B_def : B = b ^ k := rfl

theorem modulus_def : modulus = q := rfl

end Goldilocks
end Parameters
end SuperNeo
