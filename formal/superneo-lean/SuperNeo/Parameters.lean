import SuperNeo.Field
import SuperNeo.Dimensions

namespace SuperNeo
namespace Parameters
namespace Goldilocks

def modulus : Nat := SuperNeo.Goldilocks.q
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
  unfold sanityProp eta d nF b k Kmax B T modulus extDegreeK SuperNeo.eta SuperNeo.d SuperNeo.Goldilocks.q
  decide

theorem b_lt_modulus_half : b < modulus / 2 := by
  rcases concreteParameters with ⟨_, _, _, _, _, _, _, _, hb, _⟩
  exact hb

theorem B_eq_16384 : B = 16384 := by
  rcases concreteParameters with ⟨_, _, _, _, _, _, hB, _, _, _⟩
  exact hB

theorem B_def : B = b ^ k := rfl

theorem modulus_def : modulus = SuperNeo.Goldilocks.q := rfl

theorem modulus_pos : 0 < modulus := by
  simpa [modulus_def] using SuperNeo.Goldilocks.q_pos

theorem modulus_gt_one : 1 < modulus := by
  simpa [modulus_def] using SuperNeo.Goldilocks.q_gt_one

theorem eta_eq_81 : eta = 81 := by
  rcases concreteParameters with ⟨hEta, _, _, _, _, _, _, _, _, _⟩
  exact hEta

theorem d_eq_54 : d = 54 := by
  rcases concreteParameters with ⟨_, hD, _, _, _, _, _, _, _, _⟩
  exact hD

theorem nF_eq_2_pow_30 : nF = 2 ^ 30 := by
  rcases concreteParameters with ⟨_, _, hNF, _, _, _, _, _, _, _⟩
  simpa using hNF

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

theorem T_eq_216 : T = 216 := by
  rcases concreteParameters with ⟨_, _, _, _, _, _, _, hT, _, _⟩
  exact hT

theorem extDegreeK_eq_2 : extDegreeK = 2 := by
  rcases concreteParameters with ⟨_, _, _, _, _, _, _, _, _, hExt⟩
  exact hExt

theorem kappa_eq_18 : kappa = 18 := rfl

theorem kappa_pos : 0 < kappa := by
  decide

theorem b_pos : 0 < b := by
  rw [b_eq_2]
  decide

theorem b_nonzero : b ≠ 0 :=
  Nat.ne_of_gt b_pos

theorem k_pos : 0 < k := by
  rw [k_eq_14]
  decide

theorem Kmax_pos : 0 < Kmax := by
  rw [Kmax_eq_61]
  decide

theorem T_pos : 0 < T := by
  rw [T_eq_216]
  decide

theorem extDegreeK_pos : 0 < extDegreeK := by
  rw [extDegreeK_eq_2]
  decide

theorem modulus_div_two_pos : 0 < modulus / 2 := by
  -- immediate for the concrete Goldilocks modulus
  decide

theorem b_lt_modulus : b < modulus := by
  exact Nat.lt_trans b_lt_modulus_half (Nat.div_lt_self modulus_pos (by decide))

theorem B_pos : 0 < B := by
  rw [B_def]
  exact Nat.pow_pos b_pos

theorem B_nonzero : B ≠ 0 :=
  Nat.ne_of_gt B_pos

theorem B_lt_modulus : B < modulus := by
  rw [B_eq_16384, modulus_def]
  decide

end Goldilocks
end Parameters
end SuperNeo
