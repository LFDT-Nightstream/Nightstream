namespace SuperNeo

/-- Cyclotomic index used in the Goldilocks SuperNeo instantiation. -/
def eta : Nat := 81

/-- Cyclotomic polynomial degree: Phi_81(X) = X^54 + X^27 + 1. -/
def d : Nat := 54

/-- Field-vector length induced by ring-vector length. -/
def nF (nR : Nat) : Nat := d * nR

/-- Public-input field length induced by ring-input length. -/
def nFIn (nRIn : Nat) : Nat := d * nRIn

/-- Basic shape checks for the concrete Goldilocks setting. -/
def goldilocksShapeSanity : Bool :=
  decide (eta = 81 ∧ d = 54 ∧ nF 1 = 54 ∧ nF 2 = 108 ∧ nFIn 3 = 162)

def goldilocksShapeProp : Prop :=
  eta = 81 ∧ d = 54 ∧ nF 1 = 54 ∧ nF 2 = 108 ∧ nFIn 3 = 162

theorem goldilocksShapeSanity_sound
  (hOk : goldilocksShapeSanity = true) :
  goldilocksShapeProp := by
  unfold goldilocksShapeSanity at hOk
  simpa [goldilocksShapeProp] using (decide_eq_true_eq.mp hOk)

theorem goldilocksShape : goldilocksShapeProp := by
  unfold goldilocksShapeProp eta d nF nFIn
  decide

theorem eta_eq_81 : eta = 81 := rfl

theorem d_eq_54 : d = 54 := rfl

theorem nF_def (nR : Nat) : nF nR = d * nR := rfl

theorem nFIn_def (nRIn : Nat) : nFIn nRIn = d * nRIn := rfl

end SuperNeo
