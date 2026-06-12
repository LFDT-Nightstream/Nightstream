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

theorem eta_pos : 0 < eta := by
  decide

theorem d_pos : 0 < d := by
  decide

theorem nF_def (nR : Nat) : nF nR = d * nR := rfl

theorem nFIn_def (nRIn : Nat) : nFIn nRIn = d * nRIn := rfl

@[simp] theorem nF_zero : nF 0 = 0 := by
  simp [nF]

@[simp] theorem nF_one : nF 1 = d := by
  simp [nF]

@[simp] theorem nF_two : nF 2 = d + d := by
  simp [nF, Nat.mul_two]

@[simp] theorem nF_add (a b : Nat) :
    nF (a + b) = nF a + nF b := by
  simp [nF, Nat.mul_add]

@[simp] theorem nF_mul (a b : Nat) :
    nF (a * b) = nF a * b := by
  simp [nF, Nat.mul_assoc]

@[simp] theorem nFIn_zero : nFIn 0 = 0 := by
  simp [nFIn]

@[simp] theorem nFIn_one : nFIn 1 = d := by
  simp [nFIn]

@[simp] theorem nFIn_add (a b : Nat) :
    nFIn (a + b) = nFIn a + nFIn b := by
  simp [nFIn, Nat.mul_add]

@[simp] theorem nFIn_mul (a b : Nat) :
    nFIn (a * b) = nFIn a * b := by
  simp [nFIn, Nat.mul_assoc]

theorem nF_eq_54_mul (nR : Nat) :
    nF nR = 54 * nR := by
  simpa [d_eq_54] using nF_def nR

theorem nFIn_eq_54_mul (nRIn : Nat) :
    nFIn nRIn = 54 * nRIn := by
  simpa [d_eq_54] using nFIn_def nRIn

theorem nF_pos_of_pos {nR : Nat} (h : 0 < nR) : 0 < nF nR := by
  simpa [nF] using Nat.mul_pos d_pos h

theorem nFIn_pos_of_pos {nRIn : Nat} (h : 0 < nRIn) : 0 < nFIn nRIn := by
  simpa [nFIn] using Nat.mul_pos d_pos h

theorem nF_mono {a b : Nat} (h : a ≤ b) : nF a ≤ nF b := by
  simpa [nF] using Nat.mul_le_mul_left d h

theorem nFIn_mono {a b : Nat} (h : a ≤ b) : nFIn a ≤ nFIn b := by
  simpa [nFIn] using Nat.mul_le_mul_left d h

end SuperNeo
