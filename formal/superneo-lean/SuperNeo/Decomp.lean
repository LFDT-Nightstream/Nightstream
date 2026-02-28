import SuperNeo.Norm

/-! Balanced digit decomposition and round-trip properties (P6). -/
namespace SuperNeo

open F

/-- Centered integer representative in [-(q-1)/2, (q-1)/2]. -/
def centeredInt (a : F) : Int :=
  if a.val <= halfQ then
    Int.ofNat a.val
  else
    Int.ofNat a.val - Int.ofNat q
private def q0Stage (a : Int) (b : Nat) : Int :=
  if a >= 0 then a / Int.ofNat b else - ((-a) / Int.ofNat b)
private def r0Stage (a : Int) (b : Nat) : Int :=
  a - q0Stage a b * Int.ofNat b
private def r1Stage (a : Int) (b : Nat) : Int :=
  let bi := Int.ofNat b
  let half := Int.ofNat (b / 2)
  let r0 := r0Stage a b
  if r0 > half then r0 - bi else r0
private def balancedResidue (a : Int) (b : Nat) : Int :=
  let bi := Int.ofNat b
  let half := Int.ofNat (b / 2)
  let r1 := r1Stage a b
  if r1 < -half then r1 + bi else r1
private theorem r0Stage_decompose (a : Int) (b : Nat) :
  a = q0Stage a b * Int.ofNat b + r0Stage a b := by
  unfold r0Stage
  have h : a - q0Stage a b * Int.ofNat b + q0Stage a b * Int.ofNat b = a := by
    simpa using (Int.sub_add_cancel a (q0Stage a b * Int.ofNat b))
  omega
private theorem r0Stage_eq_mod_of_nonneg
  (a : Int) (b : Nat)
  (ha : a >= 0) :
  r0Stage a b = a % Int.ofNat b := by
  have hq : q0Stage a b = a / Int.ofNat b := by
    simp [q0Stage, ha]
  have hdecomp : Int.ofNat b * (a / Int.ofNat b) + a % Int.ofNat b = a :=
    Int.mul_ediv_add_emod a (Int.ofNat b)
  have hdecomp' : (a / Int.ofNat b) * Int.ofNat b + a % Int.ofNat b = a := by
    simpa [Int.mul_comm] using hdecomp
  unfold r0Stage
  rw [hq]
  omega
private theorem r0Stage_eq_neg_mod_of_neg
  (a : Int) (b : Nat)
  (ha : ¬ a >= 0) :
  r0Stage a b = - ((-a) % Int.ofNat b) := by
  have hq : q0Stage a b = - ((-a) / Int.ofNat b) := by
    simp [q0Stage, ha]
  have hdecomp : Int.ofNat b * ((-a) / Int.ofNat b) + (-a) % Int.ofNat b = -a :=
    Int.mul_ediv_add_emod (-a) (Int.ofNat b)
  have hdecomp' : ((-a) / Int.ofNat b) * Int.ofNat b + (-a) % Int.ofNat b = -a := by
    simpa [Int.mul_comm] using hdecomp
  have hnegMul :
      -((-a) / Int.ofNat b) * Int.ofNat b =
        - (((-a) / Int.ofNat b) * Int.ofNat b) := by
    simpa using (Int.neg_mul ((-a) / Int.ofNat b) (Int.ofNat b))
  unfold r0Stage
  rw [hq, hnegMul]
  omega
private theorem r0Stage_range
  (a : Int) {b : Nat}
  (hb : 2 ≤ b) :
  -(Int.ofNat b) < r0Stage a b ∧ r0Stage a b < Int.ofNat b := by
  have hbPos : 0 < b := Nat.lt_of_lt_of_le (by decide : 0 < 2) hb
  have hbiNe : Int.ofNat b ≠ 0 := Int.ofNat_ne_zero.mpr (Nat.ne_of_gt hbPos)
  by_cases ha : a >= 0
  · have hEq : r0Stage a b = a % Int.ofNat b := r0Stage_eq_mod_of_nonneg a b ha
    have hLoMod : 0 ≤ a % Int.ofNat b := Int.emod_nonneg a hbiNe
    have hHiMod : a % Int.ofNat b < Int.ofNat b := by
      simpa using (Int.emod_lt a hbiNe)
    constructor
    · rw [hEq]
      omega
    · rw [hEq]
      exact hHiMod
  · have hEq : r0Stage a b = - ((-a) % Int.ofNat b) := r0Stage_eq_neg_mod_of_neg a b ha
    have hLoMod : 0 ≤ (-a) % Int.ofNat b := Int.emod_nonneg (-a) hbiNe
    have hHiMod : (-a) % Int.ofNat b < Int.ofNat b := by
      simpa using (Int.emod_lt (-a) hbiNe)
    constructor
    · rw [hEq]
      omega
    · rw [hEq]
      omega
private theorem balancedResidue_range
  (a : Int) {b : Nat}
  (hb : 2 ≤ b) :
  -Int.ofNat (b / 2) ≤ balancedResidue a b ∧
    balancedResidue a b ≤ Int.ofNat (b / 2) := by
  let bi : Int := Int.ofNat b
  let half : Int := Int.ofNat (b / 2)
  have hr0 : -bi < r0Stage a b ∧ r0Stage a b < bi := by
    simpa [bi] using r0Stage_range a hb
  have hHalfNonneg : 0 ≤ half := by
    change 0 ≤ Int.ofNat (b / 2)
    exact Int.natCast_nonneg (b / 2)
  have hTwoHalfLeBi : 2 * half ≤ bi := by
    have hNat : 2 * (b / 2) ≤ b := Nat.mul_div_le b 2
    change (2 : Int) * Int.ofNat (b / 2) ≤ Int.ofNat b
    change Int.ofNat (2 * (b / 2)) ≤ Int.ofNat b
    exact (Int.ofNat_le).2 hNat
  have hBiLeTwoHalfPlusOne : bi ≤ 2 * half + 1 := by
    have hmod : b % 2 ≤ 1 := Nat.le_of_lt_succ (Nat.mod_lt b (by decide : 0 < 2))
    have hdecomp : b = (b / 2) * 2 + b % 2 := by
      simpa [Nat.mul_comm] using (Nat.div_add_mod b 2).symm
    have hNat : b ≤ 2 * (b / 2) + 1 := by
      calc
        b = (b / 2) * 2 + b % 2 := hdecomp
        _ ≤ (b / 2) * 2 + 1 := Nat.add_le_add_left hmod _
        _ = 2 * (b / 2) + 1 := by simp [Nat.mul_comm]
    change Int.ofNat b ≤ (2 : Int) * Int.ofNat (b / 2) + 1
    change Int.ofNat b ≤ Int.ofNat (2 * (b / 2) + 1)
    exact (Int.ofNat_le).2 hNat
  have hR1Cases :
      r1Stage a b = (if r0Stage a b > half then r0Stage a b - bi else r0Stage a b) := by
    simp [r1Stage, bi, half]
  have hResCases :
      balancedResidue a b =
        (if r1Stage a b < -half then r1Stage a b + bi else r1Stage a b) := by
    simp [balancedResidue, bi, half]
  rcases hr0 with ⟨hr0Lo, hr0Hi⟩
  by_cases hUp : r0Stage a b > half
  · have hR1 : r1Stage a b = r0Stage a b - bi := by
      simpa [hUp] using hR1Cases
    by_cases hDn : r1Stage a b < -half
    · exfalso
      omega
    · have hRes : balancedResidue a b = r1Stage a b := by
        simpa [hDn] using hResCases
      have hLo : -half ≤ r1Stage a b := by
        omega
      have hHi : r1Stage a b ≤ half := by
        omega
      simpa [hRes] using And.intro hLo hHi
  · have hR1 : r1Stage a b = r0Stage a b := by
      simpa [hUp] using hR1Cases
    by_cases hDn : r1Stage a b < -half
    · have hRes : balancedResidue a b = r1Stage a b + bi := by
        simpa [hDn] using hResCases
      have hLo : -half ≤ r1Stage a b + bi := by
        omega
      have hHi : r1Stage a b + bi ≤ half := by
        omega
      simpa [hRes] using And.intro hLo hHi
    · have hRes : balancedResidue a b = r1Stage a b := by
        simpa [hDn] using hResCases
      have hLo : -half ≤ r1Stage a b := by
        omega
      have hHi : r1Stage a b ≤ half := by
        omega
      simpa [hRes] using And.intro hLo hHi

private theorem balancedResidue_divisible
  (a : Int) {b : Nat} (_hb : 2 ≤ b) :
  ∃ q : Int, a = q * Int.ofNat b + balancedResidue a b := by
  let bi : Int := Int.ofNat b
  let half : Int := Int.ofNat (b / 2)
  let q0 : Int := q0Stage a b
  let r0 : Int := r0Stage a b
  let r1 : Int := r1Stage a b
  have hA : a = q0 * bi + r0 := by
    simpa [q0, r0, bi] using r0Stage_decompose a b
  have hr1Cases : r1 = (if r0 > half then r0 - bi else r0) := by
    simp [r1, r1Stage, bi, half, r0]
  have hr1Def : r1Stage a b = r1 := by
    simp [r1]
  by_cases hUp : r0 > half
  · have hr1 : r1 = r0 - bi := by simpa [hUp] using hr1Cases
    by_cases hDn : r1 < -half
    · have hBr : balancedResidue a b = r1 + bi := by
        have hDn' : r1 < -(Int.ofNat (b / 2)) := by
          simpa [half] using hDn
        unfold balancedResidue
        rw [hr1Def]
        rw [if_pos hDn']
      refine ⟨q0, ?_⟩
      rw [hBr, hr1]
      simp [bi]
      omega
    · have hBr : balancedResidue a b = r1 := by
        have hDn' : ¬ r1 < -(Int.ofNat (b / 2)) := by
          simpa [half] using hDn
        unfold balancedResidue
        rw [hr1Def]
        rw [if_neg hDn']
      refine ⟨q0 + 1, ?_⟩
      have hMul : (q0 + 1) * bi = q0 * bi + bi := by
        simpa [Int.one_mul] using (Int.add_mul q0 1 bi)
      have hStep : a = (q0 * bi + bi) + (r0 - bi) := by
        omega
      calc
        a = (q0 * bi + bi) + (r0 - bi) := hStep
        _ = (q0 + 1) * bi + balancedResidue a b := by
              simpa [hMul, hBr, hr1]
        _ = (q0 + 1) * Int.ofNat b + balancedResidue a b := by
              simp [bi]
  · have hr1 : r1 = r0 := by simpa [hUp] using hr1Cases
    by_cases hDn : r1 < -half
    · have hBr : balancedResidue a b = r1 + bi := by
        have hDn' : r1 < -(Int.ofNat (b / 2)) := by
          simpa [half] using hDn
        unfold balancedResidue
        rw [hr1Def]
        rw [if_pos hDn']
      refine ⟨q0 - 1, ?_⟩
      have hMul : (q0 - 1) * bi = q0 * bi - bi := by
        have h : (q0 + (-1)) * bi = q0 * bi + (-1) * bi := by
          simpa using (Int.add_mul q0 (-1) bi)
        calc
          (q0 - 1) * bi = (q0 + (-1)) * bi := by rfl
          _ = q0 * bi + (-1) * bi := h
          _ = q0 * bi - bi := by simpa [Int.sub_eq_add_neg]
      have hStep : a = (q0 * bi - bi) + (r0 + bi) := by
        omega
      have hRes : balancedResidue a b = r0 + bi := by
        simpa [hBr, hr1]
      calc
        a = (q0 * bi - bi) + (r0 + bi) := hStep
        _ = ((q0 - 1) * bi) + (r0 + bi) := by
              simpa [hMul]
        _ = (q0 - 1) * Int.ofNat b + balancedResidue a b := by
              simpa [bi, hRes]
    · have hBr : balancedResidue a b = r1 := by
        have hDn' : ¬ r1 < -(Int.ofNat (b / 2)) := by
          simpa [half] using hDn
        unfold balancedResidue
        rw [hr1Def]
        rw [if_neg hDn']
      refine ⟨q0, ?_⟩
      rw [hBr, hr1]
      exact hA

/-- One-step decomposition invariant at the scalar stage. -/
private theorem balancedResidue_step_exists
  (cur : Int) {b : Nat} (hb : 2 ≤ b) :
  ∃ q : Int, cur = q * Int.ofNat b + balancedResidue cur b := by
  exact balancedResidue_divisible cur hb

/-- Divisibility form of the one-step invariant: `cur - r` is a multiple of `b`. -/
private theorem balancedResidue_step_sub_mul
  (cur : Int) {b : Nat} (hb : 2 ≤ b) :
  ∃ q : Int, cur - balancedResidue cur b = q * Int.ofNat b := by
  rcases balancedResidue_step_exists cur hb with ⟨q, hq⟩
  refine ⟨q, ?_⟩
  omega

/-- One scalar decomposition update step (internal model for loop invariants). -/
private def splitScalarStep (b : Nat) (st : Int × Array F) : Int × Array F :=
  let r := balancedResidue st.1 b
  let bi := Int.ofNat b
  let cur' :=
    if st.1 - r >= 0 then
      (st.1 - r) / bi
    else
      - ((-(st.1 - r)) / bi)
  (cur', st.2.push (F.ofInt r))

/-- Scalar split state after `k` iterations (recursive model for induction proofs). -/
private def splitScalarState (a : F) (b : Nat) : Nat → Int × Array F
  | 0 => (centeredInt a, (#[] : Array F))
  | k + 1 => splitScalarStep b (splitScalarState a b k)

private theorem splitScalarStep_snd
  (b : Nat) (st : Int × Array F) :
  (splitScalarStep b st).2 = st.2.push (F.ofInt (balancedResidue st.1 b)) := by
  simp [splitScalarStep]

private theorem splitScalarStep_fst
  (b : Nat) (st : Int × Array F) :
  (splitScalarStep b st).1 =
    (if st.1 - balancedResidue st.1 b >= 0 then
      (st.1 - balancedResidue st.1 b) / Int.ofNat b
    else
      - ((-(st.1 - balancedResidue st.1 b)) / Int.ofNat b)) := by
  simp [splitScalarStep]

private theorem splitScalarState_zero
  (a : F) (b : Nat) :
  splitScalarState a b 0 = (centeredInt a, (#[] : Array F)) := by
  rfl

private theorem splitScalarState_succ
  (a : F) (b k : Nat) :
  splitScalarState a b (k + 1) = splitScalarStep b (splitScalarState a b k) := by
  rfl

private theorem splitScalarState_succ_snd
  (a : F) (b k : Nat) :
  (splitScalarState a b (k + 1)).2 =
    (splitScalarState a b k).2.push (F.ofInt (balancedResidue (splitScalarState a b k).1 b)) := by
  simpa [splitScalarState_succ] using splitScalarStep_snd b (splitScalarState a b k)

private theorem splitScalarState_succ_fst
  (a : F) (b k : Nat) :
  (splitScalarState a b (k + 1)).1 =
    (if (splitScalarState a b k).1 - balancedResidue (splitScalarState a b k).1 b >= 0 then
      ((splitScalarState a b k).1 - balancedResidue (splitScalarState a b k).1 b) / Int.ofNat b
    else
      - ((-((splitScalarState a b k).1 - balancedResidue (splitScalarState a b k).1 b)) / Int.ofNat b)) := by
  simpa [splitScalarState_succ] using splitScalarStep_fst b (splitScalarState a b k)

private theorem splitScalarState_snd_size
  (a : F) (b k : Nat) :
  (splitScalarState a b k).2.size = k := by
  induction k with
  | zero =>
      simp [splitScalarState]
  | succ k ih =>
      simp [splitScalarState, splitScalarStep_snd, ih]

private theorem splitScalarState_step_sub_mul
  (a : F) {b k : Nat} (hb : 2 ≤ b) :
  ∃ q : Int,
    (splitScalarState a b k).1 - balancedResidue (splitScalarState a b k).1 b =
      q * Int.ofNat b := by
  exact balancedResidue_step_sub_mul (cur := (splitScalarState a b k).1) hb

private theorem splitScalarState_step_decompose
  (a : F) {b k : Nat} (hb : 2 ≤ b) :
  ∃ q : Int,
    (splitScalarState a b k).1 =
      q * Int.ofNat b + balancedResidue (splitScalarState a b k).1 b := by
  exact balancedResidue_step_exists (cur := (splitScalarState a b k).1) hb

private theorem signedDiv_of_mul
  (q : Int) {b : Nat} (hb : 0 < b) :
  (if q * Int.ofNat b >= 0 then
      (q * Int.ofNat b) / Int.ofNat b
    else
      - ((-(q * Int.ofNat b)) / Int.ofNat b)) = q := by
  let bi : Int := Int.ofNat b
  have hBiNe : bi ≠ 0 := Int.ofNat_ne_zero.mpr (Nat.ne_of_gt hb)
  by_cases hNonneg : 0 ≤ q * bi
  · have hDiv : (q * bi) / bi = q := by
      simpa [Int.mul_comm] using (Int.mul_ediv_cancel_left q hBiNe)
    calc
      (if q * bi >= 0 then (q * bi) / bi else - ((-(q * bi)) / bi))
          = (q * bi) / bi := by simp [hNonneg]
      _ = q := hDiv
  · have hDivNeg : (-(q * bi)) / bi = -q := by
      have hNegMul : -(q * bi) = bi * (-q) := by
        calc
          -(q * bi) = -(bi * q) := by
                simpa [Int.mul_comm]
          _ = bi * (-q) := by
                simpa using (Int.mul_neg bi q).symm
      calc
        (-(q * bi)) / bi = (bi * (-q)) / bi := by
              simpa [hNegMul]
        _ = -q := by
              simpa using (Int.mul_ediv_cancel_left (-q) hBiNe)
    have hNotNonneg : ¬ q * bi >= 0 := hNonneg
    calc
      (if q * bi >= 0 then (q * bi) / bi else - ((-(q * bi)) / bi))
          = - ((-(q * bi)) / bi) := by simp [hNotNonneg]
      _ = - (-q) := by simpa [hDivNeg]
      _ = q := by simp

private theorem splitScalarStep_fst_eq_of_sub_mul
  (b : Nat) (st : Int × Array F) {q : Int}
  (hb : 0 < b)
  (hSubMul : st.1 - balancedResidue st.1 b = q * Int.ofNat b) :
  (splitScalarStep b st).1 = q := by
  have hFst := splitScalarStep_fst b st
  rw [hFst]
  simpa [hSubMul] using (signedDiv_of_mul q (b := b) hb)

private theorem splitScalarState_succ_fst_eq_of_decompose
  (a : F) {b k : Nat} {q : Int}
  (hb : 0 < b)
  (hDecomp :
    (splitScalarState a b k).1 =
      q * Int.ofNat b + balancedResidue (splitScalarState a b k).1 b) :
  (splitScalarState a b (k + 1)).1 = q := by
  have hSubMul :
      (splitScalarState a b k).1 - balancedResidue (splitScalarState a b k).1 b =
        q * Int.ofNat b := by
    omega
  have hStep :
      (splitScalarStep b (splitScalarState a b k)).1 = q :=
    splitScalarStep_fst_eq_of_sub_mul b (splitScalarState a b k) (q := q) hb hSubMul
  simpa [splitScalarState_succ] using hStep

private theorem splitScalarState_succ_fst_witness
  (a : F) {b k : Nat} (hb : 2 ≤ b) :
  ∃ q : Int,
    (splitScalarState a b k).1 =
      q * Int.ofNat b + balancedResidue (splitScalarState a b k).1 b ∧
    (splitScalarState a b (k + 1)).1 = q := by
  have hbPos : 0 < b := Nat.lt_of_lt_of_le (by decide : 0 < 2) hb
  rcases splitScalarState_step_sub_mul (a := a) (b := b) (k := k) hb with ⟨q, hSubMul⟩
  refine ⟨q, ?_, ?_⟩
  · omega
  · have hDecomp :
      (splitScalarState a b k).1 =
        q * Int.ofNat b + balancedResidue (splitScalarState a b k).1 b := by
      omega
    exact splitScalarState_succ_fst_eq_of_decompose (a := a) (b := b) (k := k)
      (q := q) hbPos hDecomp

private theorem splitScalarState_step_recompose
  (a : F) {b k : Nat} (hb : 2 ≤ b) :
  (splitScalarState a b k).1 =
    (splitScalarState a b (k + 1)).1 * Int.ofNat b +
      balancedResidue (splitScalarState a b k).1 b := by
  rcases splitScalarState_succ_fst_witness (a := a) (b := b) (k := k) hb with
      ⟨q, hDecomp, hFst⟩
  calc
    (splitScalarState a b k).1
        = q * Int.ofNat b + balancedResidue (splitScalarState a b k).1 b := hDecomp
    _ = (splitScalarState a b (k + 1)).1 * Int.ofNat b +
          balancedResidue (splitScalarState a b k).1 b := by
          simpa [hFst]

private def splitScalarRecomposeInt (a : F) (b : Nat) : Nat → Int
  | 0 => 0
  | k + 1 =>
      splitScalarRecomposeInt a b k +
        (Int.ofNat b) ^ k * balancedResidue (splitScalarState a b k).1 b

private theorem splitScalarState_recomposeInt_invariant
  (a : F) {b k : Nat} (hb : 2 ≤ b) :
  centeredInt a =
    (Int.ofNat b) ^ k * (splitScalarState a b k).1 +
      splitScalarRecomposeInt a b k := by
  induction k with
  | zero =>
      simp [splitScalarState, splitScalarRecomposeInt]
  | succ k ih =>
      let bi : Int := Int.ofNat b
      let st : Int := (splitScalarState a b k).1
      let st' : Int := (splitScalarState a b (k + 1)).1
      let r : Int := balancedResidue (splitScalarState a b k).1 b
      have hStep : st = st' * bi + r := by
        simpa [st, st', bi, r] using
          (splitScalarState_step_recompose (a := a) (b := b) (k := k) hb)
      have hMulSwap : (bi ^ k * st') * bi = (bi ^ k * bi) * st' := by
        calc
          (bi ^ k * st') * bi = bi ^ k * (st' * bi) := by rw [Int.mul_assoc]
          _ = bi ^ k * (bi * st') := by rw [Int.mul_comm st' bi]
          _ = (bi ^ k * bi) * st' := by rw [Int.mul_assoc]
      have ih' : centeredInt a = bi ^ k * st + splitScalarRecomposeInt a b k := by
        simpa [bi, st] using ih
      calc
        centeredInt a = bi ^ k * st + splitScalarRecomposeInt a b k := ih'
        _ = bi ^ k * (st' * bi + r) + splitScalarRecomposeInt a b k := by
              rw [hStep]
        _ = (bi ^ k * (st' * bi) + bi ^ k * r) + splitScalarRecomposeInt a b k := by
              rw [Int.mul_add]
        _ = (((bi ^ k * st') * bi) + bi ^ k * r) + splitScalarRecomposeInt a b k := by
              rw [Int.mul_assoc]
        _ = (((bi ^ k * bi) * st') + bi ^ k * r) + splitScalarRecomposeInt a b k := by
              rw [hMulSwap]
        _ = ((bi ^ k * bi) * st') + (splitScalarRecomposeInt a b k + bi ^ k * r) := by
              simpa [Int.add_assoc, Int.add_left_comm, Int.add_comm]
        _ = bi ^ (k + 1) * st' + splitScalarRecomposeInt a b (k + 1) := by
              simp [splitScalarRecomposeInt, Int.pow_succ, bi, st', r]

private theorem splitScalarRecomposeInt_eq_range_fold
  (a : F) (b : Nat) :
  ∀ k : Nat,
    splitScalarRecomposeInt a b k =
      (List.range k).foldl
        (fun acc i =>
          acc + (Int.ofNat b) ^ i * balancedResidue (splitScalarState a b i).1 b)
        0 := by
  intro k
  induction k with
  | zero =>
      simp [splitScalarRecomposeInt]
  | succ k ih =>
      calc
        splitScalarRecomposeInt a b (k + 1)
            = splitScalarRecomposeInt a b k +
                (Int.ofNat b) ^ k * balancedResidue (splitScalarState a b k).1 b := by
                  simp [splitScalarRecomposeInt]
        _ = (List.range k).foldl
              (fun acc i =>
                acc + (Int.ofNat b) ^ i * balancedResidue (splitScalarState a b i).1 b)
              0 +
              (Int.ofNat b) ^ k * balancedResidue (splitScalarState a b k).1 b := by
                simp [ih]
        _ = (List.range (k + 1)).foldl
              (fun acc i =>
                acc + (Int.ofNat b) ^ i * balancedResidue (splitScalarState a b i).1 b)
              0 := by
                simp [List.range_succ, List.foldl_append, Int.add_assoc]

theorem splitScalarState_recomposeInt_invariant_fold
  (a : F) {b k : Nat} (hb : 2 ≤ b) :
  centeredInt a =
    (Int.ofNat b) ^ k * (splitScalarState a b k).1 +
      (List.range k).foldl
        (fun acc i =>
          acc + (Int.ofNat b) ^ i * balancedResidue (splitScalarState a b i).1 b)
        0 := by
  simpa [splitScalarRecomposeInt_eq_range_fold] using
    (splitScalarState_recomposeInt_invariant (a := a) (b := b) (k := k) hb)

theorem centeredInt_eq_residue_fold_of_splitScalarState_fst_zero
  (a : F) {b k : Nat}
  (hb : 2 ≤ b)
  (hZero : (splitScalarState a b k).1 = 0) :
  centeredInt a =
    (List.range k).foldl
      (fun acc i =>
        acc + (Int.ofNat b) ^ i * balancedResidue (splitScalarState a b i).1 b)
      0 := by
  have hInv := splitScalarState_recomposeInt_invariant_fold (a := a) (b := b) (k := k) hb
  simpa [hZero] using hInv

/-- Int-level residue fold used by theorem-native scalar reconstruction proofs. -/
def splitScalarResidueFoldInt (a : F) (b k : Nat) : Int :=
  (List.range k).foldl
    (fun acc i =>
      acc + (Int.ofNat b) ^ i * balancedResidue (splitScalarState a b i).1 b)
    0

theorem centeredInt_eq_splitScalarResidueFoldInt_of_splitScalarState_fst_zero
  (a : F) {b k : Nat}
  (hb : 2 ≤ b)
  (hZero : (splitScalarState a b k).1 = 0) :
  centeredInt a = splitScalarResidueFoldInt a b k := by
  simpa [splitScalarResidueFoldInt] using
    (centeredInt_eq_residue_fold_of_splitScalarState_fst_zero
      (a := a) (b := b) (k := k) hb hZero)

theorem splitScalarState_fst_zero_of_centeredInt_eq_splitScalarResidueFoldInt
  (a : F) {b k : Nat} (hb : 2 ≤ b)
  (hEq : centeredInt a = splitScalarResidueFoldInt a b k) : (splitScalarState a b k).1 = 0 := by
  have hInv := splitScalarState_recomposeInt_invariant_fold (a := a) (b := b) (k := k) hb
  have hInv' : centeredInt a = (Int.ofNat b) ^ k * (splitScalarState a b k).1 + splitScalarResidueFoldInt a b k := by
    simpa [splitScalarResidueFoldInt] using hInv
  have hMulZero : (Int.ofNat b) ^ k * (splitScalarState a b k).1 = 0 := by omega
  have hBasePos : 0 < (Int.ofNat b) := by exact Int.natCast_pos.mpr (Nat.lt_of_lt_of_le (by decide : 0 < 2) hb)
  have hPowPos : 0 < (Int.ofNat b) ^ k := by exact Int.pow_pos hBasePos
  exact (Int.mul_eq_zero.mp hMulZero).resolve_left (Int.ne_of_gt hPowPos)
theorem centeredInt_eq_splitScalarResidueFoldInt_base2_succ_of_eq
  (a : F) {k : Nat}
  (hEq : centeredInt a = splitScalarResidueFoldInt a 2 k) :
  centeredInt a = splitScalarResidueFoldInt a 2 (k + 1) := by
  have hZero : (splitScalarState a 2 k).1 = 0 :=
    splitScalarState_fst_zero_of_centeredInt_eq_splitScalarResidueFoldInt
      (a := a) (b := 2) (k := k) (hb := by decide) hEq
  have hFold : splitScalarResidueFoldInt a 2 (k + 1) = splitScalarResidueFoldInt a 2 k := by
    have hBr0 : balancedResidue 0 2 = 0 := by native_decide
    simp [splitScalarResidueFoldInt, List.range_succ, List.foldl_append, hZero, hBr0]
  calc
    centeredInt a = splitScalarResidueFoldInt a 2 k := hEq
    _ = splitScalarResidueFoldInt a 2 (k + 1) := hFold.symm
/-- Balanced base-b split of one field element into k digits. -/
def splitBalancedScalar (a : F) (b k : Nat) : Array F :=
  if b < 2 then
    Array.replicate k (0 : F)
  else
    (splitScalarState a b k).2

theorem splitBalancedScalar_eq_state_snd_of_base_ge_two
  {a : F} {b k : Nat}
  (hb : 2 ≤ b) :
  splitBalancedScalar a b k = (splitScalarState a b k).2 := by
  unfold splitBalancedScalar
  have hbNotLt : ¬ b < 2 := Nat.not_lt.mpr hb
  simp [hbNotLt]

theorem splitBalancedScalar_size (a : F) (b k : Nat) :
  (splitBalancedScalar a b k).size = k := by
  unfold splitBalancedScalar
  by_cases hb : b < 2
  · simp [hb]
  · simp [hb, splitScalarState_snd_size]

theorem splitBalancedScalar_size_of_base_ge_two
  {a : F} {b k : Nat}
  (hb : 2 ≤ b) :
  (splitBalancedScalar a b k).size = k := by
  simpa [splitBalancedScalar_eq_state_snd_of_base_ge_two hb] using
    (splitScalarState_snd_size a b k)

private theorem splitScalarState_digit_get
  (a : F) (b : Nat) :
  ∀ k i (hi : i < k),
    ((splitScalarState a b k).2)[i]'(by
      simpa [splitScalarState_snd_size] using hi) =
      F.ofInt (balancedResidue (splitScalarState a b i).1 b) := by
  intro k
  induction k with
  | zero =>
      intro i hi
      exact (Nat.not_lt_zero i hi).elim
  | succ k ih =>
      intro i hi
      by_cases hEq : i = k
      · subst i
        have hSize : (splitScalarState a b k).2.size = k :=
          splitScalarState_snd_size a b k
        have hLast :
            ((splitScalarState a b k).2.push
              (F.ofInt (balancedResidue (splitScalarState a b k).1 b)))[
                (splitScalarState a b k).2.size] =
            F.ofInt (balancedResidue (splitScalarState a b k).1 b) := by
          simpa using
            (Array.getElem_push_eq
              (xs := (splitScalarState a b k).2)
              (x := F.ofInt (balancedResidue (splitScalarState a b k).1 b)))
        simpa [splitScalarState_succ_snd, hSize] using hLast
      · have hik : i < k := by omega
        have hSize : (splitScalarState a b k).2.size = k :=
          splitScalarState_snd_size a b k
        have hLtSize : i < (splitScalarState a b k).2.size := by
          simpa [hSize] using hik
        have hGetLt :
            ((splitScalarState a b k).2.push
              (F.ofInt (balancedResidue (splitScalarState a b k).1 b)))[i]'(by
                simpa [hSize] using hi) =
            ((splitScalarState a b k).2)[i]'hLtSize := by
          simpa using
            (Array.getElem_push_lt
              (xs := (splitScalarState a b k).2)
              (x := F.ofInt (balancedResidue (splitScalarState a b k).1 b))
              (h := hLtSize))
        calc
          ((splitScalarState a b (k + 1)).2)[i]'(by
            simpa [splitScalarState_snd_size] using hi)
              = ((splitScalarState a b k).2)[i]'hLtSize := by
                  simpa [splitScalarState_succ_snd] using hGetLt
          _ = F.ofInt (balancedResidue (splitScalarState a b i).1 b) := by
                exact ih i hik

theorem splitBalancedScalar_digit_get_of_base_ge_two
  {a : F} {b k i : Nat}
  (hb : 2 ≤ b)
  (hi : i < k) :
  (splitBalancedScalar a b k)[i]'(by
      simpa [splitBalancedScalar_size_of_base_ge_two hb] using hi) =
    F.ofInt (balancedResidue (splitScalarState a b i).1 b) := by
  simpa [splitBalancedScalar_eq_state_snd_of_base_ge_two hb] using
    (splitScalarState_digit_get a b k i hi)

theorem splitBalancedScalar_digit_bounded_witness_of_base_ge_two
  {a : F} {b k i : Nat}
  (hb : 2 ≤ b)
  (hi : i < k) :
  ∃ r : Int,
    -Int.ofNat (b / 2) ≤ r ∧
    r ≤ Int.ofNat (b / 2) ∧
    (splitBalancedScalar a b k)[i]'(by
      simpa [splitBalancedScalar_size_of_base_ge_two hb] using hi) = F.ofInt r := by
  refine ⟨balancedResidue (splitScalarState a b i).1 b, ?_, ?_, ?_⟩
  · exact (balancedResidue_range (a := (splitScalarState a b i).1) (b := b) hb).1
  · exact (balancedResidue_range (a := (splitScalarState a b i).1) (b := b) hb).2
  · exact splitBalancedScalar_digit_get_of_base_ge_two (a := a) (b := b) (k := k) (i := i) hb hi

private theorem normInfF_lt_two_of_int_between_neg_one_and_one
  {r : Int}
  (hLo : (-1 : Int) ≤ r)
  (hHi : r ≤ 1) :
  normInfF (F.ofInt r) < 2 := by
  have hCases : r = -1 ∨ r = 0 ∨ r = 1 := by
    omega
  rcases hCases with hNeg | hRest
  · subst r
    native_decide
  · rcases hRest with hZero | hOne
    · subst r
      native_decide
    · subst r
      native_decide

theorem splitBalancedScalar_digit_bound_of_base_two
  {a : F} {k i : Nat}
  (hi : i < k) :
  normInfF ((splitBalancedScalar a 2 k)[i]'(by
      simpa [splitBalancedScalar_size_of_base_ge_two (by decide : 2 ≤ 2)] using hi)) < 2 := by
  rcases splitBalancedScalar_digit_bounded_witness_of_base_ge_two
      (a := a) (b := 2) (k := k) (i := i) (by decide : 2 ≤ 2) hi with
      ⟨r, hLo, hHi, hEq⟩
  have hLo' : (-1 : Int) ≤ r := by
    simpa using hLo
  have hHi' : r ≤ (1 : Int) := by
    simpa using hHi
  have hNorm : normInfF (F.ofInt r) < 2 :=
    normInfF_lt_two_of_int_between_neg_one_and_one hLo' hHi'
  simpa [hEq] using hNorm

theorem splitBalancedScalar_digitsWithinBaseProp_of_base_two
  {a : F} {k : Nat} :
  ∀ i (hi : i < (splitBalancedScalar a 2 k).size),
    normInfF ((splitBalancedScalar a 2 k)[i]'hi) < 2 := by
  intro i hi
  have hik : i < k := by
    simpa [splitBalancedScalar_size_of_base_ge_two (by decide : 2 ≤ 2)] using hi
  simpa [splitBalancedScalar_size_of_base_ge_two (by decide : 2 ≤ 2)] using
    (splitBalancedScalar_digit_bound_of_base_two (a := a) (k := k) (i := i) hik)

theorem splitBalancedScalar_digitsWithinBase_of_base_two
  {a : F} {k : Nat} :
  (splitBalancedScalar a 2 k).all (fun x => normInfF x < 2) = true := by
  apply (Array.all_eq_true).2
  intro i hi
  exact decide_eq_true (splitBalancedScalar_digitsWithinBaseProp_of_base_two (a := a) (k := k) i hi)

theorem splitBalancedScalar_get_zero_of_base_lt_two
  {a : F} {b k i : Nat}
  (hb : b < 2)
  (hi : i < (splitBalancedScalar a b k).size) :
  (splitBalancedScalar a b k)[i]'hi = 0 := by
  simp [splitBalancedScalar, hb] at hi ⊢

/-- Balanced base-b split of a vector into k digit-vectors. -/
def splitBalancedVec (z : Array F) (b k : Nat) : Array (Array F) :=
  if b < 2 then
    Array.replicate k (Array.replicate z.size (0 : F))
  else
    Array.ofFn (fun i : Fin k =>
      Array.ofFn (fun j : Fin z.size =>
        (splitBalancedScalar z[j.1]! b k)[i.1]!))

theorem splitBalancedVec_get_zero_of_base_lt_two
  {z : Array F} {b k i j : Nat}
  (hb : b < 2)
  (hi : i < (splitBalancedVec z b k).size)
  (hj : j < ((splitBalancedVec z b k)[i]'hi).size) :
  ((splitBalancedVec z b k)[i]'hi)[j]'hj = 0 := by
  simp [splitBalancedVec, hb] at hi hj ⊢

/-- Structural size invariant: vector split always produces exactly `k` digit rows. -/
theorem splitBalancedVec_size (z : Array F) (b k : Nat) :
  (splitBalancedVec z b k).size = k := by
  unfold splitBalancedVec
  by_cases hb : b < 2
  · simp [hb]
  · simp [hb]

theorem splitBalancedVec_isEmpty_eq_false_of_k_pos
  {z : Array F} {b k : Nat}
  (hK : 0 < k) :
  (splitBalancedVec z b k).isEmpty = false := by
  have hSize : (splitBalancedVec z b k).size = k := splitBalancedVec_size z b k
  have hPos : 0 < (splitBalancedVec z b k).size := by
    simpa [hSize] using hK
  unfold Array.isEmpty
  have hNe : (splitBalancedVec z b k).size ≠ 0 := Nat.ne_of_gt hPos
  simp [hNe]

/-- Structural row-width invariant: each split digit row has width `z.size`. -/
theorem splitBalancedVec_row_size
  {z : Array F} {b k i : Nat}
  (hi : i < (splitBalancedVec z b k).size) :
  ((splitBalancedVec z b k)[i]'hi).size = z.size := by
  unfold splitBalancedVec at hi ⊢
  by_cases hb : b < 2
  · simp [hb] at hi ⊢
  · simp [hb] at hi ⊢

theorem splitBalancedVec_entry_eq_splitBalancedScalar_of_base_ge_two
  {z : Array F} {b k i j : Nat}
  (hb : 2 ≤ b)
  (hi : i < (splitBalancedVec z b k).size)
  (hj : j < ((splitBalancedVec z b k)[i]'hi).size) :
  ((splitBalancedVec z b k)[i]'hi)[j]'hj = (splitBalancedScalar z[j]! b k)[i]! := by
  have hbNotLt : ¬ b < 2 := Nat.not_lt.mpr hb
  have hiK : i < k := by
    simpa [splitBalancedVec, hbNotLt] using hi
  have hjZ : j < z.size := by
    simpa [splitBalancedVec_row_size (z := z) (b := b) (k := k) (i := i) hi] using hj
  simp [splitBalancedVec, hbNotLt, hjZ]

theorem splitBalancedVec_entry_eq_splitBalancedScalar_of_base_ge_two'
  {z : Array F} {b k i j : Nat}
  (hb : 2 ≤ b)
  (hiK : i < k)
  (hjZ : j < z.size) :
  ((splitBalancedVec z b k)[i]'(by
      simpa [splitBalancedVec, Nat.not_lt.mpr hb] using hiK))[j]'(by
      simpa [splitBalancedVec, Nat.not_lt.mpr hb] using hjZ) =
    (splitBalancedScalar z[j]! b k)[i]! := by
  simp [splitBalancedVec, Nat.not_lt.mpr hb, hjZ]

theorem splitBalancedVec_entry_bounded_witness_of_base_ge_two
  {z : Array F} {b k i j : Nat}
  (hb : 2 ≤ b)
  (hi : i < (splitBalancedVec z b k).size)
  (hj : j < ((splitBalancedVec z b k)[i]'hi).size) :
  ∃ r : Int,
    -Int.ofNat (b / 2) ≤ r ∧
    r ≤ Int.ofNat (b / 2) ∧
    ((splitBalancedVec z b k)[i]'hi)[j]'hj = F.ofInt r := by
  have hiK : i < k := by
    simpa [splitBalancedVec, Nat.not_lt.mpr hb] using hi
  have hjZ : j < z.size := by
    simpa [splitBalancedVec_row_size (z := z) (b := b) (k := k) (i := i) hi] using hj
  have hEntry :
      ((splitBalancedVec z b k)[i]'hi)[j]'hj = (splitBalancedScalar z[j]! b k)[i]! := by
    simpa using
      (splitBalancedVec_entry_eq_splitBalancedScalar_of_base_ge_two'
        (z := z) (b := b) (k := k) (i := i) (j := j) hb hiK hjZ)
  rcases splitBalancedScalar_digit_bounded_witness_of_base_ge_two
      (a := z[j]!) (b := b) (k := k) (i := i) hb hiK with
      ⟨r, hLo, hHi, hEq⟩
  have hEqBang : (splitBalancedScalar z[j]! b k)[i]! = F.ofInt r := by
    simpa [splitBalancedScalar_size_of_base_ge_two hb, hiK] using hEq
  refine ⟨r, hLo, hHi, ?_⟩
  simpa [hEntry] using hEqBang

theorem splitBalancedVec_digit_bound_of_base_two
  {z : Array F} {k i j : Nat}
  (hi : i < (splitBalancedVec z 2 k).size)
  (hj : j < ((splitBalancedVec z 2 k)[i]'hi).size) :
  normInfF (((splitBalancedVec z 2 k)[i]'hi)[j]'hj) < 2 := by
  have hiK : i < k := by
    simpa [splitBalancedVec, (show ¬ (2 < 2) by decide)] using hi
  have hjZ : j < z.size := by
    simpa [splitBalancedVec_row_size (z := z) (b := 2) (k := k) (i := i) hi] using hj
  have hEntry :
      ((splitBalancedVec z 2 k)[i]'hi)[j]'hj =
        (splitBalancedScalar z[j]! 2 k)[i]! := by
    simpa using
      (splitBalancedVec_entry_eq_splitBalancedScalar_of_base_ge_two'
        (z := z) (b := 2) (k := k) (i := i) (j := j)
        (hb := by decide) hiK hjZ)
  have hScalar :
      normInfF ((splitBalancedScalar z[j]! 2 k)[i]'(by
        simpa [splitBalancedScalar_size_of_base_ge_two (by decide : 2 ≤ 2)] using hiK)) < 2 :=
    splitBalancedScalar_digit_bound_of_base_two (a := z[j]!) (k := k) (i := i) hiK
  have hScalarBang : normInfF ((splitBalancedScalar z[j]! 2 k)[i]!) < 2 := by
    simpa [splitBalancedScalar_size_of_base_ge_two (by decide : 2 ≤ 2), hiK] using hScalar
  simpa [hEntry] using hScalarBang

/-- Recompose z = Σ b^i z_i from split digits. -/
private def powF (x : F) : Nat → F
  | 0 => 1
  | n + 1 => powF x n * x

/-- Field-level residue fold aligned with `splitScalarResidueFoldInt`. -/
def splitScalarResidueFoldF (a : F) (b k : Nat) : F :=
  (List.range k).foldl
    (fun acc i =>
      acc + powF (F.ofNat b) i * F.ofInt (balancedResidue (splitScalarState a b i).1 b))
    0

/--
Assumption bundle for transporting the scalar decomposition identity from `Int`
into `F` through `F.ofInt`.
-/
def p6OfIntSemiringAssumption : Prop :=
  (∀ x y : Int, F.ofInt (x + y) = F.ofInt x + F.ofInt y) ∧
  (∀ x y : Int, F.ofInt (x * y) = F.ofInt x * F.ofInt y) ∧
  (∀ n : Nat, F.ofInt (Int.ofNat n) = F.ofNat n)

private theorem powF_eq_ofInt_pow_of_p6OfIntSemiringAssumption
  (hOfInt : p6OfIntSemiringAssumption) (b : Nat) :
  ∀ i : Nat, powF (F.ofNat b) i = F.ofInt ((Int.ofNat b) ^ i) := by
  intro i
  induction i with
  | zero =>
      exact (hOfInt.2.2 1).symm
  | succ i ih =>
      have hBase : F.ofNat b = F.ofInt (Int.ofNat b) := (hOfInt.2.2 b).symm
      calc
        powF (F.ofNat b) (i + 1)
            = powF (F.ofNat b) i * F.ofNat b := by
                simp [powF]
        _ = F.ofInt ((Int.ofNat b) ^ i) * F.ofInt (Int.ofNat b) := by
              rw [ih, hBase]
        _ = F.ofInt (((Int.ofNat b) ^ i) * Int.ofNat b) := by
              symm
              exact hOfInt.2.1 ((Int.ofNat b) ^ i) (Int.ofNat b)
        _ = F.ofInt ((Int.ofNat b) ^ (i + 1)) := by
              simp [Int.pow_succ]

theorem splitScalarResidueFoldF_eq_ofInt_splitScalarResidueFoldInt_of_p6OfIntSemiringAssumption
  (hOfInt : p6OfIntSemiringAssumption) (a : F) (b k : Nat) :
  splitScalarResidueFoldF a b k = F.ofInt (splitScalarResidueFoldInt a b k) := by
  induction k with
  | zero =>
      simpa [splitScalarResidueFoldF, splitScalarResidueFoldInt] using
        (hOfInt.2.2 0).symm
  | succ k ih =>
      have hPow :
          powF (F.ofNat b) k = F.ofInt ((Int.ofNat b) ^ k) :=
        powF_eq_ofInt_pow_of_p6OfIntSemiringAssumption hOfInt b k
      have hMul :
          F.ofInt (((Int.ofNat b) ^ k) * balancedResidue (splitScalarState a b k).1 b) =
            F.ofInt ((Int.ofNat b) ^ k) *
              F.ofInt (balancedResidue (splitScalarState a b k).1 b) :=
        hOfInt.2.1 ((Int.ofNat b) ^ k) (balancedResidue (splitScalarState a b k).1 b)
      have hAdd :
          F.ofInt
              (splitScalarResidueFoldInt a b k +
                ((Int.ofNat b) ^ k) * balancedResidue (splitScalarState a b k).1 b) =
            F.ofInt (splitScalarResidueFoldInt a b k) +
              F.ofInt (((Int.ofNat b) ^ k) * balancedResidue (splitScalarState a b k).1 b) :=
        hOfInt.1 (splitScalarResidueFoldInt a b k)
          (((Int.ofNat b) ^ k) * balancedResidue (splitScalarState a b k).1 b)
      calc
        splitScalarResidueFoldF a b (k + 1)
            = splitScalarResidueFoldF a b k +
                powF (F.ofNat b) k *
                  F.ofInt (balancedResidue (splitScalarState a b k).1 b) := by
                simp [splitScalarResidueFoldF, List.range_succ, List.foldl_append]
        _ = F.ofInt (splitScalarResidueFoldInt a b k) +
              F.ofInt ((Int.ofNat b) ^ k) *
                F.ofInt (balancedResidue (splitScalarState a b k).1 b) := by
              simp [ih, hPow]
        _ = F.ofInt (splitScalarResidueFoldInt a b k) +
              F.ofInt (((Int.ofNat b) ^ k) * balancedResidue (splitScalarState a b k).1 b) := by
              rw [← hMul]
        _ = F.ofInt
              (splitScalarResidueFoldInt a b k +
                ((Int.ofNat b) ^ k) * balancedResidue (splitScalarState a b k).1 b) := by
              simpa using hAdd.symm
        _ = F.ofInt (splitScalarResidueFoldInt a b (k + 1)) := by
              simp [splitScalarResidueFoldInt, List.range_succ, List.foldl_append]

def recomposeSplitDigits (digits : Array (Array F)) (b : Nat) : Array F :=
  if digits.isEmpty then
    #[]
  else
    let m := (digits[0]!).size
    Array.ofFn (fun j : Fin m =>
      (List.range digits.size).foldl
        (fun acc i => acc + powF (F.ofNat b) i * (digits[i]![j.1]!))
        (0 : F))

def digitsWithinBase (digits : Array (Array F)) (b : Nat) : Bool :=
  digits.all (fun row => row.all (fun x => normInfF x < b))

def digitsWithinBaseProp (digits : Array (Array F)) (b : Nat) : Prop :=
  ∀ i (hi : i < digits.size) j (hj : j < (digits[i]'hi).size),
    normInfF ((digits[i]'hi)[j]'hj) < b

theorem recomposeSplitDigits_get_of_nonempty
  {digits : Array (Array F)} {b j : Nat}
  (hNE : digits.isEmpty = false)
  (hj : j < (digits[0]!).size) :
  (recomposeSplitDigits digits b)[j]'(by
      simpa [recomposeSplitDigits, hNE] using hj) =
      (List.range digits.size).foldl
      (fun acc i => acc + powF (F.ofNat b) i * (digits[i]![j]!))
      (0 : F) := by
  simp [recomposeSplitDigits, hNE]

theorem digitsWithinBase_sound
  {digits : Array (Array F)} {b : Nat}
  (hOk : digitsWithinBase digits b = true) :
  digitsWithinBaseProp digits b := by
  intro i hi j hj
  have hRow :
      (digits[i]'hi).all (fun x => normInfF x < b) = true :=
    (Array.all_eq_true.mp hOk) i hi
  exact decide_eq_true_eq.mp ((Array.all_eq_true.mp hRow) j hj)

theorem digitsWithinBase_complete
  {digits : Array (Array F)} {b : Nat}
  (hProp : digitsWithinBaseProp digits b) :
  digitsWithinBase digits b = true := by
  apply (Array.all_eq_true).2
  intro i hi
  apply (Array.all_eq_true).2
  intro j hj
  exact decide_eq_true (hProp i hi j hj)

theorem digitsWithinBase_iff_prop
  {digits : Array (Array F)} {b : Nat} :
  digitsWithinBase digits b = true ↔ digitsWithinBaseProp digits b := by
  constructor
  · exact digitsWithinBase_sound
  · exact digitsWithinBase_complete

theorem splitBalancedVec_digitsWithinBaseProp_of_base_two
  {z : Array F} {k : Nat} :
  digitsWithinBaseProp (splitBalancedVec z 2 k) 2 := by
  intro i hi j hj
  exact splitBalancedVec_digit_bound_of_base_two (z := z) (k := k) (i := i) (j := j) hi hj

theorem splitBalancedVec_digitsWithinBase_of_base_two
  {z : Array F} {k : Nat} :
  digitsWithinBase (splitBalancedVec z 2 k) 2 = true := by
  exact digitsWithinBase_complete (splitBalancedVec_digitsWithinBaseProp_of_base_two (z := z) (k := k))

theorem recomposeSplitDigits_splitBalancedVec_get_eq_fold_of_base_ge_two
  {z : Array F} {b k j : Nat}
  (_hb : 2 ≤ b)
  (hk : 0 < k)
  (hj : j < z.size) :
  (recomposeSplitDigits (splitBalancedVec z b k) b)[j]'(by
      have hNE : (splitBalancedVec z b k).isEmpty = false :=
        splitBalancedVec_isEmpty_eq_false_of_k_pos (z := z) (b := b) (k := k) hk
      have h0 : 0 < (splitBalancedVec z b k).size := by
        simpa [splitBalancedVec_size (z := z) (b := b) (k := k)] using hk
      have hRow0 : ((splitBalancedVec z b k)[0]!).size = z.size := by
        simpa [h0] using
          (splitBalancedVec_row_size (z := z) (b := b) (k := k) (i := 0) h0)
      simpa [recomposeSplitDigits, hNE, hRow0] using hj) =
    (List.range (splitBalancedVec z b k).size).foldl
      (fun acc i => acc + powF (F.ofNat b) i * (((splitBalancedVec z b k)[i]!)[j]!))
      (0 : F) := by
  have hNE : (splitBalancedVec z b k).isEmpty = false :=
    splitBalancedVec_isEmpty_eq_false_of_k_pos (z := z) (b := b) (k := k) hk
  have h0 : 0 < (splitBalancedVec z b k).size := by
    simpa [splitBalancedVec_size (z := z) (b := b) (k := k)] using hk
  have hRow0 : ((splitBalancedVec z b k)[0]!).size = z.size := by
    simpa [h0] using
      (splitBalancedVec_row_size (z := z) (b := b) (k := k) (i := 0) h0)
  simpa [hRow0] using
    (recomposeSplitDigits_get_of_nonempty
      (digits := splitBalancedVec z b k) (b := b) (j := j) hNE
      (by simpa [hRow0] using hj))

private theorem foldl_add_term_eq_of_term_eq
  (l : List Nat)
  (f g : Nat → F)
  (hEq : ∀ i, i ∈ l → f i = g i) :
  ∀ init : F,
    l.foldl (fun acc i => acc + f i) init =
      l.foldl (fun acc i => acc + g i) init := by
  intro init
  induction l generalizing init with
  | nil =>
      simp
  | cons a t ih =>
      have hHead : f a = g a := hEq a (by simp)
      have hTailEq : ∀ i, i ∈ t → f i = g i := by
        intro i hi
        exact hEq i (by simp [hi])
      simpa [hHead] using (ih hTailEq (init := init + g a))

theorem recomposeSplitDigits_splitBalancedVec_get_eq_scalar_fold_of_base_ge_two
  {z : Array F} {b k j : Nat}
  (hb : 2 ≤ b)
  (hk : 0 < k)
  (hj : j < z.size) :
  (recomposeSplitDigits (splitBalancedVec z b k) b)[j]'(by
      have hNE : (splitBalancedVec z b k).isEmpty = false :=
        splitBalancedVec_isEmpty_eq_false_of_k_pos (z := z) (b := b) (k := k) hk
      have h0 : 0 < (splitBalancedVec z b k).size := by
        simpa [splitBalancedVec_size (z := z) (b := b) (k := k)] using hk
      have hRow0 : ((splitBalancedVec z b k)[0]!).size = z.size := by
        simpa [h0] using
          (splitBalancedVec_row_size (z := z) (b := b) (k := k) (i := 0) h0)
      simpa [recomposeSplitDigits, hNE, hRow0] using hj) =
    (List.range k).foldl
      (fun acc i => acc + powF (F.ofNat b) i * ((splitBalancedScalar z[j]! b k)[i]!))
      (0 : F) := by
  have hFoldVec :
      (recomposeSplitDigits (splitBalancedVec z b k) b)[j]'(by
          have hNE : (splitBalancedVec z b k).isEmpty = false :=
            splitBalancedVec_isEmpty_eq_false_of_k_pos (z := z) (b := b) (k := k) hk
          have h0 : 0 < (splitBalancedVec z b k).size := by
            simpa [splitBalancedVec_size (z := z) (b := b) (k := k)] using hk
          have hRow0 : ((splitBalancedVec z b k)[0]!).size = z.size := by
            simpa [h0] using
              (splitBalancedVec_row_size (z := z) (b := b) (k := k) (i := 0) h0)
          simpa [recomposeSplitDigits, hNE, hRow0] using hj) =
        (List.range (splitBalancedVec z b k).size).foldl
          (fun acc i => acc + powF (F.ofNat b) i * (((splitBalancedVec z b k)[i]!)[j]!))
          (0 : F) :=
    recomposeSplitDigits_splitBalancedVec_get_eq_fold_of_base_ge_two
      (z := z) (b := b) (k := k) (j := j) hb hk hj
  have hFoldVec' :
      (recomposeSplitDigits (splitBalancedVec z b k) b)[j]'(by
          have hNE : (splitBalancedVec z b k).isEmpty = false :=
            splitBalancedVec_isEmpty_eq_false_of_k_pos (z := z) (b := b) (k := k) hk
          have h0 : 0 < (splitBalancedVec z b k).size := by
            simpa [splitBalancedVec_size (z := z) (b := b) (k := k)] using hk
          have hRow0 : ((splitBalancedVec z b k)[0]!).size = z.size := by
            simpa [h0] using
              (splitBalancedVec_row_size (z := z) (b := b) (k := k) (i := 0) h0)
          simpa [recomposeSplitDigits, hNE, hRow0] using hj) =
        (List.range k).foldl
          (fun acc i => acc + powF (F.ofNat b) i * (((splitBalancedVec z b k)[i]!)[j]!))
          (0 : F) := by
    simpa [splitBalancedVec_size (z := z) (b := b) (k := k)] using hFoldVec
  have hRewrite :
      (List.range k).foldl
        (fun acc i => acc + powF (F.ofNat b) i * (((splitBalancedVec z b k)[i]!)[j]!))
        (0 : F) =
      (List.range k).foldl
        (fun acc i => acc + powF (F.ofNat b) i * ((splitBalancedScalar z[j]! b k)[i]!))
        (0 : F) := by
    apply foldl_add_term_eq_of_term_eq (l := List.range k)
    intro i hiMem
    have hiK : i < k := List.mem_range.mp hiMem
    have hEntry :
        (((splitBalancedVec z b k)[i]!)[j]!) =
          (splitBalancedScalar z[j]! b k)[i]! := by
      simpa [splitBalancedVec, Nat.not_lt.mpr hb, hiK, hj] using
        (splitBalancedVec_entry_eq_splitBalancedScalar_of_base_ge_two'
          (z := z) (b := b) (k := k) (i := i) (j := j) hb hiK hj)
    simpa [hEntry]
  exact hFoldVec'.trans hRewrite

theorem recomposeSplitDigits_splitBalancedVec_get_eq_residue_fold_of_base_ge_two
  {z : Array F} {b k j : Nat}
  (hb : 2 ≤ b)
  (hk : 0 < k)
  (hj : j < z.size) :
  (recomposeSplitDigits (splitBalancedVec z b k) b)[j]'(by
      have hNE : (splitBalancedVec z b k).isEmpty = false :=
        splitBalancedVec_isEmpty_eq_false_of_k_pos (z := z) (b := b) (k := k) hk
      have h0 : 0 < (splitBalancedVec z b k).size := by
        simpa [splitBalancedVec_size (z := z) (b := b) (k := k)] using hk
      have hRow0 : ((splitBalancedVec z b k)[0]!).size = z.size := by
        simpa [h0] using
          (splitBalancedVec_row_size (z := z) (b := b) (k := k) (i := 0) h0)
      simpa [recomposeSplitDigits, hNE, hRow0] using hj) =
    (List.range k).foldl
      (fun acc i =>
        acc + powF (F.ofNat b) i * F.ofInt (balancedResidue (splitScalarState z[j]! b i).1 b))
      (0 : F) := by
  have hScalarFold :=
    recomposeSplitDigits_splitBalancedVec_get_eq_scalar_fold_of_base_ge_two
      (z := z) (b := b) (k := k) (j := j) hb hk hj
  have hRewrite :
      (List.range k).foldl
        (fun acc i => acc + powF (F.ofNat b) i * ((splitBalancedScalar z[j]! b k)[i]!))
        (0 : F) =
      (List.range k).foldl
        (fun acc i =>
          acc + powF (F.ofNat b) i * F.ofInt (balancedResidue (splitScalarState z[j]! b i).1 b))
        (0 : F) := by
    apply foldl_add_term_eq_of_term_eq (l := List.range k)
    intro i hiMem
    have hiK : i < k := List.mem_range.mp hiMem
    have hDigit :
        (splitBalancedScalar z[j]! b k)[i]! =
          F.ofInt (balancedResidue (splitScalarState z[j]! b i).1 b) := by
      simpa [splitBalancedScalar_size_of_base_ge_two hb, hiK] using
        (splitBalancedScalar_digit_get_of_base_ge_two
          (a := z[j]!) (b := b) (k := k) (i := i) hb hiK)
    simpa [hDigit]
  exact hScalarFold.trans hRewrite

theorem recomposeSplitDigits_splitBalancedVec_get_eq_splitScalarResidueFoldF_of_base_ge_two
  {z : Array F} {b k j : Nat}
  (hb : 2 ≤ b)
  (hk : 0 < k)
  (hj : j < z.size) :
  (recomposeSplitDigits (splitBalancedVec z b k) b)[j]'(by
      have hNE : (splitBalancedVec z b k).isEmpty = false :=
        splitBalancedVec_isEmpty_eq_false_of_k_pos (z := z) (b := b) (k := k) hk
      have h0 : 0 < (splitBalancedVec z b k).size := by
        simpa [splitBalancedVec_size (z := z) (b := b) (k := k)] using hk
      have hRow0 : ((splitBalancedVec z b k)[0]!).size = z.size := by
        simpa [h0] using
          (splitBalancedVec_row_size (z := z) (b := b) (k := k) (i := 0) h0)
      simpa [recomposeSplitDigits, hNE, hRow0] using hj) =
    splitScalarResidueFoldF z[j]! b k := by
  simpa [splitScalarResidueFoldF] using
    (recomposeSplitDigits_splitBalancedVec_get_eq_residue_fold_of_base_ge_two
      (z := z) (b := b) (k := k) (j := j) hb hk hj)

theorem recomposeSplitDigits_splitBalancedVec_get_eq_ofInt_centeredInt_of_base_ge_two_of_state_zero
  {z : Array F} {b k j : Nat}
  (hb : 2 ≤ b)
  (hk : 0 < k)
  (hj : j < z.size)
  (hZero : (splitScalarState z[j]! b k).1 = 0)
  (hOfInt : p6OfIntSemiringAssumption) :
  (recomposeSplitDigits (splitBalancedVec z b k) b)[j]'(by
      have hNE : (splitBalancedVec z b k).isEmpty = false :=
        splitBalancedVec_isEmpty_eq_false_of_k_pos (z := z) (b := b) (k := k) hk
      have h0 : 0 < (splitBalancedVec z b k).size := by
        simpa [splitBalancedVec_size (z := z) (b := b) (k := k)] using hk
      have hRow0 : ((splitBalancedVec z b k)[0]!).size = z.size := by
        simpa [h0] using
          (splitBalancedVec_row_size (z := z) (b := b) (k := k) (i := 0) h0)
      simpa [recomposeSplitDigits, hNE, hRow0] using hj) =
    F.ofInt (centeredInt z[j]!) := by
  have hFoldF :
      (recomposeSplitDigits (splitBalancedVec z b k) b)[j]'(by
          have hNE : (splitBalancedVec z b k).isEmpty = false :=
            splitBalancedVec_isEmpty_eq_false_of_k_pos (z := z) (b := b) (k := k) hk
          have h0 : 0 < (splitBalancedVec z b k).size := by
            simpa [splitBalancedVec_size (z := z) (b := b) (k := k)] using hk
          have hRow0 : ((splitBalancedVec z b k)[0]!).size = z.size := by
            simpa [h0] using
              (splitBalancedVec_row_size (z := z) (b := b) (k := k) (i := 0) h0)
          simpa [recomposeSplitDigits, hNE, hRow0] using hj) =
        splitScalarResidueFoldF z[j]! b k := by
    exact recomposeSplitDigits_splitBalancedVec_get_eq_splitScalarResidueFoldF_of_base_ge_two
      (z := z) (b := b) (k := k) (j := j) hb hk hj
  have hFoldBridge :
      splitScalarResidueFoldF z[j]! b k =
        F.ofInt (splitScalarResidueFoldInt z[j]! b k) :=
    splitScalarResidueFoldF_eq_ofInt_splitScalarResidueFoldInt_of_p6OfIntSemiringAssumption
      hOfInt (z[j]!) b k
  have hCentered :
      centeredInt z[j]! = splitScalarResidueFoldInt z[j]! b k :=
    centeredInt_eq_splitScalarResidueFoldInt_of_splitScalarState_fst_zero
      (a := z[j]!) (b := b) (k := k) hb hZero
  calc
    (recomposeSplitDigits (splitBalancedVec z b k) b)[j]'(by
        have hNE : (splitBalancedVec z b k).isEmpty = false :=
          splitBalancedVec_isEmpty_eq_false_of_k_pos (z := z) (b := b) (k := k) hk
        have h0 : 0 < (splitBalancedVec z b k).size := by
          simpa [splitBalancedVec_size (z := z) (b := b) (k := k)] using hk
        have hRow0 : ((splitBalancedVec z b k)[0]!).size = z.size := by
          simpa [h0] using
            (splitBalancedVec_row_size (z := z) (b := b) (k := k) (i := 0) h0)
        simpa [recomposeSplitDigits, hNE, hRow0] using hj)
      = splitScalarResidueFoldF z[j]! b k := hFoldF
    _ = F.ofInt (splitScalarResidueFoldInt z[j]! b k) := hFoldBridge
    _ = F.ofInt (centeredInt z[j]!) := by simp [hCentered]

theorem recomposeSplitDigits_splitBalancedVec_get_eq_entry_of_base_ge_two_of_state_zero
  {z : Array F} {b k j : Nat}
  (hb : 2 ≤ b)
  (hk : 0 < k)
  (hj : j < z.size)
  (hZero : (splitScalarState z[j]! b k).1 = 0)
  (hOfInt : p6OfIntSemiringAssumption)
  (hCentered : ∀ x : F, F.ofInt (centeredInt x) = x) :
  (recomposeSplitDigits (splitBalancedVec z b k) b)[j]'(by
      have hNE : (splitBalancedVec z b k).isEmpty = false :=
        splitBalancedVec_isEmpty_eq_false_of_k_pos (z := z) (b := b) (k := k) hk
      have h0 : 0 < (splitBalancedVec z b k).size := by
        simpa [splitBalancedVec_size (z := z) (b := b) (k := k)] using hk
      have hRow0 : ((splitBalancedVec z b k)[0]!).size = z.size := by
        simpa [h0] using
          (splitBalancedVec_row_size (z := z) (b := b) (k := k) (i := 0) h0)
      simpa [recomposeSplitDigits, hNE, hRow0] using hj) = z[j]! := by
  calc
    (recomposeSplitDigits (splitBalancedVec z b k) b)[j]'(by
        have hNE : (splitBalancedVec z b k).isEmpty = false :=
          splitBalancedVec_isEmpty_eq_false_of_k_pos (z := z) (b := b) (k := k) hk
        have h0 : 0 < (splitBalancedVec z b k).size := by
          simpa [splitBalancedVec_size (z := z) (b := b) (k := k)] using hk
        have hRow0 : ((splitBalancedVec z b k)[0]!).size = z.size := by
          simpa [h0] using
            (splitBalancedVec_row_size (z := z) (b := b) (k := k) (i := 0) h0)
        simpa [recomposeSplitDigits, hNE, hRow0] using hj)
      = F.ofInt (centeredInt z[j]!) :=
          recomposeSplitDigits_splitBalancedVec_get_eq_ofInt_centeredInt_of_base_ge_two_of_state_zero
            (z := z) (b := b) (k := k) (j := j) hb hk hj hZero hOfInt
    _ = z[j]! := hCentered (z[j]!)

/-- Public alias for the scalar terminal-state-zero side condition used in P6 reconstruction. -/
def splitScalarTerminalZeroProp (z : Array F) (b k : Nat) : Prop :=
  ∀ j (_hj : j < z.size), (splitScalarState z[j]! b k).1 = 0

theorem recomposeSplitDigits_splitBalancedVec_eq_of_base_ge_two_of_state_zero
  {z : Array F} {b k : Nat}
  (hb : 2 ≤ b)
  (hk : 0 < k)
  (hZero : splitScalarTerminalZeroProp z b k)
  (hOfInt : p6OfIntSemiringAssumption)
  (hCentered : ∀ x : F, F.ofInt (centeredInt x) = x) :
  recomposeSplitDigits (splitBalancedVec z b k) b = z := by
  have hNE : (splitBalancedVec z b k).isEmpty = false :=
    splitBalancedVec_isEmpty_eq_false_of_k_pos (z := z) (b := b) (k := k) hk
  have h0 : 0 < (splitBalancedVec z b k).size := by
    simpa [splitBalancedVec_size (z := z) (b := b) (k := k)] using hk
  have hRow0 : ((splitBalancedVec z b k)[0]!).size = z.size := by
    simpa [h0] using
      (splitBalancedVec_row_size (z := z) (b := b) (k := k) (i := 0) h0)
  apply Array.ext
  · simpa [recomposeSplitDigits, hNE, hRow0]
  · intro j hjL hjR
    have hEntry :
        (recomposeSplitDigits (splitBalancedVec z b k) b)[j]'hjL = z[j]! := by
      have hEntry' :=
        recomposeSplitDigits_splitBalancedVec_get_eq_entry_of_base_ge_two_of_state_zero
          (z := z) (b := b) (k := k) (j := j)
          hb hk hjR (hZero j hjR) hOfInt hCentered
      simpa using hEntry'
    simpa [hjR] using hEntry

theorem digitsWithinBaseProp_mono
  {digits : Array (Array F)} {b B : Nat}
  (hDigits : digitsWithinBaseProp digits b)
  (hMono : b ≤ B) :
  digitsWithinBaseProp digits B := by
  intro i hi j hj
  exact Nat.lt_of_lt_of_le (hDigits i hi j hj) hMono

theorem digitsWithinBase_mono
  {digits : Array (Array F)} {b B : Nat}
  (hDigits : digitsWithinBase digits b = true)
  (hMono : b ≤ B) :
  digitsWithinBase digits B = true := by
  exact digitsWithinBase_complete
    (digitsWithinBaseProp_mono (digitsWithinBase_sound hDigits) hMono)

def splitRoundTrip (z : Array F) (b k : Nat) : Bool :=
  if b < 2 then
    false
  else
    let digits := splitBalancedVec z b k
    decide (recomposeSplitDigits digits b = z) && digitsWithinBase digits b

theorem splitRoundTrip_false_of_base_lt_two
  {z : Array F} {b k : Nat}
  (hb : b < 2) :
  splitRoundTrip z b k = false := by
  simp [splitRoundTrip, hb]

theorem splitRoundTrip_eq_checks_of_base_ge_two
  {z : Array F} {b k : Nat}
  (hb : 2 ≤ b) :
  splitRoundTrip z b k =
    (decide (recomposeSplitDigits (splitBalancedVec z b k) b = z) &&
      digitsWithinBase (splitBalancedVec z b k) b) := by
  unfold splitRoundTrip
  have hbNotLt : ¬ b < 2 := Nat.not_lt.mpr hb
  simp [hbNotLt]

theorem splitRoundTrip_true_iff_checks_of_base_ge_two
  {z : Array F} {b k : Nat}
  (hb : 2 ≤ b) :
  splitRoundTrip z b k = true ↔
    decide (recomposeSplitDigits (splitBalancedVec z b k) b = z) = true ∧
      digitsWithinBase (splitBalancedVec z b k) b = true := by
  rw [splitRoundTrip_eq_checks_of_base_ge_two hb]
  simp [Bool.and_eq_true]

theorem splitRoundTrip_digits_size
  {z : Array F} {b k : Nat}
  (_hOk : splitRoundTrip z b k = true) :
  (splitBalancedVec z b k).size = k := by
  exact splitBalancedVec_size z b k

theorem splitRoundTrip_sound
  {z : Array F} {b k : Nat}
  (hOk : splitRoundTrip z b k = true) :
  b ≥ 2 ∧
    let digits := splitBalancedVec z b k
    recomposeSplitDigits digits b = z ∧ digitsWithinBase digits b = true := by
  unfold splitRoundTrip at hOk
  by_cases hb : b < 2
  · simp [hb] at hOk
  · have hbGe : b ≥ 2 := Nat.le_of_not_lt hb
    simp [hb] at hOk
    have hAnd :
      decide (recomposeSplitDigits (splitBalancedVec z b k) b = z) = true ∧
        digitsWithinBase (splitBalancedVec z b k) b = true := by
      simpa [Bool.and_eq_true] using hOk
    refine ⟨hbGe, ?_⟩
    exact ⟨decide_eq_true_eq.mp hAnd.1, hAnd.2⟩

theorem splitRoundTrip_complete
  {z : Array F} {b k : Nat}
  (hProp : b ≥ 2 ∧
    let digits := splitBalancedVec z b k
    recomposeSplitDigits digits b = z ∧ digitsWithinBase digits b = true) :
  splitRoundTrip z b k = true := by
  rcases hProp with ⟨hbGe, hRest⟩
  have hbNotLt : ¬ b < 2 := Nat.not_lt.mpr hbGe
  rcases hRest with ⟨hRec, hDigits⟩
  unfold splitRoundTrip
  simp [hbNotLt, hRec, hDigits]

theorem splitRoundTrip_sound_prop
  {z : Array F} {b k : Nat}
  (hOk : splitRoundTrip z b k = true) :
  b ≥ 2 ∧
    let digits := splitBalancedVec z b k
    recomposeSplitDigits digits b = z ∧ digitsWithinBaseProp digits b := by
  rcases splitRoundTrip_sound hOk with ⟨hbGe, hRest⟩
  rcases hRest with ⟨hRec, hDigits⟩
  exact ⟨hbGe, hRec, digitsWithinBase_sound hDigits⟩

theorem splitRoundTrip_complete_prop
  {z : Array F} {b k : Nat}
  (hProp : b ≥ 2 ∧
    let digits := splitBalancedVec z b k
    recomposeSplitDigits digits b = z ∧ digitsWithinBaseProp digits b) :
  splitRoundTrip z b k = true := by
  rcases hProp with ⟨hbGe, hRest⟩
  rcases hRest with ⟨hRec, hDigits⟩
  exact splitRoundTrip_complete ⟨hbGe, hRec, digitsWithinBase_complete hDigits⟩

theorem splitRoundTrip_iff_prop
  {z : Array F} {b k : Nat} :
  splitRoundTrip z b k = true ↔
    (b ≥ 2 ∧
      let digits := splitBalancedVec z b k
      recomposeSplitDigits digits b = z ∧ digitsWithinBaseProp digits b) := by
  constructor
  · exact splitRoundTrip_sound_prop
  · exact splitRoundTrip_complete_prop

/-- Theorem-native P6 assumption boundary at fixed `(b,k)`. -/
def p6DecompAssumption (b k : Nat) : Prop :=
  b ≥ 2 ∧
    ∀ z : Array F,
      recomposeSplitDigits (splitBalancedVec z b k) b = z ∧
        digitsWithinBaseProp (splitBalancedVec z b k) b

/-- Check-style P6 assumption boundary at fixed `(b,k)`. -/
def p6DecompCheckAssumption (b k : Nat) : Prop :=
  b ≥ 2 ∧ ∀ z : Array F, splitRoundTrip z b k = true

theorem p6DecompAssumption_base_ge_two
  {b k : Nat}
  (hAssm : p6DecompAssumption b k) :
  b ≥ 2 := by
  exact hAssm.1

theorem p6DecompAssumption_recompose_eq
  {b k : Nat} {z : Array F}
  (hAssm : p6DecompAssumption b k) :
  recomposeSplitDigits (splitBalancedVec z b k) b = z := by
  exact (hAssm.2 z).1

theorem p6DecompAssumption_digitsWithinBaseProp
  {b k : Nat} {z : Array F}
  (hAssm : p6DecompAssumption b k) :
  digitsWithinBaseProp (splitBalancedVec z b k) b := by
  exact (hAssm.2 z).2

theorem p6DecompCheckAssumption_splitRoundTrip
  {b k : Nat} {z : Array F}
  (hCheck : p6DecompCheckAssumption b k) :
  splitRoundTrip z b k = true := by
  exact hCheck.2 z

theorem p6DecompAssumption_of_checkAssumption
  {b k : Nat}
  (hCheck : p6DecompCheckAssumption b k) :
  p6DecompAssumption b k := by
  rcases hCheck with ⟨hb, hAll⟩
  refine ⟨hb, ?_⟩
  intro z
  exact (splitRoundTrip_sound_prop (z := z) (b := b) (k := k) (hAll z)).2

theorem p6DecompCheckAssumption_of_assumption
  {b k : Nat}
  (hAssm : p6DecompAssumption b k) :
  p6DecompCheckAssumption b k := by
  rcases hAssm with ⟨hb, hAll⟩
  refine ⟨hb, ?_⟩
  intro z
  exact splitRoundTrip_complete_prop
    (z := z) (b := b) (k := k) ⟨hb, hAll z⟩

theorem p6DecompAssumption_iff_checkAssumption
  {b k : Nat} :
  p6DecompAssumption b k ↔ p6DecompCheckAssumption b k := by
  constructor
  · exact p6DecompCheckAssumption_of_assumption
  · exact p6DecompAssumption_of_checkAssumption

theorem splitRoundTrip_true_of_p6DecompAssumption
  {b k : Nat} {z : Array F}
  (hAssm : p6DecompAssumption b k) :
  splitRoundTrip z b k = true := by
  exact p6DecompCheckAssumption_splitRoundTrip
    (z := z) (p6DecompCheckAssumption_of_assumption hAssm)

theorem splitRoundTrip_base_ge_two
  {z : Array F} {b k : Nat}
  (hOk : splitRoundTrip z b k = true) :
  b ≥ 2 := by
  exact (splitRoundTrip_sound_prop hOk).1

theorem splitRoundTrip_recompose_eq
  {z : Array F} {b k : Nat}
  (hOk : splitRoundTrip z b k = true) :
  recomposeSplitDigits (splitBalancedVec z b k) b = z := by
  exact (splitRoundTrip_sound_prop hOk).2.1

theorem splitRoundTrip_recompose_size_eq
  {z : Array F} {b k : Nat}
  (hOk : splitRoundTrip z b k = true) :
  (recomposeSplitDigits (splitBalancedVec z b k) b).size = z.size := by
  simpa [splitRoundTrip_recompose_eq hOk]

theorem splitRoundTrip_digitsWithinBaseProp
  {z : Array F} {b k : Nat}
  (hOk : splitRoundTrip z b k = true) :
  digitsWithinBaseProp (splitBalancedVec z b k) b := by
  exact (splitRoundTrip_sound_prop hOk).2.2

theorem splitRoundTrip_digit_bound
  {z : Array F} {b k i j : Nat}
  (hOk : splitRoundTrip z b k = true)
  (hi : i < (splitBalancedVec z b k).size)
  (hj : j < ((splitBalancedVec z b k)[i]'hi).size) :
  normInfF (((splitBalancedVec z b k)[i]'hi)[j]'hj) < b := by
  exact splitRoundTrip_digitsWithinBaseProp hOk i hi j hj

theorem splitRoundTrip_digit_row_size
  {z : Array F} {b k i : Nat}
  (_hOk : splitRoundTrip z b k = true)
  (hi : i < (splitBalancedVec z b k).size) :
  ((splitBalancedVec z b k)[i]'hi).size = z.size := by
  exact splitBalancedVec_row_size hi

/-- Tiny sanity check for Definition 3 + split_b style decomposition. -/
def decompSanity : Bool :=
  let z := #[F.ofNat 3, F.ofInt (-2), F.ofNat (q - 1)]
  splitRoundTrip z 2 8

end SuperNeo
