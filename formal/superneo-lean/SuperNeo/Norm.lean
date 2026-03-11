import SuperNeo.Ring

namespace SuperNeo

local instance : NeZero Goldilocks.q := ⟨Nat.ne_of_gt Goldilocks.q_pos⟩

/-- Infinity norm on field elements via centered representatives. -/
def normInfF (x : F) : Nat :=
  F.centeredAbs x

/-- Infinity norm on coefficient arrays. -/
def normInfCoeffs (a : Coeffs) : Nat :=
  a.foldl (fun acc x => Nat.max acc (normInfF x)) 0

/-- Alias used by protocol-facing statements. -/
def maxRhoNorm (a : Coeffs) : Nat :=
  normInfCoeffs a

@[simp] theorem maxRhoNorm_eq_normInfCoeffs (a : Coeffs) :
    maxRhoNorm a = normInfCoeffs a := rfl

@[simp] theorem normInfF_zero : normInfF (0 : F) = 0 := by
  have hRep : F.centeredRep (0 : F) = Int.ofNat (0 : F).val := by
    apply F.centeredRep_eq_of_le_halfQ
    simp
  simp [normInfF, F.centeredAbs, hRep]

theorem normInfF_eq_zero_iff (x : F) :
    normInfF x = 0 ↔ x = 0 := by
  simpa [normInfF] using F.centeredAbs_eq_zero_iff x

@[simp] theorem normInfCoeffs_empty : normInfCoeffs (#[] : Coeffs) = 0 := by
  simp [normInfCoeffs]

theorem normInfCoeffs_nonneg (a : Coeffs) : 0 ≤ normInfCoeffs a :=
  Nat.zero_le _

theorem maxRhoNorm_nonneg (a : Coeffs) : 0 ≤ maxRhoNorm a :=
  normInfCoeffs_nonneg a

/-- Assumption bundle: vector-add norm bound from operand norms. -/
def vecAddNormBoundFromOperands (BA BB B : Nat) : Prop :=
  ∀ a b : Coeffs, a.size = b.size →
    normInfCoeffs a ≤ BA →
    normInfCoeffs b ≤ BB →
    normInfCoeffs (vecAdd a b) ≤ B

/-- Assumption bundle: vector-scale norm bound from operand norms. -/
def vecScaleNormBoundFromOperands (BS BA B : Nat) : Prop :=
  ∀ s : F, ∀ a : Coeffs,
    normInfF s ≤ BS →
    normInfCoeffs a ≤ BA →
    normInfCoeffs (vecScale s a) ≤ B

/-- Assumption bundle: ring multiplication norm bound from operand norms. -/
def mulRqNormBoundFromOperands (BA BB B : Nat) : Prop :=
  ∀ a b : Coeffs,
    normInfCoeffs a ≤ BA →
    normInfCoeffs b ≤ BB →
    normInfCoeffs (mulRq a b) ≤ B

/-- Assumption bundle: cyclotomic ring multiplication norm bound from operand norms. -/
def mulRqPhiNormBoundFromOperands (BA BB B : Nat) : Prop :=
  ∀ a b : Coeffs,
    normInfCoeffs a ≤ BA →
    normInfCoeffs b ≤ BB →
    normInfCoeffs (mulRqPhi a b) ≤ B

/-- Assumption bundle: subtraction norm bound from operand norms. -/
def coeffSubNormBoundFromOperands (BA BB B : Nat) : Prop :=
  ∀ a b : Coeffs, a.size = b.size →
    normInfCoeffs a ≤ BA →
    normInfCoeffs b ≤ BB →
    normInfCoeffs (vecAdd a (vecScale (-1) b)) ≤ B

def AllChallengeCoeffs (a : Coeffs) : Prop :=
  ∀ i : Fin a.size, normInfF a[i] ≤ 2

theorem allChallengeCoeffs_empty : AllChallengeCoeffs (#[] : Coeffs) := by
  intro i
  exact False.elim (Nat.not_lt_zero _ i.2)

theorem allChallengeCoeffs_mono
    {a : Coeffs}
    {B C : Nat}
    (hB : ∀ i : Fin a.size, normInfF a[i] ≤ B)
    (hBC : B ≤ C) :
    ∀ i : Fin a.size, normInfF a[i] ≤ C := by
  intro i
  exact Nat.le_trans (hB i) hBC

theorem allChallengeCoeffs_of_bound
    {a : Coeffs}
    (hB : ∀ i : Fin a.size, normInfF a[i] ≤ 2) :
    AllChallengeCoeffs a :=
  hB

theorem allChallengeCoeffs_weaken
    {a : Coeffs}
    (h : AllChallengeCoeffs a) :
    ∀ i : Fin a.size, normInfF a[i] ≤ 2 := h

theorem vecAddNormBoundFromOperands_of_global
    {BA BB B : Nat}
    (hGlobal : ∀ a b : Coeffs, a.size = b.size →
      normInfCoeffs (vecAdd a b) ≤ B) :
    vecAddNormBoundFromOperands BA BB B := by
  intro a b hSize _hA _hB
  exact hGlobal a b hSize

theorem vecScaleNormBoundFromOperands_of_global
    {BS BA B : Nat}
    (hGlobal : ∀ s : F, ∀ a : Coeffs, normInfCoeffs (vecScale s a) ≤ B) :
    vecScaleNormBoundFromOperands BS BA B := by
  intro s a _hS _hA
  exact hGlobal s a

theorem mulRqNormBoundFromOperands_of_global
    {BA BB B : Nat}
    (hGlobal : ∀ a b : Coeffs, normInfCoeffs (mulRq a b) ≤ B) :
    mulRqNormBoundFromOperands BA BB B := by
  intro a b _hA _hB
  exact hGlobal a b

theorem mulRqPhiNormBoundFromOperands_of_global
    {BA BB B : Nat}
    (hGlobal : ∀ a b : Coeffs, normInfCoeffs (mulRqPhi a b) ≤ B) :
    mulRqPhiNormBoundFromOperands BA BB B := by
  intro a b _hA _hB
  exact hGlobal a b

theorem coeffSubNormBoundFromOperands_of_global
    {BA BB B : Nat}
    (hGlobal : ∀ a b : Coeffs, a.size = b.size →
      normInfCoeffs (vecAdd a (vecScale (-1) b)) ≤ B) :
    coeffSubNormBoundFromOperands BA BB B := by
  intro a b hSize _hA _hB
  exact hGlobal a b hSize

private theorem acc_le_foldl_max_of_fn
  {α : Type}
  (l : List α)
  (f : α → Nat)
  (acc : Nat) :
  acc ≤ l.foldl (fun a y => Nat.max a (f y)) acc := by
  induction l generalizing acc with
  | nil =>
      simp
  | cons a t ih =>
      have h₁ : acc ≤ Nat.max acc (f a) := Nat.le_max_left _ _
      have h₂ :
          Nat.max acc (f a) ≤
            t.foldl (fun b y => Nat.max b (f y)) (Nat.max acc (f a)) :=
        ih (acc := Nat.max acc (f a))
      simpa [List.foldl] using Nat.le_trans h₁ h₂

private theorem le_foldl_max_of_mem_fn
  {α : Type}
  (l : List α)
  (f : α → Nat)
  (acc : Nat)
  (x : α)
  (hx : x ∈ l) :
  f x ≤ l.foldl (fun a y => Nat.max a (f y)) acc := by
  induction l generalizing acc with
  | nil =>
      cases hx
  | cons a t ih =>
      simp at hx
      rcases hx with hxa | hxt
      · subst hxa
        have h₁ : f x ≤ Nat.max acc (f x) := Nat.le_max_right _ _
        have h₂ : Nat.max acc (f x) ≤ t.foldl (fun b y => Nat.max b (f y)) (Nat.max acc (f x)) :=
          acc_le_foldl_max_of_fn t f (Nat.max acc (f x))
        simpa [List.foldl] using Nat.le_trans h₁ h₂
      · have hTail : f x ≤ t.foldl (fun b y => Nat.max b (f y)) (Nat.max acc (f a)) :=
          ih (acc := Nat.max acc (f a)) hxt
        simpa [List.foldl] using hTail

private theorem foldl_max_le_of_forall_le_fn
  {α : Type}
  (l : List α)
  (f : α → Nat)
  (acc m : Nat)
  (hAcc : acc ≤ m)
  (hAll : ∀ x ∈ l, f x ≤ m) :
  l.foldl (fun a y => Nat.max a (f y)) acc ≤ m := by
  induction l generalizing acc with
  | nil =>
      simpa using hAcc
  | cons a t ih =>
      have ha : f a ≤ m := hAll a (by simp)
      have hAcc' : Nat.max acc (f a) ≤ m := (Nat.max_le).2 ⟨hAcc, ha⟩
      have hAll' : ∀ x ∈ t, f x ≤ m := by
        intro x hx
        exact hAll x (by simp [hx])
      simpa [List.foldl] using ih (acc := Nat.max acc (f a)) hAcc' hAll'

theorem normInfF_getElem_le_normInfCoeffs
  (a : Coeffs) (i : Fin a.size) :
  normInfF a[i] ≤ normInfCoeffs a := by
  have hmem : a[i] ∈ a.toList := Array.getElem_mem_toList (xs := a) i.2
  unfold normInfCoeffs
  rw [← Array.foldl_toList]
  exact le_foldl_max_of_mem_fn a.toList normInfF 0 a[i] hmem

theorem normInfF_coeffAt_le_normInfCoeffs
  (a : Coeffs) (i : Nat) :
  normInfF (coeffAt a i) ≤ normInfCoeffs a := by
  by_cases hi : i < d
  · by_cases his : i < a.size
    · have hle : normInfF (a[i]'his) ≤ normInfCoeffs a :=
        normInfF_getElem_le_normInfCoeffs a ⟨i, his⟩
      simpa [coeffAt, hi, Array.getD, his] using hle
    · have hcoeff : coeffAt a i = (0 : F) := by
        simp [coeffAt, hi, Array.getD, his]
      rw [hcoeff, normInfF_zero]
      exact Nat.zero_le _
  · have hcoeff : coeffAt a i = (0 : F) := by
      simp [coeffAt, hi]
    rw [hcoeff, normInfF_zero]
    exact Nat.zero_le _

theorem normInfF_mul_le_of_normInfF_left_le_four
  (x y : F)
  (hx : normInfF x ≤ 4) :
  normInfF (x * y) ≤ 4 * normInfF y := by
  simpa [normInfF] using F.centeredAbs_mul_le_of_centeredAbs_left_le_four x y hx

theorem normInfCoeffs_le_of_forall_getElem
  {a : Coeffs} {M : Nat}
  (hAll : ∀ i : Fin a.size, normInfF a[i] ≤ M) :
  normInfCoeffs a ≤ M := by
  unfold normInfCoeffs
  rw [← Array.foldl_toList]
  refine foldl_max_le_of_forall_le_fn
    (l := a.toList)
    (f := normInfF)
    (acc := 0)
    (m := M)
    (by exact Nat.zero_le _)
    ?_
  intro x hx
  have hxArr : x ∈ a := (Array.mem_def).2 hx
  rcases (Array.mem_iff_getElem).1 hxArr with ⟨i, hi, hEq⟩
  subst hEq
  exact hAll ⟨i, hi⟩

theorem normInfCoeffs_le_of_forall_coeffAt
  {a : Coeffs} {M : Nat}
  (ha : a.size = d)
  (hAll : ∀ i : Nat, i < d → normInfF (coeffAt a i) ≤ M) :
  normInfCoeffs a ≤ M := by
  apply normInfCoeffs_le_of_forall_getElem
  intro i
  have hi : i.1 < d := by simpa [ha] using i.2
  have hcoeff := hAll i.1 hi
  simpa [coeffAt, hi, Array.getD, i.2] using hcoeff

private theorem normInfF_foldl_add_le_with_init
  (l : List Nat)
  (t : Nat → F)
  (init : F)
  (M : Nat)
  (hM : ∀ j ∈ l, normInfF (t j) ≤ M) :
  normInfF (l.foldl (fun acc j => acc + t j) init) ≤ normInfF init + l.length * M := by
  induction l generalizing init with
  | nil =>
      simp
  | cons j js ih =>
      have hj : normInfF (t j) ≤ M := hM j (by simp)
      have hTail : ∀ k ∈ js, normInfF (t k) ≤ M := by
        intro k hk
        exact hM k (by simp [hk])
      calc
        normInfF ((j :: js).foldl (fun acc k => acc + t k) init)
            = normInfF (js.foldl (fun acc k => acc + t k) (init + t j)) := by
                simp [List.foldl]
        _ ≤ normInfF (init + t j) + js.length * M := ih (init := init + t j) hTail
        _ ≤ (normInfF init + normInfF (t j)) + js.length * M := by
              exact Nat.add_le_add_right (by simpa [normInfF] using F.centeredAbs_add_le init (t j)) _
        _ ≤ (normInfF init + M) + js.length * M := by
              exact Nat.add_le_add_right (Nat.add_le_add_left hj (normInfF init)) _
        _ = normInfF init + (M + js.length * M) := by
              omega
        _ = normInfF init + (js.length + 1) * M := by
              simp [Nat.succ_mul, Nat.add_comm, Nat.add_left_comm, Nat.add_assoc]
        _ = normInfF init + (j :: js).length * M := by
              simp

private theorem normInfF_foldl_add_range_le
  (d : Nat)
  (t : Nat → F)
  (M : Nat)
  (hM : ∀ j ∈ List.range d, normInfF (t j) ≤ M) :
  normInfF ((List.range d).foldl (fun acc j => acc + t j) 0) ≤ d * M := by
  have h :=
    normInfF_foldl_add_le_with_init (l := List.range d) (t := t) (init := 0) (M := M) hM
  simpa [normInfF_zero, List.length_range] using h

private theorem foldl_acc_if_eq_foldl_add_if
  (l : List Nat)
  (n : Nat)
  (f : Nat → F)
  (acc : F) :
  l.foldl
      (fun acc i =>
        if hIn : i ≤ n ∧ n - i < d then
          acc + f i
        else
          acc)
      acc
    =
    l.foldl
      (fun acc i =>
        acc + (if hIn : i ≤ n ∧ n - i < d then f i else 0))
      acc := by
  induction l generalizing acc with
  | nil =>
      simp
  | cons i is ih =>
      by_cases hIn : i ≤ n ∧ n - i < d
      · simpa [List.foldl, hIn] using ih (acc := acc + f i)
      · simpa [List.foldl, hIn] using ih (acc := acc)

set_option maxHeartbeats 1000000 in
theorem normInfF_coeffAt_mulRqPhi_le_of_left_four
  (a b : Coeffs)
  (i : Nat) (hi : i < d)
  {B : Nat}
  (hA : normInfCoeffs a ≤ 4)
  (hB : normInfCoeffs b ≤ B) :
  normInfF (coeffAt (mulRqPhi a b) i) ≤ 12 * d * B := by
  let rawM : Nat := d * (4 * B)
  have hRawBound : ∀ n : Nat, normInfF (rawConvCoeff a b n) ≤ rawM := by
    intro n
    let term : Nat → F := fun j =>
      if hIn : j ≤ n ∧ n - j < d then
        coeffAt a j * coeffAt b (n - j)
      else
        0
    have hTermBound : ∀ j ∈ List.range d, normInfF (term j) ≤ 4 * B := by
      intro j hj
      by_cases hIn : j ≤ n ∧ n - j < d
      · have hAj : normInfF (coeffAt a j) ≤ 4 :=
          Nat.le_trans (normInfF_coeffAt_le_normInfCoeffs a j) hA
        have hBj : normInfF (coeffAt b (n - j)) ≤ B :=
          Nat.le_trans (normInfF_coeffAt_le_normInfCoeffs b (n - j)) hB
        have hMul :
            normInfF (coeffAt a j * coeffAt b (n - j)) ≤
              4 * normInfF (coeffAt b (n - j)) := by
          exact normInfF_mul_le_of_normInfF_left_le_four
            (coeffAt a j) (coeffAt b (n - j)) hAj
        have hLe : normInfF (coeffAt a j * coeffAt b (n - j)) ≤ 4 * B :=
          Nat.le_trans hMul (Nat.mul_le_mul_left 4 hBj)
        simpa [term, hIn] using hLe
      · have : normInfF (term j) = 0 := by simp [term, hIn, normInfF_zero]
        rw [this]
        exact Nat.zero_le _
    have hFold :
        normInfF ((List.range d).foldl (fun acc j => acc + term j) 0) ≤ d * (4 * B) :=
      normInfF_foldl_add_range_le d term (4 * B) hTermBound
    have hRaw :
        rawConvCoeff a b n = (List.range d).foldl (fun acc j => acc + term j) 0 := by
      unfold rawConvCoeff term
      rw [foldl_acc_if_eq_foldl_add_if
        (l := List.range d)
        (n := n)
        (f := fun j => coeffAt a j * coeffAt b (n - j))
        (acc := 0)]
    simpa [rawM, hRaw] using hFold
  have hCoeff :
      coeffAt (mulRqPhi a b) i =
        let n := i
        let base := rawConvCoeff a b n
        if n < 26 then
          base - rawConvCoeff a b (n + 54) + rawConvCoeff a b (n + 81)
        else if n = 26 then
          base - rawConvCoeff a b 80
        else
          base - rawConvCoeff a b (n + 27) := by
    simp [coeffAt, hi, mulRqPhi]
  rw [hCoeff]
  have h3Mle12 : rawM + rawM + rawM ≤ 12 * d * B := by
    let x : Nat := d * B
    have hRawX : rawM = 4 * x := by
      unfold rawM x
      calc
        d * (4 * B) = d * (B * 4) := by rw [Nat.mul_comm 4 B]
        _ = (d * B) * 4 := by rw [Nat.mul_assoc]
        _ = 4 * (d * B) := by rw [Nat.mul_comm]
    have hLeft : rawM + rawM + rawM = 12 * x := by
      rw [hRawX]
      calc
        4 * x + 4 * x + 4 * x
            = (4 * x + 4 * x) + 4 * x := by
                simp [Nat.add_assoc]
        _ = (4 + 4) * x + 4 * x := by
              rw [← Nat.add_mul]
        _ = ((4 + 4) + 4) * x := by
              rw [← Nat.add_mul]
        _ = 12 * x := by simp
    have hRight : 12 * d * B = 12 * x := by
      unfold x
      rw [Nat.mul_assoc]
    exact Nat.le_of_eq (hLeft.trans hRight.symm)
  have h2Mle12 : rawM + rawM ≤ 12 * d * B := by
    have h2le3 : rawM + rawM ≤ rawM + rawM + rawM := by
      simpa [Nat.add_assoc] using Nat.le_add_right (rawM + rawM) rawM
    exact Nat.le_trans h2le3 h3Mle12
  by_cases hLt26 : i < 26
  · have hBase : normInfF (rawConvCoeff a b i) ≤ rawM := hRawBound i
    have h54 : normInfF (rawConvCoeff a b (i + 54)) ≤ rawM := hRawBound (i + 54)
    have h81 : normInfF (rawConvCoeff a b (i + 81)) ≤ rawM := hRawBound (i + 81)
    have hSub :
        normInfF (rawConvCoeff a b i - rawConvCoeff a b (i + 54)) ≤
          normInfF (rawConvCoeff a b i) + normInfF (rawConvCoeff a b (i + 54)) := by
      simpa [normInfF] using
        F.centeredAbs_sub_le (rawConvCoeff a b i) (rawConvCoeff a b (i + 54))
    have hAdd :
        normInfF (rawConvCoeff a b i - rawConvCoeff a b (i + 54) + rawConvCoeff a b (i + 81)) ≤
          normInfF (rawConvCoeff a b i - rawConvCoeff a b (i + 54)) + normInfF (rawConvCoeff a b (i + 81)) := by
      simpa [normInfF] using
        F.centeredAbs_add_le
          (rawConvCoeff a b i - rawConvCoeff a b (i + 54))
          (rawConvCoeff a b (i + 81))
    have hLe3M :
        normInfF (rawConvCoeff a b i - rawConvCoeff a b (i + 54) + rawConvCoeff a b (i + 81)) ≤
          rawM + rawM + rawM := by
      exact Nat.le_trans hAdd (Nat.add_le_add hSub h81 |> fun h => Nat.le_trans h (Nat.add_le_add (Nat.add_le_add hBase h54) (Nat.le_refl _)))
    have hLe :
        normInfF (rawConvCoeff a b i - rawConvCoeff a b (i + 54) + rawConvCoeff a b (i + 81)) ≤
          12 * d * B := Nat.le_trans hLe3M h3Mle12
    have hEval :
        (let n := i
         let base := rawConvCoeff a b n
         if n < 26 then
           base - rawConvCoeff a b (n + 54) + rawConvCoeff a b (n + 81)
         else if n = 26 then
           base - rawConvCoeff a b 80
         else
           base - rawConvCoeff a b (n + 27)) =
        rawConvCoeff a b i - rawConvCoeff a b (i + 54) + rawConvCoeff a b (i + 81) := by
      simp [hLt26]
    simpa [hEval] using hLe
  · by_cases hEq26 : i = 26
    · have hBase : normInfF (rawConvCoeff a b i) ≤ rawM := hRawBound i
      have h80 : normInfF (rawConvCoeff a b 80) ≤ rawM := hRawBound 80
      have hSub :
          normInfF (rawConvCoeff a b i - rawConvCoeff a b 80) ≤
            normInfF (rawConvCoeff a b i) + normInfF (rawConvCoeff a b 80) := by
        simpa [normInfF] using
          F.centeredAbs_sub_le (rawConvCoeff a b i) (rawConvCoeff a b 80)
      have hLe2M : normInfF (rawConvCoeff a b i - rawConvCoeff a b 80) ≤ rawM + rawM :=
        Nat.le_trans hSub (Nat.add_le_add hBase h80)
      have hLe : normInfF (rawConvCoeff a b i - rawConvCoeff a b 80) ≤ 12 * d * B :=
        Nat.le_trans hLe2M h2Mle12
      have hEval :
          (let n := i
           let base := rawConvCoeff a b n
           if n < 26 then
             base - rawConvCoeff a b (n + 54) + rawConvCoeff a b (n + 81)
           else if n = 26 then
             base - rawConvCoeff a b 80
           else
             base - rawConvCoeff a b (n + 27)) =
          rawConvCoeff a b i - rawConvCoeff a b 80 := by
        simp [hLt26, hEq26]
      simpa [hEval] using hLe
    · have hBase : normInfF (rawConvCoeff a b i) ≤ rawM := hRawBound i
      have h27 : normInfF (rawConvCoeff a b (i + 27)) ≤ rawM := hRawBound (i + 27)
      have hSub :
          normInfF (rawConvCoeff a b i - rawConvCoeff a b (i + 27)) ≤
            normInfF (rawConvCoeff a b i) + normInfF (rawConvCoeff a b (i + 27)) := by
        simpa [normInfF] using
          F.centeredAbs_sub_le (rawConvCoeff a b i) (rawConvCoeff a b (i + 27))
      have hLe2M : normInfF (rawConvCoeff a b i - rawConvCoeff a b (i + 27)) ≤ rawM + rawM :=
        Nat.le_trans hSub (Nat.add_le_add hBase h27)
      have hLe : normInfF (rawConvCoeff a b i - rawConvCoeff a b (i + 27)) ≤ 12 * d * B :=
        Nat.le_trans hLe2M h2Mle12
      have hEval :
          (let n := i
           let base := rawConvCoeff a b n
           if n < 26 then
             base - rawConvCoeff a b (n + 54) + rawConvCoeff a b (n + 81)
           else if n = 26 then
             base - rawConvCoeff a b 80
           else
             base - rawConvCoeff a b (n + 27)) =
          rawConvCoeff a b i - rawConvCoeff a b (i + 27) := by
        simp [hLt26, hEq26]
      simpa [hEval] using hLe

set_option maxHeartbeats 1000000 in
theorem mulRqPhiNormBoundFromOperands_four_d (B : Nat) :
  mulRqPhiNormBoundFromOperands 4 B (12 * d * B) := by
  intro a b hA hB
  have hsize : (mulRqPhi a b).size = d := mulRqPhi_size a b
  apply normInfCoeffs_le_of_forall_coeffAt hsize
  intro i hi
  exact normInfF_coeffAt_mulRqPhi_le_of_left_four a b i hi hA hB

theorem mulRqPhiNormBoundFromOperands_four_of_three_d_le
  {T B : Nat}
  (hTd : 3 * d ≤ T) :
  mulRqPhiNormBoundFromOperands 4 B (4 * T * B) := by
  intro a b hA hB
  have hBase : normInfCoeffs (mulRqPhi a b) ≤ 12 * d * B :=
    mulRqPhiNormBoundFromOperands_four_d B a b hA hB
  have hGrow : 12 * d * B ≤ 4 * T * B := by
    have h4 : 4 * (3 * d) ≤ 4 * T := Nat.mul_le_mul_left 4 hTd
    have h4B : (4 * (3 * d)) * B ≤ (4 * T) * B := Nat.mul_le_mul_right B h4
    calc
      12 * d * B = (4 * (3 * d)) * B := by
        simp [Nat.mul_assoc, Nat.mul_left_comm, Nat.mul_comm]
      _ ≤ (4 * T) * B := h4B
      _ = 4 * T * B := by
        simp [Nat.mul_assoc]
  exact Nat.le_trans hBase hGrow

theorem mulRqNormBoundFromOperands_four_d (B : Nat) :
  mulRqNormBoundFromOperands 4 B (12 * d * B) := by
  intro a b hA hB
  simpa [mulRq] using mulRqPhiNormBoundFromOperands_four_d B a b hA hB

theorem mulRqNormBoundFromOperands_four_of_d_le
  {T B : Nat}
  (hTd : 3 * d ≤ T) :
  mulRqNormBoundFromOperands 4 B (4 * T * B) := by
  intro a b hA hB
  simpa [mulRq] using
    mulRqPhiNormBoundFromOperands_four_of_three_d_le (T := T) (B := B) hTd a b hA hB

private theorem vecAdd_vecScale_neg_eq_ofFn_sub
  (a b : Coeffs)
  (hSize : a.size = b.size) :
  vecAdd a (vecScale (-1) b) =
    Array.ofFn (fun i : Fin a.size =>
      a[i] - b[i.1]'(by simpa [hSize] using i.2)) := by
  apply Array.ext
  · simp [vecAdd, hSize, vecScale]
  · intro j hj1 hj2
    have hEq : a.size = (vecScale (-1) b).size := by
      simpa [vecScale_size, hSize]
    have hjA : j < a.size := by
      simpa [vecAdd, hEq] using hj1
    have hjB : j < b.size := by
      simpa [hSize] using hjA
    simp [vecAdd, hEq, vecScale, hjA, hjB]
    have hOneMul : (1 : F) * b[j]'hjB = b[j]'hjB := by
      calc
        (1 : F) * b[j]'hjB = b[j]'hjB * 1 := by
          simpa using (Lean.Grind.Fin.mul_comm (n := Goldilocks.q) (1 : F) (b[j]'hjB))
        _ = b[j]'hjB := by
          simpa using (Lean.Grind.Fin.mul_one (n := Goldilocks.q) (b[j]'hjB))
    have hnegmul : (-1 : F) * b[j]'hjB = -(b[j]'hjB) := by
      calc
        (-1 : F) * b[j]'hjB = -(1 * b[j]'hjB) := by
          simpa using (Lean.Grind.Fin.neg_mul (n := Goldilocks.q) (1 : F) (b[j]'hjB))
        _ = -(b[j]'hjB) := by
          simpa [hOneMul]
    simpa [hnegmul] using
      (Lean.Grind.Fin.sub_eq_add_neg (n := Goldilocks.q) (a[j]'hjA) (b[j]'hjB)).symm

theorem coeffSubNormBoundFromOperands_two_two_four :
  coeffSubNormBoundFromOperands 2 2 4 := by
  intro a b hSize hA hB
  rw [vecAdd_vecScale_neg_eq_ofFn_sub a b hSize]
  apply normInfCoeffs_le_of_forall_getElem
  intro i
  have hiA : i.1 < a.size := by
    simpa using i.2
  have hiB : i.1 < b.size := by
    simpa [hSize] using hiA
  have hAi : normInfF (a[i.1]'hiA) ≤ 2 :=
    Nat.le_trans (normInfF_getElem_le_normInfCoeffs a ⟨i.1, hiA⟩) hA
  have hBi : normInfF (b[i.1]'hiB) ≤ 2 :=
    Nat.le_trans (normInfF_getElem_le_normInfCoeffs b ⟨i.1, hiB⟩) hB
  have hSub :
    normInfF (a[i.1]'hiA - b[i.1]'hiB) ≤
      normInfF (a[i.1]'hiA) + normInfF (b[i.1]'hiB) := by
    simpa [normInfF] using F.centeredAbs_sub_le (a[i.1]'hiA) (b[i.1]'hiB)
  have hLe4 : normInfF (a[i.1]'hiA) + normInfF (b[i.1]'hiB) ≤ 4 := by
    exact Nat.add_le_add hAi hBi
  have hOfFn :
      (Array.ofFn (fun i : Fin a.size =>
        a[i] - b[i.1]'(by simpa [hSize] using i.2)))[i] =
        a[i.1]'hiA - b[i.1]'hiB := by
    change
      (Array.ofFn (fun i : Fin a.size =>
        a[i] - b[i.1]'(by simpa [hSize] using i.2)))[i.1]'i.2 =
        a[i.1]'hiA - b[i.1]'hiB
    simpa [hiA, hiB] using
      (Array.getElem_ofFn
        (f := fun i : Fin a.size =>
          a[i] - b[i.1]'(by simpa [hSize] using i.2))
        (h := i.2))
  calc
    normInfF ((Array.ofFn (fun i : Fin a.size =>
      a[i] - b[i.1]'(by simpa [hSize] using i.2)))[i])
        = normInfF (a[i.1]'hiA - b[i.1]'hiB) := by
            exact congrArg normInfF hOfFn
    _ ≤ normInfF (a[i.1]'hiA) + normInfF (b[i.1]'hiB) := hSub
    _ ≤ 4 := hLe4

end SuperNeo
