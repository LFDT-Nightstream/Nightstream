import SuperNeo.ProofSystem.Lattice
import SuperNeo.RingMulComm
import SuperNeo.SamplingSet
import Init.GrindInstances.Ring.Fin

namespace SuperNeo.ProofSystem

/--
Boundary laws used by extractor linearity steps.
These are threaded explicitly until Ring/Norm closure discharges them.
-/
local instance : NeZero Goldilocks.q := ⟨Nat.ne_of_gt Goldilocks.q_pos⟩

structure LatticeReductionLaws (params : AjtaiParams) where
  samplingCarrier : SamplingCarrier
  strongSampling : strongSamplingExpansionProp samplingCarrier params.relaxedExpansion

namespace LatticeReductionLaws

/--
Canonical constructor from an explicit carrier-level strong-sampling theorem.
-/
def ofCarrier
  {params : AjtaiParams}
  (C : SamplingCarrier)
  (hStrong : strongSamplingExpansionProp C params.relaxedExpansion) :
  LatticeReductionLaws params where
  samplingCarrier := C
  strongSampling := hStrong

/--
Canonical constructor specialized to the paper-facing carrier `paperCarrier`.
-/
def ofPaperCarrier
  {params : AjtaiParams}
  (hStrong : strongSamplingExpansionProp paperCarrier params.relaxedExpansion) :
  LatticeReductionLaws params :=
  ofCarrier paperCarrier hStrong

/--
Derive the paper-facing strong-sampling contract from subtraction/multiplication
norm bundles, specialized at `params.relaxedExpansion`.
-/
theorem paperStrongSampling_of_bounds
  {params : AjtaiParams}
  {D : Nat}
  (hSub : coeffSubNormBoundFromOperands 2 2 D)
  (hMul : ∀ B : Nat, mulRqPhiNormBoundFromOperands D B (4 * params.relaxedExpansion * B)) :
  strongSamplingExpansionProp paperCarrier params.relaxedExpansion := by
  simpa using
    (SuperNeo.strongSamplingExpansionProp_of_paperCarrier
      (T := params.relaxedExpansion) (D := D) hSub hMul)

/--
Concrete paper-carrier strong-sampling derived from proved norm bundles, under
the arithmetic side-condition `3*d ≤ params.relaxedExpansion`.
-/
theorem paperStrongSampling_of_three_d_le
  {params : AjtaiParams}
  (hTd : 3 * d ≤ params.relaxedExpansion) :
  strongSamplingExpansionProp paperCarrier params.relaxedExpansion := by
  simpa using
    (SuperNeo.strongSamplingExpansionProp_paperCarrier_of_three_d_le
      (T := params.relaxedExpansion) hTd)

/--
Constructor specialized to `paperCarrier`, derived from theorem-native norm
bundles through `SamplingSet.strongSamplingExpansionProp_of_paperCarrier`.
-/
def ofPaperCarrierFromBounds
  {params : AjtaiParams}
  {D : Nat}
  (hSub : coeffSubNormBoundFromOperands 2 2 D)
  (hMul : ∀ B : Nat, mulRqPhiNormBoundFromOperands D B (4 * params.relaxedExpansion * B)) :
  LatticeReductionLaws params :=
  ofPaperCarrier
    (paperStrongSampling_of_bounds (params := params) (D := D) hSub hMul)

/--
Constructor specialized to `paperCarrier`, using concrete proved norm bundles
and the side-condition `3*d ≤ params.relaxedExpansion`.
-/
def ofPaperCarrierFromThreeDLe
  {params : AjtaiParams}
  (hTd : 3 * d ≤ params.relaxedExpansion) :
  LatticeReductionLaws params :=
  ofPaperCarrier
    (paperStrongSampling_of_three_d_le (params := params) hTd)

end LatticeReductionLaws

private theorem f_sub_eq_add_neg (a b : F) : a - b = a + -b := by
  apply Fin.ext
  change (Goldilocks.q - b.val + a.val) % Goldilocks.q =
    (a.val + (Goldilocks.q - b.val) % Goldilocks.q) % Goldilocks.q
  calc
    (Goldilocks.q - b.val + a.val) % Goldilocks.q
        = (((Goldilocks.q - b.val) % Goldilocks.q) + (a.val % Goldilocks.q)) % Goldilocks.q := by
            simpa using (Nat.add_mod (Goldilocks.q - b.val) a.val Goldilocks.q)
    _ = (a.val + (Goldilocks.q - b.val) % Goldilocks.q) % Goldilocks.q := by
          have ha : a.val % Goldilocks.q = a.val := Nat.mod_eq_of_lt a.isLt
          simp [ha, Nat.add_comm]

private theorem f_mul_sub (a b c : F) : a * (b - c) = a * b - a * c := by
  calc
    a * (b - c) = a * (b + -c) := by
      simp [f_sub_eq_add_neg]
    _ = a * b + a * -c := by
      simpa using (Lean.Grind.Fin.left_distrib (n := Goldilocks.q) a b (-c))
    _ = a * b + -(a * c) := by
      have hneg : a * -c = -(a * c) := by
        calc
          a * -c = -c * a := by
            simpa using (Lean.Grind.Fin.mul_comm (n := Goldilocks.q) a (-c))
          _ = -(c * a) := by
            simpa using (Lean.Grind.Fin.neg_mul (n := Goldilocks.q) c a)
          _ = -(a * c) := by
            simpa using congrArg (fun t => -t) (Lean.Grind.Fin.mul_comm (n := Goldilocks.q) c a)
      simp [hneg]
    _ = a * b - a * c := by
      simp [f_sub_eq_add_neg]

private theorem f_neg_add (a b : F) : -(a + b) = -a + -b := by
  have hNegOneMul (x : F) : -x = (-1 : F) * x := by
    calc
      -x = -((1 : F) * x) := by
        have hone : (1 : F) * x = x := by
          calc
            (1 : F) * x = x * (1 : F) := by
              simpa using (Lean.Grind.Fin.mul_comm (n := Goldilocks.q) (1 : F) x)
            _ = x := by
              simpa using (Lean.Grind.Fin.mul_one (n := Goldilocks.q) x)
        simp [hone]
      _ = (-1 : F) * x := by
        simpa using (Lean.Grind.Fin.neg_mul (n := Goldilocks.q) (1 : F) x).symm
  have hA : ((-1 : F) * a) = -a := by
    simpa using (hNegOneMul a).symm
  have hB : ((-1 : F) * b) = -b := by
    simpa using (hNegOneMul b).symm
  calc
    -(a + b) = (-1 : F) * (a + b) := by
      simpa using hNegOneMul (a + b)
    _ = ((-1 : F) * a) + ((-1 : F) * b) := by
      simpa using (Lean.Grind.Fin.left_distrib (n := Goldilocks.q) (-1 : F) a b)
    _ = -a + -b := by
      simpa [hA, hB]

private theorem f_sub_add_sub (x y u v : F) :
    (x - y) + (u - v) = (x + u) - (y + v) := by
  calc
    (x - y) + (u - v) = (x + -y) + (u + -v) := by
      simp [f_sub_eq_add_neg]
    _ = x + (-y + (u + -v)) := by
      simpa using (Lean.Grind.Fin.add_assoc (n := Goldilocks.q) x (-y) (u + -v))
    _ = x + ((u + -y) + -v) := by
      have hcomm : -y + u = u + -y := by
        simpa using (Lean.Grind.Fin.add_comm (n := Goldilocks.q) (-y) u)
      calc
        x + (-y + (u + -v)) = x + ((-y + u) + -v) := by
          simp [Lean.Grind.Fin.add_assoc]
        _ = x + ((u + -y) + -v) := by
          simp [hcomm]
    _ = (x + u) + (-y + -v) := by
      calc
        x + ((u + -y) + -v) = x + (u + (-y + -v)) := by
          simpa using congrArg (fun t => x + t)
            (Lean.Grind.Fin.add_assoc (n := Goldilocks.q) u (-y) (-v))
        _ = (x + u) + (-y + -v) := by
          simpa using (Lean.Grind.Fin.add_assoc (n := Goldilocks.q) x u (-y + -v)).symm
    _ = (x + u) + (-(y + v)) := by
      simp [f_neg_add, Lean.Grind.Fin.add_comm]
    _ = (x + u) - (y + v) := by
      simp [f_sub_eq_add_neg]

private theorem foldl_sub_linearity_F
  (l : List Nat)
  (t1 t2 : Nat → F)
  (acc1 acc2 : F) :
  l.foldl (fun acc j => acc + (t1 j - t2 j)) (acc1 - acc2) =
    (l.foldl (fun acc j => acc + t1 j) acc1) -
      (l.foldl (fun acc j => acc + t2 j) acc2) := by
  induction l generalizing acc1 acc2 with
  | nil =>
      simp
  | cons j js ih =>
      have hstep :
          (acc1 - acc2) + (t1 j - t2 j) =
            (acc1 + t1 j) - (acc2 + t2 j) := by
        simpa using f_sub_add_sub acc1 acc2 (t1 j) (t2 j)
      simpa [List.foldl, hstep] using ih (acc1 := acc1 + t1 j) (acc2 := acc2 + t2 j)

private theorem foldl_add_linearity_F
  (l : List Nat)
  (t1 t2 : Nat → F)
  (acc1 acc2 : F) :
  l.foldl (fun acc j => acc + (t1 j + t2 j)) (acc1 + acc2) =
    (l.foldl (fun acc j => acc + t1 j) acc1) +
      (l.foldl (fun acc j => acc + t2 j) acc2) := by
  induction l generalizing acc1 acc2 with
  | nil =>
      simp
  | cons j js ih =>
      have hstep :
          (acc1 + acc2) + (t1 j + t2 j) =
            (acc1 + t1 j) + (acc2 + t2 j) := by
        calc
          (acc1 + acc2) + (t1 j + t2 j)
              = acc1 + (acc2 + (t1 j + t2 j)) := by
                  simpa using
                    (Lean.Grind.Fin.add_assoc (n := Goldilocks.q) acc1 acc2 (t1 j + t2 j))
          _ = acc1 + (t1 j + (acc2 + t2 j)) := by
                have hMid : acc2 + (t1 j + t2 j) = t1 j + (acc2 + t2 j) := by
                  calc
                    acc2 + (t1 j + t2 j) = (acc2 + t1 j) + t2 j := by
                      simpa using
                        (Lean.Grind.Fin.add_assoc (n := Goldilocks.q) acc2 (t1 j) (t2 j)).symm
                    _ = (t1 j + acc2) + t2 j := by
                          simpa using congrArg (fun t => t + t2 j)
                            (Lean.Grind.Fin.add_comm (n := Goldilocks.q) acc2 (t1 j))
                    _ = t1 j + (acc2 + t2 j) := by
                          simpa using
                            (Lean.Grind.Fin.add_assoc (n := Goldilocks.q) (t1 j) acc2 (t2 j))
                simpa [hMid]
          _ = (acc1 + t1 j) + (acc2 + t2 j) := by
                simpa using
                  (Lean.Grind.Fin.add_assoc (n := Goldilocks.q) acc1 (t1 j) (acc2 + t2 j)).symm
      simpa [List.foldl, hstep] using ih (acc1 := acc1 + t1 j) (acc2 := acc2 + t2 j)

private theorem list_foldl_congr
  {α β : Type}
  (f g : α → β → α)
  (init : α)
  (l : List β)
  (hfg : ∀ acc b, f acc b = g acc b) :
  List.foldl f init l = List.foldl g init l := by
  induction l generalizing init with
  | nil =>
      simp
  | cons b bs ih =>
      simp [hfg, ih]

private theorem list_foldl_congr_mem
  {α β : Type}
  (f g : α → β → α)
  (init : α)
  (l : List β)
  (hfg : ∀ acc b, b ∈ l → f acc b = g acc b) :
  List.foldl f init l = List.foldl g init l := by
  induction l generalizing init with
  | nil =>
      simp
  | cons b bs ih =>
      have hHead : f init b = g init b := by
        exact hfg init b (by simp)
      calc
        List.foldl f init (b :: bs)
            = List.foldl f (f init b) bs := by
                rfl
        _ = List.foldl f (g init b) bs := by
              rw [hHead]
        _ = List.foldl g (g init b) bs := by
              apply ih
              intro acc b' hb'
              exact hfg acc b' (by simp [hb'])
        _ = List.foldl g init (b :: bs) := by
              rfl

private theorem coeffAt_eq_get_of_size_d
  (a : Coeffs) (ha : a.size = d) (i : Nat) (hi : i < d) :
  coeffAt a i = a[i]'(by simpa [ha] using hi) := by
  unfold coeffAt
  have his : i < a.size := by simpa [ha] using hi
  simp [hi, Array.getD, his]

private theorem get_eq_coeffAt_of_size_d
  (a : Coeffs) (ha : a.size = d) (i : Nat) (hi : i < d) :
  a[i]'(by simpa [ha] using hi) = coeffAt a i := by
  simpa using (coeffAt_eq_get_of_size_d a ha i hi).symm

private theorem coeffAt_subRq
  (x y : Coeffs) (k : Nat) (hk : k < d) :
  coeffAt (subRq x y) k = coeffAt x k - coeffAt y k := by
  unfold subRq coeffAt
  simp [hk, Array.getD]

private theorem coeffAt_vecAdd_of_size_d
  (x y : Coeffs)
  (hx : x.size = d) (hy : y.size = d)
  (k : Nat) (hk : k < d) :
  coeffAt (vecAdd x y) k = coeffAt x k + coeffAt y k := by
  have hxy : x.size = y.size := by simpa [hx, hy]
  have hkx : k < x.size := by simpa [hx] using hk
  have hky : k < y.size := by simpa [hy] using hk
  unfold coeffAt
  simp [vecAdd, hxy, hk, Array.getD, hkx, hky, coeffAt, hx, hy]

private theorem foldl_vecAdd_size_d
  (l : List Nat) (t : Nat → Coeffs) (init : Coeffs)
  (hinit : init.size = d)
  (ht : ∀ j, (t j).size = d) :
  (l.foldl (fun acc j => vecAdd acc (t j)) init).size = d := by
  induction l generalizing init with
  | nil =>
      simpa using hinit
  | cons j js ih =>
      have htj : (t j).size = d := ht j
      have hEq : init.size = (t j).size := by simpa [hinit, htj]
      have hinit' : (vecAdd init (t j)).size = d := by
        calc
          (vecAdd init (t j)).size = init.size := vecAdd_size_of_eq hEq
          _ = d := hinit
      simpa [List.foldl] using ih (init := vecAdd init (t j)) hinit'

private theorem coeffAt_foldl_vecAdd
  (l : List Nat) (t : Nat → Coeffs) (init : Coeffs)
  (hinit : init.size = d)
  (ht : ∀ j, (t j).size = d)
  (k : Nat) (hk : k < d) :
  coeffAt (l.foldl (fun acc j => vecAdd acc (t j)) init) k =
    l.foldl (fun acc j => acc + coeffAt (t j) k) (coeffAt init k) := by
  induction l generalizing init with
  | nil =>
      simp
  | cons j js ih =>
      have htj : (t j).size = d := ht j
      have hEq : init.size = (t j).size := by simpa [hinit, htj]
      have hinit' : (vecAdd init (t j)).size = d := by
        calc
          (vecAdd init (t j)).size = init.size := vecAdd_size_of_eq hEq
          _ = d := hinit
      have hCoeff :
          coeffAt (vecAdd init (t j)) k = coeffAt init k + coeffAt (t j) k := by
        exact coeffAt_vecAdd_of_size_d init (t j) hinit htj k hk
      calc
        coeffAt ((j :: js).foldl (fun acc j => vecAdd acc (t j)) init) k
            = coeffAt (js.foldl (fun acc j => vecAdd acc (t j)) (vecAdd init (t j))) k := by
                rfl
        _ = js.foldl (fun acc j => acc + coeffAt (t j) k) (coeffAt (vecAdd init (t j)) k) := by
              exact ih (init := vecAdd init (t j)) hinit'
        _ = js.foldl (fun acc j => acc + coeffAt (t j) k) (coeffAt init k + coeffAt (t j) k) := by
              rw [hCoeff]
        _ = (j :: js).foldl (fun acc j => acc + coeffAt (t j) k) (coeffAt init k) := by
              rfl

private theorem dotRq_size (xs ys : Array Coeffs) :
  (dotRq xs ys).size = d := by
  unfold dotRq
  have hinit : zeroRq.size = d := by simp [zeroRq, d]
  have ht : ∀ j, (mulRq (xs.getD j zeroRq) (ys.getD j zeroRq)).size = d := by
    intro j
    simp [mulRq_size]
  simpa using foldl_vecAdd_size_d
    (l := List.range (Nat.min xs.size ys.size))
    (t := fun j => mulRq (xs.getD j zeroRq) (ys.getD j zeroRq))
    (init := zeroRq)
    hinit
    ht

private theorem coeffAt_dotRq
  (xs ys : Array Coeffs) (k : Nat) (hk : k < d) :
  coeffAt (dotRq xs ys) k =
    (List.range (Nat.min xs.size ys.size)).foldl
      (fun acc j => acc + coeffAt (mulRq (xs.getD j zeroRq) (ys.getD j zeroRq)) k)
      0 := by
  unfold dotRq
  have hinit : zeroRq.size = d := by simp [zeroRq, d]
  have ht : ∀ j, (mulRq (xs.getD j zeroRq) (ys.getD j zeroRq)).size = d := by
    intro j
    simp [mulRq_size]
  simpa [coeffAt_zeroRq] using coeffAt_foldl_vecAdd
    (l := List.range (Nat.min xs.size ys.size))
    (t := fun j => mulRq (xs.getD j zeroRq) (ys.getD j zeroRq))
    (init := zeroRq)
    hinit
    ht
    k
    hk

private theorem subVec_getD_of_lt
  (n : Nat) (v1 v2 : Array Coeffs) (j : Nat) (hj : j < n) :
  (subVec n v1 v2).getD j zeroRq = subRq (v1.getD j zeroRq) (v2.getD j zeroRq) := by
  unfold subVec
  simp [Array.getD, hj]

private theorem rawConvCoeff_sub_right
  (a b c : Coeffs) (n : Nat) :
  rawConvCoeff a (subRq b c) n = rawConvCoeff a b n - rawConvCoeff a c n := by
  let g : Nat → F := fun j =>
    if hIn : j ≤ n ∧ n - j < d then
      coeffAt a j * coeffAt (subRq b c) (n - j)
    else
      0
  let g1 : Nat → F := fun j =>
    if hIn : j ≤ n ∧ n - j < d then
      coeffAt a j * coeffAt b (n - j)
    else
      0
  let g2 : Nat → F := fun j =>
    if hIn : j ≤ n ∧ n - j < d then
      coeffAt a j * coeffAt c (n - j)
    else
      0
  have hLeft :
      rawConvCoeff a (subRq b c) n =
        (List.range d).foldl (fun acc j => acc + g j) 0 := by
    unfold rawConvCoeff g
    apply list_foldl_congr
    intro acc j
    by_cases hIn : j ≤ n ∧ n - j < d
    · simp [hIn]
    · simp [hIn]
  have hRight1 :
      rawConvCoeff a b n =
        (List.range d).foldl (fun acc j => acc + g1 j) 0 := by
    unfold rawConvCoeff g1
    apply list_foldl_congr
    intro acc j
    by_cases hIn : j ≤ n ∧ n - j < d
    · simp [hIn]
    · simp [hIn]
  have hRight2 :
      rawConvCoeff a c n =
        (List.range d).foldl (fun acc j => acc + g2 j) 0 := by
    unfold rawConvCoeff g2
    apply list_foldl_congr
    intro acc j
    by_cases hIn : j ≤ n ∧ n - j < d
    · simp [hIn]
    · simp [hIn]
  have hTerm : ∀ j, g j = g1 j - g2 j := by
    intro j
    by_cases hIn : j ≤ n ∧ n - j < d
    · have hSub : coeffAt (subRq b c) (n - j) = coeffAt b (n - j) - coeffAt c (n - j) := by
        exact coeffAt_subRq b c (n - j) hIn.2
      calc
        g j = coeffAt a j * coeffAt (subRq b c) (n - j) := by simp [g, hIn]
        _ = coeffAt a j * coeffAt b (n - j) - coeffAt a j * coeffAt c (n - j) := by
              simpa [hSub] using f_mul_sub (coeffAt a j) (coeffAt b (n - j)) (coeffAt c (n - j))
        _ = g1 j - g2 j := by simp [g1, g2, hIn]
    · simp [g, g1, g2, hIn]
  have hfold := foldl_sub_linearity_F (l := List.range d) g1 g2 0 0
  have hzero : ((0 : F) - 0) = 0 := by
    exact (F.sub_eq_zero_iff (0 : F) 0).2 rfl
  have hfold0 :
      (List.range d).foldl (fun acc j => acc + (g1 j - g2 j)) 0 =
        (List.range d).foldl (fun acc j => acc + g1 j) 0 -
          (List.range d).foldl (fun acc j => acc + g2 j) 0 := by
    simpa [hzero] using hfold
  calc
    rawConvCoeff a (subRq b c) n
        = (List.range d).foldl (fun acc j => acc + g j) 0 := hLeft
    _ = (List.range d).foldl (fun acc j => acc + (g1 j - g2 j)) 0 := by
          apply list_foldl_congr
          intro acc j
          exact congrArg (fun t => acc + t) (hTerm j)
    _ = (List.range d).foldl (fun acc j => acc + g1 j) 0 -
          (List.range d).foldl (fun acc j => acc + g2 j) 0 := hfold0
    _ = rawConvCoeff a b n - rawConvCoeff a c n := by
          simp [hRight1, hRight2]

set_option maxHeartbeats 2000000 in
private theorem coeffAt_mulRq_sub_right
  (a b c : Coeffs) (i : Nat) (hi : i < d) :
  coeffAt (mulRq a (subRq b c)) i =
    coeffAt (subRq (mulRq a b) (mulRq a c)) i := by
  by_cases hLt26 : i < 26
  · calc
      coeffAt (mulRq a (subRq b c)) i
          = rawConvCoeff a (subRq b c) i
              - rawConvCoeff a (subRq b c) (i + 54)
              + rawConvCoeff a (subRq b c) (i + 81) := by
                exact coeffAt_mulRq_lt26 a (subRq b c) i hi hLt26
      _ = (rawConvCoeff a b i - rawConvCoeff a c i)
            - (rawConvCoeff a b (i + 54) - rawConvCoeff a c (i + 54))
            + (rawConvCoeff a b (i + 81) - rawConvCoeff a c (i + 81)) := by
              simp [rawConvCoeff_sub_right]
      _ = (rawConvCoeff a b i - rawConvCoeff a b (i + 54) + rawConvCoeff a b (i + 81))
            - (rawConvCoeff a c i - rawConvCoeff a c (i + 54) + rawConvCoeff a c (i + 81)) := by
              grind
      _ = coeffAt (mulRq a b) i - coeffAt (mulRq a c) i := by
            simp [coeffAt_mulRq_lt26, hi, hLt26]
      _ = coeffAt (subRq (mulRq a b) (mulRq a c)) i := by
            symm
            exact coeffAt_subRq (mulRq a b) (mulRq a c) i hi
  · by_cases hEq26 : i = 26
    · subst hEq26
      have hi26 : (26 : Nat) < d := by decide
      calc
        coeffAt (mulRq a (subRq b c)) 26
            = rawConvCoeff a (subRq b c) 26
                - rawConvCoeff a (subRq b c) 80 := by
                  exact coeffAt_mulRq_eq26 a (subRq b c) 26 hi26 rfl
        _ = (rawConvCoeff a b 26 - rawConvCoeff a c 26)
              - (rawConvCoeff a b 80 - rawConvCoeff a c 80) := by
                simp [rawConvCoeff_sub_right]
        _ = (rawConvCoeff a b 26 - rawConvCoeff a b 80)
              - (rawConvCoeff a c 26 - rawConvCoeff a c 80) := by
                grind
        _ = coeffAt (mulRq a b) 26 - coeffAt (mulRq a c) 26 := by
              simp [coeffAt_mulRq_eq26, hi26]
        _ = coeffAt (subRq (mulRq a b) (mulRq a c)) 26 := by
              symm
              exact coeffAt_subRq (mulRq a b) (mulRq a c) 26 hi26
    · have hGt26 : 26 < i := by omega
      calc
        coeffAt (mulRq a (subRq b c)) i
            = rawConvCoeff a (subRq b c) i
                - rawConvCoeff a (subRq b c) (i + 27) := by
                  exact coeffAt_mulRq_gt26 a (subRq b c) i hi hGt26
        _ = (rawConvCoeff a b i - rawConvCoeff a c i)
              - (rawConvCoeff a b (i + 27) - rawConvCoeff a c (i + 27)) := by
                simp [rawConvCoeff_sub_right]
        _ = (rawConvCoeff a b i - rawConvCoeff a b (i + 27))
              - (rawConvCoeff a c i - rawConvCoeff a c (i + 27)) := by
                grind
        _ = coeffAt (mulRq a b) i - coeffAt (mulRq a c) i := by
              simp [coeffAt_mulRq_gt26, hi, hGt26]
        _ = coeffAt (subRq (mulRq a b) (mulRq a c)) i := by
              symm
              exact coeffAt_subRq (mulRq a b) (mulRq a c) i hi

set_option maxHeartbeats 800000 in
theorem mulRq_vecAdd_right
  (a b c : Coeffs)
  (hb : b.size = d)
  (hc : c.size = d) :
  mulRq a (vecAdd b c) = vecAdd (mulRq a b) (mulRq a c) := by
  exact SuperNeo.mulRq_vecAdd_right a b c hb hc

private theorem vecScale_zero_of_size_d
  (x : Coeffs)
  (hx : x.size = d) :
  vecScale (0 : F) x = zeroRq := by
  apply Array.ext
  · simp [vecScale, zeroRq, hx]
  · intro i hi₁ hi₂
    have hi : i < d := by
      simpa [vecScale, hx, zeroRq] using hi₁
    calc
      (vecScale (0 : F) x)[i]'hi₁ = coeffAt (vecScale (0 : F) x) i := by
        exact get_eq_coeffAt_of_size_d _ (by simp [vecScale, hx]) i hi
      _ = (0 : F) * coeffAt x i := by
        exact coeffAt_vecScale_of_size_d (s := (0 : F)) (x := x) hx i hi
      _ = 0 := by simpa using (Lean.Grind.Fin.zero_mul (n := Goldilocks.q) (coeffAt x i))
      _ = coeffAt zeroRq i := (coeffAt_zeroRq i).symm
      _ = zeroRq[i]'hi₂ := by
        exact coeffAt_eq_get_of_size_d zeroRq zeroRq_size i hi

private theorem mulRq_zero_right (a : Coeffs) :
    mulRq a zeroRq = zeroRq := by
  calc
    mulRq a zeroRq = mulRq a (vecScale (0 : F) oneRq) := by
      rw [vecScale_zero_of_size_d oneRq oneRq_size]
    _ = vecScale (0 : F) (mulRq a oneRq) := by
      exact mulRq_vecScale_right (s := (0 : F)) (a := a) (x := oneRq) oneRq_size
    _ = zeroRq := by
      exact vecScale_zero_of_size_d (mulRq a oneRq) (mulRq_size a oneRq)

set_option maxHeartbeats 800000 in
private theorem foldl_vecAdd_mul_right_gen
  (delta acc : Coeffs)
  (hacc : acc.size = d)
  (l : List Nat)
  (t : Nat → Coeffs)
  (ht : ∀ j, (t j).size = d) :
  l.foldl (fun acc' j => vecAdd acc' (mulRq delta (t j))) (mulRq delta acc) =
    mulRq delta (l.foldl (fun acc' j => vecAdd acc' (t j)) acc) := by
  induction l generalizing acc with
  | nil =>
      simp
  | cons j js ih =>
      have htj : (t j).size = d := ht j
      have hnext : (vecAdd acc (t j)).size = d := by
        calc
          (vecAdd acc (t j)).size = acc.size := by
            exact vecAdd_size_of_eq (hacc.trans htj.symm)
          _ = d := hacc
      simpa [List.foldl, mulRq_vecAdd_right delta acc (t j) hacc htj] using
        ih (vecAdd acc (t j)) hnext

set_option maxHeartbeats 800000 in
private theorem dotRq_smulVec_derived
  (xs : Array Coeffs)
  (delta : Coeffs)
  (v : Array Coeffs) :
  dotRq xs (smulVec delta v) = mulRq delta (dotRq xs v) := by
  let n := Nat.min xs.size v.size
  let t : Nat → Coeffs := fun j => mulRq (xs.getD j zeroRq) (v.getD j zeroRq)
  have ht : ∀ j, (t j).size = d := by
    intro j
    simp [t, mulRq_size]
  have hfold :
      (List.range n).foldl (fun acc j => vecAdd acc (mulRq delta (t j))) zeroRq =
        mulRq delta ((List.range n).foldl (fun acc j => vecAdd acc (t j)) zeroRq) := by
    calc
      (List.range n).foldl (fun acc j => vecAdd acc (mulRq delta (t j))) zeroRq
          =
        (List.range n).foldl (fun acc j => vecAdd acc (mulRq delta (t j))) (mulRq delta zeroRq) := by
            rw [mulRq_zero_right]
      _ = mulRq delta ((List.range n).foldl (fun acc j => vecAdd acc (t j)) zeroRq) := by
            exact foldl_vecAdd_mul_right_gen delta zeroRq zeroRq_size (List.range n) t ht
  calc
    dotRq xs (smulVec delta v)
        = (List.range n).foldl (fun acc j => vecAdd acc (mulRq delta (t j))) zeroRq := by
            unfold dotRq
            rw [show Nat.min xs.size (smulVec delta v).size = n by simp [n, smulVec]]
            apply list_foldl_congr_mem
            intro acc j hjMem
            have hj : j < n := by simpa [n, List.mem_range] using hjMem
            have hjv : j < v.size := Nat.lt_of_lt_of_le hj (Nat.min_le_right _ _)
            have hget :
                (smulVec delta v).getD j zeroRq = mulRq delta (v.getD j zeroRq) := by
              unfold smulVec
              simp [hjv]
            calc
              vecAdd acc (mulRq (xs.getD j zeroRq) ((smulVec delta v).getD j zeroRq))
                  = vecAdd acc (mulRq (xs.getD j zeroRq) (mulRq delta (v.getD j zeroRq))) := by
                      rw [hget]
              _ = vecAdd acc (mulRq delta (mulRq (xs.getD j zeroRq) (v.getD j zeroRq))) := by
                    rw [← mulRq_leftActionComm delta (xs.getD j zeroRq) (v.getD j zeroRq)]
              _ = vecAdd acc (mulRq delta (t j)) := by simp [t]
    _ = mulRq delta ((List.range n).foldl (fun acc j => vecAdd acc (t j)) zeroRq) := hfold
    _ = mulRq delta (dotRq xs v) := by
          unfold dotRq
          simp [n, t]

theorem matVecMul_smulVec_derived
  {params : AjtaiParams}
  (matrixFlat : Array Coeffs)
  (delta : Coeffs)
  (v : Array Coeffs) :
  matVecMul params matrixFlat (smulVec delta v) =
    smulVec delta (matVecMul params matrixFlat v) := by
  apply Array.ext
  · simp [matVecMul, smulVec]
  · intro i hi₁ hi₂
    simpa [matVecMul, smulVec] using
      dotRq_smulVec_derived (matRow params.msgLen matrixFlat i) delta v

set_option maxHeartbeats 800000 in
private theorem smulVec_comm_derived
  {params : AjtaiParams}
  (delta1 delta2 : Coeffs)
  (v : Array Coeffs) :
  smulVec delta1 (smulVec delta2 v) = smulVec delta2 (smulVec delta1 v) := by
  unfold smulVec
  apply Array.ext
  · simp
  · intro i hi₁ hi₂
    have hi : i < v.size := by simpa using hi₁
    rw [Array.getElem_map, Array.getElem_map, Array.getElem_map, Array.getElem_map]
    exact mulRqPhi_leftActionComm delta1 delta2 v[i]


set_option maxHeartbeats 800000 in
theorem dotRq_subVec_linearity
  {params : AjtaiParams}
  (xs v1 v2 : Array Coeffs)
  (hv1 : v1.size = params.msgLen)
  (hv2 : v2.size = params.msgLen) :
  dotRq xs (subVec params.msgLen v1 v2) = subRq (dotRq xs v1) (dotRq xs v2) := by
  apply Array.ext
  · simp [dotRq_size, subRq]
  · intro i hi₁ hi₂
    have hi : i < d := by
      simpa [dotRq_size] using hi₁
    let n : Nat := Nat.min xs.size params.msgLen
    let t1 : Nat → F := fun j =>
      coeffAt (mulRq (xs.getD j zeroRq) (v1.getD j zeroRq)) i
    let t2 : Nat → F := fun j =>
      coeffAt (mulRq (xs.getD j zeroRq) (v2.getD j zeroRq)) i
    let tL : Nat → F := fun j =>
      coeffAt (mulRq (xs.getD j zeroRq) ((subVec params.msgLen v1 v2).getD j zeroRq)) i
    have htermL :
        ∀ j, j < n → tL j = t1 j - t2 j := by
      intro j hj
      have hjMsg : j < params.msgLen := Nat.lt_of_lt_of_le hj (Nat.min_le_right _ _)
      have hGet :
          (subVec params.msgLen v1 v2).getD j zeroRq =
            subRq (v1.getD j zeroRq) (v2.getD j zeroRq) :=
        subVec_getD_of_lt params.msgLen v1 v2 j hjMsg
      have hMulCoeff :
          coeffAt
              (mulRq (xs.getD j zeroRq) ((subVec params.msgLen v1 v2).getD j zeroRq))
              i
            =
          coeffAt
              (subRq
                (mulRq (xs.getD j zeroRq) (v1.getD j zeroRq))
                (mulRq (xs.getD j zeroRq) (v2.getD j zeroRq)))
              i := by
        rw [hGet]
        exact coeffAt_mulRq_sub_right
          (xs.getD j zeroRq) (v1.getD j zeroRq) (v2.getD j zeroRq) i hi
      have hCoeff :
          coeffAt
              (subRq
                (mulRq (xs.getD j zeroRq) (v1.getD j zeroRq))
                (mulRq (xs.getD j zeroRq) (v2.getD j zeroRq)))
              i
            =
          coeffAt (mulRq (xs.getD j zeroRq) (v1.getD j zeroRq)) i -
            coeffAt (mulRq (xs.getD j zeroRq) (v2.getD j zeroRq)) i := by
        exact coeffAt_subRq
          (mulRq (xs.getD j zeroRq) (v1.getD j zeroRq))
          (mulRq (xs.getD j zeroRq) (v2.getD j zeroRq))
          i
          hi
      calc
        tL j = coeffAt
                (subRq
                  (mulRq (xs.getD j zeroRq) (v1.getD j zeroRq))
                  (mulRq (xs.getD j zeroRq) (v2.getD j zeroRq)))
                i := by
                  simpa [tL] using hMulCoeff
        _ = coeffAt (mulRq (xs.getD j zeroRq) (v1.getD j zeroRq)) i -
              coeffAt (mulRq (xs.getD j zeroRq) (v2.getD j zeroRq)) i := hCoeff
        _ = t1 j - t2 j := by
              simp [t1, t2]
    have hleftFold :
        (List.range n).foldl (fun acc j => acc + tL j) 0 =
          (List.range n).foldl (fun acc j => acc + (t1 j - t2 j)) 0 := by
      exact list_foldl_congr_mem
        (fun acc j => acc + tL j)
        (fun acc j => acc + (t1 j - t2 j))
        0
        (List.range n)
        (by
          intro acc j hjMem
          have hj : j < n := by simpa [List.mem_range] using hjMem
          exact congrArg (fun t => acc + t) (htermL j hj))
    have hfold := foldl_sub_linearity_F (l := List.range n) t1 t2 0 0
    have hzero : ((0 : F) - 0) = 0 := by
      exact (F.sub_eq_zero_iff (0 : F) 0).2 rfl
    have hfold0 :
        (List.range n).foldl (fun acc j => acc + (t1 j - t2 j)) 0 =
          (List.range n).foldl (fun acc j => acc + t1 j) 0 -
            (List.range n).foldl (fun acc j => acc + t2 j) 0 := by
      simpa [hzero] using hfold
    have hLeft :
        coeffAt (dotRq xs (subVec params.msgLen v1 v2)) i =
          (List.range n).foldl (fun acc j => acc + tL j) 0 := by
      have hSubSize : (subVec params.msgLen v1 v2).size = params.msgLen := by
        simp [subVec_size]
      calc
        coeffAt (dotRq xs (subVec params.msgLen v1 v2)) i
            = (List.range (Nat.min xs.size (subVec params.msgLen v1 v2).size)).foldl
                (fun acc j =>
                  acc + coeffAt (mulRq (xs.getD j zeroRq) ((subVec params.msgLen v1 v2).getD j zeroRq)) i)
                0 := coeffAt_dotRq xs (subVec params.msgLen v1 v2) i hi
        _ = (List.range n).foldl (fun acc j => acc + tL j) 0 := by
              simp [n, hSubSize, tL]
    have hRight1 :
        coeffAt (dotRq xs v1) i = (List.range n).foldl (fun acc j => acc + t1 j) 0 := by
      calc
        coeffAt (dotRq xs v1) i
            = (List.range (Nat.min xs.size v1.size)).foldl
                (fun acc j => acc + coeffAt (mulRq (xs.getD j zeroRq) (v1.getD j zeroRq)) i)
                0 := coeffAt_dotRq xs v1 i hi
        _ = (List.range n).foldl (fun acc j => acc + t1 j) 0 := by
              simp [n, hv1, t1]
    have hRight2 :
        coeffAt (dotRq xs v2) i = (List.range n).foldl (fun acc j => acc + t2 j) 0 := by
      calc
        coeffAt (dotRq xs v2) i
            = (List.range (Nat.min xs.size v2.size)).foldl
                (fun acc j => acc + coeffAt (mulRq (xs.getD j zeroRq) (v2.getD j zeroRq)) i)
                0 := coeffAt_dotRq xs v2 i hi
        _ = (List.range n).foldl (fun acc j => acc + t2 j) 0 := by
              simp [n, hv2, t2]
    have hmainCoeff :
        coeffAt (dotRq xs (subVec params.msgLen v1 v2)) i =
          coeffAt (subRq (dotRq xs v1) (dotRq xs v2)) i := by
      calc
        coeffAt (dotRq xs (subVec params.msgLen v1 v2)) i
            = (List.range n).foldl (fun acc j => acc + tL j) 0 := hLeft
        _ = (List.range n).foldl (fun acc j => acc + (t1 j - t2 j)) 0 := hleftFold
        _ = (List.range n).foldl (fun acc j => acc + t1 j) 0 -
              (List.range n).foldl (fun acc j => acc + t2 j) 0 := hfold0
        _ = coeffAt (dotRq xs v1) i - coeffAt (dotRq xs v2) i := by
              rw [hRight1, hRight2]
        _ = coeffAt (subRq (dotRq xs v1) (dotRq xs v2)) i := by
              symm
              exact coeffAt_subRq (dotRq xs v1) (dotRq xs v2) i hi
    have hiR : i < d := by simpa [subRq] using hi₂
    simpa [coeffAt, subRq, hi, hiR, Array.getD, hi₁, hi₂, dotRq_size] using hmainCoeff

set_option maxHeartbeats 800000 in
theorem subRq_eq_zero_of_shapes
  (x y : Coeffs)
  (hx : hasRingDegreeShape x)
  (hy : hasRingDegreeShape y)
  (hsub : subRq x y = zeroRq) :
  x = y := by
  apply Array.ext
  · exact hx.trans hy.symm
  · intro i hix hiy
    have hixd : i < d := by
      calc
        i < x.size := hix
        _ = d := hx
    have hiyd : i < d := by
      calc
        i < y.size := hiy
        _ = d := hy
    have hCoeffSub :
        coeffAt x i - coeffAt y i = (0 : F) := by
      have hEqCoeff :
          coeffAt (subRq x y) i = coeffAt zeroRq i := by
        simpa using congrArg (fun a => coeffAt a i) hsub
      calc
        coeffAt x i - coeffAt y i = coeffAt (subRq x y) i := by
          symm
          exact coeffAt_subRq x y i hixd
        _ = coeffAt zeroRq i := hEqCoeff
        _ = 0 := coeffAt_zeroRq i
    have hCoeff : coeffAt x i = coeffAt y i :=
      (F.sub_eq_zero_iff (coeffAt x i) (coeffAt y i)).1 hCoeffSub
    calc
      x[i]'hix = coeffAt x i := (get_eq_coeffAt_of_size_d x hx i hixd)
      _ = coeffAt y i := hCoeff
      _ = y[i]'hiy := (coeffAt_eq_get_of_size_d y hy i hiyd)

private theorem allRingDegreeShape_get_of_lt
  {v : Array Coeffs}
  (hShape : allRingDegreeShape v)
  {i : Nat}
  (hi : i < v.size) :
  hasRingDegreeShape (v[i]'hi) :=
  hShape ⟨i, hi⟩

private theorem allRingDegreeShape_smulVec
  (delta : Coeffs) (v : Array Coeffs) :
  allRingDegreeShape (smulVec delta v) := by
  intro i
  rcases i with ⟨idx, hidx⟩
  have hMap :
      (smulVec delta v)[idx]'hidx =
        mulRqPhi delta (v[idx]'(by simpa [smulVec] using hidx)) := by
    simp [smulVec]
  unfold hasRingDegreeShape
  simpa [hMap, mulRqPhi_size]

theorem subVec_ne_zero_of_ne
  (n : Nat) (v1 v2 : Array Coeffs) :
  v1.size = n →
  v2.size = n →
  allRingDegreeShape v1 →
  allRingDegreeShape v2 →
  v1 ≠ v2 →
  subVec n v1 v2 ≠ zeroVec n := by
  intro hv1 hv2 hShape1 hShape2 hNe hEq
  apply hNe
  apply Array.ext
  · simpa [hv1, hv2]
  · intro i hi1 hi2
    have hi : i < n := by simpa [hv1] using hi1
    have hi2' : i < v2.size := by simpa [hv2] using hi
    have hEntry :
        (subVec n v1 v2).getD i zeroRq = (zeroVec n).getD i zeroRq := by
      simpa using congrArg (fun a => a.getD i zeroRq) hEq
    have hSub : subRq (v1.getD i zeroRq) (v2.getD i zeroRq) = zeroRq := by
      have hSubVec :
          (subVec n v1 v2).getD i zeroRq =
            subRq (v1.getD i zeroRq) (v2.getD i zeroRq) :=
        subVec_getD_of_lt n v1 v2 i hi
      have hEq0 :
          subRq (v1.getD i zeroRq) (v2.getD i zeroRq) =
            (zeroVec n).getD i zeroRq := by
        exact hSubVec.symm.trans hEntry
      simpa [zeroVec, hi] using hEq0
    have hShapeI1 : hasRingDegreeShape (v1.getD i zeroRq) := by
      have hShapeAt : hasRingDegreeShape (v1[i]'hi1) := allRingDegreeShape_get_of_lt hShape1 hi1
      simpa [Array.getD, hi1] using hShapeAt
    have hShapeI2 : hasRingDegreeShape (v2.getD i zeroRq) := by
      have hShapeAt : hasRingDegreeShape (v2[i]'hi2') := allRingDegreeShape_get_of_lt hShape2 hi2'
      simpa [Array.getD, hi2'] using hShapeAt
    have hEqGet : v1.getD i zeroRq = v2.getD i zeroRq :=
      subRq_eq_zero_of_shapes _ _ hShapeI1 hShapeI2 hSub
    calc
      v1[i]'hi1 = v1.getD i zeroRq := by simp [Array.getD, hi1]
      _ = v2.getD i zeroRq := hEqGet
      _ = v2[i]'hi2 := by simp [Array.getD, hi2]

set_option maxHeartbeats 800000 in
theorem matVecMul_subVec
  {params : AjtaiParams}
  (matrixFlat : Array Coeffs)
  (v1 v2 : Array Coeffs)
  (hv1 : v1.size = params.msgLen)
  (hv2 : v2.size = params.msgLen) :
  matVecMul params matrixFlat (subVec params.msgLen v1 v2) =
    subVec params.kappa (matVecMul params matrixFlat v1) (matVecMul params matrixFlat v2) := by
  apply Array.ext
  · simp [matVecMul, subVec]
  · intro i hi1 hi2
    have hiK : i < params.kappa := by simpa [matVecMul] using hi1
    have hGet :
        (matVecMul params matrixFlat (subVec params.msgLen v1 v2)).getD i zeroRq =
          subRq
            ((matVecMul params matrixFlat v1).getD i zeroRq)
            ((matVecMul params matrixFlat v2).getD i zeroRq) := by
      calc
        (matVecMul params matrixFlat (subVec params.msgLen v1 v2)).getD i zeroRq
            = dotRq (matRow params.msgLen matrixFlat i) (subVec params.msgLen v1 v2) := by
                simp [matVecMul, Array.getD, hiK]
        _ = subRq (dotRq (matRow params.msgLen matrixFlat i) v1)
              (dotRq (matRow params.msgLen matrixFlat i) v2) := by
              exact dotRq_subVec_linearity (params := params)
                (matRow params.msgLen matrixFlat i) v1 v2 hv1 hv2
        _ = subRq
              ((matVecMul params matrixFlat v1).getD i zeroRq)
              ((matVecMul params matrixFlat v2).getD i zeroRq) := by
              simp [matVecMul, Array.getD, hiK]
    calc
      (matVecMul params matrixFlat (subVec params.msgLen v1 v2))[i]'hi1
          = (matVecMul params matrixFlat (subVec params.msgLen v1 v2)).getD i zeroRq := by
              simp [Array.getD, hi1]
      _ = subRq
            ((matVecMul params matrixFlat v1).getD i zeroRq)
            ((matVecMul params matrixFlat v2).getD i zeroRq) := hGet
      _ = (subVec params.kappa (matVecMul params matrixFlat v1) (matVecMul params matrixFlat v2)).getD i zeroRq := by
            simp [subVec, hiK]
      _ = (subVec params.kappa (matVecMul params matrixFlat v1) (matVecMul params matrixFlat v2))[i]'hi2 := by
            simp [Array.getD, hiK]

private theorem acc_le_foldl_max_of_fn
  {α : Type}
  (l : List α)
  (f : α → Nat)
  (acc : Nat) :
  acc ≤ l.foldl (fun a x => Nat.max a (f x)) acc := by
  induction l generalizing acc with
  | nil =>
      simp
  | cons a t ih =>
      have h₁ : acc ≤ Nat.max acc (f a) := Nat.le_max_left _ _
      have h₂ : Nat.max acc (f a) ≤ t.foldl (fun b x => Nat.max b (f x)) (Nat.max acc (f a)) :=
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

private theorem foldl_max_normInfF_replicate_zero
  (n acc : Nat) :
  (List.replicate n (0 : F)).foldl
      (fun a y => Nat.max a (normInfF y))
      acc = acc := by
  induction n generalizing acc with
  | zero =>
      simp
  | succ n ih =>
      calc
        (List.replicate (n + 1) (0 : F)).foldl
            (fun a y => Nat.max a (normInfF y))
            acc
            = (List.replicate n (0 : F)).foldl
                (fun a y => Nat.max a (normInfF y))
                (Nat.max acc (normInfF (0 : F))) := by
                  simp [List.replicate, List.foldl]
        _ = (List.replicate n (0 : F)).foldl
              (fun a y => Nat.max a (normInfF y))
              acc := by
                rw [SuperNeo.normInfF_zero]
                simp
        _ = acc := ih acc

private theorem normInfCoeffs_zeroRq : normInfCoeffs zeroRq = 0 := by
  unfold normInfCoeffs zeroRq
  rw [← Array.foldl_toList, Array.toList_replicate]
  simpa using foldl_max_normInfF_replicate_zero d 0

private theorem normInfCoeffs_le_normInfVec_of_mem
  {v : Array Coeffs} {x : Coeffs}
  (hx : x ∈ v.toList) :
  normInfCoeffs x ≤ normInfVec v := by
  unfold normInfVec
  rw [← Array.foldl_toList]
  exact le_foldl_max_of_mem_fn v.toList normInfCoeffs 0 x hx

private theorem normInfF_coeffAt_le_normInfCoeffs
  (a : Coeffs) (i : Nat) (hi : i < d) :
  normInfF (coeffAt a i) ≤ normInfCoeffs a := by
  by_cases his : i < a.size
  · have hmem : a[i]'his ∈ a.toList := Array.getElem_mem_toList (xs := a) his
    have hle : normInfF (a[i]'his) ≤ normInfCoeffs a := by
      unfold normInfCoeffs
      rw [← Array.foldl_toList]
      exact le_foldl_max_of_mem_fn a.toList normInfF 0 (a[i]'his) hmem
    simpa [coeffAt, hi, Array.getD, his] using hle
  · have hcoeff : coeffAt a i = (0 : F) := by
      simp [coeffAt, hi, Array.getD, his]
    rw [hcoeff]
    rw [SuperNeo.normInfF_zero]
    exact Nat.zero_le (normInfCoeffs a)

private theorem normInfCoeffs_subRq_le (x y : Coeffs) :
  normInfCoeffs (subRq x y) ≤ normInfCoeffs x + normInfCoeffs y := by
  unfold normInfCoeffs subRq
  rw [← Array.foldl_toList, Array.toList_ofFn]
  refine foldl_max_le_of_forall_le_fn
    (l := List.ofFn (fun i : Fin d => coeffAt x i.1 - coeffAt y i.1))
    (f := normInfF)
    (acc := 0)
    (m := normInfCoeffs x + normInfCoeffs y)
    (by exact Nat.zero_le _)
    ?_
  intro z hz
  rcases (List.mem_ofFn.mp hz) with ⟨i, rfl⟩
  have hx : normInfF (coeffAt x i.1) ≤ normInfCoeffs x :=
    normInfF_coeffAt_le_normInfCoeffs x i.1 i.2
  have hy : normInfF (coeffAt y i.1) ≤ normInfCoeffs y :=
    normInfF_coeffAt_le_normInfCoeffs y i.1 i.2
  have hsub :
      normInfF (coeffAt x i.1 - coeffAt y i.1) ≤
        normInfF (coeffAt x i.1) + normInfF (coeffAt y i.1) := by
    simpa [normInfF] using F.centeredAbs_sub_le (coeffAt x i.1) (coeffAt y i.1)
  exact Nat.le_trans hsub (Nat.add_le_add hx hy)

private theorem normInfCoeffs_getD_le_normInfVec
  (v : Array Coeffs) (i : Nat) :
  normInfCoeffs (v.getD i zeroRq) ≤ normInfVec v := by
  by_cases his : i < v.size
  · have hmem : v[i]'his ∈ v.toList := Array.getElem_mem_toList (xs := v) his
    have hle : normInfCoeffs (v[i]'his) ≤ normInfVec v :=
      normInfCoeffs_le_normInfVec_of_mem hmem
    simpa [Array.getD, his] using hle
  · have hget : v.getD i zeroRq = zeroRq := by
      simp [Array.getD, his]
    rw [hget]
    calc
      normInfCoeffs zeroRq = 0 := normInfCoeffs_zeroRq
      _ ≤ normInfVec v := Nat.zero_le _

private theorem normInfVec_subVec_le_derived
  (n : Nat) (v1 v2 : Array Coeffs) :
  normInfVec (subVec n v1 v2) ≤ normInfVec v1 + normInfVec v2 := by
  unfold normInfVec subVec
  rw [← Array.foldl_toList, Array.toList_ofFn]
  refine foldl_max_le_of_forall_le_fn
    (l := List.ofFn (fun i : Fin n =>
      subRq (v1.getD i.1 zeroRq) (v2.getD i.1 zeroRq)))
    (f := normInfCoeffs)
    (acc := 0)
    (m := normInfVec v1 + normInfVec v2)
    (by exact Nat.zero_le _)
    ?_
  intro z hz
  rcases (List.mem_ofFn.mp hz) with ⟨i, rfl⟩
  have hSub :
      normInfCoeffs (subRq (v1.getD i.1 zeroRq) (v2.getD i.1 zeroRq)) ≤
        normInfCoeffs (v1.getD i.1 zeroRq) + normInfCoeffs (v2.getD i.1 zeroRq) :=
    normInfCoeffs_subRq_le (v1.getD i.1 zeroRq) (v2.getD i.1 zeroRq)
  have h1 : normInfCoeffs (v1.getD i.1 zeroRq) ≤ normInfVec v1 :=
    normInfCoeffs_getD_le_normInfVec v1 i.1
  have h2 : normInfCoeffs (v2.getD i.1 zeroRq) ≤ normInfVec v2 :=
    normInfCoeffs_getD_le_normInfVec v2 i.1
  exact Nat.le_trans hSub (Nat.add_le_add h1 h2)

private theorem normInfVec_smulVec_le_of_diff
  {params : AjtaiParams}
  (laws : LatticeReductionLaws params)
  (delta : Coeffs)
  (v : Array Coeffs)
  (B : Nat)
  (hδ : samplingDiffSet laws.samplingCarrier delta)
  (hB : normInfVec v ≤ B) :
  normInfVec (smulVec delta v) ≤ 4 * params.relaxedExpansion * B := by
  unfold normInfVec smulVec
  rw [← Array.foldl_toList, Array.toList_map]
  refine foldl_max_le_of_forall_le_fn
    (l := v.toList.map (fun x => mulRqPhi delta x))
    (f := normInfCoeffs)
    (acc := 0)
    (m := 4 * params.relaxedExpansion * B)
    (by exact Nat.zero_le _)
    ?_
  intro x hx
  rcases List.mem_map.mp hx with ⟨z, hzMem, rfl⟩
  have hzNorm : normInfCoeffs z ≤ B := by
    exact Nat.le_trans (normInfCoeffs_le_normInfVec_of_mem hzMem) hB
  exact laws.strongSampling delta hδ z B hzNorm

/-- Norm-transfer boundary for extracted witness from a standard binding collision. -/
theorem bindingCollision_subWitness_norm_lt_msisNormBound
  {params : AjtaiParams}
  (_laws : LatticeReductionLaws params)
  (hExpPos : 0 < params.relaxedExpansion)
  (coll : BindingCollision params) :
  normInfVec (subVec params.msgLen coll.opening1.witness coll.opening2.witness) <
    params.msisNormBound := by
  rcases coll.opens1 with ⟨_hCwf1, _hOwf1, hNs1, _hEq1⟩
  rcases coll.opens2 with ⟨_hCwf2, _hOwf2, hNs2, _hEq2⟩

  have hSubLe :
    normInfVec (subVec params.msgLen coll.opening1.witness coll.opening2.witness) ≤
      coll.opening1.normBound + coll.opening2.normBound := by
    have h :=
      normInfVec_subVec_le_derived (n := params.msgLen)
        coll.opening1.witness coll.opening2.witness
    exact Nat.le_trans h (Nat.add_le_add hNs1 hNs2)

  have hSumLt :
      coll.opening1.normBound + coll.opening2.normBound <
        params.bindingNormBound + params.bindingNormBound :=
    Nat.add_lt_add coll.bounded1 coll.bounded2

  have hLtBB :
      normInfVec (subVec params.msgLen coll.opening1.witness coll.opening2.witness) <
        params.bindingNormBound + params.bindingNormBound :=
    Nat.lt_of_le_of_lt hSubLe hSumLt

  have h1le : 1 ≤ params.relaxedExpansion :=
    Nat.succ_le_of_lt hExpPos

  have h8le : 8 ≤ 8 * params.relaxedExpansion := by
    simpa using (Nat.mul_le_mul_left 8 h1le)

  have h2le8 : (2 : Nat) ≤ 8 := by decide
  have h2le : (2 : Nat) ≤ 8 * params.relaxedExpansion :=
    Nat.le_trans h2le8 h8le

  have hBBLe : params.bindingNormBound + params.bindingNormBound ≤ params.msisNormBound := by
    unfold AjtaiParams.msisNormBound
    have h := Nat.mul_le_mul_right params.bindingNormBound h2le
    simpa [Nat.two_mul, Nat.mul_assoc] using h

  exact Nat.lt_of_lt_of_le hLtBB hBBLe

/-- Norm-transfer boundary for extracted witness from a relaxed binding collision. -/
theorem relaxedBindingCollision_subWitness_norm_lt_msisNormBound
  {params : AjtaiParams}
  (laws : LatticeReductionLaws params)
  (hExpPos : 0 < params.relaxedExpansion)
  (coll : RelaxedBindingCollision params laws.samplingCarrier) :
  normInfVec
      (subVec params.msgLen
        (smulVec coll.delta1 coll.opening2.witness)
        (smulVec coll.delta2 coll.opening1.witness)) <
    params.msisNormBound := by
  rcases coll.opens1 with ⟨_hCwf1, _hOwf1, hNs1, _hEq1⟩
  rcases coll.opens2 with ⟨_hCwf2, _hOwf2, hNs2, _hEq2⟩

  have h4pos : 0 < (4 : Nat) := by decide
  have hPos : 0 < 4 * params.relaxedExpansion :=
    Nat.mul_pos h4pos hExpPos

  let w1 := smulVec coll.delta1 coll.opening2.witness
  let w2 := smulVec coll.delta2 coll.opening1.witness

  have hw1_le :
      normInfVec w1 ≤ (4 * params.relaxedExpansion) * coll.opening2.normBound :=
    by
      simpa [w1] using
        normInfVec_smulVec_le_of_diff
          (params := params)
          laws
          coll.delta1
          coll.opening2.witness
          coll.opening2.normBound
          coll.inDiff1
          hNs2

  have hw2_le :
      normInfVec w2 ≤ (4 * params.relaxedExpansion) * coll.opening1.normBound := by
    simpa [w2] using
      normInfVec_smulVec_le_of_diff
        (params := params)
        laws
        coll.delta2
        coll.opening1.witness
        coll.opening1.normBound
        coll.inDiff2
        hNs1

  have hsub_le :
      normInfVec (subVec params.msgLen w1 w2) ≤ normInfVec w1 + normInfVec w2 :=
    normInfVec_subVec_le_derived (n := params.msgLen) w1 w2

  have htotal_le :
      normInfVec (subVec params.msgLen w1 w2) ≤
        (4 * params.relaxedExpansion) * coll.opening2.normBound +
        (4 * params.relaxedExpansion) * coll.opening1.normBound := by
    exact Nat.le_trans hsub_le (Nat.add_le_add hw1_le hw2_le)

  have h1lt :
      (4 * params.relaxedExpansion) * coll.opening2.normBound <
        (4 * params.relaxedExpansion) * params.bindingNormBound :=
    Nat.mul_lt_mul_of_pos_left coll.bounded2 hPos

  have h2lt :
      (4 * params.relaxedExpansion) * coll.opening1.normBound <
        (4 * params.relaxedExpansion) * params.bindingNormBound :=
    Nat.mul_lt_mul_of_pos_left coll.bounded1 hPos

  have hsumlt :
      (4 * params.relaxedExpansion) * coll.opening2.normBound +
      (4 * params.relaxedExpansion) * coll.opening1.normBound <
      (4 * params.relaxedExpansion) * params.bindingNormBound +
      (4 * params.relaxedExpansion) * params.bindingNormBound :=
    Nat.add_lt_add h1lt h2lt

  have hlt :
      normInfVec (subVec params.msgLen w1 w2) <
      (4 * params.relaxedExpansion) * params.bindingNormBound +
      (4 * params.relaxedExpansion) * params.bindingNormBound :=
    Nat.lt_of_le_of_lt htotal_le hsumlt

  have hRhs :
      (4 * params.relaxedExpansion) * params.bindingNormBound +
      (4 * params.relaxedExpansion) * params.bindingNormBound =
      params.msisNormBound := by
    unfold AjtaiParams.msisNormBound
    calc
      (4 * params.relaxedExpansion) * params.bindingNormBound +
          (4 * params.relaxedExpansion) * params.bindingNormBound
          = 2 * ((4 * params.relaxedExpansion) * params.bindingNormBound) := by
              simpa using
                (Eq.symm (Nat.two_mul ((4 * params.relaxedExpansion) * params.bindingNormBound)))
      _ = (2 * (4 * params.relaxedExpansion)) * params.bindingNormBound := by
              simp [Nat.mul_assoc]
      _ = (8 * params.relaxedExpansion) * params.bindingNormBound := by
              simp [Nat.mul_left_comm, Nat.mul_comm]
      _ = 8 * params.relaxedExpansion * params.bindingNormBound := by
              simp [Nat.mul_assoc]

  exact lt_of_lt_of_eq hlt hRhs

/--
Extractor: a standard Ajtai binding collision yields a homogeneous MSIS witness.
-/
theorem msisBreakEvent_of_bindingCollision
  {params : AjtaiParams}
  (laws : LatticeReductionLaws params)
  (hExpPos : 0 < params.relaxedExpansion)
  (coll : BindingCollision params) :
  MSISBreakEvent params := by
  rcases coll.opens1 with ⟨hCwf1, hOwf1, _hNs1, hEq1⟩
  rcases coll.opens2 with ⟨_hCwf2, hOwf2, _hNs2, hEq2⟩
  let chal : MSISChallenge params :=
    ⟨Commitment.ppMatrixFlat params coll.commitment, zeroVec params.kappa⟩
  refine ⟨chal, rfl, ?_⟩
  let w := subVec params.msgLen coll.opening1.witness coll.opening2.witness
  refine ⟨{
    witness := w
    bounded := ?_
    satisfies := ?_
  }⟩
  · refine ⟨by simp [w], ?_, ?_⟩
    · have hwNeZero :
        subVec params.msgLen coll.opening1.witness coll.opening2.witness ≠
          zeroVec params.msgLen := by
        exact subVec_ne_zero_of_ne params.msgLen coll.opening1.witness coll.opening2.witness
          hOwf1.1
          hOwf2.1
          hOwf1.2
          hOwf2.2
          coll.distinct
      simpa [w, zeroVec] using hwNeZero
    · simpa [w] using
        bindingCollision_subWitness_norm_lt_msisNormBound (params := params) laws hExpPos coll
  · refine ⟨?_, ?_⟩
    · refine ⟨?_, ?_⟩
      · exact Commitment.ppMatrixFlat_size_of_wf (params := params) (c := coll.commitment) hCwf1
      · simp [chal, zeroVec]
    · calc
        matVecMul params chal.matrix w
            = subVec params.kappa
                (matVecMul params chal.matrix coll.opening1.witness)
                (matVecMul params chal.matrix coll.opening2.witness) := by
                simpa [w] using
                  matVecMul_subVec (params := params) chal.matrix coll.opening1.witness coll.opening2.witness
                    hOwf1.1
                    hOwf2.1
        _ = subVec params.kappa
              (Commitment.valueVec params coll.commitment)
              (Commitment.valueVec params coll.commitment) := by
              simp [chal, hEq1, hEq2]
        _ = zeroVec params.kappa := by
              simpa using subVec_self params.kappa (Commitment.valueVec params coll.commitment)
        _ = chal.target := by
              rfl

/--
Extractor: a relaxed Ajtai binding collision yields a homogeneous MSIS witness.
-/
theorem msisBreakEvent_of_relaxedBindingCollision
  {params : AjtaiParams}
  (laws : LatticeReductionLaws params)
  (hExpPos : 0 < params.relaxedExpansion)
  (coll : RelaxedBindingCollision params laws.samplingCarrier) :
  MSISBreakEvent params := by
  rcases coll.opens1 with ⟨hCwf1, hOwf1, _hNs1, hEq1⟩
  rcases coll.opens2 with ⟨_hCwf2, hOwf2, _hNs2, hEq2⟩
  let chal : MSISChallenge params :=
    ⟨Commitment.ppMatrixFlat params coll.commitment, zeroVec params.kappa⟩
  refine ⟨chal, rfl, ?_⟩
  let w1 := smulVec coll.delta1 coll.opening2.witness
  let w2 := smulVec coll.delta2 coll.opening1.witness
  let w := subVec params.msgLen w1 w2
  refine ⟨{
    witness := w
    bounded := ?_
    satisfies := ?_
  }⟩
  · refine ⟨by simp [w], ?_, ?_⟩
    · have hwNeZero : subVec params.msgLen w1 w2 ≠ zeroVec params.msgLen := by
        have hw1Size : w1.size = params.msgLen := by
          simpa [w1, smulVec] using hOwf2.1
        have hw2Size : w2.size = params.msgLen := by
          simpa [w2, smulVec] using hOwf1.1
        exact subVec_ne_zero_of_ne params.msgLen w1 w2
          hw1Size
          hw2Size
          (by simpa [w1] using allRingDegreeShape_smulVec coll.delta1 coll.opening2.witness)
          (by simpa [w2] using allRingDegreeShape_smulVec coll.delta2 coll.opening1.witness)
          (by simpa [w1, w2] using coll.distinct)
      simpa [w, zeroVec] using hwNeZero
    · simpa [w, w1, w2] using
        relaxedBindingCollision_subWitness_norm_lt_msisNormBound (params := params) laws hExpPos coll
  · refine ⟨?_, ?_⟩
    · refine ⟨?_, ?_⟩
      · exact Commitment.ppMatrixFlat_size_of_wf (params := params) (c := coll.commitment) hCwf1
      · simp [chal, zeroVec]
    · calc
        matVecMul params chal.matrix w
            = subVec params.kappa
                (matVecMul params chal.matrix w1)
                (matVecMul params chal.matrix w2) := by
                simpa [w] using matVecMul_subVec (params := params) chal.matrix w1 w2
                  (by simpa [w1, smulVec] using hOwf2.1)
                  (by simpa [w2, smulVec] using hOwf1.1)
        _ = subVec params.kappa
              (smulVec coll.delta1 (matVecMul params chal.matrix coll.opening2.witness))
              (smulVec coll.delta2 (matVecMul params chal.matrix coll.opening1.witness)) := by
              simp [w1, w2, matVecMul_smulVec_derived (params := params)]
        _ = subVec params.kappa
              (smulVec coll.delta1 (smulVec coll.delta2 (Commitment.valueVec params coll.commitment)))
              (smulVec coll.delta2 (smulVec coll.delta1 (Commitment.valueVec params coll.commitment))) := by
              simp [chal, hEq1, hEq2]
        _ = subVec params.kappa
              (smulVec coll.delta2 (smulVec coll.delta1 (Commitment.valueVec params coll.commitment)))
              (smulVec coll.delta2 (smulVec coll.delta1 (Commitment.valueVec params coll.commitment))) := by
              rw [smulVec_comm_derived (params := params) coll.delta1 coll.delta2
                (Commitment.valueVec params coll.commitment)]
        _ = zeroVec params.kappa := by
              simpa using
                subVec_self params.kappa
                  (smulVec coll.delta2 (smulVec coll.delta1 (Commitment.valueVec params coll.commitment)))
        _ = chal.target := by
              rfl

/-- Truth-valued probability model: `Pr P = 1` iff `P`, else `0`. -/
noncomputable def truthProb : ProbModel where
  Pr := fun P => by
    classical
    exact if P then (1 : Rat) else 0
  prNonneg := by
    intro P
    classical
    by_cases hP : P
    · simp [hP]
    · simp [hP]
  prLeOne := by
    intro P
    classical
    by_cases hP : P
    · simp [hP]
    · simp [hP]
  prFalse := by
    classical
    simp
  prMonotone := by
    intro P Q hImp
    classical
    by_cases hP : P
    · have hQ : Q := hImp hP
      simp [hP, hQ]
    · simp [hP]
      by_cases hQ : Q
      · simp [hQ]
      · simp [hQ]
  prUnionLeAdd := by
    intro P Q
    classical
    by_cases hP : P <;> by_cases hQ : Q <;> simp [hP, hQ]

/--
Unpack MSIS hardness into an explicit negligible error plus canonical advantage
bound surface.
-/
theorem msisAdvantageBound_of_hardness
  {params : AjtaiParams}
  (h : MSISHardnessAssumption params) :
  ∃ eps : ErrorFn, IsNegligible eps ∧ MSISAdvantageBound params eps := by
  rcases h with ⟨eps, hNeg, hBound⟩
  exact ⟨eps, hNeg, hBound⟩

namespace MSISHardnessBoundary

/--
Canonical boundary package reconstructed from the theorem-level MSIS hardness
assumption.
-/
noncomputable def ofHardness
  {params : AjtaiParams}
  (h : MSISHardnessAssumption params) :
  MSISHardnessBoundary params :=
  let eps := Classical.choose h
  let hRest := Classical.choose_spec h
  { epsMSIS := eps
    advantageBound := hRest.2
    negligibleEpsMSIS := hRest.1 }

/--
The reconstructed boundary package immediately recovers the input theorem-level
MSIS hardness assumption.
-/
theorem ofHardness_hardnessFromFields
  {params : AjtaiParams}
  (h : MSISHardnessAssumption params) :
  MSISHardnessAssumption params := by
  exact (ofHardness h).hardnessFromFields

end MSISHardnessBoundary

/--
Package an Ajtai binding advantage bound together with negligible error.
-/
theorem no_ajtaiBindingCollision_of_advantageBound
  {params : AjtaiParams}
  {eps : ErrorFn}
  (hBound : AjtaiBindingAdvantageBound params eps)
  (hNeg : IsNegligible eps) :
  AjtaiBindingAssumption params := by
  exact ⟨eps, hNeg, hBound⟩

/--
Package an Ajtai relaxed-binding advantage bound together with negligible error.
-/
theorem no_ajtaiRelaxedBindingCollision_of_advantageBound
  {params : AjtaiParams}
  {C : SamplingCarrier}
  {eps : ErrorFn}
  (hBound : AjtaiRelaxedBindingAdvantageBound params C eps)
  (hNeg : IsNegligible eps) :
  AjtaiRelaxedBindingAssumption params C := by
  exact ⟨eps, hNeg, hBound⟩

namespace AjtaiBindingBoundary

/-- Canonical hardness view for an Ajtai binding boundary package. -/
def hardness
  {params : AjtaiParams}
  (h : AjtaiBindingBoundary params) : AjtaiBindingAssumption params :=
  no_ajtaiBindingCollision_of_advantageBound h.advantageBound h.negligibleEpsBinding

/-- Canonical hardness derivation from package fields. -/
theorem hardnessFromFields
  {params : AjtaiParams}
  (h : AjtaiBindingBoundary params) : AjtaiBindingAssumption params :=
  h.hardness

/-- Normalize any package by overwriting redundant `hardness` proof from aligned fields. -/
def normalize
  {params : AjtaiParams}
  (h : AjtaiBindingBoundary params) : AjtaiBindingBoundary params :=
  h

theorem normalize_hardnessFromFields
  {params : AjtaiParams}
  (h : AjtaiBindingBoundary params) :
  (normalize h).hardness = h.hardnessFromFields := by
  rfl

end AjtaiBindingBoundary

namespace AjtaiRelaxedBindingBoundary

/-- Canonical hardness view for an Ajtai relaxed-binding boundary package. -/
def hardness
  {params : AjtaiParams}
  {C : SamplingCarrier}
  (h : AjtaiRelaxedBindingBoundary params C) : AjtaiRelaxedBindingAssumption params C :=
  no_ajtaiRelaxedBindingCollision_of_advantageBound h.advantageBound h.negligibleEpsRelaxedBinding

/-- Canonical relaxed-hardness derivation from package fields. -/
theorem hardnessFromFields
  {params : AjtaiParams}
  {C : SamplingCarrier}
  (h : AjtaiRelaxedBindingBoundary params C) : AjtaiRelaxedBindingAssumption params C :=
  h.hardness

/-- Normalize any relaxed package by overwriting redundant `hardness` proof from aligned fields. -/
def normalize
  {params : AjtaiParams}
  {C : SamplingCarrier}
  (h : AjtaiRelaxedBindingBoundary params C) : AjtaiRelaxedBindingBoundary params C :=
  h

theorem normalize_hardnessFromFields
  {params : AjtaiParams}
  {C : SamplingCarrier}
  (h : AjtaiRelaxedBindingBoundary params C) :
  (normalize h).hardness = h.hardnessFromFields := by
  rfl

end AjtaiRelaxedBindingBoundary

/--
Abstract reduction interface from MSIS hardness to Ajtai commitment security.
This remains theorem-facing only; implication theorems are derived below.
-/
structure MSISToAjtaiReductions (params : AjtaiParams) where
  laws : LatticeReductionLaws params
  relaxedExpansionPos : 0 < params.relaxedExpansion
  epsBinding : ErrorFn
  epsRelaxedBinding : ErrorFn
  bindingAdvantageBound : AjtaiBindingAdvantageBound params epsBinding
  relaxedBindingAdvantageBound :
    AjtaiRelaxedBindingAdvantageBound params laws.samplingCarrier epsRelaxedBinding
  negligibleEpsBinding : IsNegligible epsBinding
  negligibleEpsRelaxedBinding : IsNegligible epsRelaxedBinding

namespace MSISToAjtaiReductions

/--
Canonical constructor from an already-built lattice-law package.
-/
def ofLaws
  {params : AjtaiParams}
  (laws : LatticeReductionLaws params)
  (hExpPos : 0 < params.relaxedExpansion)
  (epsBinding epsRelaxedBinding : ErrorFn)
  (hBindBound : AjtaiBindingAdvantageBound params epsBinding)
  (hRelaxedBound : AjtaiRelaxedBindingAdvantageBound params laws.samplingCarrier epsRelaxedBinding)
  (hBindNeg : IsNegligible epsBinding)
  (hRelaxedNeg : IsNegligible epsRelaxedBinding) :
  MSISToAjtaiReductions params where
  laws := laws
  relaxedExpansionPos := hExpPos
  epsBinding := epsBinding
  epsRelaxedBinding := epsRelaxedBinding
  bindingAdvantageBound := hBindBound
  relaxedBindingAdvantageBound := hRelaxedBound
  negligibleEpsBinding := hBindNeg
  negligibleEpsRelaxedBinding := hRelaxedNeg

/--
Canonical constructor specialized to `paperCarrier`, when strong-sampling is
already available for that carrier at `params.relaxedExpansion`.
-/
def ofPaperCarrier
  {params : AjtaiParams}
  (hStrong : strongSamplingExpansionProp paperCarrier params.relaxedExpansion)
  (hExpPos : 0 < params.relaxedExpansion)
  (epsBinding epsRelaxedBinding : ErrorFn)
  (hBindBound : AjtaiBindingAdvantageBound params epsBinding)
  (hRelaxedBound :
    AjtaiRelaxedBindingAdvantageBound params paperCarrier epsRelaxedBinding)
  (hBindNeg : IsNegligible epsBinding)
  (hRelaxedNeg : IsNegligible epsRelaxedBinding) :
  MSISToAjtaiReductions params :=
  ofLaws
    (laws := LatticeReductionLaws.ofPaperCarrier hStrong)
    hExpPos
    epsBinding epsRelaxedBinding
    hBindBound
    (by simpa using hRelaxedBound)
    hBindNeg hRelaxedNeg

/--
Paper-carrier constructor from subtraction/multiplication norm bundles.
This threads `strongSampling` via
`LatticeReductionLaws.ofPaperCarrierFromBounds`.
-/
def ofPaperCarrierFromBounds
  {params : AjtaiParams}
  {D : Nat}
  (hSub : coeffSubNormBoundFromOperands 2 2 D)
  (hMul : ∀ B : Nat, mulRqPhiNormBoundFromOperands D B (4 * params.relaxedExpansion * B))
  (hExpPos : 0 < params.relaxedExpansion)
  (epsBinding epsRelaxedBinding : ErrorFn)
  (hBindBound : AjtaiBindingAdvantageBound params epsBinding)
  (hRelaxedBound :
    AjtaiRelaxedBindingAdvantageBound params paperCarrier epsRelaxedBinding)
  (hBindNeg : IsNegligible epsBinding)
  (hRelaxedNeg : IsNegligible epsRelaxedBinding) :
  MSISToAjtaiReductions params :=
  ofLaws
    (laws := LatticeReductionLaws.ofPaperCarrierFromBounds
      (params := params) (D := D) hSub hMul)
    hExpPos
    epsBinding epsRelaxedBinding
    hBindBound
    (by simpa using hRelaxedBound)
    hBindNeg hRelaxedNeg

/--
Paper-carrier constructor from the concrete norm-bundle closure path.
This requires `3*d ≤ params.relaxedExpansion` to derive strong sampling.
-/
def ofPaperCarrierFromThreeDLe
  {params : AjtaiParams}
  (hTd : 3 * d ≤ params.relaxedExpansion)
  (hExpPos : 0 < params.relaxedExpansion)
  (epsBinding epsRelaxedBinding : ErrorFn)
  (hBindBound : AjtaiBindingAdvantageBound params epsBinding)
  (hRelaxedBound :
    AjtaiRelaxedBindingAdvantageBound params paperCarrier epsRelaxedBinding)
  (hBindNeg : IsNegligible epsBinding)
  (hRelaxedNeg : IsNegligible epsRelaxedBinding) :
  MSISToAjtaiReductions params :=
  ofLaws
    (laws := LatticeReductionLaws.ofPaperCarrierFromThreeDLe
      (params := params) hTd)
    hExpPos
    epsBinding epsRelaxedBinding
    hBindBound
    (by simpa using hRelaxedBound)
    hBindNeg hRelaxedNeg

/-- Derived Ajtai binding boundary from MSIS hardness, via explicit extractor. -/
theorem toBinding
  {params : AjtaiParams}
  (hRed : MSISToAjtaiReductions params)
  (_hMsis : MSISHardnessAssumption params) :
  AjtaiBindingAssumption params := by
  exact ⟨hRed.epsBinding, hRed.negligibleEpsBinding, hRed.bindingAdvantageBound⟩

/-- Derived Ajtai relaxed-binding boundary from MSIS hardness, via explicit extractor. -/
theorem toRelaxedBinding
  {params : AjtaiParams}
  (hRed : MSISToAjtaiReductions params)
  (_hMsis : MSISHardnessAssumption params) :
  AjtaiRelaxedBindingAssumption params hRed.laws.samplingCarrier := by
  exact
    ⟨hRed.epsRelaxedBinding, hRed.negligibleEpsRelaxedBinding, hRed.relaxedBindingAdvantageBound⟩

end MSISToAjtaiReductions

/-- Derive Ajtai binding from MSIS via the declared reduction surface. -/
theorem ajtaiBinding_of_msis
  {params : AjtaiParams}
  (hRed : MSISToAjtaiReductions params)
  (hMsis : MSISHardnessAssumption params) :
  AjtaiBindingAssumption params := by
  exact hRed.toBinding hMsis

/-- Derive Ajtai relaxed binding from MSIS via the declared reduction surface. -/
theorem ajtaiRelaxedBinding_of_msis
  {params : AjtaiParams}
  (hRed : MSISToAjtaiReductions params)
  (hMsis : MSISHardnessAssumption params) :
  AjtaiRelaxedBindingAssumption params hRed.laws.samplingCarrier := by
  exact hRed.toRelaxedBinding hMsis

/-- Package both Ajtai boundaries derived from MSIS under one reduction interface. -/
theorem ajtaiBoundaries_of_msis
  {params : AjtaiParams}
  (hRed : MSISToAjtaiReductions params)
  (hMsis : MSISHardnessAssumption params) :
  AjtaiBindingAssumption params ∧
    AjtaiRelaxedBindingAssumption params hRed.laws.samplingCarrier := by
  exact ⟨ajtaiBinding_of_msis hRed hMsis, ajtaiRelaxedBinding_of_msis hRed hMsis⟩

/-- Build the canonical Ajtai binding boundary package from MSIS hardness + reduction surface. -/
def ajtaiBindingBoundary_of_msis
  {params : AjtaiParams}
  (hRed : MSISToAjtaiReductions params)
  (_hMsis : MSISHardnessAssumption params) :
  AjtaiBindingBoundary params where
  epsBinding := hRed.epsBinding
  advantageBound := hRed.bindingAdvantageBound
  negligibleEpsBinding := hRed.negligibleEpsBinding

/-- Build the canonical Ajtai relaxed-binding boundary package from MSIS hardness + reduction surface. -/
def ajtaiRelaxedBindingBoundary_of_msis
  {params : AjtaiParams}
  (hRed : MSISToAjtaiReductions params)
  (_hMsis : MSISHardnessAssumption params) :
  AjtaiRelaxedBindingBoundary params hRed.laws.samplingCarrier where
  epsRelaxedBinding := hRed.epsRelaxedBinding
  advantageBound := hRed.relaxedBindingAdvantageBound
  negligibleEpsRelaxedBinding := hRed.negligibleEpsRelaxedBinding

end SuperNeo.ProofSystem
