import SuperNeo.ProofSystem.SumCheck.General

namespace SuperNeo.ProofSystem

namespace Sumcheck

open scoped BigOperators

/--
Pad a verifier-coin prefix to a full challenge vector by appending zeroes.

This lets us turn prefix-indexed objects into total `Array F` inputs for the
existing online prover surface while preserving the prefix that matters.
-/
def prefixPadCoins (inst : Instance) (pre : Array F) : Array F :=
  Array.ofFn (fun i : Fin inst.rounds =>
    if hi : i.1 < pre.size then pre[i.1]! else 0)

@[simp] theorem prefixPadCoins_size
  (inst : Instance)
  (pre : Array F) :
  (prefixPadCoins inst pre).size = inst.rounds := by
  simp [prefixPadCoins]

theorem prefixPadCoins_get_prefix
  (inst : Instance)
  (pre : Array F)
  {i : Nat}
  (hi : i < pre.size)
  (hRounds : i < inst.rounds) :
  (prefixPadCoins inst pre)[i]! = pre[i]! := by
  rw [getElem!_pos (c := prefixPadCoins inst pre) (i := i)]
  · simp [prefixPadCoins, hi]
  · simpa [prefixPadCoins] using hRounds

theorem prefixPadCoins_getElem
  (inst : Instance)
  (pre : Array F)
  {i : Nat}
  (hi : i < pre.size)
  (hRounds : i < inst.rounds) :
  (prefixPadCoins inst pre)[i]'(by simpa [prefixPadCoins_size] using hRounds) = pre[i] := by
  have hPad : i < (prefixPadCoins inst pre).size := by
    simpa [prefixPadCoins_size] using hRounds
  show (prefixPadCoins inst pre)[i]'hPad = pre[i]
  simp [prefixPadCoins, hi]

theorem prefixPadCoins_extract
  (inst : Instance)
  (pre : Array F)
  (hSize : pre.size ≤ inst.rounds) :
  (prefixPadCoins inst pre).extract 0 pre.size = pre := by
  apply Array.ext
  · simp [prefixPadCoins, hSize]
  · intro i hiL hiR
    rw [Array.getElem_extract]
    simpa [Nat.zero_add] using
      prefixPadCoins_getElem inst pre hiR (Nat.lt_of_lt_of_le hiR hSize)

/-- One multilinear folding layer, matching `MLE.mleByFoldingExec`. -/
def foldLayerLocal (vals : Array F) (ri : F) : Array F :=
  Array.ofFn (fun i : Fin (vals.size / 2) =>
    vals[2 * i.1]! * ((1 : F) - ri) + vals[2 * i.1 + 1]! * ri)

@[simp] theorem foldLayerLocal_size
  (vals : Array F)
  (ri : F) :
  (foldLayerLocal vals ri).size = vals.size / 2 := by
  simp [foldLayerLocal]

private theorem foldLayerLocal_eq_foldLayer
  (vals : Array F)
  (ri : F) :
  foldLayerLocal vals ri = SuperNeo.foldLayer vals ri := by
  rfl

/--
Residual table after fixing the first `i` verifier challenges.

This is the honest multilinear table obtained by repeatedly folding the original
truth table with the sampled prefix `coins[0..i)`.
-/
def SoundnessGame.residualTable
  (g : SoundnessGame) :
  Nat → Array F → Array F
  | 0, _ => g.table
  | i + 1, coins => foldLayerLocal (g.residualTable i coins) (coins[i]!)

theorem SoundnessGame.residualTable_invariant_of_prefix
  (g : SoundnessGame)
  (i : Nat)
  {coins1 coins2 : Array F}
  (hPrefix : ∀ j : Nat, j < i → coins1[j]! = coins2[j]!) :
  g.residualTable i coins1 = g.residualTable i coins2 := by
  induction i with
  | zero =>
      rfl
  | succ i ih =>
      have hTail :
          ∀ j : Nat, j < i → coins1[j]! = coins2[j]! := by
        intro j hj
        exact hPrefix j (Nat.lt_trans hj (Nat.lt_succ_self i))
      have hIH := ih hTail
      have hHead : coins1[i]! = coins2[i]! := hPrefix i (Nat.lt_succ_self i)
      simp [SoundnessGame.residualTable, hIH, hHead]

theorem SoundnessGame.residualTable_eq_of_extract
  (g : SoundnessGame)
  (i : Nat)
  {coins : Array F}
  (hSize : i ≤ g.inst.rounds)
  (hCoins : coins.size = g.inst.rounds) :
  g.residualTable i coins =
    g.residualTable i (prefixPadCoins g.inst (coins.extract 0 i)) := by
  apply g.residualTable_invariant_of_prefix
  intro j hj
  have hjCoins : j < coins.size := by
    simpa [hCoins] using (Nat.lt_of_lt_of_le hj hSize)
  have hjExtract : j < (coins.extract 0 i).size := by
    simpa [Array.size_extract, hSize, hCoins] using hj
  rw [prefixPadCoins_get_prefix g.inst (coins.extract 0 i)]
  · rw [getElem!_pos (c := coins.extract 0 i) (i := j) hjExtract]
    rw [getElem!_pos (c := coins) (i := j) hjCoins]
    simpa [Nat.zero_add] using
      (Array.getElem_extract (xs := coins) (start := 0) (stop := i) (i := j) hjExtract)
  · simpa [Array.size_extract, hSize, hCoins] using hj
  · simpa [prefixPadCoins_size, hCoins] using hjCoins

theorem SoundnessGame.residualTable_size
  (g : SoundnessGame)
  (i : Nat)
  (coins : Array F)
  (hi : i ≤ g.inst.rounds) :
  (g.residualTable i coins).size = 2 ^ (g.inst.rounds - i) := by
  induction i generalizing coins with
  | zero =>
      simpa [SoundnessGame.residualTable] using g.tableSize
  | succ i ih =>
      have hiLt : i < g.inst.rounds := by omega
      have hPrev := ih coins (Nat.le_of_lt hiLt)
      calc
        (g.residualTable (i + 1) coins).size
            = (g.residualTable i coins).size / 2 := by
                simp [SoundnessGame.residualTable]
        _ = (2 ^ (g.inst.rounds - i)) / 2 := by
              rw [hPrev]
        _ = 2 ^ (g.inst.rounds - (i + 1)) := by
              have hExp : g.inst.rounds - i = (g.inst.rounds - (i + 1)) + 1 := by
                omega
              rw [hExp, Nat.pow_succ]
              simp

private theorem extract_head_eq
  (coins : Array F)
  {i stop : Nat}
  (hi : i < stop)
  (hStop : stop ≤ coins.size) :
  (coins.extract i stop)[0]! = coins[i]! := by
  have hPos : 0 < (coins.extract i stop).size := by
    rw [Array.size_extract_of_le hStop]
    omega
  rw [getElem!_pos (c := coins.extract i stop) (i := 0) hPos]
  rw [getElem!_pos (c := coins) (i := i)]
  · simpa using (Array.getElem_extract (xs := coins) (start := i) (stop := stop) (i := 0) (by
      rw [Array.size_extract_of_le hStop]
      omega))
  · exact Nat.lt_of_lt_of_le hi hStop

private theorem extract_tail_eq
  (coins : Array F)
  {i stop : Nat}
  (hi : i < stop)
  (hStop : stop ≤ coins.size) :
  (coins.extract i stop).extract 1 (coins.extract i stop).size =
    coins.extract (i + 1) stop := by
  rw [Array.extract_extract]
  rw [Array.size_extract_of_le hStop]
  simp [Nat.min_eq_right, hi.le, Nat.add_assoc]

theorem SoundnessGame.residualTable_mleByFolding_tail
  (g : SoundnessGame)
  (i : Nat)
  (coins : Array F)
  (hi : i ≤ g.inst.rounds)
  (hCoins : coins.size = g.inst.rounds) :
  mleByFolding (g.residualTable i coins) (coins.extract i coins.size) =
    mleByFolding g.table coins := by
  induction i generalizing coins with
  | zero =>
      have hExtract : coins.extract 0 coins.size = coins := by
        simpa using (Array.extract_eq_self_of_le (as := coins) (i := 0) (j := coins.size) (by simp))
      simpa [SoundnessGame.residualTable, hExtract]
  | succ i ih =>
      have hiLt : i < g.inst.rounds := by omega
      have hStop : coins.size ≤ coins.size := le_rfl
      have hTailNe : (coins.extract i coins.size).size ≠ 0 := by
        rw [Array.size_extract_of_le hStop]
        omega
      have hStep :=
        SuperNeo.mleByFolding_step
          (v := g.residualTable i coins)
          (r := coins.extract i coins.size)
          hTailNe
      have hHead :
          (coins.extract i coins.size)[0]! = coins[i]! :=
        extract_head_eq coins (hi := by omega) hStop
      have hTail :
          (coins.extract i coins.size).extract 1 (coins.extract i coins.size).size =
            coins.extract (i + 1) coins.size :=
        extract_tail_eq coins (hi := by omega) hStop
      have hStep' :
          mleByFolding (g.residualTable i coins) (coins.extract i coins.size) =
            mleByFolding
              (SuperNeo.foldLayer (g.residualTable i coins) (coins[i]!))
              (coins.extract (i + 1) coins.size) := by
        rw [hStep, hHead, hTail]
      calc
        mleByFolding (g.residualTable (i + 1) coins) (coins.extract (i + 1) coins.size)
            = mleByFolding
                (SuperNeo.foldLayer (g.residualTable i coins) (coins[i]!))
                (coins.extract (i + 1) coins.size) := by
                  simp [SoundnessGame.residualTable, foldLayerLocal_eq_foldLayer]
        _ = mleByFolding (g.residualTable i coins) (coins.extract i coins.size) := by
              simpa using hStep'.symm
        _ = mleByFolding g.table coins := by
              exact ih coins (Nat.le_of_lt hiLt) hCoins

/-- Sum of all values in an array. -/
def arraySum (vals : Array F) : F :=
  Finset.sum (Finset.range vals.size) (fun i => vals[i]!)

@[simp] theorem arraySum_empty :
  arraySum (#[] : Array F) = 0 := by
  simp [arraySum]

@[simp] theorem arraySum_ofFn
  (n : Nat)
  (f : Fin n → F) :
  arraySum (Array.ofFn f) =
    Finset.sum (Finset.range n) (fun i => if h : i < n then f ⟨i, h⟩ else 0) := by
  rw [arraySum]
  simp only [Array.size_ofFn]
  apply Finset.sum_congr rfl
  intro i hi
  have hiNat : i < n := Finset.mem_range.mp hi
  have hi' : i < (Array.ofFn f).size := by simpa using hi
  rw [getElem!_pos (c := Array.ofFn f) (i := i) hi']
  simpa [hiNat] using (Array.getElem_ofFn (f := f) (i := i) hi')

private theorem sum_range_pairs
  (n : Nat)
  (f : Nat → F) :
  Finset.sum (Finset.range n) (fun j => f (2 * j) + f (2 * j + 1)) =
    Finset.sum (Finset.range (2 * n)) f := by
  induction n with
  | zero =>
      simp
  | succ n ih =>
      calc
        Finset.sum (Finset.range (n + 1)) (fun j => f (2 * j) + f (2 * j + 1))
            = Finset.sum (Finset.range n) (fun j => f (2 * j) + f (2 * j + 1)) +
                (f (2 * n) + f (2 * n + 1)) := by
                  rw [Finset.sum_range_succ]
        _ = Finset.sum (Finset.range (2 * n)) f + (f (2 * n) + f (2 * n + 1)) := by
              rw [ih]
        _ = Finset.sum (Finset.range (2 * n)) f + f (2 * n) + f (2 * n + 1) := by
              abel
        _ = Finset.sum (Finset.range (2 * n + 1)) f + f (2 * n + 1) := by
              rw [Finset.sum_range_succ]
        _ = Finset.sum (Finset.range (2 * (n + 1))) f := by
              have hEq : 2 * (n + 1) = 2 * n + 1 + 1 := by omega
              simpa [hEq] using (Finset.sum_range_succ (f := f) (n := 2 * n + 1)).symm

private theorem arraySum_even_add_odd
  (vals : Array F)
  {n : Nat}
  (hSize : vals.size = 2 * n) :
  arraySum vals =
    arraySum (Array.ofFn (fun j : Fin n => vals[2 * j.1]!)) +
      arraySum (Array.ofFn (fun j : Fin n => vals[2 * j.1 + 1]!)) := by
  have hEven :
      arraySum (Array.ofFn (fun j : Fin n => vals[2 * j.1]!)) =
        Finset.sum (Finset.range n) (fun j => vals[2 * j]!) := by
    rw [arraySum_ofFn]
    apply Finset.sum_congr rfl
    intro x hx
    have hx' : x < n := Finset.mem_range.mp hx
    simp [hx']
  have hOdd :
      arraySum (Array.ofFn (fun j : Fin n => vals[2 * j.1 + 1]!)) =
        Finset.sum (Finset.range n) (fun j => vals[2 * j + 1]!) := by
    rw [arraySum_ofFn]
    apply Finset.sum_congr rfl
    intro x hx
    have hx' : x < n := Finset.mem_range.mp hx
    simp [hx']
  calc
    arraySum vals
        = Finset.sum (Finset.range (2 * n)) (fun k => vals[k]!) := by
              simp [arraySum, hSize]
    _ = Finset.sum (Finset.range n) (fun j => vals[2 * j]! + vals[2 * j + 1]!) := by
          symm
          exact sum_range_pairs n (fun k => vals[k]!)
    _ = Finset.sum (Finset.range n) (fun j => vals[2 * j]!) +
          Finset.sum (Finset.range n) (fun j => vals[2 * j + 1]!) := by
          rw [Finset.sum_add_distrib]
    _ = arraySum (Array.ofFn (fun j : Fin n => vals[2 * j.1]!)) +
          arraySum (Array.ofFn (fun j : Fin n => vals[2 * j.1 + 1]!)) := by
          rw [hEven, hOdd]

private theorem arraySum_foldLayerLocal
  (vals : Array F)
  (ri : F) :
  arraySum (foldLayerLocal vals ri) =
    arraySum (Array.ofFn (fun j : Fin (vals.size / 2) => vals[2 * j.1]!)) * ((1 : F) - ri) +
      arraySum (Array.ofFn (fun j : Fin (vals.size / 2) => vals[2 * j.1 + 1]!)) * ri := by
  have hEven :
      arraySum (Array.ofFn (fun j : Fin (vals.size / 2) => vals[2 * j.1]!)) =
        Finset.sum (Finset.range (vals.size / 2)) (fun j => vals[2 * j]!) := by
    rw [arraySum_ofFn]
    apply Finset.sum_congr rfl
    intro x hx
    have hx' : x < vals.size / 2 := Finset.mem_range.mp hx
    simp [hx']
  have hOdd :
      arraySum (Array.ofFn (fun j : Fin (vals.size / 2) => vals[2 * j.1 + 1]!)) =
        Finset.sum (Finset.range (vals.size / 2)) (fun j => vals[2 * j + 1]!) := by
    rw [arraySum_ofFn]
    apply Finset.sum_congr rfl
    intro x hx
    have hx' : x < vals.size / 2 := Finset.mem_range.mp hx
    simp [hx']
  calc
    arraySum (foldLayerLocal vals ri)
        = Finset.sum (Finset.range (vals.size / 2))
            (fun j => vals[2 * j]! * ((1 : F) - ri) + vals[2 * j + 1]! * ri) := by
              rw [foldLayerLocal, arraySum_ofFn]
              apply Finset.sum_congr rfl
              intro x hx
              have hx' : x < vals.size / 2 := Finset.mem_range.mp hx
              simp [hx']
    _ = Finset.sum (Finset.range (vals.size / 2)) (fun j => vals[2 * j]! * ((1 : F) - ri)) +
          Finset.sum (Finset.range (vals.size / 2)) (fun j => vals[2 * j + 1]! * ri) := by
            rw [Finset.sum_add_distrib]
    _ = Finset.sum (Finset.range (vals.size / 2)) (fun j => vals[2 * j]!) * ((1 : F) - ri) +
          Finset.sum (Finset.range (vals.size / 2)) (fun j => vals[2 * j + 1]!) * ri := by
            rw [← Finset.sum_mul, ← Finset.sum_mul]
    _ = arraySum (Array.ofFn (fun j : Fin (vals.size / 2) => vals[2 * j.1]!)) * ((1 : F) - ri) +
          arraySum (Array.ofFn (fun j : Fin (vals.size / 2) => vals[2 * j.1 + 1]!)) * ri := by
            rw [hEven, hOdd]

theorem sumcheckTableSum_eq_arraySum
  (vals : Array F) :
  SuperNeo.sumcheckTableSum vals = arraySum vals := by
  unfold SuperNeo.sumcheckTableSum arraySum
  rw [← Array.foldr_toList]
  rw [← List.sum_eq_foldr]
  rw [show vals.toList = List.ofFn (fun i : Fin vals.size => vals[i.1]!) by
    simpa using (List.ofFn_getElem vals.toList).symm]
  rw [List.sum_ofFn, Fin.sum_univ_eq_sum_range]

private theorem fToZMod_zero :
  fToZMod (0 : F) = 0 := by
  simp [fToZMod]

private theorem fToZMod_one :
  fToZMod (1 : F) = 1 := by
  simp [fToZMod]

private theorem fToZMod_add
  (a b : F) :
  fToZMod (a + b) = fToZMod a + fToZMod b := by
  simp [fToZMod, SuperNeo.F.val_add]

private theorem fToZMod_mul
  (a b : F) :
  fToZMod (a * b) = fToZMod a * fToZMod b := by
  simp [fToZMod, SuperNeo.F.val_mul]

private theorem fToZMod_neg
  (a : F) :
  fToZMod (-a) = -fToZMod a := by
  simp [fToZMod, SuperNeo.F.val_neg]

private theorem fToZMod_sub
  (a b : F) :
  fToZMod (a - b) = fToZMod a - fToZMod b := by
  simp [sub_eq_add_neg, fToZMod_add, fToZMod_neg]

private theorem fToZMod_pow
  (a : F)
  (n : Nat) :
  fToZMod (a ^ n) = (fToZMod a) ^ n := by
  induction n with
  | zero =>
      simp [pow_zero, fToZMod_one]
  | succ n ih =>
      simp [pow_succ, fToZMod_mul, ih]

private theorem sumcheckEvalPoly_ofFn_succ
  (n : Nat)
  (v : Fin (n + 1) → F)
  (x : F) :
  SuperNeo.sumcheckEvalPoly (Array.ofFn v) x =
    v 0 + x * SuperNeo.sumcheckEvalPoly (Array.ofFn (fun i : Fin n => v i.succ)) x := by
  unfold SuperNeo.sumcheckEvalPoly
  rw [← Array.foldr_toList]
  simp
  have hTail :
      List.foldr (fun c acc => c + x * acc) 0
          (Array.ofFn (fun i : Fin n => v i.succ)).toList =
        Array.foldr (fun c acc => c + x * acc) 0
          (Array.ofFn (fun i : Fin n => v i.succ)) := by
    simpa using
      (Array.foldr_toList
        (f := fun c acc => c + x * acc)
        (init := (0 : F))
        (xs := Array.ofFn (fun i : Fin n => v i.succ)))
  simpa [SuperNeo.sumcheckEvalPoly] using congrArg (fun z => v 0 + x * z) hTail

private theorem sumcheckEvalPoly_zero_ofFn
  (n : Nat)
  (x : F) :
  SuperNeo.sumcheckEvalPoly (Array.ofFn (fun _ : Fin n => (0 : F))) x = 0 := by
  induction n with
  | zero =>
      simp [SuperNeo.sumcheckEvalPoly]
  | succ n ih =>
      rw [sumcheckEvalPoly_ofFn_succ]
      simp [ih]

private theorem sumcheckEvalPoly_headOnly_ofFn
  (n : Nat)
  (c x : F) :
  SuperNeo.sumcheckEvalPoly
      (Array.ofFn (fun k : Fin (n + 1) => if k.1 = 0 then c else 0)) x = c := by
  rw [sumcheckEvalPoly_ofFn_succ]
  simp [sumcheckEvalPoly_zero_ofFn]

private theorem polynomial_ofFn_succ
  (n : Nat)
  (v : Fin (n + 1) → Fq) :
  Polynomial.ofFn (n + 1) v =
    Polynomial.C (v 0) + Polynomial.X * Polynomial.ofFn n (fun i : Fin n => v i.succ) := by
  rw [Polynomial.ofFn_eq_sum_monomial]
  rw [Fin.sum_univ_succ]
  rw [Polynomial.ofFn_eq_sum_monomial]
  simp [Finset.mul_sum, Polynomial.X_mul_monomial]

private theorem sumcheckEvalPoly_fToZMod_ofFn
  (n : Nat)
  (v : Fin n → F)
  (x : F) :
  fToZMod (SuperNeo.sumcheckEvalPoly (Array.ofFn v) x) =
    (Polynomial.ofFn n (fun i : Fin n => fToZMod (v i))).eval (fToZMod x) := by
  induction n with
  | zero =>
      simp [SuperNeo.sumcheckEvalPoly, Polynomial.ofFn_zero', fToZMod_zero]
  | succ n ih =>
      rw [sumcheckEvalPoly_ofFn_succ]
      rw [polynomial_ofFn_succ]
      simp [Polynomial.eval_add, Polynomial.eval_mul, Polynomial.eval_X,
        Polynomial.eval_C, fToZMod_add, fToZMod_mul, ih]

theorem sumcheckEvalPoly_fToZMod
  (poly : Array F)
  (x : F) :
  fToZMod (SuperNeo.sumcheckEvalPoly poly x) =
    (sumcheckPolynomialZMod poly).eval (fToZMod x) := by
  let v : Fin poly.size → F := fun i => poly[i.1]!
  have hPoly : Array.ofFn v = poly := by
    apply Array.ext
    · simp [v]
    · intro i h1 h2
      simp [v]
  calc
    fToZMod (SuperNeo.sumcheckEvalPoly poly x)
        = fToZMod (SuperNeo.sumcheckEvalPoly (Array.ofFn v) x) := by
            rw [hPoly]
    _ = (Polynomial.ofFn poly.size (fun i : Fin poly.size => fToZMod (v i))).eval (fToZMod x) :=
          sumcheckEvalPoly_fToZMod_ofFn poly.size v x
    _ = (sumcheckPolynomialZMod poly).eval (fToZMod x) := by
          simp [sumcheckPolynomialZMod, v]

private theorem sumcheckPolynomialZMod_sub_eq
  {n : Nat}
  (a b : Array F)
  (hA : a.size = n)
  (hB : b.size = n) :
  sumcheckPolynomialZMod
      (Array.ofFn (fun k : Fin n => a[k.1]! - b[k.1]!)) =
    sumcheckPolynomialZMod a - sumcheckPolynomialZMod b := by
  ext i
  by_cases hi : i < n
  · have hAi : i < a.size := by simpa [hA] using hi
    have hBi : i < b.size := by simpa [hB] using hi
    simp [sumcheckPolynomialZMod, hi, hAi, hBi, fToZMod_sub]
  · have hNi : n ≤ i := Nat.le_of_not_lt hi
    have hAi : a.size ≤ i := by simpa [hA] using hNi
    have hBi : b.size ≤ i := by simpa [hB] using hNi
    have hSizeSub :
        (Array.ofFn (fun k : Fin n => a[k.1]! - b[k.1]!)).size = n := by
      simp
    have hL0 :
        ((Polynomial.ofFn n
            (fun k : Fin n => fToZMod (a[k.1]! - b[k.1]!))).coeff i) = 0 :=
      Polynomial.ofFn_coeff_eq_zero_of_ge
        (v := fun k : Fin n => fToZMod (a[k.1]! - b[k.1]!))
        hNi
    have hPolySub :
        sumcheckPolynomialZMod
            (Array.ofFn (fun k : Fin n => a[k.1]! - b[k.1]!)) =
          Polynomial.ofFn n
            (fun k : Fin n => fToZMod (a[k.1]! - b[k.1]!)) := by
      ext j
      by_cases hj : j < n
      · simp [sumcheckPolynomialZMod, hSizeSub, hj, Array.getElem_ofFn]
      · have hnj : n ≤ j := Nat.le_of_not_lt hj
        simp [sumcheckPolynomialZMod, hSizeSub, hj, hnj, Array.getElem_ofFn]
    have hL :
        (sumcheckPolynomialZMod
          (Array.ofFn (fun k : Fin n => a[k.1]! - b[k.1]!))).coeff i = 0 := by
      rw [hPolySub]
      exact hL0
    have hRA : (sumcheckPolynomialZMod a).coeff i = 0 := by
      simpa [sumcheckPolynomialZMod] using
        (Polynomial.ofFn_coeff_eq_zero_of_ge
          (v := fun k : Fin a.size => fToZMod a[k.1]!)
          hAi)
    have hRB : (sumcheckPolynomialZMod b).coeff i = 0 := by
      simpa [sumcheckPolynomialZMod] using
        (Polynomial.ofFn_coeff_eq_zero_of_ge
          (v := fun k : Fin b.size => fToZMod b[k.1]!)
          hBi)
    simp [hL, hRA, hRB]

private theorem sumcheckEvalPoly_sub_eq
  {n : Nat}
  (a b : Array F)
  (x : F)
  (hA : a.size = n)
  (hB : b.size = n) :
  SuperNeo.sumcheckEvalPoly
      (Array.ofFn (fun k : Fin n => a[k.1]! - b[k.1]!)) x =
    SuperNeo.sumcheckEvalPoly a x - SuperNeo.sumcheckEvalPoly b x := by
  apply fToZMod_injective
  rw [sumcheckEvalPoly_fToZMod, fToZMod_sub, sumcheckEvalPoly_fToZMod, sumcheckEvalPoly_fToZMod]
  rw [sumcheckPolynomialZMod_sub_eq a b hA hB]
  simp

private theorem linear_interp_eval
  (a b x : F) :
  a + x * (b - a) = a * ((1 : F) - x) + b * x := by
  apply fToZMod_injective
  simp [fToZMod_add, fToZMod_mul, fToZMod_sub, fToZMod_one]
  ring

private theorem list_sum_map_ite_eq_mul_filter_length_local
  {α : Type}
  (l : List α)
  (P : α → Bool)
  (c : Nat) :
  (l.map (fun x => if P x then c else 0)).sum = c * (l.filter P).length := by
  induction l with
  | nil =>
      simp
  | cons x xs ih =>
      by_cases hPx : P x
      · simp [hPx, ih, Nat.mul_add, Nat.add_assoc, Nat.add_left_comm, Nat.add_comm]
      · simp [hPx, ih, Nat.mul_add, Nat.add_assoc, Nat.add_left_comm, Nat.add_comm]

theorem fullFieldCoinEventCountBool_pushInvariant
  (m : Nat)
  (B : Array F → Bool)
  (hInv : ∀ a ∈ fullFieldCoinSpace m, ∀ r : F, B (a.push r) = B a) :
  fullFieldCoinEventCountBool (m + 1) B =
    Goldilocks.q * fullFieldCoinEventCountBool m B := by
  classical
  unfold fullFieldCoinEventCountBool
  rw [fullFieldCoinSpace, List.filter_flatMap, List.length_flatMap]
  have hInner :
      List.map
        (fun a =>
          (List.filter B
            (fullFieldChallengeDomain.map (fun r => a.push r))).length)
        (fullFieldCoinSpace m)
      =
      List.map
        (fun a => if B a then Goldilocks.q else 0)
        (fullFieldCoinSpace m) := by
    apply List.ext_get
    · simp
    · intro j hj1 hj2
      simp at hj1
      have hjMem : (fullFieldCoinSpace m).get ⟨j, hj1⟩ ∈ fullFieldCoinSpace m := by
        exact List.get_mem _ _
      let a : Array F := (fullFieldCoinSpace m).get ⟨j, hj1⟩
      have hPredEq : (fun r : F => B (a.push r)) = fun _ : F => B a := by
        funext r
        exact hInv a hjMem r
      have hFilterEq :
          (List.filter B
            (fullFieldChallengeDomain.map (fun r => a.push r))).length
            = (if B a then Goldilocks.q else 0) := by
        calc
          (List.filter B
            (fullFieldChallengeDomain.map (fun r => a.push r))).length
              = (List.filter (B ∘ fun r => a.push r) fullFieldChallengeDomain).length := by
                  simp [List.filter_map]
          _ = (List.filter (fun _ : F => B a) fullFieldChallengeDomain).length := by
                simpa using congrArg (fun p => (List.filter p fullFieldChallengeDomain).length) hPredEq
          _ = (if B a then Goldilocks.q else 0) := by
                by_cases hBa : B a
                · simp [hBa, fullFieldChallengeDomain_length]
                · simp [hBa]
      dsimp [a] at hFilterEq
      simpa using hFilterEq
  rw [hInner]
  have hSum :=
    list_sum_map_ite_eq_mul_filter_length_local
      (l := fullFieldCoinSpace m)
      (P := B)
      (c := Goldilocks.q)
  simpa [Nat.mul_comm] using hSum

theorem fullFieldCoinEventCount_pushInvariant
  (m : Nat)
  (E : Array F → Prop)
  (hInv : ∀ a ∈ fullFieldCoinSpace m, ∀ r : F, E (a.push r) ↔ E a) :
  fullFieldCoinEventCount (m + 1) E =
    Goldilocks.q * fullFieldCoinEventCount m E := by
  classical
  let B : Array F → Bool := fun coins => decide (E coins)
  have hInvBool : ∀ a ∈ fullFieldCoinSpace m, ∀ r : F, B (a.push r) = B a := by
    intro a ha r
    by_cases hEa : E a
    · have hPush : E (a.push r) := (hInv a ha r).2 hEa
      simp [B, hEa, hPush]
    · have hPush : ¬ E (a.push r) := by
        intro hPush'
        exact hEa ((hInv a ha r).1 hPush')
      simp [B, hEa, hPush]
  unfold fullFieldCoinEventCount
  simpa [B] using fullFieldCoinEventCountBool_pushInvariant m B hInvBool

theorem prefixRootMembership_push_iff
  (m i : Nat)
  (hi : i < m)
  (rootSet : Array F → Finset F)
  {a : Array F}
  (ha : a ∈ fullFieldCoinSpace m)
  (r : F) :
  ((a.push r)[i]! ∈ rootSet ((a.push r).extract 0 i)) ↔
    (a[i]! ∈ rootSet (a.extract 0 i)) := by
  have hSize : a.size = m := mem_fullFieldCoinSpace_size ha
  have hiA : i < a.size := by simpa [hSize] using hi
  have hiPush : i < (a.push r).size := by
    simpa [Array.size_push, hSize] using Nat.lt_trans hi (Nat.lt_succ_self m)
  have hExtract :
      (a.push r).extract 0 i = a.extract 0 i := by
    exact Array.extract_push_of_le (h := Nat.le_of_lt hiA)
  have hGet :
      (a.push r)[i]! = a[i]! := by
    rw [getElem!_pos (c := a.push r) (i := i) hiPush]
    rw [getElem!_pos (c := a) (i := i) hiA]
    simpa using Array.getElem_push_lt (xs := a) (x := r) (i := i) hiA
  simp [hExtract, hGet]

theorem fullFieldCoinEventCount_prefixRootSet_le
  (m i d : Nat)
  (hi : i < m)
  (rootSet : Array F → Finset F)
  (hBound :
    ∀ a ∈ fullFieldCoinSpace i, (rootSet a).card ≤ d) :
  fullFieldCoinEventCount m
      (fun coins => coins[i]! ∈ rootSet (coins.extract 0 i)) ≤
    d * Goldilocks.q ^ (m - 1) := by
  have hDecomp : ∃ t, m = i + 1 + t := by
    refine ⟨m - (i + 1), ?_⟩
    omega
  rcases hDecomp with ⟨t, rfl⟩
  induction t with
  | zero =>
      have hBase :
          fullFieldCoinEventCount (i + 1)
              (fun coins => coins[i]! ∈ rootSet (coins.extract 0 i)) ≤
            (fullFieldCoinSpace i).length * d := by
        simpa using
          (fullFieldCoinEventCount_lastPrefixRootSet_le
            (m := i) (d := d) (rootSet := rootSet) hBound)
      calc
        fullFieldCoinEventCount (i + 1)
            (fun coins => coins[i]! ∈ rootSet (coins.extract 0 i))
            ≤ (fullFieldCoinSpace i).length * d := hBase
        _ = d * Goldilocks.q ^ ((i + 1) - 1) := by
              simp [fullFieldCoinSpace_length, Nat.mul_comm, Nat.mul_left_comm, Nat.mul_assoc]
  | succ t ih =>
      have hPush :
          fullFieldCoinEventCount (i + 1 + t + 1)
              (fun coins => coins[i]! ∈ rootSet (coins.extract 0 i))
            =
          Goldilocks.q *
            fullFieldCoinEventCount (i + 1 + t)
              (fun coins => coins[i]! ∈ rootSet (coins.extract 0 i)) := by
        refine fullFieldCoinEventCount_pushInvariant (m := i + 1 + t)
          (E := fun coins => coins[i]! ∈ rootSet (coins.extract 0 i)) ?_
        intro a ha r
        exact prefixRootMembership_push_iff (m := i + 1 + t) (i := i)
          (hi := by omega) (rootSet := rootSet) ha r
      calc
        fullFieldCoinEventCount (i + 1 + t + 1)
            (fun coins => coins[i]! ∈ rootSet (coins.extract 0 i))
            = Goldilocks.q *
                fullFieldCoinEventCount (i + 1 + t)
                  (fun coins => coins[i]! ∈ rootSet (coins.extract 0 i)) := hPush
        _ ≤ Goldilocks.q * (d * Goldilocks.q ^ ((i + 1 + t) - 1)) := by
              exact Nat.mul_le_mul_left Goldilocks.q (ih (by omega))
        _ = d * Goldilocks.q ^ ((i + 1 + t + 1) - 1) := by
              have hExp : ((i + 1 + t) - 1) + 1 = ((i + 1 + t + 1) - 1) := by omega
              calc
                Goldilocks.q * (d * Goldilocks.q ^ ((i + 1 + t) - 1))
                    = d * (Goldilocks.q ^ ((i + 1 + t) - 1) * Goldilocks.q) := by
                        simp [Nat.mul_assoc, Nat.mul_left_comm, Nat.mul_comm]
                _ = d * Goldilocks.q ^ (((i + 1 + t) - 1) + 1) := by
                        simp [Nat.pow_succ, Nat.mul_assoc, Nat.mul_left_comm, Nat.mul_comm]
                _ = d * Goldilocks.q ^ ((i + 1 + t + 1) - 1) := by
                        rw [hExp]

/-- Honest SumCheck round polynomial at round `i`, derived from the residual table. -/
def SoundnessGame.honestRoundPoly
  (g : SoundnessGame)
  (i : Nat)
  (coins : Array F) : Array F :=
  let vals := g.residualTable i coins
  let v0 := arraySum (Array.ofFn (fun j : Fin (vals.size / 2) => vals[2 * j.1]!))
  let v1 := arraySum (Array.ofFn (fun j : Fin (vals.size / 2) => vals[2 * j.1 + 1]!))
  Array.ofFn (fun k : Fin (g.inst.maxDegree + 1) =>
    if h0 : k.1 = 0 then
      v0
    else if h1 : k.1 = 1 then
      v1 - v0
    else
      0)

theorem SoundnessGame.honestRoundPoly_invariant_of_prefix
  (g : SoundnessGame)
  (i : Nat)
  {coins1 coins2 : Array F}
  (hPrefix : ∀ j : Nat, j < i → coins1[j]! = coins2[j]!) :
  g.honestRoundPoly i coins1 = g.honestRoundPoly i coins2 := by
  have hRes :
      g.residualTable i coins1 = g.residualTable i coins2 :=
    g.residualTable_invariant_of_prefix i hPrefix
  let mk : Array F → Array F := fun vals =>
    let v0 := arraySum (Array.ofFn (fun j : Fin (vals.size / 2) => vals[2 * j.1]!))
    let v1 := arraySum (Array.ofFn (fun j : Fin (vals.size / 2) => vals[2 * j.1 + 1]!))
    Array.ofFn (fun k : Fin (g.inst.maxDegree + 1) =>
      if h0 : k.1 = 0 then
        v0
      else if h1 : k.1 = 1 then
        v1 - v0
      else
        0)
  have hMk : mk (g.residualTable i coins1) = mk (g.residualTable i coins2) := congrArg mk hRes
  simpa [SoundnessGame.honestRoundPoly, mk] using hMk

@[simp] theorem SoundnessGame.honestRoundPoly_size
  (g : SoundnessGame)
  (i : Nat)
  (coins : Array F) :
  (g.honestRoundPoly i coins).size = g.inst.maxDegree + 1 := by
  simp [SoundnessGame.honestRoundPoly]

/--
Prefix-indexed honest round polynomial.

This is the faithful witness object: it depends on the verifier prefix of length
`i`, not on a single global round polynomial shared across all prefixes.
-/
def SoundnessGame.honestRoundPolyOfPrefix
  (g : SoundnessGame)
  (i : Nat)
  (pre : Array F) : Array F :=
  g.honestRoundPoly i (prefixPadCoins g.inst pre)

/-- Claimed value carried by the prover transcript after `i` rounds. -/
def SoundnessGame.claimedAt
  (g : SoundnessGame) : Nat → Array F → F
  | 0, _ => g.inst.claimedValue
  | i + 1, coins => SuperNeo.sumcheckEvalPoly (g.prover.roundPoly i coins) (coins[i]!)

/-- Honest residual claim after fixing the first `i` verifier challenges. -/
def SoundnessGame.honestClaim
  (g : SoundnessGame)
  (i : Nat)
  (coins : Array F) : F :=
  SuperNeo.sumcheckTableSum (g.residualTable i coins)

/-- Prefix-indexed prover claim surface. -/
def SoundnessGame.claimedAtPrefix
  (g : SoundnessGame)
  (i : Nat)
  (pre : Array F) : F :=
  g.claimedAt i (prefixPadCoins g.inst pre)

/-- Prefix-indexed honest residual claim surface. -/
def SoundnessGame.honestClaimOfPrefix
  (g : SoundnessGame)
  (i : Nat)
  (pre : Array F) : F :=
  g.honestClaim i (prefixPadCoins g.inst pre)

/-- Prefix-local false-claim predicate. -/
def SoundnessGame.falseClaimAtPrefix
  (g : SoundnessGame)
  (i : Nat)
  (pre : Array F) : Prop :=
  g.claimedAtPrefix i pre ≠ g.honestClaimOfPrefix i pre

theorem SoundnessGame.claimedAt_zero
  (g : SoundnessGame)
  (coins : Array F) :
  g.claimedAt 0 coins = g.inst.claimedValue := by
  rfl

theorem SoundnessGame.honestClaim_zero
  (g : SoundnessGame)
  (coins : Array F) :
  g.honestClaim 0 coins = SuperNeo.sumcheckTableSum g.table := by
  simp [SoundnessGame.honestClaim, SoundnessGame.residualTable]

theorem SoundnessGame.falseClaimAtPrefix_zero
  (g : SoundnessGame)
  (pre : Array F) :
  g.falseClaimAtPrefix 0 pre := by
  simpa [SoundnessGame.falseClaimAtPrefix, SoundnessGame.claimedAtPrefix,
    SoundnessGame.honestClaimOfPrefix, SoundnessGame.claimedAt_zero,
    SoundnessGame.honestClaim_zero, ne_comm] using g.falseClaim

theorem SoundnessGame.claimedAt_invariant_of_prefix
  (g : SoundnessGame)
  (i : Nat)
  (hi : i ≤ g.inst.rounds)
  {coins1 coins2 : Array F}
  (hPrefix : ∀ j : Nat, j < i → coins1[j]! = coins2[j]!) :
  g.claimedAt i coins1 = g.claimedAt i coins2 := by
  induction i with
  | zero =>
      rfl
  | succ i ih =>
      have hRound :
          g.prover.roundPoly i coins1 = g.prover.roundPoly i coins2 :=
        g.prover.nonanticipatory i (by omega) (by
            intro j hj
            exact hPrefix j (by omega))
      have hHead : coins1[i]! = coins2[i]! := hPrefix i (Nat.lt_succ_self i)
      simp [SoundnessGame.claimedAt, hRound, hHead]

theorem SoundnessGame.honestClaim_invariant_of_prefix
  (g : SoundnessGame)
  (i : Nat)
  {coins1 coins2 : Array F}
  (hPrefix : ∀ j : Nat, j < i → coins1[j]! = coins2[j]!) :
  g.honestClaim i coins1 = g.honestClaim i coins2 := by
  simp [SoundnessGame.honestClaim, g.residualTable_invariant_of_prefix i hPrefix]

theorem SoundnessGame.claimedAt_eq_of_extract
  (g : SoundnessGame)
  (i : Nat)
  {coins : Array F}
  (hSize : i ≤ g.inst.rounds)
  (hCoins : coins.size = g.inst.rounds) :
  g.claimedAt i coins =
    g.claimedAtPrefix i (coins.extract 0 i) := by
  unfold SoundnessGame.claimedAtPrefix
  exact g.claimedAt_invariant_of_prefix i hSize (by
    intro j hj
    have hjCoins : j < coins.size := by
      simpa [hCoins] using (Nat.lt_of_lt_of_le hj hSize)
    have hjExtract : j < (coins.extract 0 i).size := by
      simpa [Array.size_extract, hSize, hCoins] using hj
    rw [prefixPadCoins_get_prefix g.inst (coins.extract 0 i)]
    · rw [getElem!_pos (c := coins.extract 0 i) (i := j) hjExtract]
      rw [getElem!_pos (c := coins) (i := j) hjCoins]
      simpa [Nat.zero_add] using
        (Array.getElem_extract (xs := coins) (start := 0) (stop := i) (i := j) hjExtract)
    · simpa [Array.size_extract, hSize, hCoins] using hj
    · simpa [prefixPadCoins_size, hCoins] using hjCoins)

theorem SoundnessGame.honestClaim_eq_of_extract
  (g : SoundnessGame)
  (i : Nat)
  {coins : Array F}
  (hSize : i ≤ g.inst.rounds)
  (hCoins : coins.size = g.inst.rounds) :
  g.honestClaim i coins =
    g.honestClaimOfPrefix i (coins.extract 0 i) := by
  unfold SoundnessGame.honestClaimOfPrefix
  exact g.honestClaim_invariant_of_prefix i (by
    intro j hj
    have hjCoins : j < coins.size := by
      simpa [hCoins] using (Nat.lt_of_lt_of_le hj hSize)
    have hjExtract : j < (coins.extract 0 i).size := by
      simpa [Array.size_extract, hSize, hCoins] using hj
    rw [prefixPadCoins_get_prefix g.inst (coins.extract 0 i)]
    · rw [getElem!_pos (c := coins.extract 0 i) (i := j) hjExtract]
      rw [getElem!_pos (c := coins) (i := j) hjCoins]
      simpa [Nat.zero_add] using
        (Array.getElem_extract (xs := coins) (start := 0) (stop := i) (i := j) hjExtract)
    · simpa [Array.size_extract, hSize, hCoins] using hj
    · simpa [prefixPadCoins_size, hCoins] using hjCoins)

theorem SoundnessGame.proverRoundPoly_eq_of_extract
  (g : SoundnessGame)
  (i : Nat)
  {coins : Array F}
  (hi : i < g.inst.rounds)
  (hCoins : coins.size = g.inst.rounds) :
  g.prover.roundPoly i coins =
    g.prover.roundPoly i (prefixPadCoins g.inst (coins.extract 0 i)) := by
  symm
  exact g.prover.nonanticipatory i hi (by
    intro j hj
    have hjCoins : j < coins.size := by
      simpa [hCoins] using (Nat.lt_trans hj hi)
    have hjExtract : j < (coins.extract 0 i).size := by
      simpa [Array.size_extract, hi.le, hCoins] using hj
    rw [prefixPadCoins_get_prefix g.inst (coins.extract 0 i)]
    · rw [getElem!_pos (c := coins.extract 0 i) (i := j) hjExtract]
      rw [getElem!_pos (c := coins) (i := j) hjCoins]
      simpa [Nat.zero_add] using
        (Array.getElem_extract (xs := coins) (start := 0) (stop := i) (i := j) hjExtract)
    · simpa [Array.size_extract, hi.le, hCoins] using hj
    · simpa [prefixPadCoins_size, hCoins] using hjCoins)

theorem SoundnessGame.transcript_roundPoly_get!
  (g : SoundnessGame)
  (coins : Array F)
  {i : Nat}
  (hi : i < g.inst.rounds) :
  (g.transcript coins).roundPolys[i]! = g.prover.roundPoly i coins := by
  simp [SoundnessGame.transcript, hi]

theorem SoundnessGame.transcript_challenge_get!
  (g : SoundnessGame)
  (coins : Array F)
  {i : Nat}
  (hi : i < coins.size) :
  (g.transcript coins).challenges[i]! = coins[i]! := by
  simp [SoundnessGame.transcript, hi]

theorem SoundnessGame.acceptsOn_round_sum_eq_claimedAt
  (g : SoundnessGame)
  (coins : Array F)
  (hAcc : g.acceptsOn coins)
  {i : Nat}
  (hi : i < g.inst.rounds)
  (hRoundsPos : 0 < g.inst.rounds) :
  SuperNeo.sumcheckEvalPoly (g.prover.roundPoly i coins) 0 +
      SuperNeo.sumcheckEvalPoly (g.prover.roundPoly i coins) 1 =
    g.claimedAt i coins := by
  have hCore : SuperNeo.sumcheckAcceptedCore g.inst (g.transcript coins) := hAcc.1
  have hRoundCons : SuperNeo.sumcheckRoundConsistent g.inst (g.transcript coins) := hCore.2.2.1
  have hCoinsSize : coins.size = g.inst.rounds := by
    simpa [SoundnessGame.transcript] using hRoundCons.1
  cases i with
  | zero =>
      have hInit :
          SuperNeo.sumcheckEvalPoly (g.prover.roundPoly 0 coins) 0 +
              SuperNeo.sumcheckEvalPoly (g.prover.roundPoly 0 coins) 1 =
            g.inst.claimedValue := by
        have hZeroNe : ¬ (g.transcript coins).roundPolys.size = 0 := by
          simpa [SoundnessGame.transcript] using Nat.ne_of_gt hRoundsPos
        have hPoly0 :
            (g.transcript coins).roundPolys[0]! = g.prover.roundPoly 0 coins := by
          simpa [SoundnessGame.transcript] using
            g.transcript_roundPoly_get! coins (i := 0) hRoundsPos
        have hInitCore : SuperNeo.sumcheckInitialRoundConsistent g.inst (g.transcript coins) :=
          hCore.2.2.2.2.1
        unfold SuperNeo.sumcheckInitialRoundConsistent at hInitCore
        simp [hZeroNe] at hInitCore
        simpa [hPoly0] using hInitCore
      simpa [SoundnessGame.claimedAt] using hInit
  | succ i =>
      have hiPrev : i + 1 < (g.transcript coins).roundPolys.size := by
        simpa [SoundnessGame.transcript] using hi
      have hFoldCore : SuperNeo.sumcheckFoldConsistent (g.transcript coins) := hCore.2.2.2.2.2
      have hFold :=
        hFoldCore.2 i hiPrev
      have hPoly :
          (g.transcript coins).roundPolys[i + 1]! = g.prover.roundPoly (i + 1) coins := by
        simpa [SoundnessGame.transcript] using
          g.transcript_roundPoly_get! coins (i := i + 1) hi
      have hChallenge :
          (g.transcript coins).challenges[i]! = coins[i]! := by
        exact g.transcript_challenge_get! coins (i := i) (by
          simpa [hCoinsSize] using (Nat.lt_trans (Nat.lt_succ_self i) hi))
      have hPrevPoly :
          (g.transcript coins).roundPolys[i]! = g.prover.roundPoly i coins := by
        simpa [SoundnessGame.transcript] using
          g.transcript_roundPoly_get! coins (i := i) (by
            simpa [hCoinsSize] using (Nat.lt_trans (Nat.lt_succ_self i) hi))
      simpa [SoundnessGame.claimedAt, hPoly, hPrevPoly, hChallenge] using hFold

theorem SoundnessGame.acceptsOn_final_claim_eq_honest
  (g : SoundnessGame)
  (coins : Array F)
  (hAcc : g.acceptsOn coins)
  (hRoundsPos : 0 < g.inst.rounds) :
  g.claimedAt g.inst.rounds coins = mleByFolding g.table coins := by
  have hRoundsNe : g.inst.rounds ≠ 0 := Nat.ne_of_gt hRoundsPos
  have hLastLt : g.inst.rounds - 1 < g.inst.rounds := by omega
  have hFinal :
      SuperNeo.sumcheckEvalPoly (g.prover.roundPoly (g.inst.rounds - 1) coins)
          (coins[g.inst.rounds - 1]!) = mleByFolding g.table coins := by
    simpa [SoundnessGame.acceptsOn, SoundnessGame.transcript,
      SuperNeo.sumcheckFinalOracleConsistentWithTable, hRoundsNe, hLastLt] using hAcc.2.2.2
  have hRounds :
      g.inst.rounds = (g.inst.rounds - 1) + 1 := by
    omega
  rw [hRounds, SoundnessGame.claimedAt]
  simpa using hFinal

private theorem sumcheckEvalPoly_zero_of_sumcheckPolynomialZMod_eq_zero
  (poly : Array F)
  (x : F)
  (hZero : sumcheckPolynomialZMod poly = 0) :
  SuperNeo.sumcheckEvalPoly poly x = 0 := by
  apply fToZMod_injective
  rw [sumcheckEvalPoly_fToZMod, hZero]
  simp [fToZMod_zero]

private theorem sumcheckEvalPoly_linear_interp_ofFn
  (n : Nat)
  (v0 v1 x : F) :
  SuperNeo.sumcheckEvalPoly
      (Array.ofFn (fun k : Fin (n + 2) =>
        if h0 : k.1 = 0 then
          v0
        else if h1 : k.1 = 1 then
          v1 - v0
        else
          0)) x =
    v0 * ((1 : F) - x) + v1 * x := by
  rw [sumcheckEvalPoly_ofFn_succ]
  rw [sumcheckEvalPoly_ofFn_succ]
  simp [sumcheckEvalPoly_zero_ofFn, linear_interp_eval]

private theorem prefixPadCoins_push_eq_on_prefix
  (inst : Instance)
  (pre : Array F)
  (x : F)
  (i : Nat)
  (hSize : pre.size = i)
  (hi : i < inst.rounds)
  {j : Nat}
  (hj : j < i) :
  (prefixPadCoins inst (pre.push x))[j]! = (prefixPadCoins inst pre)[j]! := by
  have hjPre : j < pre.size := by simpa [hSize] using hj
  have hjPush : j < (pre.push x).size := by
    simpa [Array.size_push, hSize] using Nat.lt_succ_of_lt hjPre
  have hjRounds : j < inst.rounds := Nat.lt_trans hj hi
  rw [prefixPadCoins_get_prefix inst (pre.push x) (i := j)]
  · rw [prefixPadCoins_get_prefix inst pre (i := j)]
    · rw [getElem!_pos (c := pre.push x) (i := j) hjPush]
      · rw [getElem!_pos (c := pre) (i := j) hjPre]
        simpa using Array.getElem_push_lt (xs := pre) (x := x) (i := j) hjPre
    · exact hjPre
    · exact hjRounds
  · exact hjPush
  · exact hjRounds

/-- Prefix-local round-sum consistency for the prover polynomial at round `i`. -/
def SoundnessGame.prefixRoundConsistent
  (g : SoundnessGame)
  (i : Nat)
  (pre : Array F) : Prop :=
  SuperNeo.sumcheckEvalPoly (g.prover.roundPoly i (prefixPadCoins g.inst pre)) 0 +
      SuperNeo.sumcheckEvalPoly (g.prover.roundPoly i (prefixPadCoins g.inst pre)) 1 =
    g.claimedAtPrefix i pre

theorem SoundnessGame.claimedAtPrefix_succ
  (g : SoundnessGame)
  (i : Nat)
  (pre : Array F)
  (r : F)
  (hSize : pre.size = i)
  (hi : i < g.inst.rounds) :
  g.claimedAtPrefix (i + 1) (pre.push r) =
    SuperNeo.sumcheckEvalPoly (g.prover.roundPoly i (prefixPadCoins g.inst pre)) r := by
  unfold SoundnessGame.claimedAtPrefix SoundnessGame.claimedAt
  have hPoly :
      g.prover.roundPoly i (prefixPadCoins g.inst (pre.push r)) =
        g.prover.roundPoly i (prefixPadCoins g.inst pre) := by
    apply g.prover.nonanticipatory i hi
    intro j hj
    exact prefixPadCoins_push_eq_on_prefix g.inst pre r i hSize hi hj
  have hiPush : i < (pre.push r).size := by
    simpa [Array.size_push, hSize]
  have hVal :
      (prefixPadCoins g.inst (pre.push r))[i]! = r := by
    rw [prefixPadCoins_get_prefix g.inst (pre.push r) (i := i)]
    · rw [getElem!_pos (c := pre.push r) (i := i) hiPush]
      simpa [hSize] using Array.getElem_push_eq (xs := pre) (x := r)
    · exact hiPush
    · exact hi
  simpa [hPoly, hVal]

theorem SoundnessGame.honestClaimOfPrefix_split
  (g : SoundnessGame)
  (i : Nat)
  (pre : Array F)
  (hSize : pre.size = i)
  (hi : i < g.inst.rounds) :
  g.honestClaimOfPrefix (i + 1) (pre.push 0) +
      g.honestClaimOfPrefix (i + 1) (pre.push 1) =
    g.honestClaimOfPrefix i pre := by
  let coins := prefixPadCoins g.inst pre
  let vals := g.residualTable i coins
  let v0 := arraySum (Array.ofFn (fun j : Fin (vals.size / 2) => vals[2 * j.1]!))
  let v1 := arraySum (Array.ofFn (fun j : Fin (vals.size / 2) => vals[2 * j.1 + 1]!))
  have hRes0 :
      g.residualTable i (prefixPadCoins g.inst (pre.push 0)) = vals := by
    apply g.residualTable_invariant_of_prefix i
    intro j hj
    exact prefixPadCoins_push_eq_on_prefix g.inst pre 0 i hSize hi hj
  have hRes1 :
      g.residualTable i (prefixPadCoins g.inst (pre.push 1)) = vals := by
    apply g.residualTable_invariant_of_prefix i
    intro j hj
    exact prefixPadCoins_push_eq_on_prefix g.inst pre 1 i hSize hi hj
  have hiPush0 : i < (pre.push (0 : F)).size := by
    simpa [Array.size_push, hSize]
  have hiPush1 : i < (pre.push (1 : F)).size := by
    simpa [Array.size_push, hSize]
  have hVal0 :
      (prefixPadCoins g.inst (pre.push (0 : F)))[i]! = 0 := by
    rw [prefixPadCoins_get_prefix g.inst (pre.push (0 : F)) (i := i)]
    · rw [getElem!_pos (c := pre.push (0 : F)) (i := i) hiPush0]
      simpa [hSize] using Array.getElem_push_eq (xs := pre) (x := (0 : F))
    · exact hiPush0
    · exact hi
  have hVal1 :
      (prefixPadCoins g.inst (pre.push (1 : F)))[i]! = 1 := by
    rw [prefixPadCoins_get_prefix g.inst (pre.push (1 : F)) (i := i)]
    · rw [getElem!_pos (c := pre.push (1 : F)) (i := i) hiPush1]
      simpa [hSize] using Array.getElem_push_eq (xs := pre) (x := (1 : F))
    · exact hiPush1
    · exact hi
  have hNext0 :
      g.residualTable (i + 1) (prefixPadCoins g.inst (pre.push 0)) =
        foldLayerLocal vals 0 := by
    simpa [SoundnessGame.residualTable, vals, hRes0, hVal0]
  have hNext1 :
      g.residualTable (i + 1) (prefixPadCoins g.inst (pre.push 1)) =
        foldLayerLocal vals 1 := by
    simpa [SoundnessGame.residualTable, vals, hRes1, hVal1]
  have hCurrentSize :
      vals.size = 2 * (2 ^ (g.inst.rounds - (i + 1))) := by
    have hVals : vals.size = 2 ^ (g.inst.rounds - i) := by
      simpa [vals] using g.residualTable_size i coins hi.le
    have hExp : g.inst.rounds - i = (g.inst.rounds - (i + 1)) + 1 := by
      omega
    calc
      vals.size = 2 ^ (g.inst.rounds - i) := hVals
      _ = 2 * (2 ^ (g.inst.rounds - (i + 1))) := by
            rw [hExp, Nat.pow_succ]
            ring
  have hCurrent :
      g.honestClaimOfPrefix i pre = v0 + v1 := by
    unfold SoundnessGame.honestClaimOfPrefix SoundnessGame.honestClaim
    rw [sumcheckTableSum_eq_arraySum]
    have hEven : vals.size = 2 * (vals.size / 2) := by
      omega
    change arraySum vals =
      arraySum (Array.ofFn (fun j : Fin (vals.size / 2) => vals[2 * j.1]!)) +
        arraySum (Array.ofFn (fun j : Fin (vals.size / 2) => vals[2 * j.1 + 1]!))
    simpa [v0, v1] using (arraySum_even_add_odd vals hEven)
  have hNextClaim0 :
      g.honestClaimOfPrefix (i + 1) (pre.push 0) = v0 := by
    unfold SoundnessGame.honestClaimOfPrefix SoundnessGame.honestClaim
    rw [sumcheckTableSum_eq_arraySum, hNext0]
    simpa [v0, v1] using arraySum_foldLayerLocal vals (0 : F)
  have hNextClaim1 :
      g.honestClaimOfPrefix (i + 1) (pre.push 1) = v1 := by
    unfold SoundnessGame.honestClaimOfPrefix SoundnessGame.honestClaim
    rw [sumcheckTableSum_eq_arraySum, hNext1]
    simpa [v0, v1] using arraySum_foldLayerLocal vals (1 : F)
  calc
    g.honestClaimOfPrefix (i + 1) (pre.push 0) +
        g.honestClaimOfPrefix (i + 1) (pre.push 1)
        = v0 + v1 := by rw [hNextClaim0, hNextClaim1]
    _ = g.honestClaimOfPrefix i pre := hCurrent.symm

theorem SoundnessGame.honestClaimOfPrefix_succ
  (g : SoundnessGame)
  (i : Nat)
  (pre : Array F)
  (r : F)
  (hSize : pre.size = i)
  (hi : i < g.inst.rounds)
  (hDegPos : 0 < g.inst.maxDegree) :
  g.honestClaimOfPrefix (i + 1) (pre.push r) =
    SuperNeo.sumcheckEvalPoly (g.honestRoundPolyOfPrefix i pre) r := by
  rcases Nat.exists_eq_succ_of_ne_zero (Nat.ne_of_gt hDegPos) with ⟨n, hDeg⟩
  let coins := prefixPadCoins g.inst pre
  let vals := g.residualTable i coins
  let v0 := arraySum (Array.ofFn (fun j : Fin (vals.size / 2) => vals[2 * j.1]!))
  let v1 := arraySum (Array.ofFn (fun j : Fin (vals.size / 2) => vals[2 * j.1 + 1]!))
  have hRes :
      g.residualTable i (prefixPadCoins g.inst (pre.push r)) = vals := by
    apply g.residualTable_invariant_of_prefix i
    intro j hj
    exact prefixPadCoins_push_eq_on_prefix g.inst pre r i hSize hi hj
  have hiPush : i < (pre.push r).size := by
    simpa [Array.size_push, hSize]
  have hVal :
      (prefixPadCoins g.inst (pre.push r))[i]! = r := by
    rw [prefixPadCoins_get_prefix g.inst (pre.push r) (i := i)]
    · rw [getElem!_pos (c := pre.push r) (i := i) hiPush]
      simpa [hSize] using Array.getElem_push_eq (xs := pre) (x := r)
    · exact hiPush
    · exact hi
  have hNext :
      g.residualTable (i + 1) (prefixPadCoins g.inst (pre.push r)) =
        foldLayerLocal vals r := by
    simpa [SoundnessGame.residualTable, vals, hRes, hVal]
  have hEval :
      SuperNeo.sumcheckEvalPoly
          (Array.ofFn (fun k : Fin (n + 2) =>
            if h0 : k.1 = 0 then
              v0
            else if h1 : k.1 = 1 then
              v1 - v0
            else
              0)) r =
        v0 * ((1 : F) - r) + v1 * r :=
    sumcheckEvalPoly_linear_interp_ofFn n v0 v1 r
  have hEvalHonest :
      SuperNeo.sumcheckEvalPoly (g.honestRoundPolyOfPrefix i pre) r =
        v0 * ((1 : F) - r) + v1 * r := by
    rw [SoundnessGame.honestRoundPolyOfPrefix, SoundnessGame.honestRoundPoly, hDeg]
    simpa [coins, vals, v0, v1] using hEval
  calc
    g.honestClaimOfPrefix (i + 1) (pre.push r)
        = arraySum (foldLayerLocal vals r) := by
            unfold SoundnessGame.honestClaimOfPrefix SoundnessGame.honestClaim
            rw [sumcheckTableSum_eq_arraySum, hNext]
    _ = v0 * ((1 : F) - r) + v1 * r := by
          simpa [v0, v1, linear_interp_eval] using (arraySum_foldLayerLocal vals r)
    _ = SuperNeo.sumcheckEvalPoly (g.honestRoundPolyOfPrefix i pre) r := by
          exact hEvalHonest.symm

def SoundnessGame.prefixGapPoly
  (g : SoundnessGame)
  (i : Nat)
  (pre : Array F) : Array F :=
  Array.ofFn (fun k : Fin (g.inst.maxDegree + 1) =>
    (g.prover.roundPoly i (prefixPadCoins g.inst pre))[k.1]! -
      (g.honestRoundPolyOfPrefix i pre)[k.1]!)

@[simp] theorem SoundnessGame.prefixGapPoly_size
  (g : SoundnessGame)
  (i : Nat)
  (pre : Array F) :
  (g.prefixGapPoly i pre).size = g.inst.maxDegree + 1 := by
  simp [SoundnessGame.prefixGapPoly]

theorem SoundnessGame.prefixGapPoly_eval_sub
  (g : SoundnessGame)
  (i : Nat)
  (pre : Array F)
  (r : F)
  (hi : i < g.inst.rounds) :
  SuperNeo.sumcheckEvalPoly (g.prefixGapPoly i pre) r =
    SuperNeo.sumcheckEvalPoly (g.prover.roundPoly i (prefixPadCoins g.inst pre)) r -
      SuperNeo.sumcheckEvalPoly (g.honestRoundPolyOfPrefix i pre) r := by
  simpa [SoundnessGame.prefixGapPoly, SoundnessGame.honestRoundPoly_size] using
    (sumcheckEvalPoly_sub_eq
      (a := g.prover.roundPoly i (prefixPadCoins g.inst pre))
      (b := g.honestRoundPolyOfPrefix i pre)
      (x := r)
      (hA := g.prover.roundPolyShape i hi (prefixPadCoins g.inst pre))
      (hB := g.honestRoundPoly_size i (prefixPadCoins g.inst pre)))

theorem SoundnessGame.prefixRoundConsistent_of_acceptsOn
  (g : SoundnessGame)
  (coins : Array F)
  (hAcc : g.acceptsOn coins)
  {i : Nat}
  (hi : i < g.inst.rounds)
  (hRoundsPos : 0 < g.inst.rounds) :
  g.prefixRoundConsistent i (coins.extract 0 i) := by
  have hCoinsSize : coins.size = g.inst.rounds := by
    have hCore : SuperNeo.sumcheckAcceptedCore g.inst (g.transcript coins) := hAcc.1
    have hRoundCons : SuperNeo.sumcheckRoundConsistent g.inst (g.transcript coins) := hCore.2.2.1
    simpa [SoundnessGame.transcript] using hRoundCons.1
  unfold SoundnessGame.prefixRoundConsistent
  rw [← g.proverRoundPoly_eq_of_extract i hi hCoinsSize]
  rw [← g.claimedAt_eq_of_extract i hi.le hCoinsSize]
  exact g.acceptsOn_round_sum_eq_claimedAt coins hAcc hi hRoundsPos

theorem SoundnessGame.not_falseClaimAtPrefix_rounds_of_acceptsOn
  (g : SoundnessGame)
  (coins : Array F)
  (hAcc : g.acceptsOn coins)
  (hRoundsPos : 0 < g.inst.rounds) :
  ¬ g.falseClaimAtPrefix g.inst.rounds (coins.extract 0 g.inst.rounds) := by
  have hCoinsSize : coins.size = g.inst.rounds := by
    have hCore : SuperNeo.sumcheckAcceptedCore g.inst (g.transcript coins) := hAcc.1
    have hRoundCons : SuperNeo.sumcheckRoundConsistent g.inst (g.transcript coins) := hCore.2.2.1
    simpa [SoundnessGame.transcript] using hRoundCons.1
  have hClaimEq :
      g.claimedAtPrefix g.inst.rounds (coins.extract 0 g.inst.rounds) =
        g.honestClaimOfPrefix g.inst.rounds (coins.extract 0 g.inst.rounds) := by
    have hHonestFinal :
        g.honestClaim g.inst.rounds coins = mleByFolding g.table coins := by
      have hTail :
          coins.extract g.inst.rounds coins.size = #[] := by
        simpa [hCoinsSize] using
          (Array.extract_eq_nil_of_ge
            (as := coins) (i := g.inst.rounds) (j := coins.size) (by omega))
      have hFold :
          mleByFolding (g.residualTable g.inst.rounds coins) #[] =
            mleByFolding g.table coins := by
        simpa [hTail] using
          (g.residualTable_mleByFolding_tail g.inst.rounds coins le_rfl hCoinsSize)
      have hResNe : (g.residualTable g.inst.rounds coins).size ≠ 0 := by
        have hResSize := g.residualTable_size g.inst.rounds coins le_rfl
        simpa [hResSize]
      have hHead :
          mleByFolding (g.residualTable g.inst.rounds coins) #[] =
            (g.residualTable g.inst.rounds coins)[0]! := by
        exact mleByFolding_empty (g.residualTable g.inst.rounds coins) hResNe
      have hArray :
          arraySum (g.residualTable g.inst.rounds coins) =
            (g.residualTable g.inst.rounds coins)[0]! := by
        have hResSize := g.residualTable_size g.inst.rounds coins le_rfl
        simp [arraySum, hResSize]
      have hCore :
          arraySum (g.residualTable g.inst.rounds coins) = mleByFolding g.table coins := by
        calc
          arraySum (g.residualTable g.inst.rounds coins)
              = (g.residualTable g.inst.rounds coins)[0]! := hArray
          _ = mleByFolding (g.residualTable g.inst.rounds coins) #[] := hHead.symm
          _ = mleByFolding g.table coins := hFold
      simpa [SoundnessGame.honestClaim, sumcheckTableSum_eq_arraySum] using hCore
    calc
      g.claimedAtPrefix g.inst.rounds (coins.extract 0 g.inst.rounds)
          = g.claimedAt g.inst.rounds coins := by
              symm
              exact g.claimedAt_eq_of_extract g.inst.rounds le_rfl hCoinsSize
      _ = mleByFolding g.table coins := g.acceptsOn_final_claim_eq_honest coins hAcc hRoundsPos
      _ = g.honestClaim g.inst.rounds coins := hHonestFinal.symm
      _ = g.honestClaimOfPrefix g.inst.rounds (coins.extract 0 g.inst.rounds) := by
            exact g.honestClaim_eq_of_extract g.inst.rounds le_rfl hCoinsSize
  simpa [SoundnessGame.falseClaimAtPrefix] using hClaimEq

private theorem exists_true_false_transition
  (P : Nat → Prop)
  {n : Nat}
  (h0 : P 0)
  (hn : ¬ P n) :
  ∃ i, i < n ∧ P i ∧ ¬ P (i + 1) := by
  induction n generalizing P with
  | zero =>
      exact False.elim (hn h0)
  | succ n ih =>
      by_cases h1 : P 1
      · rcases ih (P := fun k => P (k + 1)) h1 hn with ⟨i, hi, hPi, hNext⟩
        exact ⟨i + 1, by omega, hPi, hNext⟩
      · exact ⟨0, Nat.succ_pos _, h0, h1⟩

theorem SoundnessGame.prefixGapPoly_eval_zero_of_repair
  (g : SoundnessGame)
  (i : Nat)
  (pre : Array F)
  (r : F)
  (hSize : pre.size = i)
  (hi : i < g.inst.rounds)
  (hRound : g.prefixRoundConsistent i pre)
  (hRepair : ¬ g.falseClaimAtPrefix (i + 1) (pre.push r))
  (hDegPos : 0 < g.inst.maxDegree) :
  SuperNeo.sumcheckEvalPoly (g.prefixGapPoly i pre) r = 0 := by
  unfold SoundnessGame.falseClaimAtPrefix at hRepair
  have hRepairEq :
      g.claimedAtPrefix (i + 1) (pre.push r) =
        g.honestClaimOfPrefix (i + 1) (pre.push r) := by
    exact not_ne_iff.mp hRepair
  rw [g.prefixGapPoly_eval_sub i pre r hi]
  have hClaim :
      SuperNeo.sumcheckEvalPoly (g.prover.roundPoly i (prefixPadCoins g.inst pre)) r =
        g.claimedAtPrefix (i + 1) (pre.push r) := by
    exact (g.claimedAtPrefix_succ i pre r hSize hi).symm
  have hHonest :
      SuperNeo.sumcheckEvalPoly (g.honestRoundPolyOfPrefix i pre) r =
        g.honestClaimOfPrefix (i + 1) (pre.push r) := by
    exact (g.honestClaimOfPrefix_succ i pre r hSize hi hDegPos).symm
  rw [hClaim, hHonest]
  exact sub_eq_zero.mpr hRepairEq

theorem SoundnessGame.prefixGapPoly_nonzero_of_falseClaim
  (g : SoundnessGame)
  (i : Nat)
  (pre : Array F)
  (hSize : pre.size = i)
  (hi : i < g.inst.rounds)
  (hRound : g.prefixRoundConsistent i pre)
  (hFalse : g.falseClaimAtPrefix i pre)
  (hDegPos : 0 < g.inst.maxDegree) :
  sumcheckPolynomialZMod (g.prefixGapPoly i pre) ≠ 0 := by
  intro hZero
  have hGapEval0 :
      SuperNeo.sumcheckEvalPoly (g.prover.roundPoly i (prefixPadCoins g.inst pre)) 0 =
        SuperNeo.sumcheckEvalPoly (g.honestRoundPolyOfPrefix i pre) 0 := by
    have hEval0 :
        SuperNeo.sumcheckEvalPoly (g.prefixGapPoly i pre) 0 = 0 :=
      sumcheckEvalPoly_zero_of_sumcheckPolynomialZMod_eq_zero (g.prefixGapPoly i pre) 0 hZero
    rw [g.prefixGapPoly_eval_sub i pre 0 hi] at hEval0
    exact sub_eq_zero.mp hEval0
  have hGapEval1 :
      SuperNeo.sumcheckEvalPoly (g.prover.roundPoly i (prefixPadCoins g.inst pre)) 1 =
        SuperNeo.sumcheckEvalPoly (g.honestRoundPolyOfPrefix i pre) 1 := by
    have hEval1 :
        SuperNeo.sumcheckEvalPoly (g.prefixGapPoly i pre) 1 = 0 :=
      sumcheckEvalPoly_zero_of_sumcheckPolynomialZMod_eq_zero (g.prefixGapPoly i pre) 1 hZero
    rw [g.prefixGapPoly_eval_sub i pre 1 hi] at hEval1
    exact sub_eq_zero.mp hEval1
  have hRepair0 :
      g.claimedAtPrefix (i + 1) (pre.push 0) =
        g.honestClaimOfPrefix (i + 1) (pre.push 0) := by
    rw [g.claimedAtPrefix_succ i pre 0 hSize hi,
      g.honestClaimOfPrefix_succ i pre 0 hSize hi hDegPos]
    exact hGapEval0
  have hRepair1 :
      g.claimedAtPrefix (i + 1) (pre.push 1) =
        g.honestClaimOfPrefix (i + 1) (pre.push 1) := by
    rw [g.claimedAtPrefix_succ i pre 1 hSize hi,
      g.honestClaimOfPrefix_succ i pre 1 hSize hi hDegPos]
    exact hGapEval1
  have hCurrent :
      g.claimedAtPrefix i pre = g.honestClaimOfPrefix i pre := by
    calc
      g.claimedAtPrefix i pre
          = SuperNeo.sumcheckEvalPoly (g.prover.roundPoly i (prefixPadCoins g.inst pre)) 0 +
              SuperNeo.sumcheckEvalPoly (g.prover.roundPoly i (prefixPadCoins g.inst pre)) 1 := by
                symm
                exact hRound
      _ = g.claimedAtPrefix (i + 1) (pre.push 0) +
            g.claimedAtPrefix (i + 1) (pre.push 1) := by
              rw [g.claimedAtPrefix_succ i pre 0 hSize hi,
                g.claimedAtPrefix_succ i pre 1 hSize hi]
      _ = g.honestClaimOfPrefix (i + 1) (pre.push 0) +
            g.honestClaimOfPrefix (i + 1) (pre.push 1) := by
              rw [hRepair0, hRepair1]
      _ = g.honestClaimOfPrefix i pre := by
            exact g.honestClaimOfPrefix_split i pre hSize hi
  exact hFalse hCurrent


end Sumcheck

end SuperNeo.ProofSystem
