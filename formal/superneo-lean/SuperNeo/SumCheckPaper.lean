import SuperNeo.SumCheckCore
import SuperNeo.PolynomialBridge
import Mathlib

namespace SuperNeo

/-- Paper-facing degree bound for a coefficient-array encoding of a round polynomial. -/
def sumcheckRoundPolyDegreeLe (inst : SumCheckInstance) (poly : Array F) : Prop :=
  poly.size ≤ inst.maxDegree + 1

theorem sumcheckRoundPolyShape_degreeLe
  {inst : SumCheckInstance}
  {poly : Array F}
  (h : sumcheckRoundPolyShape inst poly) :
  sumcheckRoundPolyDegreeLe inst poly := by
  simpa [sumcheckRoundPolyDegreeLe, sumcheckRoundPolyShape] using Nat.le_of_eq h

/-- Paper-facing per-round degree bound extracted from the transcript encoding. -/
def sumcheckRoundDegrees
  (inst : SumCheckInstance)
  (tr : SumCheckTranscript) : Prop :=
  ∀ i : Fin tr.roundPolys.size,
    sumcheckRoundPolyDegreeLe inst tr.roundPolys[i.1]

/-- Paper-facing verifier checks: transcript shape, degree bound, initial, and fold. -/
def sumcheckVerifierAccepted
  (inst : SumCheckInstance)
  (tr : SumCheckTranscript) : Prop :=
  sumcheckRoundConsistent inst tr ∧
  sumcheckRoundDegrees inst tr ∧
  sumcheckInitialRoundConsistent inst tr ∧
  sumcheckFoldConsistent tr

theorem sumcheckVerifierAccepted_of_accepted
  {inst : SumCheckInstance}
  {tr : SumCheckTranscript}
  (hAcc : sumcheckAccepted inst tr) :
  sumcheckVerifierAccepted inst tr := by
  refine ⟨hAcc.2.2.1, ?_, hAcc.2.2.2.2.1, hAcc.2.2.2.2.2⟩
  intro i
  exact sumcheckRoundPolyShape_degreeLe (hAcc.2.2.2.1 i)

/-- Sum of all values in an array. -/
private def arraySum (vals : Array F) : F :=
  Finset.sum (Finset.range vals.size) (fun i => vals[i]!)

@[simp] private theorem arraySum_empty :
  arraySum (#[] : Array F) = 0 := by
  simp [arraySum]

@[simp] private theorem arraySum_ofFn
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
  {n : Nat}
  (vals : Array F)
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
        = Finset.sum (Finset.range (2 * n)) (fun i => vals[i]!) := by
            simp [arraySum, hSize]
    _ = Finset.sum (Finset.range n) (fun j => vals[2 * j]! + vals[2 * j + 1]!) := by
          symm
          exact sum_range_pairs n (fun i => vals[i]!)
    _ = Finset.sum (Finset.range n) (fun j => vals[2 * j]!) +
          Finset.sum (Finset.range n) (fun j => vals[2 * j + 1]!) := by
            rw [Finset.sum_add_distrib]
    _ = arraySum (Array.ofFn (fun j : Fin n => vals[2 * j.1]!)) +
          arraySum (Array.ofFn (fun j : Fin n => vals[2 * j.1 + 1]!)) := by
            rw [hEven, hOdd]

private theorem sumcheckEvalPoly_ofFn_succ
  (n : Nat)
  (v : Fin (n + 1) → F)
  (x : F) :
  sumcheckEvalPoly (Array.ofFn v) x =
    v 0 + x * sumcheckEvalPoly (Array.ofFn (fun i : Fin n => v i.succ)) x := by
  unfold sumcheckEvalPoly
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
  simpa [sumcheckEvalPoly] using congrArg (fun z => v 0 + x * z) hTail

private theorem sumcheckEvalPoly_zero_ofFn
  (n : Nat)
  (x : F) :
  sumcheckEvalPoly (Array.ofFn (fun _ : Fin n => (0 : F))) x = 0 := by
  induction n with
  | zero =>
      simp [sumcheckEvalPoly]
  | succ n ih =>
      rw [sumcheckEvalPoly_ofFn_succ]
      simp [ih]

private theorem sumcheckEvalPoly_headOnly_ofFn
  (n : Nat)
  (c x : F) :
  sumcheckEvalPoly
      (Array.ofFn (fun k : Fin (n + 1) => if k.1 = 0 then c else 0)) x = c := by
  rw [sumcheckEvalPoly_ofFn_succ]
  simp [sumcheckEvalPoly_zero_ofFn]

private theorem linear_interp_eval
  (a b x : F) :
  a + x * (b - a) = a * ((1 : F) - x) + b * x := by
  apply fToZMod_injective
  simp [fToZMod_add, fToZMod_mul, fToZMod_sub, fToZMod_one]
  ring

private theorem sumcheckEvalPoly_linear_interp_ofFn
  (n : Nat)
  (v0 v1 x : F) :
  sumcheckEvalPoly
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

private theorem arraySum_foldLayer
  (vals : Array F)
  (ri : F) :
  arraySum (foldLayer vals ri) =
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
    arraySum (foldLayer vals ri)
        = Finset.sum (Finset.range (vals.size / 2))
            (fun j => vals[2 * j]! * ((1 : F) - ri) + vals[2 * j + 1]! * ri) := by
              rw [foldLayer, arraySum_ofFn]
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

private theorem sumcheckTableSum_eq_arraySum
  (vals : Array F) :
  sumcheckTableSum vals = arraySum vals := by
  unfold sumcheckTableSum arraySum
  rw [← Array.foldr_toList]
  rw [← List.sum_eq_foldr]
  rw [show vals.toList = List.ofFn (fun i : Fin vals.size => vals[i.1]!) by
    simpa using (List.ofFn_getElem vals.toList).symm]
  rw [List.sum_ofFn, Fin.sum_univ_eq_sum_range]

private theorem sumcheckTableSum_foldLayer_split
  (vals : Array F)
  (hEven : vals.size = 2 * (vals.size / 2)) :
  sumcheckTableSum (foldLayer vals 0) + sumcheckTableSum (foldLayer vals 1) =
    sumcheckTableSum vals := by
  let evenVals := Array.ofFn (fun j : Fin (vals.size / 2) => vals[2 * j.1]!)
  let oddVals := Array.ofFn (fun j : Fin (vals.size / 2) => vals[2 * j.1 + 1]!)
  have h0 :
      sumcheckTableSum (foldLayer vals 0) = arraySum evenVals := by
    rw [sumcheckTableSum_eq_arraySum, arraySum_foldLayer]
    simp [evenVals, oddVals]
  have h1 :
      sumcheckTableSum (foldLayer vals 1) = arraySum oddVals := by
    rw [sumcheckTableSum_eq_arraySum, arraySum_foldLayer]
    simp [evenVals, oddVals]
  have hSum :
      sumcheckTableSum vals = arraySum evenVals + arraySum oddVals := by
    rw [sumcheckTableSum_eq_arraySum]
    simpa [evenVals, oddVals] using arraySum_even_add_odd (n := vals.size / 2) vals hEven
  calc
    sumcheckTableSum (foldLayer vals 0) + sumcheckTableSum (foldLayer vals 1)
        = arraySum evenVals + arraySum oddVals := by rw [h0, h1]
    _ = sumcheckTableSum vals := hSum.symm

private def sumcheckResidualTable
  (table : Array F) :
  Nat → Array F → Array F
  | 0, _ => table
  | i + 1, coins => foldLayer (sumcheckResidualTable table i coins) (coins[i]!)

private theorem sumcheckResidualTable_size
  {inst : SumCheckInstance}
  (table : Array F)
  (hTable : table.size = 2 ^ inst.rounds) :
  ∀ i coins, i ≤ inst.rounds →
    (sumcheckResidualTable table i coins).size = 2 ^ (inst.rounds - i)
  := by
  intro i
  induction i with
  | zero =>
      intro coins _hi
      simpa [sumcheckResidualTable] using hTable
  | succ i ih =>
      intro coins hi
      have hiLt : i < inst.rounds := by omega
      have hPrev := ih coins (Nat.le_of_lt hiLt)
      calc
        (sumcheckResidualTable table (i + 1) coins).size
            = (sumcheckResidualTable table i coins).size / 2 := by
                simp [sumcheckResidualTable]
        _ = (2 ^ (inst.rounds - i)) / 2 := by
              rw [hPrev]
        _ = 2 ^ (inst.rounds - (i + 1)) := by
              have hExp : inst.rounds - i = (inst.rounds - (i + 1)) + 1 := by omega
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

private theorem sumcheckResidualTable_mleByFolding_tail
  {inst : SumCheckInstance}
  (table : Array F)
  (hTable : table.size = 2 ^ inst.rounds) :
  ∀ i coins, i ≤ inst.rounds → coins.size = inst.rounds →
    mleByFolding (sumcheckResidualTable table i coins) (coins.extract i coins.size) =
      mleByFolding table coins
  := by
  intro i
  induction i with
  | zero =>
      intro coins _hi hCoins
      have hExtract : coins.extract 0 coins.size = coins := by
        simpa using (Array.extract_eq_self_of_le (as := coins) (i := 0) (j := coins.size) (by simp))
      simpa [sumcheckResidualTable, hExtract]
  | succ i ih =>
      intro coins hi hCoins
      have hiLt : i < inst.rounds := by omega
      have hStop : coins.size ≤ coins.size := le_rfl
      have hTailNe : (coins.extract i coins.size).size ≠ 0 := by
        rw [Array.size_extract_of_le hStop]
        omega
      have hStep :=
        mleByFolding_step
          (v := sumcheckResidualTable table i coins)
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
          mleByFolding (sumcheckResidualTable table i coins) (coins.extract i coins.size) =
            mleByFolding
              (foldLayer (sumcheckResidualTable table i coins) (coins[i]!))
              (coins.extract (i + 1) coins.size) := by
        rw [hStep, hHead, hTail]
      calc
        mleByFolding (sumcheckResidualTable table (i + 1) coins) (coins.extract (i + 1) coins.size)
            = mleByFolding
                (foldLayer (sumcheckResidualTable table i coins) (coins[i]!))
                (coins.extract (i + 1) coins.size) := by
                  simp [sumcheckResidualTable]
        _ = mleByFolding (sumcheckResidualTable table i coins) (coins.extract i coins.size) := by
              simpa using hStep'.symm
        _ = mleByFolding table coins := by
              exact ih coins (Nat.le_of_lt hiLt) hCoins

private def sumcheckHonestRoundPolyForCoins
  (inst : SumCheckInstance)
  (table : Array F)
  (i : Nat)
  (coins : Array F) : Array F :=
  let vals := sumcheckResidualTable table i coins
  let v0 := arraySum (Array.ofFn (fun j : Fin (vals.size / 2) => vals[2 * j.1]!))
  let v1 := arraySum (Array.ofFn (fun j : Fin (vals.size / 2) => vals[2 * j.1 + 1]!))
  Array.ofFn (fun k : Fin (inst.maxDegree + 1) =>
    if h0 : k.1 = 0 then
      v0
    else if h1 : k.1 = 1 then
      v1 - v0
    else
      0)

private theorem sumcheckHonestRoundPolyForCoins_size
  (inst : SumCheckInstance)
  (table : Array F)
  (i : Nat)
  (coins : Array F) :
  (sumcheckHonestRoundPolyForCoins inst table i coins).size = inst.maxDegree + 1 := by
  simp [sumcheckHonestRoundPolyForCoins]

private theorem sumcheckHonestRoundPolyForCoins_eval_zero
  (inst : SumCheckInstance)
  (table : Array F)
  (i : Nat)
  (coins : Array F) :
  sumcheckEvalPoly (sumcheckHonestRoundPolyForCoins inst table i coins) 0 =
    sumcheckTableSum (foldLayer (sumcheckResidualTable table i coins) 0) := by
  cases hDeg : inst.maxDegree with
  | zero =>
      unfold sumcheckHonestRoundPolyForCoins
      rw [hDeg]
      rw [sumcheckEvalPoly_ofFn_succ, sumcheckTableSum_eq_arraySum, arraySum_foldLayer]
      simp
  | succ n =>
      unfold sumcheckHonestRoundPolyForCoins
      rw [sumcheckTableSum_eq_arraySum, arraySum_foldLayer, sumcheckEvalPoly_ofFn_succ]
      simp [sumcheckEvalPoly_zero_ofFn, hDeg]

private theorem sumcheckHonestRoundPolyForCoins_eval
  (inst : SumCheckInstance)
  (table : Array F)
  (i : Nat)
  (coins : Array F)
  (r : F)
  (hDegPos : 0 < inst.maxDegree) :
  sumcheckEvalPoly (sumcheckHonestRoundPolyForCoins inst table i coins) r =
    sumcheckTableSum (foldLayer (sumcheckResidualTable table i coins) r) := by
  rcases Nat.exists_eq_succ_of_ne_zero (Nat.ne_of_gt hDegPos) with ⟨n, hDeg⟩
  let vals := sumcheckResidualTable table i coins
  let v0 := arraySum (Array.ofFn (fun j : Fin (vals.size / 2) => vals[2 * j.1]!))
  let v1 := arraySum (Array.ofFn (fun j : Fin (vals.size / 2) => vals[2 * j.1 + 1]!))
  have hFold :
      sumcheckTableSum (foldLayer vals r) = v0 * ((1 : F) - r) + v1 * r := by
    simpa [sumcheckTableSum_eq_arraySum, v0, v1] using arraySum_foldLayer vals r
  unfold sumcheckHonestRoundPolyForCoins
  rw [hDeg]
  calc
    sumcheckEvalPoly
        (Array.ofFn
          (fun k : Fin (n + 2) =>
            if h0 : k.1 = 0 then
              v0
            else if h1 : k.1 = 1 then
              v1 - v0
            else
              0))
        r = v0 * ((1 : F) - r) + v1 * r := by
          simpa [linear_interp_eval] using sumcheckEvalPoly_linear_interp_ofFn n v0 v1 r
    _ = sumcheckTableSum (foldLayer vals r) := hFold.symm
    _ = sumcheckTableSum (foldLayer (sumcheckResidualTable table i coins) r) := by
          simp [vals]

private def sumcheckHonestTranscriptForStatement
  {inst : SumCheckInstance}
  (stmt : SumCheckStatement inst)
  (challenges : Array F)
  (_hSize : challenges.size = inst.rounds) : SumCheckTranscript :=
  { challenges := challenges
    roundPolys := Array.ofFn (fun i : Fin inst.rounds =>
      sumcheckHonestRoundPolyForCoins inst stmt.table i.1 challenges) }

private theorem sumcheckHonestTranscriptForStatement_roundPoly_get!
  {inst : SumCheckInstance}
  (stmt : SumCheckStatement inst)
  {challenges : Array F}
  {hSize}
  {i : Nat}
  (hi : i < inst.rounds) :
  (sumcheckHonestTranscriptForStatement stmt challenges hSize).roundPolys[i]! =
    sumcheckHonestRoundPolyForCoins inst stmt.table i challenges := by
  simp [sumcheckHonestTranscriptForStatement, hi]

private theorem sumcheckHonestTranscriptForStatement_accepted
  {inst : SumCheckInstance}
  (stmt : SumCheckStatement inst)
  (challenges : Array F)
  (hSize : challenges.size = inst.rounds) :
  sumcheckAccepted inst (sumcheckHonestTranscriptForStatement stmt challenges hSize) := by
  refine ⟨stmt.parameterConsistent, stmt.degreeCompatible, ?_, ?_, ?_, ?_⟩
  · exact ⟨hSize, by simp [sumcheckHonestTranscriptForStatement]⟩
  · intro i
    have hi : i.1 < inst.rounds := by
      simpa [sumcheckHonestTranscriptForStatement] using i.2
    simpa [sumcheckRoundPolyShape, sumcheckHonestTranscriptForStatement, hi] using
      sumcheckHonestRoundPolyForCoins_size inst stmt.table i.1 challenges
  · unfold sumcheckInitialRoundConsistent
    by_cases hRounds : inst.rounds = 0
    · have hZeroSize :
        (sumcheckHonestTranscriptForStatement stmt challenges hSize).roundPolys.size = 0 := by
          simp [sumcheckHonestTranscriptForStatement, hRounds]
      simp [hZeroSize]
    · have hNonzeroSize :
        (sumcheckHonestTranscriptForStatement stmt challenges hSize).roundPolys.size ≠ 0 := by
          simp [sumcheckHonestTranscriptForStatement, hRounds]
      have hRoundsPos : 0 < inst.rounds := Nat.pos_of_ne_zero hRounds
      have hDegPos :=
        sumcheckDegreeCompatible.maxDegree_pos_of_rounds_pos stmt.degreeCompatible hRoundsPos
      have hPoly0 :
          (sumcheckHonestTranscriptForStatement stmt challenges hSize).roundPolys[0]! =
            sumcheckHonestRoundPolyForCoins inst stmt.table 0 challenges :=
        sumcheckHonestTranscriptForStatement_roundPoly_get! stmt (hSize := hSize) hRoundsPos
      have hEven : stmt.table.size = 2 * (stmt.table.size / 2) := by
        rw [stmt.tableSize]
        rcases Nat.exists_eq_succ_of_ne_zero hRounds with ⟨n, hn⟩
        rw [hn]
        simp [Nat.pow_succ, Nat.mul_comm, Nat.mul_left_comm, Nat.mul_assoc]
      have hClaim :
          sumcheckEvalPoly (sumcheckHonestTranscriptForStatement stmt challenges hSize).roundPolys[0]! 0 +
              sumcheckEvalPoly (sumcheckHonestTranscriptForStatement stmt challenges hSize).roundPolys[0]! 1 =
            inst.claimedValue := by
        calc
          sumcheckEvalPoly (sumcheckHonestTranscriptForStatement stmt challenges hSize).roundPolys[0]! 0 +
              sumcheckEvalPoly (sumcheckHonestTranscriptForStatement stmt challenges hSize).roundPolys[0]! 1
              = sumcheckTableSum (foldLayer stmt.table 0) + sumcheckTableSum (foldLayer stmt.table 1) := by
                  rw [hPoly0,
                    sumcheckHonestRoundPolyForCoins_eval_zero,
                    sumcheckHonestRoundPolyForCoins_eval _ _ _ _ _ hDegPos]
                  simpa [sumcheckResidualTable]
          _ = sumcheckTableSum stmt.table := by
                simpa [sumcheckResidualTable] using sumcheckTableSum_foldLayer_split stmt.table hEven
          _ = inst.claimedValue := stmt.hypercubeSumEqClaimed
      simpa [sumcheckInitialRoundConsistent, hNonzeroSize] using hClaim
  · refine ⟨by simp [sumcheckHonestTranscriptForStatement, hSize], ?_⟩
    intro i hi
    have hiNext : i + 1 < inst.rounds := by
      simpa [sumcheckHonestTranscriptForStatement] using hi
    have hiCur : i < inst.rounds := Nat.lt_trans (Nat.lt_succ_self i) hiNext
    have hRoundsPos : 0 < inst.rounds := Nat.lt_trans (Nat.zero_lt_succ _) hiNext
    have hDegPos :=
      sumcheckDegreeCompatible.maxDegree_pos_of_rounds_pos stmt.degreeCompatible hRoundsPos
    have hPolyCur :
        (sumcheckHonestTranscriptForStatement stmt challenges hSize).roundPolys[i]! =
          sumcheckHonestRoundPolyForCoins inst stmt.table i challenges :=
      sumcheckHonestTranscriptForStatement_roundPoly_get! stmt (hSize := hSize) hiCur
    have hPolyNext :
        (sumcheckHonestTranscriptForStatement stmt challenges hSize).roundPolys[i + 1]! =
          sumcheckHonestRoundPolyForCoins inst stmt.table (i + 1) challenges :=
      sumcheckHonestTranscriptForStatement_roundPoly_get! stmt (hSize := hSize) hiNext
    have hEven :
        (sumcheckResidualTable stmt.table (i + 1) challenges).size =
          2 * ((sumcheckResidualTable stmt.table (i + 1) challenges).size / 2) := by
      have hResSize :
          (sumcheckResidualTable stmt.table (i + 1) challenges).size =
            2 ^ (inst.rounds - (i + 1)) := by
        exact sumcheckResidualTable_size (inst := inst) stmt.table stmt.tableSize (i + 1) challenges hiNext.le
      rw [hResSize]
      have hDiffPos : 0 < inst.rounds - (i + 1) := by omega
      rcases Nat.exists_eq_succ_of_ne_zero (Nat.ne_of_gt hDiffPos) with ⟨n, hn⟩
      rw [hn]
      simp [Nat.pow_succ, Nat.mul_comm]
    calc
      sumcheckEvalPoly (sumcheckHonestTranscriptForStatement stmt challenges hSize).roundPolys[i + 1]! 0 +
          sumcheckEvalPoly (sumcheckHonestTranscriptForStatement stmt challenges hSize).roundPolys[i + 1]! 1
          = sumcheckTableSum (foldLayer (sumcheckResidualTable stmt.table (i + 1) challenges) 0) +
              sumcheckTableSum (foldLayer (sumcheckResidualTable stmt.table (i + 1) challenges) 1) := by
                rw [hPolyNext,
                  sumcheckHonestRoundPolyForCoins_eval_zero,
                  sumcheckHonestRoundPolyForCoins_eval _ _ _ _ _ hDegPos]
      _ = sumcheckTableSum (sumcheckResidualTable stmt.table (i + 1) challenges) := by
            simpa using
              sumcheckTableSum_foldLayer_split (sumcheckResidualTable stmt.table (i + 1) challenges) hEven
      _ = sumcheckEvalPoly (sumcheckHonestTranscriptForStatement stmt challenges hSize).roundPolys[i]!
            (sumcheckHonestTranscriptForStatement stmt challenges hSize).challenges[i]! := by
            rw [hPolyCur]
            simp [sumcheckHonestTranscriptForStatement, hiCur]
            exact (sumcheckHonestRoundPolyForCoins_eval inst stmt.table i challenges
              (challenges[i]!) hDegPos).symm

private theorem sumcheckHonestTranscriptForStatement_finalOracle
  {inst : SumCheckInstance}
  (stmt : SumCheckStatement inst)
  (challenges : Array F)
  (hSize : challenges.size = inst.rounds) :
  sumcheckFinalOracleConsistent inst stmt
    (sumcheckHonestTranscriptForStatement stmt challenges hSize) := by
  have hAcc :=
    sumcheckHonestTranscriptForStatement_accepted stmt challenges hSize
  refine ⟨hAcc.2.2.1, ?_⟩
  by_cases hZero : inst.rounds = 0
  · simp [sumcheckFinalOracleConsistent, hZero]
    have hTableSize : stmt.table.size = 1 := by simpa [hZero] using stmt.tableSize
    have hTableNe : stmt.table.size ≠ 0 := by simpa [hTableSize]
    calc
      mleByFolding stmt.table #[] = stmt.table[0]! := by
        exact mleByFolding_empty stmt.table hTableNe
      _ = sumcheckTableSum stmt.table := by
            symm
            exact sumcheckTableSum_size_one stmt.table hTableSize
      _ = inst.claimedValue := stmt.hypercubeSumEqClaimed
  · have hRoundsPos : 0 < inst.rounds := Nat.pos_of_ne_zero hZero
    have hDegPos :=
      sumcheckDegreeCompatible.maxDegree_pos_of_rounds_pos stmt.degreeCompatible hRoundsPos
    have hLastLt : inst.rounds - 1 < inst.rounds := by omega
    have hLastCoins : inst.rounds - 1 < challenges.size := by
      simpa [hSize] using hLastLt
    have hRoundPoly :
        (sumcheckHonestTranscriptForStatement stmt challenges hSize).roundPolys[inst.rounds - 1]! =
          sumcheckHonestRoundPolyForCoins inst stmt.table (inst.rounds - 1) challenges :=
      sumcheckHonestTranscriptForStatement_roundPoly_get! stmt (hSize := hSize) hLastLt
    have hTailEval :
        mleByFolding (sumcheckResidualTable stmt.table (inst.rounds - 1) challenges)
          (challenges.extract (inst.rounds - 1) challenges.size) =
        sumcheckEvalPoly (sumcheckHonestRoundPolyForCoins inst stmt.table (inst.rounds - 1) challenges)
          (challenges[inst.rounds - 1]!) := by
      have hTailSize :
          (challenges.extract (inst.rounds - 1) challenges.size).size = 1 := by
        rw [Array.size_extract_of_le le_rfl]
        omega
      have hTailHead :
          (challenges.extract (inst.rounds - 1) challenges.size)[0]! = challenges[inst.rounds - 1]! :=
        extract_head_eq challenges hLastCoins le_rfl
      have hTailNe :
          (challenges.extract (inst.rounds - 1) challenges.size).size ≠ 0 := by
        simp [hTailSize]
      have hStep :=
        mleByFolding_step
          (v := sumcheckResidualTable stmt.table (inst.rounds - 1) challenges)
          (r := challenges.extract (inst.rounds - 1) challenges.size)
          hTailNe
      have hAfter :
          (challenges.extract (inst.rounds - 1) challenges.size).extract 1
              (challenges.extract (inst.rounds - 1) challenges.size).size = #[] := by
        simp [hTailSize]
      have hFoldedSize :
          (foldLayer (sumcheckResidualTable stmt.table (inst.rounds - 1) challenges)
            (challenges[inst.rounds - 1]!)).size = 1 := by
        have hResSize :
            (sumcheckResidualTable stmt.table (inst.rounds - 1) challenges).size = 2 := by
          have hBase := sumcheckResidualTable_size (inst := inst) stmt.table stmt.tableSize
            (inst.rounds - 1) challenges (by omega)
          have hExp : inst.rounds - (inst.rounds - 1) = 1 := by omega
          simpa [hExp, Nat.pow_one] using hBase
        simpa [hResSize, foldLayer_size]
      have hFoldedNe :
          (foldLayer (sumcheckResidualTable stmt.table (inst.rounds - 1) challenges)
            (challenges[inst.rounds - 1]!)).size ≠ 0 := by
        simp [hFoldedSize]
      calc
        mleByFolding (sumcheckResidualTable stmt.table (inst.rounds - 1) challenges)
            (challenges.extract (inst.rounds - 1) challenges.size)
            = mleByFolding
                (foldLayer (sumcheckResidualTable stmt.table (inst.rounds - 1) challenges)
                  (challenges[inst.rounds - 1]!))
                #[] := by
                  rw [hStep, hTailHead, hAfter]
        _ = (foldLayer (sumcheckResidualTable stmt.table (inst.rounds - 1) challenges)
              (challenges[inst.rounds - 1]!))[0]! := by
              exact mleByFolding_empty _ hFoldedNe
        _ = sumcheckTableSum
              (foldLayer (sumcheckResidualTable stmt.table (inst.rounds - 1) challenges)
                (challenges[inst.rounds - 1]!)) := by
              have hOne : (foldLayer (sumcheckResidualTable stmt.table (inst.rounds - 1) challenges)
                (challenges[inst.rounds - 1]!)).size = 1 := hFoldedSize
              symm
              exact sumcheckTableSum_size_one _ hOne
        _ = sumcheckEvalPoly
              (sumcheckHonestRoundPolyForCoins inst stmt.table (inst.rounds - 1) challenges)
              (challenges[inst.rounds - 1]!) := by
              symm
              exact sumcheckHonestRoundPolyForCoins_eval inst stmt.table (inst.rounds - 1)
                challenges (challenges[inst.rounds - 1]!) hDegPos
    have hTail :
        mleByFolding (sumcheckResidualTable stmt.table (inst.rounds - 1) challenges)
          (challenges.extract (inst.rounds - 1) challenges.size) =
        mleByFolding stmt.table challenges := by
      exact sumcheckResidualTable_mleByFolding_tail (inst := inst) stmt.table stmt.tableSize
        (inst.rounds - 1) challenges (by omega) hSize
    simp [sumcheckFinalOracleConsistent, hZero, hRoundPoly]
    exact hTailEval.symm.trans hTail

/-- Canonical Definition-6 theorem object from a concrete table statement. -/
def sumcheckDefinition6Statement_of_statement
  {inst : SumCheckInstance}
  (stmt : SumCheckStatement inst) :
  SumCheckDefinition6Statement inst where
  statement := stmt
  honestTranscript := fun challenges hSize =>
    sumcheckHonestTranscriptForStatement stmt challenges hSize
  transcriptChallenges := by
    intro challenges hSize
    rfl
  accepted := by
    intro challenges hSize
    exact sumcheckHonestTranscriptForStatement_accepted stmt challenges hSize
  finalOracle := by
    intro challenges hSize
    exact sumcheckHonestTranscriptForStatement_finalOracle stmt challenges hSize

/-- Build a Definition-6 paper witness from legacy base-claim assumptions. -/
theorem sumcheckPaperClaimTrue_of_baseClaim_constructive
  {inst : SumCheckInstance}
  (hClaim : sumcheckBaseClaimTrue inst) :
  sumcheckPaperClaimTrue inst := by
  exact ⟨sumcheckDefinition6Statement_of_statement (sumcheckWitnessStatement inst hClaim)⟩

/-- Build a Definition-6 paper witness from accepted transcript evidence. -/
theorem sumcheckPaperClaimTrue_of_accepted
  {inst : SumCheckInstance}
  {tr : SumCheckTranscript}
  (hAcc : sumcheckAccepted inst tr) :
  sumcheckPaperClaimTrue inst := by
  exact sumcheckPaperClaimTrue_of_baseClaim_constructive
    ⟨sumcheckAccepted_parameter_consistent hAcc, sumcheckAccepted_degree_compatible hAcc⟩

/--
Any statement/transcript consistency witness yields claim truth for the
Definition-6 theorem surface.
-/
theorem sumcheckPaperClaimTrue_of_statementTranscriptConsistent
  {inst : SumCheckInstance}
  {stmt : SumCheckStatement inst}
  {tr : SumCheckTranscript}
  (hConsistent : sumcheckStatementTranscriptConsistent inst stmt tr) :
  sumcheckPaperClaimTrue inst := by
  exact sumcheckPaperClaimTrue_of_baseClaim_constructive
    ⟨sumcheckAccepted_parameter_consistent hConsistent.1,
      sumcheckAccepted_degree_compatible hConsistent.1⟩

/-- Constructive completeness closure for the Definition-6 paper semantics. -/
theorem sumcheckCompleteness_constructive : SumcheckCompletenessAssumption := by
  intro inst hClaim
  rcases hClaim with ⟨stmt⟩
  let challenges := Array.replicate inst.rounds (0 : F)
  have hSize : challenges.size = inst.rounds := by simp [challenges]
  exact ⟨stmt.honestTranscript challenges hSize, stmt.accepted challenges hSize⟩

/-- Structural completeness theorem for the standalone Definition-6 surface. -/
abbrev sumcheckStructuralCompleteness_constructive : SumcheckCompletenessAssumption :=
  sumcheckCompleteness_constructive

/--
Constructive structural closure for the standalone Definition-6 SumCheck
surface. This is not the probabilistic Lund soundness theorem.
-/
theorem sumcheckSoundness_constructive : SumcheckSoundnessAssumption := by
  intro inst tr hAcc
  exact sumcheckPaperClaimTrue_of_accepted hAcc

/-- Structural soundness theorem for the standalone Definition-6 surface. -/
abbrev sumcheckStructuralSoundness_constructive : SumcheckSoundnessAssumption :=
  sumcheckSoundness_constructive

/-- Constructive Definition-6 SumCheck assumption bundle. -/
def sumcheckAssumptions_constructive : SumCheckAssumptions :=
  { soundness := sumcheckSoundness_constructive
    completeness := sumcheckCompleteness_constructive }

theorem sumcheckPaperClaimTrue_iff_baseClaim
  {inst : SumCheckInstance} :
  sumcheckPaperClaimTrue inst ↔ sumcheckBaseClaimTrue inst := by
  constructor
  · intro h
    rcases h with ⟨stmt⟩
    exact ⟨stmt.statement.parameterConsistent, stmt.statement.degreeCompatible⟩
  · intro h
    exact sumcheckPaperClaimTrue_of_baseClaim_constructive h

theorem sumcheckClaimTrue_iff_baseClaim
  {inst : SumCheckInstance} :
  sumcheckClaimTrue inst ↔ sumcheckBaseClaimTrue inst :=
  sumcheckPaperClaimTrue_iff_baseClaim

end SuperNeo
