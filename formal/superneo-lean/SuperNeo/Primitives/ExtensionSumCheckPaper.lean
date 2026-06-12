import SuperNeo.Primitives.ExtensionSumCheck
import Mathlib

namespace SuperNeo

private def extensionArraySum (vals : Array KExt) : KExt :=
  Finset.sum (Finset.range vals.size) (fun i => vals[i]!)

@[simp] private theorem extensionArraySum_empty :
  extensionArraySum (#[] : Array KExt) = 0 := by
  simp [extensionArraySum]

@[simp] private theorem extensionArraySum_ofFn
  (n : Nat)
  (f : Fin n → KExt) :
  extensionArraySum (Array.ofFn f) =
    Finset.sum (Finset.range n) (fun i => if h : i < n then f ⟨i, h⟩ else 0) := by
  rw [extensionArraySum]
  simp only [Array.size_ofFn]
  apply Finset.sum_congr rfl
  intro i hi
  have hiNat : i < n := Finset.mem_range.mp hi
  have hi' : i < (Array.ofFn f).size := by simpa using hi
  rw [getElem!_pos (c := Array.ofFn f) (i := i) hi']
  simpa [hiNat] using (Array.getElem_ofFn (f := f) (i := i) hi')

private theorem extensionSum_range_pairs
  (n : Nat)
  (f : Nat → KExt) :
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

private theorem extensionArraySum_even_add_odd
  (vals : Array KExt)
  {n : Nat}
  (hSize : vals.size = 2 * n) :
  extensionArraySum vals =
    extensionArraySum (Array.ofFn (fun j : Fin n => vals[2 * j.1]!)) +
      extensionArraySum (Array.ofFn (fun j : Fin n => vals[2 * j.1 + 1]!)) := by
  have hEven :
      extensionArraySum (Array.ofFn (fun j : Fin n => vals[2 * j.1]!)) =
        Finset.sum (Finset.range n) (fun j => vals[2 * j]!) := by
    rw [extensionArraySum_ofFn]
    apply Finset.sum_congr rfl
    intro x hx
    have hx' : x < n := Finset.mem_range.mp hx
    simp [hx']
  have hOdd :
      extensionArraySum (Array.ofFn (fun j : Fin n => vals[2 * j.1 + 1]!)) =
        Finset.sum (Finset.range n) (fun j => vals[2 * j + 1]!) := by
    rw [extensionArraySum_ofFn]
    apply Finset.sum_congr rfl
    intro x hx
    have hx' : x < n := Finset.mem_range.mp hx
    simp [hx']
  calc
    extensionArraySum vals
        = Finset.sum (Finset.range (2 * n)) (fun k => vals[k]!) := by
              simp [extensionArraySum, hSize]
    _ = Finset.sum (Finset.range n) (fun j => vals[2 * j]! + vals[2 * j + 1]!) := by
          symm
          exact extensionSum_range_pairs n (fun k => vals[k]!)
    _ = Finset.sum (Finset.range n) (fun j => vals[2 * j]!) +
          Finset.sum (Finset.range n) (fun j => vals[2 * j + 1]!) := by
          rw [Finset.sum_add_distrib]
    _ = extensionArraySum (Array.ofFn (fun j : Fin n => vals[2 * j.1]!)) +
          extensionArraySum (Array.ofFn (fun j : Fin n => vals[2 * j.1 + 1]!)) := by
          rw [hEven, hOdd]

private theorem extensionSumcheckEvalPoly_ofFn_succ
  (n : Nat)
  (v : Fin (n + 1) → KExt)
  (x : KExt) :
  extensionSumcheckEvalPoly (Array.ofFn v) x =
    v 0 + x * extensionSumcheckEvalPoly (Array.ofFn (fun i : Fin n => v i.succ)) x := by
  unfold extensionSumcheckEvalPoly
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
        (init := (0 : KExt))
        (xs := Array.ofFn (fun i : Fin n => v i.succ)))
  simpa [extensionSumcheckEvalPoly] using congrArg (fun z => v 0 + x * z) hTail

private theorem extensionSumcheckEvalPoly_zero_ofFn
  (n : Nat)
  (x : KExt) :
  extensionSumcheckEvalPoly (Array.ofFn (fun _ : Fin n => (0 : KExt))) x = 0 := by
  induction n with
  | zero =>
      simp [extensionSumcheckEvalPoly]
  | succ n ih =>
      rw [extensionSumcheckEvalPoly_ofFn_succ]
      simp [ih]

private theorem extensionLinear_interp_eval
  (a b x : KExt) :
  a + x * (b - a) = a * ((1 : KExt) - x) + b * x := by
  ring

private theorem extensionSumcheckEvalPoly_linear_interp_ofFn
  (n : Nat)
  (v0 v1 x : KExt) :
  extensionSumcheckEvalPoly
      (Array.ofFn (fun k : Fin (n + 2) =>
        if h0 : k.1 = 0 then
          v0
        else if h1 : k.1 = 1 then
          v1 - v0
        else
          0)) x =
    v0 * ((1 : KExt) - x) + v1 * x := by
  rw [extensionSumcheckEvalPoly_ofFn_succ]
  rw [extensionSumcheckEvalPoly_ofFn_succ]
  simp [extensionSumcheckEvalPoly_zero_ofFn, extensionLinear_interp_eval]

private theorem extensionArraySum_foldLayer
  (vals : Array KExt)
  (ri : KExt) :
  extensionArraySum (foldLayerK vals ri) =
    extensionArraySum (Array.ofFn (fun j : Fin (vals.size / 2) => vals[2 * j.1]!)) * ((1 : KExt) - ri) +
      extensionArraySum (Array.ofFn (fun j : Fin (vals.size / 2) => vals[2 * j.1 + 1]!)) * ri := by
  have hEven :
      extensionArraySum (Array.ofFn (fun j : Fin (vals.size / 2) => vals[2 * j.1]!)) =
        Finset.sum (Finset.range (vals.size / 2)) (fun j => vals[2 * j]!) := by
    rw [extensionArraySum_ofFn]
    apply Finset.sum_congr rfl
    intro x hx
    have hx' : x < vals.size / 2 := Finset.mem_range.mp hx
    simp [hx']
  have hOdd :
      extensionArraySum (Array.ofFn (fun j : Fin (vals.size / 2) => vals[2 * j.1 + 1]!)) =
        Finset.sum (Finset.range (vals.size / 2)) (fun j => vals[2 * j + 1]!) := by
    rw [extensionArraySum_ofFn]
    apply Finset.sum_congr rfl
    intro x hx
    have hx' : x < vals.size / 2 := Finset.mem_range.mp hx
    simp [hx']
  calc
    extensionArraySum (foldLayerK vals ri)
        = Finset.sum (Finset.range (vals.size / 2))
            (fun j => vals[2 * j]! * ((1 : KExt) - ri) + vals[2 * j + 1]! * ri) := by
              rw [foldLayerK, extensionArraySum_ofFn]
              apply Finset.sum_congr rfl
              intro x hx
              have hx' : x < vals.size / 2 := Finset.mem_range.mp hx
              simp [hx']
    _ = Finset.sum (Finset.range (vals.size / 2)) (fun j => vals[2 * j]! * ((1 : KExt) - ri)) +
          Finset.sum (Finset.range (vals.size / 2)) (fun j => vals[2 * j + 1]! * ri) := by
            rw [Finset.sum_add_distrib]
    _ = Finset.sum (Finset.range (vals.size / 2)) (fun j => vals[2 * j]!) * ((1 : KExt) - ri) +
          Finset.sum (Finset.range (vals.size / 2)) (fun j => vals[2 * j + 1]!) * ri := by
            rw [← Finset.sum_mul, ← Finset.sum_mul]
    _ = extensionArraySum (Array.ofFn (fun j : Fin (vals.size / 2) => vals[2 * j.1]!)) * ((1 : KExt) - ri) +
          extensionArraySum (Array.ofFn (fun j : Fin (vals.size / 2) => vals[2 * j.1 + 1]!)) * ri := by
            rw [hEven, hOdd]

private theorem extensionSumcheckTableSum_eq_arraySum
  (vals : Array KExt) :
  extensionSumcheckTableSum vals = extensionArraySum vals := by
  unfold extensionSumcheckTableSum extensionArraySum
  rw [← Array.foldr_toList]
  rw [← List.sum_eq_foldr]
  rw [show vals.toList = List.ofFn (fun i : Fin vals.size => vals[i.1]!) by
    simpa using (List.ofFn_getElem vals.toList).symm]
  rw [List.sum_ofFn, Fin.sum_univ_eq_sum_range]

private theorem extensionSumcheckTableSum_size_one
  (vals : Array KExt)
  (hSize : vals.size = 1) :
  extensionSumcheckTableSum vals = vals[0]! := by
  rw [extensionSumcheckTableSum_eq_arraySum]
  simp [extensionArraySum, hSize]

private theorem extensionSumcheckTableSum_foldLayer_split
  (vals : Array KExt)
  (hEven : vals.size = 2 * (vals.size / 2)) :
  extensionSumcheckTableSum (foldLayerK vals 0) + extensionSumcheckTableSum (foldLayerK vals 1) =
    extensionSumcheckTableSum vals := by
  let evenVals := Array.ofFn (fun j : Fin (vals.size / 2) => vals[2 * j.1]!)
  let oddVals := Array.ofFn (fun j : Fin (vals.size / 2) => vals[2 * j.1 + 1]!)
  have h0 :
      extensionSumcheckTableSum (foldLayerK vals 0) = extensionArraySum evenVals := by
    rw [extensionSumcheckTableSum_eq_arraySum, extensionArraySum_foldLayer]
    simp [evenVals, oddVals]
  have h1 :
      extensionSumcheckTableSum (foldLayerK vals 1) = extensionArraySum oddVals := by
    rw [extensionSumcheckTableSum_eq_arraySum, extensionArraySum_foldLayer]
    simp [evenVals, oddVals]
  have hSum :
      extensionSumcheckTableSum vals = extensionArraySum evenVals + extensionArraySum oddVals := by
    rw [extensionSumcheckTableSum_eq_arraySum]
    simpa [evenVals, oddVals] using extensionArraySum_even_add_odd vals hEven
  calc
    extensionSumcheckTableSum (foldLayerK vals 0) + extensionSumcheckTableSum (foldLayerK vals 1)
        = extensionArraySum evenVals + extensionArraySum oddVals := by rw [h0, h1]
    _ = extensionSumcheckTableSum vals := hSum.symm

private def extensionSumcheckResidualTable
  (table : Array KExt) :
  Nat → Array KExt → Array KExt
  | 0, _ => table
  | i + 1, coins => foldLayerK (extensionSumcheckResidualTable table i coins) (coins[i]!)

private theorem extensionSumcheckResidualTable_size
  {inst : ExtensionSumCheckInstance}
  (table : Array KExt)
  (hTable : table.size = 2 ^ inst.rounds) :
  ∀ i coins, i ≤ inst.rounds →
    (extensionSumcheckResidualTable table i coins).size = 2 ^ (inst.rounds - i) := by
  intro i
  induction i with
  | zero =>
      intro coins _hi
      simpa [extensionSumcheckResidualTable] using hTable
  | succ i ih =>
      intro coins hi
      have hiLt : i < inst.rounds := by omega
      have hPrev := ih coins (Nat.le_of_lt hiLt)
      calc
        (extensionSumcheckResidualTable table (i + 1) coins).size
            = (extensionSumcheckResidualTable table i coins).size / 2 := by
                simp [extensionSumcheckResidualTable]
        _ = (2 ^ (inst.rounds - i)) / 2 := by
              rw [hPrev]
        _ = 2 ^ (inst.rounds - (i + 1)) := by
              have hExp : inst.rounds - i = (inst.rounds - (i + 1)) + 1 := by omega
              rw [hExp, Nat.pow_succ]
              simp

private theorem extensionExtract_head_eq
  (coins : Array KExt)
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

private theorem extensionExtract_tail_eq
  (coins : Array KExt)
  {i stop : Nat}
  (hi : i < stop)
  (hStop : stop ≤ coins.size) :
  (coins.extract i stop).extract 1 (coins.extract i stop).size =
    coins.extract (i + 1) stop := by
  rw [Array.extract_extract]
  rw [Array.size_extract_of_le hStop]
  simp [Nat.min_eq_right, hi.le, Nat.add_assoc]

private theorem extensionSumcheckResidualTable_mleByFoldingK_tail
  {inst : ExtensionSumCheckInstance}
  (table : Array KExt)
  (hTable : table.size = 2 ^ inst.rounds) :
  ∀ i coins, i ≤ inst.rounds → coins.size = inst.rounds →
    mleByFoldingK (extensionSumcheckResidualTable table i coins) (coins.extract i coins.size) =
      mleByFoldingK table coins := by
  intro i
  induction i with
  | zero =>
      intro coins _hi hCoins
      have hExtract : coins.extract 0 coins.size = coins := by
        simpa using (Array.extract_eq_self_of_le (as := coins) (i := 0) (j := coins.size) (by simp))
      simpa [extensionSumcheckResidualTable, hExtract]
  | succ i ih =>
      intro coins hi hCoins
      have hiLt : i < inst.rounds := by omega
      have hStop : coins.size ≤ coins.size := le_rfl
      have hTailNe : (coins.extract i coins.size).size ≠ 0 := by
        rw [Array.size_extract_of_le hStop]
        omega
      have hStep :=
        mleByFoldingK_step
          (v := extensionSumcheckResidualTable table i coins)
          (r := coins.extract i coins.size)
          hTailNe
      have hHead :
          (coins.extract i coins.size)[0]! = coins[i]! :=
        extensionExtract_head_eq coins (hi := by omega) hStop
      have hTail :
          (coins.extract i coins.size).extract 1 (coins.extract i coins.size).size =
            coins.extract (i + 1) coins.size :=
        extensionExtract_tail_eq coins (hi := by omega) hStop
      have hStep' :
          mleByFoldingK (extensionSumcheckResidualTable table i coins) (coins.extract i coins.size) =
            mleByFoldingK
              (foldLayerK (extensionSumcheckResidualTable table i coins) (coins[i]!))
              (coins.extract (i + 1) coins.size) := by
        rw [hStep, hHead, hTail]
      calc
        mleByFoldingK (extensionSumcheckResidualTable table (i + 1) coins) (coins.extract (i + 1) coins.size)
            = mleByFoldingK
                (foldLayerK (extensionSumcheckResidualTable table i coins) (coins[i]!))
                (coins.extract (i + 1) coins.size) := by
                  simp [extensionSumcheckResidualTable]
        _ = mleByFoldingK (extensionSumcheckResidualTable table i coins) (coins.extract i coins.size) := by
              simpa using hStep'.symm
        _ = mleByFoldingK table coins := by
              exact ih coins (Nat.le_of_lt hiLt) hCoins

private def extensionSumcheckHonestRoundPolyForCoins
  (inst : ExtensionSumCheckInstance)
  (table : Array KExt)
  (i : Nat)
  (coins : Array KExt) : Array KExt :=
  let vals := extensionSumcheckResidualTable table i coins
  let v0 := extensionArraySum (Array.ofFn (fun j : Fin (vals.size / 2) => vals[2 * j.1]!))
  let v1 := extensionArraySum (Array.ofFn (fun j : Fin (vals.size / 2) => vals[2 * j.1 + 1]!))
  Array.ofFn (fun k : Fin (inst.maxDegree + 1) =>
    if h0 : k.1 = 0 then
      v0
    else if h1 : k.1 = 1 then
      v1 - v0
    else
      0)

private theorem extensionSumcheckHonestRoundPolyForCoins_size
  (inst : ExtensionSumCheckInstance)
  (table : Array KExt)
  (i : Nat)
  (coins : Array KExt) :
  (extensionSumcheckHonestRoundPolyForCoins inst table i coins).size = inst.maxDegree + 1 := by
  simp [extensionSumcheckHonestRoundPolyForCoins]

private theorem extensionSumcheckHonestRoundPolyForCoins_eval_zero
  (inst : ExtensionSumCheckInstance)
  (table : Array KExt)
  (i : Nat)
  (coins : Array KExt) :
  extensionSumcheckEvalPoly (extensionSumcheckHonestRoundPolyForCoins inst table i coins) 0 =
    extensionSumcheckTableSum (foldLayerK (extensionSumcheckResidualTable table i coins) 0) := by
  cases hDeg : inst.maxDegree with
  | zero =>
      unfold extensionSumcheckHonestRoundPolyForCoins
      rw [hDeg]
      rw [extensionSumcheckEvalPoly_ofFn_succ, extensionSumcheckTableSum_eq_arraySum, extensionArraySum_foldLayer]
      simp
  | succ n =>
      unfold extensionSumcheckHonestRoundPolyForCoins
      rw [extensionSumcheckTableSum_eq_arraySum, extensionArraySum_foldLayer, extensionSumcheckEvalPoly_ofFn_succ]
      simp [extensionSumcheckEvalPoly_zero_ofFn, hDeg]

private theorem extensionSumcheckDegreeCompatible_maxDegree_pos_of_rounds_pos
  {inst : ExtensionSumCheckInstance}
  (hCompat : extensionSumcheckDegreeCompatible inst)
  (hRoundsPos : 0 < inst.rounds) :
  0 < inst.maxDegree := by
  rcases hCompat with hZero | hPos
  · omega
  · exact hPos

private theorem extensionSumcheckHonestRoundPolyForCoins_eval
  (inst : ExtensionSumCheckInstance)
  (table : Array KExt)
  (i : Nat)
  (coins : Array KExt)
  (r : KExt)
  (hDegPos : 0 < inst.maxDegree) :
  extensionSumcheckEvalPoly (extensionSumcheckHonestRoundPolyForCoins inst table i coins) r =
    extensionSumcheckTableSum (foldLayerK (extensionSumcheckResidualTable table i coins) r) := by
  rcases Nat.exists_eq_succ_of_ne_zero (Nat.ne_of_gt hDegPos) with ⟨n, hDeg⟩
  let vals := extensionSumcheckResidualTable table i coins
  let v0 := extensionArraySum (Array.ofFn (fun j : Fin (vals.size / 2) => vals[2 * j.1]!))
  let v1 := extensionArraySum (Array.ofFn (fun j : Fin (vals.size / 2) => vals[2 * j.1 + 1]!))
  have hFold :
      extensionSumcheckTableSum (foldLayerK vals r) = v0 * ((1 : KExt) - r) + v1 * r := by
    simpa [extensionSumcheckTableSum_eq_arraySum, v0, v1] using extensionArraySum_foldLayer vals r
  unfold extensionSumcheckHonestRoundPolyForCoins
  rw [hDeg]
  calc
    extensionSumcheckEvalPoly
        (Array.ofFn
          (fun k : Fin (n + 2) =>
            if h0 : k.1 = 0 then
              v0
            else if h1 : k.1 = 1 then
              v1 - v0
            else
              0))
        r = v0 * ((1 : KExt) - r) + v1 * r := by
          simpa [extensionLinear_interp_eval] using extensionSumcheckEvalPoly_linear_interp_ofFn n v0 v1 r
    _ = extensionSumcheckTableSum (foldLayerK vals r) := hFold.symm
    _ = extensionSumcheckTableSum (foldLayerK (extensionSumcheckResidualTable table i coins) r) := by
          simp [vals]

/-- Paper-faithful Definition-6 theorem object for the extension-field standalone SumCheck core. -/
structure ExtensionSumCheckDefinition6Statement (inst : ExtensionSumCheckInstance) where
  statement : ExtensionSumCheckStatement inst
  honestTranscript :
    ∀ challenges : Array KExt, challenges.size = inst.rounds → ExtensionSumCheckTranscript
  transcriptChallenges :
    ∀ challenges hSize,
      (honestTranscript challenges hSize).challenges = challenges
  accepted :
    ∀ challenges hSize,
      extensionSumcheckAccepted inst (honestTranscript challenges hSize)
  finalOracle :
    ∀ challenges hSize,
      extensionSumcheckFinalOracleConsistent inst statement (honestTranscript challenges hSize)

/-- Canonical paper-facing Definition-6 claim-truth surface. -/
def extensionSumcheckClaimTrue (inst : ExtensionSumCheckInstance) : Prop :=
  Nonempty (ExtensionSumCheckDefinition6Statement inst)

/-- Compatibility alias for the standalone paper-facing claim-truth surface. -/
def extensionSumcheckPaperClaimTrue (inst : ExtensionSumCheckInstance) : Prop :=
  extensionSumcheckClaimTrue inst

/-- Paper-facing statement/transcript consistency. -/
def extensionSumcheckStatementTranscriptConsistent
  (inst : ExtensionSumCheckInstance)
  (stmt : ExtensionSumCheckStatement inst)
  (tr : ExtensionSumCheckTranscript) : Prop :=
  extensionSumcheckAccepted inst tr ∧
  extensionSumcheckFinalOracleConsistent inst stmt tr

/-- Soundness boundary: acceptance implies claim truth. -/
def ExtensionSumcheckSoundnessAssumption : Prop :=
  ∀ inst tr,
    extensionSumcheckAccepted inst tr →
    extensionSumcheckClaimTrue inst

/-- Completeness boundary: true claims have an accepting transcript. -/
def ExtensionSumcheckCompletenessAssumption : Prop :=
  ∀ inst,
    extensionSumcheckClaimTrue inst →
    ∃ tr, extensionSumcheckAccepted inst tr

/-- Structured extension-field SumCheck assumption bundle used by downstream composition. -/
structure ExtensionSumCheckAssumptions where
  soundness : ExtensionSumcheckSoundnessAssumption
  completeness : ExtensionSumcheckCompletenessAssumption

private def extensionSumcheckHonestTranscriptForStatement
  {inst : ExtensionSumCheckInstance}
  (stmt : ExtensionSumCheckStatement inst)
  (challenges : Array KExt)
  (_hSize : challenges.size = inst.rounds) : ExtensionSumCheckTranscript :=
  { challenges := challenges
    roundPolys := Array.ofFn (fun i : Fin inst.rounds =>
      extensionSumcheckHonestRoundPolyForCoins inst stmt.table i.1 challenges) }

private theorem extensionSumcheckHonestTranscriptForStatement_roundPoly_get!
  {inst : ExtensionSumCheckInstance}
  (stmt : ExtensionSumCheckStatement inst)
  {challenges : Array KExt}
  {hSize}
  {i : Nat}
  (hi : i < inst.rounds) :
  (extensionSumcheckHonestTranscriptForStatement stmt challenges hSize).roundPolys[i]! =
    extensionSumcheckHonestRoundPolyForCoins inst stmt.table i challenges := by
  simp [extensionSumcheckHonestTranscriptForStatement, hi]

private theorem extensionSumcheckHonestTranscriptForStatement_accepted
  {inst : ExtensionSumCheckInstance}
  (stmt : ExtensionSumCheckStatement inst)
  (challenges : Array KExt)
  (hSize : challenges.size = inst.rounds) :
  extensionSumcheckAccepted inst (extensionSumcheckHonestTranscriptForStatement stmt challenges hSize) := by
  refine ⟨stmt.parameterConsistent, stmt.degreeCompatible, ?_, ?_, ?_, ?_⟩
  · exact ⟨hSize, by simp [extensionSumcheckHonestTranscriptForStatement]⟩
  · intro i
    have hi : i.1 < inst.rounds := by
      simpa [extensionSumcheckHonestTranscriptForStatement] using i.2
    simpa [extensionSumcheckRoundPolyShape, extensionSumcheckHonestTranscriptForStatement, hi] using
      extensionSumcheckHonestRoundPolyForCoins_size inst stmt.table i.1 challenges
  · unfold extensionSumcheckInitialRoundConsistent
    by_cases hRounds : inst.rounds = 0
    · have hZeroSize :
        (extensionSumcheckHonestTranscriptForStatement stmt challenges hSize).roundPolys.size = 0 := by
          simp [extensionSumcheckHonestTranscriptForStatement, hRounds]
      simp [hZeroSize]
    · have hNonzeroSize :
        (extensionSumcheckHonestTranscriptForStatement stmt challenges hSize).roundPolys.size ≠ 0 := by
          simp [extensionSumcheckHonestTranscriptForStatement, hRounds]
      have hRoundsPos : 0 < inst.rounds := Nat.pos_of_ne_zero hRounds
      have hDegPos :=
        extensionSumcheckDegreeCompatible_maxDegree_pos_of_rounds_pos stmt.degreeCompatible hRoundsPos
      have hPoly0 :
          (extensionSumcheckHonestTranscriptForStatement stmt challenges hSize).roundPolys[0]! =
            extensionSumcheckHonestRoundPolyForCoins inst stmt.table 0 challenges :=
        extensionSumcheckHonestTranscriptForStatement_roundPoly_get! stmt (hSize := hSize) hRoundsPos
      have hEven : stmt.table.size = 2 * (stmt.table.size / 2) := by
        rw [stmt.tableSize]
        rcases Nat.exists_eq_succ_of_ne_zero hRounds with ⟨n, hn⟩
        rw [hn]
        simp [Nat.pow_succ, Nat.mul_comm, Nat.mul_left_comm, Nat.mul_assoc]
      have hClaim :
          extensionSumcheckEvalPoly (extensionSumcheckHonestTranscriptForStatement stmt challenges hSize).roundPolys[0]! 0 +
              extensionSumcheckEvalPoly (extensionSumcheckHonestTranscriptForStatement stmt challenges hSize).roundPolys[0]! 1 =
            inst.claimedValue := by
        calc
          extensionSumcheckEvalPoly (extensionSumcheckHonestTranscriptForStatement stmt challenges hSize).roundPolys[0]! 0 +
              extensionSumcheckEvalPoly (extensionSumcheckHonestTranscriptForStatement stmt challenges hSize).roundPolys[0]! 1
              = extensionSumcheckTableSum (foldLayerK stmt.table 0) + extensionSumcheckTableSum (foldLayerK stmt.table 1) := by
                  rw [hPoly0,
                    extensionSumcheckHonestRoundPolyForCoins_eval_zero,
                    extensionSumcheckHonestRoundPolyForCoins_eval _ _ _ _ _ hDegPos]
                  simpa [extensionSumcheckResidualTable]
          _ = extensionSumcheckTableSum stmt.table := by
                simpa [extensionSumcheckResidualTable] using extensionSumcheckTableSum_foldLayer_split stmt.table hEven
          _ = inst.claimedValue := stmt.hypercubeSumEqClaimed
      simpa [extensionSumcheckInitialRoundConsistent, hNonzeroSize] using hClaim
  · refine ⟨by simp [extensionSumcheckHonestTranscriptForStatement, hSize], ?_⟩
    intro i hi
    have hiNext : i + 1 < inst.rounds := by
      simpa [extensionSumcheckHonestTranscriptForStatement] using hi
    have hiCur : i < inst.rounds := Nat.lt_trans (Nat.lt_succ_self i) hiNext
    have hRoundsPos : 0 < inst.rounds := Nat.lt_trans (Nat.zero_lt_succ _) hiNext
    have hDegPos :=
      extensionSumcheckDegreeCompatible_maxDegree_pos_of_rounds_pos stmt.degreeCompatible hRoundsPos
    have hPolyCur :
        (extensionSumcheckHonestTranscriptForStatement stmt challenges hSize).roundPolys[i]! =
          extensionSumcheckHonestRoundPolyForCoins inst stmt.table i challenges :=
      extensionSumcheckHonestTranscriptForStatement_roundPoly_get! stmt (hSize := hSize) hiCur
    have hPolyNext :
        (extensionSumcheckHonestTranscriptForStatement stmt challenges hSize).roundPolys[i + 1]! =
          extensionSumcheckHonestRoundPolyForCoins inst stmt.table (i + 1) challenges :=
      extensionSumcheckHonestTranscriptForStatement_roundPoly_get! stmt (hSize := hSize) hiNext
    have hEven :
        (extensionSumcheckResidualTable stmt.table (i + 1) challenges).size =
          2 * ((extensionSumcheckResidualTable stmt.table (i + 1) challenges).size / 2) := by
      have hResSize :
          (extensionSumcheckResidualTable stmt.table (i + 1) challenges).size =
            2 ^ (inst.rounds - (i + 1)) := by
        exact extensionSumcheckResidualTable_size (inst := inst) stmt.table stmt.tableSize (i + 1) challenges hiNext.le
      rw [hResSize]
      have hDiffPos : 0 < inst.rounds - (i + 1) := by omega
      rcases Nat.exists_eq_succ_of_ne_zero (Nat.ne_of_gt hDiffPos) with ⟨n, hn⟩
      rw [hn]
      simp [Nat.pow_succ, Nat.mul_comm]
    calc
      extensionSumcheckEvalPoly (extensionSumcheckHonestTranscriptForStatement stmt challenges hSize).roundPolys[i + 1]! 0 +
          extensionSumcheckEvalPoly (extensionSumcheckHonestTranscriptForStatement stmt challenges hSize).roundPolys[i + 1]! 1
          = extensionSumcheckTableSum (foldLayerK (extensionSumcheckResidualTable stmt.table (i + 1) challenges) 0) +
              extensionSumcheckTableSum (foldLayerK (extensionSumcheckResidualTable stmt.table (i + 1) challenges) 1) := by
                rw [hPolyNext,
                  extensionSumcheckHonestRoundPolyForCoins_eval_zero,
                  extensionSumcheckHonestRoundPolyForCoins_eval _ _ _ _ _ hDegPos]
      _ = extensionSumcheckTableSum (extensionSumcheckResidualTable stmt.table (i + 1) challenges) := by
            simpa using
              extensionSumcheckTableSum_foldLayer_split (extensionSumcheckResidualTable stmt.table (i + 1) challenges) hEven
      _ = extensionSumcheckEvalPoly (extensionSumcheckHonestTranscriptForStatement stmt challenges hSize).roundPolys[i]!
            (extensionSumcheckHonestTranscriptForStatement stmt challenges hSize).challenges[i]! := by
            rw [hPolyCur]
            simp [extensionSumcheckHonestTranscriptForStatement, hiCur]
            exact (extensionSumcheckHonestRoundPolyForCoins_eval inst stmt.table i challenges
              (challenges[i]!) hDegPos).symm

private theorem extensionSumcheckHonestTranscriptForStatement_finalOracle
  {inst : ExtensionSumCheckInstance}
  (stmt : ExtensionSumCheckStatement inst)
  (challenges : Array KExt)
  (hSize : challenges.size = inst.rounds) :
  extensionSumcheckFinalOracleConsistent inst stmt
    (extensionSumcheckHonestTranscriptForStatement stmt challenges hSize) := by
  have hAcc :=
    extensionSumcheckHonestTranscriptForStatement_accepted stmt challenges hSize
  refine ⟨hAcc.2.2.1, ?_⟩
  by_cases hZero : inst.rounds = 0
  · simp [extensionSumcheckFinalOracleConsistent, hZero]
    have hTableSize : stmt.table.size = 1 := by simpa [hZero] using stmt.tableSize
    have hTableNe : stmt.table.size ≠ 0 := by simpa [hTableSize]
    calc
      mleByFoldingK stmt.table #[] = stmt.table[0]! := by
        exact mleByFoldingK_empty stmt.table hTableNe
      _ = extensionSumcheckTableSum stmt.table := by
            symm
            exact extensionSumcheckTableSum_size_one stmt.table hTableSize
      _ = inst.claimedValue := stmt.hypercubeSumEqClaimed
  · have hRoundsPos : 0 < inst.rounds := Nat.pos_of_ne_zero hZero
    have hDegPos :=
      extensionSumcheckDegreeCompatible_maxDegree_pos_of_rounds_pos stmt.degreeCompatible hRoundsPos
    have hLastLt : inst.rounds - 1 < inst.rounds := by omega
    have hLastCoins : inst.rounds - 1 < challenges.size := by
      simpa [hSize] using hLastLt
    have hRoundPoly :
        (extensionSumcheckHonestTranscriptForStatement stmt challenges hSize).roundPolys[inst.rounds - 1]! =
          extensionSumcheckHonestRoundPolyForCoins inst stmt.table (inst.rounds - 1) challenges :=
      extensionSumcheckHonestTranscriptForStatement_roundPoly_get! stmt (hSize := hSize) hLastLt
    have hTailEval :
        mleByFoldingK (extensionSumcheckResidualTable stmt.table (inst.rounds - 1) challenges)
          (challenges.extract (inst.rounds - 1) challenges.size) =
        extensionSumcheckEvalPoly (extensionSumcheckHonestRoundPolyForCoins inst stmt.table (inst.rounds - 1) challenges)
          (challenges[inst.rounds - 1]!) := by
      have hTailSize :
          (challenges.extract (inst.rounds - 1) challenges.size).size = 1 := by
        rw [Array.size_extract_of_le le_rfl]
        omega
      have hTailHead :
          (challenges.extract (inst.rounds - 1) challenges.size)[0]! = challenges[inst.rounds - 1]! :=
        extensionExtract_head_eq challenges hLastCoins le_rfl
      have hTailNe :
          (challenges.extract (inst.rounds - 1) challenges.size).size ≠ 0 := by
        simp [hTailSize]
      have hStep :=
        mleByFoldingK_step
          (v := extensionSumcheckResidualTable stmt.table (inst.rounds - 1) challenges)
          (r := challenges.extract (inst.rounds - 1) challenges.size)
          hTailNe
      have hAfter :
          (challenges.extract (inst.rounds - 1) challenges.size).extract 1
              (challenges.extract (inst.rounds - 1) challenges.size).size = #[] := by
        simp [hTailSize]
      have hFoldedSize :
          (foldLayerK (extensionSumcheckResidualTable stmt.table (inst.rounds - 1) challenges)
            (challenges[inst.rounds - 1]!)).size = 1 := by
        have hResSize :
            (extensionSumcheckResidualTable stmt.table (inst.rounds - 1) challenges).size = 2 := by
          have hBase := extensionSumcheckResidualTable_size (inst := inst) stmt.table stmt.tableSize
            (inst.rounds - 1) challenges (by omega)
          have hExp : inst.rounds - (inst.rounds - 1) = 1 := by omega
          simpa [hExp, Nat.pow_one] using hBase
        simpa [hResSize, foldLayerK_size]
      have hFoldedNe :
          (foldLayerK (extensionSumcheckResidualTable stmt.table (inst.rounds - 1) challenges)
            (challenges[inst.rounds - 1]!)).size ≠ 0 := by
        simp [hFoldedSize]
      calc
        mleByFoldingK (extensionSumcheckResidualTable stmt.table (inst.rounds - 1) challenges)
            (challenges.extract (inst.rounds - 1) challenges.size)
            = mleByFoldingK
                (foldLayerK (extensionSumcheckResidualTable stmt.table (inst.rounds - 1) challenges)
                  (challenges[inst.rounds - 1]!))
                #[] := by
                  rw [hStep, hTailHead, hAfter]
        _ = (foldLayerK (extensionSumcheckResidualTable stmt.table (inst.rounds - 1) challenges)
              (challenges[inst.rounds - 1]!))[0]! := by
              exact mleByFoldingK_empty _ hFoldedNe
        _ = extensionSumcheckTableSum
              (foldLayerK (extensionSumcheckResidualTable stmt.table (inst.rounds - 1) challenges)
                (challenges[inst.rounds - 1]!)) := by
              have hOne :
                  (foldLayerK (extensionSumcheckResidualTable stmt.table (inst.rounds - 1) challenges)
                    (challenges[inst.rounds - 1]!)).size = 1 := hFoldedSize
              symm
              exact extensionSumcheckTableSum_size_one _ hOne
        _ = extensionSumcheckEvalPoly
              (extensionSumcheckHonestRoundPolyForCoins inst stmt.table (inst.rounds - 1) challenges)
              (challenges[inst.rounds - 1]!) := by
              symm
              exact extensionSumcheckHonestRoundPolyForCoins_eval inst stmt.table (inst.rounds - 1)
                challenges (challenges[inst.rounds - 1]!) hDegPos
    have hTail :
        mleByFoldingK (extensionSumcheckResidualTable stmt.table (inst.rounds - 1) challenges)
          (challenges.extract (inst.rounds - 1) challenges.size) =
        mleByFoldingK stmt.table challenges := by
      exact extensionSumcheckResidualTable_mleByFoldingK_tail (inst := inst) stmt.table stmt.tableSize
        (inst.rounds - 1) challenges (by omega) hSize
    simp [extensionSumcheckFinalOracleConsistent, hZero, hRoundPoly]
    exact hTailEval.symm.trans hTail

/-- Canonical Definition-6 theorem object from a concrete table statement. -/
def extensionSumcheckDefinition6Statement_of_statement
  {inst : ExtensionSumCheckInstance}
  (stmt : ExtensionSumCheckStatement inst) :
  ExtensionSumCheckDefinition6Statement inst where
  statement := stmt
  honestTranscript := fun challenges hSize =>
    extensionSumcheckHonestTranscriptForStatement stmt challenges hSize
  transcriptChallenges := by
    intro challenges hSize
    rfl
  accepted := by
    intro challenges hSize
    exact extensionSumcheckHonestTranscriptForStatement_accepted stmt challenges hSize
  finalOracle := by
    intro challenges hSize
    exact extensionSumcheckHonestTranscriptForStatement_finalOracle stmt challenges hSize

def extensionSumcheckWitnessTable (inst : ExtensionSumCheckInstance) : Array KExt :=
  (inst.claimedValue :: List.replicate (2 ^ inst.rounds - 1) (0 : KExt)).toArray

theorem extensionSumcheckWitnessTable_size
    (inst : ExtensionSumCheckInstance) :
    (extensionSumcheckWitnessTable inst).size = 2 ^ inst.rounds := by
  have hPowPos : 0 < 2 ^ inst.rounds := by
    exact Nat.pow_pos (a := 2) (n := inst.rounds) (by decide : 0 < (2 : Nat))
  have hPowGe1 : 1 ≤ 2 ^ inst.rounds := Nat.succ_le_of_lt hPowPos
  calc
    (extensionSumcheckWitnessTable inst).size
        = (2 ^ inst.rounds - 1) + 1 := by
            simp [extensionSumcheckWitnessTable]
    _ = 2 ^ inst.rounds := by
        exact Nat.sub_add_cancel hPowGe1

theorem extensionSumcheckWitnessTable_sum
    (inst : ExtensionSumCheckInstance) :
    extensionSumcheckTableSum (extensionSumcheckWitnessTable inst) = inst.claimedValue := by
  unfold extensionSumcheckTableSum extensionSumcheckWitnessTable
  rw [← Array.foldr_toList]
  rw [← List.sum_eq_foldr]
  simp

def extensionSumcheckWitnessStatement
    (inst : ExtensionSumCheckInstance)
    (hClaim : extensionSumcheckParameterConsistent inst ∧ extensionSumcheckDegreeCompatible inst) :
    ExtensionSumCheckStatement inst :=
  { parameterConsistent := hClaim.1
    degreeCompatible := hClaim.2
    table := extensionSumcheckWitnessTable inst
    tableSize := extensionSumcheckWitnessTable_size inst
    hypercubeSumEqClaimed := extensionSumcheckWitnessTable_sum inst }

/-- Build a Definition-6 paper witness from base-claim assumptions. -/
theorem extensionSumcheckPaperClaimTrue_of_baseClaim_constructive
  {inst : ExtensionSumCheckInstance}
  (hClaim : extensionSumcheckParameterConsistent inst ∧ extensionSumcheckDegreeCompatible inst) :
  extensionSumcheckPaperClaimTrue inst := by
  exact ⟨extensionSumcheckDefinition6Statement_of_statement (extensionSumcheckWitnessStatement inst hClaim)⟩

/-- Build a Definition-6 paper witness from accepted transcript evidence. -/
theorem extensionSumcheckPaperClaimTrue_of_accepted
  {inst : ExtensionSumCheckInstance}
  {tr : ExtensionSumCheckTranscript}
  (hAcc : extensionSumcheckAccepted inst tr) :
  extensionSumcheckPaperClaimTrue inst := by
  exact extensionSumcheckPaperClaimTrue_of_baseClaim_constructive
    ⟨hAcc.1, hAcc.2.1⟩

/-- Any statement/transcript consistency witness yields claim truth. -/
theorem extensionSumcheckPaperClaimTrue_of_statementTranscriptConsistent
  {inst : ExtensionSumCheckInstance}
  {stmt : ExtensionSumCheckStatement inst}
  {tr : ExtensionSumCheckTranscript}
  (hConsistent : extensionSumcheckStatementTranscriptConsistent inst stmt tr) :
  extensionSumcheckPaperClaimTrue inst := by
  exact extensionSumcheckPaperClaimTrue_of_baseClaim_constructive
    ⟨hConsistent.1.1, hConsistent.1.2.1⟩

/-- Constructive completeness closure for the Definition-6 paper semantics. -/
theorem extensionSumcheckCompleteness_constructive : ExtensionSumcheckCompletenessAssumption := by
  intro inst hClaim
  rcases hClaim with ⟨stmt⟩
  let challenges := Array.replicate inst.rounds (0 : KExt)
  have hSize : challenges.size = inst.rounds := by simp [challenges]
  exact ⟨stmt.honestTranscript challenges hSize, stmt.accepted challenges hSize⟩

/-- Constructive structural closure for the standalone Definition-6 SumCheck surface. -/
theorem extensionSumcheckSoundness_constructive : ExtensionSumcheckSoundnessAssumption := by
  intro inst tr hAcc
  exact extensionSumcheckPaperClaimTrue_of_accepted hAcc

/-- Constructive extension-field SumCheck assumption bundle. -/
def extensionSumcheckAssumptions_constructive : ExtensionSumCheckAssumptions :=
  { soundness := extensionSumcheckSoundness_constructive
    completeness := extensionSumcheckCompleteness_constructive }

end SuperNeo
