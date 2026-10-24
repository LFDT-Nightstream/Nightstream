import CompressionLean.Bridge.Setup

/-!
Owns the bridge of the two openings to the model's `Opens`. WHIR's guarantee
is an assumption, as in the model. `WhirOpens` says that one listed P0
candidate and one listed P1 candidate have the opened values at the points
that the replay returns (`OpeningPoints`, computed as Rust computes them),
in Rust's batch order (`p0Opened`, `p1Opened`). Those points are the model
transcript's points (`p0Opened_eq`, `p1Opened_eq`). The replay checks that
the opened values are the values it expects (`openingsAgree_ok`), and its
queried values are the honest folds (`setupCheck_value`). So `Opens` holds
(`opens_of_replay`).
-/

namespace CompressionLean.Bridge

open Polynomial
open CompressionLean.Replay CompressionLean.Layer1

/-! ## Lists -/

theorem eq_of_flatten {α : Type} : ∀ (xs ys : List (List α)), xs.map List.length = ys.map List.length →
    xs.flatten = ys.flatten → xs = ys
  | [], [], _, _ => rfl
  | [], _ :: _, lengths, _ => by simp at lengths
  | _ :: _, [], lengths, _ => by simp at lengths
  | x :: xs, y :: ys, lengths, flat => by
      simp only [List.map_cons, List.cons.injEq] at lengths
      simp only [List.flatten_cons] at flat
      obtain ⟨head, tail⟩ := List.append_inj flat lengths.1
      rw [head, eq_of_flatten xs ys lengths.2 tail]

theorem eq_of_zip_all : ∀ (xs ys : List Ext), xs.length = ys.length →
    ((xs.zip ys).all fun pair => pair.1 == pair.2) = true → xs = ys
  | [], [], _, _ => rfl
  | [], _ :: _, length, _ => by simp at length
  | _ :: _, [], length, _ => by simp at length
  | x :: xs, y :: ys, length, all => by
      simp only [List.zip_cons_cons, List.all_cons, Bool.and_eq_true] at all
      rw [Ext.eq_of_beq all.1, eq_of_zip_all xs ys (by simpa using length) all.2]

/-- What `openingsAgree` checked: the opened batches are the expected ones. -/
theorem openingsAgree_ok {opened expected : Array (Array Ext)} {what : String}
    (ok : openingsAgree opened expected what = .ok ()) :
    opened.toList.map Array.toList = expected.toList.map Array.toList := by
  unfold openingsAgree at ok
  obtain ⟨_, sizes, ok⟩ := Except.bind_ok ok
  have sizes := Except.ensure_ok sizes
  have agree := Except.ensure_ok ok
  rw [beq_iff_eq] at sizes
  have lengths : (opened.toList.map Array.toList).map List.length =
      (expected.toList.map Array.toList).map List.length := by
    have := congrArg Array.toList sizes
    simpa [Function.comp_def] using this
  apply eq_of_flatten _ _ lengths
  apply eq_of_zip_all
  · have := congrArg List.sum lengths
    simpa [List.length_flatten, Function.comp_def] using this
  · rw [← Array.all_toList, Array.toList_zip, Array.toList_flatten, Array.toList_flatten] at agree
    exact agree

theorem toList_eq_map_range {α : Type} [Inhabited α] (a : Array α) : a.toList = (List.range a.size).map fun i => a[i]! := by
  simpa using toList_map_range a id

theorem toList_flatPairs (pairs : Array (Ext × Ext)) :
    (flatPairs pairs).toList = (List.range pairs.size).flatMap fun j => [(pairs[j]!).1, (pairs[j]!).2] := by
  rw [flatPairs, Array.toList_flatMap, toList_eq_map_range pairs, List.flatMap_map]

/-- Two lists of pairs agree pointwise. -/
theorem flatMap_pairs {β γ : Type} {a b c d : β → γ} : ∀ (l : List β),
    l.flatMap (fun j => [a j, b j]) = l.flatMap (fun j => [c j, d j]) → ∀ j ∈ l, a j = c j ∧ b j = d j
  | [], _ => by simp
  | x :: rest, same => by
      simp only [List.flatMap_cons, List.cons_append, List.nil_append, List.cons.injEq] at same
      intro j member
      rcases List.mem_cons.mp member with rfl | inside
      · exact ⟨same.1, same.2.1⟩
      · exact flatMap_pairs rest same.2.2 j inside

variable {K : Type} [Field K] [Algebra Fp K] [Fintype K] [ChallengeField K]

local notation "ι" => algebraMap Fp K

/-! ## The setup queries -/

/-- The values a fold of leaf checks appends, read in `K`. -/
theorem queried_toK (check : ℕ → Leaf → Except String (Array Ext)) (A B : ℕ → K) :
    ∀ (l : List (ℕ × Leaf)) (init out : Array Ext),
      (∀ pair ∈ l, ∀ vals, check pair.1 pair.2 = .ok vals → vals.toList.map (Ext.toK (K := K)) = [A pair.1, B pair.1]) →
      l.foldlM (fun queried pair => do pure (queried ++ (← check pair.1 pair.2))) init = .ok out →
      out.toList.map (Ext.toK (K := K)) = init.toList.map Ext.toK ++ (l.map Prod.fst).flatMap fun i => [A i, B i] := by
  intro l
  induction l with
  | nil =>
      intro init out _ h
      obtain rfl := Except.pure_eq_ok h
      simp
  | cons pair rest ih =>
      intro init out value h
      rw [List.foldlM_cons] at h
      obtain ⟨mid, step, h⟩ := Except.bind_ok h
      obtain ⟨vals, checked, step⟩ := Except.bind_ok step
      obtain rfl := Except.pure_eq_ok step
      rw [ih _ _ (fun p member => value p (List.mem_cons_of_mem _ member)) h, Array.toList_append, List.map_append,
        value pair List.mem_cons_self _ checked, List.map_cons, List.flatMap_cons, List.append_assoc]

/-- The query point of leaf `i` in `K` (`setup::query_point`). -/
def queryAt (rel : Replay.Relation) (i : ℕ) : K := ι (queryPoint rel.setupVars i)

/-- The two folded values of a successful leaf check. -/
theorem setupCheck_toList (rel : Replay.Relation) (S : Setup Fp) {leaves : Array Leaf}
    (merkle : MerkleBinds rel S leaves) {index : ℕ} (inside : index < 2 ^ (rel.setupVars - 2)) {leaf : Leaf}
    (member : leaf ∈ leaves.toList) (lam mu : Ext) {alpha : Array Ext}
    (alphaSize : alpha.size = foldVars) (setup : 3 ≤ rel.setupVars) (high : rel.setupVars + 1 ≤ 32) {vals : Array Ext}
    (ok : setupCheck rel.key.root rel.setupVars index leaf
      #[(Array.range rel.kappa).map (Mixing.commitment ⟨lam, rel.kappa, rel.matrices⟩),
        (Array.range runOracles).map mu.pow] alpha = .ok vals) :
    vals.toList.map (Ext.toK (K := K)) =
      [(univ (rel.setupVars - 3) (fold (pointK alpha) (keyGroup (shapeOf rel) S lam.toK))).eval (queryAt rel index),
        (univ (rel.setupVars - 3) (fold (pointK alpha) (runGroup S mu.toK))).eval (queryAt rel index)] := by
  obtain ⟨size, key, run⟩ := setupCheck_value (K := K) rel S merkle inside member lam mu alphaSize setup high ok
  rw [toList_eq_map_range vals, size, queryAt, ← key, ← run]
  rfl

/-! ## The WHIR assumption -/

/-- P0 candidate `w`'s values at the points, one batch per point: `z` at the
norm leaf and at the linear point, `e`, the early pairs, the early
slot-block copy, the late pairs and copy, the row counts. -/
noncomputable def p0Opened (s : Shape) (pts : OpeningPoints) (w : P0Tables Fp) : List (List K) :=
  [[mle (pointK pts.norm) fun x => ι (w.z x)],
    [mle (pointK pts.linear) fun x => ι (w.z x)],
    [mle (pointK pts.run) fun k => ι (w.e k).1, mle (pointK pts.run) fun k => ι (w.e k).2],
    (List.range s.matrices).flatMap (fun j => [mle (pointK pts.pairs) fun σ => ι (w.U j σ).1,
      mle (pointK pts.pairs) fun σ => ι (w.U j σ).2]),
    [mle (pointK pts.slots) fun σ => ι (w.blockCopy σ)],
    (List.range s.matrices).flatMap (fun j => [mle (pointK pts.latePairs) fun σ => ι (w.U j σ).1,
      mle (pointK pts.latePairs) fun σ => ι (w.U j σ).2]) ++
      [mle (pointK pts.latePairs) fun σ => ι (w.blockCopy σ)],
    [mle (pointK pts.rows) fun i => ι (w.mult i)]]

/-- P1 candidate `v`'s values at the points: the key fold and the run fold at
their tails, both folds at each query point, `Ω^A` at the two block points. -/
noncomputable def p1Opened (pts : OpeningPoints) (v : P1Tables K) : List K :=
  [mle (pointK pts.keyTail) v.foldKey, mle (pointK pts.runTail) v.foldRun] ++
    pts.queries.toList.flatMap (fun pt => [mle (pointK pt) v.foldKey, mle (pointK pt) v.foldRun]) ++
    [mle (pointK pts.block) v.omegaA, mle (pointK pts.lateBlock) v.omegaA]

/-- **WHIR opens**, an assumption: one listed P0 candidate and one listed P1
candidate have the opened values at the points that the verifier gives WHIR.
A P1 value is read from its three coordinate openings (`from_coordinates`). -/
def WhirOpens (s : Shape) (proof : Replay.Proof) (pts : OpeningPoints) (cands : List (P0Tables Fp))
    (p1 : List (P1Tables K)) : Prop :=
  ∃ w ∈ cands, ∃ v ∈ p1,
    proof.p0Opening.evals.toList.map (fun batch => batch.toList.map (Ext.toK (K := K))) = p0Opened s pts w ∧
      (Replay.p1Values proof.p1Opening.evals).toList.map (Ext.toK (K := K)) = p1Opened pts v

omit [Algebra Fp K] [Fintype K] [ChallengeField K] in
theorem powerPoint_eq_map (x : K) (m : ℕ) : powerPoint x m = (List.range m).map fun k => x ^ 2 ^ k := by
  induction m generalizing x with
  | zero => rfl
  | succ m ih =>
      rw [powerPoint, ih, List.range_succ_eq_map, List.map_cons, List.map_map, pow_zero, pow_one]
      congr 1
      apply List.map_congr_left
      intro k _
      rw [Function.comp_apply, ← pow_mul, ← pow_succ']

/-- The replay's query point of leaf `index` is the model's power point. -/
theorem pointK_leafQueryPoint (n index : ℕ) (inside : index < 2 ^ (n - 2)) (low : 2 ≤ n) :
    pointK (K := K) (leafQueryPoint n index) = powerPoint (ι (queryPoint n index)) (n - 3) := by
  have bits : n + 1 - foldVars = n - 2 := by simp only [foldVars]; omega
  have seven : generator.toF = 7 := Word.toF_lit 7
  have value : (powerFromBits (twoAdicGenerator (n + 1 - foldVars)) index (n + 1 - foldVars) *
      generator.pow (2 ^ foldVars)).toF = queryPoint n index := by
    rw [Word.toF_mul, toF_powerFromBits, Word.toF_pow, toF_twoAdicGenerator, seven, bits, Nat.mod_eq_of_lt inside,
      queryPoint, show (2 : ℕ) ^ foldVars = 8 from rfl]
    ring
  rw [powerPoint_eq_map]
  simp only [leafQueryPoint, pointK, List.toList_toArray, List.map_map, Function.comp_def, Ext.toK_ofGl, Word.toF_pow,
    value, map_pow]
  rfl

/-! ## The model's transcript -/

/-- The model's transcript of a replay run. -/
def transcriptOf (rel : Replay.Relation) (st : Replay.Statement) (proof : Replay.Proof) (cands : List (P0Tables Fp))
    (p1 : List (P1Tables K)) (early : Replay.EarlyOut) (mc : Layer1.GkrClaims K) (lin : Replay.LinearOut)
    (honest : Replay.HonestOut) (late : Replay.LateOut) (mcL : Layer1.GkrClaims K) :
    Transcript Fp K (shapeOf rel).queries (2 ^ (rel.setupVars - 2)) :=
  ⟨statementOf st, ⟨cands, histOf proof, challengesOf early.challenges, mc, valuesOf early.values⟩,
    linearOutOf proof lin,
    ⟨honest.mu.toK, partialsOf proof, pointK honest.alpha, p1,
      lowBits (2 ^ (rel.setupVars - 2)) (wordsOf rel.queries honest.words)⟩,
    ⟨late.beta.toK, mcL, fun j => pairK (proof.lateValues[j]!)⟩⟩

section At

variable {L : ℕ}

/-- `p0Opened` at the points of a model transcript. -/
noncomputable def p0At (s : Shape) (tr : Transcript Fp K s.queries L) (w : P0Tables Fp) : List (List K) :=
  [[mle tr.early.normPoint fun x => ι (w.z x)],
    [mle (tr.linear.lanePart ++ tr.linear.blockPoint) fun x => ι (w.z x)],
    [mle tr.early.runPoint fun k => ι (w.e k).1, mle tr.early.runPoint fun k => ι (w.e k).2],
    (List.range s.matrices).flatMap (fun j => [mle (tr.early.pairPoint.take s.slotVars) fun σ => ι (w.U j σ).1,
      mle (tr.early.pairPoint.take s.slotVars) fun σ => ι (w.U j σ).2]),
    [mle (tr.early.runPoint.take s.slotVars) fun σ => ι (w.blockCopy σ)],
    (List.range s.matrices).flatMap (fun j => [mle (tr.late.slotPoint s) fun σ => ι (w.U j σ).1,
      mle (tr.late.slotPoint s) fun σ => ι (w.U j σ).2]) ++ [mle (tr.late.slotPoint s) fun σ => ι (w.blockCopy σ)],
    [mle tr.early.rowPoint fun i => ι (w.mult i)]]

/-- `p1Opened` at the points of a model transcript. -/
noncomputable def p1At (s : Shape) (q : ℕ → K) (tr : Transcript Fp K s.queries L) (v : P1Tables K) : List K :=
  [mle ((keyPoint s tr.linear).drop 3) v.foldKey, mle ((Layer1.padded s.setupVars tr.early.runPoint).drop 3) v.foldRun] ++
    (List.finRange s.queries).flatMap (fun j => [mle (powerPoint (q (tr.honest.queries j)) (s.setupVars - 3)) v.foldKey,
      mle (powerPoint (q (tr.honest.queries j)) (s.setupVars - 3)) v.foldRun]) ++
    [mle tr.linear.blockPoint v.omegaA, mle tr.late.blockPoint v.omegaA]

end At

/-! ## Opens -/

/-- The replay's query indices are the model's. -/
theorem queries_toList {rel : Replay.Relation} {proof : Replay.Proof} {early : Replay.EarlyOut}
    {lin : Replay.LinearOut} {honest : Replay.HonestOut} (run : HonestRun rel proof early lin honest)
    (setup : 3 ≤ rel.setupVars) :
    honest.queries.toList = (List.finRange rel.queries).map fun j =>
      ((lowBits (2 ^ (rel.setupVars - 2)) (wordsOf rel.queries honest.words) j : Fin _) : ℕ) := by
  rw [run.queries]
  apply List.ext_getElem
  · simp [run.words]
  · intro i h1 h2
    have lt : i < honest.words.size := by simpa [run.words] using h2
    simp only [Array.toList_map, List.getElem_map, Array.getElem_toList, List.getElem_finRange, lowBits, lowBit,
      wordsOf, Fin.val_cast]
    have bits : rel.setupVars + 1 - foldVars = rel.setupVars - 2 := by simp only [foldVars]; omega
    rw [getElem!_pos honest.words i lt]
    simp only [bits]
    rfl

/-- The P0 values at the verifier's points are the values at the model
transcript's points. -/
theorem p0Opened_eq (rel : Replay.Relation) (st : Replay.Statement) (proof : Replay.Proof)
    (cands : List (P0Tables Fp)) (p1 : List (P1Tables K)) {early : Replay.EarlyOut} {mc : Layer1.GkrClaims K}
    (agree : ClaimsAgree rel.earlyShapes.size early.gkr mc)
    (openings : early.openings = earlyOpeningsOf rel.key.layout early.challenges early.gkr proof.earlyValues)
    (lin : Replay.LinearOut) (honest : Replay.HonestOut) {late : Replay.LateOut} {mcL : Layer1.GkrClaims K}
    (agreeL : ClaimsAgree rel.lateShapes.size late.gkr mcL)
    (lateOpenings : late.openings = lateOpeningsOf rel.key.layout late.beta late.gkr) (w : P0Tables Fp) :
    p0Opened (shapeOf rel) (openingPoints rel early lin honest late) w =
      p0At (shapeOf rel) (transcriptOf rel st proof cands p1 early mc lin honest late mcL) w := by
  have small : ∀ u, u < 5 → u < rel.earlyShapes.size := fun u lt => by rw [earlyShapes_size]; exact lt
  have smallL : ∀ u, u < 2 → u < rel.lateShapes.size := fun u lt => by rw [lateShapes_size]; exact lt
  set pts := openingPoints rel early lin honest late
  set tr := transcriptOf rel st proof cands p1 early mc lin honest late mcL
  have norm : pointK (K := K) pts.norm = tr.early.normPoint := (agree.leafPoint 0 (small 0 (by norm_num))).symm
  have linear : pointK (K := K) pts.linear = tr.linear.lanePart ++ tr.linear.blockPoint := pointK_append _ _
  have run : pointK (K := K) pts.run = tr.early.runPoint := by
    show pointK early.openings.runPoint = _
    rw [openings]
    exact (agree.leafPoint 3 (small 3 (by norm_num))).symm
  have pairs : pointK (K := K) pts.pairs = tr.early.pairPoint.take (shapeOf rel).slotVars := by
    show pointK early.openings.pairPoint = _
    rw [openings]
    show pointK ((early.gkr.leafPoint[4]!).extract 0 rel.key.layout.slotVariables) = _
    rw [pointK_extract_take, ← agree.leafPoint 4 (small 4 (by norm_num))]
    rfl
  have slots : pointK (K := K) pts.slots = tr.early.runPoint.take (shapeOf rel).slotVars := by
    show pointK (early.openings.runPoint.extract 0 rel.key.layout.slotVariables) = _
    rw [pointK_extract_take, openings]
    show (pointK (early.gkr.leafPoint[3]!)).take _ = _
    rw [← agree.leafPoint 3 (small 3 (by norm_num))]
    rfl
  have latePairs : pointK (K := K) pts.latePairs = tr.late.slotPoint (shapeOf rel) := by
    show pointK late.openings.pairPoint = _
    rw [lateOpenings]
    show pointK ((late.gkr.leafPoint[0]!).extract 0 rel.key.layout.slotVariables) = _
    rw [pointK_extract_take, ← agreeL.leafPoint 0 (smallL 0 (by norm_num))]
    rfl
  have rows : pointK (K := K) pts.rows = tr.early.rowPoint := by
    show pointK early.openings.rowPoint = _
    rw [openings]
    exact (agree.leafPoint 2 (small 2 (by norm_num))).symm
  simp only [p0Opened, p0At, norm, linear, run, pairs, slots, latePairs, rows]

/-- The P1 values at the verifier's points are the values at the model
transcript's points. -/
theorem p1Opened_eq (rel : Replay.Relation) (st : Replay.Statement) (proof : Replay.Proof) (setup : 3 ≤ rel.setupVars)
    (cands : List (P0Tables Fp)) (p1 : List (P1Tables K)) {early : Replay.EarlyOut} {mc : Layer1.GkrClaims K}
    (agree : ClaimsAgree rel.earlyShapes.size early.gkr mc)
    (openings : early.openings = earlyOpeningsOf rel.key.layout early.challenges early.gkr proof.earlyValues)
    {lin : Replay.LinearOut} (linRun : LinearRun (K := K) rel proof lin)
    {honest : Replay.HonestOut} (honestRun : HonestRun rel proof early lin honest)
    {late : Replay.LateOut} {mcL : Layer1.GkrClaims K} (agreeL : ClaimsAgree rel.lateShapes.size late.gkr mcL)
    (lateOpenings : late.openings = lateOpeningsOf rel.key.layout late.beta late.gkr) (v : P1Tables K) :
    p1Opened (openingPoints rel early lin honest late) v =
      p1At (shapeOf rel) (queryAt rel) (transcriptOf rel st proof cands p1 early mc lin honest late mcL) v := by
  have small : ∀ u, u < 5 → u < rel.earlyShapes.size := fun u lt => by rw [earlyShapes_size]; exact lt
  have smallL : ∀ u, u < 2 → u < rel.lateShapes.size := fun u lt => by rw [lateShapes_size]; exact lt
  set pts := openingPoints rel early lin honest late
  set tr := transcriptOf rel st proof cands p1 early mc lin honest late mcL
  have keyTail : pointK (K := K) pts.keyTail = (keyPoint (shapeOf rel) tr.linear).drop 3 := by
    show pointK ((Replay.padded rel.setupVars (lin.lanePoint ++ lin.blockPoint)).extract foldVars _) = _
    rw [pointK_extract_drop, pointK_padded, pointK_append, linRun.lanePoint]
    rfl
  have runTail : pointK (K := K) pts.runTail = (Layer1.padded (shapeOf rel).setupVars tr.early.runPoint).drop 3 := by
    show pointK ((Replay.padded rel.setupVars early.openings.runPoint).extract foldVars _) = _
    rw [pointK_extract_drop, pointK_padded, openings]
    show (Layer1.padded rel.setupVars (pointK (early.gkr.leafPoint[3]!))).drop foldVars = _
    rw [← agree.leafPoint 3 (small 3 (by norm_num))]
    rfl
  have queries : pts.queries.toList.flatMap (fun pt => [mle (pointK pt) v.foldKey, mle (pointK pt) v.foldRun]) =
      (List.finRange (shapeOf rel).queries).flatMap (fun j =>
        [mle (powerPoint (queryAt rel (tr.honest.queries j)) ((shapeOf rel).setupVars - 3)) v.foldKey,
          mle (powerPoint (queryAt rel (tr.honest.queries j)) ((shapeOf rel).setupVars - 3)) v.foldRun]) := by
    show (honest.queries.map (leafQueryPoint rel.setupVars)).toList.flatMap _ = _
    rw [Array.toList_map, List.flatMap_map, queries_toList honestRun setup, List.flatMap_map]
    apply List.flatMap_congr
    intro j _
    rw [pointK_leafQueryPoint _ _ (lowBits (2 ^ (rel.setupVars - 2)) (wordsOf rel.queries honest.words) j).isLt
      (by omega)]
    rfl
  have block : pointK (K := K) pts.block = tr.linear.blockPoint := rfl
  have lateBlock : pointK (K := K) pts.lateBlock = tr.late.blockPoint := by
    show pointK late.openings.blockPoint = _
    rw [lateOpenings]
    exact (agreeL.leafPoint 1 (smallL 1 (by norm_num))).symm
  simp only [p1Opened, p1At, keyTail, runTail, queries, block, lateBlock]

/-- The replay's opening checks, under WHIR at the points that the replay
returns, give the model's `Opens` at the model's transcript of the run. -/
theorem opens_of_replay (rel : Replay.Relation) (st : Replay.Statement) (proof : Replay.Proof)
    (shape : ProofShape rel proof) (S : Setup Fp) (merkle : MerkleBinds rel S proof.leaves) (setup : 3 ≤ rel.setupVars)
    (high : rel.setupVars + 1 ≤ 32) (cands : List (P0Tables Fp)) (p1 : List (P1Tables K))
    {early : Replay.EarlyOut} {mc : Layer1.GkrClaims K} (agree : ClaimsAgree rel.earlyShapes.size early.gkr mc)
    (values : early.values = proof.earlyValues)
    (zLeaf : early.zLeaf = early.challenges.beta - early.gkr.leafValue 0 0)
    (openings : early.openings = earlyOpeningsOf rel.key.layout early.challenges early.gkr proof.earlyValues)
    {lin : Replay.LinearOut} (linRun : LinearRun (K := K) rel proof lin)
    {honest : Replay.HonestOut} (honestRun : HonestRun rel proof early lin honest)
    {late : Replay.LateOut} {mcL : Layer1.GkrClaims K} (agreeL : ClaimsAgree rel.lateShapes.size late.gkr mcL)
    (latePairs : late.pairs = proof.lateValues)
    (lateOpenings : late.openings = lateOpeningsOf rel.key.layout late.beta late.gkr)
    (p1ok : openingsAgree #[p1Values proof.p1Opening.evals] #[p1Expected proof honest lin late] "P1 openings" =
      .ok ())
    (p0ok : openingsAgree proof.p0Opening.evals (p0Expected early lin late) "P0 openings" = .ok ())
    (whir : WhirOpens (shapeOf rel) proof (openingPoints rel early lin honest late) cands p1) :
    Opens (shapeOf rel) S (queryAt rel) (transcriptOf rel st proof cands p1 early mc lin honest late mcL) := by
  obtain ⟨w, wm, v, vm, whir0, whir1⟩ := whir
  rw [p0Opened_eq rel st proof cands p1 agree openings lin honest agreeL lateOpenings] at whir0
  rw [p1Opened_eq rel st proof setup cands p1 agree openings linRun honestRun agreeL lateOpenings] at whir1
  have small : ∀ u, u < 5 → u < rel.earlyShapes.size := fun u lt => by rw [earlyShapes_size]; exact lt
  have smallL : ∀ u, u < 2 → u < rel.lateShapes.size := fun u lt => by rw [lateShapes_size]; exact lt
  -- The P0 batches.
  have p0 : ((p0Expected early lin late).toList.map Array.toList).map (List.map (Ext.toK (K := K))) =
      p0At (shapeOf rel) (transcriptOf rel st proof cands p1 early mc lin honest late mcL) w := by
    rw [← openingsAgree_ok p0ok, List.map_map]
    exact whir0
  simp only [p0Expected, p0At, Array.toList_append, List.map_cons, List.map_nil, List.map_append,
    toList_flatPairs, List.map_flatMap, List.cons.injEq, and_true] at p0
  obtain ⟨norm, linear, ⟨eRe, eIm⟩, earlyPairs, block, latePart, mult⟩ := p0
  have pairSize : early.values.pairs.size = rel.matrices := by rw [values]; exact shape.pairs
  rw [pairSize] at earlyPairs
  rw [latePairs, shape.lateValues] at latePart
  obtain ⟨latePairsEq, lateBlock⟩ := List.append_inj' latePart (by simp)
  -- The P1 values.
  have agree1 := openingsAgree_ok p1ok
  simp only [List.map_cons, List.map_nil, List.cons.injEq, and_true] at agree1
  have p1v : (p1Expected proof honest lin late).toList.map (Ext.toK (K := K)) =
      p1At (shapeOf rel) (queryAt rel) (transcriptOf rel st proof cands p1 early mc lin honest late mcL) v := by
    rw [← agree1]
    exact whir1
  simp only [p1Expected, p1At, Array.toList_append, List.map_append, List.map_cons, List.map_nil] at p1v
  obtain ⟨front, ends⟩ := List.append_inj' p1v (by simp)
  obtain ⟨folds, middle⟩ := List.append_inj front (by simp)
  simp only [List.cons.injEq, and_true] at ends folds
  obtain ⟨omegaLinear, omegaLate⟩ := ends
  obtain ⟨keyFold, runFold⟩ := folds
  -- The queried values are the honest folds.
  have alphaSize : honest.alpha.size = foldVars := by rw [honestRun.alpha]; rfl
  have queriedRun := honestRun.queried
  rw [linRun.mixing] at queriedRun
  have leavesSize : honest.queries.size ≤ proof.leaves.size := by
    rw [honestRun.queries, Array.size_map, honestRun.words, shape.leaves]
  have inside : ∀ pair ∈ (honest.queries.zip proof.leaves).toList,
      pair.1 < 2 ^ (rel.setupVars - 2) ∧ pair.2 ∈ proof.leaves.toList := by
    rintro ⟨index, leaf⟩ member
    rw [Array.toList_zip] at member
    obtain ⟨queriesMember, leavesMember⟩ := List.of_mem_zip member
    refine ⟨?_, leavesMember⟩
    rw [honestRun.queries, Array.toList_map] at queriesMember
    obtain ⟨word, _, rfl⟩ := List.mem_map.mp queriesMember
    rw [show rel.setupVars + 1 - foldVars = rel.setupVars - 2 by simp only [foldVars]; omega]
    exact Nat.mod_lt _ (by positivity)
  have queried := queried_toK (K := K) (fun i leaf => setupCheck rel.key.root rel.setupVars i leaf
      #[(Array.range rel.kappa).map (Mixing.commitment ⟨lin.lam, rel.kappa, rel.matrices⟩),
        (Array.range runOracles).map honest.mu.pow] honest.alpha)
    (fun i => (univ (rel.setupVars - 3) (fold (pointK honest.alpha)
      (keyGroup (shapeOf rel) S lin.lam.toK))).eval (queryAt rel i))
    (fun i => (univ (rel.setupVars - 3) (fold (pointK honest.alpha) (runGroup S honest.mu.toK))).eval (queryAt rel i))
    _ #[] honest.queried (fun pair member _ ok => setupCheck_toList rel S merkle (inside pair member).1
      (inside pair member).2 lin.lam honest.mu alphaSize setup high ok) queriedRun
  rw [Array.toList_zip, List.map_fst_zip (by simpa using leavesSize), queries_toList honestRun setup,
    List.flatMap_map] at queried
  simp only [List.map_nil, List.nil_append] at queried
  rw [queried] at middle
  have pointwise := flatMap_pairs (List.finRange rel.queries) middle
  have partialSize : ∀ g, g < 2 → (proof.partials[g]!).size = 2 ^ foldVars := fun g lt =>
    shape.partialSizes _ (by
      rw [getElem!_pos proof.partials g (by rw [shape.partials]; exact lt)]
      exact Array.getElem_mem_toList _)
  unfold Opens EarlyConsistent LateConsistent HonestConsistent
  refine ⟨w, wm, v, vm, ⟨?_, eRe.symm, eIm.symm, fun j lt => ?_, block.symm, ?_⟩, ⟨fun j lt => ?_, ?_, ?_⟩, linear,
    ⟨?_, ?_, fun j => ?_⟩, omegaLinear⟩
  · rw [← norm, zLeaf, Ext.toK_sub, ← agree.leafValues 0 (small 0 (by norm_num)) 0]
    rfl
  · obtain ⟨re, im⟩ := flatMap_pairs (List.range rel.matrices) earlyPairs j (List.mem_range.mpr lt)
    exact ⟨re.symm, im.symm⟩
  · rw [← mult, openings]
    exact (agree.leafValues 2 (small 2 (by norm_num)) 0).symm
  · obtain ⟨re, im⟩ := flatMap_pairs (List.range rel.matrices) latePairsEq j (List.mem_range.mpr lt)
    exact ⟨re.symm, im.symm⟩
  · rw [← (List.cons.inj lateBlock).1, lateOpenings]
    show (late.beta - late.gkr.leafValue 0 2 - (late.gkr.leafPoint[0]!)[rel.key.layout.slotVariables]!).toK = _
    rw [Ext.toK_sub, Ext.toK_sub, pointK_getD, ← agreeL.leafPoint 0 (smallL 0 (by norm_num)),
      ← agreeL.leafValues 0 (smallL 0 (by norm_num)) 2]
    rfl
  · rw [← omegaLate, lateOpenings]
    exact (agreeL.leafValues 1 (smallL 1 (by norm_num)) 0).symm
  · rw [← keyFold, toK_combine _ _ (by rw [partialSize 0 (by norm_num), alphaSize])]
    rfl
  · rw [← runFold, toK_combine _ _ (by rw [partialSize 1 (by norm_num), alphaSize])]
    rfl
  · obtain ⟨key, run⟩ := pointwise j (List.mem_finRange j)
    exact ⟨key.symm, run.symm⟩

end CompressionLean.Bridge
