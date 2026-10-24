import CompressionLean.Bridge.Linear

/-!
Owns the bridge of the late stage (`Replay.lateStage`) to the model's
`lateGame`. The replay's lane split is the model's head and tail
(`toK_head`, `toK_tail`), and its slot step over the key's segments is the
extension of the model's lane split (`toK_slotStep`), under the key
structure facts of `KeyStructure.lean`. A successful late replay is a path
of `lateGame` that draws the replay's `β` and GKR challenges and ends in the
model's `LateOut` of the replay's values, whose `LateChecks` hold
(`late_reaches`).
-/

namespace CompressionLean.Bridge

open CompressionLean.Lib
open Polynomial
open CompressionLean.Replay CompressionLean.Layer1

variable {K : Type} [Field K] [Algebra Fp K] [Fintype K] [ChallengeField K]

local notation "ι" => algebraMap Fp K

/-! ## The lane split -/

theorem toK_headFrom (tau : Array Ext) (length : ℕ) (ratio : Gl) (d : ℕ) :
    (Replay.headFrom tau length ratio (ratio.pow length) d).toK (K := K) =
      Layer1.headFrom D length (ι ratio.toF) (tableK tau) d := by
  induction d with
  | zero => rfl
  | succ d ih =>
      rw [Replay.headFrom, Layer1.headFrom]
      split_ifs
      · rw [Ext.toK_sub, Ext.toK_add, Ext.toK_scale, Ext.toK_scale, ih, Gl.toF_pow, map_pow]
        simp only [tableK]
        ring
      · rw [Ext.toK_add, Ext.toK_scale, ih]
        simp only [tableK]
        ring

theorem toK_head (tau : Array Ext) (length : ℕ) (ratio : Gl) {lane : ℕ} (inside : lane < D) :
    (((Replay.laneSplit tau length ratio).1)[lane]!).toK (K := K) =
      headValue D length (ι ratio.toF) (tableK tau) lane := by
  simp only [Replay.laneSplit]
  rw [getElem!_map_range _ _ inside, toK_headFrom]
  rfl

theorem toK_tail (tau : Array Ext) (length : ℕ) (ratio : Gl) {lane : ℕ} (inside : lane < D) :
    (((Replay.laneSplit tau length ratio).2)[lane]!).toK (K := K) =
      tailValue D length (ι ratio.toF) (tableK tau) lane := by
  simp only [Replay.laneSplit]
  rw [getElem!_map_range _ _ inside, tailValue]
  split_ifs with long
  · rw [Ext.toK_scale, prefixSum, toK_sumRange, Gl.toF_pow, map_pow]
    congr 1
    apply Finset.sum_congr rfl
    intro i _
    rw [Ext.toK_scale, Gl.toF_pow, map_pow]
    rfl
  · exact toK_default

/-! ## The slot step -/

theorem toList_map_range {α β : Type} [Inhabited α] (a : Array α) (f : α → β) :
    a.toList.map f = (List.range a.size).map fun i => f a[i]! := by
  apply List.ext_getElem (by simp)
  intro i h1 h2
  simp only [List.getElem_map, Array.getElem_toList, List.getElem_range]
  rw [getElem!_pos a i (by simpa using h1)]

section Key

variable (rel : Replay.Relation) (S : Setup Fp)

/-- Each segment's lane is a lane of the ring: its first slot has it. -/
theorem segment_lane (fits : S.Fits (shapeOf rel)) (checked : (structureOf rel.key.layout).Checked (shapeOf rel))
    (describes : (structureOf rel.key.layout).Describes (shapeOf rel) S) {i : ℕ}
    (inside : i < rel.key.layout.segments.size) : (rel.key.layout.segments[i]!).lane < D := by
  have start := segmentStart_mono (shapeOf rel) (structureOf rel.key.layout) checked i inside
  have strict : (structureOf rel.key.layout).segmentStart i < (structureOf rel.key.layout).segmentStart (i + 1) := by
    have inside' : i < (structureOf rel.key.layout).segments := inside
    unfold Structure.segmentStart
    rw [if_pos inside']
    split_ifs with next
    · exact checked.segment_mono i next
    · exact checked.segment_last i (by simp [structureOf] at next ⊢; omega)
  obtain ⟨_, _, lane⟩ := describes.slot_shape i _ inside le_rfl strict
  have := fits.lane_lt ((structureOf rel.key.layout).segmentStart i)
  rw [lane] at this
  exact this

/-- The replay's slot step over the key's segments is the extension of the
model's lane split at part `t`. -/
theorem toK_slotStep (fits : S.Fits (shapeOf rel)) (checked : (structureOf rel.key.layout).Checked (shapeOf rel))
    (describes : (structureOf rel.key.layout).Describes (shapeOf rel) S) (tau : Array Ext) (point : Array Ext)
    (t : ℕ) (part : t < 2) :
    (slotStep rel.key.layout point (rel.key.layout.classes.map fun c => Replay.laneSplit tau c.len c.ratio) t).toK
        (K := K) = mle (pointK point) (laneSplit (shapeOf rel) S (tableK tau) t) := by
  unfold slotStep
  rw [toK_step, List.map_map, toList_map_range]
  have value : ∀ i, i < rel.key.layout.segments.size →
      ((fun s : ℕ × Ext => (s.1, s.2.toK (K := K))) ∘ fun seg : Segment =>
        (seg.first, (if t = 0 then (rel.key.layout.classes.map fun c => Replay.laneSplit tau c.len c.ratio)[seg.classIndex]!.1
          else (rel.key.layout.classes.map fun c => Replay.laneSplit tau c.len c.ratio)[seg.classIndex]!.2)[seg.lane]!))
        rel.key.layout.segments[i]! =
      ((structureOf rel.key.layout).segmentFirst i,
        (if t = 0 then headValue else tailValue) (shapeOf rel).D
          ((structureOf rel.key.layout).classLength ((structureOf rel.key.layout).segmentClass i))
          (ι ((structureOf rel.key.layout).classRatio ((structureOf rel.key.layout).segmentClass i))) (tableK tau)
          ((structureOf rel.key.layout).segmentLane i)) := by
    intro i inside
    have named := checked.class_named i inside
    simp only [structureOf] at named
    have lane := segment_lane rel S fits checked describes inside
    simp only [Function.comp, structureOf]
    rw [getElem!_pos (rel.key.layout.classes.map _) _ (by simpa using named)]
    simp only [Array.getElem_map]
    rw [← getElem!_pos rel.key.layout.classes _ named]
    congr 1
    split_ifs
    · exact toK_head tau _ _ lane
    · exact toK_tail tau _ _ lane
  rw [List.map_congr_left fun i member => value i (List.mem_range.mp member)]
  rcases (show t = 0 ∨ t = 1 by omega) with rfl | rfl
  · simp only [if_true]
    exact slotStep_head (shapeOf rel) S (structureOf rel.key.layout) fits checked describes (tableK tau) _
  · simp only [show (1 : ℕ) ≠ 0 by omega, if_false]
    exact slotStep_tail (shapeOf rel) S (structureOf rel.key.layout) fits checked describes (tableK tau) _

end Key

/-! ## The stage -/

/-- The openings `checkLate` returns. -/
def lateOpeningsOf (layout : Layout) (beta : Ext) (claims : Replay.GkrClaims) : LateOpenings where
  pairPoint := (claims.leafPoint[0]!).extract 0 layout.slotVariables
  block := beta - claims.leafValue 0 2 - (claims.leafPoint[0]!)[layout.slotVariables]!
  blockPoint := claims.leafPoint[1]!
  omega := claims.leafValue 1 0

/-- What `checkLate` checked and returned. -/
structure LateChecked (layout : Layout) (beta : Ext) (m : Mixing) (lanes : Array (Array Ext × Array Ext))
    (claims : Replay.GkrClaims) (pairs : Array (Ext × Ext)) (op : LateOpenings) : Prop where
  pair : pairCheck claims.roots[0]! claims.roots[1]! "block scatter" = .ok ()
  ubarLeaf : claims.leafValue 0 0 =
    sumRange pairs.size fun j => (m.evalA j).1 * (pairs[j]!).1 + (m.evalA j).2 * (pairs[j]!).2
  laneLeaf : claims.leafValue 0 1 =
    oneMinus (claims.leafPoint[0]!)[layout.slotVariables]! *
        slotStep layout ((claims.leafPoint[0]!).extract 0 layout.slotVariables) lanes 0 +
      (claims.leafPoint[0]!)[layout.slotVariables]! *
        slotStep layout ((claims.leafPoint[0]!).extract 0 layout.slotVariables) lanes 1
  blockLeaf : claims.leafValue 1 1 = beta - Replay.identityValue claims.leafPoint[1]!
  openings : op = lateOpeningsOf layout beta claims

theorem checkLate_ok {layout : Layout} {beta : Ext} {m : Mixing} {lanes : Array (Array Ext × Array Ext)}
    {claims : Replay.GkrClaims} {pairs : Array (Ext × Ext)} {op : LateOpenings}
    (ok : checkLate layout beta m lanes claims pairs = .ok op) : LateChecked layout beta m lanes claims pairs op := by
  unfold checkLate at ok
  obtain ⟨_, pair, ok⟩ := Except.bind_ok ok
  obtain ⟨_, ubarLeaf, ok⟩ := Except.bind_ok ok
  obtain ⟨_, laneLeaf, ok⟩ := Except.bind_ok ok
  obtain ⟨_, blockLeaf, ok⟩ := Except.bind_ok ok
  exact ⟨pair, Ext.eq_of_beq (Except.ensure_ok ubarLeaf), Ext.eq_of_beq (Except.ensure_ok laneLeaf),
    Ext.eq_of_beq (Except.ensure_ok blockLeaf), (Except.pure_eq_ok ok).symm⟩

theorem lateShapes_size (rel : Replay.Relation) : rel.lateShapes.size = 2 := rfl

theorem lateShapes_agree (rel : Replay.Relation) : ShapesAgree (lateShape (shapeOf rel)) rel.lateShapes where
  trees := rfl
  depth t inside := by
    rw [lateShapes_size] at inside
    interval_cases t <;> simp [lateShape, Replay.Relation.lateShapes, shapeOf]
  factors t inside := by
    rw [lateShapes_size] at inside
    interval_cases t <;> simp [lateShape, Replay.Relation.lateShapes]

theorem pointK_extract_take (p : Array Ext) (n : ℕ) : pointK (K := K) (p.extract 0 n) = (pointK p).take n := by
  simp [pointK, Array.toList_extract, List.extract_eq_take_drop, List.map_take]

variable [DecidableEq K]

/-- The model's messages of the replay's late stage: the GKR messages and
the late values. -/
def lateMsgs (rel : Replay.Relation) (proof : Replay.Proof) : List (Σ M : Type, M) :=
  gkrMsgs (K := K) rel.lateShapes proof.late ++ [⟨ℕ → K × K, fun j => pairK (proof.lateValues[j]!)⟩]

/-- A successful late replay is a path of `lateGame` that receives the
proof's messages, draws the replay's `β` and GKR challenges and ends in the
model's `LateOut` of the replay's values; along it every `LateChecks`
condition holds. -/
theorem late_reaches (rel : Replay.Relation) (proof : Replay.Proof) (shape : ProofShape rel proof) (S : Setup Fp)
    (fits : S.Fits (shapeOf rel)) (checked : (structureOf rel.key.layout).Checked (shapeOf rel))
    (describes : (structureOf rel.key.layout).Describes (shapeOf rel) S) {lin : Replay.LinearOut}
    (linRun : LinearRun (K := K) rel proof lin) (cands : List (P0Tables Fp)) (p1 : List (P1Tables K)) (q : ℕ)
    {t t' : Replay.Transcript} {late : Replay.LateOut} (run : Runs (lateStage rel proof lin) t late t') :
    ∃ ds : List Draw, t'.draws.toList = t.draws.toList ++ ds ∧ ∃ mc : Layer1.GkrClaims K,
      ClaimsAgree rel.lateShapes.size late.gkr mc ∧
      (∀ u, u < rel.lateShapes.size → (late.gkr.leafPoint[u]!).size = (rel.lateShapes[u]!).depth) ∧
      late.pairs = proof.lateValues ∧ late.openings = lateOpeningsOf rel.key.layout late.beta late.gkr ∧
      Game.Reaches (lateGame (shapeOf rel) S lin.lam.toK (laneWeight (shapeOf rel) barOf lin.zeta.toK) cands p1)
        (lateMsgs (K := K) rel proof) (ds.map (drawn (K := K) q))
        (some ⟨late.beta.toK, mc, fun j => pairK (proof.lateValues[j]!)⟩) := by
  unfold lateStage at run
  obtain ⟨beta, t1, h1, run⟩ := run_bind run
  obtain ⟨claims, t2, h2, run⟩ := run_bind run
  obtain ⟨openings, t3, h3, run⟩ := run_bind run
  obtain ⟨_, t4, h4, run⟩ := run_bind run
  obtain ⟨rfl, rfl⟩ := run_pure run
  obtain ⟨checkOk, rfl⟩ := run_liftExcept h3
  have checked' := checkLate_ok checkOk
  obtain ⟨dsG, drawsG, mc, claimsAgree, lengths, reachG⟩ := gkr_reaches (K := K) (lateShapes_agree rel)
    (Finset.univ : Finset (Fin cands.length × Fin p1.length))
    (fun w => lateBatch (shapeOf rel) S lin.lam.toK (laneWeight (shapeOf rel) barOf lin.zeta.toK) beta.toK
      (cands.get w.1) (p1.get w.2)) (fun _ _ => ∅) q shape.late h2
  have small : ∀ u, u < 2 → u < rel.lateShapes.size := fun u lt => by rw [lateShapes_size]; exact lt
  have leafValue : ∀ u j, u < 2 → mc.leafValues u j = (claims.leafValue u j).toK :=
    fun u j lt => claimsAgree.leafValues u (small u lt) j
  have leafPoint : ∀ u, u < 2 → mc.leafPoint u = pointK (claims.leafPoint[u]!) :=
    fun u lt => claimsAgree.leafPoint u (small u lt)
  have root : ∀ u, u < 2 → mc.roots u = pairK (claims.roots[u]!) := fun u lt => claimsAgree.roots u (small u lt)
  have tau : tableK (K := K) lin.tau = laneWeight (shapeOf rel) barOf lin.zeta.toK := by
    funext l
    rw [linRun.tau, tableK_tau (shapeOf rel) rfl]
  have checks : LateChecks (shapeOf rel) S lin.lam.toK (laneWeight (shapeOf rel) barOf lin.zeta.toK)
      (⟨beta.toK, mc, fun j => pairK (proof.lateValues[j]!)⟩ : Layer1.LateOut K) := by
    refine ⟨?_, ?_, ?_, ?_⟩
    · rw [root 0 (by norm_num), root 1 (by norm_num)]
      exact pairCheck_ok checked'.pair
    · rw [leafValue 0 0 (by norm_num), checked'.ubarLeaf, toK_sumRange, shape.lateValues, linRun.mixing]
      apply Finset.sum_congr rfl
      intro j _
      rw [Ext.toK_add, Ext.toK_mul, Ext.toK_mul]
      simp only [Mixing.evalA, Ext.toK_pow, pairK, Layer1.Shape.evalARow, shapeOf]
      ring
    · rw [leafValue 0 1 (by norm_num), checked'.laneLeaf, Ext.toK_add, Ext.toK_mul, Ext.toK_mul, toK_oneMinus,
        toK_slotStep rel S fits checked describes lin.tau _ 0 (by norm_num),
        toK_slotStep rel S fits checked describes lin.tau _ 1 (by norm_num), tau, pointK_extract_take,
        pointK_getD]
      simp only [Layer1.LateOut.top, Layer1.LateOut.slotPoint, leafPoint 0 (by norm_num)]
      rfl
    · rw [leafValue 1 1 (by norm_num), checked'.blockLeaf, Ext.toK_sub, toK_identity, Layer1.LateOut.blockPoint,
        leafPoint 1 (by norm_num)]
  refine ⟨.ext beta :: dsG, ?_, mc, claimsAgree, lengths, rfl, checked'.openings, ?_⟩
  · rw [run_observeExts h4, drawsG, run_sampleExt h1]
    simp
  · unfold lateGame lateMsgs
    rw [List.map_cons, drawn_ext]
    apply Game.Reaches.draw beta.toK
    refine Game.Reaches.congr (Game.Reaches.bind reachG ?_) rfl (List.append_nil _)
    apply Game.Reaches.recv (fun j => pairK (proof.lateValues[j]!))
    rw [if_pos checks]
    exact Game.Reaches.pure _

end CompressionLean.Bridge
