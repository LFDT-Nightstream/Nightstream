import CompressionLean.Sim.Early
import CompressionLean.Sim.Honest
import CompressionLean.Bridge.Late

/-!
Owns the simulation of the late stage of `Late.lean` (`verifier.rs::verify`)
against `Replay.lateStage`: `lateStage_tracks` tracks the replay's run, and
`IsLateOut` gives what the openings read.

The circuit builds the lane tables (`mle.rs::lane_split`, `matrix.rs::Structure::lane_tables`)
with loops; the replay states each entry in closed form (`headFrom`,
`prefixSum`). They agree when `τ` has `D` lanes and every class has at most
`D` lanes. The slot steps (`matrix.rs::Structure::slot_step`) need every segment to name a
class and a lane. `Structure::check` enforces both (`matrix.rs::Structure::check`). The
leaf checks (`matrix.rs::Structure::check_late`) read three leaf values and one point
coordinate of the pair tree and two leaf values of the block tree; the
replay's GKR run gives these sizes.
-/

namespace CompressionLean.Sim

open CompressionLean.Circuit
open CompressionLean.Lib
open CompressionLean.Replay (Word Ext Relation TreeShape D)
open NightstreamFPrime.Spec (goldilocksModulus)

variable {F : Type} [CommRing F]

/-! ## Instruction shape -/

instance Mle.laneSplit_appends (tau : Array ExtVar) (length : ℕ) (ratio : Word) :
    Appends (Mle.laneSplit tau length ratio) := by
  unfold Mle.laneSplit; appends
instance laneTables_appends (layout : Replay.Layout) (tau : Array ExtVar) : Appends (laneTables layout tau) := by
  unfold laneTables; appends
instance slotStep_appends (layout : Replay.Layout) (point : Array ExtVar) (lanes : Array (Array ExtVar × Array ExtVar))
    (t : ℕ) : Appends (slotStep layout point lanes t) := by
  unfold slotStep; appends
instance checkLate_appends (layout : Replay.Layout) (beta : ExtVar) (evalA : Array (ExtVar × ExtVar))
    (lanes : Array (Array ExtVar × Array ExtVar)) (claims : Array TreeClaimView) (pairs : Array (ExtVar × ExtVar)) :
    Appends (checkLate layout beta evalA lanes claims pairs) := by
  unfold checkLate; appends
instance lateStage_appends (rel : Relation) (tau : Array ExtVar) (evalA : Array (ExtVar × ExtVar)) (late : GkrView)
    (lateValues : Array (ExtVar × ExtVar)) (d : Duplex) :
    Appends ((Circuit.lateStage rel tau evalA late lateValues).run d) := by
  unfold Circuit.lateStage; appends

/-! ## Lane tables -/

private theorem isExts_table {c : Ctx F} {m k : ℕ} {xs : Array ExtVar} {f : ℕ → Ext} (size : xs.size = k)
    (h : ∀ i < k, IsExt c m xs[i]! (f i)) : IsExts c m xs ((Array.range k).map f) :=
  ⟨by simp [size], fun i hi => by rw [getElem!_map_range k f (size ▸ hi)]; exact h i (size ▸ hi)⟩

private def IsLanes (c : Ctx F) (m : ℕ) (lanes : Array (Array ExtVar × Array ExtVar))
    (glanes : Array (Array Ext × Array Ext)) : Prop :=
  lanes.size = glanes.size ∧
    ∀ i < lanes.size, IsExts c m lanes[i]!.1 glanes[i]!.1 ∧ IsExts c m lanes[i]!.2 glanes[i]!.2

private theorem IsLanes.mono {c : Ctx F} {n m : ℕ} {lanes : Array (Array ExtVar × Array ExtVar)}
    {glanes : Array (Array Ext × Array Ext)} (h : IsLanes c n lanes glanes) (le : n ≤ m := by omega) :
    IsLanes c m lanes glanes :=
  ⟨h.1, fun i hi => ⟨(h.2 i hi).1.mono le, (h.2 i hi).2.mono le⟩⟩

section Lanes

variable [CharP F goldilocksModulus] {c : Ctx F} {n : ℕ}

/-- `lane_split` (`mle.rs::lane_split`): the head and tail lane tables in the
replay's closed form. -/
private theorem laneSplit_ensures {tau : Array ExtVar} {gtau : Array Ext} {length : ℕ} (ratio : Word)
    (ht : IsExts c n tau gtau) (size : gtau.size = D) (short : length ≤ D) :
    Ensures c n (Mle.laneSplit tau length ratio) fun r m =>
      IsExts c m r.1 (Replay.laneSplit gtau length ratio).1 ∧ IsExts c m r.2 (Replay.laneSplit gtau length ratio).2 := by
  set H : ℕ → Ext := Replay.headFrom gtau length ratio (ratio.pow length) with hH
  set S : ℕ → Ext := Replay.prefixSum gtau ratio with hS
  have hD : 1 < D := by decide
  have hget : ∀ {i}, i < D → IsExt c n tau[i]! gtau[i]! := fun hi => ht.get (by rw [size]; exact hi)
  unfold Mle.laneSplit
  dsimp only
  refine (Ensures.loop (fun k (head : Array ExtVar) m => head.size = D ∧
      ∀ l, D - 1 - k ≤ l → l < D → IsExt c m head[l]! (H (D - 1 - l))) ⟨by simp, fun l lo hi => ?_⟩
    fun k hk head m _ hhead => ?_).bind fun head m₁ _ hhead => ?_
  · have e : l = D - 1 := by omega
    subst e
    rw [getElem!_pos _ _ (by simp; omega), Array.getElem_replicate, Nat.sub_self, hH, Replay.headFrom]
    exact hget (by omega)
  · obtain ⟨hs, hh⟩ := hhead
    have hk' : k < D - 1 := by simpa using hk
    have xk : (List.range (D - 1)).reverse[k]'hk = D - 2 - k := by
      rw [List.getElem_reverse, List.getElem_range]
      simp only [List.length_range]
      omega
    rw [xk]
    refine (extScaleConstant_ensures ratio (hh (D - 2 - k + 1) (by omega) (by omega))).bind
      fun carried _ _ hcarried => ?_
    refine (extAdd_ensures (hget (by omega)).mono hcarried).bind fun value _ _ hvalue => ?_
    have next : ∀ {m' : ℕ} {v : ExtVar}, m ≤ m' → IsExt c m' v (H (k + 1)) →
        ForInStep.yield (head.set! (D - 2 - k) v) = ForInStep.yield (ForInStep.yield (head.set! (D - 2 - k) v)).value ∧
          (ForInStep.yield (head.set! (D - 2 - k) v)).value.size = D ∧ ∀ l, D - 1 - (k + 1) ≤ l → l < D →
            IsExt c m' (ForInStep.yield (head.set! (D - 2 - k) v)).value[l]! (H (D - 1 - l)) := by
      intro m' v le hv
      refine ⟨rfl, by simp [hs], fun l lo hi => ?_⟩
      simp only [ForInStep.value]
      rw [getElem!_set! _ _ _ _ (by omega)]
      split_ifs with same
      · rw [← same, show D - 1 - (D - 2 - k) = k + 1 by omega]
        exact hv
      · exact (hh l (by omega) hi).mono le
    have top : D - 1 - (D - 2 - k + 1) = k := by omega
    split
    · rename_i inside
      refine (extScaleConstant_ensures (ratio.pow length) (hget inside).mono).bind fun dropped _ _ hdropped => ?_
      refine (extSub_ensures hvalue.mono hdropped).bind fun value' _ _ hvalue' => Ensures.pure (next (by omega) ?_)
      have e : H (k + 1) = gtau[D - 2 - k]! + (H (D - 1 - (D - 2 - k + 1))).scale ratio -
          gtau[D - 2 - k + length]!.scale (ratio.pow length) := by
        rw [top, hH, Replay.headFrom, if_pos inside]
      rw [e]
      exact hvalue'
    · rename_i outside
      refine Ensures.pure (next (by omega) ?_)
      have e : H (k + 1) = gtau[D - 2 - k]! + (H (D - 1 - (D - 2 - k + 1))).scale ratio := by
        rw [top, hH, Replay.headFrom, if_neg outside]
      rw [e]
      exact hvalue
  obtain ⟨headSize, hhead⟩ := hhead
  have hhead : ∀ l < D, IsExt c m₁ head[l]! (H (D - 1 - l)) := fun l hl => hhead l (by simp) hl
  refine extZero_ensures.bind fun acc _ _ hzero => ?_
  refine (Ensures.loopArray (fun k s m => s.1.size = D ∧ (∀ i < k, IsExt c m s.1[i]! (S (i + 1))) ∧
      IsExt c m s.2.1 (S k) ∧ s.2.2 = ratio.pow k)
    ⟨by simp, fun i hi => absurd hi (by omega), hzero, Word.eq_of_toF (by rw [Word.toF_pow, pow_zero, Word.toF_one])⟩
    fun k hk s m _ hs => ?_).bind fun sums m₃ _ hsums => ?_
  · obtain ⟨hs1, hsums, hacc, hpow⟩ := hs
    have hk' : k < tau.size := by simpa using hk
    rw [show tau.zipIdx[k] = (tau[k]!, k) by simp [getElem!_pos, hk']]
    dsimp only
    refine (extScaleConstant_ensures s.2.2 (ht.2 k hk').mono).bind fun term _ _ hterm => ?_
    refine (extAdd_ensures hacc.mono hterm).bind fun acc' _ _ hacc' => Ensures.pure ?_
    have hkD : k < D := by rw [← size, ← ht.1]; exact hk'
    have next : S (k + 1) = S k + gtau[k]!.scale s.2.2 := by
      rw [hS, hpow]
      exact sumRange_succ k _
    refine ⟨rfl, by simp [hs1], fun i hi => ?_, next ▸ hacc', ?_⟩
    · show IsExt c _ (s.1.set! k acc')[i]! (S (i + 1))
      rw [getElem!_set! _ _ _ _ (by omega)]
      split_ifs with same
      · rw [← same]
        exact next ▸ hacc'
      · exact (hsums i (by omega)).mono
    · show s.2.2 * ratio = ratio.pow (k + 1)
      rw [hpow]
      exact Word.eq_of_toF (by rw [Word.toF_mul, Word.toF_pow, Word.toF_pow, pow_succ])
  have hsums : ∀ i < D, IsExt c m₃ sums.1[i]! (S (i + 1)) := fun i hi =>
    hsums.2.1 i (by rw [Array.size_zipIdx, ht.1, size]; exact hi)
  refine extZero_ensures.bind fun zero _ _ hzero' => ?_
  set T : ℕ → Ext := fun lane =>
    if lane + length ≥ D + 1 then (S (lane + length - D)).scale (ratio.pow (D - lane)) else 0 with hT
  refine (Ensures.rangeMapM D (fun lane a m => IsExt c m a (T lane)) (fun _ _ _ _ h le => h.mono le)
    fun i hi m le => ?_).bind fun tail _ _ htail => Ensures.pure ⟨?_, ?_⟩
  · split
    · rename_i long
      refine (extScaleConstant_ensures (ratio.pow (D - i)) (hsums (i + length - (D + 1)) (by omega)).mono).post
        fun _ _ q => ?_
      rw [hT]
      dsimp only
      rw [if_pos (by omega), show i + length - D = i + length - (D + 1) + 1 by omega]
      exact q
    · rename_i short'
      refine Ensures.pure ?_
      rw [hT]
      dsimp only
      rw [if_neg (by omega)]
      exact hzero'.mono
  · exact isExts_table headSize fun l hl => (hhead l hl).mono
  · obtain ⟨tsize, htail⟩ := htail
    refine isExts_table tsize fun i hi => ?_
    rw [getElem!_pos tail i (by rw [tsize]; exact hi)]
    exact htail i (by rw [tsize]; exact hi)

/-- The lane tables of every run class (`matrix.rs::Structure::lane_tables`). -/
private theorem laneTables_ensures {layout : Replay.Layout} {tau : Array ExtVar} {gtau : Array Ext}
    (ht : IsExts c n tau gtau) (size : gtau.size = D) (short : ∀ cl ∈ layout.classes, cl.len ≤ D) :
    Ensures c n (laneTables layout tau) fun r m =>
      IsLanes c m r (layout.classes.map fun cl => Replay.laneSplit gtau cl.len cl.ratio) := by
  unfold laneTables
  refine (Ensures.mapM (fun (cl : Replay.RunClass) (r : Array ExtVar × Array ExtVar) m =>
      IsExts c m r.1 (Replay.laneSplit gtau cl.len cl.ratio).1 ∧
        IsExts c m r.2 (Replay.laneSplit gtau cl.len cl.ratio).2)
    (fun _ _ _ _ h le => ⟨h.1.mono le, h.2.mono le⟩)
    fun cl hcl m le => laneSplit_ensures cl.ratio ht.mono size (short cl hcl)).post fun r m q => ?_
  obtain ⟨len, get⟩ := List.forall₂_iff_get.mp q
  have len' : r.size = layout.classes.size := by simpa using len.symm
  refine ⟨by rw [Array.size_map, len'], fun i hi => ?_⟩
  have e := get i (by simpa [len'] using hi) (by simpa using hi)
  simp only [List.get_eq_getElem, Array.getElem_toList] at e
  rw [getElem!_pos r i hi, getElem!_pos _ i (by rw [Array.size_map, ← len']; exact hi), Array.getElem_map]
  exact e

/-- `slot_step` (`matrix.rs::Structure::slot_step`): the replay's `slotStep`. -/
private theorem slotStep_ensures {layout : Replay.Layout} {point : Array ExtVar} {gpoint : Array Ext}
    {lanes : Array (Array ExtVar × Array ExtVar)} {glanes : Array (Array Ext × Array Ext)} (t : ℕ)
    (hp : IsExts c n point gpoint) (hl : IsLanes c n lanes glanes)
    (named : ∀ s ∈ layout.segments, s.classIndex < glanes.size ∧
      s.lane < (if t = 0 then glanes[s.classIndex]!.1 else glanes[s.classIndex]!.2).size) :
    Ensures c n (slotStep layout point lanes t) fun r m => IsExt c m r (Replay.slotStep layout gpoint glanes t) := by
  unfold slotStep Replay.slotStep
  refine (Mle.step_ensures (gvalues := layout.segments.map fun s =>
      (if t = 0 then glanes[s.classIndex]!.1 else glanes[s.classIndex]!.2)[s.lane]!) layout.slots hp
    ⟨by simp, fun i hi => ?_⟩ (by simp)).post fun _ _ q => ?_
  · have hi' : i < layout.segments.size := by simpa using hi
    have mem : layout.segments[i] ∈ layout.segments := Array.getElem_mem hi'
    obtain ⟨inside, lane⟩ := named _ mem
    obtain ⟨lsize, parts⟩ := hl
    rw [getElem!_pos _ i hi, getElem!_pos _ i (by simpa using hi'), Array.getElem_map, Array.getElem_map]
    have part := parts _ (lsize ▸ inside)
    split_ifs at lane ⊢
    · exact part.1.get lane
    · exact part.2.get lane
  · rw [Array.toList_map, Array.toList_map, List.zip_map'] at q
    exact q

end Lanes

/-! ## The leaf checks -/

/-- What the openings read of the late stage (`verifier.rs::verify`): the
pair point, the block value, the block point and `Ω^A` at it. -/
def IsLateOut (c : Ctx F) (m : ℕ) (v : LateOpeningsView) (out : Replay.LateOut) : Prop :=
  IsExts c m v.pairPoint out.openings.pairPoint ∧ IsExt c m v.block out.openings.block ∧
    IsExts c m v.blockPoint out.openings.blockPoint ∧ IsExt c m v.omega out.openings.omega

/-- `IsLateOut` survives later values (`verifier.rs::verify`). -/
theorem IsLateOut.mono {c : Ctx F} {n m : ℕ} {v : LateOpeningsView} {out : Replay.LateOut}
    (h : IsLateOut c n v out) (le : n ≤ m := by omega) : IsLateOut c m v out := by
  obtain ⟨pair, block, blockPoint, omega⟩ := h
  exact ⟨pair.mono le, block.mono le, blockPoint.mono le, omega.mono le⟩

private theorem checkLate_of {layout : Replay.Layout} {beta : Ext} {m : Replay.Mixing}
    {lanes : Array (Array Ext × Array Ext)} {claims : Replay.GkrClaims} {pairs : Array (Ext × Ext)}
    (pair : Replay.pairCheck claims.roots[0]! claims.roots[1]! "block scatter" = .ok ())
    (ubarLeaf : claims.leafValue 0 0 =
      Replay.sumRange pairs.size fun j => (m.evalA j).1 * (pairs[j]!).1 + (m.evalA j).2 * (pairs[j]!).2)
    (laneLeaf : claims.leafValue 0 1 =
      Replay.oneMinus (claims.leafPoint[0]!)[layout.slotVariables]! *
          Replay.slotStep layout ((claims.leafPoint[0]!).extract 0 layout.slotVariables) lanes 0 +
        (claims.leafPoint[0]!)[layout.slotVariables]! *
          Replay.slotStep layout ((claims.leafPoint[0]!).extract 0 layout.slotVariables) lanes 1)
    (blockLeaf : claims.leafValue 1 1 = beta - Replay.identityValue claims.leafPoint[1]!) :
    Replay.checkLate layout beta m lanes claims pairs = .ok (Bridge.lateOpeningsOf layout beta claims) := by
  unfold Replay.checkLate
  rw [pair]
  simp [Replay.ensure, ext_beq_of_eq ubarLeaf, ext_beq_of_eq laneLeaf, ext_beq_of_eq blockLeaf]
  rfl

section Check

variable [CharP F goldilocksModulus] {c : Ctx F} {n : ℕ}

/-- `Structure::check_late` (`matrix.rs::Structure::check_late`): once its checks hold, the
replay's `checkLate` accepts, and the openings are the replay's. -/
private theorem checkLate_ensures {layout : Replay.Layout} {beta : ExtVar} {gbeta : Ext}
    {evalA : Array (ExtVar × ExtVar)} {mix : Replay.Mixing} {lanes : Array (Array ExtVar × Array ExtVar)}
    {glanes : Array (Array Ext × Array Ext)} {claims : Array TreeClaimView} {g : Replay.GkrClaims}
    {pairs : Array (ExtVar × ExtVar)} {gpairs : Array (Ext × Ext)}
    (hbeta : IsExt c n beta gbeta) (hes : gpairs.size ≤ evalA.size)
    (hevalA : ∀ j < evalA.size, IsPair c n evalA[j]! (mix.evalA j)) (hlanes : IsLanes c n lanes glanes)
    (named : ∀ s ∈ layout.segments, s.classIndex < glanes.size ∧ s.lane < glanes[s.classIndex]!.1.size ∧
      s.lane < glanes[s.classIndex]!.2.size)
    (hclaims : IsGkrClaims c n claims g) (two : claims.size = 2) (s₀ : 3 ≤ (g.leafValues[0]!).size)
    (s₁ : 2 ≤ (g.leafValues[1]!).size) (s₂ : layout.slotVariables < (g.leafPoint[0]!).size)
    (hps : pairs.size = gpairs.size) (hpairs : ∀ j < pairs.size, IsPair c n pairs[j]! gpairs[j]!) :
    Ensures c n (checkLate layout beta evalA lanes claims pairs) fun res k =>
      Replay.checkLate layout gbeta mix glanes g gpairs = .ok (Bridge.lateOpeningsOf layout gbeta g) ∧
        IsLateOut c k res ⟨gbeta, g, gpairs, Bridge.lateOpeningsOf layout gbeta g⟩ := by
  obtain ⟨r₀, q₀, point₀, values₀⟩ := hclaims.2 0 (by omega)
  obtain ⟨r₁, q₁, point₁, values₁⟩ := hclaims.2 1 (by omega)
  have value₀ : ∀ j < (g.leafValues[0]!).size, IsExt c n claims[0]!.values[j]! (g.leafValue 0 j) :=
    fun j hj => values₀.get hj
  have value₁ : ∀ j < (g.leafValues[1]!).size, IsExt c n claims[1]!.values[j]! (g.leafValue 1 j) :=
    fun j hj => values₁.get hj
  have top : IsExt c n (claims[0]!.point)[layout.slotVariables]! (g.leafPoint[0]!)[layout.slotVariables]! :=
    point₀.get s₂
  unfold checkLate
  dsimp only
  refine (checkPair_ensures (gx := g.roots[0]!) (gy := g.roots[1]!) "block scatter" ⟨r₀, q₀⟩ ⟨r₁, q₁⟩).bind
    fun _ _ _ pair => ?_
  refine extZero_ensures.bind fun ubar₀ _ _ hzero => ?_
  refine (Ensures.loopArray (fun k u m => IsExt c m u (Replay.sumRange k fun j =>
      (mix.evalA j).1 * (gpairs[j]!).1 + (mix.evalA j).2 * (gpairs[j]!).2)) hzero fun k hk u m _ hu => ?_).bind
    fun ubar _ _ hubar => ?_
  · have hk' : k < pairs.size ∧ k < evalA.size := by simpa using hk
    rw [show (pairs.zip evalA)[k] = (pairs[k]!, evalA[k]!) by simp [getElem!_pos, hk'.1, hk'.2]]
    have hw : IsPair c m evalA[k]! (mix.evalA k) := (hevalA k hk'.2).mono
    have hv : IsPair c m pairs[k]! gpairs[k]! := (hpairs k hk'.1).mono
    refine (extMul_ensures hw.1 hv.1).bind fun a _ _ ha => ?_
    refine (extMul_ensures hw.2.mono hv.2.mono).bind fun b _ _ hb => ?_
    refine (extAdd_ensures ha.mono hb).bind fun term _ _ hterm => ?_
    exact (extAdd_ensures hu.mono hterm).bind fun u' _ _ hu' =>
      Ensures.pure ⟨rfl, by rw [sumRange_succ]; exact hu'⟩
  rw [Array.size_zip, hps, Nat.min_eq_left hes] at hubar
  refine (slotStep_ensures 0 (point₀.slice 0 layout.slotVariables).mono hlanes.mono
    fun s hs => ⟨(named s hs).1, by simpa using (named s hs).2.1⟩).bind fun head _ _ hhead => ?_
  refine (slotStep_ensures 1 (point₀.slice 0 layout.slotVariables).mono hlanes.mono
    fun s hs => ⟨(named s hs).1, by simpa using (named s hs).2.2⟩).bind fun tail _ _ htail => ?_
  refine (oneMinus_ensures top.mono).bind fun low _ _ hlow => ?_
  refine (extMul_ensures hlow hhead.mono).bind fun a _ _ ha => ?_
  refine (extMul_ensures top.mono htail.mono).bind fun b _ _ hb => ?_
  refine (extAdd_ensures ha.mono hb).bind fun lane _ _ hlane => ?_
  refine (assertExtEqual_ensures (value₀ 0 (by omega)).mono hubar.mono).bind fun _ _ _ ubarLeaf => ?_
  refine (assertExtEqual_ensures (value₀ 1 (by omega)).mono hlane.mono).bind fun _ _ _ laneLeaf => ?_
  refine (Mle.identity_ensures point₁.mono).bind fun index _ _ hindex => ?_
  refine (extSub_ensures hbeta.mono hindex).bind fun expected _ _ hexpected => ?_
  refine (assertExtEqual_ensures (value₁ 1 (by omega)).mono hexpected).bind fun _ _ _ blockLeaf => ?_
  refine (extSub_ensures hbeta.mono (value₀ 2 (by omega)).mono).bind fun rest _ _ hrest => ?_
  refine (extSub_ensures hrest top.mono).bind fun block _ _ hblock => Ensures.pure ?_
  exact ⟨checkLate_of pair ubarLeaf laneLeaf blockLeaf, (point₀.slice 0 layout.slotVariables).mono, hblock,
    point₁.mono, (value₁ 0 (by omega)).mono⟩

end Check

/-! ## The late stage -/

section Stage

variable [CharP F goldilocksModulus] {c : Ctx F} {n : ℕ}

/-- **The late stage** (`verifier.rs::verify`). Once the circuit's checks
hold, the circuit's late stage tracks the replay's, its openings are the
replay's, and its pairs are the proof's. `short` and `named` are
`Structure::check` (`matrix.rs::Structure::check`). -/
theorem lateStage_tracks {rel : Relation} {tau : Array ExtVar} {evalA : Array (ExtVar × ExtVar)} {late : GkrView}
    {lateValues : Array (ExtVar × ExtVar)} {proof : Replay.Proof} {lin : Replay.LinearOut} (hp : Permutes c.perm)
    (htau : IsExts c n tau lin.tau) (tauSize : lin.tau.size = D) (hes : evalA.size = rel.matrices)
    (hevalA : ∀ j < evalA.size, IsPair c n evalA[j]! (lin.mixing.evalA j)) (hlate : IsGkr c n late proof.late)
    (hvalues : List.Forall₂ (IsPair c n) lateValues.toList proof.lateValues.toList)
    (read : Replay.readProof rel proof = .ok ()) (short : ∀ cl ∈ rel.key.layout.classes, cl.len ≤ D)
    (named : ∀ s ∈ rel.key.layout.segments, s.classIndex < rel.key.layout.classes.size ∧ s.lane < D) :
    Tracks c n (Circuit.lateStage rel tau evalA late lateValues) (Replay.lateStage rel proof lin)
      fun v out m => IsLateOut c m v out ∧ out.pairs = proof.lateValues := by
  have shape := Bridge.readProof_ok read
  obtain ⟨rootsSize, -, -⟩ := Bridge.readGkr_ok shape.late
  set glanes := rel.key.layout.classes.map fun cl => Replay.laneSplit lin.tau cl.len cl.ratio
  have lanesNamed : ∀ s ∈ rel.key.layout.segments, s.classIndex < glanes.size ∧
      s.lane < glanes[s.classIndex]!.1.size ∧ s.lane < glanes[s.classIndex]!.2.size := fun s hs => by
    obtain ⟨inside, lane⟩ := named s hs
    rw [getElem!_pos glanes _ (by simpa [glanes] using inside), Array.getElem_map]
    simp [glanes, inside, lane, Replay.laneSplit]
  unfold Circuit.lateStage Replay.lateStage
  refine Tracks.bind (Duplex.sampleExt_tracks hp) fun beta gbeta m₁ _ hbeta => ?_
  obtain ⟨⟨hrs, hroots⟩, hls, hlayers⟩ := (hlate.mono (m := m₁)).entries
  refine Tracks.bind (gkrVerify_tracks hp hrs hroots hls hlayers shape.late).runs fun claims gc _ _ hc => ?_
  obtain ⟨hc, _, _, run⟩ := hc
  obtain ⟨roots, sizes⟩ := gkrVerify_sizes shape.late run
  have s₀ := sizes 0 (by rw [Bridge.lateShapes_size]; omega)
  have s₁ := sizes 1 (by rw [Bridge.lateShapes_size]; omega)
  have f₀ : (rel.lateShapes[0]!).factors = 2 := rfl
  have f₁ : (rel.lateShapes[1]!).factors = 1 := rfl
  have d₀ : (rel.lateShapes[0]!).depth = rel.key.layout.slotVariables + 1 := rfl
  rw [f₀, d₀] at s₀
  rw [f₁] at s₁
  have two : claims.size = 2 := by
    rw [hc.1, roots, rootsSize]
    rfl
  obtain ⟨pairsSize, pairsOk⟩ := forall₂_entries hvalues
  refine Tracks.lift (laneTables_ensures htau.mono tauSize short) fun lanes _ _ hlanes => ?_
  refine Tracks.lift (checkLate_ensures (mix := lin.mixing) (gpairs := proof.lateValues) hbeta.mono
    (by rw [shape.lateValues, hes]) (fun j hj => (hevalA j hj).mono) hlanes lanesNamed hc.mono two
    (by omega) (by omega) (by omega) pairsSize (fun j hj => by rw [getElem!_pos _ j hj]; exact (pairsOk j hj).mono))
    fun openings _ _ hopen => ?_
  refine Tracks.check hopen.1 ?_
  refine Tracks.bind (observePairs_tracks hp (hvalues.imp fun _ _ q => q.mono)) fun _ _ _ _ _ => ?_
  exact Tracks.pure ⟨hopen.2.mono, rfl⟩

end Stage

end CompressionLean.Sim
