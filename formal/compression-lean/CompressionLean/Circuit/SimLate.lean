import CompressionLean.Circuit.SimEarly
import CompressionLean.Circuit.SimHonest
import CompressionLean.Bridge.Late

/-!
Owns the simulation of the late stage of `Late.lean` (`verifier.rs:251-260`)
against `Replay.lateStage`: `lateStage_ensures` gives the replay's run, and
`IsLateOut` gives what the openings read.

The circuit builds the lane tables (`mle.rs:109-140`, `matrix.rs:217-224`)
with loops; the replay states each entry in closed form (`headFrom`,
`prefixSum`). They agree when `τ` has `D` lanes and every class has at most
`D` lanes. The slot steps (`matrix.rs:382-392`) need every segment to name a
class and a lane. `Structure::check` enforces both (`matrix.rs:142-182`). The
leaf checks (`matrix.rs:319-361`) read three leaf values and one point
coordinate of the pair tree and two leaf values of the block tree; the
replay's GKR run gives these sizes.
-/

namespace CompressionLean.Circuit

open CompressionLean.Lib
open CompressionLean.Replay (Word Ext Relation TreeShape D Runs runs_bind runs_pure runs_sampleExt
  runs_observeExts runs_liftExcept)
open NightstreamFPrime.Spec (goldilocksModulus)

variable {F : Type} [CommRing F]

private theorem isExts_table {c : Ctx F} {m k : ℕ} {xs : Array ExtVar} {f : ℕ → Ext} (size : xs.size = k)
    (h : ∀ i < k, IsExt c m xs[i]! (f i)) : IsExts c m xs ((Array.range k).map f) :=
  ⟨by simp [size], fun i hi => by rw [getElem!_map_range k f (size ▸ hi)]; exact h i (size ▸ hi)⟩

private def IsLanes (c : Ctx F) (m : ℕ) (lanes : Array (Array ExtVar × Array ExtVar))
    (glanes : Array (Array Ext × Array Ext)) : Prop :=
  lanes.size = glanes.size ∧
    ∀ i < lanes.size, IsExts c m lanes[i]!.1 glanes[i]!.1 ∧ IsExts c m lanes[i]!.2 glanes[i]!.2

private theorem IsLanes.mono {c : Ctx F} {n m : ℕ} {lanes : Array (Array ExtVar × Array ExtVar)}
    {glanes : Array (Array Ext × Array Ext)} (h : IsLanes c n lanes glanes) (le : n ≤ m) :
    IsLanes c m lanes glanes :=
  ⟨h.1, fun i hi => ⟨(h.2 i hi).1.mono le, (h.2 i hi).2.mono le⟩⟩

section Lanes

variable [CharP F goldilocksModulus] {c : Ctx F} {n : ℕ} {P : Prop}

private theorem laneSplit_sim {tau : Array ExtVar} {gtau : Array Ext} {length : ℕ} (ratio : Word)
    (ht : P → IsExts c n tau gtau) (size : P → gtau.size = D) (short : length ≤ D) :
    Sim c n P (Mle.laneSplit tau length ratio)
      (fun r m => IsExts c m r.1 (Replay.laneSplit gtau length ratio).1 ∧
        IsExts c m r.2 (Replay.laneSplit gtau length ratio).2) True := by
  set H : ℕ → Ext := Replay.headFrom gtau length ratio (ratio.pow length) with hH
  set S : ℕ → Ext := Replay.prefixSum gtau ratio with hS
  have hD : 1 < D := by decide
  have hget : ∀ {m i}, P → n ≤ m → i < D → IsExt c m tau[i]! gtau[i]! :=
    fun h le hi => ((ht h).get (by rw [size h]; exact hi)).mono le
  unfold Mle.laneSplit
  dsimp only
  refine Sim.bind₀ (Q := fun head m => head.size = D ∧ ∀ l < D, IsExt c m head[l]! (H (D - 1 - l)))
    (((Sim.loop (P := P) (I := fun k head m => head.size = D ∧
      ∀ l, D - 1 - k ≤ l → l < D → IsExt c m head[l]! (H (D - 1 - l))) (fun _ => True)
      fun k hk head m lm => ?_).post (fun _ _ _ q => ?_) fun _ => ⟨fun _ => trivial, fun _ _ _ => trivial⟩).pre
      fun h => ⟨h, by simp, fun l lo hi => ?_⟩) fun head m₁ l₁ => ?_
  · have hk' : k < D - 1 := by simpa using hk
    have xk : (List.range (D - 1)).reverse[k]'hk = D - 2 - k := by
      rw [List.getElem_reverse, List.getElem_range]
      simp only [List.length_range]
      omega
    rw [xk]
    refine (Sim.extScaleConstant ratio fun h => h.2.2 (D - 2 - k + 1) (by omega) (by omega)).bind₀
      fun carried m₂ l₂ => ?_
    refine (Sim.extAdd (fun h => hget h.1.1 (lm.trans l₂) (by omega)) fun h => h.2).bind₀ fun value m₃ l₃ => ?_
    have next : ∀ {m' : ℕ} {v : ExtVar}, n ≤ m' → m ≤ m' → head.size = D →
        (∀ l, D - 1 - k ≤ l → l < D → IsExt c m head[l]! (H (D - 1 - l))) → IsExt c m' v (H (k + 1)) →
        ForInStep.yield (head.set! (D - 2 - k) v) = ForInStep.yield (ForInStep.yield (head.set! (D - 2 - k) v)).value ∧
          (ForInStep.yield (head.set! (D - 2 - k) v)).value.size = D ∧ ∀ l, D - 1 - (k + 1) ≤ l → l < D →
            IsExt c m' (ForInStep.yield (head.set! (D - 2 - k) v)).value[l]! (H (D - 1 - l)) := by
      intro m' v _ le hs hh hv
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
      refine (Sim.extScaleConstant (ratio.pow length) fun h => hget h.1.1.1 (lm.trans (l₂.trans l₃)) inside).bind₀
        fun dropped m₄ l₄ => ?_
      refine (Sim.extSub (fun h => h.1.2.mono l₄) fun h => h.2).bind₀ fun value' m₅ l₅ => ?_
      refine Sim.pure fun h => next (lm.trans (l₂.trans (l₃.trans (l₄.trans l₅)))) (l₂.trans (l₃.trans (l₄.trans l₅)))
        h.1.1.1.1.2.1 h.1.1.1.1.2.2 ?_
      have e : H (k + 1) = gtau[D - 2 - k]! + (H (D - 1 - (D - 2 - k + 1))).scale ratio -
          gtau[D - 2 - k + length]!.scale (ratio.pow length) := by
        rw [top, hH, Replay.headFrom, if_pos inside]
      rw [e]
      exact h.2
    · rename_i outside
      refine Sim.pure fun h => next (lm.trans (l₂.trans l₃)) (l₂.trans l₃) h.1.1.2.1 h.1.1.2.2 ?_
      have e : H (k + 1) = gtau[D - 2 - k]! + (H (D - 1 - (D - 2 - k + 1))).scale ratio := by
        rw [top, hH, Replay.headFrom, if_neg outside]
      rw [e]
      exact h.2
  · exact ⟨q.1, fun l hl => q.2 l (by simp) hl⟩
  · have e : l = D - 1 := by omega
    subst e
    rw [getElem!_pos _ _ (by simp; omega), Array.getElem_replicate, Nat.sub_self, hH, Replay.headFrom]
    exact hget h le_rfl (by omega)
  refine Sim.extZero.bind₀ fun acc m₂ l₂ => ?_
  refine Sim.bind₀ (Q := fun s m => s.1.size = D ∧ ∀ i < D, IsExt c m s.1[i]! (S (i + 1)))
    (((Sim.loop₀ (I := fun k s m => s.1.size = D ∧ (∀ i < k, IsExt c m s.1[i]! (S (i + 1))) ∧
      IsExt c m s.2.1 (S k) ∧ s.2.2 = ratio.pow k) fun k hk s m lm => ?_).post (fun h _ _ q => ?_)
      fun _ => Iff.rfl).pre fun h => ⟨h, by simp, fun i hi => absurd hi (by omega), h.2,
      Word.eq_of_toF (by rw [Word.toF_pow, pow_zero, Word.toF_one])⟩)
    fun sums m₃ l₃ => ?_
  · have hk' : k < tau.size := by simpa using hk
    rw [show tau.zipIdx[k] = (tau[k]!, k) by simp [getElem!_pos, hk']]
    dsimp only
    refine (Sim.extScaleConstant s.2.2 fun h => ((ht h.1.1.1).2 k hk').mono (l₁.trans (l₂.trans lm))).bind₀
      fun term m₄ l₄ => ?_
    refine (Sim.extAdd (fun h => h.1.2.2.2.1.mono l₄) fun h => h.2).bind₀ fun acc' m₅ l₅ => ?_
    refine Sim.pure fun h => ?_
    obtain ⟨⟨⟨⟨⟨hP, -⟩, -⟩, hs, hsums, -, hpow⟩, -⟩, hacc⟩ := h
    have hkD : k < D := by rw [← size hP, ← (ht hP).1]; exact hk'
    have next : S (k + 1) = S k + gtau[k]!.scale s.2.2 := by
      rw [hS, hpow]
      exact sumRange_succ k _
    refine ⟨rfl, by simp [hs], fun i hi => ?_, next ▸ hacc, by
      show s.2.2 * ratio = ratio.pow (k + 1)
      rw [hpow]
      exact Word.eq_of_toF (by rw [Word.toF_mul, Word.toF_pow, Word.toF_pow, pow_succ])⟩
    show IsExt c m₅ (s.1.set! k acc')[i]! (S (i + 1))
    rw [getElem!_set! _ _ _ _ (by omega)]
    split_ifs with same
    · rw [← same]
      exact next ▸ hacc
    · exact (hsums i (by omega)).mono (l₄.trans l₅)
  · refine ⟨q.1, fun i hi => q.2.1 i ?_⟩
    rw [Array.size_zipIdx, (ht h.1.1.1).1, size h.1.1.1]
    exact hi
  refine Sim.extZero.bind₀ fun zero m₄ l₄ => ?_
  set T : ℕ → Ext := fun lane =>
    if lane + length ≥ D + 1 then (S (lane + length - D)).scale (ratio.pow (D - lane)) else 0 with hT
  refine (Sim.rangeMapM D (fun lane a m => IsExt c m a (T lane)) (fun _ _ _ _ h le => h.mono le)
    fun i hi m le => ?_).bind₀ fun tail m₅ l₅ => Sim.pure fun h => ⟨?_, ?_⟩
  · split
    · rename_i long
      refine (Sim.extScaleConstant (ratio.pow (D - i)) fun h => (h.1.2.2 (i + length - (D + 1)) (by omega)).mono
        (l₄.trans le)).post (fun _ _ _ q => ?_) fun _ => Iff.rfl
      rw [hT]
      dsimp only
      rw [if_pos (by omega), show i + length - D = i + length - (D + 1) + 1 by omega]
      exact q
    · rename_i short'
      refine Sim.pure fun h => ?_
      rw [hT]
      dsimp only
      rw [if_neg (by omega)]
      exact h.2.mono le
  · exact isExts_table h.1.1.1.1.2.1 fun l hl =>
      (h.1.1.1.1.2.2 l hl).mono (l₂.trans (l₃.trans (l₄.trans l₅)))
  · refine isExts_table h.2.1 fun i hi => ?_
    rw [getElem!_pos tail i (by rw [h.2.1]; exact hi)]
    exact h.2.2 i (by rw [h.2.1]; exact hi)

private theorem laneTables_sim {layout : Replay.Layout} {tau : Array ExtVar} {gtau : Array Ext}
    (ht : P → IsExts c n tau gtau) (size : P → gtau.size = D) (short : ∀ cl ∈ layout.classes, cl.len ≤ D) :
    Sim c n P (laneTables layout tau) (fun r m => IsLanes c m r
      (layout.classes.map fun cl => Replay.laneSplit gtau cl.len cl.ratio)) True := by
  unfold laneTables
  refine (Sim.mapM (fun (cl : Replay.RunClass) (r : Array ExtVar × Array ExtVar) m =>
      IsExts c m r.1 (Replay.laneSplit gtau cl.len cl.ratio).1 ∧
        IsExts c m r.2 (Replay.laneSplit gtau cl.len cl.ratio).2)
    (fun _ _ _ _ h le => ⟨h.1.mono le, h.2.mono le⟩)
    fun cl hcl m le => laneSplit_sim cl.ratio (fun h => (ht h).mono le) size (short cl hcl)).post
    (fun _ r m q => ?_) fun _ => Iff.rfl
  obtain ⟨len, get⟩ := List.forall₂_iff_get.mp q
  have len' : r.size = layout.classes.size := by simpa using len.symm
  refine ⟨by rw [Array.size_map, len'], fun i hi => ?_⟩
  have e := get i (by simpa [len'] using hi) (by simpa using hi)
  simp only [List.get_eq_getElem, Array.getElem_toList] at e
  rw [getElem!_pos r i hi, getElem!_pos _ i (by rw [Array.size_map, ← len']; exact hi), Array.getElem_map]
  exact e

private theorem slotStep_sim {layout : Replay.Layout} {point : Array ExtVar} {gpoint : Array Ext}
    {lanes : Array (Array ExtVar × Array ExtVar)} {glanes : Array (Array Ext × Array Ext)} (t : ℕ)
    (hp : P → IsExts c n point gpoint) (hl : P → IsLanes c n lanes glanes)
    (named : ∀ s ∈ layout.segments, s.classIndex < glanes.size ∧
      s.lane < (if t = 0 then glanes[s.classIndex]!.1 else glanes[s.classIndex]!.2).size) :
    Sim c n P (slotStep layout point lanes t) (fun r m => IsExt c m r (Replay.slotStep layout gpoint glanes t))
      True := by
  unfold slotStep Replay.slotStep
  refine (Sim.step (gvalues := layout.segments.map fun s =>
      (if t = 0 then glanes[s.classIndex]!.1 else glanes[s.classIndex]!.2)[s.lane]!) layout.slots hp
    (fun h => ⟨by simp, fun i hi => ?_⟩) fun _ => by simp).post (fun _ _ _ q => ?_) fun _ => Iff.rfl
  · have hi' : i < layout.segments.size := by simpa using hi
    have mem : layout.segments[i] ∈ layout.segments := Array.getElem_mem hi'
    obtain ⟨inside, lane⟩ := named _ mem
    obtain ⟨lsize, parts⟩ := hl h
    rw [getElem!_pos _ i hi, getElem!_pos _ i (by simpa using hi'), Array.getElem_map, Array.getElem_map]
    have part := parts _ (lsize ▸ inside)
    split_ifs at lane ⊢
    · exact part.1.get lane
    · exact part.2.get lane
  · rw [Array.toList_map, Array.toList_map, List.zip_map'] at q
    exact q

end Lanes

/-- What the openings read of the late stage (`verifier.rs:251-260`): the
pair point, the block value, the block point and `Ω^A` at it. -/
def IsLateOut (c : Ctx F) (m : ℕ) (v : LateOpeningsView) (out : Replay.LateOut) : Prop :=
  IsExts c m v.pairPoint out.openings.pairPoint ∧ IsExt c m v.block out.openings.block ∧
    IsExts c m v.blockPoint out.openings.blockPoint ∧ IsExt c m v.omega out.openings.omega

/-- `IsLateOut` survives later values (`verifier.rs:251-260`). -/
theorem IsLateOut.mono {c : Ctx F} {n m : ℕ} {v : LateOpeningsView} {out : Replay.LateOut}
    (h : IsLateOut c n v out) (le : n ≤ m) : IsLateOut c m v out :=
  ⟨h.1.mono le, h.2.1.mono le, h.2.2.1.mono le, h.2.2.2.mono le⟩

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

variable [CharP F goldilocksModulus] {c : Ctx F} {n : ℕ} {P : Prop}

private theorem checkLate_ensures {layout : Replay.Layout} {beta : ExtVar} {gbeta : Ext}
    {evalA : Array (ExtVar × ExtVar)} {mix : Replay.Mixing} {lanes : Array (Array ExtVar × Array ExtVar)}
    {glanes : Array (Array Ext × Array Ext)} {claims : Array TreeClaimView} {g : Replay.GkrClaims}
    {pairs : Array (ExtVar × ExtVar)} {gpairs : Array (Ext × Ext)}
    (hbeta : P → IsExt c n beta gbeta)
    (hevalA : P → gpairs.size ≤ evalA.size ∧ ∀ j < evalA.size, IsPair c n evalA[j]! (mix.evalA j))
    (hlanes : P → IsLanes c n lanes glanes)
    (named : ∀ s ∈ layout.segments, s.classIndex < glanes.size ∧ s.lane < glanes[s.classIndex]!.1.size ∧
      s.lane < glanes[s.classIndex]!.2.size)
    (hclaims : P → IsGkrClaims c n claims g ∧ claims.size = 2)
    (hsizes : P → 3 ≤ (g.leafValues[0]!).size ∧ 2 ≤ (g.leafValues[1]!).size ∧
      layout.slotVariables < (g.leafPoint[0]!).size)
    (hpairs : P → pairs.size = gpairs.size ∧ ∀ j < pairs.size, IsPair c n pairs[j]! gpairs[j]!) :
    Ensures c n P (checkLate layout beta evalA lanes claims pairs) (fun res k =>
      Replay.checkLate layout gbeta mix glanes g gpairs = .ok (Bridge.lateOpeningsOf layout gbeta g) ∧
        IsLateOut c k res ⟨gbeta, g, gpairs, Bridge.lateOpeningsOf layout gbeta g⟩) := by
  have tree : ∀ {k} u, u < 2 → P → n ≤ k → IsExt c k claims[u]!.root.1 (g.roots[u]!).1 ∧
      IsExt c k claims[u]!.root.2 (g.roots[u]!).2 ∧ IsExts c k claims[u]!.point g.leafPoint[u]! ∧
      IsExts c k claims[u]!.values g.leafValues[u]! := fun u hu h le => by
    have q := (hclaims h).1.2 u (by rw [(hclaims h).2]; exact hu)
    exact ⟨q.1.mono le, q.2.1.mono le, q.2.2.1.mono le, q.2.2.2.mono le⟩
  have value : ∀ {k} u j, u < 2 → P → n ≤ k → j < (g.leafValues[u]!).size →
      IsExt c k claims[u]!.values[j]! (g.leafValue u j) := fun u j hu h le hj => (tree u hu h le).2.2.2.get hj
  have top : ∀ {k}, P → n ≤ k →
      IsExt c k (claims[0]!.point)[layout.slotVariables]! (g.leafPoint[0]!)[layout.slotVariables]! :=
    fun h le => (tree 0 (by omega) h le).2.2.1.get (hsizes h).2.2
  unfold checkLate
  dsimp only
  refine Ensures.bind (checkPair_ensures (gx := g.roots[0]!) (gy := g.roots[1]!) "block scatter"
    (fun h => ⟨(tree 0 (by omega) h le_rfl).1, (tree 0 (by omega) h le_rfl).2.1⟩)
    fun h => ⟨(tree 1 (by omega) h le_rfl).1, (tree 1 (by omega) h le_rfl).2.1⟩) fun _ m₁ l₁ => ?_
  set U := Replay.sumRange gpairs.size fun j => (mix.evalA j).1 * (gpairs[j]!).1 + (mix.evalA j).2 * (gpairs[j]!).2
  set T := (g.leafPoint[0]!)[layout.slotVariables]!
  set slot := (g.leafPoint[0]!).extract 0 layout.slotVariables
  set L := Replay.oneMinus T * Replay.slotStep layout slot glanes 0 + T * Replay.slotStep layout slot glanes 1
  refine Ensures.post (Sim.ensures (Q := fun res k => IsLateOut c k res ⟨gbeta, g, gpairs,
      Bridge.lateOpeningsOf layout gbeta g⟩) (C := g.leafValue 0 0 = U ∧ g.leafValue 0 1 = L ∧
      g.leafValue 1 1 = gbeta - Replay.identityValue g.leafPoint[1]! ∧ True) ?_)
    fun h _ _ q => ⟨checkLate_of h.2 q.2.1 q.2.2.1 q.2.2.2.1, q.1⟩
  refine Sim.extZero.bind₀ fun ubar₀ m₂ l₂ => ?_
  refine (((Sim.loop₀ (I := fun k u m => IsExt c m u (Replay.sumRange k fun j =>
      (mix.evalA j).1 * (gpairs[j]!).1 + (mix.evalA j).2 * (gpairs[j]!).2)) fun k hk u m lm => ?_).pre
    fun h => ⟨h, h.2⟩).post (Q' := fun u m => IsExt c m u U) (fun h _ _ q => ?_) fun _ => Iff.rfl).bind₀
    fun ubar m₃ l₃ => ?_
  · have hk' : k < pairs.size ∧ k < evalA.size := by simpa using hk
    have L₀ : n ≤ m := l₁.trans (l₂.trans lm)
    rw [show (pairs.zip evalA)[k] = (pairs[k]!, evalA[k]!) by simp [getElem!_pos, hk'.1, hk'.2]]
    refine (Sim.extMul (fun h => ((hevalA h.1.1.1).2 k hk'.2).1.mono L₀)
      fun h => ((hpairs h.1.1.1).2 k hk'.1).1.mono L₀).bind₀ fun a m₄ l₄ => ?_
    refine (Sim.extMul (fun h => ((hevalA h.1.1.1.1).2 k hk'.2).2.mono (L₀.trans l₄))
      fun h => ((hpairs h.1.1.1.1).2 k hk'.1).2.mono (L₀.trans l₄)).bind₀ fun b m₅ l₅ => ?_
    refine (Sim.extAdd (fun h => h.1.2.mono l₅) fun h => h.2).bind₀ fun term m₆ l₆ => ?_
    refine (Sim.extAdd (fun h => h.1.1.1.2.mono (l₄.trans (l₅.trans l₆))) fun h => h.2).bind₀ fun u' m₇ l₇ => ?_
    exact Sim.pure fun h => ⟨rfl, by rw [sumRange_succ]; exact h.2⟩
  · rw [Array.size_zip, (hpairs h.1.1).1, Nat.min_eq_left (hevalA h.1.1).1] at q
    exact q
  have L₃ : n ≤ m₃ := l₁.trans (l₂.trans l₃)
  refine Sim.pre (P' := (P ∧ Replay.pairCheck g.roots[0]! g.roots[1]! "block scatter" = .ok ()) ∧
    IsExt c m₃ ubar U) ?_ fun h => ⟨h.1.1, h.2⟩
  refine (slotStep_sim 0 (fun h => (tree 0 (by omega) h.1.1 L₃).2.2.1.slice 0 layout.slotVariables)
    (fun h => (hlanes h.1.1).mono L₃) fun s hs => ⟨(named s hs).1, by simpa using (named s hs).2.1⟩).bind₀
    fun head m₄ l₄ => ?_
  refine (slotStep_sim 1 (fun h => (tree 0 (by omega) h.1.1.1 (L₃.trans l₄)).2.2.1.slice 0 layout.slotVariables)
    (fun h => (hlanes h.1.1.1).mono (L₃.trans l₄)) fun s hs => ⟨(named s hs).1, by simpa using (named s hs).2.2⟩).bind₀
    fun tail m₅ l₅ => ?_
  refine (Sim.oneMinus fun h => top h.1.1.1.1 (L₃.trans (l₄.trans l₅))).bind₀ fun low m₆ l₆ => ?_
  refine (Sim.extMul (fun h => h.2) fun h => h.1.1.2.mono (l₅.trans l₆)).bind₀ fun a m₇ l₇ => ?_
  refine (Sim.extMul (fun h => top h.1.1.1.1.1.1 (L₃.trans (l₄.trans (l₅.trans (l₆.trans l₇)))))
    fun h => h.1.1.2.mono (l₆.trans l₇)).bind₀ fun b m₈ l₈ => ?_
  refine (Sim.extAdd (fun h => h.1.2.mono l₈) fun h => h.2).bind₀ fun lane m₉ l₉ => ?_
  have L₉ : m₃ ≤ m₉ := l₄.trans (l₅.trans (l₆.trans (l₇.trans (l₈.trans l₉))))
  refine Sim.pre (P' := ((P ∧ Replay.pairCheck g.roots[0]! g.roots[1]! "block scatter" = .ok ()) ∧
    IsExt c m₉ ubar U) ∧ IsExt c m₉ lane L) ?_ fun h => ⟨⟨h.1.1.1.1.1.1.1, h.1.1.1.1.1.1.2.mono L₉⟩, h.2⟩
  have N₉ : n ≤ m₉ := L₃.trans L₉
  refine (Sim.assertExtEqual (fun h => value 0 0 (by omega) h.1.1.1 N₉ (by have := (hsizes h.1.1.1).1; omega))
    fun h => h.1.2).bind fun _ m₁₀ l₁₀ => ?_
  refine (Sim.assertExtEqual (fun h => value 0 1 (by omega) h.1.1.1.1 (N₉.trans l₁₀)
    (by have := (hsizes h.1.1.1.1).1; omega)) fun h => h.1.2.mono l₁₀).bind fun _ m₁₁ l₁₁ => ?_
  have N₁₁ : n ≤ m₁₁ := N₉.trans (l₁₀.trans l₁₁)
  refine (Sim.identity fun h => (tree 1 (by omega) h.1.1.1.1.1 N₁₁).2.2.1).bind₀ fun index m₁₂ l₁₂ => ?_
  refine (Sim.extSub (fun h => (hbeta h.1.1.1.1.1.1).mono (N₁₁.trans l₁₂)) fun h => h.2).bind₀
    fun expected m₁₃ l₁₃ => ?_
  refine (Sim.assertExtEqual (fun h => value 1 1 (by omega) h.1.1.1.1.1.1.1 (N₁₁.trans (l₁₂.trans l₁₃))
    (by have := (hsizes h.1.1.1.1.1.1.1).2.1; omega)) fun h => h.2).bind fun _ m₁₄ l₁₄ => ?_
  have N₁₄ : n ≤ m₁₄ := N₁₁.trans (l₁₂.trans (l₁₃.trans l₁₄))
  refine (Sim.extSub (fun h => (hbeta h.1.1.1.1.1.1.1.1).mono N₁₄) fun h => value 0 2 (by omega)
    h.1.1.1.1.1.1.1.1 N₁₄ (by have := (hsizes h.1.1.1.1.1.1.1.1).1; omega)).bind₀ fun rest m₁₅ l₁₅ => ?_
  refine (Sim.extSub (fun h => h.2) fun h => top h.1.1.1.1.1.1.1.1.1 (N₁₄.trans l₁₅)).bind₀ fun block m₁₆ l₁₆ => ?_
  have N₁₆ : n ≤ m₁₆ := N₁₄.trans (l₁₅.trans l₁₆)
  refine Sim.pure fun h => ?_
  have hP : P := h.1.1.1.1.1.1.1.1.1.1
  exact ⟨(tree 0 (by omega) hP N₁₆).2.2.1.slice 0 layout.slotVariables, h.2, (tree 1 (by omega) hP N₁₆).2.2.1,
    value 1 0 (by omega) hP N₁₆ (by have := (hsizes hP).2.1; omega)⟩

end Check

section Stage

variable [CharP F goldilocksModulus] {c : Ctx F} {n : ℕ} {P : Prop}

/-- **The late stage** (`verifier.rs:251-260`). Once the circuit's checks
hold, the replay's late stage runs from the matching transcript, the
circuit's openings are the replay's, its pairs are the proof's, and the
duplexes still match. `short` and `named` are `Structure::check`
(`matrix.rs:142-182`). -/
theorem lateStage_ensures {rel : Relation} {tau : Array ExtVar} {evalA : Array (ExtVar × ExtVar)} {late : GkrView}
    {lateValues : Array (ExtVar × ExtVar)} {proof : Replay.Proof} {lin : Replay.LinearOut} {d : Duplex}
    {t : Replay.Transcript} (hp : Permutes c.perm) (hd : P → IsDuplex c n d t.duplex)
    (htau : P → IsExts c n tau lin.tau) (tauSize : P → lin.tau.size = D)
    (hevalA : P → evalA.size = rel.matrices ∧ ∀ j < evalA.size, IsPair c n evalA[j]! (lin.mixing.evalA j))
    (hlate : P → IsGkr c n late proof.late)
    (hvalues : P → List.Forall₂ (IsPair c n) lateValues.toList proof.lateValues.toList)
    (read : Replay.readProof rel proof = .ok ()) (short : ∀ cl ∈ rel.key.layout.classes, cl.len ≤ D)
    (named : ∀ s ∈ rel.key.layout.segments, s.classIndex < rel.key.layout.classes.size ∧ s.lane < D) :
    Ensures c n P ((Circuit.lateStage rel tau evalA late lateValues).run d)
      (fun res m => ∃ out t', Runs (Replay.lateStage rel proof lin) t out t' ∧ IsLateOut c m res.1 out ∧
        out.pairs = proof.lateValues ∧ IsDuplex c m res.2 t'.duplex) := by
  have shape := Bridge.readProof_ok read
  obtain ⟨rootsSize, -, -⟩ := Bridge.readGkr_ok shape.late
  set glanes := rel.key.layout.classes.map fun cl => Replay.laneSplit lin.tau cl.len cl.ratio
  have lanesNamed : ∀ s ∈ rel.key.layout.segments, s.classIndex < glanes.size ∧
      s.lane < glanes[s.classIndex]!.1.size ∧ s.lane < glanes[s.classIndex]!.2.size := fun s hs => by
    obtain ⟨inside, lane⟩ := named s hs
    rw [getElem!_pos glanes _ (by simpa [glanes] using inside), Array.getElem_map]
    simp [glanes, inside, lane, Replay.laneSplit]
  unfold Circuit.lateStage
  refine Ensures.tbind (Sim.sampleExt hp hd).ensures fun beta d₁ m₁ l₁ => ?_
  set gbeta : Ext := ⟨t.duplex.sample.1, t.duplex.sample.2.sample.1, t.duplex.sample.2.sample.2.sample.1⟩
  set t₁ : Replay.Transcript := ⟨t.duplex.sample.2.sample.2.sample.2, t.draws.push (.ext gbeta)⟩
  refine Ensures.tbind (gkrVerify_ensures (t := t₁) hp (fun h => h.2.1.2)
    (fun h => ((hlate h.1).mono l₁).entries.1) (fun h => ((hlate h.1).mono l₁).entries.2) shape.late)
    fun claims d₂ m₂ l₂ => ?_
  refine (Ensures.exists_and (X := fun x : Replay.GkrClaims × Replay.Transcript =>
      Runs (Replay.gkrVerify proof.late rel.lateShapes) t₁ x.1 x.2 ∧ IsGkrClaims c m₂ claims x.1 ∧
        IsDuplex c m₂ d₂ x.2.duplex) (⟨#[], #[], #[], #[], #[]⟩, t) fun x => ?_).pre
    fun h => ⟨h.1, let ⟨gc, t₂, hx⟩ := h.2; ⟨(gc, t₂), hx⟩⟩
  obtain ⟨gc, t₂⟩ := x
  dsimp only
  refine Ensures.pre (P' := P ∧ IsExt c m₂ beta gbeta ∧ Runs (Replay.gkrVerify proof.late rel.lateShapes) t₁ gc t₂ ∧
    IsGkrClaims c m₂ claims gc ∧ IsDuplex c m₂ d₂ t₂.duplex) ?_
    fun h => ⟨h.1.1, h.1.2.1.1.mono l₂, h.2.1, h.2.2.1, h.2.2.2⟩
  have L₂ : n ≤ m₂ := l₁.trans l₂
  refine Ensures.tbind (Ensures.lift (laneTables_sim (fun h => (htau h.1).mono L₂) (fun h => tauSize h.1)
    short).ensures) fun lanes d₃ m₃ l₃ => ?_
  have L₃ : n ≤ m₃ := L₂.trans l₃
  refine Ensures.tbind (Ensures.lift (checkLate_ensures (mix := lin.mixing) (gpairs := proof.lateValues)
    (fun h => h.1.2.1.mono l₃)
    (fun h => ⟨by rw [shape.lateValues, (hevalA h.1.1).1], fun j hj => ((hevalA h.1.1).2 j hj).mono L₃⟩)
    (fun h => h.2.1.1) lanesNamed
    (fun h => ⟨h.1.2.2.2.1.mono l₃, by
      rw [h.1.2.2.2.1.1, (gkrVerify_sizes shape.late h.1.2.2.1).1, rootsSize]
      rfl⟩)
    (fun h => by
      obtain ⟨-, sizes⟩ := gkrVerify_sizes shape.late h.1.2.2.1
      have s₀ := sizes 0 (by rw [Bridge.lateShapes_size]; omega)
      have s₁ := sizes 1 (by rw [Bridge.lateShapes_size]; omega)
      have f₀ : (rel.lateShapes[0]!).factors = 2 := rfl
      have f₁ : (rel.lateShapes[1]!).factors = 1 := rfl
      have d₀ : (rel.lateShapes[0]!).depth = rel.key.layout.slotVariables + 1 := rfl
      rw [f₀, d₀] at s₀
      rw [f₁] at s₁
      exact ⟨by omega, by omega, by omega⟩)
    fun h => by
      obtain ⟨size, pairs⟩ := forall₂_entries (hvalues h.1.1)
      exact ⟨size, fun j hj => by rw [getElem!_pos _ j hj]; exact (pairs j hj).mono L₃⟩))
    fun openings d₄ m₄ l₄ => ?_
  refine Ensures.tbind (Sim.observePairs (r := t₂.duplex) hp (fun h => by
      have e : d₄ = d₂ := h.2.2.trans h.1.2.2
      rw [e]
      exact h.1.1.2.2.2.2.mono (l₃.trans l₄))
    fun h => (hvalues h.1.1.1).imp fun _ _ q => q.mono (L₃.trans l₄)).ensures fun _ d₅ m₅ l₅ => ?_
  rw [StateT.run_pure]
  refine Ensures.pure fun h => ?_
  obtain ⟨⟨⟨⟨-, -, run, -, -⟩, -⟩, ⟨checkOk, hopen⟩, -⟩, hd₅, -⟩ := h
  refine ⟨⟨gbeta, gc, proof.lateValues, Bridge.lateOpeningsOf rel.key.layout gbeta gc⟩, ?_⟩
  refine ⟨⟨((Replay.flatPairs proof.lateValues).flatMap fun v => #[v.c0, v.c1, v.c2]).foldl Replay.Duplex.observe
      t₂.duplex, t₂.draws⟩, ?_, ?_, ?_, ?_⟩
  · unfold Replay.lateStage
    exact runs_bind (runs_sampleExt t) (runs_bind run (runs_bind (runs_liftExcept checkOk _)
      (runs_bind (runs_observeExts _ _) (runs_pure _ _))))
  · exact hopen.mono l₅
  · rfl
  · exact hd₅

end Stage

end CompressionLean.Circuit
