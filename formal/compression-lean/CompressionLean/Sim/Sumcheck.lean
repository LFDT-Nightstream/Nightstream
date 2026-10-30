import CompressionLean.Sim.Merkle
import CompressionLean.Sim.Tracks
import CompressionLean.Ports.Circuit.Sumcheck

namespace CompressionLean.Sim

open CompressionLean.Spec
open CompressionLean.Circuit
open CompressionLean.Replay (Ext)
open NightstreamFPrime.Spec (goldilocksModulus)

variable {F : Type} [CommRing F]


abbrev observeAllExts (vs : Array ExtVar) : Transcript PUnit :=
  forIn vs PUnit.unit fun value _ => do
    Duplex.observeExt value
    Pure.pure (ForInStep.yield PUnit.unit)

instance sumcheckReplay_appends (rounds : Array (Array ExtVar)) (claim : ExtVar) (d : Duplex) :
    Appends ((Circuit.sumcheckReplay rounds claim).run d) := by
  unfold Circuit.sumcheckReplay; appends

section Rounds

variable [CharP F goldilocksModulus] {c : Ctx F} {n : ℕ}

theorem observeExts_ensures {d : Duplex} {r : Replay.Duplex} {vs : Array ExtVar} {gs : Array Ext}
    (hp : Permutes c.perm) (hd : IsDuplex c n d r) (hv : IsExts c n vs gs) :
    Ensures c n ((observeAllExts vs).run d)
      fun res m => IsDuplex c m res.2 ((gs.flatMap fun g => #[g.c0, g.c1, g.c2]).foldl Replay.Duplex.observe r) := by
  rw [observeAllExts, ← Array.forIn_toList]
  refine (Ensures.loopT (fun k _ e m => IsDuplex c m e
      (((gs.toList.take k).flatMap fun g => [g.c0, g.c1, g.c2]).foldl Replay.Duplex.observe r))
    (by simpa using hd) fun k hk _ e m _ he => ?_).post fun _ _ q => ?_
  · have hk' : k < vs.size := by simpa using hk
    have hg : k < gs.size := hv.1 ▸ hk'
    rw [StateT.run_bind]
    refine (Duplex.observeExt_ensures (e := gs[k]!) hp he ?_).bind fun res _ _ hres => ?_
    · rw [Array.getElem_toList, ← getElem!_pos vs k hk']
      exact (hv.2 k hk').mono
    · rw [StateT.run_pure]
      refine Ensures.pure ⟨rfl, ?_⟩
      rw [List.take_add_one, List.getElem?_eq_getElem (by simpa using hg), Option.toList_some, List.flatMap_append,
        List.foldl_append, Array.getElem_toList, ← getElem!_pos gs k hg]
      simp only [List.flatMap_cons, List.flatMap_nil, List.append_nil, List.foldl_cons, List.foldl_nil]
      exact hres
  · rw [show vs.toList.length = gs.toList.length by simp [hv.1], List.take_length] at q
    rw [← Array.foldl_toList, Array.toList_flatMap]
    exact q

theorem observeExts_tracks {vs : Array ExtVar} {gs : Array Ext} (hp : Permutes c.perm) (hv : IsExts c n vs gs) :
    Tracks c n (observeAllExts vs) (Replay.observeExts gs) fun _ _ _ => True := fun _ tr hd =>
  (observeExts_ensures hp hd hv).post fun _ _ q => ⟨(), _, Bridge.runs_observeExts gs tr, q, trivial⟩

theorem sumcheckReplay_tracks {rounds : Array (Array ExtVar)} {grounds : Array (Array Ext)} {claim : ExtVar}
    {gclaim : Ext} (hp : Permutes c.perm) (hr : rounds.size = grounds.size)
    (hm : ∀ i < rounds.size, IsExts c n rounds[i]! grounds[i]!) (hne : ∀ g ∈ grounds, 0 < g.size)
    (hc : IsExt c n claim gclaim) :
    Tracks c n (Circuit.sumcheckReplay rounds claim) (Replay.sumcheckReplay grounds gclaim)
      fun res out m => IsExts c m res.1 out.1 ∧ IsExt c m res.2 out.2 := by
  intro d t hd
  unfold Circuit.sumcheckReplay
  dsimp only
  refine Ensures.tbind (Q := fun r m => ∃ point final t', Runs (Replay.sumcheckRounds grounds.toList #[] gclaim) t
      (point, final) t' ∧ IsExts c m r.1.1 point ∧ IsExt c m r.1.2 final ∧ IsDuplex c m r.2 t'.duplex) ?_
    fun res e m _ hres => ?_
  · rw [← Array.forIn_toList (xs := rounds)]
    refine (Ensures.loopT (fun k acc e m => ∃ point final t',
        Runs (Replay.sumcheckRounds (grounds.toList.take k) #[] gclaim) t (point, final) t' ∧
          IsExts c m acc.1 point ∧ IsExt c m acc.2 final ∧ IsDuplex c m e t'.duplex)
      ⟨#[], gclaim, t, Bridge.runs_pure _ _, IsExts.empty, hc, hd⟩
      fun k hk acc e m _ hI => ?_).post fun r m q => ?_
    · obtain ⟨gpt, gcl, tk, hrun, hpt, hcl, hdk⟩ := hI
      have hk' : k < rounds.size := by simpa using hk
      have hg : k < grounds.size := hr ▸ hk'
      have hmsg : IsExts c m rounds.toList[k] grounds[k]! := by
        rw [Array.getElem_toList, ← getElem!_pos rounds k hk']
        exact (hm k hk').mono
      have h0 : IsExt c m rounds.toList[k][0]! (grounds[k]!)[0]! :=
        hmsg.2 0 (by rw [hmsg.1, getElem!_pos grounds k hg]; exact hne _ (Array.getElem_mem _))
      refine Ensures.liftBind (extSub_ensures hcl h0) fun atOne _ _ hat => ?_
      refine Ensures.tbind (observeExts_ensures hp hdk.mono hmsg.mono) fun _ _ _ _ hd₂ => ?_
      refine Ensures.tbind (Duplex.sampleExt_ensures hp hd₂) fun r _ _ _ h₃ => ?_
      obtain ⟨hr₃, hd₃⟩ := h₃
      refine Ensures.liftBind (interpolate_ensures (gvalues := Replay.roundValues grounds[k]! gcl) ?_ hr₃)
        fun cl' _ _ hcl' => ?_
      · unfold Replay.roundValues
        exact ((IsExts.empty.push h0.mono).push hat.mono).append (hmsg.mono.extract 1)
      rw [StateT.run_pure]
      refine Ensures.pure ⟨rfl, ?_⟩
      rw [List.take_add_one, List.getElem?_eq_getElem (by simpa using hg), Option.toList_some,
        Bridge.sumcheckRounds_append, Array.getElem_toList, ← getElem!_pos grounds k hg]
      exact ⟨_, _, _, Bridge.runs_bind hrun rfl, hpt.mono.push hr₃.mono, hcl', hd₃.mono⟩
    · rw [show rounds.toList.length = grounds.toList.length by simp [hr], List.take_length] at q
      exact q
  · obtain ⟨point, final, t', hrun, hpt, hfin, hd'⟩ := hres
    rw [StateT.run_pure]
    exact Ensures.pure ⟨(point, final), t', hrun, hd', hpt, hfin⟩

end Rounds

end CompressionLean.Sim
