import CompressionLean.Circuit.SimRun
import CompressionLean.Circuit.Verifier

/-!
Owns the simulation of the verifier's own transcript steps
(`Verifier.lean`, `Early.lean`, `Whir.lean`) against the replay's duplex:
constants observed as words (`absorb`, the relation words of `verify`) and
a commitment's root binding (`observeRoot`). These are the first steps of
the stage lemmas that `VerifyAgrees` (`Final/Agrees.lean`) waits for.
-/

namespace CompressionLean.Circuit

open CompressionLean.Replay (Gl)
open NightstreamFPrime.Spec (goldilocksModulus)

variable {F : Type} [CommRing F]

namespace Sim

variable [CharP F goldilocksModulus] {c : Ctx F} {n : ℕ} {P : Prop}

/-- `absorb`: each word as a constant, observed. -/
theorem absorb {d : Duplex} {r : Replay.Duplex} (hp : Permutes c.perm) (hd : P → IsDuplex c n d r)
    (words : Array Gl) :
    Sim c n P ((Whir.absorb words).run d) (fun res m => IsDuplex c m res.2 (words.foldl Replay.Duplex.observe r))
      True := by
  unfold Whir.absorb
  rw [← Array.forIn_toList, ← Array.foldl_toList]
  refine Sim.tbind₀ (Q := fun res m => IsDuplex c m res.2 (words.toList.foldl Replay.Duplex.observe r)) ?_
    fun _ e m _ => ?_
  · refine ((Sim.loopT (I := fun k _ e m => IsDuplex c m e ((words.toList.take k).foldl Replay.Duplex.observe r))
      (D := fun _ => True) fun k hk _ e m _ => ?_).pre fun h => ⟨h, by simpa using hd h⟩).post
      (fun _ _ _ h => by rw [List.take_length] at h; exact h) fun _ => by simp
    refine Sim.tbind₀ (Sim.lift (Sim.const words.toList[k])) fun w e₁ m₁ l₁ => ?_
    refine Sim.tbind₀ (observe hp (r := (words.toList.take k).foldl Replay.Duplex.observe r)
      (fun h => (show e₁ = e from h.2.2) ▸ h.1.2.mono l₁) fun h => h.2.1)
      fun _ e₂ m₂ l₂ => ?_
    rw [StateT.run_pure]
    refine Sim.pure fun h => ⟨rfl, ?_⟩
    rw [List.take_succ_eq_append_getElem hk, List.foldl_append]
    exact h.2
  · rw [StateT.run_pure]
    exact Sim.pure fun h => h.2

/-- `observe_root`: the separator as constants, then the root's lanes. -/
theorem observeRoot {d : Duplex} {r : Replay.Duplex} (hp : Permutes c.perm) (hd : P → IsDuplex c n d r)
    (seed : Array Gl) {root : DigestVar} {words : Array Gl} (hw : P → IsWords c n (Array.ofFn root) words) :
    Sim c n P ((Circuit.observeRoot seed root).run d)
      (fun res m => IsDuplex c m res.2 (words.foldl Replay.Duplex.observe (seed.foldl Replay.Duplex.observe r)))
      True := by
  have split : Circuit.observeRoot seed root = (Whir.absorb seed >>= fun _ => Duplex.observeSlice (Array.ofFn root)) :=
    by unfold Circuit.observeRoot Whir.absorb; simp only [bind_assoc, pure_bind]
  rw [split]
  exact Sim.tbind₀ (absorb hp hd seed) fun _ e m l => observeSlice hp (fun h => h.2) fun h => (hw h.1).mono l

end Sim

end CompressionLean.Circuit
