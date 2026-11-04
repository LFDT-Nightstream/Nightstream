import NightstreamFPrime.Lifecycle.Nebula.Framing
import NightstreamFPrime.Lifecycle.Nebula.Carry
import NightstreamFPrime.Spec.Nebula.EtaLine

/-! A random-oracle model of the Nebula `η` transcript (CW-10.8). The step circuit computes the
product-test challenge `η` with Poseidon2 from the plan digest, the open timestamp and the three
roots (`etaChallenges`, `MemoryApp.etaBlocks`). The model replaces that hash by an oracle on the
same inputs (`Point`, `pointOf`): `romContext p oracle` is the memory context `context p` with
`η` read from `oracle`. The sponge itself is an oracle (`deployed`) on which the model is the
deployed context (`romContext_deployed`), so the model's events at that oracle are the deployed
events. -/

set_option autoImplicit false

namespace NightstreamFPrime.Lifecycle.Nebula.EtaOracle

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Nebula
open NightstreamFPrime.Lifecycle.Nebula

/-- The inputs of the `η` transcript: the plan digest, the open timestamp as a word, and the ops,
memory and final roots. -/
abbrev Point := Digest × F × Digest × Digest × Digest

/-- An oracle for the `η` transcript. -/
abbrev Oracle := Point → K × K

/-- The oracle point of an `η` input. -/
def pointOf (input : EtaInput Digest) : Point :=
  (input.planDigest, natWord input.ts, input.opsRoot, input.memRoot, input.finalRoot)

/-- The memory context with `η` read from `oracle`. -/
def romContext (p : Plan) (oracle : Oracle) : Context K Digest :=
  (context p).withEta fun _ input => oracle (pointOf input)

/-- The Poseidon2 `η` transcript as an oracle. -/
def deployed : Oracle := fun point =>
  etaChallenges ⟨point.1, point.2.1.val, point.2.2.1, point.2.2.2.1, point.2.2.2.2⟩

theorem deployed_pointOf (input : EtaInput Digest) :
    deployed (pointOf input) = etaChallenges input := by
  have word : natWord (natWord input.ts).val = natWord input.ts := by
    apply Fin.ext
    simp [natWord, Poseidon2.ofNat, Nat.mod_mod_of_dvd]
  unfold deployed pointOf etaChallenges
  simp only [etaInputBlocks, word]

/-- At the deployed oracle, the model is the deployed memory context. -/
theorem romContext_deployed (p : Plan) : romContext p deployed = context p := by
  show (⟨p, hash, (fun _ input => deployed (pointOf input)), planDigest p.toGeometry⟩ :
      Context K Digest) = ⟨p, hash, fun _ => etaChallenges, planDigest p.toGeometry⟩
  congr 1
  funext _ input
  exact deployed_pointOf input

/-- Two `η` inputs with canonical timestamps have the same oracle point only if they are equal. -/
theorem pointOf_injective {input input' : EtaInput Digest} (small : input.ts < goldilocksModulus)
    (small' : input'.ts < goldilocksModulus) (same : pointOf input = pointOf input') :
    input = input' := by
  simp only [pointOf, Prod.mk.injEq] at same
  obtain ⟨plan, ts, ops, mem, final⟩ := same
  have tsEq : input.ts = input'.ts := by
    have := congrArg Fin.val ts
    rwa [natWord_val small, natWord_val small'] at this
  cases input
  cases input'
  simp_all

end NightstreamFPrime.Lifecycle.Nebula.EtaOracle
