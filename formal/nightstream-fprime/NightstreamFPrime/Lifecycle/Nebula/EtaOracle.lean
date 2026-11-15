import NightstreamFPrime.Lifecycle.Nebula.Framing
import NightstreamFPrime.Lifecycle.Nebula.Carry
import NightstreamFPrime.Spec.Nebula.EtaLine
import NightstreamFPrime.Spec.FieldExtensionRing

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

open scoped NightstreamFPrime.Spec.FieldExtensionRing in
/-- The oracles are finite. Instance search on `Point → K × K` exceeds its size limit. -/
noncomputable instance oracleFintype : Fintype Oracle := Pi.instFintype

/-- The oracle point of an `η` input. -/
def pointOf (input : EtaInput Digest) : Point :=
  (input.planDigest, natWord input.ts, input.opsRoot, input.memRoot, input.finalRoot)

/-- The `η` input of an oracle point. -/
def inputOf (point : Point) : EtaInput Digest :=
  ⟨point.1, point.2.1.val, point.2.2.1, point.2.2.2.1, point.2.2.2.2⟩

/-- An `η` input with a canonical timestamp is the input of its oracle point. -/
theorem inputOf_pointOf {input : EtaInput Digest} (small : input.ts < fieldModulus) :
    inputOf (pointOf input) = input := by
  cases input
  simp only [inputOf, pointOf, natWord_val small]

/-- The challenge function of an oracle: every segment reads `oracle` at its input's point. -/
def etaOf (oracle : Oracle) : ℕ → EtaInput Digest → K × K := fun _ input => oracle (pointOf input)

/-- The memory context with `η` read from `oracle`. -/
def romContext (p : Plan) (oracle : Oracle) : Context K Digest :=
  (context p).withEta (etaOf oracle)

/-- The Poseidon2 `η` transcript as an oracle. -/
def deployed : Oracle := fun point => etaChallenges (inputOf point)

theorem deployed_pointOf (input : EtaInput Digest) :
    deployed (pointOf input) = etaChallenges input := by
  have word : natWord (natWord input.ts).val = natWord input.ts := by
    apply Fin.ext
    simp [natWord, Poseidon2.ofNat, Nat.mod_mod_of_dvd]
  unfold deployed inputOf pointOf etaChallenges
  simp only [etaInputBlocks, word]

/-- At the deployed oracle, the challenge function is the deployed one. -/
theorem etaOf_deployed (p : Plan) : etaOf deployed = (context p).eta := by
  funext _ input
  exact deployed_pointOf input

/-- At the deployed oracle, the model is the deployed memory context. -/
theorem romContext_deployed (p : Plan) : romContext p deployed = context p := by
  show (⟨p, hash, (fun _ input => deployed (pointOf input)), planDigest p.toGeometry⟩ :
      Context K Digest) = ⟨p, hash, fun _ => etaChallenges, planDigest p.toGeometry⟩
  congr 1
  funext _ input
  exact deployed_pointOf input

/-- Two `η` inputs with canonical timestamps have the same oracle point only if they are equal. -/
theorem pointOf_injective {input input' : EtaInput Digest} (small : input.ts < fieldModulus)
    (small' : input'.ts < fieldModulus) (same : pointOf input = pointOf input') :
    input = input' := by
  rw [← inputOf_pointOf small, same, inputOf_pointOf small']

end NightstreamFPrime.Lifecycle.Nebula.EtaOracle
