import NightstreamFPrime.Lifecycle.BoundedDepth.Step
import NightstreamFPrime.Spec.Sponge

/-! The circuit key digest (`SPEC.md` section 4.4, step 2): the Poseidon2 hash of the domain
chunk, the package identity and the circuit index. Rust `circuit_key`
(`crates/nightstream/src/lifecycle/bounded_depth/mod.rs`) computes the same words.

The hash input fixes the identity and the circuit for an identity of any length
(`input_injective`), so two keys with one digest are equal or give two distinct hash inputs
with one hash (`digest_binds`). `Spec.Sponge.collision_blocks` turns the second case into a
sponge collision. The digest has 4 words (`digest_length`), so the keys of an identity have the
shape that the step results need (`keys_shaped`). -/

set_option autoImplicit false

namespace NightstreamFPrime.Lifecycle.BoundedDepth.CircuitKeyDigest

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Lifecycle.PaperAlgebra

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤ RingCarrierLayout.carrierWidth logicalWidth}

def domain : List F := domainChunk "Nightstream/OptionB/circuit-key/v1"

/-- The hash input of circuit `circuit`: 0 is the light circuit and 1 is the refresh circuit. -/
def input (identity : List F) (circuit : Fin 2) : List F :=
  domain ++ identity ++ [Poseidon2.ofNat circuit.val]

def digest (identity : List F) (circuit : Fin 2) : KeyDigest :=
  Poseidon2.hash (input identity circuit)

theorem digest_length (identity : List F) (circuit : Fin 2) :
    (digest identity circuit).length = 4 :=
  Poseidon2.hash_length _

theorem domain_prefix (identity : List F) (circuit : Fin 2) : domain <+: input identity circuit := by
  simp only [input, List.append_assoc]
  exact List.prefix_append _ _

/-- The hash input fixes the identity and the circuit. -/
theorem input_injective {identity identity' : List F} {circuit circuit' : Fin 2}
    (same : input identity circuit = input identity' circuit') :
    identity = identity' ∧ circuit = circuit' := by
  simp only [input, List.append_assoc] at same
  obtain ⟨identities, last⟩ := List.append_inj' (List.append_cancel_left same) rfl
  refine ⟨identities, ?_⟩
  have word : Poseidon2.ofNat circuit.val = Poseidon2.ofNat circuit'.val := by simpa using last
  fin_cases circuit <;> fin_cases circuit' <;> first | rfl | exact absurd word (by decide)

/-- Two keys with one digest are equal, or their hash inputs are distinct and have one Poseidon2
hash. -/
theorem digest_binds {identity identity' : List F} {circuit circuit' : Fin 2}
    (same : digest identity circuit = digest identity' circuit') :
    (identity, circuit) = (identity', circuit') ∨
      (input identity circuit ≠ input identity' circuit' ∧
        Poseidon2.hash (input identity circuit) = Poseidon2.hash (input identity' circuit')) := by
  by_cases inputs : input identity circuit = input identity' circuit'
  · obtain ⟨identities, circuits⟩ := input_injective inputs
    exact Or.inl (by rw [identities, circuits])
  · exact Or.inr ⟨inputs, same⟩

/-- The circuit keys of a package identity: the digest of each circuit with its relation. -/
def keys (identity : List F)
    (relations : Fin 2 → ProductionKey.LogicalRelation logicalWidth publicFits) :
    Fin 2 → CircuitKey logicalWidth publicFits :=
  fun circuit => ⟨digest identity circuit, relations circuit⟩

theorem keys_shaped (identity : List F)
    (relations : Fin 2 → ProductionKey.LogicalRelation logicalWidth publicFits) (key : Fin 2) :
    (keys identity relations key).digest.length = 4 :=
  digest_length identity key

end NightstreamFPrime.Lifecycle.BoundedDepth.CircuitKeyDigest
