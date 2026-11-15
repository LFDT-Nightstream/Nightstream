import NightstreamFPrime.Export.Codec
import NightstreamFPrime.Lifecycle.BoundedDepth.CircuitKeyDigest

/-! Vectors of the hashed encodings that Rust computes outside the step packages
(`SECURITY_MODEL.md`, section Hashed encodings). The value is `[schema, circuitKeys]`.
`circuitKeys` holds `[identity, digest 0, digest 1]` of
`Lifecycle.BoundedDepth.CircuitKeyDigest.digest` for each identity of `identities`. Rust must
give the same words (`crates/nightstream/tests/bounded_depth_lifecycle/circuit_key.rs`). -/

namespace NightstreamFPrime.Export.EncodingParity

open NightstreamFPrime.Export.Codec
open NightstreamFPrime.Spec
open NightstreamFPrime.Lifecycle.BoundedDepth

def schema : Nat := 1

def words (values : List F) : Value :=
  .array (values.map fun value => .atom value.val)

def identities : List (List F) :=
  [List.replicate 4 0, (List.range 4).map Poseidon2.ofNat,
    (List.range 4).map fun word => Poseidon2.ofNat (fieldModulus - 1 - word)]

def circuitKeys : Value :=
  .array (identities.map fun identity =>
    .array [words identity, words (CircuitKeyDigest.digest identity 0),
      words (CircuitKeyDigest.digest identity 1)])

def parityValue : Value := .array [.atom schema, circuitKeys]

end NightstreamFPrime.Export.EncodingParity
