import NightstreamFPrime.Export.ParityEmitter
import NightstreamFPrime.Spec.Poseidon2

/-! Known answers of `Spec.Poseidon2.hash` for the Rust test `sponge_hash_matches_lean_reference`
(`crates/neo-ccs/tests/poseidon2_lean_vectors.rs`): for each length from 0 to 25, the words
`0, 1, 2, …` and the zero words. The value is `[q, [[input, digest], …]]`. -/

namespace NightstreamFPrime.Export.Poseidon2Vectors

open NightstreamFPrime.Spec
open NightstreamFPrime.Export.Codec

def inputs : List (List F) :=
  (List.range 26).map (fun length => (List.range length).map Poseidon2.ofNat) ++
    (List.range 26).map (fun length => List.replicate length 0)

def words (values : List F) : Value :=
  .array (values.map fun value => .atom value.val)

def vectors : Value :=
  .array [.atom fieldModulus,
    .array (inputs.map fun input => .array [words input, words (Poseidon2.hash input)])]

end NightstreamFPrime.Export.Poseidon2Vectors

def main (arguments : List String) : IO UInt32 :=
  NightstreamFPrime.Export.ParityEmitter.run "emitted_poseidon2_hash_vectors"
    NightstreamFPrime.Export.Poseidon2Vectors.vectors arguments
