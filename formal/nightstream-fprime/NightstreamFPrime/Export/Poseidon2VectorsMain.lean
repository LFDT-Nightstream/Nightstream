import NightstreamFPrime.Export.ParityEmitter
import NightstreamFPrime.Spec.Poseidon2

/-! Cross-check vectors of the Poseidon2 instance that Nightstream and ProtogaLattice share
(`decisions/poseidon2-p64-instance.md`). The value is
`[q, [width, rate, digestLen, fullRounds, partialRounds], permutations, hashes]`. `permutations`
holds `[input, Spec.Poseidon2.permute input]` for the words `0, 1, …, 15` and three chained calls
from it, the zero state and the state of `q − 1` words. `hashes` holds
`[input, Spec.Poseidon2.hash input]` for each length from 0 to 25, with the words `0, 1, 2, …` and
with zero words. The Rust test `crates/neo-ccs/tests/poseidon2_lean_vectors.rs` reads both. -/

namespace NightstreamFPrime.Export.Poseidon2Vectors

open NightstreamFPrime.Spec
open NightstreamFPrime.Export.Codec

def chain : Nat → Poseidon2.State → List Poseidon2.State
  | 0, _ => []
  | count + 1, state => state :: chain count (Poseidon2.permute state)

def permutationInputs : List Poseidon2.State :=
  chain 4 ((List.range Poseidon2.width).map Poseidon2.ofNat) ++
    [Poseidon2.zeroState, List.replicate Poseidon2.width (Poseidon2.ofNat (fieldModulus - 1))]

def hashInputs : List (List F) :=
  (List.range 26).map (fun length => (List.range length).map Poseidon2.ofNat) ++
    (List.range 26).map (fun length => List.replicate length 0)

def words (values : List F) : Value :=
  .array (values.map fun value => .atom value.val)

def vectors : Value :=
  .array [.atom fieldModulus,
    .array ([Poseidon2.width, Poseidon2.rate, Poseidon2.digestLen, 2 * Poseidon2.halfFullRounds,
      Poseidon2.partialRounds].map .atom),
    .array (permutationInputs.map fun input => .array [words input, words (Poseidon2.permute input)]),
    .array (hashInputs.map fun input => .array [words input, words (Poseidon2.hash input)])]

end NightstreamFPrime.Export.Poseidon2Vectors

def main (arguments : List String) : IO UInt32 :=
  NightstreamFPrime.Export.ParityEmitter.run "emitted_poseidon2_vectors"
    NightstreamFPrime.Export.Poseidon2Vectors.vectors arguments
