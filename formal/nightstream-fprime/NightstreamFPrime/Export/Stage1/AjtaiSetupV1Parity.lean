import NightstreamFPrime.Export.Codec
import NightstreamFPrime.Export.Stage1.Poseidon2HashChainV1SetupAuthority
import NightstreamFPrime.Spec.AjtaiSetupV1

namespace NightstreamFPrime.Export.Stage1.AjtaiSetupV1Parity

open NightstreamFPrime.Export.Codec
open NightstreamFPrime.Spec (fieldModulus ringDegree)
open NightstreamFPrime.Spec.AjtaiSetupV1

def schema : Nat := 5

def testSeedBytes : List Nat := List.range 32

def testSeed : Seed where
  bytes := testSeedBytes
  length_eq := by simp [testSeedBytes]
  canonical := by
    intro byte member
    simp [testSeedBytes] at member
    omega

def natListValue (values : List Nat) : Value :=
  .array (values.map Value.atom)

def coordinateValue (seed : Seed) (row block lane : Nat) : Value :=
  .array [.atom row, .atom block, .atom lane,
    .atom (elementCoefficientNat seed.bytes row block lane)]

/-- Streams of `keyWords` words for the sampler `keptCoefficientNat`. No index of the production
key gives a rejected word in practice (59 of the `2^64` words), so these words are synthetic:
one rejected word; alternate kept and rejected words, so the fallback gives the last
coefficients; 20 rejected words, so the last word is the last coefficient; 21 rejected words, so
the fallback gives one coefficient; each word rejected; the words next to `q`. -/
def samplerStreams : List (List Nat) :=
  ([fun word => if word = 0 then fieldModulus else word,
    fun word => if word % 2 = 0 then fieldModulus - 1 - word else fieldModulus + word % 59,
    fun word => if word < 20 then 2 ^ 64 - 1 - word else 1000 + word,
    fun word => if word % 4 = 3 then fieldModulus + word % 59 else fieldModulus - 1 - word,
    fun _ => 2 ^ 64 - 1,
    fun word => fieldModulus - 2 + word % 4] : List (Nat → Nat)).map
    fun stream => (List.range keyWords).map stream

def samplerValue (words : List Nat) : Value :=
  .array [natListValue words,
    natListValue ((List.range ringDegree).map (keptCoefficientNat words))]

/-- Entry 3 is the FIPS 202 SHAKE128 example for the empty message. -/
def parityValue : Value :=
  .array [
    .atom schema,
    natListValue setupIdBytes,
    natListValue testSeed.bytes,
    natListValue (Shake128.bytes [] 32),
    natListValue Poseidon2HashChainV1SetupAuthority.productionSeed.bytes,
    .array [
      coordinateValue Poseidon2HashChainV1SetupAuthority.productionSeed 0 0 0,
      coordinateValue Poseidon2HashChainV1SetupAuthority.productionSeed 0 0 (ringDegree - 1),
      coordinateValue Poseidon2HashChainV1SetupAuthority.productionSeed
        1 32768 17,
      coordinateValue Poseidon2HashChainV1SetupAuthority.productionSeed
        (Poseidon2HashChainV1SetupAuthority.verifierRows - 1)
        (Poseidon2HashChainV1SetupAuthority.messageColumns - 1) (ringDegree - 1)],
    natListValue Poseidon2HashChainV1SetupAuthority.authorityNats,
    natListValue (Shake128.bytes
      (elementInput Poseidon2HashChainV1SetupAuthority.productionSeed.bytes 1 32768)
      (8 * keyWords)),
    .array (samplerStreams.map samplerValue)]

end NightstreamFPrime.Export.Stage1.AjtaiSetupV1Parity
