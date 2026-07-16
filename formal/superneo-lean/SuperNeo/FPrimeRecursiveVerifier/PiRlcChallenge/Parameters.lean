/-!
Owns: the fixed dimensions shared by every Pi_RLC challenge branch.

Does not own: sampler equations, transcript state, or production configuration.

Emits constraints: no.

Authority boundary: these constants are the theorem-level fixed shape; a
concrete bridge must prove that Rust uses the same values.

| Parameter | Value | Mathematical role |
|---|---:|---|
| `chunkModulus` | 65,536 | Canonical 16-bit chunk range |
| `rejectionBucket` | 65,535 | Unique rejected value |
| `alphabetSize` | 5 | Centered challenge alphabet size |
| `quotientBits` | 14 | Width of the mod-5 quotient witness |
| `chunksPerSample` | 64 | Four digests times sixteen chunks |
| `outputLength` | 54 | Coefficients selected per rho |
| `slackBits` | 4 | Width of accepted-count slack |
| `maxRejections` | 10 | Largest rejection count accepted by the fixed sample |
| `selectionWindow` | 11 | Candidate indices needed for each selected output |
| `rhoCount` | 15 | Number of transcript-chained Pi_RLC challenges |
-/

namespace SuperNeo.FPrimeRecursiveVerifier.PiRlcChallenge

def chunkModulus : Nat := 65_536
def rejectionBucket : Nat := 65_535
def alphabetSize : Nat := 5
def quotientBits : Nat := 14
def chunksPerSample : Nat := 64
def outputLength : Nat := 54
def slackBits : Nat := 4
def maxRejections : Nat := chunksPerSample - outputLength
def selectionWindow : Nat := maxRejections + 1
def rhoCount : Nat := 15

end SuperNeo.FPrimeRecursiveVerifier.PiRlcChallenge
