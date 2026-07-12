import Nightstream.Implementation.Encoding.FPrime
import Nightstream.Implementation.R1CS.Correspondence.FPrime.FPrimeEncodingSound

/-!
Executable `ENC-CANON` regressions: raw-length rejection, canonical lane
rejection, round-trip, the exact 532-row honest assignment, and a public-bit
forgery rejected at the first equality row.
-/

set_option maxRecDepth 32768
set_option maxHeartbeats 4000000

namespace NightstreamTests.FPrimeEncoding

open Nightstream.Implementation
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.FPrimeEncoding

example : Encoding.FPrime.acceptsDigestByteLength (List.replicate 32 0) = true := by
  decide

example : Encoding.FPrime.acceptsDigestByteLength (List.replicate 31 0) = false := by
  decide

example : Encoding.FPrime.acceptsEncInstLength (List.replicate 256 false) = true := by
  decide

example : Encoding.FPrime.acceptsEncInstLength (List.replicate 255 false) = false := by
  decide

def sampleDigest : Encoding.FPrime.Digest := fun lane =>
  match lane with
  | ⟨0, _⟩ => ⟨0, by decide⟩
  | ⟨1, _⟩ => ⟨5, by decide⟩
  | ⟨2, _⟩ => ⟨0x0123456789ABCDEF, by decide⟩
  | ⟨3, _⟩ => ⟨18446744069414584320, by decide⟩
  | ⟨n + 4, impossible⟩ => by omega

example : Encoding.FPrime.decodeEncInst
    (Encoding.FPrime.encodeEncInst sampleDigest) = some sampleDigest := by
  exact Encoding.FPrime.encInst_roundtrip sampleDigest

def noncanonicalLanes : Fin 4 → BitVec 64 := fun lane =>
  if lane = 0 then BitVec.ofNat 64 goldilocksP else 0

example : Encoding.FPrime.decodeEncInst noncanonicalLanes = none := by
  native_decide

def canonicalSample : Nat → Nat :=
  assignmentOf CanonicalU64.honestWitness

/-- Four copies of the canonical-u64 sample in the exact production encoding
layout. This is constructed independently of the Rust witness exporter. -/
def honestAssignment (column : Nat) : Nat :=
  if column = 0 then 1
  else if column < 5 then canonicalSample 1
  else if column < 261 then canonicalSample (2 + (column - 5) % 64)
  else if column < 525 then canonicalSample (2 + (column - 261) % 66)
  else 0

example : Satisfies rows honestAssignment := by native_decide

def wrongBitAssignment (column : Nat) : Nat :=
  if column = 5 then 0 else honestAssignment column

example : ¬ Satisfies rows wrongBitAssignment := by native_decide

example : ∀ row ∈ rows.take 69, RowHolds wrongBitAssignment row := by
  native_decide

example : ¬ RowHolds wrongBitAssignment ((rows.drop 69).head (by decide)) := by
  native_decide

end NightstreamTests.FPrimeEncoding
