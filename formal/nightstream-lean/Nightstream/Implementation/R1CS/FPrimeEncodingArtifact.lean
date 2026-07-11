import Nightstream.Implementation.R1CS.CanonicalU64Artifact

/-!
Exact row program emitted by Rust `enforce_public_bits_encode_digest`.

The production helper allocates four digest variables and 256 public bit
variables before invoking this row program. For each lane it emits the exact
canonical-u64 artifact, then 64 equality rows connecting the public bits to
that canonical decomposition. The Rust conformance test checks this layout,
row/column count, honest assignment, forged bit rejection, source anchor, and
artifact hash against the live builder.
-/

namespace Nightstream.Implementation.R1CS.FPrimeEncoding

open Nightstream.Implementation.R1CS

set_option maxRecDepth 32768

def schemaVersion : Nat := 1
def artifactKind : String := "r1cs/f-prime-enc-inst"
def sourceAnchor : String := "enforce_public_bits_encode_digest"
def artifactSha256 : String := "e3b0601fdcf5afc1446391e2f4b52679944ee627acf307a638d75a587c220d66"
def witnessSha256 : String := "0024a6e6b64e47eaa6877b7de81c23d8b5b5bb601cfc23d7c8bec095cf034d1c"

def laneCount : Nat := 4
def bitsPerLane : Nat := 64
def rowCount : Nat := 532
def colCount : Nat := 525

def digestCol (lane : Nat) : Nat := lane + 1
def publicBitCol (lane bit : Nat) : Nat := 5 + lane * 64 + bit
def canonicalAuxStart (lane : Nat) : Nat := 261 + lane * 66

def canonicalMap (lane : Nat) : Nat → Nat
  | 0 => 0
  | 1 => digestCol lane
  | index => canonicalAuxStart lane + (index - 2)

/-- Exact `(publicBit - canonicalBit) * 1 = 0` row. -/
def equalityRow (lane bit : Nat) : Row :=
  ⟨[(publicBitCol lane bit, 1),
    (canonicalMap lane (CanonicalU64.bitCol bit), goldilocksP - 1)],
   [(0, 1)], []⟩

def laneRows (lane : Nat) : List Row :=
  CanonicalU64.rows.map (renameRow (canonicalMap lane)) ++
    (List.range 64).map (equalityRow lane)

def rows : List Row :=
  (List.range 4).flatMap laneRows

theorem rows_length : rows.length = rowCount := by decide

end Nightstream.Implementation.R1CS.FPrimeEncoding
