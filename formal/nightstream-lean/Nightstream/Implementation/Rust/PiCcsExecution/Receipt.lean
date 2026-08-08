import Nightstream.SuperNeo.Concrete.Algebra
import Nightstream.SuperNeo.SumCheck.FixedPhase.RawCertificate

/-!
Canonical transport data for one Rust `Pi_CCS` execution.

Owns: the independent Lean representation of the Rust receipt, canonical
Goldilocks limb checks, and the exact fixed-profile proof-byte decoder.

Does not own: transcript replay, the selected CCS relation, paper acceptance,
or a claim that any receipt came from Rust.

Emits constraints: no.

Assurance tier: model-level. The byte decoder is executable Lean semantics for
the selected Rust codec tag `1102`, version `1`, 24 rounds, and 10 extension
coefficients per round.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Rust.PiCcsExecution

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.SumCheck

/-- Canonical low/high Goldilocks representatives of one extension value. -/
structure RawK where
  low : Nat
  high : Nat
deriving Repr, DecidableEq, Inhabited

namespace RawK

/-- Both limbs are canonical Goldilocks representatives. -/
def wellFormed (value : RawK) : Bool :=
  decide (value.low < goldilocksModulus) &&
    decide (value.high < goldilocksModulus)

/-- Total decode. Receipt acceptance separately requires `wellFormed`, so
modular reduction cannot create an accepted alternate encoding. -/
def decode (value : RawK) : K where
  c0 := ⟨value.low % goldilocksModulus, Nat.mod_lt _ (by decide)⟩
  c1 := ⟨value.high % goldilocksModulus, Nat.mod_lt _ (by decide)⟩

/-- Exact limb order used by the Rust transcript. -/
def fields (value : RawK) : List Nat := [value.low, value.high]

end RawK

/-- Verifier-visible statement captured before proof messages are replayed. -/
structure PiCcsCanonicalStatement where
  relationId : List Nat
  transcriptState : List Nat
  transcriptAbsorbed : Nat
  publicFields : List Nat
  piCcsStatementFields : List Nat
  priorPoint : List RawK
  claimedCoefficients : List RawK
deriving Repr, DecidableEq

/-- Raw prover data for one execution. -/
structure PiCcsExecutionProof where
  proofBytes : List Nat
  fullOutput : List RawK
deriving Repr, DecidableEq

/-- Exact selected Rust proof tag. -/
def proofTag : Nat := 1102

/-- Exact selected Rust proof-codec version. -/
def proofVersion : Nat := 1

/-- The selected rectangular relation has 24 SumCheck variables. -/
def roundCount : Nat := 24

/-- Degree nine gives ten fixed coefficient slots per round. -/
def roundCoefficientCount : Nat := 10

/-- Four `u64` header words plus 24 × 10 quadratic-extension values. -/
def proofByteCount : Nat := 32 + roundCount * roundCoefficientCount * 16

/-- Read one little-endian `u64` word from a byte array. The exact-length and
byte-range checks are separate fail-closed conditions. Array access keeps the
fixed 3,872-byte parser linear. -/
def readU64LE (bytes : Array Nat) (offset : Nat) : Nat :=
  (List.range 8).foldl
    (fun value index => value + bytes.getD (offset + index) 0 * 256 ^ index) 0

/-- Byte offset of one extension coefficient. -/
def coefficientOffset (round coefficient : Nat) : Nat :=
  32 + (round * roundCoefficientCount + coefficient) * 16

/-- Decode one coefficient from its two little-endian limbs. -/
def proofCoefficient (bytes : Array Nat) (round coefficient : Nat) : RawK :=
  let offset := coefficientOffset round coefficient
  { low := readU64LE bytes offset
    high := readU64LE bytes (offset + 8) }

/-- All serialized coefficients use canonical Goldilocks limbs. -/
def proofCoefficientsWellFormed (bytes : Array Nat) : Bool :=
  (List.range roundCount).all fun round =>
    (List.range roundCoefficientCount).all fun coefficient =>
      (proofCoefficient bytes round coefficient).wellFormed

/-- Exact selected fixed-profile proof codec check. -/
def proofBytesWellFormed (bytes : List Nat) : Bool :=
  let bytesArray := bytes.toArray
  decide (bytes.length = proofByteCount) &&
    bytes.all (fun byte => decide (byte < 256)) &&
    decide (readU64LE bytesArray 0 = proofTag) &&
    decide (readU64LE bytesArray 8 = proofVersion) &&
    decide (readU64LE bytesArray 16 = roundCount) &&
    decide (readU64LE bytesArray 24 = roundCoefficientCount) &&
    proofCoefficientsWellFormed bytesArray

/-- Decode the fixed certificate after transport checks. This function is
total; `proofBytesWellFormed` is the authority that rejects malformed input. -/
def proofCertificate (bytes : List Nat) : SumCheck.Finite.Certificate K :=
  let bytesArray := bytes.toArray
  { rounds := (List.range roundCount).map fun round =>
      { coefficients := (List.range roundCoefficientCount).map fun coefficient =>
          (proofCoefficient bytesArray round coefficient).decode } }

end Nightstream.Implementation.Rust.PiCcsExecution
