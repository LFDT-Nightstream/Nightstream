import Nightstream.Implementation.R1CS.Semantics

/-!
Contract: canonical byte/field and `enc_inst` encodings at the F' boundary.

Owns the exact accepted value domain: four Goldilocks residues, each encoded as
one little-endian 64-bit lane, and the fixed public input `[1 | 4 * 64 bits]`.
Raw containers are rejected unless they have 32 bytes or 256 bits. Decoding
rejects every 64-bit lane at or above the Goldilocks modulus. The round-trip
and injectivity theorems are over accepted values, so modular aliases cannot
enter through serialization.

This module models Rust `encode_x_out_public_bits`,
`encode_f_prime_public_input`, and `canonical_digest32_fields`. Exact circuit
row correspondence is proved separately in `FPrimeEncodingSound`.
-/

namespace Nightstream.Implementation.Encoding.FPrime

open Nightstream.Implementation.R1CS

/-- One canonical Goldilocks representative. -/
abbrev Lane := { value : Nat // value < goldilocksP }

/-- Four digest lanes, matching Rust's `[F; 4]`. -/
abbrev Digest := Fin 4 → Lane

/-- The typed 32-byte representation, grouped as four little-endian words. -/
abbrev DigestBytes := Fin 4 → BitVec 64

/-- The typed 256-bit `enc_inst` body, grouped by digest lane. -/
abbrev EncInst := Fin 4 → BitVec 64

def encodeLane (lane : Lane) : BitVec 64 :=
  BitVec.ofNat 64 lane.1

def encodeBytes (digest : Digest) : DigestBytes :=
  fun lane => encodeLane (digest lane)

def encodeEncInst (digest : Digest) : EncInst :=
  fun lane => encodeLane (digest lane)

/-- Canonical decoding. The `if` is the exact `value < ORDER_U64` gate in
Rust's byte-to-field path. -/
def decodeLanes (encoded : Fin 4 → BitVec 64) : Option Digest :=
  if canonical : ∀ lane, (encoded lane).toNat < goldilocksP then
    some (fun lane => ⟨(encoded lane).toNat, canonical lane⟩)
  else
    none

abbrev decodeBytes := decodeLanes
abbrev decodeEncInst := decodeLanes

/-- Executable shape gate for an untrusted byte container. Rust's typed
`[u8; 32]` carries this invariant after deserialization. -/
def acceptsDigestByteLength (bytes : List UInt8) : Bool :=
  bytes.length == 32

/-- Executable shape gate used before the F' circuit emits encoding rows. -/
def acceptsEncInstLength (bits : List Bool) : Bool :=
  bits.length == 256

theorem digestBytes_length_checked (bytes : List UInt8) :
    acceptsDigestByteLength bytes = true ↔ bytes.length = 32 := by
  simp [acceptsDigestByteLength]

theorem encInst_length_checked (bits : List Bool) :
    acceptsEncInstLength bits = true ↔ bits.length = 256 := by
  simp [acceptsEncInstLength]

private theorem lane_toNat_encode (lane : Lane) :
    (encodeLane lane).toNat = lane.1 := by
  simp only [encodeLane, BitVec.toNat_ofNat]
  apply Nat.mod_eq_of_lt
  exact Nat.lt_trans lane.2 (by decide)

theorem decode_encode (digest : Digest) :
    decodeLanes (fun lane => encodeLane (digest lane)) = some digest := by
  unfold decodeLanes
  split
  · rename_i canonical
    congr 1
    funext lane
    apply Subtype.ext
    exact lane_toNat_encode (digest lane)
  · rename_i notCanonical
    exfalso
    apply notCanonical
    intro lane
    rw [lane_toNat_encode]
    exact (digest lane).2

theorem digestBytes_roundtrip (digest : Digest) :
    decodeBytes (encodeBytes digest) = some digest := by
  exact decode_encode digest

theorem encInst_roundtrip (digest : Digest) :
    decodeEncInst (encodeEncInst digest) = some digest := by
  exact decode_encode digest

theorem encodeLane_injective : Function.Injective encodeLane := by
  intro left right equal
  apply Subtype.ext
  have values := congrArg BitVec.toNat equal
  simpa only [lane_toNat_encode] using values

theorem digestBytes_injective : Function.Injective encodeBytes := by
  intro left right equal
  funext lane
  apply encodeLane_injective
  exact congrFun equal lane

theorem encInst_injective : Function.Injective encodeEncInst := by
  intro left right equal
  funext lane
  apply encodeLane_injective
  exact congrFun equal lane

/-- Equality of all 256 public bits is enough to recover the four field
lanes. This is the bit-level form consumed by the circuit correspondence. -/
theorem encInst_bits_injective {left right : Digest}
    (equalBits : ∀ lane (bit : Nat), bit < 64 →
      (encodeEncInst left lane).getLsbD bit =
        (encodeEncInst right lane).getLsbD bit) :
    left = right := by
  apply encInst_injective
  funext lane
  apply BitVec.eq_of_getLsbD_eq
  intro bit bitLt
  exact equalBits lane bit bitLt

/-- Full F' CCS public input: a verifier-fixed affine-one coordinate and the
canonical 256-bit recursive link. -/
structure PublicInput where
  one : Nat
  body : EncInst

def encodePublicInput (digest : Digest) : PublicInput where
  one := 1
  body := encodeEncInst digest

def PublicInput.Accepted (input : PublicInput) : Prop :=
  input.one = 1 ∧ ∃ digest, decodeEncInst input.body = some digest

theorem publicInput_shape (digest : Digest) :
    (encodePublicInput digest).one = 1 ∧
    decodeEncInst (encodePublicInput digest).body = some digest := by
  exact ⟨rfl, encInst_roundtrip digest⟩

theorem publicInput_injective : Function.Injective encodePublicInput := by
  intro left right equal
  apply encInst_injective
  exact congrArg PublicInput.body equal

end Nightstream.Implementation.Encoding.FPrime
