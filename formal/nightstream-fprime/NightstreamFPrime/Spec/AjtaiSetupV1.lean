import NightstreamFPrime.Spec.AjtaiSetupV1.ChaCha20
import NightstreamFPrime.Spec.Poseidon2

/-!
Owns the exact compact Ajtai setup selected by
`nightstream-ajtai-chacha20-wide256-v1`.

The verifier owns one canonical 32-byte seed. Each key coefficient uses one
RFC-8439 ChaCha20 block with nonce `row_u32_le || block_u64_le`, counter equal
to the coefficient lane, and reduction of the first 256 output bits modulo
the Goldilocks prime. The key remains an indexed finite function. There is no
rejection, retry, fallback, or expanded key list.
-/

namespace NightstreamFPrime.Spec.AjtaiSetupV1

/-- ASCII bytes of `nightstream-ajtai-chacha20-wide256-v1`. -/
def setupIdBytes : List Nat :=
  [110, 105, 103, 104, 116, 115, 116, 114, 101, 97, 109, 45, 97, 106,
    116, 97, 105, 45, 99, 104, 97, 99, 104, 97, 50, 48, 45, 119, 105,
    100, 101, 50, 53, 54, 45, 118, 49]

@[simp] theorem setupIdBytes_length : setupIdBytes.length = 37 := by
  rfl

/-- Canonical verifier-owned 256-bit setup seed. -/
structure Seed where
  bytes : List Nat
  length_eq : bytes.length = 32
  canonical : forall byte, byte ∈ bytes -> byte < 256

/-- Total wide-reduction coefficient function. -/
def wideCoefficientNat (seed : List Nat) (row block lane : Nat) : Nat :=
  ChaCha20.first256Nat seed row block lane % goldilocksModulus

theorem wideCoefficientNat_lt (seed : List Nat) (row block lane : Nat) :
    wideCoefficientNat seed row block lane < goldilocksModulus := by
  exact Nat.mod_lt _ (by decide)

/-- The dimensions are type-level verifier authority. The only stored value
is the canonical setup seed. -/
structure Setup (_verifierRows _messageColumns : Nat) where
  seed : Seed

namespace Setup

/-- One canonical coefficient selected by an in-bounds key coordinate. -/
def coefficientNat {verifierRows messageColumns : Nat}
    (setup : Setup verifierRows messageColumns)
    (row : Fin verifierRows) (block : Fin messageColumns)
    (lane : Fin ringDegree) : Nat :=
  wideCoefficientNat setup.seed.bytes row.val block.val lane.val

theorem coefficientNat_lt {verifierRows messageColumns : Nat}
    (setup : Setup verifierRows messageColumns)
    (row : Fin verifierRows) (block : Fin messageColumns)
    (lane : Fin ringDegree) :
    setup.coefficientNat row block lane < goldilocksModulus := by
  exact wideCoefficientNat_lt _ _ _ _

/-- Exact lazy Ajtai key consumed by the SuperNeo relation. -/
def verifierKey {verifierRows messageColumns : Nat}
    (setup : Setup verifierRows messageColumns) :
    Fin verifierRows → Fin messageColumns → RingF :=
  fun row block lane =>
    ⟨setup.coefficientNat row block lane,
      setup.coefficientNat_lt row block lane⟩

/-- Canonical non-hashed setup descriptor. It binds the exact setup ID,
dimensions, seed byte count, and verifier-owned seed. -/
def authorityNats {verifierRows messageColumns : Nat}
    (setup : Setup verifierRows messageColumns) : List Nat :=
  [setupIdBytes.length] ++ setupIdBytes ++
    [verifierRows, messageColumns, setup.seed.bytes.length] ++ setup.seed.bytes

/-- Poseidon2 field mapping of the complete setup authority. -/
def authorityWords {verifierRows messageColumns : Nat}
    (setup : Setup verifierRows messageColumns) : List F :=
  setup.authorityNats.map Poseidon2.ofNat

@[simp] theorem authorityNats_length {verifierRows messageColumns : Nat}
    (setup : Setup verifierRows messageColumns) :
    setup.authorityNats.length = 73 := by
  simp [authorityNats, setup.seed.length_eq]

@[simp] theorem authorityWords_length {verifierRows messageColumns : Nat}
    (setup : Setup verifierRows messageColumns) :
    setup.authorityWords.length = 73 := by
  simp [authorityWords]

end Setup

end NightstreamFPrime.Spec.AjtaiSetupV1
