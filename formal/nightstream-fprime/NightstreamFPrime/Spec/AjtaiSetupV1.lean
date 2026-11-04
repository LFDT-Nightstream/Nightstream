import NightstreamFPrime.Spec.AjtaiSetupV1.Shake128
import NightstreamFPrime.Spec.Poseidon2

namespace NightstreamFPrime.Spec.AjtaiSetupV1

def setupIdBytes : List Nat :=
  [110, 105, 103, 104, 116, 115, 116, 114, 101, 97, 109, 45, 97, 106,
    116, 97, 105, 45, 115, 104, 97, 107, 101, 49, 50, 56, 45, 119, 105,
    100, 101, 50, 53, 54, 45, 118, 49]

@[simp] theorem setupIdBytes_length : setupIdBytes.length = 37 := by
  rfl

structure Seed where
  bytes : List Nat
  length_eq : bytes.length = 32
  canonical : forall byte, byte ∈ bytes -> byte < 256

def littleEndianBytes : Nat → Nat → List Nat
  | 0, _ => []
  | count + 1, value => value % 256 :: littleEndianBytes count (value / 256)

@[simp] theorem littleEndianBytes_length (count value : Nat) :
    (littleEndianBytes count value).length = count := by
  induction count generalizing value with
  | zero => rfl
  | succ count ih => simp [littleEndianBytes, ih]

def elementInput (seed : List Nat) (row block : Nat) : List Nat :=
  setupIdBytes ++ seed ++ littleEndianBytes 4 row ++ littleEndianBytes 8 block

def elementLanes (seed : List Nat) (row block : Nat) : List UInt64 :=
  Shake128.lanes (elementInput seed row block) (4 * ringDegree)

def laneWord (lanes : List UInt64) (lane : Nat) : Nat :=
  (lanes.getD (4 * lane) 0).toNat + 2 ^ 64 * ((lanes.getD (4 * lane + 1) 0).toNat +
    2 ^ 64 * ((lanes.getD (4 * lane + 2) 0).toNat +
      2 ^ 64 * (lanes.getD (4 * lane + 3) 0).toNat))

theorem laneWord_lt (lanes : List UInt64) (lane : Nat) : laneWord lanes lane < 2 ^ 256 := by
  have w0 := UInt64.toNat_lt (lanes.getD (4 * lane) 0)
  have w1 := UInt64.toNat_lt (lanes.getD (4 * lane + 1) 0)
  have w2 := UInt64.toNat_lt (lanes.getD (4 * lane + 2) 0)
  have w3 := UInt64.toNat_lt (lanes.getD (4 * lane + 3) 0)
  unfold laneWord
  omega

def wideCoefficientNat (seed : List Nat) (row block lane : Nat) : Nat :=
  laneWord (elementLanes seed row block) lane % goldilocksModulus

theorem wideCoefficientNat_lt (seed : List Nat) (row block lane : Nat) :
    wideCoefficientNat seed row block lane < goldilocksModulus := by
  exact Nat.mod_lt _ (by decide)

structure Setup (_verifierRows _messageColumns : Nat) where
  seed : Seed

namespace Setup

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

def verifierKey {verifierRows messageColumns : Nat}
    (setup : Setup verifierRows messageColumns) :
    Fin verifierRows → Fin messageColumns → RingF :=
  fun row block lane =>
    ⟨setup.coefficientNat row block lane,
      setup.coefficientNat_lt row block lane⟩

def authorityNats {verifierRows messageColumns : Nat}
    (setup : Setup verifierRows messageColumns) : List Nat :=
  [setupIdBytes.length] ++ setupIdBytes ++
    [verifierRows, messageColumns, setup.seed.bytes.length] ++ setup.seed.bytes

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
