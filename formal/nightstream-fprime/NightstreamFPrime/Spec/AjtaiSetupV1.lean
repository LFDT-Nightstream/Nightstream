import NightstreamFPrime.Spec.AjtaiSetupV1.Shake128
import NightstreamFPrime.Spec.Poseidon2

namespace NightstreamFPrime.Spec.AjtaiSetupV1

/-- The ASCII bytes of `nightstream-ajtai-shake128-reject64-v1`. -/
def setupIdBytes : List Nat :=
  [110, 105, 103, 104, 116, 115, 116, 114, 101, 97, 109, 45, 97, 106,
    116, 97, 105, 45, 115, 104, 97, 107, 101, 49, 50, 56, 45, 114, 101,
    106, 101, 99, 116, 54, 52, 45, 118, 49]

@[simp] theorem setupIdBytes_length : setupIdBytes.length = 38 := by
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

/-- One key element reads four blocks of the SHAKE128 rate: 84 words of 8 bytes. -/
def keyWords : Nat := 84

def elementWords (seed : List Nat) (row block : Nat) : List Nat :=
  (Shake128.lanes (elementInput seed row block) keyWords).map UInt64.toNat

theorem elementWords_length (seed : List Nat) (row block : Nat) :
    (elementWords seed row block).length = keyWords := by
  rw [elementWords, List.length_map, Shake128.lanes_length]

theorem elementWords_lt (seed : List Nat) (row block : Nat) :
    ∀ word ∈ elementWords seed row block, word < 2 ^ 64 := by
  intro word member
  obtain ⟨lane, _, rfl⟩ := List.mem_map.mp member
  exact UInt64.toNat_lt lane

def keptWords (words : List Nat) : List Nat :=
  words.filter (· < fieldModulus)

def keptCoefficientNat (words : List Nat) (lane : Nat) : Nat :=
  (keptWords words).getD lane 0

theorem keptCoefficientNat_of_kept (words : List Nat) (lane : Nat)
    (inside : lane < (keptWords words).length) :
    keptCoefficientNat words lane = (keptWords words)[lane] := by
  rw [keptCoefficientNat, List.getD_eq_getElem?_getD, List.getElem?_eq_getElem inside,
    Option.getD_some]

theorem keptCoefficientNat_fallback (words : List Nat) (lane : Nat)
    (outside : (keptWords words).length ≤ lane) :
    keptCoefficientNat words lane = 0 := by
  rw [keptCoefficientNat, List.getD_eq_getElem?_getD, List.getElem?_eq_none outside,
    Option.getD_none]

theorem keptCoefficientNat_lt (words : List Nat) (lane : Nat) :
    keptCoefficientNat words lane < fieldModulus := by
  rcases Nat.lt_or_ge lane (keptWords words).length with inside | outside
  · rw [keptCoefficientNat_of_kept words lane inside]
    exact of_decide_eq_true (List.mem_filter.mp (List.getElem_mem inside)).2
  · rw [keptCoefficientNat_fallback words lane outside]
    decide

def elementCoefficientNat (seed : List Nat) (row block lane : Nat) : Nat :=
  keptCoefficientNat (elementWords seed row block) lane

theorem elementCoefficientNat_lt (seed : List Nat) (row block lane : Nat) :
    elementCoefficientNat seed row block lane < fieldModulus :=
  keptCoefficientNat_lt _ _

structure Setup (_verifierRows _messageColumns : Nat) where
  seed : Seed

namespace Setup

def coefficientNat {verifierRows messageColumns : Nat}
    (setup : Setup verifierRows messageColumns)
    (row : Fin verifierRows) (block : Fin messageColumns)
    (lane : Fin ringDegree) : Nat :=
  elementCoefficientNat setup.seed.bytes row.val block.val lane.val

theorem coefficientNat_lt {verifierRows messageColumns : Nat}
    (setup : Setup verifierRows messageColumns)
    (row : Fin verifierRows) (block : Fin messageColumns)
    (lane : Fin ringDegree) :
    setup.coefficientNat row block lane < fieldModulus := by
  exact elementCoefficientNat_lt _ _ _ _

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
    setup.authorityNats.length = 74 := by
  simp [authorityNats, setup.seed.length_eq]

@[simp] theorem authorityWords_length {verifierRows messageColumns : Nat}
    (setup : Setup verifierRows messageColumns) :
    setup.authorityWords.length = 74 := by
  simp [authorityWords]

end Setup

end NightstreamFPrime.Spec.AjtaiSetupV1
