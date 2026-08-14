/-!
Contract: independent typed values for the PaddedRowIdentityMemoryV2 memory
semantics.

Assurance tier: model-level.

Owns the V2 integer bounds, ROM/RAM address map, memory records, application
access kinds, and the challenge-independent validity predicate for one access.

Does not own fingerprints, commitments, F-prime, circuit rows, Rust layouts,
or cryptographic assumptions.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Protocol.Nebula

/-- V2 timestamps are canonical 23-bit nonnegative integers. -/
def timestampBits : Nat := 23

def timestampLimit : Nat := 2 ^ timestampBits

/-- V2 memory values are canonical 32-bit nonnegative integers. -/
def valueBits : Nat := 32

def valueLimit : Nat := 2 ^ valueBits

/-- V2 declares at most 17 bits of active accesses in one segment. -/
def operationCountBits : Nat := 17

def operationCountLimit : Nat := 2 ^ operationCountBits

def romCells : Nat := 2 ^ 12

def ramCells : Nat := 2 ^ 16

def scannedCells : Nat := romCells + ramCells

inductive MemorySpace where
  | rom
  | ram
deriving DecidableEq, Repr

def MemorySpace.capacity : MemorySpace → Nat
  | .rom => romCells
  | .ram => ramCells

/-- The scan uses ROM first and RAM second. -/
def globalIndex : MemorySpace → Nat → Nat
  | .rom, address => address
  | .ram, address => romCells + address

inductive AccessKind where
  | read
  | write (requestedValue : Nat)
deriving DecidableEq, Repr

/-- One element of the RS, WS, IS, or FS multiset. -/
structure MemTuple where
  timestamp : Nat
  globalIndex : Nat
  value : Nat
deriving DecidableEq, Repr

@[ext]
theorem MemTuple.ext
    {left right : MemTuple}
    (timestamp : left.timestamp = right.timestamp)
    (globalIndex : left.globalIndex = right.globalIndex)
    (value : left.value = right.value) : left = right := by
  cases left
  cases right
  simp_all

/-- One application memory access before timestamp scheduling is checked. -/
structure Access where
  space : MemorySpace
  address : Nat
  kind : AccessKind
  read : MemTuple
  write : MemTuple
deriving DecidableEq, Repr

/-- Challenge-independent application and memory-record agreement. -/
structure Access.WellFormed (access : Access) : Prop where
  addressInRange : access.address < access.space.capacity
  readIndex : access.read.globalIndex = globalIndex access.space access.address
  writeIndex : access.write.globalIndex = globalIndex access.space access.address
  readValueInRange : access.read.value < valueLimit
  writeValueInRange : access.write.value < valueLimit
  valueRule :
    match access.kind with
    | .read => access.write.value = access.read.value
    | .write requestedValue =>
        access.space = .ram ∧
        requestedValue < valueLimit ∧
        access.write.value = requestedValue

/-- One access at a verifier-owned global timestamp. The successor timestamp
must remain in the 23-bit range. -/
structure Access.ValidAt (access : Access) (timestampIn : Nat) : Prop where
  wellFormed : access.WellFormed
  timestampInRange : timestampIn < timestampLimit
  timestampOutRange : timestampIn + 1 < timestampLimit
  readBeforeWrite : access.read.timestamp < timestampIn + 1
  writeTimestamp : access.write.timestamp = timestampIn + 1

/-- Exact operation order for one segment. Every active access consumes one
global integer timestamp. -/
inductive Ordered : Nat → List Access → Nat → Prop
  | nil (timestamp : Nat) : Ordered timestamp [] timestamp
  | cons
      {timestampIn timestampOut : Nat}
      {access : Access}
      {rest : List Access}
      (valid : access.ValidAt timestampIn)
      (tail : Ordered (timestampIn + 1) rest timestampOut) :
      Ordered timestampIn (access :: rest) timestampOut

theorem Ordered.append
    {timestampIn timestampMiddle timestampOut : Nat}
    {first second : List Access}
    (left : Ordered timestampIn first timestampMiddle)
    (right : Ordered timestampMiddle second timestampOut) :
    Ordered timestampIn (first ++ second) timestampOut := by
  induction left with
  | nil => simpa using right
  | cons valid _ inductionHypothesis =>
      exact .cons valid (inductionHypothesis right)

theorem Ordered.timestampOut_eq
    {timestampIn timestampOut : Nat}
    {accesses : List Access}
    (ordered : Ordered timestampIn accesses timestampOut) :
    timestampOut = timestampIn + accesses.length := by
  induction ordered with
  | nil => simp
  | cons _ _ inductionHypothesis =>
      simp only [List.length_cons]
      omega

end Nightstream.Protocol.Nebula
