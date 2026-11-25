import NightstreamFPrime.Spec.AjtaiSetupV1.Shake128

/-! Keccak-f[1600] on 25 lanes of 64 bits (the permutation of `AjtaiSetupV1/Shake128.lean`, FIPS
202 Section 3), the overwrite-mode duplex sponge of spongefish 0.8.0 with a rate of 136 bytes
(`duplex_sponge.rs`, `DuplexSponge<KeccakF1600, 200, 136>`), and the FIPS 202 hashes SHAKE128
and SHA3-256 (Sections 5.1 and 6) on byte arrays.

- Byte `i` of the state is byte `i % 8` of lane `i / 8`, little endian, as
  `KeccakF1600::permute_mut` reads and writes it.
- `Duplex.absorb` writes the input over the rate from the absorb position and applies the
  permutation only when the rate is full and input remains; it sets the squeeze position to the
  rate. `Duplex.squeeze` sets the absorb position to 0 and applies the permutation when the
  squeeze position is at the rate. An empty absorb or squeeze does not change the state.
- `Duplex.events` is the list of the absorbs and squeezes of a sponge, the last one first. With
  `Duplex.ideal = some answer`, a squeeze returns `answer events count` for the events before
  it: the sponge in the reading `[KECCAK]` of `SECURITY_MODEL.md`. `Duplex.empty` has no `ideal`
  and is the sponge of the code.
- `hash rate domain` is the FIPS 202 sponge with the padding `domain ‖ 0* ‖ 0x80`:
  `shake128` has the rate 168 and the domain `0x1F`, `sha3_256` the rate 136 and `0x06`.

The model of Akita is a random oracle; these definitions have no specification in Lean. The
evidence for them is the FIPS 202 vectors and the vectors of real proofs of
`tests/AkitaVectors.lean`. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

open NightstreamFPrime.Spec.AjtaiSetupV1 (Shake128.State Shake128.permute)

namespace Keccak

/-- The lanes of a state in the order of their bytes. -/
def ofState (s : Shake128.State) : Array UInt64 :=
  #[s.a00, s.a10, s.a20, s.a30, s.a40, s.a01, s.a11, s.a21, s.a31, s.a41,
    s.a02, s.a12, s.a22, s.a32, s.a42, s.a03, s.a13, s.a23, s.a33, s.a43,
    s.a04, s.a14, s.a24, s.a34, s.a44]

def toState (l : Array UInt64) : Shake128.State :=
  ⟨l[0]!, l[1]!, l[2]!, l[3]!, l[4]!, l[5]!, l[6]!, l[7]!, l[8]!, l[9]!, l[10]!, l[11]!,
    l[12]!, l[13]!, l[14]!, l[15]!, l[16]!, l[17]!, l[18]!, l[19]!, l[20]!, l[21]!, l[22]!,
    l[23]!, l[24]!⟩

def permute (l : Array UInt64) : Array UInt64 := ofState (Shake128.permute (toState l))

def zeroLanes : Array UInt64 := Array.replicate 25 0

def getByte (l : Array UInt64) (i : Nat) : UInt8 :=
  (l[i / 8]! >>> (8 * (i % 8)).toUInt64).toUInt8

def setByte (l : Array UInt64) (i : Nat) (b : UInt8) : Array UInt64 :=
  let shift := (8 * (i % 8)).toUInt64
  l.modify (i / 8) fun w => (w &&& ~~~((0xff : UInt64) <<< shift)) ||| (b.toUInt64 <<< shift)

def xorByte (l : Array UInt64) (i : Nat) (b : UInt8) : Array UInt64 :=
  l.modify (i / 8) fun w => w ^^^ (b.toUInt64 <<< (8 * (i % 8)).toUInt64)

/-- The FIPS 202 sponge of rate `rate` bytes with the padding `domain ‖ 0* ‖ 0x80`, and `count`
bytes of output. -/
def hash (rate : Nat) (domain : UInt8) (input : ByteArray) (count : Nat) : ByteArray := Id.run do
  let padded := input ++ ByteArray.mk (Array.replicate (rate - input.size % rate) 0)
  let padded := padded.set! input.size (padded.get! input.size ^^^ domain)
  let padded := padded.set! (padded.size - 1) (padded.get! (padded.size - 1) ^^^ 0x80)
  let mut lanes := zeroLanes
  for block in [0:padded.size / rate] do
    for i in [0:rate] do
      lanes := xorByte lanes i (padded.get! (block * rate + i))
    lanes := permute lanes
  let mut out := ByteArray.emptyWithCapacity count
  for i in [0:count] do
    if i ≠ 0 ∧ i % rate = 0 then lanes := permute lanes
    out := out.push (getByte lanes (i % rate))
  return out

/-- SHAKE128 (FIPS 202 Section 6.2) with `count` bytes of output. -/
def shake128 (input : ByteArray) (count : Nat) : ByteArray := hash 168 0x1F input count

/-- SHA3-256 (FIPS 202 Section 6.1). -/
def sha3_256 (input : ByteArray) : ByteArray := hash 136 0x06 input 32

end Keccak

/-- An absorb of bytes or a squeeze of `count` bytes. -/
inductive Duplex.Event
  | absorb (input : ByteArray)
  | squeeze (count : Nat)

/-- The duplex sponge of spongefish over Keccak-f[1600]: the state, the absorb position and the
squeeze position, in bytes; its events, the last one first; and the answers of its squeezes when
it is ideal. -/
structure Duplex where
  lanes : Array UInt64
  absorbPos : Nat
  squeezePos : Nat
  events : List Duplex.Event := []
  ideal : Option (List Duplex.Event → Nat → ByteArray) := none

namespace Duplex

def rate : Nat := 136

/-- The state of `DuplexSponge::with_permutation`. -/
def empty : Duplex := { lanes := Keccak.zeroLanes, absorbPos := 0, squeezePos := rate }

/-- The state and the absorb position after the bytes of `input`. -/
def absorbed (s : Duplex) (input : ByteArray) : Array UInt64 × Nat := Id.run do
  let mut lanes := s.lanes
  let mut pos := s.absorbPos
  for b in input do
    if pos = rate then
      lanes := Keccak.permute lanes
      pos := 0
    lanes := Keccak.setByte lanes pos b
    pos := pos + 1
  return (lanes, pos)

def absorb (s : Duplex) (input : ByteArray) : Duplex :=
  if input.size = 0 then { s with events := .absorb input :: s.events }
  else
    { lanes := (s.absorbed input).1, absorbPos := (s.absorbed input).2, squeezePos := rate,
      events := .absorb input :: s.events, ideal := s.ideal }

/-- The next `count` bytes of the state, with the state and the squeeze position after them. -/
def squeezed (s : Duplex) (count : Nat) : ByteArray × Array UInt64 × Nat := Id.run do
  let mut lanes := s.lanes
  let mut pos := s.squeezePos
  let mut out := ByteArray.emptyWithCapacity count
  for _ in [0:count] do
    if pos = rate then
      lanes := Keccak.permute lanes
      pos := 0
    out := out.push (Keccak.getByte lanes pos)
    pos := pos + 1
  return (out, lanes, pos)

def squeeze (s : Duplex) (count : Nat) : ByteArray × Duplex :=
  if count = 0 then (ByteArray.empty, { s with events := .squeeze count :: s.events })
  else match s.ideal with
    | some answer => (answer s.events count, { s with events := .squeeze count :: s.events })
    | none =>
        ((s.squeezed count).1,
          { lanes := (s.squeezed count).2.1, absorbPos := 0, squeezePos := (s.squeezed count).2.2,
            events := .squeeze count :: s.events, ideal := none })

theorem absorb_events (s : Duplex) (input : ByteArray) :
    (s.absorb input).events = .absorb input :: s.events := by
  unfold absorb
  split <;> rfl

theorem absorb_ideal (s : Duplex) (input : ByteArray) : (s.absorb input).ideal = s.ideal := by
  unfold absorb
  split <;> rfl

theorem squeeze_events (s : Duplex) (count : Nat) :
    (s.squeeze count).2.events = .squeeze count :: s.events := by
  unfold squeeze
  split
  · rfl
  · split <;> rfl

theorem squeeze_ideal (s : Duplex) (count : Nat) : (s.squeeze count).2.ideal = s.ideal := by
  unfold squeeze
  split
  · rfl
  · split
    · rfl
    · rename_i none_
      exact none_.symm

/-- A squeeze of an ideal sponge is the answer for the events before it. -/
theorem squeeze_of_ideal (s : Duplex) {count : Nat} (positive : count ≠ 0)
    {answer : List Event → Nat → ByteArray} (ideal : s.ideal = some answer) :
    (s.squeeze count).1 = answer s.events count := by
  unfold squeeze
  simp only [positive, ↓reduceIte, ideal]

end Duplex

end NightstreamFPrime.Spec.Compression.Akita.Replay
