/-!
Native-efficient ChaCha8 stream used by large generated coefficient schedules.

The arithmetic state is `UInt32`, so `native_decide` executes the ARX core
with machine-word operations instead of allocating arbitrary-precision `Nat`
values for every add, xor, shift, and reduction. The public result remains a
list of canonical naturals. Exact Rust parity is pinned by generated stream
and SeededPhi81 row fixtures.
-/

namespace Nightstream.Implementation.R1CS.ChaCha8Fast

private def getWord (state : Array UInt32) (index : Nat) : UInt32 :=
  state.getD index 0

private def rotateLeft32 (value : UInt32) (amount : Nat) : UInt32 :=
  let amount := amount % 32
  if amount = 0 then value
  else
    (value <<< UInt32.ofNat amount) |||
      (value >>> UInt32.ofNat (32 - amount))

private def quarterRound (state : Array UInt32)
    (ai bi ci di : Nat) : Array UInt32 :=
  let a1 := getWord state ai + getWord state bi
  let d1 := rotateLeft32 (getWord state di ^^^ a1) 16
  let c1 := getWord state ci + d1
  let b1 := rotateLeft32 (getWord state bi ^^^ c1) 12
  let a2 := a1 + b1
  let d2 := rotateLeft32 (d1 ^^^ a2) 8
  let c2 := c1 + d2
  let b2 := rotateLeft32 (b1 ^^^ c2) 7
  (((state.set! ai a2).set! bi b2).set! ci c2).set! di d2

private def doubleRound (state : Array UInt32) : Array UInt32 :=
  let state := quarterRound state 0 4 8 12
  let state := quarterRound state 1 5 9 13
  let state := quarterRound state 2 6 10 14
  let state := quarterRound state 3 7 11 15
  let state := quarterRound state 0 5 10 15
  let state := quarterRound state 1 6 11 12
  let state := quarterRound state 2 7 8 13
  quarterRound state 3 4 9 14

private def runDoubleRounds : Nat → Array UInt32 → Array UInt32
  | 0, state => state
  | rounds + 1, state => runDoubleRounds rounds (doubleRound state)

private def littleEndian32 (bytes : List Nat) (offset : Nat) : UInt32 :=
  UInt32.ofNat <|
    bytes.getD offset 0 +
      256 * bytes.getD (offset + 1) 0 +
      65536 * bytes.getD (offset + 2) 0 +
      16777216 * bytes.getD (offset + 3) 0

private def initialState (seed : List Nat) (block : Nat) : Array UInt32 :=
  #[0x61707865, 0x3320646e, 0x79622d32, 0x6b206574,
    littleEndian32 seed 0, littleEndian32 seed 4,
    littleEndian32 seed 8, littleEndian32 seed 12,
    littleEndian32 seed 16, littleEndian32 seed 20,
    littleEndian32 seed 24, littleEndian32 seed 28,
    UInt32.ofNat block, UInt32.ofNat (block / 4294967296), 0, 0]

private def blockWord32s (seed : List Nat) (block : Nat) : List UInt32 :=
  let initial := initialState seed block
  let permuted := runDoubleRounds 4 initial
  (List.range 16).map fun index =>
    getWord permuted index + getWord initial index

/-- The sixteen words of one block as canonical naturals. -/
def blockWords (seed : List Nat) (block : Nat) : List Nat :=
  (blockWord32s seed block).map UInt32.toNat

private def word32s (seed : List Nat) (wordStart count : Nat) : List UInt32 :=
  let firstBlock := wordStart / 16
  let offset := wordStart % 16
  let blockCount := (offset + count + 15) / 16
  ((List.range blockCount).flatMap fun blockOffset =>
      blockWord32s seed (firstBlock + blockOffset)).drop offset |>.take count

/-- Finite logical `u32` stream slice, represented canonically in `Nat`. -/
def words (seed : List Nat) (wordStart count : Nat) : List Nat :=
  (word32s seed wordStart count).map UInt32.toNat

/-- Pair consecutive little-endian words exactly as `RngCore::next_u64`. -/
def u64s (seed : List Nat) (wordStart count : Nat) : List Nat :=
  let stream := word32s seed wordStart (2 * count)
  (List.range count).map fun index =>
    (stream.getD (2 * index) 0).toNat +
      4294967296 * (stream.getD (2 * index + 1) 0).toNat

end Nightstream.Implementation.R1CS.ChaCha8Fast
