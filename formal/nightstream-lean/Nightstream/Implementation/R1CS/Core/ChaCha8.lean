/-!
Contract: pure executable ChaCha8 word stream used by compact seeded Phi81 rows.

Owns the 32-bit ARX permutation and the 64-bit-counter, zero-stream layout of
`rand_chacha::ChaCha8Rng::from_seed`.  The definition is deliberately small
and pure: callers request a finite slice of the logical `u32` stream, so the
four-block buffering strategy used by the Rust crate is not part of the
semantics. Internal ARX definitions remain visible so the optimized machine
implementation can be refined to this model without trusting fixtures.

This is not a cryptographic security proof for ChaCha8.  Its assurance role is
exact coefficient reproduction.  Rust-generated conformance vectors pin the
translation against the concrete `rand_chacha` version used by `neo-ccs`.
-/

namespace Nightstream.Implementation.R1CS.ChaCha8

def wordModulus : Nat := 4294967296

def add32 (left right : Nat) : Nat := (left + right) % wordModulus

def xor32 (left right : Nat) : Nat := Nat.xor left right % wordModulus

def rotateLeft32 (value amount : Nat) : Nat :=
  let amount := amount % 32
  if amount = 0 then value % wordModulus
  else
    ((Nat.shiftLeft value amount) % wordModulus +
      Nat.shiftRight value (32 - amount)) % wordModulus

def getWord (state : Array Nat) (index : Nat) : Nat :=
  state.getD index 0

def quarterRound (state : Array Nat)
    (ai bi ci di : Nat) : Array Nat :=
  let a1 := add32 (getWord state ai) (getWord state bi)
  let d1 := rotateLeft32 (xor32 (getWord state di) a1) 16
  let c1 := add32 (getWord state ci) d1
  let b1 := rotateLeft32 (xor32 (getWord state bi) c1) 12
  let a2 := add32 a1 b1
  let d2 := rotateLeft32 (xor32 d1 a2) 8
  let c2 := add32 c1 d2
  let b2 := rotateLeft32 (xor32 b1 c2) 7
  (((state.set! ai a2).set! bi b2).set! ci c2).set! di d2

def doubleRound (state : Array Nat) : Array Nat :=
  let state := quarterRound state 0 4 8 12
  let state := quarterRound state 1 5 9 13
  let state := quarterRound state 2 6 10 14
  let state := quarterRound state 3 7 11 15
  let state := quarterRound state 0 5 10 15
  let state := quarterRound state 1 6 11 12
  let state := quarterRound state 2 7 8 13
  quarterRound state 3 4 9 14

def runDoubleRounds : Nat → Array Nat → Array Nat
  | 0, state => state
  | rounds + 1, state => runDoubleRounds rounds (doubleRound state)

def littleEndian32 (bytes : List Nat) (offset : Nat) : Nat :=
  (bytes.getD offset 0 +
    256 * bytes.getD (offset + 1) 0 +
    65536 * bytes.getD (offset + 2) 0 +
    16777216 * bytes.getD (offset + 3) 0) % wordModulus

def initialState (seed : List Nat) (block : Nat) : Array Nat :=
  #[0x61707865, 0x3320646e, 0x79622d32, 0x6b206574,
    littleEndian32 seed 0, littleEndian32 seed 4,
    littleEndian32 seed 8, littleEndian32 seed 12,
    littleEndian32 seed 16, littleEndian32 seed 20,
    littleEndian32 seed 24, littleEndian32 seed 28,
    block % wordModulus, (block / wordModulus) % wordModulus, 0, 0]

/-- The sixteen `u32` words of one ChaCha8 block. -/
def blockWords (seed : List Nat) (block : Nat) : List Nat :=
  let initial := initialState seed block
  let permuted := runDoubleRounds 4 initial
  (List.range 16).map fun index =>
    add32 (getWord permuted index) (getWord initial index)

/-- Finite slice of the logical word stream.  This is extensionally equal to
the buffered Rust RNG stream when `seed` has 32 bytes. -/
def words (seed : List Nat) (wordStart count : Nat) : List Nat :=
  let firstBlock := wordStart / 16
  let offset := wordStart % 16
  let blockCount := (offset + count + 15) / 16
  ((List.range blockCount).flatMap fun blockOffset =>
      blockWords seed (firstBlock + blockOffset)).drop offset |>.take count

def u64s (seed : List Nat) (wordStart count : Nat) : List Nat :=
  let stream := words seed wordStart (2 * count)
  (List.range count).map fun index =>
    stream.getD (2 * index) 0 + wordModulus * stream.getD (2 * index + 1) 0

end Nightstream.Implementation.R1CS.ChaCha8
