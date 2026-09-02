/-!
Owns the pure RFC-8439 ChaCha20 block function used by
`nightstream-ajtai-chacha20-wide256-v1`.

The 96-bit nonce is `row_u32_le || block_u64_le`; the block counter is the
coefficient lane. This file defines exact coefficient-generation semantics.
It does not claim ChaCha20 pseudorandomness.
-/

namespace NightstreamFPrime.Spec.AjtaiSetupV1.ChaCha20

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

/-- RFC-8439 state with the Nightstream indexed nonce and counter framing. -/
def initialState (seed : List Nat) (row block lane : Nat) : Array Nat :=
  #[0x61707865, 0x3320646e, 0x79622d32, 0x6b206574,
    littleEndian32 seed 0, littleEndian32 seed 4,
    littleEndian32 seed 8, littleEndian32 seed 12,
    littleEndian32 seed 16, littleEndian32 seed 20,
    littleEndian32 seed 24, littleEndian32 seed 28,
    lane % wordModulus, row % wordModulus,
    block % wordModulus, (block / wordModulus) % wordModulus]

/-- The sixteen `u32` words of one RFC-8439 ChaCha20 block. -/
def blockWords (seed : List Nat) (row block lane : Nat) : List Nat :=
  let initial := initialState seed row block lane
  let permuted := runDoubleRounds 10 initial
  (List.range 16).map fun index =>
    add32 (getWord permuted index) (getWord initial index)

/-- The first 32 block bytes interpreted as one little-endian integer. -/
def first256Nat (seed : List Nat) (row block lane : Nat) : Nat :=
  ((blockWords seed row block lane).take 8).reverse.foldl
    (fun value word => value * wordModulus + word) 0

end NightstreamFPrime.Spec.AjtaiSetupV1.ChaCha20
