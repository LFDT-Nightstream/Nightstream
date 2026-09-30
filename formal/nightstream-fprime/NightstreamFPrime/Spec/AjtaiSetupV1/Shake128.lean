/-!
Owns SHAKE128 (FIPS 202, Sections 3, 5.1 and 6.2) as used by
`nightstream-ajtai-shake128-wide256-v1`.

The Keccak-f[1600] state is one structure of 25 unboxed 64-bit lanes; field
`aXY` is lane `A[X, Y]`. Message and output bytes use the FIPS 202 lane order
`5Y + X`, each lane little-endian. This file defines exact semantics. It makes
no claim that SHAKE128 behaves as a random oracle.
-/

namespace NightstreamFPrime.Spec.AjtaiSetupV1.Shake128

/-- The Keccak-f[1600] state. Field `aXY` holds lane `A[X, Y]`. -/
structure State where
  a00 : UInt64
  a10 : UInt64
  a20 : UInt64
  a30 : UInt64
  a40 : UInt64
  a01 : UInt64
  a11 : UInt64
  a21 : UInt64
  a31 : UInt64
  a41 : UInt64
  a02 : UInt64
  a12 : UInt64
  a22 : UInt64
  a32 : UInt64
  a42 : UInt64
  a03 : UInt64
  a13 : UInt64
  a23 : UInt64
  a33 : UInt64
  a43 : UInt64
  a04 : UInt64
  a14 : UInt64
  a24 : UInt64
  a34 : UInt64
  a44 : UInt64

def State.zero : State :=
  ⟨0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0⟩

/-- Rotation to the left by `amount < 64` bits. Shift amounts are taken
modulo 64, so amount zero returns the lane unchanged. -/
@[inline] def rotl (value amount : UInt64) : UInt64 :=
  (value <<< amount) ||| (value >>> (64 - amount))

/-- Round constants `RC[i]` of step ι (FIPS 202, Section 3.2.5). -/
def roundConstants : List UInt64 :=
  [0x0000000000000001, 0x0000000000008082, 0x800000000000808a, 0x8000000080008000,
    0x000000000000808b, 0x0000000080000001, 0x8000000080008081, 0x8000000000008009,
    0x000000000000008a, 0x0000000000000088, 0x0000000080008009, 0x000000008000000a,
    0x000000008000808b, 0x800000000000008b, 0x8000000000008089, 0x8000000000008003,
    0x8000000000008002, 0x8000000000000080, 0x000000000000800a, 0x800000008000000a,
    0x8000000080008081, 0x8000000000008080, 0x0000000080000001, 0x8000000080008008]

/-- One round `ι ∘ χ ∘ π ∘ ρ ∘ θ` (FIPS 202, Section 3.3). The rotation
offsets are those of FIPS 202 Table 2. -/
def round (s : State) (roundConstant : UInt64) : State :=
  -- θ: C[x] = ⊕ A[x, y]; D[x] = C[x - 1] ⊕ rot(C[x + 1], 1).
  let c0 := s.a00 ^^^ s.a01 ^^^ s.a02 ^^^ s.a03 ^^^ s.a04
  let c1 := s.a10 ^^^ s.a11 ^^^ s.a12 ^^^ s.a13 ^^^ s.a14
  let c2 := s.a20 ^^^ s.a21 ^^^ s.a22 ^^^ s.a23 ^^^ s.a24
  let c3 := s.a30 ^^^ s.a31 ^^^ s.a32 ^^^ s.a33 ^^^ s.a34
  let c4 := s.a40 ^^^ s.a41 ^^^ s.a42 ^^^ s.a43 ^^^ s.a44
  let d0 := c4 ^^^ rotl c1 1
  let d1 := c0 ^^^ rotl c2 1
  let d2 := c1 ^^^ rotl c3 1
  let d3 := c2 ^^^ rotl c4 1
  let d4 := c3 ^^^ rotl c0 1
  -- ρ and π: B[x, y] = rot(A[(x + 3y) mod 5, x] ⊕ D[(x + 3y) mod 5], r[(x + 3y) mod 5, x]).
  let b00 := rotl (s.a00 ^^^ d0) 0
  let b10 := rotl (s.a11 ^^^ d1) 44
  let b20 := rotl (s.a22 ^^^ d2) 43
  let b30 := rotl (s.a33 ^^^ d3) 21
  let b40 := rotl (s.a44 ^^^ d4) 14
  let b01 := rotl (s.a30 ^^^ d3) 28
  let b11 := rotl (s.a41 ^^^ d4) 20
  let b21 := rotl (s.a02 ^^^ d0) 3
  let b31 := rotl (s.a13 ^^^ d1) 45
  let b41 := rotl (s.a24 ^^^ d2) 61
  let b02 := rotl (s.a10 ^^^ d1) 1
  let b12 := rotl (s.a21 ^^^ d2) 6
  let b22 := rotl (s.a32 ^^^ d3) 25
  let b32 := rotl (s.a43 ^^^ d4) 8
  let b42 := rotl (s.a04 ^^^ d0) 18
  let b03 := rotl (s.a40 ^^^ d4) 27
  let b13 := rotl (s.a01 ^^^ d0) 36
  let b23 := rotl (s.a12 ^^^ d1) 10
  let b33 := rotl (s.a23 ^^^ d2) 15
  let b43 := rotl (s.a34 ^^^ d3) 56
  let b04 := rotl (s.a20 ^^^ d2) 62
  let b14 := rotl (s.a31 ^^^ d3) 55
  let b24 := rotl (s.a42 ^^^ d4) 39
  let b34 := rotl (s.a03 ^^^ d0) 41
  let b44 := rotl (s.a14 ^^^ d1) 2
  -- χ: A[x, y] = B[x, y] ⊕ (¬B[x + 1, y] ∧ B[x + 2, y]); ι on A[0, 0].
  { a00 := b00 ^^^ (~~~b10 &&& b20) ^^^ roundConstant,
    a10 := b10 ^^^ (~~~b20 &&& b30),
    a20 := b20 ^^^ (~~~b30 &&& b40),
    a30 := b30 ^^^ (~~~b40 &&& b00),
    a40 := b40 ^^^ (~~~b00 &&& b10),
    a01 := b01 ^^^ (~~~b11 &&& b21),
    a11 := b11 ^^^ (~~~b21 &&& b31),
    a21 := b21 ^^^ (~~~b31 &&& b41),
    a31 := b31 ^^^ (~~~b41 &&& b01),
    a41 := b41 ^^^ (~~~b01 &&& b11),
    a02 := b02 ^^^ (~~~b12 &&& b22),
    a12 := b12 ^^^ (~~~b22 &&& b32),
    a22 := b22 ^^^ (~~~b32 &&& b42),
    a32 := b32 ^^^ (~~~b42 &&& b02),
    a42 := b42 ^^^ (~~~b02 &&& b12),
    a03 := b03 ^^^ (~~~b13 &&& b23),
    a13 := b13 ^^^ (~~~b23 &&& b33),
    a23 := b23 ^^^ (~~~b33 &&& b43),
    a33 := b33 ^^^ (~~~b43 &&& b03),
    a43 := b43 ^^^ (~~~b03 &&& b13),
    a04 := b04 ^^^ (~~~b14 &&& b24),
    a14 := b14 ^^^ (~~~b24 &&& b34),
    a24 := b24 ^^^ (~~~b34 &&& b44),
    a34 := b34 ^^^ (~~~b44 &&& b04),
    a44 := b44 ^^^ (~~~b04 &&& b14) }

/-- Keccak-f[1600]: the 24 rounds of Keccak-p[1600, 24]. -/
def permute (s : State) : State :=
  roundConstants.foldl round s

/-- SHAKE128 rate: 168 bytes, 21 lanes. -/
def rateBytes : Nat := 168

/-- SHAKE padding after `used` bytes of the last rate block (FIPS 202,
Section 6.2 and Algorithm 9): the suffix bits `1111`, then pad10*1. For
`used < rateBytes` it fills that block with `rateBytes - used` bytes. -/
def padding (used : Nat) : List Nat :=
  if used + 1 = rateBytes then [0x9F]
  else 0x1F :: (List.replicate (rateBytes - used - 2) 0 ++ [0x80])

/-- The padded message: a nonempty multiple of the rate. -/
def pad (message : List Nat) : List Nat :=
  message ++ padding (message.length % rateBytes)

/-- One lane from eight bytes, little-endian. -/
def laneOf (b0 b1 b2 b3 b4 b5 b6 b7 : Nat) : UInt64 :=
  UInt64.ofNat (b0 + 256 * (b1 + 256 * (b2 + 256 * (b3 + 256 * (b4 + 256 *
    (b5 + 256 * (b6 + 256 * b7)))))))

/-- The little-endian lanes of a byte string. A padded message has a
multiple of eight bytes, so no partial lane is dropped. -/
def lanesOfBytes : List Nat → List UInt64
  | b0 :: b1 :: b2 :: b3 :: b4 :: b5 :: b6 :: b7 :: rest =>
      laneOf b0 b1 b2 b3 b4 b5 b6 b7 :: lanesOfBytes rest
  | _ => []

/-- Absorb whole rate blocks: XOR 21 lanes into the rate lanes `5y + x`,
then permute (FIPS 202, Algorithm 8, step 6). -/
def absorbLanes (s : State) : List UInt64 → State
  | l0 :: l1 :: l2 :: l3 :: l4 :: l5 :: l6 :: l7 :: l8 :: l9 :: l10 :: l11 :: l12 :: l13 ::
      l14 :: l15 :: l16 :: l17 :: l18 :: l19 :: l20 :: rest =>
    absorbLanes (permute { s with
      a00 := s.a00 ^^^ l0, a10 := s.a10 ^^^ l1, a20 := s.a20 ^^^ l2,
      a30 := s.a30 ^^^ l3, a40 := s.a40 ^^^ l4,
      a01 := s.a01 ^^^ l5, a11 := s.a11 ^^^ l6, a21 := s.a21 ^^^ l7,
      a31 := s.a31 ^^^ l8, a41 := s.a41 ^^^ l9,
      a02 := s.a02 ^^^ l10, a12 := s.a12 ^^^ l11, a22 := s.a22 ^^^ l12,
      a32 := s.a32 ^^^ l13, a42 := s.a42 ^^^ l14,
      a03 := s.a03 ^^^ l15, a13 := s.a13 ^^^ l16, a23 := s.a23 ^^^ l17,
      a33 := s.a33 ^^^ l18, a43 := s.a43 ^^^ l19,
      a04 := s.a04 ^^^ l20 }) rest
  | _ => s

def absorb (message : List Nat) : State :=
  absorbLanes State.zero (lanesOfBytes (pad message))

/-- The 21 rate lanes in output order. -/
def rateLanes (s : State) : List UInt64 :=
  [s.a00, s.a10, s.a20, s.a30, s.a40, s.a01, s.a11, s.a21, s.a31, s.a41,
    s.a02, s.a12, s.a22, s.a32, s.a42, s.a03, s.a13, s.a23, s.a33, s.a43, s.a04]

/-- Squeeze `blocks` rate blocks; the state is permuted only between them. -/
def squeezeBlocks (s : State) : Nat → List UInt64
  | 0 => []
  | 1 => rateLanes s
  | blocks + 2 => rateLanes s ++ squeezeBlocks (permute s) (blocks + 1)

/-- The first `count` output lanes of SHAKE128(message). -/
def lanes (message : List Nat) (count : Nat) : List UInt64 :=
  (squeezeBlocks (absorb message) ((count + 20) / 21)).take count

/-- The eight little-endian bytes of one lane. -/
def laneBytes (lane : UInt64) : List Nat :=
  (List.range 8).map fun byte => lane.toNat / 256 ^ byte % 256

/-- The first `count` output bytes of SHAKE128(message). -/
def bytes (message : List Nat) (count : Nat) : List Nat :=
  ((lanes message ((count + 7) / 8)).flatMap laneBytes).take count

theorem squeezeBlocks_length (s : State) (blocks : Nat) :
    (squeezeBlocks s blocks).length = 21 * blocks := by
  induction blocks generalizing s with
  | zero => rfl
  | succ blocks ih =>
      cases blocks with
      | zero => rfl
      | succ blocks =>
          simp only [squeezeBlocks, List.length_append, ih]
          simp only [rateLanes, List.length_cons, List.length_nil]
          omega

/-- Every requested output lane is a squeezed lane. -/
theorem lanes_length (message : List Nat) (count : Nat) :
    (lanes message count).length = count := by
  simp only [lanes, List.length_take, squeezeBlocks_length]
  omega

end NightstreamFPrime.Spec.AjtaiSetupV1.Shake128
