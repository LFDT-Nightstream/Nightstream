import Mathlib.Order.Basic
import NightstreamFPrime.Spec.Algebra

/-! Owns the verifier-owned Nebula memory plan of SPEC.md section 7.1:
the geometry, the program images, validity rules 1 to 5, and the
sizes derived from the geometry. The relation, the rows, and `plan_digest`
take a `Geometry`, so they read no image; only the initial memory reads a
`Plan`. It does not own records, checks, or the security setup check
(rule 6). -/

namespace NightstreamFPrime.Spec.Nebula

/-- SPEC.md section 7.1 without the images. `n` is the segment length `N`. One package
serves every program of a geometry. -/
structure Geometry where
  r : ℕ
  μ : ℕ
  wTs : ℕ
  bOps : ℕ
  bScan : ℕ
  n : ℕ
  sMax : ℕ

/-- SPEC.md section 7.1: a geometry and a program, the ROM and RAM images, read at
addresses below `2 ^ r` and `2 ^ μ`. -/
structure Plan extends Geometry where
  rom : ℕ → ℕ
  ram : ℕ → ℕ

namespace Geometry

/-- ROM cells `R = 2 ^ r`. -/
def romSize (p : Geometry) : ℕ := 2 ^ p.r

/-- RAM cells `M = 2 ^ μ`. -/
def ramSize (p : Geometry) : ℕ := 2 ^ p.μ

/-- All cells `R + M` of the flat address space (SPEC.md section 7.2). -/
def cells (p : Geometry) : ℕ := p.romSize + p.ramSize

/-- Bits of an address field: enough for a ROM address and for a RAM address. -/
def addrWidth (p : Geometry) : ℕ := max p.r p.μ

/-- Bits of one operation slot (SPEC.md section 7.2). -/
def opWidth (p : Geometry) : ℕ := 3 + p.addrWidth + 64 + p.wTs

/-- Bits of one scan slot (SPEC.md section 7.2). -/
def scanWidth (p : Geometry) : ℕ := 32 + p.wTs

/-- `W_seg` of SPEC.md section 7.1: enough bits for every value in `[0, S_max]`. The open
row decomposes `S_max − 1 − seg_idx` with it. -/
def segWidth (p : Geometry) : ℕ := p.sMax.log2 + 1

/-- Upper bound `m_mem` on the size of each side of one segment's check. -/
def maxTuples (p : Geometry) : ℕ := p.cells + p.n * p.bOps

/-- SPEC.md section 7.1, rules 1 to 5. Rule 3 bounds the timestamp width by 31, so that
`v + 2^32 · t` of a fingerprint (SPEC.md section 7.3) is one canonical field element. Rule 3
includes the lane widths: each lane has fewer than `q` bits, so the length prefix of its
packed block (SPEC.md section 7.4) is one canonical field element. -/
structure Valid (p : Geometry) : Prop where
  exactCover : p.n * p.bScan = p.cells
  timestampRange : p.sMax * p.n * p.bOps < 2 ^ p.wTs
  fieldEncoding : p.wTs ≤ 31
  laneBits : p.bOps * p.opWidth < fieldModulus ∧ p.bScan * p.scanWidth < fieldModulus
  belowModulus : p.cells < fieldModulus
  positive : 0 < p.bOps ∧ 0 < p.bScan ∧ 0 < p.n ∧ 0 < p.sMax

theorem two_pow_addrWidth_le_cells (p : Geometry) : 2 ^ p.addrWidth ≤ p.cells := by
  unfold addrWidth cells romSize ramSize
  rw [Nat.max_def]
  split
  · exact Nat.le_add_left _ _
  · exact Nat.le_add_right _ _

/-- An address field holds fewer than 64 bits, so its value is a canonical field element. -/
theorem addrWidth_le {p : Geometry} (valid : p.Valid) : p.addrWidth ≤ 63 := by
  have below := Nat.lt_of_le_of_lt (two_pow_addrWidth_le_cells p) valid.belowModulus
  refine Nat.le_of_not_lt fun large => ?_
  have big : 2 ^ 64 ≤ 2 ^ p.addrWidth := Nat.pow_le_pow_right (by decide) large
  have : fieldModulus < 2 ^ 64 := by decide
  omega

end Geometry

/-- SPEC.md section 7.1, rules 1 to 5 for the geometry, and the image word width. -/
structure Plan.Valid (p : Plan) : Prop extends Geometry.Valid p.toGeometry where
  romWords : ∀ a < p.romSize, p.rom a < 2 ^ 32
  ramWords : ∀ a < p.ramSize, p.ram a < 2 ^ 32

end NightstreamFPrime.Spec.Nebula
