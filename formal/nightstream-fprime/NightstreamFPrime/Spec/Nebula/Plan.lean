import Mathlib.Order.Basic
import NightstreamFPrime.Spec.Algebra

/-! Owns the verifier-owned Nebula memory plan of `specs/nebula-superneo.md`
§4: the geometry, the program images, validity rules 1–5 of §4.2, and the
sizes derived from the geometry. The relation, the rows, and `plan_digest`
take a `Geometry`, so they read no image; only the initial memory reads a
`Plan`. It does not own records, checks, or the security setup check
(rule 6). -/

namespace NightstreamFPrime.Spec.Nebula

/-- Spec §4.1 without the images. `n` is the segment length `N`. One package
serves every program of a geometry. -/
structure Geometry where
  r : ℕ
  μ : ℕ
  wTs : ℕ
  bOps : ℕ
  bScan : ℕ
  n : ℕ
  sMax : ℕ

/-- Spec §4.1: a geometry and a program, the ROM and RAM images, read at
addresses below `2 ^ r` and `2 ^ μ`. -/
structure Plan extends Geometry where
  rom : ℕ → ℕ
  ram : ℕ → ℕ

namespace Geometry

/-- ROM cells `R = 2 ^ r`. -/
def romSize (p : Geometry) : ℕ := 2 ^ p.r

/-- RAM cells `M = 2 ^ μ`. -/
def ramSize (p : Geometry) : ℕ := 2 ^ p.μ

/-- All cells `R + M` of the flat address space (spec §5). -/
def cells (p : Geometry) : ℕ := p.romSize + p.ramSize

/-- Bits of one operation slot (spec §6.1). -/
def opWidth (p : Geometry) : ℕ := 3 + p.μ + 64 + p.wTs

/-- Bits of one scan slot (spec §6.2). -/
def scanWidth (p : Geometry) : ℕ := 32 + p.wTs

/-- `W_seg` of spec §4.2: enough bits for every value in `[0, S_max]`. The open
row decomposes `S_max − 1 − seg_idx` with it. -/
def segWidth (p : Geometry) : ℕ := p.sMax.log2 + 1

/-- Upper bound `m_mem` on the size of each side of one segment's check. -/
def maxTuples (p : Geometry) : ℕ := p.cells + p.n * p.bOps

/-- Spec §4.2 rules 1–5. Rule 3 includes the lane widths: each lane has fewer
than `q` bits, so the length prefix of its packed block (spec §9.1) is one
canonical field element. -/
structure Valid (p : Geometry) : Prop where
  exactCover : p.n * p.bScan = p.cells
  timestampRange : p.sMax * p.n * p.bOps < 2 ^ p.wTs
  fieldEncoding : p.wTs ≤ 62
  laneBits : p.bOps * p.opWidth < goldilocksModulus ∧ p.bScan * p.scanWidth < goldilocksModulus
  addressWidth : p.r ≤ p.μ
  belowModulus : p.cells < goldilocksModulus
  positive : 0 < p.bOps ∧ 0 < p.bScan ∧ 0 < p.n ∧ 0 < p.sMax

theorem romSize_le_ramSize {p : Geometry} (valid : p.Valid) : p.romSize ≤ p.ramSize :=
  Nat.pow_le_pow_right Nat.zero_lt_two valid.addressWidth

end Geometry

/-- Spec §4.2 rules 1–5 for the geometry, and the image word width of §4.1. -/
structure Plan.Valid (p : Plan) : Prop extends Geometry.Valid p.toGeometry where
  romWords : ∀ a < p.romSize, p.rom a < 2 ^ 32
  ramWords : ∀ a < p.ramSize, p.ram a < 2 ^ 32

end NightstreamFPrime.Spec.Nebula
