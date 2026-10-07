import Mathlib.Order.Basic
import NightstreamFPrime.Spec.Algebra

/-! Owns the verifier-owned Nebula memory plan of `specs/nebula-superneo.md`
§4: its fields, validity rules 1–5 of §4.2, and the sizes derived from it.
It does not own records, checks, or the security setup check (rule 6). -/

namespace NightstreamFPrime.Spec.Nebula

/-- Spec §4.1. `n` is the segment length `N`; `rom` and `ram` are the images,
read at addresses below `2 ^ r` and `2 ^ μ`. -/
structure Plan where
  r : ℕ
  μ : ℕ
  wTs : ℕ
  bOps : ℕ
  bScan : ℕ
  n : ℕ
  sMax : ℕ
  rom : ℕ → ℕ
  ram : ℕ → ℕ

namespace Plan

/-- ROM cells `R = 2 ^ r`. -/
def romSize (p : Plan) : ℕ := 2 ^ p.r

/-- RAM cells `M = 2 ^ μ`. -/
def ramSize (p : Plan) : ℕ := 2 ^ p.μ

/-- All cells `R + M` of the flat address space (spec §5). -/
def cells (p : Plan) : ℕ := p.romSize + p.ramSize

/-- Bits of one operation slot (spec §6.1). -/
def opWidth (p : Plan) : ℕ := 3 + p.μ + 64 + p.wTs

/-- Bits of one scan slot (spec §6.2). -/
def scanWidth (p : Plan) : ℕ := 32 + p.wTs

/-- Upper bound `m_mem` on the size of each side of one segment's check. -/
def maxTuples (p : Plan) : ℕ := p.cells + p.n * p.bOps

/-- Spec §4.2 rules 1–5, and the image word width of §4.1. -/
structure Valid (p : Plan) : Prop where
  exactCover : p.n * p.bScan = p.cells
  timestampRange : p.sMax * p.n * p.bOps < 2 ^ p.wTs
  fieldEncoding : p.wTs ≤ 62
  addressWidth : p.r ≤ p.μ
  belowModulus : p.cells < goldilocksModulus
  positive : 0 < p.bOps ∧ 0 < p.bScan ∧ 0 < p.n ∧ 0 < p.sMax
  romWords : ∀ a < p.romSize, p.rom a < 2 ^ 32
  ramWords : ∀ a < p.ramSize, p.ram a < 2 ^ 32

theorem romSize_le_ramSize {p : Plan} (valid : p.Valid) : p.romSize ≤ p.ramSize :=
  Nat.pow_le_pow_right Nat.zero_lt_two valid.addressWidth

end Plan

end NightstreamFPrime.Spec.Nebula
