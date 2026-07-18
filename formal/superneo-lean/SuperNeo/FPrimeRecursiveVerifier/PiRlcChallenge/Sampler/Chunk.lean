import Mathlib.Data.Fin.Basic
import Mathlib.Tactic
import SuperNeo.Primitives.Field
import SuperNeo.FPrimeRecursiveVerifier.PiRlcChallenge.Parameters

/-!
Owns: native arithmetic semantics for one little-endian 16-bit sampler chunk.

Does not own: transcript origin, list-level acceptance/selection, or concrete
Goldilocks R1CS row conformance.

Emits constraints: no.

Authority boundary: the input is already a canonical `Fin 65536` chunk; this
file proves its unique decomposition and semantic outputs.

| Predicate/theorem | Rust stage | Guarantee | Assumptions | Permits row removal? |
|---|---|---|---|---|
| `Accepted`, `accepted_iff_lt_bucket` | `challenge.sampler.chunk.accept` | Accepts exactly values below 65,535 | Canonical 16-bit chunk | No — Rust refinement open |
| `ChunkArithmeticHolds`, `arithmeticWitness_unique` | `challenge.sampler.chunk.mod5` | Unique bounded quotient and residue reconstruct the chunk | Canonical chunk and bounded witness | No — Rust refinement open |
| `symbol_mem_alphabet`, `mod5Polynomial_zero_iff` | `challenge.sampler.chunk.symbol_and_prefix` | Residue maps exactly into `[-2, 2]` | Canonical residue | No — Rust refinement open |
| `chunkArithmetic_exact` | `challenge.sampler.chunk` | Arithmetic witness exists exactly for the canonical transition | Model equations above | No — Rust refinement open |

Concrete selection-row refinement belongs in `Refinement/SelectionRows.lean`.
-/

namespace SuperNeo.FPrimeRecursiveVerifier.PiRlcChallenge

/-- Canonical 16-bit sampler input. -/
abbrev Chunk := Fin chunkModulus

/-- Mathematical accept predicate for the unbiased size-five sampler. -/
def Accepted (chunk : Chunk) : Prop :=
  chunk.val ≠ rejectionBucket

/-- Executable accept bit used for list filtering and conformance artifacts. -/
def acceptBit (chunk : Chunk) : Bool :=
  decide (chunk.val ≠ rejectionBucket)

/-- Canonical mod-5 quotient. -/
def quotient (chunk : Chunk) : Nat :=
  chunk.val / alphabetSize

/-- Canonical mod-5 residue. -/
def residue (chunk : Chunk) : Nat :=
  chunk.val % alphabetSize

/-- Centered production coefficient in `[-2, 2]`. -/
def symbol (chunk : Chunk) : Int :=
  Int.ofNat (residue chunk) - 2

/-- Exact root polynomial emitted by Rust for the unsigned mod-5 index. -/
def mod5Polynomial (value : F) : F :=
  value * (value - 1) * (value - 2) * (value - 3) * (value - 4)

/-- One chunk's public semantic outputs. -/
structure ChunkOutput where
  accepted : Bool
  symbol : Int
  nextPrefix : Nat
deriving Repr, DecidableEq

/-- Private quotient and residue used by the arithmetic leaf. -/
structure ChunkArithmeticWitness where
  quotient : Nat
  residue : Nat
deriving Repr, DecidableEq

/-- Exact transition implemented by one validated chunk. -/
def ChunkTransition
    (chunk : Chunk) (priorCount : Nat) (output : ChunkOutput) : Prop :=
  output.accepted = acceptBit chunk ∧
    output.symbol = symbol chunk ∧
    output.nextPrefix = priorCount + if acceptBit chunk then 1 else 0

/-- Bounded quotient/remainder equations emitted by the mod-5 arithmetic leaf. -/
def ChunkArithmeticHolds
    (chunk : Chunk) (witness : ChunkArithmeticWitness) : Prop :=
  witness.quotient < 2 ^ quotientBits ∧
    witness.residue < alphabetSize ∧
    chunk.val = alphabetSize * witness.quotient + witness.residue

theorem accepted_iff_lt_bucket (chunk : Chunk) :
    Accepted chunk ↔ chunk.val < rejectionBucket := by
  simp only [Accepted, rejectionBucket]
  have hChunk : chunk.val < 65_536 := by
    simpa only [chunkModulus] using chunk.isLt
  omega

theorem acceptBit_eq_true_iff (chunk : Chunk) :
    acceptBit chunk = true ↔ Accepted chunk := by
  simp [acceptBit, Accepted]

theorem residue_lt_alphabet (chunk : Chunk) :
    residue chunk < alphabetSize := by
  exact Nat.mod_lt _ (by decide : 0 < alphabetSize)

theorem quotient_fits_bits (chunk : Chunk) :
    quotient chunk < 2 ^ quotientBits := by
  have hChunk := chunk.isLt
  simp only [chunkModulus] at hChunk
  simp [quotient, alphabetSize, quotientBits]
  omega

theorem chunk_decomposition (chunk : Chunk) :
    chunk.val = alphabetSize * quotient chunk + residue chunk := by
  simpa [quotient, residue] using (Nat.div_add_mod chunk.val alphabetSize).symm

theorem canonicalArithmeticHolds (chunk : Chunk) :
    ChunkArithmeticHolds chunk
      { quotient := quotient chunk, residue := residue chunk } := by
  exact ⟨quotient_fits_bits chunk, residue_lt_alphabet chunk,
    chunk_decomposition chunk⟩

theorem arithmeticWitness_unique
    (chunk : Chunk) (witness : ChunkArithmeticWitness)
    (hHolds : ChunkArithmeticHolds chunk witness) :
    witness.quotient = quotient chunk ∧
      witness.residue = residue chunk := by
  rcases hHolds with ⟨_, hResidue, hEquation⟩
  constructor
  · change witness.quotient = chunk.val / 5
    change witness.residue < 5 at hResidue
    change chunk.val = 5 * witness.quotient + witness.residue at hEquation
    omega
  · have hMod := congrArg (fun value => value % alphabetSize) hEquation
    have hResidueEq : residue chunk = witness.residue := by
      simpa [residue, Nat.add_mod, Nat.mul_mod,
        Nat.mod_eq_of_lt hResidue] using hMod
    exact hResidueEq.symm

theorem symbol_mem_alphabet (chunk : Chunk) :
    (-2 : Int) ≤ symbol chunk ∧ symbol chunk ≤ 2 := by
  have hResidue := residue_lt_alphabet chunk
  change residue chunk < 5 at hResidue
  interval_cases hValue : residue chunk <;> simp [symbol, hValue]

/--
The Rust product chain accepts exactly the five unsigned residues. Assumes the
Goldilocks field; does not cover quotient recomposition or transcript origin.
Maps to `chunk::enforce_mod5_index`.
-/
theorem mod5Polynomial_zero_iff (value : F) :
    mod5Polynomial value = 0 ↔
      value = 0 ∨ value = 1 ∨ value = 2 ∨ value = 3 ∨ value = 4 := by
  simp only [mod5Polynomial, mul_eq_zero, sub_eq_zero]
  tauto

theorem chunkArithmetic_exact (chunk : Chunk) :
    ∃! witness, ChunkArithmeticHolds chunk witness := by
  refine ⟨
    { quotient := quotient chunk, residue := residue chunk },
    canonicalArithmeticHolds chunk,
    ?_⟩
  intro witness hWitness
  rcases arithmeticWitness_unique chunk witness hWitness with ⟨hQ, hR⟩
  cases witness
  simp_all

end SuperNeo.FPrimeRecursiveVerifier.PiRlcChallenge
