import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.SelectionRows
import Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler.ProductionAlphabet

/-!
Profile-independent source order for one bounded `Pi_RLC` scalar sampler.

Assurance tier: executable mathematical correspondence. This file defines how
the fixed 64 candidates are addressed and how a decoded alphabet coefficient
is represented in the Goldilocks field. It does not inspect any owner,
transcript trace, or generated row artifact.

Owns: the `block -> lane -> 16-bit part` candidate address; exact recomposition
to source order; and centered-field encoding of one verifier-decoded symbol.

Does not own: candidate generation, Poseidon2 provenance, accept/reject
decisions, accepted-prefix counts, selection rows, scalar assembly, Rust
conformance, row removal, or costs.

Emits constraints: no.

Authority boundary: this is an independent indexing and encoding convention.
A production column acquires this meaning only through a separate refinement
theorem.

| Protocol | Phase | Constraint family | Mathematical object | Guarantee |
|---|---|---|---|---|
| `Pi_RLC` | bounded sampler | candidate address | `block/lane/part` | `candidate = 16*block + 4*lane + part` |
| `Pi_RLC` | symbol decoding | centered encoding | coefficient in `{0,1,2,3,4}` | canonical Goldilocks representative of `coefficient - 2` |
-/

namespace Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.CandidateOrder

open Nightstream.Implementation.R1CS
open Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler

/-- Hierarchical address of one candidate in block-major, lane-major, then
16-bit-part order. -/
structure Address where
  block : Fin 4
  lane : Fin 4
  part : Fin 4

/-- Independent quotient/remainder decoding of one source-order index. -/
def address (candidate : Fin SelectionRows.candidateCount) : Address :=
  { block := ⟨candidate.val / 16, by
      have candidateLt := candidate.isLt
      simp only [SelectionRows.candidateCount] at candidateLt
      omega⟩
    lane := ⟨(candidate.val % 16) / 4, by
      have remainderLt := Nat.mod_lt candidate.val (by decide : 0 < 16)
      omega⟩
    part := ⟨candidate.val % 4, Nat.mod_lt _ (by decide)⟩ }

/-- Recomposition records the precise source-order convention used by the
semantic candidate vector. -/
theorem address_recomposes (candidate : Fin SelectionRows.candidateCount) :
    16 * (address candidate).block.val +
        4 * (address candidate).lane.val + (address candidate).part.val =
      candidate.val := by
  have outer := Nat.div_add_mod candidate.val 16
  have inner := Nat.div_add_mod (candidate.val % 16) 4
  simp only [address]
  omega

/-- Canonical Goldilocks representative of the centered coefficient
`coefficient - 2`. -/
def centeredField (coefficient : ProductionAlphabet.Coefficient) : Nat :=
  (coefficient.val + (goldilocksP - 2)) % goldilocksP

end Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.CandidateOrder
