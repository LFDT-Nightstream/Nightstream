import Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler

/-!
Independent mathematics for the production `Pi_RLC` coefficient alphabet.

Protocol: SuperNeo `Pi_RLC` inside the candidate noninteractive NIFS.
Phase: one scalar challenge's coefficient rejection sampler.
Constraint family: 16-bit candidate classification, five-symbol decoding,
first-accepted selection, and the fixed 54-of-64 success boundary.

Owns: the exact numerical sampler parameters; the verifier-owned accepted
domain; centered coefficient bounds; a bijection between accepted chunks and
`Fin 13107 × Fin 5`; the conditional bounded/reference equivalence at 54 of
64; and the theorem that every successful least cursor passes six complete
eight-candidate digest windows.

Does not own: candidate generation, transcript serialization, Poseidon2,
successor transcript state, rotation/ring-scalar assembly, the SuperNeo
strong-set theorem, probability claims, Rust, R1CS, or constraint counts.

Emits constraints: no.

Authority boundary: acceptance and decoding are mathematical functions of the
candidate chunk. The prover supplies neither. The 64-candidate theorem is
conditional on a successful bounded execution; it does not assume that every
stream succeeds and does not yet prove that a concrete transcript produces
this stream or advances by exactly eight Poseidon2 digests.

| Protocol | Phase | Mathematical object | Exact obligation |
|---|---|---|---|
| `Pi_RLC` | parameters | `parameter_values` | fix 16-bit chunks, rejection `65535`, alphabet `5`, need `54`, and bound `64 = 8 × 8` |
| `Pi_RLC` | acceptance | `accepts_eq_true_iff` | accept exactly chunks below the rejection bucket |
| `Pi_RLC` | decoding | `centeredValue_bounds` | every decoded symbol represents an integer in `[-2, 2]` |
| `Pi_RLC` | balance | `acceptedFactorization` | accepted chunks are exactly `Fin 13107 × Fin 5` |
| `Pi_RLC` | bounded/reference | `sample54of64_eq_some_iff_reference_within` | fixed execution equals the least terminating reference run inside 64 candidates |
| `Pi_RLC` | digest window | `successful_cursor_after_sixth_digest` | success implies `48 < consumed ≤ 64` |
-/

namespace Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler.ProductionAlphabet

open Nightstream.SuperNeo.Sampling

/-- Number of distinct 16-bit candidates. -/
def chunkModulus : Nat := 65536

/-- The unique 16-bit candidate excluded by rejection sampling. -/
def rejectionBucket : Nat := 65535

/-- Number of centered coefficient symbols `{-2, -1, 0, 1, 2}`. -/
def alphabetSize : Nat := 5

/-- Number of coefficients in one production rotation/ring scalar. -/
def coefficientCount : Nat := 54

/-- Number of exact 16-bit candidates exposed by one four-lane digest. -/
def chunksPerDigest : Nat := 8

/-- Fixed number of digest rounds in the bounded recursive verifier. -/
def digestRounds : Nat := 8

/-- Fixed candidate prefix inspected by the recursive verifier. -/
def candidateBound : Nat := 64

/-- Number of complete five-symbol blocks in the accepted domain. -/
def acceptedQuotientCount : Nat := 13107

theorem parameter_values :
    chunkModulus = 65536 /\
    rejectionBucket = 65535 /\
    alphabetSize = 5 /\
    coefficientCount = 54 /\
    chunksPerDigest = 8 /\
    digestRounds = 8 /\
    candidateBound = 64 /\
    candidateBound = digestRounds * chunksPerDigest /\
    rejectionBucket = acceptedQuotientCount * alphabetSize := by
  decide

/-- One candidate chunk, canonical by construction. -/
abbrev Chunk := Fin chunkModulus

/-- One alphabet index. Its centered integer interpretation is defined below. -/
abbrev Coefficient := Fin alphabetSize

/-- Verifier-owned rejection predicate. -/
def accepts (chunk : Chunk) : Bool :=
  decide (chunk.val < rejectionBucket)

/-- Verifier-owned residue decoder. -/
def symbol (chunk : Chunk) : Coefficient :=
  ⟨chunk.val % alphabetSize, Nat.mod_lt _ (by decide)⟩

/-- The production-shaped mathematical verifier for one coefficient. -/
def verifier : FirstAccepted.Verifier Chunk Coefficient where
  accepts := accepts
  symbol := symbol

theorem accepts_eq_true_iff (chunk : Chunk) :
    verifier.accepts chunk = true ↔ chunk.val < rejectionBucket := by
  simp [verifier, accepts]

/-- Since a chunk is already below `65536`, rejection is exactly equality with
the final bucket. -/
theorem accepts_eq_true_iff_ne_rejectionBucket (chunk : Chunk) :
    verifier.accepts chunk = true ↔ chunk.val ≠ rejectionBucket := by
  rw [accepts_eq_true_iff]
  have upper := chunk.isLt
  simp [chunkModulus, rejectionBucket] at upper ⊢
  omega

/-- Centered integer represented by an alphabet residue. -/
def centeredValue (coefficient : Coefficient) : Int :=
  (coefficient.val : Int) - 2

/-- The decoder's codomain itself proves the exact centered strong-set bound
for individual coefficients; no circuit witness is trusted for this fact. -/
theorem centeredValue_bounds (coefficient : Coefficient) :
    (-2 : Int) ≤ centeredValue coefficient /\ centeredValue coefficient ≤ 2 := by
  have upper : coefficient.val < 5 := by
    simpa [alphabetSize] using coefficient.isLt
  unfold centeredValue
  omega

/-- Accepted chunks, with acceptance carried as proof data. -/
abbrev AcceptedChunk :=
  { chunk : Chunk // verifier.accepts chunk = true }

theorem AcceptedChunk.val_lt_rejectionBucket (chunk : AcceptedChunk) :
    chunk.val.val < rejectionBucket :=
  (accepts_eq_true_iff chunk.val).mp chunk.property

/-- Quotient/remainder coordinates for one accepted chunk. -/
def factor (chunk : AcceptedChunk) :
    Fin acceptedQuotientCount × Coefficient :=
  let quotient := chunk.val.val / alphabetSize
  have quotientLt : quotient < acceptedQuotientCount := by
    have accepted : chunk.val.val < 65535 := by
      simpa [rejectionBucket] using chunk.val_lt_rejectionBucket
    change chunk.val.val / 5 < 13107
    omega
  ⟨⟨quotient, quotientLt⟩, symbol chunk.val⟩

/-- Recompose an accepted chunk from its quotient and alphabet residue. -/
def combine (coordinates : Fin acceptedQuotientCount × Coefficient) :
    AcceptedChunk :=
  let value := coordinates.1.val * alphabetSize + coordinates.2.val
  have accepted : value < rejectionBucket := by
    have quotientLt : coordinates.1.val < 13107 := by
      simpa [acceptedQuotientCount] using coordinates.1.isLt
    have residueLt : coordinates.2.val < 5 := by
      simpa [alphabetSize] using coordinates.2.isLt
    change coordinates.1.val * 5 + coordinates.2.val < 65535
    omega
  have canonical : value < chunkModulus := by
    have acceptedNumeric : value < 65535 := by
      simpa [rejectionBucket] using accepted
    change value < 65536
    omega
  ⟨⟨value, canonical⟩, (accepts_eq_true_iff _).mpr accepted⟩

theorem combine_factor (chunk : AcceptedChunk) :
    combine (factor chunk) = chunk := by
  apply Subtype.ext
  apply Fin.ext
  change chunk.val.val / 5 * 5 + chunk.val.val % 5 = chunk.val.val
  simpa [Nat.mul_comm] using Nat.div_add_mod chunk.val.val 5

theorem factor_combine
    (coordinates : Fin acceptedQuotientCount × Coefficient) :
    factor (combine coordinates) = coordinates := by
  rcases coordinates with ⟨quotient, residue⟩
  have residueLt : residue.val < 5 := by
    simpa [alphabetSize] using residue.isLt
  apply Prod.ext
  · apply Fin.ext
    change (quotient.val * 5 + residue.val) / 5 = quotient.val
    omega
  · apply Fin.ext
    change (quotient.val * 5 + residue.val) % 5 = residue.val
    omega

/-- Exact accepted-domain decomposition. `factor` and `combine` are mutual
inverses, so accepted chunks are in bijection with
`Fin 13107 × Fin 5`. In particular, every residue has the same quotient
coordinate set. This balance argument does not depend on implementation rows. -/
theorem acceptedFactorization :
    (∀ chunk, combine (factor chunk) = chunk) /\
      (∀ coordinates, factor (combine coordinates) = coordinates) :=
  ⟨combine_factor, factor_combine⟩

/-- Filtering cannot create candidates. This elementary bound is stated here
so the digest-window proof does not depend on an implementation counter. -/
theorem acceptedCount_le_length (candidates : List Chunk) :
    FirstAccepted.acceptedCount verifier candidates ≤ candidates.length := by
  induction candidates with
  | nil =>
      simp [FirstAccepted.acceptedCount, FirstAccepted.acceptedCandidates]
  | cons head tail inductionHypothesis =>
      cases accepted : verifier.accepts head with
      | false =>
          simpa [FirstAccepted.acceptedCount,
            FirstAccepted.acceptedCandidates, accepted] using
              Nat.le_succ_of_le inductionHypothesis
      | true =>
          simpa [FirstAccepted.acceptedCount,
            FirstAccepted.acceptedCandidates, accepted] using
              Nat.succ_le_succ inductionHypothesis

/-- Exact conditional equivalence specialized to the production 54-of-64
parameters. It asserts no unconditional termination or success probability. -/
theorem sample54of64_eq_some_iff_reference_within
    {stream : FirstAccepted.CandidateStream Chunk}
    {output : List Coefficient} :
    FirstAccepted.boundedSample verifier coefficientCount
        (FirstAccepted.streamPrefix stream candidateBound) = some output ↔
      ∃ consumed,
        consumed ≤ candidateBound /\
          FirstAccepted.ReferenceExecution verifier coefficientCount stream
            output consumed := by
  exact FirstAccepted.boundedSample_eq_some_iff_referenceExecution_within

/-- The least successful cursor must occur after candidate 48 and no later
than candidate 64. Thus a whole-digest implementation necessarily completes
at least six eight-candidate windows. The fixed implementation consumes all
eight windows independently of this least cursor. -/
theorem successful_cursor_after_sixth_digest
    {stream : FirstAccepted.CandidateStream Chunk}
    (execution : FirstAccepted.BoundedExecution verifier coefficientCount
      stream candidateBound) :
    6 * chunksPerDigest < execution.consumed /\
      execution.consumed ≤ digestRounds * chunksPerDigest := by
  have needLeConsumed : coefficientCount ≤ execution.consumed := by
    calc
      coefficientCount ≤
          FirstAccepted.acceptedCount verifier
            (FirstAccepted.streamPrefix stream execution.consumed) :=
        execution.reference.enough
      _ ≤ (FirstAccepted.streamPrefix stream execution.consumed).length :=
        acceptedCount_le_length _
      _ = execution.consumed := FirstAccepted.streamPrefix_length _ _
  have needNumeric : 54 ≤ execution.consumed := by
    simpa [coefficientCount] using needLeConsumed
  have withinNumeric : execution.consumed ≤ 64 := by
    simpa [candidateBound] using execution.within
  constructor
  · change 48 < execution.consumed
    omega
  · change execution.consumed ≤ 64
    exact withinNumeric

end Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler.ProductionAlphabet
