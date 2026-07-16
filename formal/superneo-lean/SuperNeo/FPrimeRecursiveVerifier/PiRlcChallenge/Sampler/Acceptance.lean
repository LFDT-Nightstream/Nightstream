import SuperNeo.FPrimeRecursiveVerifier.PiRlcChallenge.Sampler.Chunk

/-!
Owns: arithmetic semantics for the fixed enough-accepts gate.

Does not own: chunk validity, symbol selection, or transcript origin.

Emits constraints: no.

Authority boundary: this predicate counts validated chunk accept bits; it does
not make the chunks authoritative.

| Predicate/theorem | Rust stage | Guarantee | Assumptions | Permits row removal? |
|---|---|---|---|---|
| `EnoughAccepts` | `challenge.sampler.acceptance_bound` | At least 54 supplied chunks accept | Validated chunks | No — Rust refinement open |
| `AcceptanceArithmeticHolds` | `challenge.sampler.acceptance_bound` | `acceptedCount = 54 + slack` with four-bit slack | Supplied witness | No — Rust refinement open |
| `enoughAccepts_iff_rejected_le` | `challenge.sampler.acceptance_bound` | Enough accepts iff at most ten of 64 chunks reject | Fixed length 64 | No — Rust refinement open |
| `acceptanceArithmetic_exact` | `challenge.sampler.acceptance_bound` | Arithmetic witness exists exactly when acceptance holds | Fixed length 64 | No — Rust refinement open |
-/

namespace SuperNeo.FPrimeRecursiveVerifier.PiRlcChallenge

/-- Number of accepted chunks in transcript order. -/
def acceptedCount (chunks : List Chunk) : Nat :=
  (chunks.filter acceptBit).length

/-- Number of rejected chunks in transcript order. -/
def rejectedCount (chunks : List Chunk) : Nat :=
  (chunks.filter fun chunk => !(acceptBit chunk)).length

/-- Semantic gate required before selecting 54 coefficients. -/
def EnoughAccepts (chunks : List Chunk) : Prop :=
  outputLength ≤ acceptedCount chunks

/-- Private four-bit slack emitted by the Rust acceptance leaf. -/
structure AcceptanceWitness where
  slack : Nat
deriving Repr, DecidableEq

/-- Native arithmetic equations corresponding to the acceptance leaf. -/
def AcceptanceArithmeticHolds
    (chunks : List Chunk) (witness : AcceptanceWitness) : Prop :=
  acceptedCount chunks = outputLength + witness.slack ∧
    witness.slack < 2 ^ slackBits

theorem accepted_add_rejected (chunks : List Chunk) :
    acceptedCount chunks + rejectedCount chunks = chunks.length := by
  simpa [acceptedCount, rejectedCount, Nat.add_comm] using
    (List.length_eq_length_filter_add (l := chunks) acceptBit).symm

theorem enoughAccepts_iff_rejected_le
    (chunks : List Chunk)
    (hLength : chunks.length = chunksPerSample) :
    EnoughAccepts chunks ↔ rejectedCount chunks ≤ 10 := by
  have hPartition := accepted_add_rejected chunks
  simp only [EnoughAccepts, outputLength, chunksPerSample] at *
  omega

theorem acceptanceArithmetic_sound
    (chunks : List Chunk) (witness : AcceptanceWitness)
    (hHolds : AcceptanceArithmeticHolds chunks witness) :
    EnoughAccepts chunks := by
  rcases hHolds with ⟨hCount, _⟩
  simp only [EnoughAccepts]
  omega

theorem acceptanceArithmetic_complete
    (chunks : List Chunk)
    (hLength : chunks.length = chunksPerSample)
    (hEnough : EnoughAccepts chunks) :
    AcceptanceArithmeticHolds chunks
      { slack := acceptedCount chunks - outputLength } := by
  have hPartition := accepted_add_rejected chunks
  constructor
  · simp only [EnoughAccepts] at hEnough
    change acceptedCount chunks =
      outputLength + (acceptedCount chunks - outputLength)
    omega
  · change acceptedCount chunks - outputLength < 2 ^ slackBits
    simp only [chunksPerSample] at hLength
    simp only [EnoughAccepts] at hEnough
    norm_num [outputLength, slackBits]
    omega

theorem acceptanceArithmetic_exact
    (chunks : List Chunk)
    (hLength : chunks.length = chunksPerSample) :
    (∃ witness, AcceptanceArithmeticHolds chunks witness) ↔
      EnoughAccepts chunks := by
  constructor
  · rintro ⟨witness, hWitness⟩
    exact acceptanceArithmetic_sound chunks witness hWitness
  · intro hEnough
    exact ⟨_, acceptanceArithmetic_complete chunks hLength hEnough⟩

end SuperNeo.FPrimeRecursiveVerifier.PiRlcChallenge
