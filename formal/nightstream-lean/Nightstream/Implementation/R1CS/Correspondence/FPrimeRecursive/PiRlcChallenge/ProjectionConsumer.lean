import Nightstream.Implementation.R1CS.Artifacts.FPrimeRecursive.PiRlcChallenge.ChallengeWiring
import Nightstream.Implementation.R1CS.Correspondence.FPrimeRecursive.PiRlcChallenge.Sampler.FirstAccepted
import Nightstream.Implementation.R1CS.Correspondence.FPrimeRecursive.PiRlcProjection.YZcolNormalForm
import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive

/-!
Active PiRLC sampler-output handoff to the projection normal form.

Owns: decoding the artifact-checked sampler output columns as fifteen Phi81
challenge rings; proving that this is exactly the challenge view consumed by
the active `y_zcol` projection identities; and transporting the conditional
active row theorem into a field-derived challenge vector.

Does not own: identity of `EmbeddedRowsSatisfied` with the complete Rust relation,
Poseidon2 transcript replay, the post-PiCCS transcript state, projection
arithmetic, encoded costs, or row removal.

Emits constraints: no.

Authority boundary: explicit accepted-row embeddings determine a
field-derived first-accepted vector. It becomes the verifier's PiRLC challenge
vector only after a separate theorem binds all digest fields to the
verifier-owned Poseidon2 transcript. The zero-cost producer/consumer alias
cannot supply that authority.

| Protocol -> phase -> family | Mathematical obligation | Authority class | Remaining boundary |
|---|---|---|---|
| `pi_rlc.challenge.output -> projection_shared.rho_inputs` | all 810 physical columns are identical | direct dataflow | none |
| `pi_rlc.challenge.output.decode` | each 54-column vector decodes as one `RingF` | computed | none |
| `pi_rlc.challenge.field_derived` | accepted rows imply independent first-accepted field chunks | checked | complete Rust-row identity |
| `pi_rlc.challenge.authority` | field-derived vectors equal verifier-derived challenges | security boundary | Poseidon2 replay and post-PiCCS cursor |
-/

namespace Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcChallenge.ProjectionConsumer

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper
open Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc
open Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcChallenge
open Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcChallenge.SamplerLayout
open Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcProjection.YZcolNormalForm
open Nightstream.SuperNeo.Concrete

def semanticChallengeCount : Nat :=
  Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.arity.total

theorem semantic_challenge_count : semanticChallengeCount = sourceCount := by
  rfl

def samplerColumns (index : Fin sourceCount) : List Nat :=
  ChallengeWiring.samplerOutputColumnsFor index

def decodedSamplerChallenges (assignment : Nat → Nat) :
    Fin sourceCount → RingF :=
  fun index => ringOfList (values assignment (samplerColumns index))

/-- Independent field-derived challenge vector. It intentionally depends on
the canonical digest-field columns, but carries no Poseidon2 provenance. -/
def fieldDerivedChallenges
    (assignment : Nat → Nat)
    (canonical : PiRlcChallenge.Transcript.ChunkOrder.CanonicalAssignment
      assignment) : Fin sourceCount → RingF :=
  fun index => ringOfList
    ((Sampler.FirstAccepted.semanticFieldOutput assignment canonical index).map
      residue)

/-- The active sampler-layout and challenge-wiring facades name the same
physical output column for every scalar and coefficient. -/
theorem sampler_output_column_eq_layout
    (index : Fin sourceCount)
    (coefficient : Fin ChallengeWiring.coefficientCount) :
    ChallengeWiring.samplerOutputColumn index coefficient =
      outputColumn index coefficient := by
  simp [ChallengeWiring.samplerOutputColumn,
    ChallengeWiring.Generated.outputBase,
    ChallengeWiring.Generated.rhoStride,
    ChallengeWiring.Generated.coefficientStride,
    outputColumn, tailFirstAllocated, Data.tailFirstAllocatedBase,
    Data.tailFirstAllocatedStride, Data.outputOffset, Data.outputStride,
    FPrimeRecursivePiRlcChallengeWiringData.outputBase,
    FPrimeRecursivePiRlcChallengeWiringData.rhoStride,
    FPrimeRecursivePiRlcChallengeWiringData.coefficientStride,
    FPrimeRecursivePiRlcChallengeSamplerLayoutData.tailFirstAllocatedBase,
    FPrimeRecursivePiRlcChallengeSamplerLayoutData.tailFirstAllocatedStride,
    FPrimeRecursivePiRlcChallengeSamplerLayoutData.outputOffset,
    FPrimeRecursivePiRlcChallengeSamplerLayoutData.outputStride] <;>
    omega

theorem sampler_columns_eq_layout
    (index : Fin sourceCount) :
    samplerColumns index = outputColumnsFor index := by
  unfold samplerColumns ChallengeWiring.samplerOutputColumnsFor outputColumnsFor
  exact congrArg
    (fun values : Fin ChallengeWiring.coefficientCount → Nat =>
      List.ofFn values)
    (funext fun coefficient => sampler_output_column_eq_layout index coefficient)

/-- Exact physical producer/consumer alias for every active challenge. -/
theorem sampler_columns_eq_projection_columns :
    ∀ index : Fin sourceCount,
      samplerColumns index = (limb0Pair index).rhoColumns := by
  set_option maxRecDepth 100000 in
    decide

/-- The projection normal form and the sampler output owner decode the same
physical columns, with no equality row or digest between them. -/
theorem decodedSamplerChallenges_eq_decodedChallenges
    (assignment : Nat → Nat) :
    decodedSamplerChallenges assignment = decodedChallenges assignment := by
  funext index
  unfold decodedSamplerChallenges decodedChallenges lowChallengeRings
  rw [sampler_columns_eq_projection_columns index]

/-- The still-open semantic authority premise for the active sampler output.
It must be discharged from the active source rows and verifier-owned transcript,
not from the projection consumer or a prover-carried digest. -/
def SamplerChallengesBound
    (assignment : Nat → Nat)
    (challenges : Fin sourceCount → RingF) : Prop :=
  decodedSamplerChallenges assignment = challenges

/-- Explicit active row acceptance determines the exact field-derived Phi81
challenge vector. This closes sampler arithmetic and output routing, but not
the transcript-authority or complete-Rust-row boundaries. -/
theorem samplerChallengesBound_fieldDerived
    (prime : EuclidPrime goldilocksP)
    {fullRows : List Row} {assignment : Nat → Nat}
    (canonical : PiRlcChallenge.Transcript.ChunkOrder.CanonicalAssignment
      assignment)
    (one : assignment 0 = 1)
    (accepted : Sampler.Rows.EmbeddedRowsSatisfied fullRows assignment) :
    SamplerChallengesBound assignment
      (fieldDerivedChallenges assignment canonical) := by
  funext index
  have outputs := Sampler.FirstAccepted.accepted_refines
    prime canonical one accepted index
  apply congrArg ringOfList
  unfold values
  rw [sampler_columns_eq_layout index]
  simpa [Sampler.FirstAccepted.productionOutput,
    fieldDerivedChallenges] using congrArg (List.map residue) outputs

/-- Once active sampler refinement supplies the explicit authority premise,
the challenge field of `SemanticColumnsMatch` follows without another check. -/
theorem decodedChallenges_eq_of_samplerBound
    {assignment : Nat → Nat}
    {challenges : Fin sourceCount → RingF}
    (bound : SamplerChallengesBound assignment challenges) :
    decodedChallenges assignment = challenges := by
  rw [← decodedSamplerChallenges_eq_decodedChallenges assignment]
  exact bound

end Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcChallenge.ProjectionConsumer
