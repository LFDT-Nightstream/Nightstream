import Mathlib.Tactic
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPreludeXOut
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingVariableHashRecipeCertificate

/-!
Facade and structural certificate for the two exact Rust Prelude XOut hashes.

Owns fixed profile identity, exact source-row and column geometry, complete
32-field input order, nine-round sponge structure, and four-lane outputs.
It does not evaluate either row program or give authority to its input values.

Assurance tier: artifact-checked for
`FPRIME-STREAMING-PRELUDE-XOUT-ROWS-V1`, Nightstream b2/k16.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPreludeXOut

open Nightstream.Implementation.R1CS.Poseidon2Sponge
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplay.Artifact
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPreludeXOut.Artifact
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingVariableHashRecipe.Artifact
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingVariableHashRecipeCertificate
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPreludeXOut

def artifact : RawArtifact := rawArtifact

structure Valid (candidate : RawArtifact) : Prop where
  schemaVersion : candidate.schemaVersion = 3
  profileId : candidate.profileId = "nightstream-goldilocks-b2-k16"
  branchScope : candidate.branchScope = "base"
  lifecycleScope : candidate.lifecycleScope = "prelude"
  stagePath : candidate.stagePath = "nebula.streaming.prelude.state_x_out"
  sourceRowCount : candidate.sourceRowCount = 677047
  sourceColumnCount : candidate.sourceColumnCount = 677038
  normalizedColumnCount : candidate.normalizedColumnCount = 677038
  publicSpans : candidate.publicSpans =
    [{ sourceStart := 671002, normalizedStart := 1, length := 64 },
      { sourceStart := 671068, normalizedStart := 65, length := 64 },
      { sourceStart := 671134, normalizedStart := 129, length := 64 },
      { sourceStart := 671200, normalizedStart := 193, length := 64 },
      { sourceStart := 676774, normalizedStart := 257, length := 64 },
      { sourceStart := 676840, normalizedStart := 321, length := 64 },
      { sourceStart := 676906, normalizedStart := 385, length := 64 },
      { sourceStart := 676972, normalizedStart := 449, length := 64 },
      { sourceStart := 671267, normalizedStart := 513, length := 64 },
      { sourceStart := 665429, normalizedStart := 577, length := 64 }]
  afterRows : candidate.afterXOut.sourceRows = { start := 665550, stop := 670984 }
  beforeRows : candidate.beforeXOut.sourceRows = { start := 671337, stop := 676771 }
  afterInputs : candidate.afterXOut.preimageColumns.map (fun binding => binding.source) =
    candidate.afterXOut.recipe.inputColumns
  beforeInputs : candidate.beforeXOut.preimageColumns.map (fun binding => binding.source) =
    candidate.beforeXOut.recipe.inputColumns
  afterOutputs : candidate.afterXOut.digestColumns.map (fun binding => binding.source) =
    candidate.afterXOut.recipe.outputColumns
  beforeOutputs : candidate.beforeXOut.digestColumns.map (fun binding => binding.source) =
    candidate.beforeXOut.recipe.outputColumns
  afterCanonicalFields :
    candidate.afterXOut.canonicalCalls.map CanonicalCall.fieldColumn =
      candidate.afterXOut.recipe.outputColumns
  beforeCanonicalFields :
    candidate.beforeXOut.canonicalCalls.map CanonicalCall.fieldColumn =
      candidate.beforeXOut.recipe.outputColumns
  afterNormalizedBitBase : candidate.afterXOut.normalizedBitBase = 1
  beforeNormalizedBitBase : candidate.beforeXOut.normalizedBitBase = 257

theorem artifact_valid : Valid artifact where
  schemaVersion := rfl
  profileId := rfl
  branchScope := rfl
  lifecycleScope := rfl
  stagePath := rfl
  sourceRowCount := rfl
  sourceColumnCount := rfl
  normalizedColumnCount := rfl
  publicSpans := rfl
  afterRows := rfl
  beforeRows := rfl
  afterInputs := rfl
  beforeInputs := rfl
  afterOutputs := rfl
  beforeOutputs := rfl
  afterCanonicalFields := rfl
  beforeCanonicalFields := rfl
  afterNormalizedBitBase := rfl
  beforeNormalizedBitBase := rfl

private theorem input_length (block : HashBlock)
    (exact : block = artifact.afterXOut ∨ block = artifact.beforeXOut) :
    block.recipe.inputColumns.length = 32 := by
  rcases exact with rfl | rfl <;> rfl

private theorem absorb_rounds (block : HashBlock)
    (exact : block = artifact.afterXOut ∨ block = artifact.beforeXOut) :
    block.recipe.absorbRounds = 8 := by
  rw [VariableHashRecipe.absorbRounds, input_length block exact]
  norm_num [rate]

private theorem output_exact (block : HashBlock)
    (exact : block = artifact.afterXOut ∨ block = artifact.beforeXOut) :
    block.recipe.outputColumns =
      (block.recipe.callOutputColumns block.recipe.absorbRounds).take 4 := by
  rcases exact with rfl | rfl <;> rfl

private theorem trace_ownedValid (block : HashBlock)
    (exact : block = artifact.afterXOut ∨ block = artifact.beforeXOut) :
    block.recipe.trace.OwnedValid := by
  exact ownedValid block.recipe (by
      rw [absorb_rounds block exact]
      omega) (by
      rw [input_length block exact, absorb_rounds block exact])
    (output_exact block exact)

theorem after_trace_ownedValid :
    artifact.afterXOut.recipe.trace.OwnedValid :=
  trace_ownedValid artifact.afterXOut (Or.inl rfl)

theorem before_trace_ownedValid :
    artifact.beforeXOut.recipe.trace.OwnedValid :=
  trace_ownedValid artifact.beforeXOut (Or.inr rfl)

theorem after_inputLength_exact :
    artifact.afterXOut.recipe.inputColumns.length = 32 :=
  input_length artifact.afterXOut (Or.inl rfl)

theorem before_inputLength_exact :
    artifact.beforeXOut.recipe.inputColumns.length = 32 :=
  input_length artifact.beforeXOut (Or.inr rfl)

private theorem valueSchedules_exact
    (block : HashBlock)
    (exact : block = artifact.afterXOut ∨ block = artifact.beforeXOut) :
    valueSchedules block.recipe.trace.rounds =
      List.replicate 8 (.absorb 4) ++ [.pad] := by
  rcases exact with rfl | rfl <;> rfl

theorem after_valueSchedules_exact :
    valueSchedules artifact.afterXOut.recipe.trace.rounds =
      List.replicate 8 (.absorb 4) ++ [.pad] :=
  valueSchedules_exact artifact.afterXOut (Or.inl rfl)

theorem before_valueSchedules_exact :
    valueSchedules artifact.beforeXOut.recipe.trace.rounds =
      List.replicate 8 (.absorb 4) ++ [.pad] :=
  valueSchedules_exact artifact.beforeXOut (Or.inr rfl)

def publicCall (block : HashBlock) (lane : Fin 4) : CanonicalCall :=
  block.canonicalCalls.getD lane.val default

def normalizedPublicBitColumn
    (block : HashBlock) (lane : Fin 4) (bit : Nat) : Nat :=
  block.normalizedBitBase + 64 * lane.val + bit

private theorem publicCall_member
    (block : HashBlock)
    (exact : block = artifact.afterXOut ∨ block = artifact.beforeXOut)
    (lane : Fin 4) :
    publicCall block lane ∈ block.canonicalCalls := by
  rcases exact with rfl | rfl <;> fin_cases lane <;>
    simp [publicCall, artifact,
      Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPreludeXOut.rawArtifact]

private theorem publicCall_fieldColumn
    (block : HashBlock)
    (exact : block = artifact.afterXOut ∨ block = artifact.beforeXOut)
    (lane : Fin 4) :
    (publicCall block lane).fieldColumn =
      block.recipe.outputColumns.getD lane.val 0 := by
  rcases exact with rfl | rfl <;> fin_cases lane <;> rfl

theorem after_publicCall_member (lane : Fin 4) :
    publicCall artifact.afterXOut lane ∈
      artifact.afterXOut.canonicalCalls :=
  publicCall_member artifact.afterXOut (Or.inl rfl) lane

theorem before_publicCall_member (lane : Fin 4) :
    publicCall artifact.beforeXOut lane ∈
      artifact.beforeXOut.canonicalCalls :=
  publicCall_member artifact.beforeXOut (Or.inr rfl) lane

theorem after_publicCall_fieldColumn (lane : Fin 4) :
    (publicCall artifact.afterXOut lane).fieldColumn =
      artifact.afterXOut.recipe.outputColumns.getD lane.val 0 :=
  publicCall_fieldColumn artifact.afterXOut (Or.inl rfl) lane

theorem before_publicCall_fieldColumn (lane : Fin 4) :
    (publicCall artifact.beforeXOut lane).fieldColumn =
      artifact.beforeXOut.recipe.outputColumns.getD lane.val 0 :=
  publicCall_fieldColumn artifact.beforeXOut (Or.inr rfl) lane

end Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPreludeXOut
