import Nightstream.Implementation.Nebula.Production.Carrier.StreamingFPrimeProgram
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingFPrimeProgram

/-!
Contract: exact comparison of the Rust-emitted streaming program with the
verifier-owned Lean program.

Assurance tier: Rust-conformant for property
`FPRIME-STREAMING-PROGRAM`.

Owns exact phase codes, phase order, repeated-phase indices, chunk geometry,
and the 400-step production count.

Does not own phase-local constraints, relation rows or columns, recursive
proof integration, same-assignment conformance, or security reduction.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 10000

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingProgramArtifact

open Nightstream.Implementation.Nebula.ProductionStreamingFPrimeProgram
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingProgram.Artifact
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingFPrimeProgram

def expectedWorkItems : List (Nat × Nat) :=
  (program productionConfig).map fun item =>
    (item.phase.code.val, item.index)

theorem profile_exact :
    profileId = "nebula-fprime-streaming-program" := by
  decide

theorem artifact_geometry_exact :
    rawProgram.stateChunkFields = 1024 /\
      rawProgram.priorStateFrameFields = 83874 /\
      rawProgram.priorStateChunks = 82 /\
      rawProgram.claimFrameFields = 88023 /\
      rawProgram.claimChunkFields = 1024 /\
      rawProgram.claimChunks = 86 /\
      rawProgram.piCcsRounds = 26 /\
      rawProgram.piRlcFamilies = 110 /\
      rawProgram.successorPrefixFrameFields = 83756 /\
      rawProgram.successorPrefixChunks = 82 /\
      rawProgram.workItemCount = 400 := by
  decide

theorem artifact_valid : ProgramValid rawProgram := by
  decide

/-- Expansion of the compact Rust runs is exactly the Lean phase program.
This compares every phase code and every repeated-phase index. -/
theorem rust_program_exact :
    rawProgram.expanded = expectedWorkItems := by
  decide

theorem rust_program_length_exact :
    rawProgram.expanded.length = 400 := by
  calc
    rawProgram.expanded.length = expectedWorkItems.length :=
      congrArg List.length rust_program_exact
    _ = (program productionConfig).length := by
      simp [expectedWorkItems]
    _ = 400 := production_program_length

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingProgramArtifact
