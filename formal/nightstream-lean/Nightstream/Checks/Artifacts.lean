import Nightstream.Checks.Common
import Nightstream.Implementation.Encoding.FPrime
import Nightstream.Implementation.R1CS.Correspondence.CanonicalU64.CanonicalU64Sound
import Nightstream.Implementation.R1CS.Correspondence.FPrime.FPrimeCounterSound
import Nightstream.Implementation.R1CS.Correspondence.FPrime.FPrimeEncodingSound
import Nightstream.Implementation.R1CS.Correspondence.FPrime.FPrimeTerminalLinkSound
import Nightstream.Implementation.R1CS.Correspondence.FPrime.FPrimeStateLinkSound
import Nightstream.Implementation.R1CS.Correspondence.FPrime.FPrimeChunkDigestSound
import Nightstream.Implementation.R1CS.Correspondence.FPrime.FPrimeCeContinuitySound
import Nightstream.Implementation.R1CS.Correspondence.FPrimeBase.FPrimeBaseStateSound
import Nightstream.Implementation.R1CS.Correspondence.FPrimeBase.FPrimeBaseProgramSound
import Nightstream.Implementation.R1CS.Correspondence.Poseidon2.Poseidon2PermutationSound
import Nightstream.Implementation.R1CS.Correspondence.Projection.PiRLCProjectionSound
import Nightstream.Implementation.R1CS.Correspondence.U64.U64IncrementSound
import Nightstream.Implementation.R1CS.Correspondence.U64.U64AddSound
import Nightstream.Implementation.R1CS.Ownership.FPrimeRecursive.FPrimeRecursiveManifest
import Nightstream.Implementation.R1CS.Ownership.FPrimeFullHistory.FPrimeFullHistoryManifest
set_option maxRecDepth 16384

namespace Nightstream.Checks.Artifacts

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.CanonicalU64

/-- Executable artifact checks over exported witnesses and manifest metadata.
Each expensive expression remains behind a thunk until its result can be
printed immediately. -/
def probes : List Nightstream.Checks.Probe :=
  [ ⟨"r1cs_artifact_honest_witness",
      fun _ => decide (Satisfies rows (assignmentOf honestWitness)), true⟩
  , ⟨"r1cs_artifact_forged_noncanonical",
      fun _ => decide (Satisfies rows (assignmentOf forgedWitness)), false⟩
  , ⟨"r1cs_u64_increment_honest",
      fun _ => decide (Satisfies U64Increment.rows (assignmentOf U64Increment.honestWitness)),
      true⟩
  , ⟨"r1cs_u64_increment_rejects_wrap",
      fun _ => decide (Satisfies U64Increment.rows (assignmentOf U64Increment.overflowWitness)),
      false⟩
  , ⟨"r1cs_u64_add_honest",
      fun _ => decide (Satisfies U64Add.rows (assignmentOf U64Add.honestWitness)), true⟩
  , ⟨"r1cs_u64_add_rejects_wrap",
      fun _ => decide (Satisfies U64Add.rows (assignmentOf U64Add.overflowWitness)), false⟩
  , ⟨"r1cs_fprime_counter_honest",
      fun _ => decide (Satisfies FPrimeCounter.rows (assignmentOf FPrimeCounter.honestWitness)),
      true⟩
  , ⟨"r1cs_fprime_counter_rejects_source_disconnect",
      fun _ => decide (Satisfies FPrimeCounter.rows
        (assignmentOf FPrimeCounter.wrongSourceWitness)), false⟩
  , ⟨"r1cs_fprime_counter_rejects_wrong_step",
      fun _ => decide (Satisfies FPrimeCounter.rows
        (assignmentOf FPrimeCounter.wrongStepWitness)), false⟩
  , ⟨"r1cs_fprime_counter_rejects_batch_size_forgery",
      fun _ => decide (Satisfies FPrimeCounter.rows
        (assignmentOf FPrimeCounter.wrongRowsWitness)), false⟩
  , ⟨"r1cs_fprime_encoding_exact_row_count",
      fun _ => FPrimeEncoding.rows.length == FPrimeEncoding.rowCount, true⟩
  , ⟨"fprime_encoding_accepts_256_bits",
      fun _ => Nightstream.Implementation.Encoding.FPrime.acceptsEncInstLength
        (List.replicate 256 false), true⟩
  , ⟨"fprime_encoding_rejects_255_bits",
      fun _ => Nightstream.Implementation.Encoding.FPrime.acceptsEncInstLength
        (List.replicate 255 false), false⟩
  , ⟨"r1cs_fprime_terminal_link_exact_row_count",
      fun _ => FPrimeTerminalLink.rows.length == FPrimeTerminalLink.rowCount, true⟩
  , ⟨"r1cs_fprime_state_link_exact_row_count",
      fun _ => FPrimeStateLink.rows.length == FPrimeStateLink.rowCount, true⟩
  , ⟨"r1cs_fprime_base_state_exact_row_count",
      fun _ => FPrimeBaseState.rows.length == FPrimeBaseState.rowCount, true⟩
  , ⟨"r1cs_fprime_base_program_exact_instruction_count",
      fun _ => FPrimeBaseProgram.instructions.length == FPrimeBaseProgram.rowCount, true⟩
  , ⟨"r1cs_fprime_chunk_digest_binding_row_count",
      fun _ => FPrimeChunkDigest.bindingRows.length == 4, true⟩
  , ⟨"r1cs_fprime_ce_continuity_exact_row_count",
      fun _ => FPrimeCeContinuity.continuityRows.length == FPrimeCeContinuity.continuityRowCount,
      true⟩
  , ⟨"r1cs_fprime_diagnostic_direct_ccs3_manifest_top_level_coverage",
      fun _ => FPrimeRecursiveManifest.covers 0 FPrimeRecursiveManifest.totalRows
        FPrimeRecursiveManifest.topLevelFamilies, true⟩
  , ⟨"r1cs_fprime_diagnostic_direct_ccs3_manifest_nifs_coverage",
      fun _ => FPrimeRecursiveManifest.covers FPrimeRecursiveManifest.nifsRowStart
        FPrimeRecursiveManifest.nifsRowEnd
        FPrimeRecursiveManifest.nifsFamilies, true⟩
  , ⟨"r1cs_fprime_full_history_manifest_rows",
      fun _ => FPrimeFullHistoryManifest.totalRows == 4193134, true⟩
  , ⟨"r1cs_fprime_full_history_manifest_columns",
      fun _ => FPrimeFullHistoryManifest.totalColumns == 3582173, true⟩
  , ⟨"r1cs_fprime_full_history_top_level_coverage",
      fun _ => FPrimeRecursiveManifest.covers 0 FPrimeFullHistoryManifest.totalRows
        FPrimeFullHistoryManifest.topLevelFamilies, true⟩
  , ⟨"r1cs_pirlc_projection_exact_row_count",
      fun _ => PiRLCProjection.rows.length == PiRLCProjection.rowCount, true⟩
  , ⟨"r1cs_pirlc_projection_honest_satisfies",
      fun _ => decide (Satisfies PiRLCProjection.rows
        (assignmentOf PiRLCProjection.honestWitness)), true⟩
  , ⟨"r1cs_pirlc_projection_bad_root_satisfies",
      fun _ => decide (Satisfies PiRLCProjection.rows
        (assignmentOf PiRLCProjection.badRootWitness)), true⟩
  , ⟨"r1cs_pirlc_diagnostic_direct_ccs3_identity_census",
      fun _ => FPrimeRecursiveManifest.projectionIdentityCount == 31, true⟩
  , ⟨"r1cs_pirlc_diagnostic_direct_ccs3_pair_census",
      fun _ => FPrimeRecursiveManifest.projectionPairCounts.all (fun count => count == 15), true⟩
  , ⟨"r1cs_poseidon2_permutation_exact_row_count",
      fun _ => Poseidon2Permutation.rows.length == Poseidon2Permutation.rowCount, true⟩
  ]

def run : IO Bool :=
  Nightstream.Checks.runProbes probes

end Nightstream.Checks.Artifacts
