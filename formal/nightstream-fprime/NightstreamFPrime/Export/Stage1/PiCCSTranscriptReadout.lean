import NightstreamFPrime.Export.PermutationOutput.Readout
import NightstreamFPrime.Export.Stage1.PoseidonRetainedBlock
import NightstreamFPrime.Layout.Stage1.PiCCSOrdinarySourceSupportData

/-!
Owns the fixed PiCCS transcript readout used by ordinary arithmetic sources.
The family start and count come from the exact phase layout. Each output is
computed from the final retained S-box values in its own permutation.
-/

namespace NightstreamFPrime.Export.Stage1.PiCCSTranscriptReadout

open NightstreamFPrime.Circuit
open NightstreamFPrime.Spec
open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Export.Package

abbrev Index := Fin PiCCSOrdinarySourceSupport.transcriptInvocationCount

def phaseStart : Nat := Spartan.sourceToSpartan PiCCSInputs.phaseOffset

theorem phaseStart_eq : phaseStart = 14751526 := by
  unfold phaseStart
  rw [PiCCSInputs.phaseOffset_eq]
  rfl

theorem sboxColumn_lt_spartanColumnCount (index : Index) (lane : Fin 8) :
    PermutationOutput.Readout.sboxColumn phaseStart index lane <
      Spartan.spartanColumnCount := by
  apply Nat.lt_of_lt_of_le (PermutationOutput.Readout.sboxColumn_lt_end phaseStart index lane)
  rw [phaseStart_eq, PiCCSOrdinarySourceSupport.transcriptInvocationCount_eq,
    Spartan.spartanColumnCount_eq]
  norm_num

def env (source : Env) : Env :=
  PermutationOutput.Readout.env phaseStart
    PiCCSOrdinarySourceSupport.transcriptInvocationCount source

def sourceColumn (index : Index) (lane : Fin 8) : Nat :=
  PiCCSInputs.phaseOffset + index.val * 592 + 584 + lane.val

theorem sourceColumn_target (index : Index) (lane : Fin 8) :
    Spartan.sourceToSpartan (sourceColumn index lane) =
      PermutationOutput.Readout.outputColumn phaseStart index lane := by
  unfold sourceColumn PermutationOutput.Readout.outputColumn
    PermutationOutput.Readout.witnessStart phaseStart
  have combined : PiCCSInputs.phaseOffset + index.val * 592 + 584 + lane.val =
      PiCCSInputs.phaseOffset + (index.val * 592 + 584 + lane.val) := by omega
  rw [combined, Spartan.sourceToSpartan_add_of_piCcsLocal
    PiCCSInputs.phaseOffset (index.val * 592 + 584 + lane.val) (by
      norm_num [PiCCSInputs.phaseOffset_eq, Spartan.piCcsPhaseOffset])]
  omega

/-- Readout preserves every source outside the exact transcript-output family,
including public columns moved by the physical permutation. -/
theorem env_source_of_notTranscript (source : Env) (column : Nat)
    (bound : column < Spartan.SourceColumnCount)
    (outside : ¬ PiCCSOrdinarySourceSupport.TranscriptOutput column) :
    env source (Spartan.sourceToSpartan column) =
      source (Spartan.sourceToSpartan column) := by
  apply PermutationOutput.Readout.env_of_decode_none
  cases found : PermutationOutput.Readout.decode phaseStart
      PiCCSOrdinarySourceSupport.transcriptInvocationCount
      (Spartan.sourceToSpartan column) with
  | none => rfl
  | some selected =>
      rcases selected with ⟨index, lane⟩
      have address := PermutationOutput.Readout.decode_source phaseStart found
      rw [← sourceColumn_target] at address
      have selectedBound : sourceColumn index lane < Spartan.SourceColumnCount :=
        PiCCSOrdinarySourceSupport.source_lt_sourceColumnCount
          (PiCCSOrdinarySourceSupport.transcript_output_source _ ⟨index, lane, rfl⟩)
      have inverse := congrArg Spartan.spartanToSource address
      rw [Spartan.spartanToSource_sourceToSpartan column bound,
        Spartan.spartanToSource_sourceToSpartan _ selectedBound] at inverse
      exact False.elim (outside ⟨index, lane, Option.some.inj inverse⟩)

private def physicalIndex (index : Index) :
    Fin PoseidonRetainedBlock.basePackage.permutationInvocations.length :=
  ⟨index.val, by
    have bound : index.val < 718 := by
      simpa only [PiCCSOrdinarySourceSupport.transcriptInvocationCount_eq] using index.isLt
    rw [PoseidonRetainedBlock.basePackage_permutationInvocations_length]
    change index.val < 7757
    omega⟩

def invocation (index : Index) : PermutationInvocation :=
  PoseidonRetainedBlock.basePackage.permutationInvocations.get (physicalIndex index)

theorem invocation_witnessStart (index : Index) :
    (invocation index).witnessStart =
      PermutationOutput.Readout.witnessStart phaseStart index := by
  have bound : index.val < 718 := by
    simpa only [PiCCSOrdinarySourceSupport.transcriptInvocationCount_eq] using index.isLt
  let selected : Fin (Data.permutationInvocations ()).length :=
    ⟨index.val, by
      rw [PoseidonRetainedBlock.data_permutationInvocations_length]
      change index.val < 7757
      omega⟩
  have listEq := congrArg
    (fun values : List PermutationInvocation => values[index.val]?)
    PoseidonRetainedBlock.basePackage_permutationInvocations_eq
  change PoseidonRetainedBlock.basePackage.permutationInvocations[index.val]? =
    (Data.permutationInvocations ())[index.val]? at listEq
  have physicalBound : index.val <
      PoseidonRetainedBlock.basePackage.permutationInvocations.length := by
    rw [PoseidonRetainedBlock.basePackage_permutationInvocations_length]
    change index.val < 7757
    omega
  have dataBound : index.val < (Data.permutationInvocations ()).length := by
    rw [PoseidonRetainedBlock.data_permutationInvocations_length]
    change index.val < 7757
    omega
  rw [List.getElem?_eq_getElem physicalBound,
    List.getElem?_eq_getElem dataBound] at listEq
  have same : invocation index = (Data.permutationInvocations ()).get selected :=
    Option.some.inj listEq
  rw [same, PermutationPlan.canonicalInvocation_witnessStart_of_transcript selected bound]
  exact Spartan.sourceToSpartan_add_of_piCcsLocal PiCCSInputs.phaseOffset
    (index.val * 592) (by
      norm_num [PiCCSInputs.phaseOffset_eq, Spartan.piCcsPhaseOffset])

/-- Accepted physical rows force the stored outputs to equal the computed
readout. This derives source agreement from constraints, not from a packet
encoding or a caller-supplied output-coherence assertion. -/
theorem env_eq_of_rows (source : Env)
    (rows : PoseidonRetainedBlock.basePackage.RowsHold source) :
    env source = source := by
  funext column
  cases found : PermutationOutput.Readout.decode phaseStart
      PiCCSOrdinarySourceSupport.transcriptInvocationCount column with
  | none => exact PermutationOutput.Readout.env_of_decode_none _ _ _ _ found
  | some selected =>
      rcases selected with ⟨index, lane⟩
      have address := PermutationOutput.Readout.decode_source phaseStart found
      rw [address]
      change PermutationOutput.Readout.env phaseStart
        PiCCSOrdinarySourceSupport.transcriptInvocationCount source
        (PermutationOutput.Readout.outputColumn phaseStart index lane) = _
      rw [PermutationOutput.Readout.env_outputColumn]
      have output := PermutationOutput.invocation_finalLayer (invocation index) source
        (PoseidonRetainedBlock.invocation_holds source rows (physicalIndex index))
      have selected := congrFun output lane
      rw [invocation_witnessStart] at selected
      exact selected.symm

end NightstreamFPrime.Export.Stage1.PiCCSTranscriptReadout
