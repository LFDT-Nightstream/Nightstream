import NightstreamFPrime.Export.PermutationOutput.Readout
import NightstreamFPrime.Export.Stage1.PoseidonRetainedBlock
import NightstreamFPrime.Layout.Stage1.PiCCSOrdinarySourceSupportData
import NightstreamFPrime.Layout.Stage1.SpartanValues

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

/-- Spartan column of the first PiCCS transcript permutation. -/
def transcriptStart : Nat := Spartan.sourceToSpartan PiCCSStarts.statementWitnessStart

theorem transcriptStart_eq : transcriptStart = 5156948 := by
  unfold transcriptStart
  rw [PiCCSStarts.statementWitnessStart_eq]
  rfl

theorem sboxColumn_lt_spartanColumnCount (index : Index) (lane : Fin 16) :
    PermutationOutput.Readout.sboxColumn transcriptStart index lane <
      Spartan.spartanColumnCount := by
  apply Nat.lt_of_lt_of_le (PermutationOutput.Readout.sboxColumn_lt_end transcriptStart index lane)
  rw [transcriptStart_eq, PiCCSOrdinarySourceSupport.transcriptInvocationCount_eq,
    Spartan.spartanColumnCount_eq]
  norm_num

def env (source : Env) : Env :=
  PermutationOutput.Readout.env transcriptStart
    PiCCSOrdinarySourceSupport.transcriptInvocationCount source

def sourceColumn (index : Index) (lane : Fin 16) : Nat :=
  PiCCSStarts.statementWitnessStart + index.val * 1096 + 1080 + lane.val

theorem sourceColumn_target (index : Index) (lane : Fin 16) :
    Spartan.sourceToSpartan (sourceColumn index lane) =
      PermutationOutput.Readout.outputColumn transcriptStart index lane := by
  unfold sourceColumn PermutationOutput.Readout.outputColumn
    PermutationOutput.Readout.witnessStart transcriptStart
  have combined : PiCCSStarts.statementWitnessStart + index.val * 1096 + 1080 + lane.val =
      PiCCSStarts.statementWitnessStart + (index.val * 1096 + 1080 + lane.val) := by omega
  rw [combined, Spartan.sourceToSpartan_add_of_piCcsLocal
    PiCCSStarts.statementWitnessStart (index.val * 1096 + 1080 + lane.val) (by
      norm_num [PiCCSStarts.statementWitnessStart_eq, Spartan.piCcsPhaseOffset])]
  omega

/-- Readout preserves every source outside the exact transcript-output family,
including public columns moved by the physical permutation. -/
theorem env_source_of_notTranscript (source : Env) (column : Nat)
    (bound : column < Spartan.SourceColumnCount)
    (outside : ¬ PiCCSOrdinarySourceSupport.TranscriptOutput column) :
    env source (Spartan.sourceToSpartan column) =
      source (Spartan.sourceToSpartan column) := by
  apply PermutationOutput.Readout.env_of_decode_none
  cases found : PermutationOutput.Readout.decode transcriptStart
      PiCCSOrdinarySourceSupport.transcriptInvocationCount
      (Spartan.sourceToSpartan column) with
  | none => rfl
  | some selected =>
      rcases selected with ⟨index, lane⟩
      have address := PermutationOutput.Readout.decode_source transcriptStart found
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
    have bound : index.val < 355 := by
      simpa only [PiCCSOrdinarySourceSupport.transcriptInvocationCount_eq] using index.isLt
    rw [PoseidonRetainedBlock.basePackage_permutationInvocations_length]
    rw [PoseidonRetainedBlock.laterInvocationCount_eq]
    omega⟩

def invocation (index : Index) : PermutationInvocation :=
  PoseidonRetainedBlock.basePackage.permutationInvocations.get (physicalIndex index)

theorem invocation_witnessStart (index : Index) :
    (invocation index).witnessStart =
      PermutationOutput.Readout.witnessStart transcriptStart index := by
  have bound : index.val < 355 := by
    simpa only [PiCCSOrdinarySourceSupport.transcriptInvocationCount_eq] using index.isLt
  let selected : Fin (Data.permutationInvocations ()).length :=
    ⟨index.val, by
      rw [PoseidonRetainedBlock.data_permutationInvocations_length]
      rw [PoseidonRetainedBlock.laterInvocationCount_eq]
      omega⟩
  have listEq := congrArg
    (fun values : List PermutationInvocation => values[index.val]?)
    PoseidonRetainedBlock.basePackage_permutationInvocations_eq
  change PoseidonRetainedBlock.basePackage.permutationInvocations[index.val]? =
    (Data.permutationInvocations ())[index.val]? at listEq
  have physicalBound : index.val <
      PoseidonRetainedBlock.basePackage.permutationInvocations.length := by
    rw [PoseidonRetainedBlock.basePackage_permutationInvocations_length]
    rw [PoseidonRetainedBlock.laterInvocationCount_eq]
    omega
  have dataBound : index.val < (Data.permutationInvocations ()).length := by
    rw [PoseidonRetainedBlock.data_permutationInvocations_length]
    rw [PoseidonRetainedBlock.laterInvocationCount_eq]
    omega
  rw [List.getElem?_eq_getElem physicalBound,
    List.getElem?_eq_getElem dataBound] at listEq
  have same : invocation index = (Data.permutationInvocations ()).get selected :=
    Option.some.inj listEq
  rw [same, PermutationPlan.canonicalInvocation_witnessStart_of_transcript selected bound]
  exact Spartan.sourceToSpartan_add_of_piCcsLocal PiCCSStarts.statementWitnessStart
    (index.val * 1096) (by
      norm_num [PiCCSStarts.statementWitnessStart_eq, Spartan.piCcsPhaseOffset])

/-- The actual transcript permutation rows force their stored outputs to
equal the computed readout. Other package rows are not required. -/
theorem env_eq_of_invocations (source : Env)
    (rows : ∀ index, PermutationInvocationHolds (PilotData.circuitPackage ())
      (invocation index) source) :
    env source = source := by
  funext column
  cases found : PermutationOutput.Readout.decode transcriptStart
      PiCCSOrdinarySourceSupport.transcriptInvocationCount column with
  | none => exact PermutationOutput.Readout.env_of_decode_none _ _ _ _ found
  | some selected =>
      rcases selected with ⟨index, lane⟩
      have address := PermutationOutput.Readout.decode_source transcriptStart found
      rw [address]
      change PermutationOutput.Readout.env transcriptStart
        PiCCSOrdinarySourceSupport.transcriptInvocationCount source
        (PermutationOutput.Readout.outputColumn transcriptStart index lane) = _
      rw [PermutationOutput.Readout.env_outputColumn]
      have output := PermutationOutput.invocation_finalLayer (invocation index) source
        (rows index)
      have selected := congrFun output lane
      rw [invocation_witnessStart] at selected
      exact selected.symm

/-- Accepted package rows supply the exact transcript invocation rows. -/
theorem env_eq_of_rows (source : Env)
    (rows : PoseidonRetainedBlock.basePackage.RowsHold source) :
    env source = source := by
  apply env_eq_of_invocations source
  intro index
  exact PoseidonRetainedBlock.invocation_holds source rows (physicalIndex index)

end NightstreamFPrime.Export.Stage1.PiCCSTranscriptReadout
