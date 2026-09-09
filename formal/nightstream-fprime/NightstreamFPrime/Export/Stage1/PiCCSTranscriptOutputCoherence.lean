import NightstreamFPrime.Export.PermutationOutput
import NightstreamFPrime.Export.Stage1.PerApplicationCanonicalEncodes
import NightstreamFPrime.Export.Stage1.PerApplicationCanonicalPreservation
import NightstreamFPrime.Export.Stage1.PiCCSTranscriptOutputForms

/-!
Owns the physical final-layer equality for each actual PiCCS permutation and
its canonical retained output form. Source coherence follows from accepted
physical rows; it is not an assumption on arbitrary raw packets.
-/

namespace NightstreamFPrime.Export.Stage1.PiCCSTranscriptOutputCoherence

open NightstreamFPrime.Circuit
open NightstreamFPrime.Spec
open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
open NightstreamFPrime.Export.Package
open PiCCSPoseidonPreservation

abbrev Program := Lifecycle.Stage1.Application.Program
abbrev InvocationIndex := Fin PiCCSPoseidonPlan.invocationCount

/-- Accepted canonical physical package rows supply the final-layer equality
for every indexed PiCCS invocation, without enumerating the invocation list. -/
theorem packageRows_imply_finalLayer (env : Env)
    (rows : PoseidonRetainedBlock.basePackage.RowsHold env)
    (index : InvocationIndex) :
    (fun lane : Fin 8 => env ((physicalInvocation index).witnessStart + 584 + lane.val)) =
      Layer.externalF (fun lane => env
        ((physicalInvocation index).witnessStart +
          (PoseidonRetainedSlots.localOutput (PoseidonRetainedSlots.finalRow lane)).val)) := by
  apply PermutationOutput.invocation_finalLayer (physicalInvocation index) env
  exact PoseidonRetainedBlock.invocation_holds env rows
    ⟨index.val, by
      rw [PoseidonRetainedBlock.basePackage_permutationInvocations_length]
      exact (laterIndex index).isLt⟩

/-- The canonical retained output form equals the physical output constrained
by the same PiCCS invocation. Only accepted package rows are required; the
owned coordinate encodings are derived from the canonical assignment. -/
theorem canonicalOutput_of_packageRows {application : Program}
    (raw : PerApplicationCanonicalAssignment.RawValues application)
    (rows : (PerApplicationPackage.package application).RowsHold
      (SourceCompiler.sourceEnv raw.base)) (index : InvocationIndex) :
    PiCCSPoseidonPreservation.outputValue
        (PerApplicationCanonicalEncodes.poseidonGeometry application)
        raw.assignment index =
      fun lane : Fin 8 =>
        PerApplicationPackage.baseEnv application (SourceCompiler.sourceEnv raw.base)
          ((physicalInvocation index).witnessStart + 584 + lane.val) := by
  let geometry := PerApplicationCanonicalEncodes.poseidonGeometry application
  have sboxes := PiCCSPoseidonPlan.retainedBlock_encodesAt geometry
    raw.assignment raw.retainedSource
    (PiRLCRetainedGeometry.laterPoseidonFits
      (PiCCSPoseidonPlan.prefixGeometry geometry))
    (PerApplicationCanonicalEncodes.retainedEncodes raw).laterPoseidon
  change SparseLayer.evalState raw.assignment
      (PiCCSPoseidonPlan.outputState geometry index) = _
  rw [PiCCSPoseidonPreservation.outputState_baseEnv geometry raw.assignment
    raw.base raw.groupValue raw.products sboxes index]
  exact (packageRows_imply_finalLayer _
    (PerApplicationCanonicalPreservation.packageRows_imply_validatedPrefix
      application (SourceCompiler.sourceEnv raw.base) rows) index).symm

/-- The shared transcript reader uses that exact row-derived output word. -/
theorem transcriptForm_of_packageRows {application : Program}
    (raw : PerApplicationCanonicalAssignment.RawValues application)
    (rows : (PerApplicationPackage.package application).RowsHold
      (SourceCompiler.sourceEnv raw.base))
    (index : PiCCSTranscriptOutputForms.TranscriptIndex) (lane : Fin 8) :
    (PiCCSTranscriptOutputForms.transcriptForm
        (PerApplicationCanonicalEncodes.poseidonGeometry application) index lane).eval
        raw.assignment =
      PerApplicationPackage.baseEnv application (SourceCompiler.sourceEnv raw.base)
        ((physicalInvocation (PiCCSTranscriptOutputForms.invocation index)).witnessStart +
          584 + lane.val) := by
  exact congrFun (canonicalOutput_of_packageRows raw rows
    (PiCCSTranscriptOutputForms.invocation index)) lane

/-- The running point reader is a lane of the same accepted transcript output. -/
theorem pointForm_of_packageRows {application : Program}
    (raw : PerApplicationCanonicalAssignment.RawValues application)
    (rows : (PerApplicationPackage.package application).RowsHold
      (SourceCompiler.sourceEnv raw.base))
    (coordinate : Fin Lifecycle.productionShape.cubeVariables) (component : Fin 2) :
    (PiCCSTranscriptOutputForms.pointForm
        (PerApplicationCanonicalEncodes.poseidonGeometry application)
        coordinate component).eval raw.assignment =
      PerApplicationPackage.baseEnv application (SourceCompiler.sourceEnv raw.base)
        ((physicalInvocation (PiCCSTranscriptOutputForms.invocation
          (PiCCSTranscriptOutputForms.pointInvocation coordinate component))).witnessStart +
          584) := by
  simpa only [Nat.add_zero] using transcriptForm_of_packageRows raw rows
    (PiCCSTranscriptOutputForms.pointInvocation coordinate component) (0 : Fin 8)

/-- The selected physical invocation and the ordinary transcript source
grid name the same output column. This is an address equality only. -/
theorem physicalTranscript_source
    (index : PiCCSTranscriptOutputForms.TranscriptIndex) (lane : Fin 8) :
    (physicalInvocation (PiCCSTranscriptOutputForms.invocation index)).witnessStart +
        584 + lane.val =
      Layout.Stage1.Spartan.sourceToSpartan
        (PiCCSTranscriptOutputForms.transcriptSource index lane) := by
  have bound : index.val < 718 := by
    simpa only [Layout.Stage1.PiCCSOrdinarySourceSupport.transcriptInvocationCount_eq]
      using index.isLt
  let selected : Fin (Data.permutationInvocations ()).length :=
    ⟨index.val, by
      rw [PoseidonRetainedBlock.data_permutationInvocations_length]
      change index.val < 7757
      omega⟩
  have same : physicalInvocation (PiCCSTranscriptOutputForms.invocation index) =
      (Data.permutationInvocations ()).get selected := by
    have listEq := congrArg
      (fun invocations : List PermutationInvocation => invocations[index.val]?)
      PoseidonRetainedBlock.basePackage_permutationInvocations_eq
    change PoseidonRetainedBlock.basePackage.permutationInvocations[index.val]? =
      (Data.permutationInvocations ())[index.val]? at listEq
    have leftBound : index.val <
        PoseidonRetainedBlock.basePackage.permutationInvocations.length := by
      rw [PoseidonRetainedBlock.basePackage_permutationInvocations_length]
      change index.val < 7757
      omega
    rw [List.getElem?_eq_getElem leftBound,
      List.getElem?_eq_getElem selected.isLt] at listEq
    exact Option.some.inj listEq
  rw [same, PermutationPlan.canonicalInvocation_witnessStart_of_transcript
    selected bound]
  change Layout.Stage1.Spartan.sourceToSpartan
      (Layout.Stage1.PiCCSInputs.phaseOffset + index.val * 592) + 584 + lane.val = _
  have shifted := Layout.Stage1.Spartan.sourceToSpartan_add_of_piCcsLocal
    (Layout.Stage1.PiCCSInputs.phaseOffset + index.val * 592) (584 + lane.val) (by
      norm_num [Layout.Stage1.PiCCSInputs.phaseOffset_eq,
        Layout.Stage1.Spartan.piCcsPhaseOffset])
  rw [Nat.add_assoc, ← shifted]
  apply congrArg Layout.Stage1.Spartan.sourceToSpartan
  unfold PiCCSTranscriptOutputForms.transcriptSource
    PiCCSTranscriptOutputForms.transcriptSourceStart
  omega

/-- Accepted physical rows identify the canonical transcript form with the
exact source column consumed by the ordinary PiCCS layout. -/
theorem transcriptForm_source_of_packageRows {application : Program}
    (raw : PerApplicationCanonicalAssignment.RawValues application)
    (rows : (PerApplicationPackage.package application).RowsHold
      (SourceCompiler.sourceEnv raw.base))
    (index : PiCCSTranscriptOutputForms.TranscriptIndex) (lane : Fin 8) :
    (PiCCSTranscriptOutputForms.transcriptForm
        (PerApplicationCanonicalEncodes.poseidonGeometry application) index lane).eval
        raw.assignment =
      PerApplicationPackage.baseEnv application (SourceCompiler.sourceEnv raw.base)
        (Layout.Stage1.Spartan.sourceToSpartan
          (PiCCSTranscriptOutputForms.transcriptSource index lane)) := by
  rw [transcriptForm_of_packageRows raw rows, physicalTranscript_source]

/-- The running point source and ordinary transcript source are the same
row-derived output. There is no separate copied-output assumption. -/
theorem pointForm_source_of_packageRows {application : Program}
    (raw : PerApplicationCanonicalAssignment.RawValues application)
    (rows : (PerApplicationPackage.package application).RowsHold
      (SourceCompiler.sourceEnv raw.base))
    (coordinate : Fin Lifecycle.productionShape.cubeVariables) (component : Fin 2) :
    (PiCCSTranscriptOutputForms.pointForm
        (PerApplicationCanonicalEncodes.poseidonGeometry application)
        coordinate component).eval raw.assignment =
      PerApplicationPackage.baseEnv application (SourceCompiler.sourceEnv raw.base)
        (Layout.Stage1.Spartan.sourceToSpartan
          (PiCCSTranscriptOutputForms.pointSource coordinate component)) := by
  rw [PiCCSTranscriptOutputForms.pointSource_eq_transcriptSource]
  exact transcriptForm_source_of_packageRows raw rows
    (PiCCSTranscriptOutputForms.pointInvocation coordinate component) 0

end NightstreamFPrime.Export.Stage1.PiCCSTranscriptOutputCoherence
