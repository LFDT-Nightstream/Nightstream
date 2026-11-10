import CompressionLean.Spec.BoundedDepthRuns
import NightstreamFPrime.Export.BoundedDepth.FinishRelation
import NightstreamFPrime.Layout.BoundedDepth.LightAssemblerCompleteness
import NightstreamFPrime.Layout.BoundedDepth.RefreshAssemblerCompleteness
import NightstreamFPrime.Layout.BoundedDepth.StepCircuits
import NightstreamFPrime.Layout.PiCCS.v1_2.Leaves.OutputBinding
import NightstreamFPrime.Layout.ProductionRelation.CcsOpening
import NightstreamFPrime.Lifecycle.Nebula.FactorWords
import NightstreamFPrime.Lifecycle.PiCCS.v1_2.Completeness
import NightstreamFPrime.Spec
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.GoldilocksCausal
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.TargetConvention
import NightstreamFPrime.Spec.Folding.PiRLC.PaperForkExtractionWork
import NightstreamFPrime.Spec.Nebula

/-! The terminal of the bounded-depth scheme's final program in F′'s terms (CW-6.5). When the finish package is the
one that the emitter writes for the finish relation's pieces (`StepPackage.packageOf`), with blocks
that pack after the shared words, finish columns in which the program's checks hold give
`FinishStep.FinishHolds` for the refresh step input whose iteration, `z0` and current state are the
statement, in finish columns that agree with the program's below the shared end
(`finishTerminal_holds`, through F′'s `FinishRelation.package_finish`). -/

namespace CompressionLean.Final.BoundedDepth

open CompressionLean.Spec CompressionLean.Spec.BoundedDepth CompressionLean.BoundedDepth
open NightstreamFPrime NightstreamFPrime.Spec NightstreamFPrime.Circuit NightstreamFPrime.Lifecycle
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.HyperNova.Construction2
open NightstreamFPrime.Layout.BoundedDepth
open NightstreamFPrime.Export.Generic
open NightstreamFPrime.Export.BoundedDepth
open NightstreamFPrime.Export.BoundedDepth.StepRelation

variable {logicalWidth : ℕ}
  {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth}

private theorem base_eq : base = 270 := rfl

theorem finishTerminal_holds (program : Stage1.Application.Program)
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (circuitKeys : Fin 2 → Lifecycle.BoundedDepth.CircuitKey logicalWidth publicFits) (next width : ℕ)
    (batches : List NightstreamFPrime.Circuit.WitnessBatch) (application : ℕ × ℕ × ℕ × ℕ × ℕ)
    (packed : CircuitRelation.packable CircuitRelation.publicColumns
      (FinishRelation.pieces program relation) = true)
    (starts : CircuitRelation.startsFrom (RefreshAssembler.sharedEnd base program.witnessWordCount)
      (FinishRelation.pieces program relation) = true)
    {stateKeys : List Replay.Word} {x : StateWords} {env : Env}
    (terminal : FinishTerminal (StepPackage.packageOf 2 CircuitRelation.publicColumns
      (FinishRelation.pieces program relation) next width batches application) stateKeys x env) :
    ∃ completed : Env,
      (∀ column, column < RefreshAssembler.sharedEnd base program.witnessWordCount →
        completed column = env column) ∧
      (refreshInput (logicalWidth := logicalWidth) (publicFits := publicFits) program
        completed).iteration = (x 0).val ∧
      (refreshInput (logicalWidth := logicalWidth) (publicFits := publicFits) program completed).z0 =
        List.ofFn (fun lane : Fin 4 => x ⟨1 + lane, by omega⟩) ∧
      (refreshInput (logicalWidth := logicalWidth) (publicFits := publicFits) program completed).zi =
        List.ofFn (fun lane : Fin 4 => x ⟨5 + lane, by omega⟩) ∧
      Lifecycle.BoundedDepth.FinishStep.FinishHolds Folding.Schedule.productionLightFoldCap ajtai
        (decodedKeys circuitKeys (refreshInterface (logicalWidth := logicalWidth)
          (publicFits := publicFits) program).verifierKeys completed)
        1 (refreshInput program completed) (finishStepOutput program 1 completed) := by
  obtain ⟨positive, -, iteration, lanes, -, rows, blocks⟩ := terminal
  obtain ⟨completed, agree, holds⟩ := FinishRelation.package_finish program relation ajtai circuitKeys next
    width batches application env packed starts rows blocks
  have below : ∀ k, k < base + 25 → completed k = env k := fun k inside =>
    agree k (by rw [RefreshAssembler.sharedEnd_eq]; omega)
  have first : (refreshInput (logicalWidth := logicalWidth) (publicFits := publicFits) program
      completed).iteration = (x 0).val := by
    show (completed base).val = _
    rw [below base (by omega), base_eq]
    exact congrArg Fin.val iteration
  refine ⟨completed, agree, first, ?_, ?_, holds (by rw [first]; exact fun zero => positive (Fin.ext zero))⟩
  · show List.ofFn (fun lane : Fin 4 => completed (base + 13 + lane)) = _
    congr 1
    funext lane
    rw [below _ (by omega), base_eq]
    exact (lanes lane).1
  · show List.ofFn (fun lane : Fin 4 => completed (base + 13 + 4 + lane)) = _
    congr 1
    funext lane
    rw [below _ (by omega), base_eq]
    exact (lanes lane).2

end CompressionLean.Final.BoundedDepth
