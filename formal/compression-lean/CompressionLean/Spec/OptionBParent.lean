import CompressionLean.Spec.OptionBRuns
import NightstreamFPrime.Export.OptionB.FinishRelation

/-! The claim of option B's final verifier in F′'s terms (CW-6.5): SuperNeo v1.2 Definition 21 for
the last fold. A refresh step input whose counter and states are the statement, whose finish
checks hold (`FinishStep.FinishHolds`: the fresh public input is the hash of the prior state and
the refresh fold gives the next slots), and whose two next slots hold under their circuits'
relations (`ParentHolds`). -/

namespace CompressionLean.Spec.OptionB

open NightstreamFPrime NightstreamFPrime.Spec NightstreamFPrime.Circuit NightstreamFPrime.Lifecycle
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Export.OptionB.StepRelation

variable {lw : ℕ} {pf : ringDegree * PaperAlgebra.publicRingColumns ≤ Phi81CarrierLayout.carrierWidth lw}

/-- The finish package that F′'s emitter writes for `program` (`EmitMain.emitFinish`): the finish
relation's pieces, packed from the public columns, with the program's witness words first in its
application region. -/
def finishPackage (program : Stage1.Application.Program) (relation : ProductionKey.LogicalRelation lw pf)
    (next width : ℕ) (batches : List WitnessBatch) (rest : ℕ × ℕ × ℕ × ℕ) :
    Export.OptionB.StepPackage.Package :=
  Export.OptionB.StepPackage.packageOf 2 Export.Generic.CircuitRelation.publicColumns
    (Export.OptionB.FinishRelation.pieces program relation) next width batches (program.witnessWordCount, rest)

/-- A refresh step input with the statement's counter and states, the finish checks, and both next
slots holding under their circuits' relations. -/
def ParentHolds (program : Stage1.Application.Program)
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := lw) (publicFits := pf))
    (circuitKeys : Fin 2 → Lifecycle.OptionB.CircuitKey lw pf) (x : StateWords) : Prop :=
  ∃ env : Env,
    (refreshInput (logicalWidth := lw) (publicFits := pf) program env).iteration = (x 0).val ∧
    (refreshInput (logicalWidth := lw) (publicFits := pf) program env).z0 =
      List.ofFn (fun lane : Fin 4 => x ⟨1 + lane, by omega⟩) ∧
    (refreshInput (logicalWidth := lw) (publicFits := pf) program env).zi =
      List.ofFn (fun lane : Fin 4 => x ⟨5 + lane, by omega⟩) ∧
    Lifecycle.OptionB.FinishStep.FinishHolds Folding.Schedule.productionLightFoldCap ajtai
      (decodedKeys circuitKeys (refreshInterface (logicalWidth := lw) (publicFits := pf) program).verifierKeys env)
      1 (refreshInput program env) (refreshOutput program 1 env) ∧
    ∀ slot, ∃ witness, CE.Holds (PaperAlgebra.semantics ajtai) productionGlobalParams
      (Lifecycle.OptionB.toClaim ((circuitKeys slot).system ajtai)
        ((refreshOutput (logicalWidth := lw) (publicFits := pf) program 1 env).runningNext slot).1) witness

end CompressionLean.Spec.OptionB
