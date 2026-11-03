import NightstreamFPrime.Export.OptionB.StepRelation
import NightstreamFPrime.Layout.OptionB.FinishAssembler

/-! The finish circuit's exported relation (CW-6.3). The finish package holds the finish circuit
(`FinishStep.main`) at the refresh step's source layout, with the refresh step's shared words, and
the decode constraints of the refresh step's output digest words (`constraints`). A satisfying
assignment of its logical relation gives, in some environment, `FinishStep.FinishHolds` for the
refresh step input and output that the words carry, once the input's iteration is positive
(`finish_holds`). The final program reads the finish package's rows. -/

namespace NightstreamFPrime.Export.OptionB.FinishRelation

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
open NightstreamFPrime.Spec.HyperNova.Construction2
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.OptionB
open NightstreamFPrime.Layout.OptionB
open NightstreamFPrime.Export.Generic
open NightstreamFPrime.Export.OptionB.StepRelation

variable {logicalWidth : Nat}
  {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}
  (program : Stage1.Application.Program)

/-- The finish circuit's constraints at the refresh step's shared end, then the decode
constraints of the refresh step's output digest words. -/
def constraints (relation : ProductionKey.LogicalRelation logicalWidth publicFits) : List Expr :=
  flatConstraints (Circuit.ops (FinishStep.main program relation 1
      (RefreshAssembler.interface (logicalWidth := logicalWidth) (publicFits := publicFits) base
        program.witnessWordCount))
    (RefreshAssembler.sharedEnd base program.witnessWordCount)) ++
  LogicalPlan.decodeConstraints (refreshDigestColumn program)

def pieces (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    List Pieces.Piece :=
  Pieces.split (PilotData.canonicalConstraints ()) (constraints program relation)

/-- A satisfying assignment of the finish circuit's logical relation gives, in some environment,
`FinishHolds` for the refresh step input and output that its words carry, when the input's
iteration is positive. -/
theorem finish_holds (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (circuitKeys : Fin 2 → CircuitKey logicalWidth publicFits) (firstFresh width : Nat)
    (fits : CircuitRelation.Fits CircuitRelation.publicColumns (pieces program relation)
      firstFresh width)
    (layout : LogicalPlan.Layout width logicalWidth)
    (assignment : Assignment F logicalWidth) (digest : Digest)
    (encoded : LogicalPlan.publicInput (publicFits := publicFits)
      (LogicalPlan.sourceEnv layout assignment) = encHash digest)
    (rowsZero : (CircuitRelation.plan _ _ firstFresh width fits layout).RowsZero assignment) :
    ∃ env : Env, (refreshInput (logicalWidth := logicalWidth) (publicFits := publicFits) program
        env).iteration ≠ 0 →
      FinishStep.FinishHolds Folding.Schedule.productionLightFoldCap ajtai
        (decodedKeys circuitKeys (refreshInterface (logicalWidth := logicalWidth)
          (publicFits := publicFits) program).verifierKeys env)
        1 (refreshInput program env) (refreshOutput program 1 env) := by
  obtain ⟨env, holds, _⟩ := CircuitRelation.rowsZero_implies_constraints
    CircuitRelation.publicColumns (constraints program relation) firstFresh width fits layout
    assignment (LogicalPlan.encoded_one layout assignment digest fits.rows.positive encoded)
    rowsZero
  exact ⟨env, fun active => FinishAssembler.rows_implies_finish program relation base ajtai _ 1 env
    (holdsFlat_implies_holds _ _ fun expression member =>
      holds expression (List.mem_append_left _ member))
    _ _ _ (refreshRepresents program ajtai circuitKeys 1 env) active⟩

end NightstreamFPrime.Export.OptionB.FinishRelation
