import NightstreamFPrime.Export.BoundedDepth.StepRelation
import NightstreamFPrime.Layout.BoundedDepth.FinishAssembler
import NightstreamFPrime.Export.BoundedDepth.StepPackage
import NightstreamFPrime.Export.Generic.PieceRows

/-! The finish circuit's exported relation (CW-6.3). The finish package holds the finish circuit
(`FinishStep.main`) at the refresh step's source layout, with the refresh step's shared words, and
the decode constraints of the refresh step's output digest words (`constraints`). A satisfying
assignment of its logical relation gives, in some environment, `FinishStep.FinishHolds` for the
refresh step input and output that the words carry, once the input's iteration is positive
(`finish_holds`). The final program checks the rows of the pieces other than the Poseidon2 blocks
and each block as a permutation instead of its S-box and binding rows; those checks give the same
in an environment that agrees with the program's columns below the shared end, when no block
starts below it (`checks_finish`), stated on the package that the emitter writes
(`package_finish`). -/

namespace NightstreamFPrime.Export.BoundedDepth.FinishRelation

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
open NightstreamFPrime.Spec.HyperNova.Construction2
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.BoundedDepth
open NightstreamFPrime.Layout.BoundedDepth
open NightstreamFPrime.Export.Generic
open NightstreamFPrime.Export.BoundedDepth.StepRelation

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

/-- The finish constraints give `FinishHolds` for an active proof. -/
theorem constraints_finish (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (circuitKeys : Fin 2 → CircuitKey logicalWidth publicFits) (env : Env)
    (holds : ConstraintsHold env (constraints program relation))
    (active : (refreshInput (logicalWidth := logicalWidth) (publicFits := publicFits) program
      env).iteration ≠ 0) :
    FinishStep.FinishHolds Folding.Schedule.productionLightFoldCap ajtai
      (decodedKeys circuitKeys (refreshInterface (logicalWidth := logicalWidth)
        (publicFits := publicFits) program).verifierKeys env)
      1 (refreshInput program env) (finishStepOutput program 1 env) :=
  FinishAssembler.rows_implies_finish program relation base ajtai _ 1 env
    (holdsFlat_implies_holds _ _ fun expression member =>
      holds expression (List.mem_append_left _ member))
    _ _ _ { refreshRepresents program relation ajtai circuitKeys 1 env with }
    (fun _ => SlotWords.SlotExpr.words_eval _ env) (fun _ => Fin.isLt _) active

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
        1 (refreshInput program env) (finishStepOutput program 1 env) := by
  obtain ⟨holds, _⟩ := CircuitRelation.rowsZero_implies_constraints
    CircuitRelation.publicColumns (constraints program relation) firstFresh width fits layout
    assignment (LogicalPlan.encoded_one layout assignment digest fits.rows.positive encoded)
    rowsZero
  exact ⟨_, constraints_finish program relation ajtai circuitKeys _ holds⟩

/-- The final program's checks on the finish package's columns `env`: the rows of the pieces
other than the Poseidon2 blocks, and each block's outputs as the permutation of its inputs. When
the blocks pack and start at or after the shared end, they give `FinishHolds` for an active proof
in an environment that agrees with `env` below the shared end. -/
theorem checks_finish (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (circuitKeys : Fin 2 → CircuitKey logicalWidth publicFits) (next : Nat) (env : Env)
    (packed : CircuitRelation.packable CircuitRelation.publicColumns
      (pieces program relation) = true)
    (starts : CircuitRelation.startsFrom (RefreshAssembler.sharedEnd base program.witnessWordCount)
      (pieces program relation) = true)
    (rows : Layout.R1CS.RowsHold env (PieceRows.lowerPieces (PilotData.canonicalConstraints ())
      (CircuitRelation.otherPieces (pieces program relation)) next).1)
    (outputs : Frame.OutputsHold (CircuitRelation.blocksOf (pieces program relation)) env) :
    ∃ completed : Env,
      (∀ column, column < RefreshAssembler.sharedEnd base program.witnessWordCount →
        completed column = env column) ∧
      ((refreshInput (logicalWidth := logicalWidth) (publicFits := publicFits) program
          completed).iteration ≠ 0 →
        FinishStep.FinishHolds Folding.Schedule.productionLightFoldCap ajtai
          (decodedKeys circuitKeys (refreshInterface (logicalWidth := logicalWidth)
            (publicFits := publicFits) program).verifierKeys completed)
          1 (refreshInput program completed) (finishStepOutput program 1 completed)) := by
  obtain ⟨holds, agree⟩ := CircuitRelation.checks_implies_pieces
    (Pieces.valid_split _ _) packed next env rows outputs
  exact ⟨_, fun column below => agree column (CircuitRelation.live_below starts below),
    constraints_finish program relation ajtai circuitKeys _ (CircuitRelation.split_holds holds)⟩

/-- `checks_finish` on the finish package that the emitter writes (`StepPackage.packageOf`): the
rows of its pieces, and each of its blocks' outputs as the permutation of its lanes' values. -/
theorem package_finish (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (circuitKeys : Fin 2 → CircuitKey logicalWidth publicFits) (next width : Nat)
    (batches : List WitnessBatch) (application : Nat × Nat × Nat × Nat × Nat) (env : Env)
    (packed : CircuitRelation.packable CircuitRelation.publicColumns
      (pieces program relation) = true)
    (starts : CircuitRelation.startsFrom (RefreshAssembler.sharedEnd base program.witnessWordCount)
      (pieces program relation) = true)
    (rows : ∀ r ∈ (StepPackage.packageOf 2 CircuitRelation.publicColumns (pieces program relation)
      next width batches application).pieces.flatMap StepPackage.PackagePiece.relationRows,
        r.Holds env)
    (blocks : ∀ block ∈ (StepPackage.packageOf 2 CircuitRelation.publicColumns
      (pieces program relation) next width batches application).blocks,
        (fun lane : Fin 16 => env (block.start + 1080 + lane)) =
          Gadgets.Poseidon2.Permutation.runF Gadgets.Poseidon2.Permutation.schedule
            fun lane => (block.lanes.getD lane 0).eval env) :
    ∃ completed : Env,
      (∀ column, column < RefreshAssembler.sharedEnd base program.witnessWordCount →
        completed column = env column) ∧
      ((refreshInput (logicalWidth := logicalWidth) (publicFits := publicFits) program
          completed).iteration ≠ 0 →
        FinishStep.FinishHolds Folding.Schedule.productionLightFoldCap ajtai
          (decodedKeys circuitKeys (refreshInterface (logicalWidth := logicalWidth)
            (publicFits := publicFits) program).verifierKeys completed)
          1 (refreshInput program completed) (finishStepOutput program 1 completed)) := by
  rw [StepPackage.pieceRows_packageOf] at rows
  rw [StepPackage.blocks_packageOf] at blocks
  refine checks_finish program relation ajtai circuitKeys next env packed starts rows fun block member => ?_
  have linear := (CircuitRelation.packable_framed packed).linear block member
  funext lane
  have outputs := congrFun (blocks block member) lane
  have inputs : (fun lane : Fin 16 => (block.lanes.getD lane 0).eval env) = block.input env := by
    funext column
    exact Pieces.lane_eval block.lanes linear env column
  rw [inputs] at outputs
  exact outputs

end NightstreamFPrime.Export.BoundedDepth.FinishRelation
