import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsParameters
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsCallFrame

/-!
Authoritative call-frame boundary for the concrete fixed-active NIFS selected
by `ConcreteNifsParameters`.

Owns: extraction of the exact running, fresh, proof, and output codec
equations from one whole `nifsVerify` call frame.

Does not own: component column views, physical NIFS rows, application
semantics, paper-event refinement, Rust, or generated artifacts.

No component decoder is a premise below. Every component equation is derived
from the one typed frame decoder used by `CallRecipe.activeSoundness`.

Emits constraints: none.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsSelectedCallFrame

open Nightstream.HyperNova.Construction2.Paper
open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DirectCalls
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsCallFrame
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsParameters
open Nightstream.Protocol.FPrime.ConcretePhi81
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.Nifs
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81

section SelectedFrame

variable {shape : SemanticShape}
variable {TranscriptState : Type}
variable {Digest AppState Witness Encoded RunningWitness FreshWitness : Type}
variable [DecidableEq AppState] [DecidableEq Encoded]
variable {publicRingColumns verifierRows : Nat}
variable {publicFits :
  ringDegree * publicRingColumns <= shape.carrierWidth}
variable {keys : Fin 1 ->
  SelectedKey shape TranscriptState publicRingColumns publicFits verifierRows}
variable {defaultRunning :
  SelectedRunning shape publicRingColumns publicFits verifierRows}
variable {machine :
  Machine
    (SelectedKey shape TranscriptState publicRingColumns publicFits
      verifierRows)
    Digest AppState Witness
    (SelectedRunning shape publicRingColumns publicFits verifierRows)
    (SelectedFresh shape publicRingColumns publicFits verifierRows)
    Encoded 1}
variable {terminalRelations :
  TerminalRelations
    (SelectedKey shape TranscriptState publicRingColumns publicFits
      verifierRows)
    (SelectedRunning shape publicRingColumns publicFits verifierRows)
    RunningWitness
    (SelectedFresh shape publicRingColumns publicFits verifierRows)
    FreshWitness 1}
variable {terminalChecks :
  Nightstream.Protocol.FPrime.CanonicalTerminalVerifier.RelationChecks
    terminalRelations}
variable {widths : Widths} {footprints : Footprints}

local notation "Selected" =>
  ConcreteNifsParameters.selected keys defaultRunning machine
    terminalRelations terminalChecks widths footprints

/-- Whole-frame decoding fixes the exact concrete running operand. -/
theorem running_decodes_of_frame_decodes
    (family : Family (typeSystem Selected))
    {context : Schema (typeSystem Selected)}
    {runningRef :
      Ref (typeSystem Selected) context (.data .running)}
    {freshRef :
      Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef :
      Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame :
      CallFrame (signature := signature Selected) family Call.nifsVerify
        (Refs.cons runningRef
          (Refs.cons freshRef (Refs.cons proofRef .nil))))
    (assignment : ColumnId → Field)
    (running :
      SelectedRunning shape publicRingColumns publicFits verifierRows)
    (fresh :
      SelectedFresh shape publicRingColumns publicFits verifierRows)
    (proof :
      SelectedProof shape TranscriptState publicRingColumns publicFits
        verifierRows)
    (decoded :
      frame.operands.Decodes family assignment
        (.cons running (.cons fresh (.cons proof .nil)))) :
    (runningOperand frame.operands).Decodes family (.data .running)
      assignment running :=
  (decodes_iff family assignment frame.operands running fresh proof).mp
    decoded |>.1

/-- Whole-frame decoding fixes the exact concrete fresh operand. -/
theorem fresh_decodes_of_frame_decodes
    (family : Family (typeSystem Selected))
    {context : Schema (typeSystem Selected)}
    {runningRef :
      Ref (typeSystem Selected) context (.data .running)}
    {freshRef :
      Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef :
      Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame :
      CallFrame (signature := signature Selected) family Call.nifsVerify
        (Refs.cons runningRef
          (Refs.cons freshRef (Refs.cons proofRef .nil))))
    (assignment : ColumnId → Field)
    (running :
      SelectedRunning shape publicRingColumns publicFits verifierRows)
    (fresh :
      SelectedFresh shape publicRingColumns publicFits verifierRows)
    (proof :
      SelectedProof shape TranscriptState publicRingColumns publicFits
        verifierRows)
    (decoded :
      frame.operands.Decodes family assignment
        (.cons running (.cons fresh (.cons proof .nil)))) :
    (freshOperand frame.operands).Decodes family (.data .fresh)
      assignment fresh :=
  (decodes_iff family assignment frame.operands running fresh proof).mp
    decoded |>.2.1

/-- Whole-frame decoding fixes the exact raw certificate carrier. -/
theorem proof_decodes_of_frame_decodes
    (family : Family (typeSystem Selected))
    {context : Schema (typeSystem Selected)}
    {runningRef :
      Ref (typeSystem Selected) context (.data .running)}
    {freshRef :
      Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef :
      Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame :
      CallFrame (signature := signature Selected) family Call.nifsVerify
        (Refs.cons runningRef
          (Refs.cons freshRef (Refs.cons proofRef .nil))))
    (assignment : ColumnId → Field)
    (running :
      SelectedRunning shape publicRingColumns publicFits verifierRows)
    (fresh :
      SelectedFresh shape publicRingColumns publicFits verifierRows)
    (proof :
      SelectedProof shape TranscriptState publicRingColumns publicFits
        verifierRows)
    (decoded :
      frame.operands.Decodes family assignment
        (.cons running (.cons fresh (.cons proof .nil)))) :
    (proofOperand frame.operands).Decodes family (.data .nifsProof)
      assignment proof :=
  (decodes_iff family assignment frame.operands running fresh proof).mp
    decoded |>.2.2

/-- The sole output bundle decodes to the exact computed running carrier used
by the selected call result. -/
theorem output_decodes_of_frame_decodes
    (family : Family (typeSystem Selected))
    {context : Schema (typeSystem Selected)}
    {runningRef :
      Ref (typeSystem Selected) context (.data .running)}
    {freshRef :
      Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef :
      Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame :
      CallFrame (signature := signature Selected) family Call.nifsVerify
        (Refs.cons runningRef
          (Refs.cons freshRef (Refs.cons proofRef .nil))))
    (assignment : ColumnId → Field)
    (output :
      SelectedRunning shape publicRingColumns publicFits verifierRows)
    (decoded :
      frame.outputs.Decodes family assignment (.cons output .nil)) :
    (unaryOutput frame.outputs).Decodes family (.data .running)
      assignment output :=
  (unaryOutput_decodes_iff family assignment frame.outputs output).mp
    decoded

/-- The exact operational result used by the physical recipe, specialized to
the values decoded from this concrete call-frame type. -/
theorem call_result_exact
    (running :
      SelectedRunning shape publicRingColumns publicFits verifierRows)
    (fresh :
      SelectedFresh shape publicRingColumns publicFits verifierRows)
    (proof :
      SelectedProof shape TranscriptState publicRingColumns publicFits
        verifierRows)
    (output :
      SelectedRunning shape publicRingColumns publicFits verifierRows) :
    callEval Selected Call.nifsVerify
        (.cons running (.cons fresh (.cons proof .nil))) =
        some (.cons output .nil) ↔
      ConcretePhi81.Accepted
          (ConcreteNifsParameters.context
            (keys
              Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected)
            running fresh proof).materialize
          proof.certificate ∧
        SelectedRunning.ofResult
            (FixedActive.resultOf
              (ConcreteNifsParameters.context
                (keys
                  Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected)
                running fresh proof).materialize
              proof.certificate) =
          output :=
  ConcreteNifsParameters.callEval_nifsVerify_eq_some_iff
    keys defaultRunning machine terminalRelations terminalChecks widths
    footprints running fresh proof output

end SelectedFrame

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsSelectedCallFrame
