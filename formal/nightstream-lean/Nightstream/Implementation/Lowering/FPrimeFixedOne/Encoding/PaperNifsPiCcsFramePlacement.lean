import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsGlobalColumnMap
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsPiCcsCallBinding

/-!
Contract: derive the complete public `Pi_CCS` placement from one typed
`nifsVerify` call frame.

Every numeric location is obtained from an existing coordinate of the
authoritative running operand and the sole global column map.  Callers supply
the selected codec projection laws, but cannot choose numeric indices or
introduce copied public values.

Emits constraints: none.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsPiCcsFramePlacement

open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.Goldilocks.NumericRowBridge
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsCallFrame
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsCodecProjection
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsGlobalColumnMap
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsPiCcsCallBinding
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsPiCcsSourceBinding
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.Nifs

universe uCommitment uPublicInput uScalar uState

abbrev K := Nightstream.SuperNeo.Concrete.K

/-- The public `Pi_CCS` placement determined by the running operand itself.

The codec need not be reconstructed here: its projection laws and width
certificate are the serialization boundary consumed by the frame-level
decoder theorem.  The numeric placement is no longer a caller input. -/
def fromFrame
    {parameters : Parameters}
    {family : Family (typeSystem parameters)}
    {context : Schema (typeSystem parameters)}
    {running :
      Ref (typeSystem parameters) context (.data .running)}
    {fresh :
      Ref (typeSystem parameters) context (.data .fresh)}
    {proof :
      Ref (typeSystem parameters) context (.data .nifsProof)}
    (frame :
      CallFrame (signature := signature parameters) family Call.nifsVerify
        (Refs.cons running (Refs.cons fresh (Refs.cons proof .nil))))
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {shape : Shape}
    {codec :
      Codec (PaperNonInteractive.Running K Commitment PublicInput shape)}
    (views : RunningViews codec)
    (widthsAgree : codec.width = running.port.layout.owners.length) :
    RunningPlacement views (runningOperand frame.operands) widthsAgree
      (columnMap frame) where
  priorPoint index :=
    let projected :=
      (views.priorPoint index).columns
        (runningOperand frame.operands) widthsAgree
    kLocation frame projected
      (runningOperand_mem frame
        ((views.priorPoint index).c0_mem
          (runningOperand frame.operands) widthsAgree))
      (runningOperand_mem frame
        ((views.priorPoint index).c1_mem
          (runningOperand frame.operands) widthsAgree))
  claimedCoefficient coordinate :=
    let projected :=
      (views.claimedCoefficient coordinate).columns
        (runningOperand frame.operands) widthsAgree
    kLocation frame projected
      (runningOperand_mem frame
        ((views.claimedCoefficient coordinate).c0_mem
          (runningOperand frame.operands) widthsAgree))
      (runningOperand_mem frame
        ((views.claimedCoefficient coordinate).c1_mem
          (runningOperand frame.operands) widthsAgree))

/-- The automatically derived placement reaches the selected paper verifier
input.  The only semantic premise is successful decoding of the actual
running operand; numeric placement is not exposed to the caller. -/
theorem decodedVerifierInput_eq_statement
    {parameters : Parameters}
    {family : Family (typeSystem parameters)}
    {context : Schema (typeSystem parameters)}
    {runningRef :
      Ref (typeSystem parameters) context (.data .running)}
    {freshRef :
      Ref (typeSystem parameters) context (.data .fresh)}
    {proofRef :
      Ref (typeSystem parameters) context (.data .nifsProof)}
    (frame :
      CallFrame (signature := signature parameters) family Call.nifsVerify
        (Refs.cons runningRef
          (Refs.cons freshRef (Refs.cons proofRef .nil))))
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    (key :
      PaperNonInteractive.Key K Commitment PublicInput Scalar State shape
        columns blockCount degreeBound)
    {codec :
      Codec (PaperNonInteractive.Running K Commitment PublicInput shape)}
    (views : RunningViews codec)
    (widthsAgree : codec.width = runningRef.port.layout.owners.length)
    (template : KPiCcsTranscript.Input shape degreeBound)
    (assignment : ColumnId → Field)
    (running : PaperNonInteractive.Running K Commitment PublicInput shape)
    (fresh : PaperNonInteractive.Fresh Commitment PublicInput shape)
    (decoded :
      codec.decode
          ((runningOperand frame.operands).values assignment) =
        some running) :
    KPiCcsOccurrence.decodedVerifierInput
        (KPiCcsTranscript.occurrenceInput
          ((fromFrame frame views widthsAgree).bindInput key template))
        (numericAssignment (columnMap frame) assignment) =
      (key.statement running fresh).verifierInput key.lift :=
  (fromFrame frame views widthsAgree).decodedVerifierInput_eq_statement
    key template assignment running fresh decoded

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsPiCcsFramePlacement
