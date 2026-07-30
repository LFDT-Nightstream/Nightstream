import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsPiCcsCallBinding
import Nightstream.Implementation.R1CS.Canonical.KPiCcsEventBinding

/-!
Contract: discharge the `Pi_CCS` event theorem's source-binding premise from
the decoded running operand and the selected paper key.

The hidden table witness is attached only through
`StrongReduction.Statement.sourceProtocolData`.  That construction is
definitionally verifier-input preserving.  The public side is reconstructed
from the actual running codec coordinates by `RunningPlacement`; therefore
the equality formerly supplied as `sourceBinding` is proved internally.

The conclusion retains the paper's exact `MixingRoot`,
`SumCheckCollision`, and `OutputMismatch` alternatives.  This module neither
absorbs them into generic failure nor asserts a probability bound.

The canonical row program evaluates the selected concrete extension carrier
with `ConcreteCarrier.extensionOps`.  The enclosing NIFS profile must later
prove that its key selects those same operations; this module does not silently
identify arbitrary key operations with the canonical carrier.

Emits constraints: none beyond the selected transcript/PiCCS row program.
-/

set_option autoImplicit false
set_option maxRecDepth 10000

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsPiCcsEventBinding

open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.Goldilocks.NumericRowBridge
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsCodecProjection
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsPiCcsSourceBinding
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsPiCcsCallBinding
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.Nifs

universe uCommitment uPublicInput uScalar uState

abbrev K := Nightstream.SuperNeo.Concrete.K
abbrev Constants := Poseidon2Schedule.Constants

private theorem numeric_constant
    (columnMap : Nat → ColumnId)
    (assignment : ColumnId → Field)
    (constantWire : assignment (columnMap 0) = 1) :
    numericAssignment columnMap assignment 0 = 1 := by
  have value := congrArg Fin.val constantWire
  simpa [numericAssignment] using value

/-- **Transcript-bound PiCCS reduction with internally constructed source
authority.**

No verifier input, source-binding equality, degree bound, challenge-set size,
or challenge family is supplied by the caller.  They are all selected by the
key, decoded call operand, and transcript rows. -/
theorem rows_imply_tableTruth_or_paperBadEvent
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
    {views : RunningViews codec}
    {layout : Layout}
    {bundle : ColumnBundle layout}
    {widthsAgree : codec.width = layout.owners.length}
    {columnMap : Nat → ColumnId}
    (placement :
      RunningPlacement views bundle widthsAgree columnMap)
    (constants : Constants)
    (template : KPiCcsTranscript.Input shape degreeBound)
    (assignment : ColumnId → Field)
    (running : PaperNonInteractive.Running K Commitment PublicInput shape)
    (fresh : PaperNonInteractive.Fresh Commitment PublicInput shape)
    (witness : StrongReduction.OutputWitness shape columns)
    (decoded : codec.decode (bundle.values assignment) = some running)
    (constantWire : assignment (columnMap 0) = 1)
    (satisfied :
      Satisfies
        (KPiCcsTranscript.rows constants
          (placement.bindInput key template))
        (numericAssignment columnMap assignment)) :
    let input := placement.bindInput key template
    let pulled := numericAssignment columnMap assignment
    let data :=
      (key.statement running fresh).sourceProtocolData key.lift witness
    (TableResidualData.toTableObligations ConcreteCarrier.extensionOps
        (SignedCoefficientObject.toTableResidualData
          ConcreteCarrier.extensionOps
          (data.toJointData ConcreteCarrier.extensionOps))).AllHold ∨
      SignedCoefficientObject.MixingRoot ConcreteCarrier.extensionOps
        (data.toJointData ConcreteCarrier.extensionOps)
        (KPiCcsEventBinding.paperAlpha constants pulled input)
        (KPiCcsEventBinding.paperGamma constants pulled input) ∨
      ProtocolPolynomial.FixedWidth.SumCheckCollision
        ConcreteCarrier.extensionOps data
        (KPiCcsEventBinding.paperAlpha constants pulled input)
        (KPiCcsEventBinding.paperGamma constants pulled input)
        degreeBound key.challengeSetSize
        (KPiCcsEventBinding.paperPoint constants pulled input)
        (KPiCcsOccurrence.decodedCertificate
          (KPiCcsTranscript.occurrenceInput input) pulled) ∨
      ProtocolPolynomial.OutputMismatch ConcreteCarrier.extensionOps data
        (KPiCcsEventBinding.paperAlpha constants pulled input)
        (KPiCcsEventBinding.paperGamma constants pulled input)
        (KPiCcsEventBinding.paperPoint constants pulled input)
        (KPiCcsOccurrence.decodedMessage
          (KPiCcsTranscript.occurrenceInput input) pulled) := by
  dsimp only
  let input := placement.bindInput key template
  let pulled := numericAssignment columnMap assignment
  let data :=
    (key.statement running fresh).sourceProtocolData key.lift witness
  have sourceBinding :
      data.toVerifierInput =
        KPiCcsOccurrence.decodedVerifierInput
          (KPiCcsTranscript.occurrenceInput input) pulled := by
    calc
      data.toVerifierInput =
          (key.statement running fresh).verifierInput key.lift := by
        exact
          (key.statement running fresh).sourceProtocolData_toVerifierInput
            key.lift witness
      _ =
          KPiCcsOccurrence.decodedVerifierInput
            (KPiCcsTranscript.occurrenceInput input) pulled := by
        exact
          (placement.decodedVerifierInput_eq_statement
            key template assignment running fresh decoded).symm
  have degreeCovers : data.toVerifierInput.sumcheckDegreeBound ≤ degreeBound := by
    rw [sourceBinding,
      placement.decodedVerifierInput_eq_statement
        key template assignment running fresh decoded]
    exact key.statement_sumcheckDegreeBound_le running fresh
  exact
    KPiCcsEventBinding.rows_imply_tableTruth_or_paperBadEvent
      constants pulled input
      (numericAssignment_canonical columnMap assignment)
      (numeric_constant columnMap assignment constantWire)
      satisfied data sourceBinding degreeCovers key.challengeSetSize

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsPiCcsEventBinding
