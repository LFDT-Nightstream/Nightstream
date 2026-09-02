import NightstreamFPrime.Export.Codec
import NightstreamFPrime.Export.Stage1.PerApplicationCanonicalAssignment

/-!
Owns the compact transport plan for the canonical per-application assignment.
Each opcode selects one existing retained block and its Lean-owned source
view. `RawValues.schedule` remains the semantic assignment authority.

The plan stores no expanded slots, coordinates, rows, or assignment values.
-/

namespace NightstreamFPrime.Export.Stage1.PerApplicationAssignmentPlan

open NightstreamFPrime.Export.Codec
open NightstreamFPrime.Export.Stage1.PerApplicationCanonicalAssignment
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Lifecycle

abbrev ProgramApplication := Lifecycle.Stage1.Application.Program

/-- Fixed retained-assignment block vocabulary. Enum order has no semantic
effect; `canonicalKinds` owns the transport order. -/
inductive BlockKind where
  | priorPoseidon
  | outputPoseidon
  | laterPoseidon
  | productGroup
  | first54Reject
  | first54Symbol
  | first54Position
  | first54Value
  | first54Product
  | productInput
  | productOutput
  | priorPoseidonInput
  | outputPoseidonInput
  | piCcsPayload
  | runningState
  | runningOutput
  | runningRoundC0
  | runningRoundC1
  | runningPiDec
  | runningFresh
  | piCcsPriorInput
  | piCcsOutputInput
  | piCcsFreshPublicInput
  | piCcsPriorLast
  | piCcsOutputLast
  | piCcsExpectedContext
  | piCcsProofLogical
  | piCcsOutputEndpoint
  | piCcsFresh
  | pilotCanonicalLocal
  | pilotCanonicalFresh
  | pilotOutputDigest
  | piDecParentCommitment
  | piDecParentPublicInput
  | piDecParentEvalK
  | piDecParentEvalA
  | piDecProof
  | piDecLogical
  | piDecFresh
  | samplerLogical
  | samplerFresh
  | applicationInput
  | applicationWitness
  | applicationOutput
  | applicationLocal
deriving Repr, DecidableEq

def BlockKind.format : Format BlockKind where
  encode
    | .priorPoseidon => .atom 0
    | .outputPoseidon => .atom 1
    | .laterPoseidon => .atom 2
    | .productGroup => .atom 3
    | .first54Reject => .atom 4
    | .first54Symbol => .atom 5
    | .first54Position => .atom 6
    | .first54Value => .atom 7
    | .first54Product => .atom 8
    | .productInput => .atom 9
    | .productOutput => .atom 10
    | .priorPoseidonInput => .atom 11
    | .outputPoseidonInput => .atom 12
    | .piCcsPayload => .atom 13
    | .runningState => .atom 14
    | .runningOutput => .atom 15
    | .runningRoundC0 => .atom 16
    | .runningRoundC1 => .atom 17
    | .runningPiDec => .atom 18
    | .runningFresh => .atom 19
    | .piCcsPriorInput => .atom 20
    | .piCcsOutputInput => .atom 21
    | .piCcsFreshPublicInput => .atom 22
    | .piCcsPriorLast => .atom 23
    | .piCcsOutputLast => .atom 24
    | .piCcsExpectedContext => .atom 25
    | .piCcsProofLogical => .atom 26
    | .piCcsOutputEndpoint => .atom 27
    | .piCcsFresh => .atom 28
    | .pilotCanonicalLocal => .atom 29
    | .pilotCanonicalFresh => .atom 30
    | .pilotOutputDigest => .atom 31
    | .piDecParentCommitment => .atom 32
    | .piDecParentPublicInput => .atom 33
    | .piDecParentEvalK => .atom 34
    | .piDecParentEvalA => .atom 35
    | .piDecProof => .atom 36
    | .piDecLogical => .atom 37
    | .piDecFresh => .atom 38
    | .samplerLogical => .atom 39
    | .samplerFresh => .atom 40
    | .applicationInput => .atom 41
    | .applicationWitness => .atom 42
    | .applicationOutput => .atom 43
    | .applicationLocal => .atom 44
  decode
    | .atom 0 => .ok .priorPoseidon
    | .atom 1 => .ok .outputPoseidon
    | .atom 2 => .ok .laterPoseidon
    | .atom 3 => .ok .productGroup
    | .atom 4 => .ok .first54Reject
    | .atom 5 => .ok .first54Symbol
    | .atom 6 => .ok .first54Position
    | .atom 7 => .ok .first54Value
    | .atom 8 => .ok .first54Product
    | .atom 9 => .ok .productInput
    | .atom 10 => .ok .productOutput
    | .atom 11 => .ok .priorPoseidonInput
    | .atom 12 => .ok .outputPoseidonInput
    | .atom 13 => .ok .piCcsPayload
    | .atom 14 => .ok .runningState
    | .atom 15 => .ok .runningOutput
    | .atom 16 => .ok .runningRoundC0
    | .atom 17 => .ok .runningRoundC1
    | .atom 18 => .ok .runningPiDec
    | .atom 19 => .ok .runningFresh
    | .atom 20 => .ok .piCcsPriorInput
    | .atom 21 => .ok .piCcsOutputInput
    | .atom 22 => .ok .piCcsFreshPublicInput
    | .atom 23 => .ok .piCcsPriorLast
    | .atom 24 => .ok .piCcsOutputLast
    | .atom 25 => .ok .piCcsExpectedContext
    | .atom 26 => .ok .piCcsProofLogical
    | .atom 27 => .ok .piCcsOutputEndpoint
    | .atom 28 => .ok .piCcsFresh
    | .atom 29 => .ok .pilotCanonicalLocal
    | .atom 30 => .ok .pilotCanonicalFresh
    | .atom 31 => .ok .pilotOutputDigest
    | .atom 32 => .ok .piDecParentCommitment
    | .atom 33 => .ok .piDecParentPublicInput
    | .atom 34 => .ok .piDecParentEvalK
    | .atom 35 => .ok .piDecParentEvalA
    | .atom 36 => .ok .piDecProof
    | .atom 37 => .ok .piDecLogical
    | .atom 38 => .ok .piDecFresh
    | .atom 39 => .ok .samplerLogical
    | .atom 40 => .ok .samplerFresh
    | .atom 41 => .ok .applicationInput
    | .atom 42 => .ok .applicationWitness
    | .atom 43 => .ok .applicationOutput
    | .atom 44 => .ok .applicationLocal
    | _ => .error "invalid per-application assignment block kind"
  decode_encode := by
    intro kind
    cases kind <;> rfl

def canonicalKinds : List BlockKind :=
  [.priorPoseidon, .outputPoseidon, .laterPoseidon, .productGroup,
    .first54Reject, .first54Symbol, .first54Position, .first54Value,
    .first54Product, .productInput, .productOutput, .priorPoseidonInput,
    .outputPoseidonInput, .piCcsPayload, .runningState, .runningOutput,
    .runningRoundC0, .runningRoundC1, .runningPiDec, .runningFresh,
    .piCcsPriorInput, .piCcsOutputInput, .piCcsFreshPublicInput,
    .piCcsPriorLast, .piCcsOutputLast, .piCcsExpectedContext,
    .piCcsProofLogical, .piCcsOutputEndpoint, .piCcsFresh,
    .pilotCanonicalLocal, .pilotCanonicalFresh, .pilotOutputDigest,
    .piDecParentCommitment, .piDecParentPublicInput, .piDecParentEvalK,
    .piDecParentEvalA, .piDecProof, .piDecLogical, .piDecFresh,
    .samplerLogical, .samplerFresh, .applicationInput, .applicationWitness,
    .applicationOutput, .applicationLocal]

@[simp] theorem canonicalKinds_length : canonicalKinds.length = 45 := by
  rfl

/-- Interpret one compact opcode through the existing Lean-owned block and
source definitions. -/
def BlockKind.expand {application : ProgramApplication}
    (raw : RawValues application) : BlockKind →
      CanonicalBlockAssignment.BlockValue
  | .priorPoseidon => Canonical.ofBlock
      (PiRLCRetainedGeometry.priorPoseidonBlock application) raw.retainedSource
  | .outputPoseidon => Canonical.ofBlock
      (PiRLCRetainedGeometry.outputPoseidonBlock application) raw.retainedSource
  | .laterPoseidon => Canonical.ofBlock
      (PiRLCRetainedGeometry.laterPoseidonBlock application) raw.retainedSource
  | .productGroup => Canonical.ofBlock
      (PiRLCRetainedGeometry.productGroupBlock application) raw.retainedSource
  | .first54Reject => Canonical.ofBlock
      (PiRLCFirst54RetainedBlocks.rejectBlock application) raw.retainedSource
  | .first54Symbol => Canonical.ofBlock
      (PiRLCFirst54RetainedBlocks.symbolBlock application) raw.retainedSource
  | .first54Position => Canonical.ofBlock
      (PiRLCFirst54RetainedBlocks.positionBlock application) raw.retainedSource
  | .first54Value => Canonical.ofBlock
      (PiRLCFirst54RetainedBlocks.valueBlock application) raw.retainedSource
  | .first54Product => Canonical.ofBlock
      (PiRLCFirst54RetainedBlocks.productBlock application) raw.retainedSource
  | .productInput => Canonical.ofBlock
      (PiRLCRetainedGeometry.productInputBlock application) raw.retainedSource
  | .productOutput => Canonical.ofBlock
      (PiRLCRetainedGeometry.productOutputBlock application) raw.retainedSource
  | .priorPoseidonInput => Canonical.ofBlock
      (PiRLCPoseidonGeometry.priorInputBlock application) raw.retainedSource
  | .outputPoseidonInput => Canonical.ofBlock
      (PiRLCPoseidonGeometry.outputInputBlock application) raw.retainedSource
  | .piCcsPayload => Canonical.ofBlock
      (PiCCSActionPayloadBlock.block application) raw.payloadSource
  | .runningState => Canonical.ofBlock
      (RunningTransitionRetainedBlocks.stateBlock application) raw.retainedSource
  | .runningOutput => Canonical.ofBlock
      (RunningTransitionRetainedBlocks.outputBlock application) raw.retainedSource
  | .runningRoundC0 => Canonical.ofBlock
      (RunningTransitionRetainedBlocks.roundC0Block application) raw.retainedSource
  | .runningRoundC1 => Canonical.ofBlock
      (RunningTransitionRetainedBlocks.roundC1Block application) raw.retainedSource
  | .runningPiDec => Canonical.ofBlock
      (RunningTransitionRetainedBlocks.piDecBlock application) raw.retainedSource
  | .runningFresh => Canonical.ofBlock
      (RunningTransitionRetainedBlocks.freshBlock application) raw.retainedSource
  | .piCcsPriorInput => Canonical.ofBlock
      (PiCCSOrdinaryRetainedBlocks.priorInputBlock application) raw.retainedSource
  | .piCcsOutputInput => Canonical.ofBlock
      (PiCCSOrdinaryRetainedBlocks.outputInputBlock application) raw.retainedSource
  | .piCcsFreshPublicInput => Canonical.ofBlock
      (PiCCSOrdinaryRetainedBlocks.freshPublicInputBlock application)
      raw.retainedSource
  | .piCcsPriorLast => Canonical.ofBlock
      (PiCCSOrdinaryRetainedBlocks.priorLastBlock application) raw.retainedSource
  | .piCcsOutputLast => Canonical.ofBlock
      (PiCCSOrdinaryRetainedBlocks.outputLastBlock application) raw.retainedSource
  | .piCcsExpectedContext => Canonical.ofBlock
      (PiCCSOrdinaryRetainedBlocks.expectedContextBlock application)
      raw.retainedSource
  | .piCcsProofLogical => Canonical.ofBlock
      (PiCCSOrdinaryRetainedBlocks.proofLogicalBlock application)
      raw.retainedSource
  | .piCcsOutputEndpoint => Canonical.ofBlock
      (PiCCSOrdinaryRetainedBlocks.outputEndpointBlock application)
      raw.retainedSource
  | .piCcsFresh => Canonical.ofBlock
      (PiCCSOrdinaryRetainedBlocks.freshBlock application) raw.retainedSource
  | .pilotCanonicalLocal => Canonical.ofBlock
      (PilotOrdinaryRetainedBlocks.canonicalLocalBlock application)
      raw.retainedSource
  | .pilotCanonicalFresh => Canonical.ofBlock
      (PilotOrdinaryRetainedBlocks.canonicalFreshBlock application)
      raw.retainedSource
  | .pilotOutputDigest => Canonical.ofBlock
      (PilotOrdinaryRetainedBlocks.outputDigestBlock application)
      raw.retainedSource
  | .piDecParentCommitment => Canonical.ofBlock
      (PiDECRetainedBlocks.parentCommitmentBlock application) raw.retainedSource
  | .piDecParentPublicInput => Canonical.ofBlock
      (PiDECRetainedBlocks.parentPublicInputBlock application) raw.retainedSource
  | .piDecParentEvalK => Canonical.ofBlock
      (PiDECRetainedBlocks.parentEvalKBlock application) raw.retainedSource
  | .piDecParentEvalA => Canonical.ofBlock
      (PiDECRetainedBlocks.parentEvalABlock application) raw.retainedSource
  | .piDecProof => Canonical.ofBlock
      (PiDECRetainedBlocks.proofBlock application) raw.retainedSource
  | .piDecLogical => Canonical.ofBlock
      (PiDECRetainedBlocks.logicalBlock application) raw.retainedSource
  | .piDecFresh => Canonical.ofBlock
      (PiDECRetainedBlocks.freshBlock application) raw.retainedSource
  | .samplerLogical => Canonical.ofBlock
      (PiRLCSamplerOrdinaryRetainedBlocks.logicalBlock application)
      raw.retainedSource
  | .samplerFresh => Canonical.ofBlock
      (PiRLCSamplerOrdinaryRetainedBlocks.freshBlock application)
      raw.retainedSource
  | .applicationInput => Canonical.ofBlock
      (ApplicationRetainedBlocks.inputBlock application) raw.applicationSource
  | .applicationWitness => Canonical.ofBlock
      (ApplicationRetainedBlocks.witnessBlock application) raw.applicationSource
  | .applicationOutput => Canonical.ofBlock
      (ApplicationRetainedBlocks.outputBlock application) raw.applicationSource
  | .applicationLocal => Canonical.ofBlock
      (ApplicationRetainedBlocks.localBlock application) raw.applicationSource

/-- Expand the fixed compact plan. The result remains a 45-entry schedule;
no retained slot or assignment coordinate is materialized. -/
def expand {application : ProgramApplication} (raw : RawValues application) :
    Canonical.Schedule :=
  canonicalKinds.map (BlockKind.expand raw)

/-- The compact plan is exactly the existing canonical assignment schedule. -/
theorem expand_eq_schedule {application : ProgramApplication}
    (raw : RawValues application) : expand raw = raw.schedule := by
  rfl

/-- Execute the Lean-authored transport program as the final retained
assignment. -/
def execute {application : ProgramApplication} (raw : RawValues application) :
    Fin (PerApplicationFixedPoint.logicalWidth application) →
      NightstreamFPrime.Spec.F :=
  Canonical.assignment (encodedHashCells raw.outputDigest) (expand raw)

/-- The executable transport program is exactly the canonical assignment
used by the final structural relation. -/
theorem execute_eq_assignment {application : ProgramApplication}
    (raw : RawValues application) : execute raw = raw.assignment := by
  unfold execute RawValues.assignment
  rw [expand_eq_schedule]

def format : Format (List BlockKind) := Codec.list BlockKind.format

theorem canonical_decode_encode :
    format.decode (format.encode canonicalKinds) = .ok canonicalKinds := by
  exact format.decode_encode canonicalKinds

end NightstreamFPrime.Export.Stage1.PerApplicationAssignmentPlan
