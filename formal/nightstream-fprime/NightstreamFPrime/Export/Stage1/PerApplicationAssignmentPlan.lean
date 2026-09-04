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

structure BlockTemplate (application : ProgramApplication) where
  sourceWidth : Nat
  block : LowNormBlock.Block sourceWidth
  source : RawValues application → Fin sourceWidth →
    NightstreamFPrime.Spec.F

/-- One opcode selects one Lean-owned block and its value-source domain.
Raw values are applied only after this template fixes the block geometry. -/
def BlockKind.template (application : ProgramApplication) :
    BlockKind → BlockTemplate application
  | .priorPoseidon =>
      ⟨_, PiRLCRetainedGeometry.priorPoseidonBlock application,
        fun raw => raw.retainedSource⟩
  | .outputPoseidon =>
      ⟨_, PiRLCRetainedGeometry.outputPoseidonBlock application,
        fun raw => raw.retainedSource⟩
  | .laterPoseidon =>
      ⟨_, PiRLCRetainedGeometry.laterPoseidonBlock application,
        fun raw => raw.retainedSource⟩
  | .productGroup =>
      ⟨_, PiRLCRetainedGeometry.productGroupBlock application,
        fun raw => raw.retainedSource⟩
  | .first54Reject =>
      ⟨_, PiRLCFirst54RetainedBlocks.rejectBlock application,
        fun raw => raw.retainedSource⟩
  | .first54Symbol =>
      ⟨_, PiRLCFirst54RetainedBlocks.symbolBlock application,
        fun raw => raw.retainedSource⟩
  | .first54Position =>
      ⟨_, PiRLCFirst54RetainedBlocks.positionBlock application,
        fun raw => raw.retainedSource⟩
  | .first54Value =>
      ⟨_, PiRLCFirst54RetainedBlocks.valueBlock application,
        fun raw => raw.retainedSource⟩
  | .first54Product =>
      ⟨_, PiRLCFirst54RetainedBlocks.productBlock application,
        fun raw => raw.retainedSource⟩
  | .productInput =>
      ⟨_, PiRLCRetainedGeometry.productInputBlock application,
        fun raw => raw.retainedSource⟩
  | .productOutput =>
      ⟨_, PiRLCRetainedGeometry.productOutputBlock application,
        fun raw => raw.retainedSource⟩
  | .priorPoseidonInput =>
      ⟨_, PiRLCPoseidonGeometry.priorInputBlock application,
        fun raw => raw.retainedSource⟩
  | .outputPoseidonInput =>
      ⟨_, PiRLCPoseidonGeometry.outputInputBlock application,
        fun raw => raw.retainedSource⟩
  | .piCcsPayload =>
      ⟨_, PiCCSActionPayloadBlock.block application,
        fun raw => raw.payloadSource⟩
  | .runningState =>
      ⟨_, RunningTransitionRetainedBlocks.stateBlock application,
        fun raw => raw.retainedSource⟩
  | .runningOutput =>
      ⟨_, RunningTransitionRetainedBlocks.outputBlock application,
        fun raw => raw.retainedSource⟩
  | .runningRoundC0 =>
      ⟨_, RunningTransitionRetainedBlocks.roundC0Block application,
        fun raw => raw.retainedSource⟩
  | .runningRoundC1 =>
      ⟨_, RunningTransitionRetainedBlocks.roundC1Block application,
        fun raw => raw.retainedSource⟩
  | .runningPiDec =>
      ⟨_, RunningTransitionRetainedBlocks.piDecBlock application,
        fun raw => raw.retainedSource⟩
  | .runningFresh =>
      ⟨_, RunningTransitionRetainedBlocks.freshBlock application,
        fun raw => raw.retainedSource⟩
  | .piCcsPriorInput =>
      ⟨_, PiCCSOrdinaryRetainedBlocks.priorInputBlock application,
        fun raw => raw.retainedSource⟩
  | .piCcsOutputInput =>
      ⟨_, PiCCSOrdinaryRetainedBlocks.outputInputBlock application,
        fun raw => raw.retainedSource⟩
  | .piCcsFreshPublicInput =>
      ⟨_, PiCCSOrdinaryRetainedBlocks.freshPublicInputBlock application,
        fun raw => raw.retainedSource⟩
  | .piCcsPriorLast =>
      ⟨_, PiCCSOrdinaryRetainedBlocks.priorLastBlock application,
        fun raw => raw.retainedSource⟩
  | .piCcsOutputLast =>
      ⟨_, PiCCSOrdinaryRetainedBlocks.outputLastBlock application,
        fun raw => raw.retainedSource⟩
  | .piCcsExpectedContext =>
      ⟨_, PiCCSOrdinaryRetainedBlocks.expectedContextBlock application,
        fun raw => raw.retainedSource⟩
  | .piCcsProofLogical =>
      ⟨_, PiCCSOrdinaryRetainedBlocks.proofLogicalBlock application,
        fun raw => raw.retainedSource⟩
  | .piCcsOutputEndpoint =>
      ⟨_, PiCCSOrdinaryRetainedBlocks.outputEndpointBlock application,
        fun raw => raw.retainedSource⟩
  | .piCcsFresh =>
      ⟨_, PiCCSOrdinaryRetainedBlocks.freshBlock application,
        fun raw => raw.retainedSource⟩
  | .pilotCanonicalLocal =>
      ⟨_, PilotOrdinaryRetainedBlocks.canonicalLocalBlock application,
        fun raw => raw.retainedSource⟩
  | .pilotCanonicalFresh =>
      ⟨_, PilotOrdinaryRetainedBlocks.canonicalFreshBlock application,
        fun raw => raw.retainedSource⟩
  | .pilotOutputDigest =>
      ⟨_, PilotOrdinaryRetainedBlocks.outputDigestBlock application,
        fun raw => raw.retainedSource⟩
  | .piDecParentCommitment =>
      ⟨_, PiDECRetainedBlocks.parentCommitmentBlock application,
        fun raw => raw.retainedSource⟩
  | .piDecParentPublicInput =>
      ⟨_, PiDECRetainedBlocks.parentPublicInputBlock application,
        fun raw => raw.retainedSource⟩
  | .piDecParentEvalK =>
      ⟨_, PiDECRetainedBlocks.parentEvalKBlock application,
        fun raw => raw.retainedSource⟩
  | .piDecParentEvalA =>
      ⟨_, PiDECRetainedBlocks.parentEvalABlock application,
        fun raw => raw.retainedSource⟩
  | .piDecProof =>
      ⟨_, PiDECRetainedBlocks.proofBlock application,
        fun raw => raw.retainedSource⟩
  | .piDecLogical =>
      ⟨_, PiDECRetainedBlocks.logicalBlock application,
        fun raw => raw.retainedSource⟩
  | .piDecFresh =>
      ⟨_, PiDECRetainedBlocks.freshBlock application,
        fun raw => raw.retainedSource⟩
  | .samplerLogical =>
      ⟨_, PiRLCSamplerOrdinaryRetainedBlocks.logicalBlock application,
        fun raw => raw.retainedSource⟩
  | .samplerFresh =>
      ⟨_, PiRLCSamplerOrdinaryRetainedBlocks.freshBlock application,
        fun raw => raw.retainedSource⟩
  | .applicationInput =>
      ⟨_, ApplicationRetainedBlocks.inputBlock application,
        fun raw => raw.applicationSource⟩
  | .applicationWitness =>
      ⟨_, ApplicationRetainedBlocks.witnessBlock application,
        fun raw => raw.applicationSource⟩
  | .applicationOutput =>
      ⟨_, ApplicationRetainedBlocks.outputBlock application,
        fun raw => raw.applicationSource⟩
  | .applicationLocal =>
      ⟨_, ApplicationRetainedBlocks.localBlock application,
        fun raw => raw.applicationSource⟩

/-- Interpret one compact opcode through its Lean-owned block template. -/
def BlockKind.expand {application : ProgramApplication}
    (raw : RawValues application) (kind : BlockKind) :
      CanonicalBlockAssignment.BlockValue :=
  let template := BlockKind.template application kind
  Canonical.ofBlock template.block (template.source raw)

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
