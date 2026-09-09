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
  | productOutput
  | priorPoseidonInput
  | outputPoseidonInput
  | piCcsPayload
  | runningRoundC0
  | runningRoundC1
  | runningPiDec
  | runningFresh
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
  | piDecLogical
  | piDecFresh
  | samplerLogical
  | samplerFresh
  | applicationWitness
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
    | .productOutput => .atom 9
    | .priorPoseidonInput => .atom 10
    | .outputPoseidonInput => .atom 11
    | .piCcsPayload => .atom 12
    | .runningRoundC0 => .atom 13
    | .runningRoundC1 => .atom 14
    | .runningPiDec => .atom 15
    | .runningFresh => .atom 16
    | .piCcsFreshPublicInput => .atom 17
    | .piCcsPriorLast => .atom 18
    | .piCcsOutputLast => .atom 19
    | .piCcsExpectedContext => .atom 20
    | .piCcsProofLogical => .atom 21
    | .piCcsOutputEndpoint => .atom 22
    | .piCcsFresh => .atom 23
    | .pilotCanonicalLocal => .atom 24
    | .pilotCanonicalFresh => .atom 25
    | .pilotOutputDigest => .atom 26
    | .piDecLogical => .atom 27
    | .piDecFresh => .atom 28
    | .samplerLogical => .atom 29
    | .samplerFresh => .atom 30
    | .applicationWitness => .atom 31
    | .applicationLocal => .atom 32
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
    | .atom 9 => .ok .productOutput
    | .atom 10 => .ok .priorPoseidonInput
    | .atom 11 => .ok .outputPoseidonInput
    | .atom 12 => .ok .piCcsPayload
    | .atom 13 => .ok .runningRoundC0
    | .atom 14 => .ok .runningRoundC1
    | .atom 15 => .ok .runningPiDec
    | .atom 16 => .ok .runningFresh
    | .atom 17 => .ok .piCcsFreshPublicInput
    | .atom 18 => .ok .piCcsPriorLast
    | .atom 19 => .ok .piCcsOutputLast
    | .atom 20 => .ok .piCcsExpectedContext
    | .atom 21 => .ok .piCcsProofLogical
    | .atom 22 => .ok .piCcsOutputEndpoint
    | .atom 23 => .ok .piCcsFresh
    | .atom 24 => .ok .pilotCanonicalLocal
    | .atom 25 => .ok .pilotCanonicalFresh
    | .atom 26 => .ok .pilotOutputDigest
    | .atom 27 => .ok .piDecLogical
    | .atom 28 => .ok .piDecFresh
    | .atom 29 => .ok .samplerLogical
    | .atom 30 => .ok .samplerFresh
    | .atom 31 => .ok .applicationWitness
    | .atom 32 => .ok .applicationLocal
    | _ => .error "invalid per-application assignment block kind"
  decode_encode := by
    intro kind
    cases kind <;> rfl

def canonicalKinds : List BlockKind :=
  [.priorPoseidon, .outputPoseidon, .laterPoseidon, .productGroup,
    .first54Reject, .first54Symbol, .first54Position, .first54Value,
    .first54Product, .productOutput, .priorPoseidonInput,
    .outputPoseidonInput, .piCcsPayload,
    .runningRoundC0, .runningRoundC1, .runningPiDec, .runningFresh,
    .piCcsFreshPublicInput,
    .piCcsPriorLast, .piCcsOutputLast, .piCcsExpectedContext,
    .piCcsProofLogical, .piCcsOutputEndpoint, .piCcsFresh,
    .pilotCanonicalLocal, .pilotCanonicalFresh, .pilotOutputDigest,
    .piDecLogical, .piDecFresh,
    .samplerLogical, .samplerFresh, .applicationWitness, .applicationLocal]

@[simp] theorem canonicalKinds_length : canonicalKinds.length = 33 := by
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
  | .applicationWitness =>
      ⟨_, ApplicationRetainedBlocks.witnessBlock application,
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

/-- Expand the fixed compact plan. The result remains a 33-entry schedule;
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
