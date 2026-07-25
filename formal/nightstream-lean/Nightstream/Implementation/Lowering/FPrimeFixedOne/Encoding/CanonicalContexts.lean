import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.CanonicalBranchPlan
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.SourceOwners

/-!
Contract: exact physical SSA contexts for the two fixed-one typed programs.

Owns:
- the input, instruction-output, branch-join, and continuation column bundles
  at every Step and Terminal source position;
- codec-width agreement for every exact context;
- the sole Boolean selector and assertion coordinates selected from those
  typed contexts.

Does not own: primitive recipes, receipt ordering, row emission, source-owner
alignment, normal-form specifications, Rust artifacts, or R1CS refinement.

Every new context is the canonical output allocation prepended to the exact
earlier context.  No numeric artifact column or optional lookup is used.

Emits constraints: no.
-/

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding

open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary

namespace CanonicalContexts

namespace Step

def input
    (parameters : Parameters) :
    Columns (stepInputSchema parameters) :=
  inputColumns (stepInputSchema parameters)

def afterStep
    (parameters : Parameters) :
    Columns
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.afterStepSchema
        parameters) :=
  (instructionColumns SourceOwners.stepApplyPath
    [Ports.committedState parameters]).append
      (input parameters)

def common
    (parameters : Parameters) :
    Columns
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.commonSchema
        parameters) :=
  (instructionColumns SourceOwners.stepSelectorPath
    [Ports.auxiliaryBit parameters]).append
      (afterStep parameters)

def afterBaseEquality
    (parameters : Parameters) :
    Columns
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.afterBaseEqualitySchema
        parameters) :=
  (instructionColumns SourceOwners.stepBaseStateEqualPath
    [Ports.auxiliaryBit parameters]).append
      (common parameters)

def afterBaseLiteral
    (parameters : Parameters) :
    Columns
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.afterBaseLiteralSchema
        parameters) :=
  (instructionColumns SourceOwners.stepBaseDefaultPath
    [Ports.committedRunning parameters]).append
      (afterBaseEquality parameters)

def afterHash
    (parameters : Parameters) :
    Columns
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.afterHashSchema
        parameters) :=
  (instructionColumns SourceOwners.stepRecursiveHashPriorPath
    [Ports.auxiliaryDigest parameters]).append
      (common parameters)

def afterFreshPublic
    (parameters : Parameters) :
    Columns
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.afterFreshPublicSchema
        parameters) :=
  (instructionColumns SourceOwners.stepRecursiveFreshPublicPath
    [Ports.auxiliaryEncoded parameters]).append
      (afterHash parameters)

def afterEncode
    (parameters : Parameters) :
    Columns
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.afterEncodeSchema
        parameters) :=
  (instructionColumns SourceOwners.stepRecursiveEncodePath
    [Ports.auxiliaryEncoded parameters]).append
      (afterFreshPublic parameters)

def afterEncodedEquality
    (parameters : Parameters) :
    Columns
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.afterEncodedEqualitySchema
        parameters) :=
  (instructionColumns SourceOwners.stepRecursiveEncodedEqualPath
    [Ports.auxiliaryBit parameters]).append
      (afterEncode parameters)

def afterNifs
    (parameters : Parameters) :
    Columns
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.afterNifsSchema
        parameters) :=
  (instructionColumns SourceOwners.stepRecursiveNifsPath
    [Ports.committedRunning parameters]).append
      (afterEncodedEquality parameters)

def joined
    (parameters : Parameters) :
    Columns
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.joinedSchema
        parameters) :=
  branchJoinColumns SourceOwners.stepBranchPath
    (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.joinedSchema
      parameters)

def continuationInput
    (parameters : Parameters) :
    Columns
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.continuationInputSchema
        parameters) :=
  (joined parameters).append (common parameters)

def afterHashNext
    (parameters : Parameters) :
    Columns
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.afterHashNextSchema
        parameters) :=
  (instructionColumns SourceOwners.stepContinuationHashPath
    [Ports.publicDigest parameters]).append
      (continuationInput parameters)

/-- Exact final Step export list in protocol result order. -/
def resultExports
    (parameters : Parameters) :
    Exports (typeSystem parameters)
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.afterHashNextSchema
        parameters)
      (stepResultSchema parameters) :=
  .cons
    (.there
      (.there
        (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.CommonRefs.zNext
          parameters)))
    (.cons
      (.there (.here (Ports.committedRunning parameters)))
      (.cons
        (.here (Ports.publicDigest parameters))
        .nil))

/-- Every final Step export preserves the complete physical port, including
its ownership layout. -/
def resultExportsCompatible
    (parameters : Parameters) :
    ExportsCompatible (resultExports parameters) :=
  .cons rfl (.cons rfl (.cons rfl .nil))

/-- Exact physical coordinates exposed by the typed Step yield. -/
def result
    (parameters : Parameters) :
    Columns (stepResultSchema parameters) :=
  exportColumns
    (resultExports parameters)
    (resultExportsCompatible parameters)
    (afterHashNext parameters)

theorem inputWidths
    (parameters : Parameters)
    (profile : Profile parameters) :
    SchemaWidthAgrees (profile.family parameters)
      (stepInputSchema parameters) :=
  profile.stepInputSchema_widthsAgree parameters

theorem afterStepWidths
    (parameters : Parameters)
    (profile : Profile parameters) :
    SchemaWidthAgrees (profile.family parameters)
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.afterStepSchema
        parameters) :=
  Profile.schemaWidthAgrees_append
    (profile.callOutputs_widthsAgree parameters .step)
    (inputWidths parameters profile)

theorem commonWidths
    (parameters : Parameters)
    (profile : Profile parameters) :
    SchemaWidthAgrees (profile.family parameters)
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.commonSchema
        parameters) :=
  Profile.schemaWidthAgrees_append
    (profile.callOutputs_widthsAgree parameters .iterationZero)
    (afterStepWidths parameters profile)

theorem afterBaseEqualityWidths
    (parameters : Parameters)
    (profile : Profile parameters) :
    SchemaWidthAgrees (profile.family parameters)
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.afterBaseEqualitySchema
        parameters) :=
  Profile.schemaWidthAgrees_append
    (profile.callOutputs_widthsAgree parameters .stateEqual)
    (commonWidths parameters profile)

theorem afterBaseLiteralWidths
    (parameters : Parameters)
    (profile : Profile parameters) :
    SchemaWidthAgrees (profile.family parameters)
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.afterBaseLiteralSchema
        parameters) :=
  Profile.schemaWidthAgrees_cons
    (profile.committedRunning_widthsAgree parameters)
    (afterBaseEqualityWidths parameters profile)

theorem afterHashWidths
    (parameters : Parameters)
    (profile : Profile parameters) :
    SchemaWidthAgrees (profile.family parameters)
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.afterHashSchema
        parameters) :=
  Profile.schemaWidthAgrees_append
    (profile.callOutputs_widthsAgree parameters .hashPrior)
    (commonWidths parameters profile)

theorem afterFreshPublicWidths
    (parameters : Parameters)
    (profile : Profile parameters) :
    SchemaWidthAgrees (profile.family parameters)
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.afterFreshPublicSchema
        parameters) :=
  Profile.schemaWidthAgrees_append
    (profile.callOutputs_widthsAgree parameters .freshPublic)
    (afterHashWidths parameters profile)

theorem afterEncodeWidths
    (parameters : Parameters)
    (profile : Profile parameters) :
    SchemaWidthAgrees (profile.family parameters)
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.afterEncodeSchema
        parameters) :=
  Profile.schemaWidthAgrees_append
    (profile.callOutputs_widthsAgree parameters .encodeInstance)
    (afterFreshPublicWidths parameters profile)

theorem afterEncodedEqualityWidths
    (parameters : Parameters)
    (profile : Profile parameters) :
    SchemaWidthAgrees (profile.family parameters)
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.afterEncodedEqualitySchema
        parameters) :=
  Profile.schemaWidthAgrees_append
    (profile.callOutputs_widthsAgree parameters .encodedEqual)
    (afterEncodeWidths parameters profile)

theorem afterNifsWidths
    (parameters : Parameters)
    (profile : Profile parameters) :
    SchemaWidthAgrees (profile.family parameters)
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.afterNifsSchema
        parameters) :=
  Profile.schemaWidthAgrees_append
    (profile.callOutputs_widthsAgree parameters .nifsVerify)
    (afterEncodedEqualityWidths parameters profile)

theorem joinedWidths
    (parameters : Parameters)
    (profile : Profile parameters) :
    SchemaWidthAgrees (profile.family parameters)
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.joinedSchema
        parameters) :=
  Profile.schemaWidthAgrees_cons
    (profile.committedRunning_widthsAgree parameters)
    (by intro port member; simp at member)

theorem continuationInputWidths
    (parameters : Parameters)
    (profile : Profile parameters) :
    SchemaWidthAgrees (profile.family parameters)
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.continuationInputSchema
        parameters) :=
  Profile.schemaWidthAgrees_append
    (joinedWidths parameters profile)
    (commonWidths parameters profile)

theorem afterHashNextWidths
    (parameters : Parameters)
    (profile : Profile parameters) :
    SchemaWidthAgrees (profile.family parameters)
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.afterHashNextSchema
        parameters) :=
  Profile.schemaWidthAgrees_append
    (profile.callOutputs_widthsAgree parameters .hashNext)
    (continuationInputWidths parameters profile)

def selector
    (parameters : Parameters)
    (profile : Profile parameters) : ColumnId :=
  CanonicalPrimitivePlan.bitCoordinate profile
    (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.CommonRefs.iterationZero
      parameters)
    (common parameters)
    (commonWidths parameters profile)

def baseCondition
    (parameters : Parameters)
    (profile : Profile parameters) : ColumnId :=
  CanonicalPrimitivePlan.bitCoordinate profile
    (.here (Ports.auxiliaryBit parameters))
    (afterBaseEquality parameters)
    (afterBaseEqualityWidths parameters profile)

def recursiveCondition
    (parameters : Parameters)
    (profile : Profile parameters) : ColumnId :=
  CanonicalPrimitivePlan.bitCoordinate profile
    (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.RecursiveRefs.encodedEqual
      parameters)
    (afterEncodedEquality parameters)
    (afterEncodedEqualityWidths parameters profile)

def baseRunning
    (parameters : Parameters) :
    Bundle (Ports.committedRunning parameters) :=
  HVec.head
    (instructionColumns SourceOwners.stepBaseDefaultPath
      [Ports.committedRunning parameters])

def recursiveRunning
    (parameters : Parameters) :
    Bundle (Ports.committedRunning parameters) :=
  HVec.head
    (instructionColumns SourceOwners.stepRecursiveNifsPath
      [Ports.committedRunning parameters])

end Step

namespace Terminal

def input
    (parameters : Parameters) :
    Columns (terminalInputSchema parameters) :=
  inputColumns (terminalInputSchema parameters)

def branchInput
    (parameters : Parameters) :
    Columns
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.branchInputSchema
        parameters) :=
  (instructionColumns SourceOwners.terminalSelectorPath
    [Ports.auxiliaryBit parameters]).append
      (input parameters)

def afterBaseEquality
    (parameters : Parameters) :
    Columns
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.afterBaseEqualitySchema
        parameters) :=
  (instructionColumns SourceOwners.terminalBaseStateEqualPath
    [Ports.auxiliaryBit parameters]).append
      (branchInput parameters)

def afterHash
    (parameters : Parameters) :
    Columns
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.afterHashSchema
        parameters) :=
  (instructionColumns SourceOwners.terminalRecursiveHashPriorPath
    [Ports.auxiliaryDigest parameters]).append
      (branchInput parameters)

def afterFreshPublic
    (parameters : Parameters) :
    Columns
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.afterFreshPublicSchema
        parameters) :=
  (instructionColumns SourceOwners.terminalRecursiveFreshPublicPath
    [Ports.auxiliaryEncoded parameters]).append
      (afterHash parameters)

def afterEncode
    (parameters : Parameters) :
    Columns
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.afterEncodeSchema
        parameters) :=
  (instructionColumns SourceOwners.terminalRecursiveEncodePath
    [Ports.auxiliaryEncoded parameters]).append
      (afterFreshPublic parameters)

def afterEncodedEquality
    (parameters : Parameters) :
    Columns
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.afterEncodedEqualitySchema
        parameters) :=
  (instructionColumns SourceOwners.terminalRecursiveEncodedEqualPath
    [Ports.auxiliaryBit parameters]).append
      (afterEncode parameters)

def afterRunningCheck
    (parameters : Parameters) :
    Columns
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.afterRunningCheckSchema
        parameters) :=
  (instructionColumns SourceOwners.terminalRecursiveRunningCheckPath
    [Ports.auxiliaryBit parameters]).append
      (afterEncodedEquality parameters)

def afterFreshCheck
    (parameters : Parameters) :
    Columns
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.afterFreshCheckSchema
        parameters) :=
  (instructionColumns SourceOwners.terminalRecursiveFreshCheckPath
    [Ports.auxiliaryBit parameters]).append
      (afterRunningCheck parameters)

theorem inputWidths
    (parameters : Parameters)
    (profile : Profile parameters) :
    SchemaWidthAgrees (profile.family parameters)
      (terminalInputSchema parameters) :=
  profile.terminalInputSchema_widthsAgree parameters

theorem branchInputWidths
    (parameters : Parameters)
    (profile : Profile parameters) :
    SchemaWidthAgrees (profile.family parameters)
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.branchInputSchema
        parameters) :=
  Profile.schemaWidthAgrees_append
    (profile.callOutputs_widthsAgree parameters .iterationZero)
    (inputWidths parameters profile)

theorem afterBaseEqualityWidths
    (parameters : Parameters)
    (profile : Profile parameters) :
    SchemaWidthAgrees (profile.family parameters)
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.afterBaseEqualitySchema
        parameters) :=
  Profile.schemaWidthAgrees_append
    (profile.callOutputs_widthsAgree parameters .stateEqual)
    (branchInputWidths parameters profile)

theorem afterHashWidths
    (parameters : Parameters)
    (profile : Profile parameters) :
    SchemaWidthAgrees (profile.family parameters)
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.afterHashSchema
        parameters) :=
  Profile.schemaWidthAgrees_append
    (profile.callOutputs_widthsAgree parameters .hashPrior)
    (branchInputWidths parameters profile)

theorem afterFreshPublicWidths
    (parameters : Parameters)
    (profile : Profile parameters) :
    SchemaWidthAgrees (profile.family parameters)
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.afterFreshPublicSchema
        parameters) :=
  Profile.schemaWidthAgrees_append
    (profile.callOutputs_widthsAgree parameters .freshPublic)
    (afterHashWidths parameters profile)

theorem afterEncodeWidths
    (parameters : Parameters)
    (profile : Profile parameters) :
    SchemaWidthAgrees (profile.family parameters)
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.afterEncodeSchema
        parameters) :=
  Profile.schemaWidthAgrees_append
    (profile.callOutputs_widthsAgree parameters .encodeInstance)
    (afterFreshPublicWidths parameters profile)

theorem afterEncodedEqualityWidths
    (parameters : Parameters)
    (profile : Profile parameters) :
    SchemaWidthAgrees (profile.family parameters)
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.afterEncodedEqualitySchema
        parameters) :=
  Profile.schemaWidthAgrees_append
    (profile.callOutputs_widthsAgree parameters .encodedEqual)
    (afterEncodeWidths parameters profile)

theorem afterRunningCheckWidths
    (parameters : Parameters)
    (profile : Profile parameters) :
    SchemaWidthAgrees (profile.family parameters)
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.afterRunningCheckSchema
        parameters) :=
  Profile.schemaWidthAgrees_append
    (profile.callOutputs_widthsAgree parameters .runningCheck)
    (afterEncodedEqualityWidths parameters profile)

theorem afterFreshCheckWidths
    (parameters : Parameters)
    (profile : Profile parameters) :
    SchemaWidthAgrees (profile.family parameters)
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.afterFreshCheckSchema
        parameters) :=
  Profile.schemaWidthAgrees_append
    (profile.callOutputs_widthsAgree parameters .freshCheck)
    (afterRunningCheckWidths parameters profile)

def selector
    (parameters : Parameters)
    (profile : Profile parameters) : ColumnId :=
  CanonicalPrimitivePlan.bitCoordinate profile
    (Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.BranchRef.iterationZero
      parameters)
    (branchInput parameters)
    (branchInputWidths parameters profile)

def baseCondition
    (parameters : Parameters)
    (profile : Profile parameters) : ColumnId :=
  CanonicalPrimitivePlan.bitCoordinate profile
    (.here (Ports.auxiliaryBit parameters))
    (afterBaseEquality parameters)
    (afterBaseEqualityWidths parameters profile)

def priorCondition
    (parameters : Parameters)
    (profile : Profile parameters) : ColumnId :=
  CanonicalPrimitivePlan.bitCoordinate profile
    (Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.RecursiveRef.encodedEqual
      parameters)
    (afterEncodedEquality parameters)
    (afterEncodedEqualityWidths parameters profile)

def runningCondition
    (parameters : Parameters)
    (profile : Profile parameters) : ColumnId :=
  CanonicalPrimitivePlan.bitCoordinate profile
    (Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.RecursiveRef.runningAccepted
      parameters)
    (afterRunningCheck parameters)
    (afterRunningCheckWidths parameters profile)

def freshCondition
    (parameters : Parameters)
    (profile : Profile parameters) : ColumnId :=
  CanonicalPrimitivePlan.bitCoordinate profile
    (Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.RecursiveRef.freshAccepted
      parameters)
    (afterFreshCheck parameters)
    (afterFreshCheckWidths parameters profile)

end Terminal

end CanonicalContexts

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding
