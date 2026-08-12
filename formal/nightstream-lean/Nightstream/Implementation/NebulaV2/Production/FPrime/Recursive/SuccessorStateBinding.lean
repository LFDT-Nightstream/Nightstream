import Nightstream.Implementation.NebulaV2.Production.Carrier.FieldNativeFullClaim
import Nightstream.Implementation.NebulaV2.Production.Memory.CarryFields
import Nightstream.Implementation.NebulaV2.Production.Memory.TranscriptHashFrame
import Nightstream.Implementation.NebulaV2.Production.NIFS.Core.NifsPublicTranscript
import Nightstream.Implementation.NebulaV2.Production.Application.WasmStateFields
import Nightstream.Implementation.NebulaV2.NIFS.PiCCS.TranscriptSemantics

/-!
Contract: complete Poseidon2 binding of one production successor state.

The frame starts from the verifier-key-bound statement state. It absorbs the
candidate identity, augmented-invocation index, cumulative real application
row count, exact initial application and memory state, complete current
field-native WASM state, exact paper-NIFS output running state, and complete
current field-native memory carry. A terminal gate then produces the four
public state-digest lanes.

The NIFS output is present as every canonical field selected by the exact
augmented-relation exponent. No prover-selected accumulator digest replaces
it. The just-consumed fresh claim is deliberately absent: HyperNova
Construction 2 hashes the updated running vector, not that fresh claim. Equal
output digests recover the complete typed state or identify one exact
successor-transcript collision event.

Does not own generated sponge rows, application transition rows, terminal
verification, Poseidon2 security, Rust refinement, candidate selection, or a
verifier key.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 10000

namespace Nightstream.Implementation.NebulaV2.ProductionSuccessorStateBinding

open Nightstream.Implementation.NebulaV2
open Nightstream.Implementation.NebulaV2.ProductionFieldNativeFullClaim
open Nightstream.Implementation.NebulaV2.ProductionProductNifsPublicTranscript
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.Protocol.NebulaV2
open Nightstream.Protocol.NebulaV2.ProductionProfileCandidates
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation

abbrev State := ProductPoseidon2.State
abbrev StatementId := ProductPoseidon2.StatementId
abbrev ApplicationState := WasmStateEncoding.Image
abbrev MemoryCarry := MemoryCarryCodec.Value

/-! ## Exact typed successor -/

@[ext]
structure Value (candidate : Id) (fullShape : Phi81Relation.Shape) where
  augmentedInvocationIndex : Nat
  realApplicationRowCount : Nat
  initialApplicationState : ApplicationState
  applicationState : ApplicationState
  running : ProductionFieldNativeFullClaim.Running fullShape
  initialMemoryCarry : MemoryCarry
  memoryCarry : MemoryCarry

/-- Canonical integer and inactive-field conditions for the successor. The
invocation field is an index, so its maximum is one less than the exact
candidate invocation count. -/
structure Value.Canonical
    {candidate : Id} {fullShape : Phi81Relation.Shape}
    (headers : FPrime.ChainHeaders Digest.Value)
    (value : Value candidate fullShape) : Prop where
  invocationIndex :
    value.augmentedInvocationIndex < maximumAugmentedInvocations candidate
  realApplicationRowCount : value.realApplicationRowCount < 2 ^ 18
  initialApplicationState : value.initialApplicationState.Canonical
  applicationState : value.applicationState.Canonical
  initialMemoryCarry : value.initialMemoryCarry.Canonical headers
  memoryCarry : value.memoryCarry.Canonical headers

/-! ## Exact successor frame -/

/-- ASCII `NSSO`. -/
def successorTag : Nat := 0x4e53534f
def successorVersion : Nat := 1

noncomputable def runningNativeFields
    {fullShape : Phi81Relation.Shape}
    (running : ProductionFieldNativeFullClaim.Running fullShape) : List Nat :=
  ProductionFieldNativeFullClaim.runningNativeValues running

theorem runningNativeFields_lengthFor
    {fullShape : Phi81Relation.Shape}
    (contract : ProductNifsCodec.FullShapeContractFor
      fullShape.rowVariables fullShape)
    (running : ProductionFieldNativeFullClaim.Running fullShape) :
    (runningNativeFields running).length =
      ProductNifsCodec.runningFieldCountFor fullShape.rowVariables := by
  rw [runningNativeFields,
    ProductionFieldNativeFullClaim.runningNativeValues, List.length_map]
  exact ProductionFieldNativeFullClaim.runningFields_lengthFor contract running

theorem runningNativeFields_length
    {fullShape : Phi81Relation.Shape}
    (contract : ProductNifsCodec.FullShapeContract fullShape)
    (running : ProductionFieldNativeFullClaim.Running fullShape) :
    (runningNativeFields running).length = 83210 := by
  rw [runningNativeFields_lengthFor contract.toSelected,
    contract.rowVariables]
  decide

theorem runningNativeFields_injective
    {fullShape : Phi81Relation.Shape} :
    Function.Injective
      (runningNativeFields (fullShape := fullShape)) := by
  exact ProductionFieldNativeFullClaim.runningNativeValues_injective

/-! ## Challenge-independent successor prefix -/

/-- Exact part of a Construction-2 successor that exists before the memory
carry is opened.  This is the only accumulator authority that the memory
challenge may use.  In particular, it contains the exact NIFS output running
state and does not contain either memory carry, so challenge derivation cannot
form a cycle through the outgoing carry. -/
@[ext]
structure PreCarryValue (candidate : Id) (fullShape : Phi81Relation.Shape) where
  augmentedInvocationIndex : Nat
  realApplicationRowCount : Nat
  initialApplicationState : ApplicationState
  applicationState : ApplicationState
  running : ProductionFieldNativeFullClaim.Running fullShape

def Value.preCarry
    {candidate : Id} {fullShape : Phi81Relation.Shape}
    (value : Value candidate fullShape) : PreCarryValue candidate fullShape :=
  { augmentedInvocationIndex := value.augmentedInvocationIndex
    realApplicationRowCount := value.realApplicationRowCount
    initialApplicationState := value.initialApplicationState
    applicationState := value.applicationState
    running := value.running }

noncomputable def preCarryBlocks
    {candidate : Id} {fullShape : Phi81Relation.Shape}
    (value : PreCarryValue candidate fullShape) : List (List Nat) :=
  [ [successorTag, successorVersion]
  , ProductionMemoryTranscriptHashFrame.profileFields candidate
  , [value.augmentedInvocationIndex, value.realApplicationRowCount]
  , ProductionWasmStateFields.encode value.initialApplicationState
  , ProductionWasmStateFields.encode value.applicationState
  , runningNativeFields value.running
  ]

theorem preCarryBlocks_lengthsFor
    {candidate : Id} {fullShape : Phi81Relation.Shape}
    (contract : ProductNifsCodec.FullShapeContractFor
      fullShape.rowVariables fullShape)
    (value : PreCarryValue candidate fullShape) :
    (preCarryBlocks value).map List.length =
      [2, 4, 2, 85, 85,
        ProductNifsCodec.runningFieldCountFor fullShape.rowVariables] := by
  simp [preCarryBlocks,
    ProductionMemoryTranscriptHashFrame.profileFields_length,
    ProductionWasmStateFields.encode_length,
    runningNativeFields_lengthFor contract]

noncomputable def preCarryFrame
    {candidate : Id} {fullShape : Phi81Relation.Shape}
    (value : PreCarryValue candidate fullShape) : List Nat :=
  (preCarryBlocks value).flatten

theorem preCarryFrame_lengthFor
    {candidate : Id} {fullShape : Phi81Relation.Shape}
    (contract : ProductNifsCodec.FullShapeContractFor
      fullShape.rowVariables fullShape)
    (value : PreCarryValue candidate fullShape) :
    (preCarryFrame value).length =
      ProductNifsCodec.runningFieldCountFor fullShape.rowVariables + 178 := by
  rw [preCarryFrame, List.length_flatten,
    preCarryBlocks_lengthsFor contract]
  simp
  omega

private theorem preCarry_components_equal
    {candidate : Id} {fullShape : Phi81Relation.Shape}
    {left right : PreCarryValue candidate fullShape}
    (equal : preCarryBlocks left = preCarryBlocks right) :
    (left.augmentedInvocationIndex = right.augmentedInvocationIndex /\
      left.realApplicationRowCount = right.realApplicationRowCount) /\
      ProductionWasmStateFields.encode left.initialApplicationState =
        ProductionWasmStateFields.encode right.initialApplicationState /\
      ProductionWasmStateFields.encode left.applicationState =
        ProductionWasmStateFields.encode right.applicationState /\
      runningNativeFields left.running = runningNativeFields right.running := by
  simpa [preCarryBlocks] using equal

/-- The challenge-independent successor prefix is lossless before hashing. -/
theorem preCarryFrame_injective
    {candidate : Id} {fullShape : Phi81Relation.Shape}
    (contract : ProductNifsCodec.FullShapeContractFor
      fullShape.rowVariables fullShape) :
    Function.Injective
      (preCarryFrame (candidate := candidate) (fullShape := fullShape)) := by
  intro left right equal
  have blockEqual : preCarryBlocks left = preCarryBlocks right :=
    WasmResultCodec.flatten_injective_of_lengths
      (preCarryBlocks_lengthsFor contract left)
      (preCarryBlocks_lengthsFor contract right) equal
  rcases preCarry_components_equal blockEqual with
    ⟨⟨invocation, rows⟩, initialApplication, application, running⟩
  apply PreCarryValue.ext
  · exact invocation
  · exact rows
  · exact ProductionWasmStateFields.encode_injective initialApplication
  · exact ProductionWasmStateFields.encode_injective application
  · exact runningNativeFields_injective running

noncomputable def successorBlocks
    {candidate : Id} {fullShape : Phi81Relation.Shape}
    (value : Value candidate fullShape) : List (List Nat) :=
  [ [successorTag, successorVersion]
  , ProductionMemoryTranscriptHashFrame.profileFields candidate
  , [value.augmentedInvocationIndex, value.realApplicationRowCount]
  , ProductionWasmStateFields.encode value.initialApplicationState
  , ProductionWasmStateFields.encode value.applicationState
  , runningNativeFields value.running
  , ProductionMemoryCarryFields.encode value.initialMemoryCarry
  , ProductionMemoryCarryFields.encode value.memoryCarry
  ]

theorem successorBlocks_lengths
    {candidate : Id} {fullShape : Phi81Relation.Shape}
    (contract : ProductNifsCodec.FullShapeContract fullShape)
    (value : Value candidate fullShape) :
    (successorBlocks value).map List.length =
      [2, 4, 2, 85, 85, 83210, 59, 59] := by
  simp [successorBlocks,
    ProductionMemoryTranscriptHashFrame.profileFields_length,
    ProductionWasmStateFields.encode_length,
    runningNativeFields_length contract,
    ProductionMemoryCarryFields.encode_length]

theorem successorBlocks_lengthsFor
    {candidate : Id} {fullShape : Phi81Relation.Shape}
    (contract : ProductNifsCodec.FullShapeContractFor
      fullShape.rowVariables fullShape)
    (value : Value candidate fullShape) :
    (successorBlocks value).map List.length =
      [2, 4, 2, 85, 85,
        ProductNifsCodec.runningFieldCountFor fullShape.rowVariables,
        59, 59] := by
  simp [successorBlocks,
    ProductionMemoryTranscriptHashFrame.profileFields_length,
    ProductionWasmStateFields.encode_length,
    runningNativeFields_lengthFor contract,
    ProductionMemoryCarryFields.encode_length]

noncomputable def successorFrame
    {candidate : Id} {fullShape : Phi81Relation.Shape}
  (value : Value candidate fullShape) : List Nat :=
  (successorBlocks value).flatten

noncomputable def carryFrame
    {candidate : Id} {fullShape : Phi81Relation.Shape}
    (value : Value candidate fullShape) : List Nat :=
  ProductionMemoryCarryFields.encode value.initialMemoryCarry ++
    ProductionMemoryCarryFields.encode value.memoryCarry

theorem carryFrame_length
    {candidate : Id} {fullShape : Phi81Relation.Shape}
    (value : Value candidate fullShape) :
    (carryFrame value).length = 118 := by
  simp [carryFrame, ProductionMemoryCarryFields.encode_length]

theorem successorFrame_eq_preCarry_append
    {candidate : Id} {fullShape : Phi81Relation.Shape}
    (value : Value candidate fullShape) :
    successorFrame value = preCarryFrame value.preCarry ++ carryFrame value := by
  simp [successorFrame, successorBlocks, preCarryFrame, preCarryBlocks,
    carryFrame, Value.preCarry]

theorem successorFrame_length
    {candidate : Id} {fullShape : Phi81Relation.Shape}
    (contract : ProductNifsCodec.FullShapeContract fullShape)
    (value : Value candidate fullShape) :
    (successorFrame value).length = 83506 := by
  rw [successorFrame, List.length_flatten,
    successorBlocks_lengths contract]
  decide

theorem successorFrame_lengthFor
    {candidate : Id} {fullShape : Phi81Relation.Shape}
    (contract : ProductNifsCodec.FullShapeContractFor
      fullShape.rowVariables fullShape)
    (value : Value candidate fullShape) :
    (successorFrame value).length =
      ProductNifsCodec.runningFieldCountFor fullShape.rowVariables + 296 := by
  rw [successorFrame, List.length_flatten,
    successorBlocks_lengthsFor contract]
  simp
  omega

private theorem successor_components_equal
    {candidate : Id} {fullShape : Phi81Relation.Shape}
    {left right : Value candidate fullShape}
    (equal : successorBlocks left = successorBlocks right) :
    (left.augmentedInvocationIndex = right.augmentedInvocationIndex /\
      left.realApplicationRowCount = right.realApplicationRowCount) /\
      ProductionWasmStateFields.encode left.initialApplicationState =
        ProductionWasmStateFields.encode right.initialApplicationState /\
      ProductionWasmStateFields.encode left.applicationState =
        ProductionWasmStateFields.encode right.applicationState /\
      runningNativeFields left.running = runningNativeFields right.running /\
      ProductionMemoryCarryFields.encode left.initialMemoryCarry =
        ProductionMemoryCarryFields.encode right.initialMemoryCarry /\
      ProductionMemoryCarryFields.encode left.memoryCarry =
        ProductionMemoryCarryFields.encode right.memoryCarry := by
  simpa [successorBlocks] using equal

/-- The complete successor frame is injective before hashing. -/
theorem successorFrame_injective
    {candidate : Id} {fullShape : Phi81Relation.Shape}
    (contract : ProductNifsCodec.FullShapeContractFor
      fullShape.rowVariables fullShape) :
    Function.Injective
      (successorFrame (candidate := candidate) (fullShape := fullShape)) := by
  intro left right equal
  have blockEqual : successorBlocks left = successorBlocks right :=
    WasmResultCodec.flatten_injective_of_lengths
      (successorBlocks_lengthsFor contract left)
      (successorBlocks_lengthsFor contract right) equal
  rcases successor_components_equal blockEqual with
    ⟨⟨invocation, rows⟩, initialApplication, application, running,
      initialCarry, carry⟩
  apply Value.ext
  · exact invocation
  · exact rows
  · exact ProductionWasmStateFields.encode_injective initialApplication
  · exact ProductionWasmStateFields.encode_injective application
  · exact runningNativeFields_injective running
  · exact ProductionMemoryCarryFields.encode_injective initialCarry
  · exact ProductionMemoryCarryFields.encode_injective carry

/-- Changing the immutable initial application state changes the exact
successor frame before hashing. This is the concrete `z0` anti-splicing
property required by HyperNova Construction 2. -/
theorem successorFrame_ne_of_initialApplicationState_ne
    {candidate : Id} {fullShape : Phi81Relation.Shape}
    (contract : ProductNifsCodec.FullShapeContractFor
      fullShape.rowVariables fullShape)
    {left right : Value candidate fullShape}
    (different :
      left.initialApplicationState ≠ right.initialApplicationState) :
    successorFrame left ≠ successorFrame right := by
  intro equal
  apply different
  exact congrArg Value.initialApplicationState
    (successorFrame_injective contract equal)

/-- Changing the immutable initial memory carry also changes the exact
successor frame before hashing. -/
theorem successorFrame_ne_of_initialMemoryCarry_ne
    {candidate : Id} {fullShape : Phi81Relation.Shape}
    (contract : ProductNifsCodec.FullShapeContractFor
      fullShape.rowVariables fullShape)
    {left right : Value candidate fullShape}
    (different : left.initialMemoryCarry ≠ right.initialMemoryCarry) :
    successorFrame left ≠ successorFrame right := by
  intro equal
  apply different
  exact congrArg Value.initialMemoryCarry
    (successorFrame_injective contract equal)

private theorem runningNativeFields_canonical
    {fullShape : Phi81Relation.Shape}
    {running : ProductionFieldNativeFullClaim.Running fullShape}
    {field : Nat} (member : field ∈ runningNativeFields running) :
    field < goldilocksP := by
  simp only [runningNativeFields,
    ProductionFieldNativeFullClaim.runningNativeValues,
    List.mem_map] at member
  obtain ⟨value, _, rfl⟩ := member
  exact value.isLt

/-- Every canonical successor frame field is a canonical Goldilocks
representative. Thus modulo-field absorption cannot create a deterministic
encoding alias. -/
theorem successorFrame_fields_canonical
    {candidate : Id} {fullShape : Phi81Relation.Shape}
    {headers : FPrime.ChainHeaders Digest.Value}
    {value : Value candidate fullShape}
    (canonical : value.Canonical headers)
    {field : Nat} (member : field ∈ successorFrame value) :
    field < goldilocksP := by
  rw [successorFrame, successorBlocks] at member
  simp only [List.flatten_cons, List.flatten_nil, List.append_nil,
    List.mem_append] at member
  rcases member with fixed | profile | counters | initialApplication |
      application | running | initialCarry | carry
  · simp only [List.mem_cons, List.not_mem_nil, or_false] at fixed
    rcases fixed with rfl | rfl <;> norm_num [successorTag, successorVersion,
      goldilocksP]
  · simp [ProductionMemoryTranscriptHashFrame.profileFields,
      version, checkedStepsPerFreshClaim, goldilocksP] at profile ⊢
    rcases profile with rfl | rfl | rfl | rfl <;>
      cases candidate <;> norm_num [version, checkedStepsPerFreshClaim,
        goldilocksP]
  · simp only [List.mem_cons, List.not_mem_nil, or_false] at counters
    rcases counters with rfl | rfl
    · exact canonical.invocationIndex.trans (by
        cases candidate <;>
          norm_num [maximumAugmentedInvocations, maximumClaims,
            maximumSegments, claimsPerSegment, stepsPerSegment,
            checkedStepsPerFreshClaim, goldilocksP])
    · exact canonical.realApplicationRowCount.trans
        (by norm_num [goldilocksP])
  · exact ProductionWasmStateFields.encode_fields_canonical
      canonical.initialApplicationState initialApplication
  · exact ProductionWasmStateFields.encode_fields_canonical
      canonical.applicationState application
  · exact runningNativeFields_canonical running
  · exact ProductionMemoryCarryFields.encode_fields_canonical
      canonical.initialMemoryCarry initialCarry
  · exact ProductionMemoryCarryFields.encode_fields_canonical
      canonical.memoryCarry carry

/-! ## One replayable paper-state message -/

private theorem absorbList_append
    (left right : List Nat) (state : State) :
    Poseidon2Duplex.absorbList ProductPoseidon2.constants
        (left ++ right) state =
      Poseidon2Duplex.absorbList ProductPoseidon2.constants right
        (Poseidon2Duplex.absorbList ProductPoseidon2.constants left state) := by
  induction left generalizing state with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      simp only [List.cons_append, Poseidon2Duplex.absorbList]
      exact inductionHypothesis _

/-- The verifier-key-bound prefix and complete successor are one prefix-free
state message. The consumed fresh claim is not part of this message. -/
noncomputable def stateBlocks
    {candidate : Id} {fullShape : Phi81Relation.Shape}
    (statementId : StatementId)
    (successor : Value candidate fullShape) : List (List Nat) :=
  [ ProductPoseidon2.statementIdentifierFields statementId
  , successorFrame successor
  ]

theorem stateBlocks_lengths
    {candidate : Id} {fullShape : Phi81Relation.Shape}
    (contract : ProductNifsCodec.FullShapeContract fullShape)
    (statementId : StatementId)
    (successor : Value candidate fullShape) :
    (stateBlocks statementId successor).map List.length = [366, 83506] := by
  simp [stateBlocks, ProductPoseidon2.statementIdentifierFields,
    ProductPoseidon2.proofPrefixFields_length,
    successorFrame_length contract]

theorem stateBlocks_lengthsFor
    {candidate : Id} {fullShape : Phi81Relation.Shape}
    (contract : ProductNifsCodec.FullShapeContractFor
      fullShape.rowVariables fullShape)
    (statementId : StatementId)
    (successor : Value candidate fullShape) :
    (stateBlocks statementId successor).map List.length =
      [366,
        ProductNifsCodec.runningFieldCountFor fullShape.rowVariables + 296] := by
  simp [stateBlocks, ProductPoseidon2.statementIdentifierFields,
    ProductPoseidon2.proofPrefixFields_length,
    successorFrame_lengthFor contract]

noncomputable def stateFrame
    {candidate : Id} {fullShape : Phi81Relation.Shape}
    (statementId : StatementId)
    (successor : Value candidate fullShape) : List Nat :=
  (stateBlocks statementId successor).flatten

theorem stateFrame_length
    {candidate : Id} {fullShape : Phi81Relation.Shape}
    (contract : ProductNifsCodec.FullShapeContract fullShape)
    (statementId : StatementId)
    (successor : Value candidate fullShape) :
    (stateFrame statementId successor).length = 83872 := by
  rw [stateFrame, List.length_flatten,
    stateBlocks_lengths contract]
  decide

theorem stateFrame_lengthFor
    {candidate : Id} {fullShape : Phi81Relation.Shape}
    (contract : ProductNifsCodec.FullShapeContractFor
      fullShape.rowVariables fullShape)
    (statementId : StatementId)
    (successor : Value candidate fullShape) :
    (stateFrame statementId successor).length =
      ProductNifsCodec.runningFieldCountFor fullShape.rowVariables + 662 := by
  rw [stateFrame, List.length_flatten,
    stateBlocks_lengthsFor contract]
  simp
  omega

/-- The complete state frame recovers the complete successor before hashing. -/
theorem stateFrame_injective
    {candidate : Id} {fullShape : Phi81Relation.Shape}
    (contract : ProductNifsCodec.FullShapeContractFor
      fullShape.rowVariables fullShape)
    (statementId : StatementId) :
    Function.Injective
      (stateFrame (candidate := candidate) (fullShape := fullShape)
        statementId) := by
  intro left right equal
  have blocksEqual : stateBlocks statementId left = stateBlocks statementId right :=
    WasmResultCodec.flatten_injective_of_lengths
      (stateBlocks_lengthsFor contract statementId left)
      (stateBlocks_lengthsFor contract statementId right) equal
  have successorEqual : successorFrame left = successorFrame right := by
    simpa [stateBlocks] using
      congrArg (fun blocks => blocks.getD 1 []) blocksEqual
  exact successorFrame_injective contract successorEqual

/-- Challenge-independent absorption of the exact Construction-2 successor
prefix.  The two memory carries are deliberately absent. -/
noncomputable def preCarryAbsorbedState
    {candidate : Id} {fullShape : Phi81Relation.Shape}
    (statementId : StatementId)
    (value : PreCarryValue candidate fullShape) : State :=
  Poseidon2Duplex.absorbList ProductPoseidon2.constants
    (preCarryFrame value)
    (ProductPoseidon2.initialStateForStatement statementId)

/-- Domain-gated accumulator authority used by the memory transcript.  The
gate is a separate Poseidon2 permutation; no raw rate lane is exported as a
digest. -/
noncomputable def preCarryState
    {candidate : Id} {fullShape : Phi81Relation.Shape}
    (statementId : StatementId)
    (value : PreCarryValue candidate fullShape) : State :=
  Poseidon2Duplex.gate ProductPoseidon2.constants
    (preCarryAbsorbedState statementId value)

/-- Operational form: replay the complete Construction-2 state from the
verifier-key-bound statement state and apply the terminal squeeze gate. -/
noncomputable def outputState
    {candidate : Id} {fullShape : Phi81Relation.Shape}
    (statementId : StatementId)
    (successor : Value candidate fullShape) : State :=
  Poseidon2Duplex.gate ProductPoseidon2.constants
    (Poseidon2Duplex.absorbList ProductPoseidon2.constants
      (carryFrame successor)
      (preCarryAbsorbedState statementId successor.preCarry))

/-- Audit form: the operational state is one replay from the fixed initial
state over the exact verifier-key-bound state frame. -/
theorem outputState_replays_stateFrame
    {candidate : Id} {fullShape : Phi81Relation.Shape}
    (statementId : StatementId)
    (successor : Value candidate fullShape) :
    outputState statementId successor =
      Poseidon2Duplex.gate ProductPoseidon2.constants
        (Poseidon2Duplex.absorbList ProductPoseidon2.constants
          (stateFrame statementId successor)
          ProductPoseidon2.initialState) := by
  rw [outputState, preCarryAbsorbedState, stateFrame, stateBlocks]
  simp only [List.flatten_cons, List.flatten_nil, List.append_nil]
  rw [absorbList_append, ← absorbList_append,
    ← successorFrame_eq_preCarry_append]
  rfl

abbrev CanonicalDigest := MemoryBoundCcsPublic.CanonicalDigest

def outputLane (lane : Fin 4) : Fin 8 :=
  ⟨lane.val, by omega⟩

theorem preCarryState_canonical
    {candidate : Id} {fullShape : Phi81Relation.Shape}
    (statementId : StatementId)
    (value : PreCarryValue candidate fullShape) :
    ProductPiCcsTranscriptSemantics.StateCanonical
      (preCarryState statementId value) := by
  unfold preCarryState Poseidon2Duplex.gate
  exact ProductPiCcsTranscriptSemantics.permute_canonical _

/-- Exact four-lane accumulator authority after the dedicated domain gate. -/
noncomputable def preCarryDigest
    {candidate : Id} {fullShape : Phi81Relation.Shape}
    (statementId : StatementId)
    (value : PreCarryValue candidate fullShape) : CanonicalDigest :=
  fun lane =>
    ⟨(preCarryState statementId value).lanes (outputLane lane), by
      simpa [goldilocksP, ShiftedTernary41V1.modulus] using
        preCarryState_canonical statementId value (outputLane lane)⟩

theorem outputState_canonical
    {candidate : Id} {fullShape : Phi81Relation.Shape}
    (statementId : StatementId)
    (successor : Value candidate fullShape) :
    ProductPiCcsTranscriptSemantics.StateCanonical
      (outputState statementId successor) := by
  unfold outputState Poseidon2Duplex.gate
  exact ProductPiCcsTranscriptSemantics.permute_canonical _

/-- Exact four-lane public digest after the terminal gate. -/
noncomputable def outputDigest
    {candidate : Id} {fullShape : Phi81Relation.Shape}
    (statementId : StatementId)
    (successor : Value candidate fullShape) : CanonicalDigest :=
  fun lane =>
    ⟨(outputState statementId successor).lanes
        (outputLane lane), by
      simpa [goldilocksP, ShiftedTernary41V1.modulus] using
        outputState_canonical statementId successor
          (outputLane lane)⟩

/-! ## Exact collision boundary -/

/-- A pre-carry value whose exact pre-hash frame uses canonical Goldilocks
representatives.  This excludes deterministic reduction aliases from the
named Poseidon2 collision event. -/
structure FieldCanonicalPreCarry
    (candidate : Id) (fullShape : Phi81Relation.Shape) where
  value : PreCarryValue candidate fullShape
  fieldsCanonical : forall field, field ∈ preCarryFrame value ->
    field < goldilocksP

/-- Exact failure event for accumulator authority.  It is not an assumption
that two different NIFS outputs have different digests; that computational
claim is isolated here. -/
def PreCarryTranscriptCollision
    (candidate : Id) (fullShape : Phi81Relation.Shape)
    (statementId : StatementId) : Prop :=
  exists left right : FieldCanonicalPreCarry candidate fullShape,
    preCarryFrame left.value ≠ preCarryFrame right.value /\
      preCarryDigest statementId left.value =
        preCarryDigest statementId right.value

/-- Equal accumulator-authority digests recover the exact pre-carry value or
expose the named Poseidon2 transcript collision. -/
theorem equal_preCarryDigest_recovers_value_or_named_failure
    {candidate : Id} {fullShape : Phi81Relation.Shape}
    (contract : ProductNifsCodec.FullShapeContractFor
      fullShape.rowVariables fullShape)
    {statementId : StatementId}
    (left right : FieldCanonicalPreCarry candidate fullShape)
    (equal : preCarryDigest statementId left.value =
      preCarryDigest statementId right.value) :
    left.value = right.value \/
      PreCarryTranscriptCollision candidate fullShape statementId := by
  by_cases frameEqual :
      preCarryFrame left.value = preCarryFrame right.value
  · exact Or.inl (preCarryFrame_injective contract frameEqual)
  · exact Or.inr ⟨left, right, frameEqual, equal⟩

structure CanonicalState
    (candidate : Id) (fullShape : Phi81Relation.Shape)
    (headers : FPrime.ChainHeaders Digest.Value) where
  value : Value candidate fullShape
  canonical : value.Canonical headers

/-- A state whose exact pre-hash successor frame uses canonical field
representatives. This condition is derivable from physical carrier placement
even before semantic counter and phase bounds are recovered. -/
structure FieldCanonicalState
    (candidate : Id) (fullShape : Phi81Relation.Shape) where
  value : Value candidate fullShape
  fieldsCanonical : forall field, field ∈ successorFrame value ->
    field < goldilocksP

/-- Collision event for the exact complete Construction-2 state transcript.
This event uses unequal canonical pre-hash frames and equal public digests. -/
def SuccessorTranscriptCollision
    (candidate : Id) (fullShape : Phi81Relation.Shape)
    (headers : FPrime.ChainHeaders Digest.Value)
    (statementId : StatementId) : Prop :=
  exists left right : CanonicalState candidate fullShape headers,
    stateFrame statementId left.value ≠ stateFrame statementId right.value /\
      outputDigest statementId left.value = outputDigest statementId right.value

/-- Collision event at the exact fixed-statement successor frame boundary.
Both unequal frames use canonical Goldilocks representatives, so this event
does not include deterministic modulo-encoding aliases. -/
def FieldCanonicalSuccessorTranscriptCollision
    (candidate : Id) (fullShape : Phi81Relation.Shape)
    (statementId : StatementId) : Prop :=
  exists left right : FieldCanonicalState candidate fullShape,
    successorFrame left.value ≠ successorFrame right.value /\
      outputDigest statementId left.value = outputDigest statementId right.value

/-- Equal public successor digests recover the exact canonical state or expose
one named collision event. No verifier result, typed equality, or soundness
implication is an assumption. -/
theorem equal_outputDigest_recovers_state_or_named_failure
    {candidate : Id}
    {fullShape : Phi81Relation.Shape}
    (contract : ProductNifsCodec.FullShapeContractFor
      fullShape.rowVariables fullShape)
    {headers : FPrime.ChainHeaders Digest.Value}
    {statementId : StatementId}
    (left right : CanonicalState candidate fullShape headers)
    (equal :
      outputDigest statementId left.value =
        outputDigest statementId right.value) :
    left.value = right.value \/
      SuccessorTranscriptCollision candidate fullShape headers statementId := by
  by_cases frameEqual :
      stateFrame statementId left.value = stateFrame statementId right.value
  · exact Or.inl (stateFrame_injective contract statementId frameEqual)
  · exact Or.inr ⟨left, right, frameEqual, equal⟩

/-- Equal digests of two field-canonical physical states recover the complete
state or expose the exact fixed-statement collision event. This form is used
for cross-invocation linking before semantic bounds are transferred. -/
theorem equal_outputDigest_recovers_field_state_or_named_failure
    {candidate : Id}
    {fullShape : Phi81Relation.Shape}
    (contract : ProductNifsCodec.FullShapeContractFor
      fullShape.rowVariables fullShape)
    {statementId : StatementId}
    (left right : FieldCanonicalState candidate fullShape)
    (equal : outputDigest statementId left.value =
      outputDigest statementId right.value) :
    left.value = right.value \/
      FieldCanonicalSuccessorTranscriptCollision candidate fullShape
        statementId := by
  by_cases frameEqual :
      successorFrame left.value = successorFrame right.value
  · exact Or.inl (successorFrame_injective contract frameEqual)
  · exact Or.inr ⟨left, right, frameEqual, equal⟩

/-- Candidate separation is present before hashing in the combined frame. -/
theorem stateFrames_ne_of_candidate_ne
    {leftCandidate rightCandidate : Id}
    (different : leftCandidate ≠ rightCandidate)
    {fullShape : Phi81Relation.Shape}
    (contract : ProductNifsCodec.FullShapeContractFor
      fullShape.rowVariables fullShape)
    (statementId : StatementId)
    (leftSuccessor : Value leftCandidate fullShape)
    (rightSuccessor : Value rightCandidate fullShape) :
    stateFrame statementId leftSuccessor ≠
      stateFrame statementId rightSuccessor := by
  intro equal
  have blockEqual :
      stateBlocks statementId leftSuccessor =
        stateBlocks statementId rightSuccessor :=
    WasmResultCodec.flatten_injective_of_lengths
      (stateBlocks_lengthsFor contract statementId leftSuccessor)
      (stateBlocks_lengthsFor contract statementId rightSuccessor) equal
  have successorEqual :
      successorBlocks leftSuccessor = successorBlocks rightSuccessor := by
    have framesEqual :
        successorFrame leftSuccessor = successorFrame rightSuccessor := by
      simpa [stateBlocks] using
        congrArg (fun blocks => blocks.getD 1 []) blockEqual
    exact WasmResultCodec.flatten_injective_of_lengths
      (successorBlocks_lengthsFor contract leftSuccessor)
      (successorBlocks_lengthsFor contract rightSuccessor) framesEqual
  have profileEqual :
      ProductionMemoryTranscriptHashFrame.profileFields leftCandidate =
        ProductionMemoryTranscriptHashFrame.profileFields rightCandidate := by
    simpa [successorBlocks] using
      congrArg (fun blocks => blocks.getD 1 []) successorEqual
  exact different
    (ProductionMemoryTranscriptHashFrame.profileFields_injective profileEqual)

end Nightstream.Implementation.NebulaV2.ProductionSuccessorStateBinding
