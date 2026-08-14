import Nightstream.Protocol.Nebula.WasmState

/-!
Contract: exact typed application-state image and field order for V2.

Assurance tier: model-level.

Owns the complete flat 55-field WASM state image, its exact field tags and bit
widths, and lossless conversion to and from `AppStateVector`.

Does not own the byte container, Goldilocks-coordinate placement, generated
rows, Rust parsing, or the verifier-key manifest hash. The implementation
codec must serialize this schema as exactly 2,293 canonical little-endian
bits and prove refinement to this file.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Protocol.Nebula.WasmStateEncoding

open Nightstream.Protocol.Nebula.WasmState

/-- Version of the authority-bearing application-state schema. -/
def schemaVersion : Nat := 1

/-- Flat typed public image. No state field remains behind an opaque digest. -/
@[ext] structure Image where
  pc : Nat
  operandStackPointer : Nat
  stackFrameBase : Nat
  outputEnabled : Bool
  outputLow : Nat
  outputHigh : Nat
  callStackDepth : Nat
  memoryPagesPresent : Bool
  memoryPagesValue : Nat
  maximumMemoryPagesPresent : Bool
  maximumMemoryPagesValue : Nat
  localsFrameBase : Nat
  halted : Bool
  trapped : Bool
  trapCode : Nat
  parameterInitializationActive : Bool
  parameterInitializationRemaining : Nat
  tailCallPending : Bool
  hostArgumentsActive : Bool
  hostArgumentsRemaining : Nat
  hostResultPending : Bool
  hostCalleeFunction : Nat
  hostEventChain : Fin 4 → Nat
  eventBuffer : Fin 8 → Nat
  eventBufferSlot : Nat
  permutationPending : Bool
  permutationRound : Nat
  permutationState : Fin 12 → Nat
  grammarMode : Bool
  grammarTurnExportFunction : Nat
  grammarEventsRemaining : Nat
  grammarEventIndex : Nat
  grammarArgumentsBase : Nat
  grammarSlotCursor : Nat

/-- Flatten the semantic state into the exact public-image schema. -/
def encode (state : AppStateVector) : Image where
  pc := state.pc
  operandStackPointer := state.operandStackPointer
  stackFrameBase := state.stackFrameBase
  outputEnabled := state.output.enabled
  outputLow := state.output.low
  outputHigh := state.output.high
  callStackDepth := state.callStackDepth
  memoryPagesPresent := state.memoryPages.present
  memoryPagesValue := state.memoryPages.value
  maximumMemoryPagesPresent := state.maximumMemoryPages.present
  maximumMemoryPagesValue := state.maximumMemoryPages.value
  localsFrameBase := state.localsFrameBase
  halted := state.halted
  trapped := state.trapped
  trapCode := state.trapCode
  parameterInitializationActive := state.parameterInitialization.active
  parameterInitializationRemaining := state.parameterInitialization.remaining
  tailCallPending := state.tailCallPending
  hostArgumentsActive := state.hostArguments.active
  hostArgumentsRemaining := state.hostArguments.remaining
  hostResultPending := state.hostResultPending
  hostCalleeFunction := state.hostCalleeFunction
  hostEventChain := state.hostEventChain
  eventBuffer := state.eventAbsorb.buffer
  eventBufferSlot := state.eventAbsorb.bufferSlot
  permutationPending := state.eventAbsorb.permutationPending
  permutationRound := state.eventAbsorb.permutationRound
  permutationState := state.eventAbsorb.permutationState
  grammarMode := state.grammarMode
  grammarTurnExportFunction := state.grammar.turnExportFunction
  grammarEventsRemaining := state.grammar.eventsRemaining
  grammarEventIndex := state.grammar.eventIndex
  grammarArgumentsBase := state.grammar.argumentsBase
  grammarSlotCursor := state.grammar.slotCursor

/-- Reconstruct the semantic state from the complete typed image. -/
def decode (image : Image) : AppStateVector where
  pc := image.pc
  operandStackPointer := image.operandStackPointer
  stackFrameBase := image.stackFrameBase
  output :=
    { enabled := image.outputEnabled
      low := image.outputLow
      high := image.outputHigh }
  callStackDepth := image.callStackDepth
  memoryPages :=
    { present := image.memoryPagesPresent
      value := image.memoryPagesValue }
  maximumMemoryPages :=
    { present := image.maximumMemoryPagesPresent
      value := image.maximumMemoryPagesValue }
  localsFrameBase := image.localsFrameBase
  halted := image.halted
  trapped := image.trapped
  trapCode := image.trapCode
  parameterInitialization :=
    { active := image.parameterInitializationActive
      remaining := image.parameterInitializationRemaining }
  tailCallPending := image.tailCallPending
  hostArguments :=
    { active := image.hostArgumentsActive
      remaining := image.hostArgumentsRemaining }
  hostResultPending := image.hostResultPending
  hostCalleeFunction := image.hostCalleeFunction
  hostEventChain := image.hostEventChain
  eventAbsorb :=
    { buffer := image.eventBuffer
      bufferSlot := image.eventBufferSlot
      permutationPending := image.permutationPending
      permutationRound := image.permutationRound
      permutationState := image.permutationState }
  grammarMode := image.grammarMode
  grammar :=
    { turnExportFunction := image.grammarTurnExportFunction
      eventsRemaining := image.grammarEventsRemaining
      eventIndex := image.grammarEventIndex
      argumentsBase := image.grammarArgumentsBase
      slotCursor := image.grammarSlotCursor }

theorem decode_encode (state : AppStateVector) :
    decode (encode state) = state :=
  rfl

theorem encode_decode (image : Image) :
    encode (decode image) = image :=
  rfl

theorem encode_injective : Function.Injective encode :=
  Function.LeftInverse.injective decode_encode

/-- Independent field tags. Their order below is part of the schema. -/
inductive FieldTag where
  | pc
  | operandStackPointer
  | stackFrameBase
  | outputEnabled
  | outputLow
  | outputHigh
  | callStackDepth
  | memoryPagesPresent
  | memoryPagesValue
  | maximumMemoryPagesPresent
  | maximumMemoryPagesValue
  | localsFrameBase
  | halted
  | trapped
  | trapCode
  | parameterInitializationActive
  | parameterInitializationRemaining
  | tailCallPending
  | hostArgumentsActive
  | hostArgumentsRemaining
  | hostResultPending
  | hostCalleeFunction
  | hostEventChain (index : Fin 4)
  | eventBuffer (index : Fin 8)
  | eventBufferSlot
  | permutationPending
  | permutationRound
  | permutationState (index : Fin 12)
  | grammarMode
  | grammarTurnExportFunction
  | grammarEventsRemaining
  | grammarEventIndex
  | grammarArgumentsBase
  | grammarSlotCursor
deriving DecidableEq, Repr

/-- Canonical little-endian bit width of each tagged field. -/
def FieldTag.bitWidth : FieldTag → Nat
  | .pc => 64
  | .operandStackPointer => 64
  | .stackFrameBase => 64
  | .outputEnabled => 1
  | .outputLow => 32
  | .outputHigh => 32
  | .callStackDepth => 64
  | .memoryPagesPresent => 1
  | .memoryPagesValue => 32
  | .maximumMemoryPagesPresent => 1
  | .maximumMemoryPagesValue => 32
  | .localsFrameBase => 64
  | .halted => 1
  | .trapped => 1
  | .trapCode => 32
  | .parameterInitializationActive => 1
  | .parameterInitializationRemaining => 32
  | .tailCallPending => 1
  | .hostArgumentsActive => 1
  | .hostArgumentsRemaining => 32
  | .hostResultPending => 1
  | .hostCalleeFunction => 32
  | .hostEventChain _ => 64
  | .eventBuffer _ => 64
  | .eventBufferSlot => 2
  | .permutationPending => 1
  | .permutationRound => 5
  | .permutationState _ => 64
  | .grammarMode => 1
  | .grammarTurnExportFunction => 32
  | .grammarEventsRemaining => 32
  | .grammarEventIndex => 32
  | .grammarArgumentsBase => 64
  | .grammarSlotCursor => 3

private theorem boolToNat_injective :
    Function.Injective Bool.toNat := by
  intro left right equal
  cases left <;> cases right <;> simp_all

/-- Typed field lookup used by the canonical bit codec. Boolean values are
the integers zero and one. -/
def Image.fieldValue (image : Image) : FieldTag → Nat
  | .pc => image.pc
  | .operandStackPointer => image.operandStackPointer
  | .stackFrameBase => image.stackFrameBase
  | .outputEnabled => image.outputEnabled.toNat
  | .outputLow => image.outputLow
  | .outputHigh => image.outputHigh
  | .callStackDepth => image.callStackDepth
  | .memoryPagesPresent => image.memoryPagesPresent.toNat
  | .memoryPagesValue => image.memoryPagesValue
  | .maximumMemoryPagesPresent => image.maximumMemoryPagesPresent.toNat
  | .maximumMemoryPagesValue => image.maximumMemoryPagesValue
  | .localsFrameBase => image.localsFrameBase
  | .halted => image.halted.toNat
  | .trapped => image.trapped.toNat
  | .trapCode => image.trapCode
  | .parameterInitializationActive =>
      image.parameterInitializationActive.toNat
  | .parameterInitializationRemaining =>
      image.parameterInitializationRemaining
  | .tailCallPending => image.tailCallPending.toNat
  | .hostArgumentsActive => image.hostArgumentsActive.toNat
  | .hostArgumentsRemaining => image.hostArgumentsRemaining
  | .hostResultPending => image.hostResultPending.toNat
  | .hostCalleeFunction => image.hostCalleeFunction
  | .hostEventChain index => image.hostEventChain index
  | .eventBuffer index => image.eventBuffer index
  | .eventBufferSlot => image.eventBufferSlot
  | .permutationPending => image.permutationPending.toNat
  | .permutationRound => image.permutationRound
  | .permutationState index => image.permutationState index
  | .grammarMode => image.grammarMode.toNat
  | .grammarTurnExportFunction => image.grammarTurnExportFunction
  | .grammarEventsRemaining => image.grammarEventsRemaining
  | .grammarEventIndex => image.grammarEventIndex
  | .grammarArgumentsBase => image.grammarArgumentsBase
  | .grammarSlotCursor => image.grammarSlotCursor

/-- The complete tagged field map retains every typed state component. -/
theorem Image.fieldValue_injective :
    Function.Injective Image.fieldValue := by
  intro left right equal
  apply Image.ext
  · exact congrFun equal .pc
  · exact congrFun equal .operandStackPointer
  · exact congrFun equal .stackFrameBase
  · exact boolToNat_injective (congrFun equal .outputEnabled)
  · exact congrFun equal .outputLow
  · exact congrFun equal .outputHigh
  · exact congrFun equal .callStackDepth
  · exact boolToNat_injective (congrFun equal .memoryPagesPresent)
  · exact congrFun equal .memoryPagesValue
  · exact boolToNat_injective
      (congrFun equal .maximumMemoryPagesPresent)
  · exact congrFun equal .maximumMemoryPagesValue
  · exact congrFun equal .localsFrameBase
  · exact boolToNat_injective (congrFun equal .halted)
  · exact boolToNat_injective (congrFun equal .trapped)
  · exact congrFun equal .trapCode
  · exact boolToNat_injective
      (congrFun equal .parameterInitializationActive)
  · exact congrFun equal .parameterInitializationRemaining
  · exact boolToNat_injective (congrFun equal .tailCallPending)
  · exact boolToNat_injective (congrFun equal .hostArgumentsActive)
  · exact congrFun equal .hostArgumentsRemaining
  · exact boolToNat_injective (congrFun equal .hostResultPending)
  · exact congrFun equal .hostCalleeFunction
  · funext index
    exact congrFun equal (.hostEventChain index)
  · funext index
    exact congrFun equal (.eventBuffer index)
  · exact congrFun equal .eventBufferSlot
  · exact boolToNat_injective (congrFun equal .permutationPending)
  · exact congrFun equal .permutationRound
  · funext index
    exact congrFun equal (.permutationState index)
  · exact boolToNat_injective (congrFun equal .grammarMode)
  · exact congrFun equal .grammarTurnExportFunction
  · exact congrFun equal .grammarEventsRemaining
  · exact congrFun equal .grammarEventIndex
  · exact congrFun equal .grammarArgumentsBase
  · exact congrFun equal .grammarSlotCursor

/-- Canonical field order. Indexed arrays are in increasing `Fin` order. -/
def schema : List FieldTag :=
  [ .pc
  , .operandStackPointer
  , .stackFrameBase
  , .outputEnabled
  , .outputLow
  , .outputHigh
  , .callStackDepth
  , .memoryPagesPresent
  , .memoryPagesValue
  , .maximumMemoryPagesPresent
  , .maximumMemoryPagesValue
  , .localsFrameBase
  , .halted
  , .trapped
  , .trapCode
  , .parameterInitializationActive
  , .parameterInitializationRemaining
  , .tailCallPending
  , .hostArgumentsActive
  , .hostArgumentsRemaining
  , .hostResultPending
  , .hostCalleeFunction
  ] ++
  List.ofFn FieldTag.hostEventChain ++
  List.ofFn FieldTag.eventBuffer ++
  [ .eventBufferSlot
  , .permutationPending
  , .permutationRound
  ] ++
  List.ofFn FieldTag.permutationState ++
  [ .grammarMode
  , .grammarTurnExportFunction
  , .grammarEventsRemaining
  , .grammarEventIndex
  , .grammarArgumentsBase
  , .grammarSlotCursor
  ]

def fieldCount : Nat := schema.length

def serializedBitCount : Nat :=
  (schema.map FieldTag.bitWidth).sum

theorem fieldCount_eq : fieldCount = 55 := by
  decide

theorem serializedBitCount_eq : serializedBitCount = 2293 := by
  decide

/-- Canonical values are exactly those whose reconstructed semantic state is
valid. This retains optional-value and inactive-countdown zero rules. -/
def Image.Canonical (image : Image) : Prop :=
  (decode image).Valid

theorem canonical_encode_iff (state : AppStateVector) :
    (encode state).Canonical ↔ state.Valid := by
  simp [Image.Canonical, decode_encode]

end Nightstream.Protocol.Nebula.WasmStateEncoding
