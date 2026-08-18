import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonLeafSchema

/-! Generated file: exact semantic-to-physical layout of every production
PiRLC replay Poseidon2 call.

Owns: source-arm identity, input/output scope, first-call class, exact source
and emitted row runs, selectors, and source-to-final slot placement.

Owns also: the canonical normalized 86-row compact trace shared by all calls
up to linear-combination operand permutation and bound through the exact
selective rewrite and decoder layout.

Does not own: Poseidon2 semantics, lifecycle
soundness, or permission to remove constraints.

Emits constraints: no.
-/

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonCallLayout

inductive RawScope where
  | input
  | output
  deriving DecidableEq, Repr

inductive RawFirstClass where
  | direct
  | partialStart
  deriving DecidableEq, Repr

structure RawRun where
  arm : Nat
  scope : RawScope
  callCount : Nat
  firstClass : RawFirstClass
  selectorColumn : Nat
  sourceRowStart : Nat
  emittedRowStart : Nat
  firstFreshCount : Nat
  freshSourceStart : Nat
  freshFinalStart : Nat
  initialCarriedSourceStart : Option Nat
  initialCarriedFinalStart : Option Nat
  initialCapacitySourceStart : Nat
  initialCapacityFinalStart : Nat
  localSourceStart : Nat
  localFinalStart : Nat
  previousCapacitySourceOffset : Nat
  deriving DecidableEq, Repr

def schemaVersion : Nat := 2
def rowTemplateSource : String := "rust:nightstream/streaming-pi-rlc-family/poseidon2-normalized-row-template/v1"
def sourceCallStride : Nat := 600
def emittedCallRows : Nat := 86
def slotWidth : Nat := 41
def localFinalStride : Nat := 3526
def evenSourceRows : Nat := 1300897
def evenSourceColumns : Nat := 1301126
def oddSourceRows : Nat := 1302097
def oddSourceColumns : Nat := 1302326
def finalRows : Nat := 491046
def finalColumns : Nat := 8858862

def rawRun0 : RawRun where
  arm := 0
  scope := .input
  callCount := 229
  firstClass := .direct
  selectorColumn := 648
  sourceRowStart := 165446
  emittedRowStart := 74375
  firstFreshCount := 4
  freshSourceStart := 1559
  freshFinalStart := 38340
  initialCarriedSourceStart := none
  initialCarriedFinalStart := none
  initialCapacitySourceStart := 166308
  initialCapacityFinalStart := 2217933
  localSourceStart := 166320
  localFinalStart := 2218425
  previousCapacitySourceOffset := 596

def rawRun1 : RawRun where
  arm := 0
  scope := .output
  callCount := 13
  firstClass := .direct
  selectorColumn := 648
  sourceRowStart := 302846
  emittedRowStart := 94069
  firstFreshCount := 4
  freshSourceStart := 2477
  freshFinalStart := 75978
  initialCarriedSourceStart := none
  initialCarriedFinalStart := none
  initialCapacitySourceStart := 166316
  initialCapacityFinalStart := 2218261
  localSourceStart := 303720
  localFinalStart := 3025879
  previousCapacitySourceOffset := 596

def rawRun2 : RawRun where
  arm := 1
  scope := .input
  callCount := 230
  firstClass := .partialStart
  selectorColumn := 649
  sourceRowStart := 165446
  emittedRowStart := 309886
  firstFreshCount := 2
  freshSourceStart := 1559
  freshFinalStart := 38340
  initialCarriedSourceStart := some 166304
  initialCarriedFinalStart := some 2217769
  initialCapacitySourceStart := 166308
  initialCapacityFinalStart := 2217933
  localSourceStart := 166320
  localFinalStart := 2218425
  previousCapacitySourceOffset := 596

def rawRun3 : RawRun where
  arm := 1
  scope := .output
  callCount := 14
  firstClass := .partialStart
  selectorColumn := 649
  sourceRowStart := 303446
  emittedRowStart := 329666
  firstFreshCount := 2
  freshSourceStart := 2477
  freshFinalStart := 75978
  initialCarriedSourceStart := some 166312
  initialCarriedFinalStart := some 2218097
  initialCapacitySourceStart := 166316
  initialCapacityFinalStart := 2218261
  localSourceStart := 304320
  localFinalStart := 3029405
  previousCapacitySourceOffset := 596

def rawRuns : List RawRun := [rawRun0, rawRun1, rawRun2, rawRun3]

end Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonCallLayout
