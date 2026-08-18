import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonCallLayout

/-!
Contract: exact leaf certificate for the four Rust-emitted production PiRLC
replay-call runs.

Assurance tier: artifact-checked for the Nightstream b2/k16 profile.

Owns exact geometry, parity and scope ownership, first-call classes, selectors,
source and emitted row starts, fresh-input slots, initial state slots, and
local Poseidon2 slot runs. It does not prove row semantics or lifecycle
authority.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonCallLayout

def expectedEvenInput : RawRun where
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

def expectedEvenOutput : RawRun where
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

def expectedOddInput : RawRun where
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

def expectedOddOutput : RawRun where
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

theorem schemaVersion_exact : schemaVersion = 2 := by rfl
theorem rowTemplateSource_exact :
    rowTemplateSource =
      "rust:nightstream/streaming-pi-rlc-family/poseidon2-normalized-row-template/v1" := by
  rfl
theorem sourceCallStride_exact : sourceCallStride = 600 := by rfl
theorem emittedCallRows_exact : emittedCallRows = 86 := by rfl
theorem slotWidth_exact : slotWidth = 41 := by rfl
theorem localFinalStride_exact : localFinalStride = 86 * 41 := by rfl
theorem evenSourceRows_exact : evenSourceRows = 1300897 := by rfl
theorem evenSourceColumns_exact : evenSourceColumns = 1301126 := by rfl
theorem oddSourceRows_exact : oddSourceRows = 1302097 := by rfl
theorem oddSourceColumns_exact : oddSourceColumns = 1302326 := by rfl
theorem finalRows_exact : finalRows = 491046 := by rfl
theorem finalColumns_exact : finalColumns = 8858862 := by rfl

theorem rawRun0_exact : rawRun0 = expectedEvenInput := by rfl
theorem rawRun1_exact : rawRun1 = expectedEvenOutput := by rfl
theorem rawRun2_exact : rawRun2 = expectedOddInput := by rfl
theorem rawRun3_exact : rawRun3 = expectedOddOutput := by rfl

theorem rawRuns_exact :
    rawRuns =
      [expectedEvenInput, expectedEvenOutput,
        expectedOddInput, expectedOddOutput] := by
  rfl

theorem rawRuns_length : rawRuns.length = 4 := by
  rw [rawRuns_exact]
  rfl

end Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonCallLayout
