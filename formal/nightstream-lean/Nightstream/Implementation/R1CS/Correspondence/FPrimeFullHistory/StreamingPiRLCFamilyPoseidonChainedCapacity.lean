import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonRunValues

/-!
Contract: exact capacity-lane image for one chained production PiRLC
Poseidon2 call.

Assurance tier: artifact-checked same-assignment value placement.

Owns: structural equality between the four Rust-emitted prior-output source
images and Lean's compact Poseidon2 output forms for lanes four through seven.

Does not own: emitted-row satisfaction, call-result correctness, complete
call chaining, final replay state placement, or lifecycle soundness.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 65536

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonChainedCapacity

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonLeafSchema
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Core
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonChainedLeafReconstruction
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonCallFamily
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonCallProjection
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonCompactTrace
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonLeafModel
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonLeafReconstruction
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonRunValues
open Nightstream.Implementation.R1CS.Poseidon2CompactTraceRefinement

private abbrev rawImages :=
  Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonChainedLeaf.rawImages

def capacityLane (lane : Fin 4) : Fin width :=
  ⟨4 + lane.val, by
    have bounded := lane.isLt
    change lane.val < 4 at bounded
    change 4 + lane.val < 8
    omega⟩

/-- The final compact forms use physical columns 564, 568, ..., 592. The
matching generated source images store them as local slots 78, 79, ..., 85.
Rust emits each sparse port in reverse operand order. -/
def rawRunOfFinalTerm (term : Nat × Nat) : RawGeometricRun where
  slot := .previousLocal (78 + (term.1 - 564) / 4)
  initial := term.2
  ratio := 3

def expectedRawCapacityPort (lane : Fin 4) : RawPort where
  explicit := []
  geometric :=
    (traceFinalForm (capacityLane lane)).reverse.map rawRunOfFinalTerm

def rawSourceTermOfFinalTerm (term : Nat × Nat) : RawSourceTerm where
  column := .local (term.1 - 9)
  coefficient := term.2

def expectedRawCapacitySource
    (lane : Fin 4) : RawSourceLinearCombination where
  constant := 0
  terms :=
    (traceFinalForm (capacityLane lane)).reverse.map
      rawSourceTermOfFinalTerm

def emptySourceValue : Wire.SourceLinearCombination where
  constant := 0
  terms := []

def expectedCapacitySource
    (lane : Fin 4) : Wire.SourceLinearCombination :=
  (Wire.decodeSourceLinearCombination
    (expectedRawCapacitySource lane)).getD emptySourceValue

/-- Exact trust-boundary comparison. This checks four small generated source
images against the independent Lean compact-trace forms. -/
theorem raw_capacity_images_exact :
    rawImages.map (fun image => image.port) =
      List.ofFn expectedRawCapacityPort := by
  rfl

/-- Decoding the generated image gives the same typed port as decoding the
independent expected image. -/
theorem priorImagePort_eq_expected (lane : Fin 4) :
    priorImagePort lane =
      (Wire.decodePort (expectedRawCapacityPort lane)).getD emptyPort := by
  fin_cases lane <;> rfl

/-- A chained call's prior-local digit and the preceding call's current-local
digit are the same absolute production assignment coordinate. -/
theorem projected_previousLocal_eq_prior_local
    (run : Run) (index : Nat)
    (assignment : Fin productionFinalColumns → F)
    (slot : Fin 86) (digit : Fin 41) :
    (projectFinalAssignment (run.callSiteAt index.succ) assignment).digit
        (.previousLocal slot) digit =
      (projectFinalAssignment (run.callSiteAt index) assignment).digit
        (.local slot) digit := by
  simp only [projectFinalAssignment, digitColumn, Run.callSiteAt,
    CallSite.previousLocalSlotStart, Run.localFinalAt, Run.leafClassAt,
    Nat.succ_ne_zero, if_false, Option.map_some]
  rw [Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonCallLayout.localFinalStride_exact]
  congr 1
  norm_num [localSlotCount, slotWidth]
  omega

/-- Radix-3 decoding preserves the exact prior-call coordinate equality. -/
theorem previousLocal_slotValue_eq_prior_local
    (run : Run) (index : Nat)
    (assignment : Fin productionFinalColumns → F) (slot : Fin 86) :
    slotValue
        (projectFinalAssignment (run.callSiteAt index.succ) assignment)
        (.previousLocal slot) =
      slotValue
        (projectFinalAssignment (run.callSiteAt index) assignment)
        (.local slot) := by
  unfold slotValue geometricAction
  congr 2
  funext digit
  rw [projected_previousLocal_eq_prior_local]

/-- Small reusable leaf certificate for the eight terminal S-box owners. -/
theorem terminal_sourceSlot (slot : Fin 8) :
    sourceSlot
        Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonLeafCertificate.decodedSteps
        (.local ⟨555 + 4 * slot.val, by
          have bounded := slot.isLt
          change slot.val < 8 at bounded
          omega⟩) =
      some (.local ⟨78 + slot.val, by
        have bounded := slot.isLt
        change slot.val < 8 at bounded
        omega⟩) := by
  fin_cases slot <;> rfl

private theorem sourceFor_localValue_of_sourceSlot
    (kind : LeafClass) (final : FinalAssignment)
    (offset : Fin 600) (slot : Wire.Slot)
    (owned : sourceSlot
      Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonLeafCertificate.decodedSteps
      (.local offset) = some slot) :
    (sourceFor kind final).localValue offset = slotValue final slot := by
  cases kind with
  | direct =>
      change
        (match sourceSlot
            Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonLeafCertificate.decodedSteps
            (.local offset) with
          | some owner => slotValue final owner
          | none => 0) = slotValue final slot
      rw [owned]
  | partialStart =>
      change
        (match sourceSlot
            Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonLeafCertificate.decodedSteps
            (.local offset) with
          | some owner => slotValue final owner
          | none => 0) = slotValue final slot
      rw [owned]
  | chained selector =>
      change
        portAction
            (match sourceSlot
                Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonLeafCertificate.decodedSteps
                (.local offset) with
              | some owner => unitSlotPort owner
              | none => emptyPort) final =
          slotValue final slot
      rw [owned]
      simp [unitSlotPort, portAction, sum, slotValue]

/-- The eight terminal compact columns are exactly local slots 78 through 85
for every leaf reconstruction class. -/
theorem sourcePhysical_terminalSlot
    (kind : LeafClass) (final : FinalAssignment) (slot : Fin 8) :
    sourcePhysical (sourceFor kind final) (564 + 4 * slot.val) =
      (slotValue final
        (.local ⟨78 + slot.val, by
          have bounded := slot.isLt
          change slot.val < 8 at bounded
          omega⟩)).val := by
  let offset : Fin 600 := ⟨555 + 4 * slot.val, by
    have bounded := slot.isLt
    change slot.val < 8 at bounded
    omega⟩
  let terminal : Fin 86 := ⟨78 + slot.val, by
    have bounded := slot.isLt
    change slot.val < 8 at bounded
    omega⟩
  have columnExact :
      564 + 4 * slot.val = sourceColumnIndex (.local offset) := by
    simp only [sourceColumnIndex]
    dsimp only [offset]
    omega
  have owned : sourceSlot
      Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonLeafCertificate.decodedSteps
      (.local offset) = some (.local terminal) := by
    dsimp only [offset, terminal]
    exact terminal_sourceSlot slot
  rw [columnExact, sourcePhysical_sourceColumn]
  exact congrArg Fin.val
    (sourceFor_localValue_of_sourceSlot kind final offset (.local terminal)
      owned)

/-- Typed source-value form of the same terminal-slot ownership fact. -/
theorem sourceValue_terminalSlot
    (kind : LeafClass) (final : FinalAssignment) (slot : Fin 8) :
    sourceValue (sourceFor kind final)
        (.local ⟨555 + 4 * slot.val, by
          have bounded := slot.isLt
          change slot.val < 8 at bounded
          omega⟩) =
      slotValue final
        (.local ⟨78 + slot.val, by
          have bounded := slot.isLt
          change slot.val < 8 at bounded
          omega⟩) := by
  apply sourceFor_localValue_of_sourceSlot
  exact terminal_sourceSlot slot

/-- One generated prior-local run has the value of the same terminal slot in
the preceding call projection. -/
theorem previousTerminalRun_action
    (run : Run) (index : Nat)
    (assignment : Fin productionFinalColumns → F)
    (slot : Fin 86) (coefficient : F) :
    geometricAction
        { slot := .previousLocal slot, initial := coefficient, ratio := 3 }
        (projectFinalAssignment (run.callSiteAt index.succ) assignment) =
      coefficient * slotValue
        (projectFinalAssignment (run.callSiteAt index) assignment)
        (.local slot) := by
  rw [geometricAction_eq_scaled_slotValue _ _ rfl,
    previousLocal_slotValue_eq_prior_local]

/-- The compact final-form evaluator and the typed capacity source form have
the same value. The only order difference is an exact list reversal. -/
theorem traceFinalForm_eval_eq_capacitySource
    (kind : LeafClass) (final : FinalAssignment) (lane : Fin 4) :
    lcEval (sourcePhysical (sourceFor kind final))
        (traceFinalForm (capacityLane lane)) =
      (sourceAction (expectedCapacitySource lane)
        (sourceFor kind final)).val := by
  calc
    lcEval (sourcePhysical (sourceFor kind final))
        (traceFinalForm (capacityLane lane)) =
        lcEval (sourcePhysical (sourceFor kind final))
          (traceFinalForm (capacityLane lane)).reverse :=
      lcEval_eq_of_perm _
        (List.reverse_perm (traceFinalForm (capacityLane lane))).symm
    _ = lcEval (sourcePhysical (sourceFor kind final))
          (sourceTerms (expectedCapacitySource lane)) := by
      fin_cases lane <;> rfl
    _ = (sourceAction (expectedCapacitySource lane)
          (sourceFor kind final)).val :=
      lcEval_sourceTerms _ _

private def capacityTerms (lane : Fin 4) : List (Fin 8 × F) :=
  match lane.val with
  | 0 => [(7, 2), (6, 2), (5, 6), (4, 4), (3, 1), (2, 1), (1, 3), (0, 2)]
  | 1 => [(7, 2), (6, 6), (5, 4), (4, 2), (3, 1), (2, 3), (1, 2), (0, 1)]
  | 2 => [(7, 6), (6, 4), (5, 2), (4, 2), (3, 3), (2, 2), (1, 1), (0, 1)]
  | _ => [(7, 4), (6, 2), (5, 2), (4, 6), (3, 2), (2, 1), (1, 1), (0, 3)]

private def capacitySourceTerm (term : Fin 8 × F) : Wire.SourceTerm where
  column := .local ⟨555 + 4 * term.1.val, by
    have bounded := term.1.isLt
    change term.1.val < 8 at bounded
    omega⟩
  coefficient := term.2

private def capacityRun (term : Fin 8 × F) : Wire.GeometricRun where
  slot := .previousLocal ⟨78 + term.1.val, by
    have bounded := term.1.isLt
    change term.1.val < 8 at bounded
    omega⟩
  initial := term.2
  ratio := 3

private def typedCapacitySource (lane : Fin 4) : Wire.SourceLinearCombination where
  constant := 0
  terms := (capacityTerms lane).map capacitySourceTerm

private def typedCapacityPort (lane : Fin 4) : Wire.Port where
  explicit := []
  geometric := (capacityTerms lane).map capacityRun

/-- Small leaf certificate: the independently decoded compact output form is
the typed four-by-eight capacity table. -/
private theorem expectedCapacitySource_eq_typed (lane : Fin 4) :
    expectedCapacitySource lane = typedCapacitySource lane := by
  fin_cases lane <;> rfl

/-- Small leaf certificate: the Rust-emitted prior image is the same typed
four-by-eight capacity table. -/
private theorem priorImagePort_eq_typed (lane : Fin 4) :
    priorImagePort lane = typedCapacityPort lane := by
  fin_cases lane <;> rfl

private theorem capacityTerms_realized
    (run : Run) (index : Nat)
    (assignment : Fin productionFinalColumns → F) :
    ∀ terms : List (Fin 8 × F),
      List.Forall₂
        (fun term geometric =>
          geometricAction geometric
              (projectFinalAssignment (run.callSiteAt index.succ) assignment) =
            term.coefficient * sourceValue
              (sourceFor (run.leafClassAt index)
                (projectFinalAssignment (run.callSiteAt index) assignment))
              term.column)
        (terms.map capacitySourceTerm) (terms.map capacityRun)
  | [] => .nil
  | term :: tail => by
      apply List.Forall₂.cons
      · simp only [capacitySourceTerm, capacityRun]
        rw [previousTerminalRun_action, sourceValue_terminalSlot]
      · exact capacityTerms_realized run index assignment tail

/-- The generated capacity port realizes the independent typed source form
term by term. -/
theorem capacityPort_realized
    (run : Run) (index : Nat)
    (assignment : Fin productionFinalColumns → F) (lane : Fin 4) :
    PortRealized (expectedCapacitySource lane) (priorImagePort lane)
      (sourceFor (run.leafClassAt index)
        (projectFinalAssignment (run.callSiteAt index) assignment))
      (projectFinalAssignment (run.callSiteAt index.succ) assignment) := by
  rw [expectedCapacitySource_eq_typed, priorImagePort_eq_typed]
  refine { constant := rfl, terms := ?_ }
  exact capacityTerms_realized run index assignment (capacityTerms lane)

/-- The expected generated prior-output port and the typed capacity source
form evaluate to the same field value on adjacent call projections. -/
theorem expectedCapacityPort_action_eq_sourceAction
    (run : Run) (index : Nat)
    (assignment : Fin productionFinalColumns → F) (lane : Fin 4) :
    portAction
        ((Wire.decodePort (expectedRawCapacityPort lane)).getD emptyPort)
        (projectFinalAssignment (run.callSiteAt index.succ) assignment) =
      sourceAction (expectedCapacitySource lane)
        (sourceFor (run.leafClassAt index)
          (projectFinalAssignment (run.callSiteAt index) assignment)) := by
  rw [← priorImagePort_eq_expected]
  exact portRealized_action (capacityPort_realized run index assignment lane)

/-- In a chained leaf, capacity input lane `4 + lane` is the corresponding
decoded prior-output image. -/
theorem chained_sourceInput_capacity
    (selector : Nat) (final : FinalAssignment) (lane : Fin 4) :
    sourceInput (sourceFor (.chained selector) final) (capacityLane lane) =
      portAction (priorImagePort lane) final := by
  fin_cases lane <;> rfl

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonChainedCapacity
