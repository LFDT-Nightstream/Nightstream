import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonValuePlacement

/-!
Contract: exact fresh-word values for each production PiRLC Poseidon2 replay
call.

Assurance tier: artifact-checked same-assignment value placement.

Owns: projection of one fresh call-input lane to the exact normalized input
or output word selected by the Rust-emitted call run.

Does not own: selector activation, emitted-row satisfaction, carried-state
placement, call chaining, complete replay, or lifecycle soundness.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 65536

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonRunValues

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Core
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonCallFamily
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonCallProjection
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonCompactTrace
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonLeafModel
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonLeafReconstruction
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonValuePlacement

def rateLane (lane : Fin 4) : Fin width :=
  ⟨lane.val, by
    have bounded := lane.isLt
    change lane.val < 4 at bounded
    change lane.val < 8
    omega⟩

private theorem sourceInput_rateLane
    (source : SourceAssignment) (lane : Fin 4) :
    sourceInput source (rateLane lane) = source.externalA lane := by
  unfold sourceInput rateLane
  simp only [Fin.val_mk]
  rw [dif_pos lane.isLt]

private theorem sourceFor_externalA
    (kind : LeafClass) (final : FinalAssignment) (lane : Fin 4) :
    sourceInput (sourceFor kind final) (rateLane lane) =
      slotValue final (.externalA lane) := by
  rw [sourceInput_rateLane]
  cases kind with
  | direct =>
      rfl
  | partialStart =>
      rfl
  | chained selector =>
      change
        portAction
            (Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonChainedLeafReconstruction.unitSlotPort
              (.externalA lane)) final =
          slotValue final (.externalA lane)
      simp [
        Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonChainedLeafReconstruction.unitSlotPort,
        portAction, sum, slotValue]

theorem externalASlotValue_eq_wordValue
    (site : CallSite) (assignment : Fin productionFinalColumns → F)
    (lane : Fin 4) (start : Nat)
    (owned : site.externalASlotStart lane = some start)
    (fits : start + 41 ≤ productionFinalColumns) :
    slotValue (projectFinalAssignment site assignment) (.externalA lane) =
      wordValue assignment start fits := by
  apply slotValue_eq_wordValue_of_digits
  intro digit
  have columnOwned :
      digitColumn site (.externalA lane) digit =
        some (start + digit.val) := by
    simp [digitColumn, owned]
  rw [projected_digit_of_some site assignment (.externalA lane) digit
    (start + digit.val) columnOwned]
  rw [absoluteValue_of_lt]

private theorem inputWord_fits_at (ordinal : Fin 918) :
    38340 + ordinal.val * 41 + 41 ≤ productionFinalColumns := by
  have upper := ordinal.isLt
  change ordinal.val < 918 at upper
  change 38340 + ordinal.val * 41 + 41 ≤ 8858862
  omega

private theorem outputWord_fits_at (ordinal : Fin 54) :
    75978 + ordinal.val * 41 + 41 ≤ productionFinalColumns := by
  have upper := ordinal.isLt
  change ordinal.val < 54 at upper
  change 75978 + ordinal.val * 41 + 41 ≤ 8858862
  omega

/-- A fresh rate lane in either exact input replay run reads the corresponding
word of the same 918-word normalized algebra frame. -/
theorem inputRun_freshValue_exact
    (run : Run) (selected : run = evenInputRun ∨ run = oddInputRun)
    (index : Fin run.raw.callCount) (lane : Fin 4) (ordinal : Nat)
    (fresh : (run.callSiteAt index.val).freshOrdinal lane = some ordinal)
    (assignment : Fin productionFinalColumns → F) :
    sourceInput
        (sourceFor (run.leafClassAt index.val)
          (projectFinalAssignment (run.callSiteAt index.val) assignment))
        (rateLane lane) =
      wordValue assignment (38340 + ordinal * 41)
        (by
          rcases inputRun_freshSlot_exact run selected index lane ordinal fresh with
            ⟨bounded, boundedValue, _⟩
          subst ordinal
          exact inputWord_fits_at bounded) := by
  rcases inputRun_freshSlot_exact run selected index lane ordinal fresh with
    ⟨bounded, boundedValue, slotExact⟩
  subst ordinal
  rw [sourceFor_externalA]
  exact externalASlotValue_eq_wordValue
    (run.callSiteAt index.val) assignment lane
    (38340 + bounded.val * 41) slotExact (inputWord_fits_at bounded)

/-- A fresh rate lane in either exact output replay run reads the corresponding
word of the same 54-word normalized algebra output frame. -/
theorem outputRun_freshValue_exact
    (run : Run) (selected : run = evenOutputRun ∨ run = oddOutputRun)
    (index : Fin run.raw.callCount) (lane : Fin 4) (ordinal : Nat)
    (fresh : (run.callSiteAt index.val).freshOrdinal lane = some ordinal)
    (assignment : Fin productionFinalColumns → F) :
    sourceInput
        (sourceFor (run.leafClassAt index.val)
          (projectFinalAssignment (run.callSiteAt index.val) assignment))
        (rateLane lane) =
      wordValue assignment (75978 + ordinal * 41)
        (by
          rcases outputRun_freshSlot_exact run selected index lane ordinal fresh with
            ⟨bounded, boundedValue, _⟩
          subst ordinal
          exact outputWord_fits_at bounded) := by
  rcases outputRun_freshSlot_exact run selected index lane ordinal fresh with
    ⟨bounded, boundedValue, slotExact⟩
  subst ordinal
  rw [sourceFor_externalA]
  exact externalASlotValue_eq_wordValue
    (run.callSiteAt index.val) assignment lane
    (75978 + bounded.val * 41) slotExact (outputWord_fits_at bounded)

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonRunValues
