import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiRLCFamilyPhysicalOverlayRows
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPiRLCFamilyXOutPreimage

/-!
Contract: same-assignment semantic adapter for one physical PiRLC family arm.

Owns the conversion from a family ordinal to its parity body, direct decoding
of both semantic `FamilyState` values from that body assignment, and the joint
result of body rows, one linked physical overlay, the public-state suffix, and
the structural full-XOut preimage fields.

Does not own normalized selective lowering, the 400-arm schedule, collision
resistance, authority for opaque outer XOut fields, recursive lifecycle
integration, or terminal verification.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 524288
set_option maxHeartbeats 8000000

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPhysicalState

open Nightstream.Implementation.Nebula
open Nightstream.Implementation.Nebula.ProductPiRlcRingCombinationSound
open Nightstream.Implementation.Nebula.ProductionStreamingPiRlcAuthority
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyDigest
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublicArtifact
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublicState
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyState
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyReplayArtifact
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyXOutPreimage
open Nightstream.SuperNeo.Concrete

private abbrev sourceLayout :=
  Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilyCompleteRows.layout

private abbrev parityFor :=
  Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilyCompleteRows.parityFor

def kindForParity : CursorParity → ArmKind
  | .even => .even
  | .odd => .odd

def kindForFamily (family : Family) : ArmKind :=
  kindForParity (parityFor family)

@[simp] theorem parityForArm_kindForParity (parity : CursorParity) :
    parityForArm (kindForParity parity) = parity := by
  cases parity <;> rfl

@[simp] theorem parityForArm_kindForFamily (family : Family) :
    parityForArm (kindForFamily family) = parityFor family := by
  simp [kindForFamily]

/-- Accepted evidence for one complete physical family arm. The body and
public suffix use the same assignment. The physical overlay has its own
assignment and joins the body only through all exact field links. -/
structure AcceptedArm (setup : InputBindingSetup) (family : Family) where
  bodyAssignment : Nat → Nat
  overlayAssignment : Nat → Nat
  bodyCanonical : ∀ column, bodyAssignment column < goldilocksP
  overlayCanonical : ∀ column, overlayAssignment column < goldilocksP
  bodyOne : bodyAssignment 0 = 1
  overlayOne : overlayAssignment 0 = 1
  range : ∀ source lane,
    bodyAssignment (sourceLayout.algebra.challengeSymbol source lane) < 5
  cursorExact :
    (familyStateAt bodyAssignment bodyCanonical (kindForFamily family)
      .before).familyCursor =
        ProductPiRlcAlgebraRows.familyOrdinal family
  links :
    Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilyPhysicalOverlayRows.FieldLinksHold
      bodyAssignment overlayAssignment
  bodySatisfied : Satisfies
    (Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilyBodyOverlayRows.bodyRowsForParity
      (parityFor family)) bodyAssignment
  overlaySatisfied : Satisfies
    (Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilyPhysicalOverlayRows.rows
      setup family) overlayAssignment
  suffixSatisfied :
    (armFor (kindForFamily family)).Satisfied bodyAssignment

namespace AcceptedArm

def beforeState {setup : InputBindingSetup} {family : Family}
    (accepted : AcceptedArm setup family) : FamilyState :=
  familyStateAt accepted.bodyAssignment accepted.bodyCanonical
    (kindForFamily family) .before

def afterState {setup : InputBindingSetup} {family : Family}
    (accepted : AcceptedArm setup family) : FamilyState :=
  familyStateAt accepted.bodyAssignment accepted.bodyCanonical
    (kindForFamily family) .after

private theorem decoded_replay_states_placed
    {setup : InputBindingSetup} {family : Family}
    (accepted : AcceptedArm setup family) :
    FamilyStatesPlaced (parityFor family) accepted.bodyAssignment
      accepted.beforeState accepted.afterState := by
  simpa [beforeState, afterState] using
    replay_states_placed (kindForFamily family) accepted.bodyAssignment
      accepted.bodyCanonical

private theorem decoded_residuals_placed
    {setup : InputBindingSetup} {family : Family}
    (accepted : AcceptedArm setup family) :
    Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputFamilyRows.ResidualsPlaced
      sourceLayout.input accepted.bodyAssignment
      accepted.beforeState.inputResidual accepted.afterState.inputResidual := by
  simpa [beforeState, afterState] using
    residuals_placed (kindForFamily family) accepted.bodyAssignment
      accepted.bodyCanonical

private theorem decoded_carry_state_placed
    {setup : InputBindingSetup} {family : Family}
    (accepted : AcceptedArm setup family) :
    Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilyCarryRows.StateColumnsPlaced
      (Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilySourceRows.carryLayout
        sourceLayout)
      accepted.bodyAssignment accepted.beforeState accepted.afterState := by
  simpa [beforeState, afterState] using
    carry_state_placed (kindForFamily family) accepted.bodyAssignment
      accepted.bodyCanonical

/-- The exact input rings decoded from the accepted family body. -/
def phaseInputs
    {setup : InputBindingSetup} {family : Family}
    (accepted : AcceptedArm setup family) : Source → RingF :=
  decodedInputs sourceLayout.algebra accepted.bodyAssignment
    accepted.bodyCanonical

/-- The exact output ring decoded from the accepted family body. -/
def phaseOutput
    {setup : InputBindingSetup} {family : Family}
    (accepted : AcceptedArm setup family) : RingF :=
  outputRing sourceLayout.algebra accepted.bodyAssignment
    accepted.bodyCanonical

/-- The accepted body and linked overlay imply the exact semantic family
phase for their directly decoded input and output rings. -/
theorem phase
    {setup : InputBindingSetup} {family : Family}
    (accepted : AcceptedArm setup family) :
    FamilyPhaseRelation setup accepted.beforeState accepted.afterState
      family accepted.phaseInputs accepted.phaseOutput := by
  exact
    Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilyPhysicalOverlayRows.rows_sound
      accepted.bodyCanonical accepted.overlayCanonical accepted.bodyOne
      accepted.overlayOne accepted.range
      (decoded_residuals_placed accepted)
      (decoded_carry_state_placed accepted)
      (decoded_replay_states_placed accepted)
      accepted.cursorExact accepted.links accepted.bodySatisfied
      accepted.overlaySatisfied

private theorem before_cursor_bound
    {setup : InputBindingSetup} {family : Family}
    (accepted : AcceptedArm setup family) :
    accepted.beforeState.familyCursor < 111 := by
  have cursorExact : accepted.beforeState.familyCursor =
      ProductPiRlcAlgebraRows.familyOrdinal family := by
    simpa [beforeState] using accepted.cursorExact
  rw [cursorExact]
  exact Nat.lt_trans
    (ProductPiRlcAlgebraRows.familyOrdinal_lt family) (by decide)

private theorem after_cursor_bound
    {setup : InputBindingSetup} {family : Family}
    (accepted : AcceptedArm setup family) :
    accepted.afterState.familyCursor < 111 := by
  have cursorExact : accepted.beforeState.familyCursor =
      ProductPiRlcAlgebraRows.familyOrdinal family := by
    simpa [beforeState] using accepted.cursorExact
  rw [accepted.phase.2.2.cursor, cursorExact]
  have ordinalBound := ProductPiRlcAlgebraRows.familyOrdinal_lt family
  omega

/-- The accepted public suffix binds all ten public words to the two decoded
semantic states. -/
theorem publicBinding
    {setup : InputBindingSetup} {family : Family}
    (accepted : AcceptedArm setup family) :
    Binding (kindForFamily family) accepted.bodyAssignment
      accepted.bodyCanonical := by
  simpa [beforeState, afterState] using
    shared_public_state_refines (kindForFamily family)
      accepted.bodyAssignment accepted.bodyCanonical accepted.bodyOne
      accepted.suffixSatisfied (before_cursor_bound accepted)
      (after_cursor_bound accepted)

/-- The accepted suffix fixes the exact structural fields of either full-XOut
preimage. The five outer four-field values remain opaque at this layer. -/
theorem xOutPreimageBinding
    {setup : InputBindingSetup} {family : Family}
    (accepted : AcceptedArm setup family) (side : StateSide) :
    PreimageBinding (kindForFamily family) side accepted.bodyAssignment
      accepted.bodyCanonical := by
  simpa [beforeState, afterState] using
    x_out_preimage_refines (kindForFamily family) side
      accepted.bodyAssignment accepted.bodyCanonical accepted.bodyOne
      accepted.suffixSatisfied (before_cursor_bound accepted)
      (after_cursor_bound accepted)

/-- The accepted body, public suffix, and linked overlay imply one exact
family phase and the complete meaning of all ten public words. No digest is
used as an algebraic authority premise. -/
theorem sound
    {setup : InputBindingSetup} {family : Family}
    (accepted : AcceptedArm setup family) :
    ∃ inputs output,
      FamilyPhaseRelation setup accepted.beforeState accepted.afterState
          family inputs output ∧
        Binding (kindForFamily family) accepted.bodyAssignment
          accepted.bodyCanonical :=
  ⟨accepted.phaseInputs, accepted.phaseOutput, accepted.phase,
    accepted.publicBinding⟩

/-- Joint physical-arm result with both the semantic phase and the exact
structural meaning of each full-XOut preimage. -/
theorem soundWithXOutPreimage
    {setup : InputBindingSetup} {family : Family}
    (accepted : AcceptedArm setup family) :
    ∃ inputs output,
      FamilyPhaseRelation setup accepted.beforeState accepted.afterState
          family inputs output ∧
        Binding (kindForFamily family) accepted.bodyAssignment
          accepted.bodyCanonical ∧
        ∀ side, PreimageBinding (kindForFamily family) side
          accepted.bodyAssignment accepted.bodyCanonical :=
  ⟨accepted.phaseInputs, accepted.phaseOutput, accepted.phase,
    accepted.publicBinding, accepted.xOutPreimageBinding⟩

end AcceptedArm

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPhysicalState
