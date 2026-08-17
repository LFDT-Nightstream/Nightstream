import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiRLCFamilyBodyOverlayRows

/-!
Contract: same-assignment authority for one physical PiRLC family overlay.

Assurance tier: generated-source and physical-link soundness.

Owns the 33,360-column physical overlay layout and the 33,359 exact field
links from one shared parity body. Proves that accepted physical overlay rows
and those links imply the same `FamilyPhaseRelation` as the source layout.

Does not own normalized link rows, overlay selection, recursive lifecycle
integration, or the terminal zero-residual check.

Emits constraints: 108 physical overlay rows. The separate link compiler
emits 33,359 field equalities.
-/

set_option autoImplicit false
set_option maxRecDepth 524288
set_option maxHeartbeats 4000000

namespace Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilyPhysicalOverlayRows

open Nightstream.Implementation.Nebula.ProductPiRlcRingCombinationSound
open Nightstream.Implementation.Nebula.ProductionStreamingPiRlcAuthority
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyReplayArtifact
open Nightstream.SuperNeo.Concrete

private abbrev InputPhaseLayout :=
  Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputPhaseRows.Layout

private abbrev sourceLayout :=
  Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilyCompleteRows.layout

private abbrev parityFor :=
  Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilyCompleteRows.parityFor

private abbrev Source :=
  Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputBinding.Source

private abbrev Lane := Fin 54

private abbrev Digit := Fin 41

private abbrev Output := Fin 108

/-- Exact local columns emitted by the Rust physical overlay builder. Only
the zero word, the 918 active words, and the 108 outputs are authoritative.
The other layout fields are not read by the seeded coordinate rows. -/
def physicalLayout : InputPhaseLayout where
  inputColumn := fun _ _ => 0
  digitStart := fun source lane =>
    42 + (source.val * 54 + lane.val) * 41
  zeroDigitStart := 1
  dColumn := 0
  kappaColumn := 0
  outputColumn := fun output => 33252 + output.val
  seededRowStart := 0

theorem physical_layout_exact :
    physicalLayout.zeroDigitStart = 1 /\
      physicalLayout.digitStart ⟨0, by decide⟩ ⟨0, by decide⟩ = 42 /\
      physicalLayout.digitStart ⟨14, by decide⟩ ⟨53, by decide⟩ = 33211 /\
      physicalLayout.outputColumn ⟨0, by decide⟩ = 33252 /\
      physicalLayout.outputColumn ⟨107, by decide⟩ = 33359 := by
  decide

/-- The exact physical rows selected for one family position. -/
def rows (setup : InputBindingSetup) (family : Family) : List Row :=
  (Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputPhaseRows.coordinateBlock
    setup physicalLayout family).rows

theorem rows_length (setup : InputBindingSetup) (family : Family) :
    (rows setup family).length = 108 := by
  exact
    Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputPhaseRows.coordinateRows_length
      setup physicalLayout family

/-- Exact equality contract produced by the three compact Rust link runs.
The body assignment remains the algebraic authority. -/
structure FieldLinksHold
    (bodyAssignment overlayAssignment : Nat → Nat) : Prop where
  zero : ∀ digit : Digit,
    bodyAssignment (sourceLayout.input.phase.zeroDigitStart + digit.val) =
      overlayAssignment (physicalLayout.zeroDigitStart + digit.val)
  active : ∀ source : Source, ∀ lane : Lane, ∀ digit : Digit,
    bodyAssignment
        (sourceLayout.input.phase.digitStart source lane + digit.val) =
      overlayAssignment (physicalLayout.digitStart source lane + digit.val)
  output : ∀ output : Output,
    bodyAssignment (sourceLayout.input.phase.outputColumn output) =
      overlayAssignment (physicalLayout.outputColumn output)

def fieldLinkCount : Nat := 41 + 15 * 54 * 41 + 108

theorem fieldLinkCount_exact : fieldLinkCount = 33359 := by
  decide

/-- The compact link runs start at the exact body and physical columns and
use the exact body and overlay strides. -/
theorem link_run_geometry_exact :
    sourceLayout.input.phase.zeroDigitStart = 51463 /\
      physicalLayout.zeroDigitStart = 1 /\
      sourceLayout.input.phase.digitStart
          ⟨0, by decide⟩ ⟨0, by decide⟩ = 51504 /\
      physicalLayout.digitStart ⟨0, by decide⟩ ⟨0, by decide⟩ = 42 /\
      sourceLayout.input.phase.outputColumn ⟨0, by decide⟩ = 163502 /\
      physicalLayout.outputColumn ⟨0, by decide⟩ = 33252 := by
  exact ⟨rfl, rfl, rfl, rfl, rfl, rfl⟩

/-- Exact body source digits transfer to the physical overlay assignment
through all 33,251 input links. -/
theorem physicalSourceColumnsExact_of_links
    {bodyAssignment overlayAssignment : Nat → Nat}
    {inputs : Source → RingF}
    (bodyExact :
      Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputPhaseRows.SourceColumnsExact
        sourceLayout.input.phase bodyAssignment inputs)
    (links : FieldLinksHold bodyAssignment overlayAssignment) :
    Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputPhaseRows.SourceColumnsExact
      physicalLayout overlayAssignment inputs := by
  constructor
  · intro source lane digit
    rw [← links.active source lane digit]
    exact bodyExact.1 source lane digit
  · intro digit
    rw [← links.zero digit]
    exact bodyExact.2 digit

/-- Accepted body rows, accepted physical overlay rows, and the exact field
links imply the production PiRLC family relation. No digest is an authority
premise. -/
theorem rows_sound
    {setup : InputBindingSetup} {family : Family}
    {bodyAssignment overlayAssignment : Nat → Nat}
    {before after : FamilyState}
    (bodyCanonical : ∀ column, bodyAssignment column < goldilocksP)
    (overlayCanonical : ∀ column, overlayAssignment column < goldilocksP)
    (bodyOne : bodyAssignment 0 = 1)
    (overlayOne : overlayAssignment 0 = 1)
    (range : ∀ source lane,
      bodyAssignment (sourceLayout.algebra.challengeSymbol source lane) < 5)
    (residualsPlaced :
      Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputFamilyRows.ResidualsPlaced
        sourceLayout.input bodyAssignment
        before.inputResidual after.inputResidual)
    (carryStatePlaced :
      Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilyCarryRows.StateColumnsPlaced
        (Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilySourceRows.carryLayout
          sourceLayout) bodyAssignment before after)
    (replayStatesPlaced :
      FamilyStatesPlaced (parityFor family)
        bodyAssignment before after)
    (cursorExact : before.familyCursor =
      ProductPiRlcAlgebraRows.familyOrdinal family)
    (links : FieldLinksHold bodyAssignment overlayAssignment)
    (bodySatisfied : Satisfies
      (Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilyBodyOverlayRows.bodyRowsForParity
        (parityFor family)) bodyAssignment)
    (overlaySatisfied : Satisfies (rows setup family) overlayAssignment) :
    FamilyPhaseRelation setup before after family
      (decodedInputs sourceLayout.algebra bodyAssignment bodyCanonical)
      (outputRing sourceLayout.algebra bodyAssignment bodyCanonical) := by
  have bodySourceExact :=
    Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilyBodyOverlayRows.sourceColumnsExact_of_bodyRows
      bodyCanonical bodyOne bodySatisfied
  have physicalSourceExact :=
    physicalSourceColumnsExact_of_links bodySourceExact links
  apply
    Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilyBodyOverlayRows.bodyRows_sound_of_output_exact
      bodyCanonical bodyOne range residualsPlaced carryStatePlaced
      replayStatesPlaced cursorExact bodySatisfied
  intro output coordinate
  let index :=
    Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputPhaseRows.residualOutputIndex
      output coordinate
  calc
    bodyAssignment (sourceLayout.input.phase.outputColumn index) =
        overlayAssignment (physicalLayout.outputColumn index) :=
      links.output index
    _ =
        (Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputBindingSetup.concretePhaseBinding
          setup family
          (decodedInputs sourceLayout.algebra bodyAssignment bodyCanonical)
          index).val := by
      exact
        Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputPhaseRows.compact_output_exact_of_rows
          overlayCanonical overlayOne physicalSourceExact overlaySatisfied
          output coordinate

/-- Complete accepted source witness for one physical PiRLC family arm. It
keeps the shared parity body and the family-dependent overlay assignments
separate, and joins them only through `FieldLinksHold`. -/
structure AcceptedRows
    (setup : InputBindingSetup) (family : Family)
    (before after : FamilyState) where
  bodyAssignment : Nat → Nat
  overlayAssignment : Nat → Nat
  bodyCanonical : ∀ column, bodyAssignment column < goldilocksP
  overlayCanonical : ∀ column, overlayAssignment column < goldilocksP
  bodyOne : bodyAssignment 0 = 1
  overlayOne : overlayAssignment 0 = 1
  range : ∀ source lane,
    bodyAssignment (sourceLayout.algebra.challengeSymbol source lane) < 5
  residualsPlaced :
    Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputFamilyRows.ResidualsPlaced
      sourceLayout.input bodyAssignment
      before.inputResidual after.inputResidual
  carryStatePlaced :
    Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilyCarryRows.StateColumnsPlaced
      (Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilySourceRows.carryLayout
        sourceLayout) bodyAssignment before after
  replayStatesPlaced :
    FamilyStatesPlaced (parityFor family) bodyAssignment before after
  cursorExact : before.familyCursor =
    ProductPiRlcAlgebraRows.familyOrdinal family
  links : FieldLinksHold bodyAssignment overlayAssignment
  bodySatisfied : Satisfies
    (Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilyBodyOverlayRows.bodyRowsForParity
      (parityFor family)) bodyAssignment
  overlaySatisfied : Satisfies (rows setup family) overlayAssignment

namespace AcceptedRows

/-- An accepted physical witness determines one exact semantic family phase.
The existential values are decoded from the body assignment; no digest or
overlay output is accepted as an independent input. -/
theorem sound
    {setup : InputBindingSetup} {family : Family}
    {before after : FamilyState}
    (accepted : AcceptedRows setup family before after) :
    ∃ inputs output,
      FamilyPhaseRelation setup before after family inputs output := by
  let inputs :=
    decodedInputs sourceLayout.algebra accepted.bodyAssignment
      accepted.bodyCanonical
  let output :=
    outputRing sourceLayout.algebra accepted.bodyAssignment
      accepted.bodyCanonical
  exact ⟨inputs, output,
    rows_sound accepted.bodyCanonical accepted.overlayCanonical
      accepted.bodyOne accepted.overlayOne accepted.range
      accepted.residualsPlaced accepted.carryStatePlaced
      accepted.replayStatesPlaced accepted.cursorExact accepted.links
      accepted.bodySatisfied accepted.overlaySatisfied⟩

end AcceptedRows

end Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilyPhysicalOverlayRows
