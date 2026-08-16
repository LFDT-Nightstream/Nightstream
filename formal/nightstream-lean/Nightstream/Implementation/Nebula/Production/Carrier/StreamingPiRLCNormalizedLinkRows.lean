import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiRLCNormalizedOverlayRows
import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiRLCNormalizedResidualRows
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPiRLCFamilyNormalizedLink

/-!
Contract: exact normalized meaning of the production PiRLC body-overlay
field-link rows.

Assurance tier: model-level, with a separate Rust-conformant slot receipt.

Owns the three corrected post-lowering source-field runs, their exact body and
overlay low-norm values, the active-selector equality-row implication, source
digit transfer into the overlay, and overlay-output transfer into the body
residual input.

Does not own shifted-ternary canonicality, selector authority, the stored Rust
matrices, the Rust witness encoder, replay rows, recursive orchestration, or
commitment hardness.

Emits constraints: no. It specifies and proves the arithmetic meaning of the
existing normalized equality-row recipe.
-/

set_option autoImplicit false
set_option maxRecDepth 524288
set_option maxHeartbeats 4000000

namespace Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedLinkRows

open Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputBinding
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial
open Nightstream.SuperNeo.Concrete

namespace Normalized

private abbrev sourceLayout :=
  Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilyCompleteRows.layout

private abbrev physicalLayout :=
  Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilyPhysicalOverlayRows.physicalLayout

abbrev BodyFinalColumns :=
  Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedResidualRows.Normalized.finalColumns

abbrev OverlayFinalColumns :=
  Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedOverlayRows.Normalized.finalColumns

theorem bodyFinalColumns_positive : 0 < BodyFinalColumns :=
  Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedResidualRows.Normalized.finalColumns_positive

theorem overlayFinalColumns_positive : 0 < OverlayFinalColumns :=
  Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedOverlayRows.Normalized.finalColumns_positive

private abbrev Lane := Fin 54
private abbrev Digit := Fin 41
private abbrev Output := Fin (shape.rows * shape.degree)
private abbrev Source :=
  Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputBinding.Source
private abbrev Family :=
  Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputBinding.Family
private abbrev Setup :=
  Nightstream.Implementation.Nebula.ProductionStreamingPiRlcAuthority.InputBindingSetup

/-- The zero digit source field moves from physical column 45,415 to
normalized source column 46,055. Its one-coordinate final slot starts at
1,059,804. -/
def bodyZeroDigitValue
    (assignment : Fin BodyFinalColumns → F) (digit : Digit) : F :=
  assignment ⟨1059804 + digit.val, by
    have upper := digit.isLt
    change 1059804 + digit.val < 2484972
    omega⟩

/-- Each active digit aliases one coordinate of the retained 41-coordinate
balanced-ternary input slot. -/
def bodyActiveDigitValue
    (assignment : Fin BodyFinalColumns → F)
    (source : Source) (lane : Lane) (digit : Digit) : F :=
  assignment ⟨19332 + (source.val * 54 + lane.val) * 41 + digit.val, by
    have sourceUpper := source.isLt
    have laneUpper := lane.isLt
    have digitUpper := digit.isLt
    change source.val < 15 at sourceUpper
    change 19332 + (source.val * 54 + lane.val) * 41 + digit.val < 2484972
    omega⟩

def overlayZeroSourceColumn (digit : Digit) :
    Fin Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedOverlayRows.Normalized.sourceColumns :=
  ⟨1 + digit.val, by
    have upper := digit.isLt
    change 1 + digit.val < 33360
    omega⟩

def overlayActiveSourceColumn
    (source : Source) (lane : Lane) (digit : Digit) :
    Fin Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedOverlayRows.Normalized.sourceColumns :=
  ⟨42 + (source.val * 54 + lane.val) * 41 + digit.val, by
    have sourceUpper := source.isLt
    have laneUpper := lane.isLt
    have digitUpper := digit.isLt
    change source.val < 15 at sourceUpper
    change 42 + (source.val * 54 + lane.val) * 41 + digit.val < 33360
    omega⟩

def overlayZeroDigitValue
    (assignment : Fin OverlayFinalColumns → F) (digit : Digit) : F :=
  Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedOverlayRows.Normalized.sourceColumnValue
    (overlayZeroSourceColumn digit) assignment

def overlayActiveDigitValue
    (assignment : Fin OverlayFinalColumns → F)
    (source : Source) (lane : Lane) (digit : Digit) : F :=
  Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedOverlayRows.Normalized.sourceColumnValue
    (overlayActiveSourceColumn source lane digit) assignment

def bodyOutputSourceColumn (output : Output) :
    Fin Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedCarryRows.Normalized.sourceColumns :=
  ⟨sourceLayout.input.phase.outputColumn output, by
    have upper := output.isLt
    change output.val < 108 at upper
    change 144278 + output.val < 146224
    omega⟩

/-- The body commitment output is the same radix-seven source value read by
the retained residual rows. Its final slot starts at 1,076,091. -/
def bodyOutputValue
    (assignment : Fin BodyFinalColumns → F) (output : Output) : F :=
  Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedCarryRows.Normalized.sourceColumnValue
    (bodyOutputSourceColumn output) assignment

def overlayOutputValue
    (assignment : Fin OverlayFinalColumns → F) (output : Output) : F :=
  Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedOverlayRows.Normalized.outputValue
    assignment output

inductive Link where
  | zero (digit : Digit)
  | active (source : Source) (lane : Lane) (digit : Digit)
  | output (index : Output)

def bodyValue
    (assignment : Fin BodyFinalColumns → F) : Link → F
  | .zero digit => bodyZeroDigitValue assignment digit
  | .active source lane digit =>
      bodyActiveDigitValue assignment source lane digit
  | .output output => bodyOutputValue assignment output

def overlayValue
    (assignment : Fin OverlayFinalColumns → F) : Link → F
  | .zero digit => overlayZeroDigitValue assignment digit
  | .active source lane digit =>
      overlayActiveDigitValue assignment source lane digit
  | .output output => overlayOutputValue assignment output

/-- Exact thirteen-port image of one link row. General selector is constant
one. The selected overlay kind gates the decoded field difference. -/
def linkPoint
    (selector body overlay : F) : Fin 13 → F :=
  Rows.productPoint 1 selector (body - overlay) 0

structure ProductionAccepted
    (selector : F)
    (bodyAssignment : Fin BodyFinalColumns → F)
    (overlayAssignment : Fin OverlayFinalColumns → F) : Prop where
  row : ∀ link,
    Semantics.evaluate
      (linkPoint selector (bodyValue bodyAssignment link)
        (overlayValue overlayAssignment link)) = 0

structure FieldLinksHold
    (bodyAssignment : Fin BodyFinalColumns → F)
    (overlayAssignment : Fin OverlayFinalColumns → F) : Prop where
  zero : ∀ digit,
    bodyZeroDigitValue bodyAssignment digit =
      overlayZeroDigitValue overlayAssignment digit
  active : ∀ source lane digit,
    bodyActiveDigitValue bodyAssignment source lane digit =
      overlayActiveDigitValue overlayAssignment source lane digit
  output : ∀ index,
    bodyOutputValue bodyAssignment index =
      overlayOutputValue overlayAssignment index

private theorem value_eq_of_accepted
    {selector : F}
    {bodyAssignment : Fin BodyFinalColumns → F}
    {overlayAssignment : Fin OverlayFinalColumns → F}
    (selectorOne : selector = 1)
    (accepted : ProductionAccepted selector bodyAssignment overlayAssignment)
    (link : Link) :
    bodyValue bodyAssignment link = overlayValue overlayAssignment link := by
  have rowAccepted := accepted.row link
  unfold linkPoint at rowAccepted
  have productEqual :=
    (Rows.evaluate_productPoint_one_eq_zero_iff
      selector
      (bodyValue bodyAssignment link - overlayValue overlayAssignment link)
      0).mp rowAccepted
  rw [selectorOne, Fin.one_mul] at productEqual
  exact Lean.Grind.AddCommGroup.sub_eq_zero_iff.mp productEqual

/-- Active normalized link rows imply every one of the three field-link
families. No digest is an authority premise. -/
theorem accepted_implies_fieldLinksHold
    {selector : F}
    {bodyAssignment : Fin BodyFinalColumns → F}
    {overlayAssignment : Fin OverlayFinalColumns → F}
    (selectorOne : selector = 1)
    (accepted : ProductionAccepted selector bodyAssignment overlayAssignment) :
    FieldLinksHold bodyAssignment overlayAssignment := by
  constructor
  · intro digit
    exact value_eq_of_accepted selectorOne accepted (.zero digit)
  · intro source lane digit
    exact value_eq_of_accepted selectorOne accepted (.active source lane digit)
  · intro output
    exact value_eq_of_accepted selectorOne accepted (.output output)

/-- Exact body digit values for the authoritative input and zero words. The
next slice proves this predicate from the normalized shifted-ternary rows. -/
structure BodySourceColumnsExact
    (bodyAssignment : Fin BodyFinalColumns → F)
    (inputs : Source → RingF) : Prop where
  active : ∀ source lane digit,
    (bodyActiveDigitValue bodyAssignment source lane digit).val =
      (Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingSetup.integerResidue
        (Nightstream.Protocol.Nebula.CompactCommit.signedDigit
          (Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputResidual.canonicalInput
            (inputs source lane)) digit)).val
  zero : ∀ digit,
    (bodyZeroDigitValue bodyAssignment digit).val =
      (Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingSetup.integerResidue 0).val

/-- Exact body digits and accepted active links give the source-column
authority required by the normalized family overlay. -/
theorem overlaySourceColumnsExact_of_links
    {bodyAssignment : Fin BodyFinalColumns → F}
    {overlayAssignment : Fin OverlayFinalColumns → F}
    {inputs : Source → RingF}
    (bodyExact : BodySourceColumnsExact bodyAssignment inputs)
    (links : FieldLinksHold bodyAssignment overlayAssignment) :
    Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputPhaseRows.SourceColumnsExact
      physicalLayout
      (Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedOverlayRows.Normalized.numericAssignment
        overlayAssignment)
      inputs := by
  constructor
  · intro source lane digit
    let column := overlayActiveSourceColumn source lane digit
    have decoded :=
      Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedOverlayRows.Normalized.numericAssignment_of_lt
        overlayAssignment column.val column.isLt
    calc
      Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedOverlayRows.Normalized.numericAssignment
          overlayAssignment (physicalLayout.digitStart source lane + digit.val) =
          (overlayActiveDigitValue overlayAssignment source lane digit).val := by
        simpa [column, overlayActiveSourceColumn, overlayActiveDigitValue,
          physicalLayout] using decoded
      _ = (bodyActiveDigitValue bodyAssignment source lane digit).val :=
        congrArg Fin.val (links.active source lane digit).symm
      _ = _ := bodyExact.active source lane digit
  · intro digit
    let column := overlayZeroSourceColumn digit
    have decoded :=
      Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedOverlayRows.Normalized.numericAssignment_of_lt
        overlayAssignment column.val column.isLt
    calc
      Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedOverlayRows.Normalized.numericAssignment
          overlayAssignment (physicalLayout.zeroDigitStart + digit.val) =
          (overlayZeroDigitValue overlayAssignment digit).val := by
        simpa [column, overlayZeroSourceColumn, overlayZeroDigitValue,
          physicalLayout] using decoded
      _ = (bodyZeroDigitValue bodyAssignment digit).val :=
        congrArg Fin.val (links.zero digit).symm
      _ = _ := bodyExact.zero digit

private theorem bodyOutputValue_numeric
    (assignment : Fin BodyFinalColumns → F) (output : Output) :
    Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedResidualRows.Normalized.numericAssignment
        assignment (sourceLayout.input.phase.outputColumn output) =
      (bodyOutputValue assignment output).val := by
  have bound : sourceLayout.input.phase.outputColumn output <
      Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedCarryRows.Normalized.sourceColumns := by
    have upper := output.isLt
    change output.val < 108 at upper
    change 144278 + output.val < 146224
    omega
  unfold
    Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedResidualRows.Normalized.numericAssignment
    Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedCarryRows.Normalized.numericAssignment
    Nightstream.Implementation.Lowering.Goldilocks.NumericRowBridge.numericAssignment
    Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedCarryRows.Normalized.decodedAssignment
  rw [Nightstream.Implementation.Nebula.ProductionFreshRelationCompilerFor.NumericBridge.finiteColumnIndex_sourceColumn_of_lt
    Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedCarryRows.Normalized.sourceColumns_positive
    bound]
  simpa [bodyOutputValue, bodyOutputSourceColumn]

/-- Output links transfer the authoritative overlay commitment into the exact
body phase-binding columns read by the normalized residual rows. -/
theorem bodyPhaseBindingPlaced_of_links
    {setup : Setup} {family : Family}
    {inputs : Source → RingF}
    {bodyAssignment : Fin BodyFinalColumns → F}
    {overlayAssignment : Fin OverlayFinalColumns → F}
    (links : FieldLinksHold bodyAssignment overlayAssignment)
    (overlayPlaced :
      Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedOverlayRows.Normalized.PhaseBindingPlaced
        setup family inputs overlayAssignment) :
    Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedResidualRows.Normalized.PhaseBindingPlaced
      setup family inputs bodyAssignment := by
  intro output
  calc
    Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedResidualRows.Normalized.numericAssignment
        bodyAssignment (sourceLayout.input.phase.outputColumn output) =
        (bodyOutputValue bodyAssignment output).val :=
      bodyOutputValue_numeric bodyAssignment output
    _ = (overlayOutputValue overlayAssignment output).val :=
      congrArg Fin.val (links.output output)
    _ = (Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputBindingSetup.concretePhaseBinding
        setup family inputs output).val :=
      congrArg Fin.val (overlayPlaced output)

/-- Accepted overlay rows, accepted active field links, and exact body source
digits imply the body phase-binding premise required by the normalized
residual theorem. -/
theorem accepted_implies_bodyPhaseBindingPlaced
    {setup : Setup} {family : Family}
    {inputs : Source → RingF}
    {selector : F}
    {bodyAssignment : Fin BodyFinalColumns → F}
    {overlayAssignment : Fin OverlayFinalColumns → F}
    (selectorOne : selector = 1)
    (constantOne : overlayAssignment ⟨0, overlayFinalColumns_positive⟩ = 1)
    (bodyExact : BodySourceColumnsExact bodyAssignment inputs)
    (linksAccepted : ProductionAccepted selector bodyAssignment overlayAssignment)
    (overlayAccepted :
      Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedOverlayRows.Normalized.ProductionAccepted
        setup family overlayAssignment) :
    Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedResidualRows.Normalized.PhaseBindingPlaced
      setup family inputs bodyAssignment := by
  let links := accepted_implies_fieldLinksHold selectorOne linksAccepted
  have sourceExact := overlaySourceColumnsExact_of_links bodyExact links
  have overlayPlaced :=
    Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedOverlayRows.Normalized.accepted_implies_phaseBindingPlaced
      constantOne sourceExact overlayAccepted
  exact bodyPhaseBindingPlaced_of_links links overlayPlaced

/-- The model constants are exactly the Rust-conformant normalized-link
receipt. -/
theorem receipt_geometry_exact :
    BodyFinalColumns =
        Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyNormalizedLink.audit.bodyFinalColumns /\
      OverlayFinalColumns =
        Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyNormalizedLink.audit.overlayFinalColumns /\
      Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyNormalizedLink.audit.publicOutputCount =
        640 /\
      Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyNormalizedLink.audit.linkCountPerFamily =
        33359 /\
      Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyNormalizedLink.audit.totalLinkCount =
        3669490 := by
  native_decide

end Normalized

end Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedLinkRows
