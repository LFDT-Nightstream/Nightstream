import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiRLCFamilyPhysicalOverlayRows
import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiRLCInputBindingProductionSetup
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPiRLCFamilyOverlayRetained

/-!
Contract: exact normalized low-norm meaning of the production PiRLC family
overlay rows.

Assurance tier: model-level, with a separate Rust-conformant geometry receipt.

Owns the 33,360-to-35,856 source-column image, the family selector column,
the exact seeded A, constant-one B, and radix-seven output C product point,
and the implication from one selected 108-row overlay arm to the concrete
108-field family commitment.

Does not own source-opening rows, body-to-overlay links, selector authority,
the stored Rust matrices, the Rust witness encoder, recursive orchestration,
or Module-SIS hardness.

Emits constraints: no. It specifies and proves the arithmetic meaning of the
existing normalized product-row recipe.
-/

set_option autoImplicit false
set_option maxRecDepth 100000
set_option maxHeartbeats 4000000

namespace Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedOverlayRows

open Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputBinding
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial
open Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.Decoder
open Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.SourceImage
open Nightstream.SuperNeo.Concrete

namespace Normalized

private abbrev physicalLayout :=
  Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilyPhysicalOverlayRows.physicalLayout

private abbrev Setup :=
  Nightstream.Implementation.Nebula.ProductionStreamingPiRlcAuthority.InputBindingSetup

def sourceColumns : Nat := 33360

def finalColumns : Nat := 35856

theorem sourceColumns_positive : 0 < sourceColumns := by
  decide

theorem finalColumns_positive : 0 < finalColumns := by
  decide

/-- Exact selector used by the normalized arm with this family ordinal. -/
def selectorColumn (family : Family) : Fin finalColumns :=
  ⟨1 + ProductPiRlcAlgebraRows.familyOrdinal family, by
    have bound := ProductPiRlcAlgebraRows.familyOrdinal_lt family
    unfold finalColumns
    omega⟩

@[simp] theorem selectorColumn_val (family : Family) :
    (selectorColumn family).val =
      1 + ProductPiRlcAlgebraRows.familyOrdinal family := by
  rfl

/-- Exact normalized slot of every nonconstant physical source column.
Digits use one direct centered coordinate. Outputs use 23 radix-seven
coordinates. -/
def sourceSlot
    (column : Fin sourceColumns) (_nonzero : column.val ≠ 0) :
    DecodedSourceSlot sourceColumns finalColumns :=
  if direct : column.val < 33252 then
    { column := column
      start := column.val + 110
      width := 1
      widthPositive := by decide
      columnsFit := by
        unfold finalColumns
        omega }
  else
    { column := column
      start := 33362 + (column.val - 33252) * 23
      width := 23
      widthPositive := by decide
      columnsFit := by
        have upper := column.isLt
        unfold sourceColumns at upper
        unfold finalColumns
        omega }

/-- Sparse field value of one physical source column on the normalized
assignment. Column zero remains the final constant column. -/
def sourceColumnValue
    (column : Fin sourceColumns) (assignment : Fin finalColumns → F) : F :=
  if zero : column.val = 0 then
    assignment ⟨0, finalColumns_positive⟩
  else
    sourceSlotValue (sourceSlot column zero) assignment

/-- Canonical numeric assignment read by the existing compact seeded-block
semantics. Out-of-domain columns fail closed to zero. -/
def numericAssignment
    (assignment : Fin finalColumns → F) (column : Nat) : Nat :=
  if bound : column < sourceColumns then
    (sourceColumnValue ⟨column, bound⟩ assignment).val
  else
    0

theorem numericAssignment_of_lt
    (assignment : Fin finalColumns → F) (column : Nat)
    (bound : column < sourceColumns) :
    numericAssignment assignment column =
      (sourceColumnValue ⟨column, bound⟩ assignment).val := by
  unfold numericAssignment
  rw [dif_pos bound]

/-- Physical source column that stores one flattened commitment output. -/
def outputSourceColumn
    (output : Fin (shape.rows * shape.degree)) : Fin sourceColumns :=
  ⟨physicalLayout.outputColumn output, by
    have upper : output.val < 108 := by
      simpa only [
        Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputBindingSetup.exact_output_width]
        using output.isLt
    change 33252 + output.val < 33360
    omega⟩

@[simp] theorem outputSourceColumn_val
    (output : Fin (shape.rows * shape.degree)) :
    (outputSourceColumn output).val = 33252 + output.val := by
  rfl

def outputValue
    (assignment : Fin finalColumns → F)
    (output : Fin (shape.rows * shape.degree)) : F :=
  sourceColumnValue (outputSourceColumn output) assignment

/-- Exact thirteen-port point scanned by Rust for one retained overlay row.
The compact A value uses the verifier-owned seeded block, B reads constant
one, and C reads the normalized radix-seven output slot. -/
def coordinatePoint
    (setup : Setup) (family : Family)
    (assignment : Fin finalColumns → F)
    (output : Fin shape.rows) (coordinate : Fin shape.degree) : Fin 13 → F :=
  Rows.productPoint
    (assignment (selectorColumn family))
    (SeededPhi81RingRefinement.residueNat
      ((Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputPhaseRows.coordinateBlock
        setup physicalLayout family).linearValue
          (numericAssignment assignment) output.val coordinate.val))
    (assignment ⟨0, finalColumns_positive⟩)
    (outputValue assignment
      (Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputPhaseRows.residualOutputIndex
        output coordinate))

/-- One family selector is active and all its 108 retained product rows
accept on the exact normalized port image. -/
structure ProductionAccepted
    (setup : Setup) (family : Family)
    (assignment : Fin finalColumns → F) : Prop where
  selectorOne : assignment (selectorColumn family) = 1
  coordinate : ∀ output coordinate,
    Semantics.evaluate
      (coordinatePoint setup family assignment output coordinate) = 0

/-- Selected product rows equate each normalized output slot with the exact
residue of the compact seeded linear value. -/
theorem accepted_coordinate_eq_linearValue
    {setup : Setup} {family : Family}
    {assignment : Fin finalColumns → F}
    (constantOne : assignment ⟨0, finalColumns_positive⟩ = 1)
    (accepted : ProductionAccepted setup family assignment)
    (output : Fin shape.rows) (coordinate : Fin shape.degree) :
    outputValue assignment
        (Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputPhaseRows.residualOutputIndex
          output coordinate) =
      SeededPhi81RingRefinement.residueNat
        ((Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputPhaseRows.coordinateBlock
          setup physicalLayout family).linearValue
            (numericAssignment assignment) output.val coordinate.val) := by
  have rowAccepted := accepted.coordinate output coordinate
  simp only [coordinatePoint, accepted.selectorOne, constantOne] at rowAccepted
  have productEqual :=
    (Rows.evaluate_productPoint_one_eq_zero_iff _ _ _).mp rowAccepted
  simpa only [Fin.mul_one] using productEqual.symm

/-- Accepted normalized rows and exact source digits place every coordinate
of the concrete family commitment. -/
theorem accepted_implies_coordinate_commitment
    {setup : Setup} {family : Family}
    {inputs : Source → RingF}
    {assignment : Fin finalColumns → F}
    (constantOne : assignment ⟨0, finalColumns_positive⟩ = 1)
    (sourceExact :
      Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputPhaseRows.SourceColumnsExact
        physicalLayout (numericAssignment assignment) inputs)
    (accepted : ProductionAccepted setup family assignment)
    (output : Fin shape.rows) (coordinate : Fin shape.degree) :
    outputValue assignment
        (Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputPhaseRows.residualOutputIndex
          output coordinate) =
      ((Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputResidual.phaseBinding
        (Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputBindingSetup.seededMatrix
          setup)
        Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputBindingSetup.coefficientMap
        family inputs output).coefficients
          coordinate) := by
  exact
    (accepted_coordinate_eq_linearValue constantOne accepted output coordinate).trans
      (Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputPhaseRows.phaseCommitment_coordinate_eq_linearValue
        (setup := setup) sourceExact output coordinate).symm

/-- Exact flattened commitment placement produced by the overlay assignment.
This is the authority fact required by the normalized residual rows. -/
def PhaseBindingPlaced
    (setup : Setup) (family : Family)
    (inputs : Source → RingF)
    (assignment : Fin finalColumns → F) : Prop :=
  ∀ output,
    outputValue assignment output =
      Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputBindingSetup.concretePhaseBinding
        setup family inputs output

theorem accepted_implies_phaseBindingPlaced
    {setup : Setup} {family : Family}
    {inputs : Source → RingF}
    {assignment : Fin finalColumns → F}
    (constantOne : assignment ⟨0, finalColumns_positive⟩ = 1)
    (sourceExact :
      Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputPhaseRows.SourceColumnsExact
        physicalLayout (numericAssignment assignment) inputs)
    (accepted : ProductionAccepted setup family assignment) :
    PhaseBindingPlaced setup family inputs assignment := by
  intro output
  let pair :=
    Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputBindingSetup.outputPair
      output
  have coordinatePlaced :=
    accepted_implies_coordinate_commitment constantOne sourceExact accepted
      pair.1 pair.2
  have outputIndexEqual :
      Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputPhaseRows.residualOutputIndex
          pair.1 pair.2 = output := by
    exact
      Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputBindingSetup.outputIndex_outputPair
        output
  rw [outputIndexEqual] at coordinatePlaced
  simpa [
    Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputBindingSetup.concretePhaseBinding,
    Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputBindingSetup.flattenCommitment,
    pair] using coordinatePlaced

/-- The model constants are the exact Rust-conformant receipt geometry. -/
theorem receipt_geometry_exact :
    sourceColumns =
        Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyOverlayRetained.audit.sourceColumns /\
      finalColumns =
        Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyOverlayRetained.audit.finalColumns /\
      Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyOverlayRetained.audit.selectorStart = 1 /\
      Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyOverlayRetained.audit.selectorCount = 110 /\
      Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyOverlayRetained.audit.sourceStarts =
        [1, 42, 33252] /\
      Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyOverlayRetained.audit.finalStarts =
        [111, 152, 33362] /\
      Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyOverlayRetained.audit.widths =
        [1, 23] /\
      Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyOverlayRetained.audit.radices =
        [2, 7] := by
  native_decide

/-- The production specialization uses the exact seed schedule and port
censuses checked by the Rust-conformant receipt. -/
theorem production_receipt_valid :
    Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyOverlayRetained.AuditValid :=
  Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyOverlayRetained.audit_valid

end Normalized

end Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedOverlayRows
