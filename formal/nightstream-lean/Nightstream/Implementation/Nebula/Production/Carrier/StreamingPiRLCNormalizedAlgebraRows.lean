import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiRLCArtifact
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPiRLCFamilyBodyAlgebraRetained
import Nightstream.Implementation.R1CS.Correspondence.SelectiveCcs.Rewrite.Artifact.SourceImage

/-!
Contract: exact normalized low-norm image of the production PiRLC algebra
rows.

Assurance tier: model-level.

Owns the 51,463-to-8,858,862 source-column substitution, the two parity-arm
selector coordinates, the exact thirteen-port product-row acceptance
predicate, and the same-assignment implication from all 49,626 normalized
rows to `FamilyPhaseRelation`.

Does not own low-norm digit range rows, the Rust matrix scan, the remaining
normalized row families, overlay links, recursive orchestration, terminal
integration, or cryptographic security assumptions.

Emits constraints: no. It specifies and proves the arithmetic meaning of the
existing normalized product-row recipe.
-/

set_option autoImplicit false
set_option maxRecDepth 100000
set_option maxHeartbeats 4000000

namespace Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedAlgebraRows

open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.Goldilocks.NumericRowBridge
open Nightstream.Implementation.Nebula.ProductionFreshRelationCompilerFor
open Nightstream.Implementation.Nebula.ProductionStreamingPiRlcAuthority
open Nightstream.Implementation.Nebula.ProductionStreamingPiRlcArtifact
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.SelectiveCcs.LeanCompiler
open Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial
open Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.Decoder
open Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.SourceImage
open Nightstream.SuperNeo.Concrete

namespace Normalized

def localColumns : Nat := 51463

def finalColumns : Nat := 8858862

theorem localColumns_positive : 0 < localColumns := by
  decide

theorem finalColumns_positive : 0 < finalColumns := by
  decide

@[simp] theorem localColumns_eq_generated :
    localColumns = Generated.columns := by
  rfl

/-- The normalized body has one selected copy of the algebra rows in each
parity arm. -/
inductive Arm where
  | even
  | odd
deriving DecidableEq, Repr

def selectorColumn : Arm → Fin finalColumns
  | .even => ⟨648, by decide⟩
  | .odd => ⟨649, by decide⟩

@[simp] theorem selectorColumn_even_val :
    (selectorColumn .even).val = 648 := by
  rfl

@[simp] theorem selectorColumn_odd_val :
    (selectorColumn .odd).val = 649 := by
  rfl

/-- The model geometry is the exact geometry scanned in both Rust retained
algebra intervals. This is a constant-size receipt check; it does not evaluate
the generated row set. -/
theorem retained_audit_geometry :
    Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyBodyAlgebraRetained.audit.sourceRows =
        49626 /\
      Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyBodyAlgebraRetained.audit.localColumns =
        localColumns /\
      Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyBodyAlgebraRetained.audit.finalColumns =
        finalColumns /\
      Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyBodyAlgebraRetained.audit.selectorColumns =
        [(selectorColumn .even).val, (selectorColumn .odd).val] /\
      Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyBodyAlgebraRetained.audit.widths =
        [41, 41, 41, 41] /\
      Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyBodyAlgebraRetained.audit.radices =
        [3, 3, 3, 3] := by
  exact ⟨rfl, rfl, rfl, rfl, rfl, rfl⟩

/-- Exact radix-three low-norm slot for each nonconstant local algebra
column. The four intervals are challenges, inputs, outputs, and products. -/
def localSlot (column : Fin localColumns) (nonzero : column.val ≠ 0) :
    DecodedSourceSlot localColumns finalColumns :=
  if challenge : column.val < 919 then
    { column := column
      start := 702 + (column.val - 1) * 41
      width := 41
      widthPositive := by decide
      columnsFit := by
        have lower : 1 ≤ column.val := Nat.one_le_iff_ne_zero.mpr nonzero
        change 702 + (column.val - 1) * 41 + 41 ≤ 8858862
        omega }
  else if input : column.val < 1837 then
    { column := column
      start := 38340 + (column.val - 919) * 41
      width := 41
      widthPositive := by decide
      columnsFit := by
        change 38340 + (column.val - 919) * 41 + 41 ≤ 8858862
        omega }
  else if output : column.val < 1891 then
    { column := column
      start := 75978 + (column.val - 1837) * 41
      width := 41
      widthPositive := by decide
      columnsFit := by
        change 75978 + (column.val - 1837) * 41 + 41 ≤ 8858862
        omega }
  else
    { column := column
      start := 78192 + (column.val - 1891) * 41
      width := 41
      widthPositive := by decide
      columnsFit := by
        have upper := column.isLt
        unfold localColumns at upper
        change 78192 + (column.val - 1891) * 41 + 41 ≤ 8858862
        omega }

def localColumnForm (column : Fin localColumns) : Form finalColumns :=
  if zero : column.val = 0 then
    constantForm finalColumns_positive
  else
    sourceSlotForm (localSlot column zero)

/-- Sparse value of one local source column on the final assignment. -/
def localColumnValue
    (column : Fin localColumns) (assignment : Fin finalColumns → F) : F :=
  if zero : column.val = 0 then
    assignment ⟨0, finalColumns_positive⟩
  else
    sourceSlotValue (localSlot column zero) assignment

/-- The one local source assignment decoded from the final normalized
assignment. Every source row and every phase value below reads this function.
-/
def decodedLocalAssignment
    (assignment : Fin finalColumns → F) : Fin localColumns → F :=
  fun column => localColumnValue column assignment

theorem evaluate_localColumnForm
    (column : Fin localColumns) (assignment : Fin finalColumns → F) :
    Form.evaluate (localColumnForm column) assignment =
      decodedLocalAssignment assignment column := by
  unfold localColumnForm decodedLocalAssignment localColumnValue
  split
  · exact evaluate_constantForm finalColumns_positive assignment
  · exact evaluate_sourceSlotForm _ assignment

theorem pulledDecodedAssignment_apply
    (assignment : Fin finalColumns → F) (column : ColumnId) :
    StableRows.pulledAssignment
        (NumericBridge.finiteColumnIndex localColumns_positive)
        (decodedLocalAssignment assignment) column =
      Form.evaluate
        (localColumnForm
          (NumericBridge.finiteColumnIndex localColumns_positive column))
        assignment := by
  unfold StableRows.pulledAssignment
  exact (evaluate_localColumnForm _ assignment).symm

theorem decodedLocalAssignment_zero
    (assignment : Fin finalColumns → F) :
    decodedLocalAssignment assignment ⟨0, localColumns_positive⟩ =
      assignment ⟨0, finalColumns_positive⟩ := by
  unfold decodedLocalAssignment localColumnValue
  rw [dif_pos rfl]

/-- Substitute each sparse source term with its exact final low-norm linear
image. The source column index is the same finite index used by the stable-row
compiler. -/
def combinationImage : LinearCombination → Form finalColumns
  | [] => Form.zero
  | term :: tail =>
      Form.add
        (Form.scale term.coefficient
          (localColumnForm
            (NumericBridge.finiteColumnIndex localColumns_positive
              term.column)))
        (combinationImage tail)

theorem evaluate_combinationImage
    (source : LinearCombination)
    (assignment : Fin finalColumns → F) :
    Form.evaluate (combinationImage source) assignment =
      LinearCombination.eval
        (StableRows.pulledAssignment
          (NumericBridge.finiteColumnIndex localColumns_positive)
          (decodedLocalAssignment assignment))
        source := by
  induction source with
  | nil =>
      change Form.evaluate Form.zero assignment = 0
      exact Form.evaluate_zero assignment
  | cons term tail inductionHypothesis =>
      rw [combinationImage, LinearCombination.eval_cons,
        Form.evaluate_add, Form.evaluate_scale, inductionHypothesis]
      rw [pulledDecodedAssignment_apply]

/-- Exact A, B, and C port images of one normalized product row. -/
structure RowImage where
  a : Form finalColumns
  b : Form finalColumns
  c : Form finalColumns

def rowImage
    (row : Nightstream.Implementation.Lowering.Goldilocks.Row) : RowImage where
  a := combinationImage row.a
  b := combinationImage row.b
  c := combinationImage row.c

def RowImage.Holds
    (image : RowImage) (assignment : Fin finalColumns → F) : Prop :=
  Form.evaluate image.a assignment * Form.evaluate image.b assignment =
    Form.evaluate image.c assignment

/-- The exact thirteen-port point for one normalized product row. Unused
ports are zero by `Rows.productPoint`. -/
def RowImage.portPoint
    (image : RowImage) (selector : F)
    (assignment : Fin finalColumns → F) : Fin 13 → F :=
  Rows.productPoint selector
    (Form.evaluate image.a assignment)
    (Form.evaluate image.b assignment)
    (Form.evaluate image.c assignment)

def RowImage.Accepted
    (image : RowImage) (selector : F)
    (assignment : Fin finalColumns → F) : Prop :=
  Semantics.evaluate (image.portPoint selector assignment) = 0

theorem rowImage_holds_iff
    (row : Nightstream.Implementation.Lowering.Goldilocks.Row)
    (assignment : Fin finalColumns → F) :
    (rowImage row).Holds assignment ↔
      row.Holds
        (StableRows.pulledAssignment
          (NumericBridge.finiteColumnIndex localColumns_positive)
          (decodedLocalAssignment assignment)) := by
  unfold RowImage.Holds Row.Holds rowImage
  rw [evaluate_combinationImage, evaluate_combinationImage,
    evaluate_combinationImage]

theorem rowImage_accepted_iff_holds
    (image : RowImage) (assignment : Fin finalColumns → F) :
    image.Accepted 1 assignment ↔ image.Holds assignment := by
  unfold RowImage.Accepted RowImage.portPoint RowImage.Holds
  exact Rows.evaluate_productPoint_one_eq_zero_iff _ _ _

/-- All normalized row occurrences accept at the selected thirteen-port
product point. -/
def AllRowsAccepted
    (selector : Fin finalColumns) :
    List OwnedRow → (Fin finalColumns → F) → Prop
  | [], _ => True
  | row :: tail, assignment =>
      (rowImage row.row).Accepted (assignment selector) assignment ∧
        AllRowsAccepted selector tail assignment

/-- The selected arm is active and all its normalized algebra rows accept.
-/
def Accepted
    (arm : Arm) (rows : List OwnedRow)
    (assignment : Fin finalColumns → F) : Prop :=
  assignment (selectorColumn arm) = 1 ∧
    AllRowsAccepted (selectorColumn arm) rows assignment

/-- Bare A/B/C satisfaction after source-column substitution. -/
def Satisfies : List OwnedRow → (Fin finalColumns → F) → Prop
  | [], _ => True
  | row :: tail, assignment =>
      (rowImage row.row).Holds assignment ∧ Satisfies tail assignment

theorem allRowsAccepted_iff_satisfies
    (selector : Fin finalColumns)
    (rows : List OwnedRow) (assignment : Fin finalColumns → F)
    (selectorOne : assignment selector = 1) :
    AllRowsAccepted selector rows assignment ↔
      Satisfies rows assignment := by
  induction rows with
  | nil => rfl
  | cons row tail inductionHypothesis =>
      simp only [AllRowsAccepted, Satisfies]
      rw [selectorOne, rowImage_accepted_iff_holds,
        inductionHypothesis]

theorem accepted_implies_satisfies
    (arm : Arm) (rows : List OwnedRow)
    (assignment : Fin finalColumns → F)
    (accepted : Accepted arm rows assignment) :
    Satisfies rows assignment := by
  exact (allRowsAccepted_iff_satisfies
    (selectorColumn arm) rows assignment accepted.1).mp accepted.2

theorem satisfies_iff_typed
    (rows : List OwnedRow) (assignment : Fin finalColumns → F) :
    Satisfies rows assignment ↔
      Nightstream.Implementation.Lowering.Goldilocks.Satisfies rows
        (StableRows.pulledAssignment
          (NumericBridge.finiteColumnIndex localColumns_positive)
          (decodedLocalAssignment assignment)) := by
  induction rows with
  | nil => rfl
  | cons row tail inductionHypothesis =>
      simp only [Satisfies,
        Nightstream.Implementation.Lowering.Goldilocks.satisfies_cons]
      rw [rowImage_holds_iff, inductionHypothesis]

def productionRows : List OwnedRow :=
  NumericBridge.ownedRows Generated.sourceRows

@[simp] theorem productionRows_length : productionRows.length = 49626 := by
  change
    (ownedRowsFrom .prelude 0 NumericBridge.sourceColumn
      Generated.sourceRows).length = 49626
  rw [ownedRowsFrom_length, Generated.sourceRows_length]

def ProductionAccepted
    (arm : Arm) (assignment : Fin finalColumns → F) : Prop :=
  Accepted arm productionRows assignment

/-- All normalized A/B/C images use the one decoded local assignment and
therefore imply every authoritative generated source row. -/
theorem satisfies_implies_source_rows
    (assignment : Fin finalColumns → F)
    (satisfied : Satisfies productionRows assignment) :
    R1CS.Satisfies Generated.sourceRows
      (Generated.numericAssignment (decodedLocalAssignment assignment)) := by
  let pulled :=
    StableRows.pulledAssignment
      (NumericBridge.finiteColumnIndex localColumns_positive)
      (decodedLocalAssignment assignment)
  have typed :
      Nightstream.Implementation.Lowering.Goldilocks.Satisfies
        productionRows pulled := by
    exact (satisfies_iff_typed productionRows assignment).mp satisfied
  have numericPulled :
      R1CS.Satisfies Generated.sourceRows
        (NumericRowBridge.numericAssignment NumericBridge.sourceColumn
          pulled) := by
    change
      Nightstream.Implementation.Lowering.Goldilocks.Satisfies
        (ownedRowsFrom .prelude 0 NumericBridge.sourceColumn
          Generated.sourceRows) pulled at typed
    exact (ownedRowsFrom_satisfies_iff .prelude 0
      NumericBridge.sourceColumn Generated.sourceRows pulled).mp
      typed
  apply (NumericBridge.satisfies_iff_of_agree Generated.sourceRows
    (by simpa [localColumns] using Generated.sourceRows_below) ?_).mp
    numericPulled
  intro column bounded
  unfold pulled NumericRowBridge.numericAssignment
    StableRows.pulledAssignment Generated.numericAssignment
  rw [NumericBridge.finiteColumnIndex_sourceColumn_of_lt
    localColumns_positive bounded]
  have generatedBound : column < Generated.columns := by
    simpa [localColumns] using bounded
  rw [dif_pos generatedBound]
  congr 2

theorem productionAccepted_implies_source_rows
    (arm : Arm) (assignment : Fin finalColumns → F)
    (accepted : ProductionAccepted arm assignment) :
    R1CS.Satisfies Generated.sourceRows
      (Generated.numericAssignment (decodedLocalAssignment assignment)) := by
  exact satisfies_implies_source_rows assignment
    (accepted_implies_satisfies arm productionRows assignment accepted)

/-- Active normalized algebra rows imply the concrete production family
phase on the same final assignment. The constant-one premise is separate
from the parity selector premise. -/
theorem productionAccepted_implies_concrete_phase
    (arm : Arm) (assignment : Fin finalColumns → F)
    (constantOne : assignment ⟨0, finalColumns_positive⟩ = 1)
    (accepted : ProductionAccepted arm assignment)
    (range : ∀ source lane,
      Generated.numericAssignment (decodedLocalAssignment assignment)
        (Generated.layout.challengeSymbol source lane) < 5)
    (inputSetup : InputBindingSetup)
    (before after : FamilyState) (family : Family)
    (challengesExact :
      decodedChallenges Generated.layout
          (Generated.numericAssignment (decodedLocalAssignment assignment))
          range =
        before.challenges)
    (cursorExact : before.familyCursor =
      ProductPiRlcAlgebraRows.familyOrdinal family)
    (transition : FamilyTransition inputSetup before after family
      (decodedInputs Generated.layout
        (Generated.numericAssignment (decodedLocalAssignment assignment))
        (Generated.numericAssignment_canonical
          (decodedLocalAssignment assignment)))
      (ProductPiRlcRingCombinationSound.outputRing Generated.layout
        (Generated.numericAssignment (decodedLocalAssignment assignment))
        (Generated.numericAssignment_canonical
          (decodedLocalAssignment assignment)))) :
    FamilyPhaseRelation inputSetup before after family
      (decodedInputs Generated.layout
        (Generated.numericAssignment (decodedLocalAssignment assignment))
        (Generated.numericAssignment_canonical
          (decodedLocalAssignment assignment)))
      (ProductPiRlcRingCombinationSound.outputRing Generated.layout
        (Generated.numericAssignment (decodedLocalAssignment assignment))
        (Generated.numericAssignment_canonical
          (decodedLocalAssignment assignment))) := by
  have decodedOne :
      decodedLocalAssignment assignment Generated.one = 1 := by
    simpa [Generated.one] using
      (decodedLocalAssignment_zero assignment).trans constantOne
  exact local_rows_imply_concrete_phase
    (Generated.numericAssignment_canonical
      (decodedLocalAssignment assignment))
    (Generated.numericAssignment_one
      (decodedLocalAssignment assignment) decodedOne)
    range
    (productionAccepted_implies_source_rows arm assignment accepted)
    inputSetup before after family challengesExact cursorExact transition

end Normalized

end Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedAlgebraRows
