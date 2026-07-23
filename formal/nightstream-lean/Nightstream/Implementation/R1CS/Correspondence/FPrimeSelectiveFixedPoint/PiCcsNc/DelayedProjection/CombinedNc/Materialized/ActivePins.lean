import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.ActivePins
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.SelectiveArtifactPairs.Artifact
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.Artifact.Row.SelectorComposition
import Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.Assignment

/-!
Exact active pins for the bounded production combined-NC profile.

Owns: the fixed source/final dimensions; the ordered 128/128/14 public-write
address certificate; its interpretation as the typed 270-coordinate public
carrier under an explicit live-encoder binding; the constant and three
selector coordinates; the exact three selector-domain rows and one
selector-total row; unique equation ownership; fail-closed decoding; and the
coefficient-level derivation of the recursive selector pin.

Does not own: proof that Rust applied the generated public writes to the
normalized assignment, the selected combined-NC rows, transcript/state
authority, raw-child authority, commitment binding, costs, or row removal.

Emits constraints: none.

The constant-one equation cannot be derived from the selector rows: those
homogeneous rows also accept the all-zero assignment.  `EncoderPrefixBound`
therefore names the direct-dataflow boundary for the constant and the two
inactive selectors.  The third selector is deliberately not a premise; it is
derived from the exact selector-total row.
-/

/-!
| Stable stage path | Obligation | Authority class |
|---|---|---|
| `f_prime.pi_ccs_nc.delayed.combined.active_pins` | Decode and validate the generated active-pin records used by the materialized bridge. | checked artifact |

-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.ActivePins

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc
open Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial.Components
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial.Ports
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial.Rows
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial.Semantics
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.SelectorComposition.Semantics
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.Decoder
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.Semantics
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.SelectiveArtifactPairs

abbrev generatedPinsRaw : RawActivePins :=
  Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.ActivePins.raw

abbrev generatedPackedCoordinates : List RawPackedPublicCoordinate :=
  Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.ActivePins.packedCoordinates

abbrev generatedPackedChunk0 : List RawPackedPublicCoordinate :=
  Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.ActivePins.PackedCoordinates.Chunk0.values

abbrev generatedPackedChunk1 : List RawPackedPublicCoordinate :=
  Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.ActivePins.PackedCoordinates.Chunk1.values

abbrev generatedPackedChunk2 : List RawPackedPublicCoordinate :=
  Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.ActivePins.PackedCoordinates.Chunk2.values

/-! ## Exact fixed profile and public-write schedule -/

def FixedProfileHeader (pins : RawActivePins) : Prop :=
  pins.schemaVersion = supportedSchemaVersion ∧
  pins.sourceRows = Metadata.sourceRelationRows ∧
  pins.sourceColumns = Metadata.sourceRelationColumns ∧
  pins.finalRows = Metadata.finalRelationRows ∧
  pins.finalColumns = Metadata.finalRelationColumns ∧
  pins.constantOneColumn = 0 ∧
  pins.constantOneValue = 1 ∧
  pins.selectorColumns = [270, 271, 272] ∧
  pins.recursiveSelectorValues = [0, 0, 1] ∧
  pins.packedLaneCount = activeLaneCount ∧
  pins.packedBlockCount = 5 ∧
  pins.publicCoordinateCount = 270 ∧
  pins.selectorDomainRows.length = 3

instance (pins : RawActivePins) : Decidable (FixedProfileHeader pins) := by
  unfold FixedProfileHeader
  infer_instance

/-- Certificate input: one proof-free `RawActivePins` header containing four
small sparse rows.  No decoded or proof-carrying structure is evaluated. -/
theorem generated_header_exact : FixedProfileHeader generatedPinsRaw := by
  native_decide

theorem generated_dimensions_agree :
    generatedPinsRaw.sourceRows = Metadata.sourceRelationRows ∧
    generatedPinsRaw.sourceColumns = Metadata.sourceRelationColumns ∧
    generatedPinsRaw.finalRows = Metadata.finalRelationRows ∧
    generatedPinsRaw.finalColumns = Metadata.finalRelationColumns :=
  ⟨generated_header_exact.2.1, generated_header_exact.2.2.1,
    generated_header_exact.2.2.2.1, generated_header_exact.2.2.2.2.1⟩

theorem generated_constant_column_exact :
    generatedPinsRaw.constantOneColumn = 0 :=
  generated_header_exact.2.2.2.2.2.1

theorem generated_constant_value_exact :
    generatedPinsRaw.constantOneValue = 1 :=
  generated_header_exact.2.2.2.2.2.2.1

def expectedPublicSource (column : Nat) : RawActivePublicSource :=
  if column = 0 then .constantOne
  else if column < 257 then .sourceField column
  else .fixedZero

def expectedPackedCoordinate (column : Nat) : RawPackedPublicCoordinate :=
  { schemaVersion := supportedSchemaVersion
    column
    block := column / activeLaneCount
    lane := column % activeLaneCount
    source := expectedPublicSource column }

def expectedPackedChunk (start count : Nat) :
    List RawPackedPublicCoordinate :=
  (List.range count).map fun offset => expectedPackedCoordinate (start + offset)

def PackedCoordinateValid (coordinate : RawPackedPublicCoordinate) : Prop :=
  coordinate.schemaVersion = supportedSchemaVersion ∧
  coordinate.column < 270 ∧
  coordinate.block < 5 ∧
  coordinate.lane < activeLaneCount ∧
  coordinate.column =
    coordinate.block * activeLaneCount + coordinate.lane ∧
  coordinate.source = expectedPublicSource coordinate.column

instance (coordinate : RawPackedPublicCoordinate) :
    Decidable (PackedCoordinateValid coordinate) := by
  unfold PackedCoordinateValid
  infer_instance

structure DecodedPackedCoordinate where
  raw : RawPackedPublicCoordinate
  valid : PackedCoordinateValid raw

def decodePackedCoordinate (raw : RawPackedPublicCoordinate) :
    Option DecodedPackedCoordinate :=
  if valid : PackedCoordinateValid raw then some ⟨raw, valid⟩ else none

theorem decodePackedCoordinate_of_valid {raw : RawPackedPublicCoordinate}
    (valid : PackedCoordinateValid raw) :
    ∃ decoded, decodePackedCoordinate raw = some decoded := by
  exact ⟨⟨raw, valid⟩, by simp [decodePackedCoordinate, valid]⟩

/-- Certificate input: exactly 128 proof-free
`RawPackedPublicCoordinate` records. -/
theorem generated_packed_chunk0_exact :
    generatedPackedChunk0 =
      expectedPackedChunk 0 128 := by
  native_decide

/-- Certificate input: exactly 128 proof-free
`RawPackedPublicCoordinate` records. -/
theorem generated_packed_chunk1_exact :
    generatedPackedChunk1 =
      expectedPackedChunk 128 128 := by
  native_decide

/-- Certificate input: exactly 14 proof-free
`RawPackedPublicCoordinate` records. -/
theorem generated_packed_chunk2_exact :
    generatedPackedChunk2 =
      expectedPackedChunk 256 14 := by
  native_decide

/-- Certificate input: exactly 128 proof-free packed-coordinate records. -/
theorem generated_packed_chunk0_valid :
    ∀ coordinate ∈ generatedPackedChunk0,
      PackedCoordinateValid coordinate := by
  native_decide

/-- Certificate input: exactly 128 proof-free packed-coordinate records. -/
theorem generated_packed_chunk1_valid :
    ∀ coordinate ∈ generatedPackedChunk1,
      PackedCoordinateValid coordinate := by
  native_decide

/-- Certificate input: exactly 14 proof-free packed-coordinate records. -/
theorem generated_packed_chunk2_valid :
    ∀ coordinate ∈ generatedPackedChunk2,
      PackedCoordinateValid coordinate := by
  native_decide

/-- Kernel-only composition of the three bounded certificates.  There is no
270-record native computation here. -/
theorem generated_packed_coordinates_exact :
    generatedPackedCoordinates =
      expectedPackedChunk 0 128 ++
      expectedPackedChunk 128 128 ++
      expectedPackedChunk 256 14 := by
  unfold generatedPackedCoordinates
  unfold Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.ActivePins.packedCoordinates
  unfold Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.ActivePins.PackedCoordinates.values
  change generatedPackedChunk0 ++ generatedPackedChunk1 ++
      generatedPackedChunk2 =
    expectedPackedChunk 0 128 ++ expectedPackedChunk 128 128 ++
      expectedPackedChunk 256 14
  rw [generated_packed_chunk0_exact, generated_packed_chunk1_exact,
    generated_packed_chunk2_exact]

theorem generated_packed_coordinate_count :
    generatedPackedCoordinates.length = 270 := by
  rw [generated_packed_coordinates_exact]
  simp [expectedPackedChunk]

theorem generated_packed_chunk_lengths :
    generatedPackedChunk0.length = 128 ∧
    generatedPackedChunk1.length = 128 ∧
    generatedPackedChunk2.length = 14 := by
  rw [generated_packed_chunk0_exact, generated_packed_chunk1_exact,
    generated_packed_chunk2_exact]
  simp [expectedPackedChunk]

theorem generated_packed_chunk_maximum :
    generatedPackedChunk0.length ≤ 128 ∧
    generatedPackedChunk1.length ≤ 128 ∧
    generatedPackedChunk2.length ≤ 128 := by
  rw [generated_packed_chunk_lengths.1,
    generated_packed_chunk_lengths.2.1,
    generated_packed_chunk_lengths.2.2]
  omega

theorem generated_packed_coordinate_decodes
    {coordinate : RawPackedPublicCoordinate}
    (member : coordinate ∈ generatedPackedCoordinates) :
    ∃ decoded, decodePackedCoordinate coordinate = some decoded := by
  unfold generatedPackedCoordinates at member
  unfold Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.ActivePins.packedCoordinates at member
  unfold Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.ActivePins.PackedCoordinates.values at member
  change coordinate ∈ generatedPackedChunk0 ++ generatedPackedChunk1 ++
    generatedPackedChunk2 at member
  rcases List.mem_append.mp member with firstTwo | chunk2
  · rcases List.mem_append.mp firstTwo with chunk0 | chunk1
    · exact decodePackedCoordinate_of_valid
        (generated_packed_chunk0_valid coordinate chunk0)
    · exact decodePackedCoordinate_of_valid
        (generated_packed_chunk1_valid coordinate chunk1)
  · exact decodePackedCoordinate_of_valid
      (generated_packed_chunk2_valid coordinate chunk2)

/-! ## Exact selector rows and ownership -/

def emptyPort : RawPort := { explicit := [], geometric := [] }

def unitPort (column : Nat) : RawPort :=
  { explicit := [{ column, coefficient := 1 }], geometric := [] }

def expectedSelectorPort (arm : Fin 3) (port : Fin selectivePortCount) :
    RawPort :=
  if port.val = 0 then unitPort (270 + arm.val)
  else if port.val = 1 then unitPort 0
  else emptyPort

def expectedSelectorRow (arm : Fin 3) : RawEmittedRow :=
  { schemaVersion := supportedSchemaVersion
    rows := Metadata.finalRelationRows
    columns := Metadata.finalRelationColumns
    emittedRow := arm.val
    runIndex := 0
    family := .selectorDomain
    arm := none
    ports := List.ofFn (expectedSelectorPort arm) }

def negativeOneWord : Nat := 18446744069414584320

def selectorTotalPort : RawPort :=
  { explicit :=
      [ { column := 0, coefficient := negativeOneWord }
      , { column := 270, coefficient := 1 }
      , { column := 271, coefficient := 1 }
      , { column := 272, coefficient := 1 }
      ]
    geometric := [] }

def expectedOneHotPort (port : Fin selectivePortCount) : RawPort :=
  if port.val = 1 then unitPort 0
  else if port.val = 4 then selectorTotalPort
  else emptyPort

def expectedOneHotRow : RawEmittedRow :=
  { schemaVersion := supportedSchemaVersion
    rows := Metadata.finalRelationRows
    columns := Metadata.finalRelationColumns
    emittedRow := 4729579
    runIndex := 5
    family := .oneHot
    arm := none
    ports := List.ofFn expectedOneHotPort }

def expectedActiveRow (index : Fin 4) : RawEmittedRow :=
  if selector : index.val < 3 then
    expectedSelectorRow ⟨index.val, selector⟩
  else
    expectedOneHotRow

def expectedActiveRows : List RawEmittedRow :=
  List.ofFn expectedActiveRow

def generatedActiveRows : List RawEmittedRow :=
  generatedPinsRaw.selectorDomainRows ++ [generatedPinsRaw.oneHotRow]

/-- Certificate input: exactly four proof-free `RawEmittedRow` records, each
with thirteen sparse ports. -/
theorem generated_active_rows_exact :
    generatedActiveRows = expectedActiveRows := by
  native_decide

theorem generated_active_row_count : generatedActiveRows.length = 4 := by
  rw [generated_active_rows_exact]
  simp [expectedActiveRows]

theorem expectedActiveRow_emittedRow (index : Fin 4) :
    (expectedActiveRow index).emittedRow =
      if index.val < 3 then index.val else 4729579 := by
  by_cases selector : index.val < 3
  · simp [expectedActiveRow, selector, expectedSelectorRow]
  · simp [expectedActiveRow, selector, expectedOneHotRow]

theorem expectedActiveRow_injective : Function.Injective expectedActiveRow := by
  intro left right equal
  have emitted := congrArg RawEmittedRow.emittedRow equal
  rw [expectedActiveRow_emittedRow, expectedActiveRow_emittedRow] at emitted
  by_cases leftSelector : left.val < 3
  · by_cases rightSelector : right.val < 3
    · rw [if_pos leftSelector, if_pos rightSelector] at emitted
      exact Fin.ext emitted
    · rw [if_pos leftSelector, if_neg rightSelector] at emitted
      have leftBound := left.isLt
      omega
  · by_cases rightSelector : right.val < 3
    · rw [if_neg leftSelector, if_pos rightSelector] at emitted
      have rightBound := right.isLt
      omega
    · apply Fin.ext
      have leftBound := left.isLt
      have rightBound := right.isLt
      omega

/-- Every generated active selector row has exactly one coefficient owner. -/
theorem generated_active_row_has_unique_owner {row : RawEmittedRow}
    (member : row ∈ generatedActiveRows) :
    ∃ index : Fin 4,
      row = expectedActiveRow index ∧
      ∀ other : Fin 4, row = expectedActiveRow other → other = index := by
  rw [generated_active_rows_exact, expectedActiveRows, List.mem_ofFn] at member
  rcases member with ⟨index, expectedEq⟩
  refine ⟨index, expectedEq.symm, ?_⟩
  intro other otherEq
  exact (expectedActiveRow_injective (expectedEq.trans otherEq)).symm

private theorem fin3_cases {predicate : Fin 3 → Prop}
    (case0 : predicate 0) (case1 : predicate 1) (case2 : predicate 2) :
    ∀ index, predicate index := by
  intro index
  refine Fin.cases case0 ?_ index
  intro index
  refine Fin.cases case1 ?_ index
  intro index
  exact Fin.cases case2
    (fun impossible : Fin 0 => Fin.elim0 impossible) index

theorem expectedSelectorRow_valid :
    ∀ arm : Fin 3, RawEmittedRowValid (expectedSelectorRow arm) := by
  apply fin3_cases
  · -- Certificate input: one proof-free row with exactly thirteen ports.
    native_decide
  · -- Certificate input: one proof-free row with exactly thirteen ports.
    native_decide
  · -- Certificate input: one proof-free row with exactly thirteen ports.
    native_decide

/-- Certificate input: one proof-free row with exactly thirteen ports. -/
theorem expectedOneHotRow_valid : RawEmittedRowValid expectedOneHotRow := by
  native_decide

theorem expectedActiveRow_valid (index : Fin 4) :
    RawEmittedRowValid (expectedActiveRow index) := by
  by_cases selector : index.val < 3
  · simpa [expectedActiveRow, selector] using
      expectedSelectorRow_valid ⟨index.val, selector⟩
  · simpa [expectedActiveRow, selector] using expectedOneHotRow_valid

/-- Generated row decoding is fail-closed and retains the unique physical
owner.  The decoder is supplied by a generic kernel theorem; no decoded
structure is passed to native evaluation. -/
theorem generated_active_row_decodes {row : RawEmittedRow}
    (member : row ∈ generatedActiveRows) :
    ∃ index decoded,
      row = expectedActiveRow index ∧
      decodeEmittedRow row = some decoded ∧
      ∀ other : Fin 4, row = expectedActiveRow other → other = index := by
  rcases generated_active_row_has_unique_owner member with
    ⟨index, exactRow, unique⟩
  rcases decodeEmittedRow_of_valid (expectedActiveRow_valid index) with
    ⟨decoded, decodes⟩
  refine ⟨index, decoded, exactRow, ?_, unique⟩
  simpa [exactRow] using decodes

/-! ## Coefficient semantics and active assignment pins -/

def GeneratedActiveRowsSatisfy (assignment : Nat → Nat) : Prop :=
  ∀ raw ∈ generatedActiveRows,
    ∀ decoded, decodeEmittedRow raw = some decoded →
      EmittedRowHolds decoded assignment

theorem expectedOneHotRow_member : expectedOneHotRow ∈ generatedActiveRows := by
  rw [generated_active_rows_exact, expectedActiveRows, List.mem_ofFn]
  refine ⟨⟨3, by decide⟩, ?_⟩
  simp [expectedActiveRow]

@[simp] theorem fieldResidue_zero : fieldResidue 0 = 0 := by decide

@[simp] theorem fieldResidue_one : fieldResidue 1 = 1 := by decide

@[simp] theorem fieldResidue_negativeOne :
    fieldResidue negativeOneWord = (-1 : F) := by
  decide

@[simp] theorem expectedOneHotRow_portForm (port : Fin selectivePortCount) :
    rawEmittedPortLinearForm expectedOneHotRow port =
      rawPortLinearForm (expectedOneHotPort port) := by
  unfold rawEmittedPortLinearForm expectedOneHotRow
  rw [List.getElem?_eq_getElem (by simp), List.getElem_ofFn]

@[simp] theorem eval_emptyPort (assignment : Nat → F) :
    evalLinearForm assignment (rawPortLinearForm emptyPort) = 0 := by
  simp [emptyPort, rawPortLinearForm, evalLinearForm]

@[simp] theorem eval_unitPort (assignment : Nat → F) (column : Nat) :
    evalLinearForm assignment (rawPortLinearForm (unitPort column)) =
      assignment column := by
  simp [unitPort, rawPortLinearForm, rawTermLinearForm, evalLinearForm,
    termValue, Fin.one_mul, Fin.zero_add]

theorem eval_selectorTotalPort (assignment : Nat → F) :
    evalLinearForm assignment (rawPortLinearForm selectorTotalPort) =
      -assignment 0 + assignment 270 + assignment 271 + assignment 272 := by
  calc
    evalLinearForm assignment (rawPortLinearForm selectorTotalPort) =
        -assignment 0 +
          (assignment 270 + (assignment 271 + assignment 272)) := by
      simp [selectorTotalPort, rawPortLinearForm, rawTermLinearForm,
        evalLinearForm, termValue, Lean.Grind.Fin.neg_mul,
        Fin.one_mul, Fin.zero_add]
    _ = (-assignment 0 + assignment 270) +
          (assignment 271 + assignment 272) :=
      (Lean.Grind.Fin.add_assoc _ _ _).symm
    _ = -assignment 0 + assignment 270 + assignment 271 + assignment 272 :=
      (Lean.Grind.Fin.add_assoc _ _ _).symm

theorem expectedOneHotPoint_exact {decoded : DecodedEmittedRow}
    (decodes : decodeEmittedRow expectedOneHotRow = some decoded)
    (assignment : Nat → Nat) :
    emittedPoint decoded assignment =
      Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Row.SelectorComposition.selectorTotalPoint
        (fieldResidue (assignment 0))
        (fieldResidue (assignment 270))
        (fieldResidue (assignment 271))
        (fieldResidue (assignment 272)) := by
  rw [emittedPoint_eq_evalRawPorts decodes]
  funext port
  rw [expectedOneHotRow_portForm]
  by_cases generalPort : port = Role.generalSelector.index
  · subst port
    change evalLinearForm (fun column => fieldResidue (assignment column))
        (rawPortLinearForm (unitPort 0)) = _
    rw [eval_unitPort]
    simp [
      Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Row.SelectorComposition.selectorTotalPoint,
      productPoint, sparsePoint, Role.index]
  · by_cases outputPort : port = Role.c.index
    · subst port
      change evalLinearForm (fun column => fieldResidue (assignment column))
          (rawPortLinearForm selectorTotalPort) = _
      rw [eval_selectorTotalPort]
      simp [
        Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Row.SelectorComposition.selectorTotalPoint,
        productPoint, sparsePoint, Role.index]
    · have portValNeOne : port.val ≠ 1 := by
        intro equal
        apply generalPort
        exact Fin.ext equal
      have portValNeFour : port.val ≠ 4 := by
        intro equal
        apply outputPort
        exact Fin.ext equal
      rw [show expectedOneHotPort port = emptyPort by
        simp [expectedOneHotPort, portValNeOne, portValNeFour]]
      rw [eval_emptyPort]
      unfold Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Row.SelectorComposition.selectorTotalPoint
      unfold productPoint sparsePoint
      simp only [List.foldl_cons, List.foldl_nil]
      split
      · rename_i equal
        exact (outputPort equal).elim
      · split
        · rfl
        · split
          · rfl
          · split
            · rename_i equal
              exact (generalPort equal).elim
            · rfl

theorem evaluate_selectorTotalPoint (constant first second third : F) :
    evaluate (Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Row.SelectorComposition.selectorTotalPoint
      constant first second third) =
      -(constant * (-constant + first + second + third)) := by
  unfold Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Row.SelectorComposition.selectorTotalPoint
  rw [evaluate_productPoint]
  simp [productResidual, productPoint, sparsePoint, Role.index,
    Fin.mul_zero, Fin.zero_add]

/-- Direct encoder bindings which the homogeneous selector equations cannot
establish.  The generated header proves these are exactly the constant-one
write and the first two entries of the recursive `[0, 0, 1]` selector. -/
structure EncoderPrefixBound (assignment : Nat → Nat) : Prop where
  constantOne :
    assignment generatedPinsRaw.constantOneColumn =
      generatedPinsRaw.constantOneValue
  firstSelectorZero : assignment 270 = 0
  secondSelectorZero : assignment 271 = 0

theorem EncoderPrefixBound.constantOne_fixed {assignment : Nat → Nat}
    (bound : EncoderPrefixBound assignment) : assignment 0 = 1 := by
  calc
    assignment 0 = assignment generatedPinsRaw.constantOneColumn :=
      congrArg assignment generated_constant_column_exact.symm
    _ = generatedPinsRaw.constantOneValue := bound.constantOne
    _ = 1 := generated_constant_value_exact

theorem fieldResidue_injective_of_canonical {left right : Nat}
    (leftCanonical : left < goldilocksP)
    (rightCanonical : right < goldilocksP)
    (equal : fieldResidue left = fieldResidue right) :
    left = right := by
  have values := congrArg Fin.val equal
  change left % goldilocksModulus = right % goldilocksModulus at values
  have modulusEquality : goldilocksP = goldilocksModulus := rfl
  rw [← modulusEquality, Nat.mod_eq_of_lt leftCanonical,
    Nat.mod_eq_of_lt rightCanonical] at values
  exact values

/-- The exact selector-total coefficients, not the `.oneHot` label, force the
remaining recursive selector once the direct constant and inactive selector
writes are fixed. -/
theorem activeRowsSatisfy_implies_steadySelectorOne
    {assignment : Nat → Nat}
    (activeRows : GeneratedActiveRowsSatisfy assignment)
    (encoder : EncoderPrefixBound assignment)
    (steadyCanonical : assignment 272 < goldilocksP) :
    assignment 272 = 1 := by
  rcases decodeEmittedRow_of_valid expectedOneHotRow_valid with
    ⟨decoded, decodes⟩
  have rowZero := activeRows expectedOneHotRow expectedOneHotRow_member
    decoded decodes
  have constantOne := encoder.constantOne_fixed
  have constantField : fieldResidue (assignment 0) = 1 := by
    simp [constantOne]
  have gapZero :
      -(fieldResidue (assignment 0) *
        (-fieldResidue (assignment 0) + fieldResidue (assignment 270) +
          fieldResidue (assignment 271) + fieldResidue (assignment 272))) = 0 := by
    unfold EmittedRowHolds emittedResidual at rowZero
    rw [expectedOneHotPoint_exact decodes,
      evaluate_selectorTotalPoint] at rowZero
    exact rowZero
  have total :=
    (Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Row.SelectorComposition.selectorGap_eq_zero_iff_total
      (fieldResidue (assignment 0))
      (fieldResidue (assignment 270))
      (fieldResidue (assignment 271))
      (fieldResidue (assignment 272)) constantField).1 gapZero
  unfold SelectorTotal at total
  rw [Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Row.SelectorComposition.selectorSum_three]
    at total
  have steadyField : fieldResidue (assignment 272) = 1 := by
    simpa [encoder.firstSelectorZero, encoder.secondSelectorZero] using total
  exact fieldResidue_injective_of_canonical
    steadyCanonical (by decide) (by simpa using steadyField)

/-- The full local premise surface for the actual generated assignment.  The
selected combined-NC rows and the four always-on selector rows remain
separate because the bounded NC projection does not contain emitted rows
`0`, `1`, `2`, or `4729579`. -/
structure GeneratedEmittedAssignmentSatisfies
    (assignment : Nat → Nat) : Prop where
  selectedRows :
    SelectiveArtifactPairs.Artifact.GeneratedEmittedRowsSatisfy assignment
  activeRows : GeneratedActiveRowsSatisfy assignment
  encoder : EncoderPrefixBound assignment
  canonical : ∀ column, assignment column < goldilocksP

/-- Exact pins needed by selected-row soundness, derived from the current
profile's active encoder and physical selector-total row. -/
theorem generatedEmittedAssignmentSatisfies_implies_pins
    {assignment : Nat → Nat}
    (satisfies : GeneratedEmittedAssignmentSatisfies assignment) :
    assignment 0 = 1 ∧
      assignment Metadata.steadySelectorColumn = 1 := by
  have constantOne := satisfies.encoder.constantOne_fixed
  have steadyOne := activeRowsSatisfy_implies_steadySelectorOne
    satisfies.activeRows satisfies.encoder (satisfies.canonical 272)
  exact ⟨constantOne, by simpa [Metadata.steadySelectorColumn] using steadyOne⟩

/-! ## Complete current-profile public-assignment interpretation -/

/-- Field interpretation of one generated active public-write source.

`sourceField` is interpreted through the independently typed legacy
assignment.  The constant and padding sources are verifier-owned. -/
def interpretedActivePublicSource
    (dimensions :
      Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.Dimensions)
    (legacy :
      Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.LegacyAssignment
        dimensions) :
    RawActivePublicSource → F
  | .constantOne => 1
  | .sourceField column =>
      if inRange : column <
          Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.legacyPublicWidth then
        legacy ⟨column, Nat.lt_of_lt_of_le inRange dimensions.legacyPublicFits⟩
      else
        0
  | .fixedZero => 0

/-- The authoritative legacy source carries the conventional constant-one
coordinate.  This semantic premise is stated locally so the current-profile
decoder does not import the stale pre-`Pi_DEC` artifact wrapper. -/
def ActiveSourceConstantOne
    (dimensions :
      Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.Dimensions)
    (legacy :
      Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.LegacyAssignment
        dimensions) : Prop :=
  legacy ⟨0, Nat.lt_of_lt_of_le (by decide)
    dimensions.legacyPublicFits⟩ = 1

/-- Direct-dataflow boundary for the live encoder applying the generated
public-write schedule to the normalized assignment.

The premise ranges over the actual generated records, rather than a caller
supplied semantic public-input proposition.  Rust/R1CS refinement must prove
this field from the encoder writes; source labels and digests alone do not. -/
structure ActivePublicWritesBound
    (assignment : Nat → Nat)
    (dimensions :
      Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.Dimensions)
    (legacy :
      Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.LegacyAssignment
        dimensions) : Prop where
  generatedWrite :
    ∀ coordinate,
      coordinate ∈ generatedPackedCoordinates →
        fieldResidue (assignment coordinate.column) =
          interpretedActivePublicSource dimensions legacy coordinate.source

/-- Every logical public column has its canonical record in the generated
128/128/14 certificate.  This is kernel composition of the three existing
bounded proof-free certificates; it performs no 270-record native
computation. -/
theorem expectedPackedCoordinate_mem_generated (column : Fin 270) :
    expectedPackedCoordinate column.val ∈ generatedPackedCoordinates := by
  rw [generated_packed_coordinates_exact]
  by_cases first : column.val < 128
  · apply List.mem_append.mpr
    left
    apply List.mem_append.mpr
    left
    unfold expectedPackedChunk
    apply List.mem_map.mpr
    exact ⟨column.val, List.mem_range.mpr first, by simp⟩
  · by_cases second : column.val < 256
    · apply List.mem_append.mpr
      left
      apply List.mem_append.mpr
      right
      unfold expectedPackedChunk
      apply List.mem_map.mpr
      refine ⟨column.val - 128, List.mem_range.mpr (by omega), ?_⟩
      apply congrArg expectedPackedCoordinate
      omega
    · apply List.mem_append.mpr
      right
      unfold expectedPackedChunk
      apply List.mem_map.mpr
      refine ⟨column.val - 256, List.mem_range.mpr (by omega), ?_⟩
      apply congrArg expectedPackedCoordinate
      omega

/-- The live-write boundary plus the generated owner certificate determines
the normalized field value of every current-profile public coordinate. -/
theorem activePublicWrite_eq_interpreted
    {assignment : Nat → Nat}
    {dimensions :
      Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.Dimensions}
    {legacy :
      Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.LegacyAssignment
        dimensions}
    (bound : ActivePublicWritesBound assignment dimensions legacy)
    (column : Fin 270) :
    fieldResidue (assignment column.val) =
      interpretedActivePublicSource dimensions legacy
        (expectedPublicSource column.val) := by
  simpa [expectedPackedCoordinate] using
    bound.generatedWrite (expectedPackedCoordinate column.val)
      (expectedPackedCoordinate_mem_generated column)

/-- View the normalized first 270 physical columns as the typed public input
of the independent five-ring carrier. -/
def normalizedPublicInput
    (assignment : Nat → Nat)
    (dimensions :
      Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.Dimensions) :
    Nightstream.SuperNeo.Concrete.Phi81Relation.PublicInput dimensions.shape :=
  fun column => fieldResidue (assignment column.val)

def activePublicColumn
    (dimensions :
      Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.Dimensions)
    (column :
      Fin dimensions.shape.publicWidth) : Fin 270 :=
  ⟨column.val, by
    have bound := column.isLt
    simpa using bound⟩

theorem interpretedActivePublicSource_expected
    (dimensions :
      Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.Dimensions)
    (legacy :
      Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.LegacyAssignment
        dimensions)
    (constantOne :
      ActiveSourceConstantOne dimensions legacy)
    (column : Fin dimensions.shape.publicWidth) :
    interpretedActivePublicSource dimensions legacy
        (expectedPublicSource column.val) =
      Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.expectedPublicInput
        dimensions legacy column := by
  by_cases zero : column.val = 0
  · have columnEq : column = ⟨0, by
        have bound := column.isLt
        simpa using bound⟩ := Fin.ext zero
    rw [columnEq]
    simpa [expectedPublicSource, interpretedActivePublicSource,
      Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.expectedPublicInput,
      ActiveSourceConstantOne]
      using constantOne.symm
  · by_cases inLegacy : column.val < 257
    · simp [expectedPublicSource, interpretedActivePublicSource,
        Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.expectedPublicInput,
        Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.legacyPublicWidth,
        zero, inLegacy]
    · simp [expectedPublicSource, interpretedActivePublicSource,
        Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.expectedPublicInput,
        Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.legacyPublicWidth,
        zero, inLegacy]

/-- Current post-PiDEC public-write refinement into the independent typed
`FPrimeCarrier270` assignment.

The theorem derives all 270 coordinate values from the actual generated owner
records and the explicit live-encoder binding.  It does not infer values from
metadata and does not claim private-assignment, commitment, or CCS/CE
authority. -/
theorem activePublicWritesBound_implies_typedPublicAssignment
    {assignment : Nat → Nat}
    (dimensions :
      Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.Dimensions)
    (legacy :
      Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.LegacyAssignment
        dimensions)
    (bound : ActivePublicWritesBound assignment dimensions legacy)
    (constantOne : ActiveSourceConstantOne dimensions legacy) :
    normalizedPublicInput assignment dimensions =
      Nightstream.SuperNeo.Concrete.Phi81Relation.projectPublicInput
        (Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.assignment
          dimensions legacy) := by
  rw [Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.projectPublicInput_exact]
  funext column
  exact (activePublicWrite_eq_interpreted bound
    (activePublicColumn dimensions column)).trans
      (interpretedActivePublicSource_expected dimensions legacy constantOne column)

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.ActivePins
