import Nightstream.Implementation.Nebula.Commitment.Lanes.ShiftedTernaryEncodingBridge
import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiCcsCoordinateBindingRows
import Nightstream.Implementation.R1CS.Canonical.GoldilocksField
import Nightstream.Implementation.R1CS.Core.AffinePins

/-!
Contract: exact source-authority rows for one production PiCCS
variable-coordinate commitment phase.

Assurance tier: generated-row soundness bridge.

Owns the 41 shared-zero equality rows, the 124-row canonical opening for each
active field in verifier-supplied order, their exact row census, and the proof
that row satisfaction implies `SourceColumnsExact` for the selector block.

Does not own the compact seeded Phi81 output rows, Rust opening-cache
conformance, commitment accumulation, phase scheduling, public-state
placement, or recursive lifecycle integration.

Emits constraints: yes.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingOpeningRows

open Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBinding
open Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingRows
open Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingSetup
open Nightstream.Implementation.Nebula.ShiftedTernaryEncodingBridge
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.ShiftedTernaryCanonicalWord
open Nightstream.Implementation.R1CS.ShiftedTernaryCompiler
open Nightstream.Implementation.R1CS.ShiftedTernarySound
open Nightstream.Protocol.Nebula
open Nightstream.Protocol.Nebula.CompactCommit
open Nightstream.Protocol.Nebula.ShiftedTernary41V1

/-- Rust allocates one 41-coordinate zero word before all active openings. -/
def zeroPins (layout : Layout) : List AffinePins.Pin :=
  List.ofFn fun digit : Fin ShiftedTernary41V1.digitCount =>
    .zero (layout.zeroDigitStart + digit.val)

def zeroRows (layout : Layout) : List Row :=
  AffinePins.rows (zeroPins layout)

/-- Exact 124-row canonical opening emitted for one active source field. -/
def openingBlockRows (layout : Layout) (field : Fin fieldCount) : List Row :=
  canonicalRows.map (Relabel.row
    (OwnerCertificate.shiftedTernaryColumnMap
      (layout.fieldColumn field) (layout.digitStart field)))

/-- Active openings occur in the caller-supplied field-position order. -/
def openingRows (layout : Layout) : List Row :=
  layout.activeFields.flatMap (openingBlockRows layout)

/-- Exact source-row prefix of Rust `enforce_commit_coordinate_fields`. -/
def sourceRows (layout : Layout) : List Row :=
  zeroRows layout ++ openingRows layout

theorem zeroPins_length (layout : Layout) :
    (zeroPins layout).length = ShiftedTernary41V1.digitCount := by
  simp [zeroPins]

theorem zeroRows_length (layout : Layout) :
    (zeroRows layout).length = 41 := by
  simp [zeroRows, AffinePins.rows, zeroPins,
    ShiftedTernary41V1.digitCount]

theorem openingBlockRows_length
    (layout : Layout) (field : Fin fieldCount) :
    (openingBlockRows layout field).length = 124 := by
  simp [openingBlockRows]
  decide

private theorem openingRowsFor_length
    (layout : Layout) (activeFields : List (Fin fieldCount)) :
    (activeFields.flatMap (openingBlockRows layout)).length =
      activeFields.length * 124 := by
  induction activeFields with
  | nil => rfl
  | cons field rest inductionHypothesis =>
      simp [openingBlockRows_length, inductionHypothesis]
      omega

theorem openingRows_length (layout : Layout) :
    (openingRows layout).length = layout.activeFields.length * 124 := by
  exact openingRowsFor_length layout layout.activeFields

theorem sourceRows_length (layout : Layout) :
    (sourceRows layout).length =
      41 + layout.activeFields.length * 124 := by
  simp [sourceRows, zeroRows_length, openingRows_length]

/-- Only active source fields need placement in this phase. -/
def ActiveFieldsPlaced
    (layout : Layout) (assignment : Nat → Nat) (fields : Fields) : Prop :=
  ∀ field ∈ layout.activeFields,
    assignment (layout.fieldColumn field) = (fields field).val

def localAssignment
    (layout : Layout) (assignment : Nat → Nat)
    (field : Fin fieldCount) : Nat → Nat :=
  ShiftedTernaryCanonicalWord.localAssignment assignment
    (layout.fieldColumn field) (layout.digitStart field)

private theorem zero_holds_of_source
    {layout : Layout} {assignment : Nat → Nat}
    (holds : Satisfies (sourceRows layout) assignment) :
    Satisfies (zeroRows layout) assignment := by
  intro row member
  exact holds row (List.mem_append_left _ member)

private theorem opening_holds_of_source
    {layout : Layout} {assignment : Nat → Nat}
    (holds : Satisfies (sourceRows layout) assignment) :
    Satisfies (openingRows layout) assignment := by
  intro row member
  exact holds row (List.mem_append_right _ member)

theorem zeroPins_canonical (layout : Layout) :
    AffinePins.PinsCanonical (zeroPins layout) := by
  intro pin member
  rcases List.mem_ofFn.mp member with ⟨digit, rfl⟩
  trivial

/-- The zero equality rows, rather than an advice convention, authorize every
inactive selector coordinate. -/
theorem zero_word_exact
    {layout : Layout} {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (holds : Satisfies (sourceRows layout) assignment)
    (digit : Fin ShiftedTernary41V1.digitCount) :
    assignment (layout.zeroDigitStart + digit.val) =
      (integerResidue 0).val := by
  have pinFacts := AffinePins.rows_sound
    (zeroPins_canonical layout) canonical one (zero_holds_of_source holds)
  have zeroFact := pinFacts
    (.zero (layout.zeroDigitStart + digit.val))
    (List.mem_ofFn.mpr ⟨digit, rfl⟩)
  simpa [AffinePins.Pin.Holds] using zeroFact

private theorem openingBlock_holds
    {layout : Layout} {assignment : Nat → Nat}
    (holds : Satisfies (openingRows layout) assignment)
    (field : Fin fieldCount) (active : field ∈ layout.activeFields) :
    Satisfies (openingBlockRows layout field) assignment := by
  intro row member
  apply holds row
  unfold openingRows
  exact List.mem_flatMap.mpr ⟨field, active, member⟩

/-- The active field's exact 124 production rows derive its canonical
shifted-ternary opening. -/
theorem opening_of_rows
    {layout : Layout} {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (holds : Satisfies (sourceRows layout) assignment)
    (field : Fin fieldCount) (active : field ∈ layout.activeFields) :
    CanonicalOpening (localAssignment layout assignment field) := by
  apply canonicalOpening_of_canonicalRows
    Nightstream.Implementation.R1CS.Canonical.GoldilocksField.goldilocks_euclidPrime
    (Relabel.canonical canonical)
  · calc
      Relabel.assignment
          (OwnerCertificate.shiftedTernaryColumnMap
            (layout.fieldColumn field) (layout.digitStart field))
          assignment 0 = assignment 0 :=
        ShiftedTernaryCanonicalWord.localAssignment_zero assignment
          (layout.fieldColumn field) (layout.digitStart field)
      _ = 1 := one
  · exact (Relabel.satisfies_mapped_iff _ _ _).mp
      (openingBlock_holds (opening_holds_of_source holds) field active)

/-- An active selector word is the exact canonical signed-digit word of its
placed field. -/
theorem active_digit_exact
    {layout : Layout} {assignment : Nat → Nat} {fields : Fields}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (placed : ActiveFieldsPlaced layout assignment fields)
    (holds : Satisfies (sourceRows layout) assignment)
    (field : Fin fieldCount) (active : field ∈ layout.activeFields)
    (digit : Fin ShiftedTernary41V1.digitCount) :
    assignment (layout.digitStart field + digit.val) =
      (integerResidue (signedDigit (fields field) digit)).val := by
  rw [integerResidue_signedDigit]
  exact productionDigit_eq_protocolDigit
    (fields field) (placed field active)
    (opening_of_rows canonical one holds field active) digit

/-- Main source-row soundness theorem. The selector's complete source
authority is derived from field placement and emitted rows; it is not a
commitment-output premise. -/
theorem sourceColumnsExact_of_rows
    {layout : Layout} {assignment : Nat → Nat} {fields : Fields}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (placed : ActiveFieldsPlaced layout assignment fields)
    (holds : Satisfies (sourceRows layout) assignment) :
    SourceColumnsExact layout assignment fields := by
  constructor
  · intro field active digit
    exact active_digit_exact canonical one placed holds field active digit
  · intro digit
    exact zero_word_exact canonical one holds digit

end Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingOpeningRows
