import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiCcsCoordinateBindingOpeningRows
import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiCcsCoordinateBindingOutputRows

/-!
Contract: complete ordered source rows for one production PiCCS
variable-coordinate commitment phase.

Assurance tier: generated-row soundness bridge.

Owns the two verifier-fixed commitment shape rows, the exact Rust row-family
order, its row census, and the theorem that all accepted rows determine the
shape, source openings, and 108 direct masked Module-SIS output fields.

Does not own Rust trace conformance, ChaCha8 implementation conformance,
phase scheduling, commitment accumulation, public-state placement, or
recursive lifecycle integration.

Emits constraints: yes.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingCompleteRows

open Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBinding
open Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingOpeningRows
open Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingOutputRows
open Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingRows
open Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingSetup
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Protocol.Nebula.CompactCommit

/-- Rust `alloc_commitment` emits these two constant rows before it allocates
the 108 commitment coordinates. -/
def shapePins (layout : Layout) : List AffinePins.Pin :=
  [.constant layout.dColumn ringDegree,
   .constant layout.kappaColumn verifierRows]

def shapeRows (layout : Layout) : List Row :=
  AffinePins.rows (shapePins layout)

theorem shapePins_canonical (layout : Layout) :
    AffinePins.PinsCanonical (shapePins layout) := by
  intro pin member
  simp [shapePins] at member
  rcases member with rfl | rfl
  · change 0 < ringDegree ∧ ringDegree < goldilocksP
    decide
  · change 0 < verifierRows ∧ verifierRows < goldilocksP
    decide

theorem shapeRows_length (layout : Layout) :
    (shapeRows layout).length = 2 := by
  simp [shapeRows, AffinePins.rows, shapePins]

/-- The two accepted rows derive the fixed rank-two degree-54 commitment
shape. Shape values are not prover-selected metadata. -/
theorem shape_exact
    {layout : Layout} {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies (shapeRows layout) assignment) :
    assignment layout.dColumn = ringDegree ∧
      assignment layout.kappaColumn = verifierRows := by
  have facts := AffinePins.rows_sound
    (shapePins_canonical layout) canonical one satisfies
  constructor
  · simpa [AffinePins.Pin.Holds] using
      facts (.constant layout.dColumn ringDegree) (by simp [shapePins])
  · simpa [AffinePins.Pin.Holds] using
      facts (.constant layout.kappaColumn verifierRows) (by simp [shapePins])

theorem coordinateRows_length
    (production : ProductionSetup) (layout : Layout) :
    (coordinateBlock production layout).rows.length = 108 := by
  rw [SeededPhi81.Block.rows_length, coordinateBlock_kappa]
  decide

/-- Exact order emitted by Rust `enforce_commit_coordinate_fields`:
zero word, active canonical openings, commitment shape, seeded map. -/
def rows (production : ProductionSetup) (layout : Layout) : List Row :=
  sourceRows layout ++
    (shapeRows layout ++ (coordinateBlock production layout).rows)

theorem rows_length
    (production : ProductionSetup) (layout : Layout) :
    (rows production layout).length =
      41 + layout.activeFields.length * 124 + 2 + 108 := by
  simp only [rows, List.length_append, sourceRows_length,
    shapeRows_length, coordinateRows_length]

theorem production_rows_length
    (production : ProductionSetup) (layout : Layout)
    (activeCount : layout.activeFields.length = 1024) :
    (rows production layout).length = 127127 := by
  rw [rows_length, activeCount]

private theorem source_satisfies
    {production : ProductionSetup} {layout : Layout}
    {assignment : Nat → Nat}
    (satisfies : Satisfies (rows production layout) assignment) :
    Satisfies (sourceRows layout) assignment := by
  intro row member
  exact satisfies row (List.mem_append_left _ member)

private theorem shape_satisfies
    {production : ProductionSetup} {layout : Layout}
    {assignment : Nat → Nat}
    (satisfies : Satisfies (rows production layout) assignment) :
    Satisfies (shapeRows layout) assignment := by
  intro row member
  exact satisfies row
    (List.mem_append_right _ (List.mem_append_left _ member))

private theorem coordinate_satisfies
    {production : ProductionSetup} {layout : Layout}
    {assignment : Nat → Nat}
    (satisfies : Satisfies (rows production layout) assignment) :
    Satisfies (coordinateBlock production layout).rows assignment := by
  intro row member
  exact satisfies row
    (List.mem_append_right _ (List.mem_append_right _ member))

/-- Semantic result of the complete ordered row family. -/
structure Exact
    (production : ProductionSetup) (layout : Layout)
    (assignment : Nat → Nat) (fields : Fields) : Prop where
  d : assignment layout.dColumn = ringDegree
  kappa : assignment layout.kappaColumn = verifierRows
  source : SourceColumnsExact layout assignment fields
  output : ∀ output : Fin verifierRows, ∀ coordinate : Fin ringDegree,
    assignment (layout.outputColumn (outputIndex output coordinate)) =
      (maskedConcreteBinding production fields layout.selected
        (outputIndex output coordinate)).val

/-- Main soundness theorem. Every conclusion comes from the active source
placement and the accepted ordered rows; no claimed commitment is a premise. -/
theorem rows_sound
    {production : ProductionSetup} {layout : Layout}
    {assignment : Nat → Nat} {fields : Fields}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (placed : ActiveFieldsPlaced layout assignment fields)
    (satisfies : Satisfies (rows production layout) assignment) :
    Exact production layout assignment fields := by
  have sourceExact := sourceColumnsExact_of_rows canonical one placed
    (source_satisfies satisfies)
  have shapeExact := shape_exact canonical one (shape_satisfies satisfies)
  refine ⟨shapeExact.1, shapeExact.2, sourceExact, ?_⟩
  intro output coordinate
  exact compact_output_exact_of_rows canonical one sourceExact
    (coordinate_satisfies satisfies) output coordinate

end Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingCompleteRows
