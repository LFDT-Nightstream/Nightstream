import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.NifsPaper.PiRlc.ClaimShape
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.NifsPaper.PublicCarrier

/-!
Checked extraction of the matrix-indexed active `y_ring` carrier.

Assurance tier: model-level representation bridge.

Owns: exact indexing of one physical evaluation row per semantic matrix, two
base-field limbs per extension coefficient, and all 54 active coefficients;
and decoding that carrier into the canonical matrix-ordered CE array.

Does not own: a production relation profile, proof that a generated claim has
the required shape, evaluation values, matrix truth, padding zeroes,
transcript authority, Rust conformance, R1CS rows, costs, or row removal.

Emits constraints: no.

Authority boundary: the semantic shape fixes the matrix count. `fromClaim`
requires independently checked outer-row and active-width facts before it may
index the physical lists; it never obtains a count from a digest or fills a
missing matrix or coefficient with a default.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.claim.evaluations.rows` | one physical row for every semantic matrix | checked | `row`, `row_mem` |
| `nifs.claim.evaluations.active_columns` | exact `(matrix, coefficient, limb)` column indexing with no default | computed | `column`, `fromClaim` |
| `nifs.claim.evaluations.decode` | column values decode in canonical matrix order | computed | `decode`, `decode_size`, `decode_get` |
| `nifs.claim.evaluations.compatibility` | checked extraction agrees with the legacy list decoder on the same claim | derived | `decode_fromClaim_eq_decodedEvaluations` |
-/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.ClaimEvaluationCarrier

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.Implementation.R1CS.PiDecStrictCompiler
open Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper
open Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.ClaimShape

/-- The proof-carrying index of one semantic matrix in the physical claim. -/
def rowIndex
    {shape : Shape} {claim : ClaimLayout}
    (alignment : Holds shape claim) (matrix : Fin shape.matrixCount) :
    Fin claim.yRingCols.length :=
  ⟨matrix.val, by
    rw [alignment.evaluationCount]
    exact matrix.isLt⟩

/-- One physical row selected without a default. -/
def row
    {shape : Shape} {claim : ClaimLayout}
    (alignment : Holds shape claim) (matrix : Fin shape.matrixCount) :
    List Nat :=
  claim.yRingCols.get (rowIndex alignment matrix)

theorem row_mem
    {shape : Shape} {claim : ClaimLayout}
    (alignment : Holds shape claim) (matrix : Fin shape.matrixCount) :
    row alignment matrix ∈ claim.yRingCols := by
  exact List.get_mem claim.yRingCols (rowIndex alignment matrix)

theorem row_active_width
    {shape : Shape} {claim : ClaimLayout}
    (alignment : Holds shape claim) (matrix : Fin shape.matrixCount) :
    2 * ringDegree <= (row alignment matrix).length := by
  exact alignment.activeEvaluationWidth _ (row_mem alignment matrix)

/-- Flat physical offset of one extension-field coefficient limb. -/
def activeColumnIndex
    (coefficient : Fin ringDegree) (limb : Fin 2) : Nat :=
  2 * coefficient.val + limb.val

theorem activeColumnIndex_lt
    (coefficient : Fin ringDegree) (limb : Fin 2) :
    activeColumnIndex coefficient limb < 2 * ringDegree := by
  unfold activeColumnIndex
  omega

/-- One active physical column, selected only after proving it is present. -/
def column
    {shape : Shape} {claim : ClaimLayout}
    (alignment : Holds shape claim) (matrix : Fin shape.matrixCount)
    (coefficient : Fin ringDegree) (limb : Fin 2) : Nat :=
  (row alignment matrix).get
    ⟨activeColumnIndex coefficient limb,
      Nat.lt_of_lt_of_le (activeColumnIndex_lt coefficient limb)
        (row_active_width alignment matrix)⟩

/-- Intrinsically complete active evaluation-column carrier. -/
structure Columns (shape : Shape) where
  column : Fin shape.matrixCount -> Fin ringDegree -> Fin 2 -> Nat

/-- Extract every active evaluation column from a checked physical claim. -/
def fromClaim
    {shape : Shape} {claim : ClaimLayout}
    (alignment : Holds shape claim) : Columns shape where
  column := column alignment

/-- Decode an intrinsically complete carrier in canonical matrix order. -/
def decode
    {shape : Shape} (assignment : Nat -> Nat) (columns : Columns shape) :
    Array Evaluation :=
  Array.ofFn fun matrix coefficient =>
    ⟨residue (assignment (columns.column matrix coefficient ⟨0, by decide⟩)),
     residue (assignment (columns.column matrix coefficient ⟨1, by decide⟩))⟩

@[simp] theorem decode_size
    {shape : Shape} (assignment : Nat -> Nat) (columns : Columns shape) :
    (decode assignment columns).size = shape.matrixCount := by
  simp [decode]

@[simp] theorem decode_get
    {shape : Shape} (assignment : Nat -> Nat) (columns : Columns shape)
    (matrix : Fin shape.matrixCount) :
    (decode assignment columns)[matrix.val]'(by
      simpa only [decode_size] using matrix.isLt) =
      fun coefficient =>
        ⟨residue (assignment
            (columns.column matrix coefficient ⟨0, by decide⟩)),
         residue (assignment
            (columns.column matrix coefficient ⟨1, by decide⟩))⟩ := by
  simp [decode]

theorem column_eq_getD
    {shape : Shape} {claim : ClaimLayout}
    (alignment : Holds shape claim) (matrix : Fin shape.matrixCount)
    (coefficient : Fin ringDegree) (limb : Fin 2) :
    column alignment matrix coefficient limb =
      (row alignment matrix).getD (activeColumnIndex coefficient limb) 0 := by
  unfold column
  rw [List.get_eq_getElem, List.getElem_eq_getD]

theorem decode_fromClaim_get
    {shape : Shape} {claim : ClaimLayout}
    (assignment : Nat -> Nat) (alignment : Holds shape claim)
    (matrix : Fin shape.matrixCount) :
    (decode assignment (fromClaim alignment))[matrix.val]'(by
      simpa only [decode_size] using matrix.isLt) =
      decodedEvaluation assignment (row alignment matrix) := by
  rw [decode_get]
  funext coefficient
  apply k_eq_of_coeffs
  · change residue (assignment
        (column alignment matrix coefficient ⟨0, by decide⟩)) =
      residue (assignment
        ((row alignment matrix).getD (2 * coefficient.val) 0))
    rw [column_eq_getD]
    rfl
  · change residue (assignment
        (column alignment matrix coefficient ⟨1, by decide⟩)) =
      residue (assignment
        ((row alignment matrix).getD (2 * coefficient.val + 1) 0))
    rw [column_eq_getD]
    rfl

/-- The checked, no-default carrier agrees with the existing total list
decoder when both read the same physically aligned claim. -/
theorem decode_fromClaim_eq_decodedEvaluations
    {shape : Shape} {claim : ClaimLayout}
    (assignment : Nat -> Nat) (alignment : Holds shape claim) :
    decode assignment (fromClaim alignment) =
      decodedEvaluations assignment claim := by
  apply Array.ext
  · simp [decodedEvaluations, alignment.evaluationCount]
  · intro index leftLt rightLt
    have indexLt : index < shape.matrixCount := by
      simpa only [decode_size] using leftLt
    let matrix : Fin shape.matrixCount := ⟨index, indexLt⟩
    calc
      (decode assignment (fromClaim alignment))[index]'leftLt =
          decodedEvaluation assignment (row alignment matrix) := by
        simpa [matrix] using decode_fromClaim_get assignment alignment matrix
      _ = (decodedEvaluations assignment claim)[index]'rightLt := by
        simp only [decodedEvaluations, List.getElem_toArray, List.getElem_map]
        congr 1

end Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.ClaimEvaluationCarrier
