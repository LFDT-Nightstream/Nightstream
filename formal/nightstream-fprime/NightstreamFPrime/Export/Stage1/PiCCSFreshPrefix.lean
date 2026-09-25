import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PrefixFold
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
import NightstreamFPrime.Export.Stage1.PiCCSSourceImages

/-!
Fold retained matrix-image rows with the existing prefix interpolation.
Rows remain Vector K and the retained prefix remains Array. Missing rows
are zero. Each port has exactly the original scalar PrefixFold meaning.
The active-row constructor retains evaluator failures. IO, cached scans and
transcript steps stay with their existing owners.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiCCSFreshPrefix

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Layout

/-- Read each active row once and retain all existing matrix ports. -/
def rows? (program : MatrixProgram.Program) (sourceRow : Nat → Option R1CS.Row)
    (assignment : Phi81Relation.Assignment PiCCSSourceImages.shape)
    (fits : program.rowCount ≤ 2 ^ cubeVariables) :
    Option (Array (Vector K Spec.ProductionRelation.matrixCount)) :=
  Array.ofFnM fun row : Fin program.rowCount =>
    (PiCCSSourceImages.freshMatrixImage? program sourceRow assignment
      (NumericBooleanDomain.vertex cubeVariables ⟨row.val, Nat.lt_of_lt_of_le row.isLt fits⟩)).map
        (fun values => values.map K.embed)

/-- The streamed producer can fold one pair directly, without temporary arrays. -/
def pairRow {matrixCount : Nat} (low high : Vector K matrixCount)
    (challenge : K) : Vector K matrixCount :=
  Vector.ofFn fun port =>
    PrefixFold.interpolate extensionOps challenge (low.get port) (high.get port)

/-- Each emitted port uses the original scalar interpolation. -/
theorem pairRow_get {matrixCount : Nat} (low high : Vector K matrixCount)
    (challenge : K) (port : Fin matrixCount) :
    (pairRow low high challenge).get port =
      PrefixFold.interpolate extensionOps challenge (low.get port) (high.get port) := by
  change (Vector.ofFn _)[port.val] = _
  rw [Vector.getElem_ofFn]

/-- Adjacent row pairs consume the next low bit. An odd high endpoint is the
zero row; an empty prefix stays empty and a singleton still consumes a challenge. -/
def foldRows {matrixCount : Nat} (rows : Array (Vector K matrixCount))
    (challenge : K) : Array (Vector K matrixCount) :=
  Array.ofFn fun pair : Fin ((rows.size + 1) / 2) =>
    pairRow
      (rows.getD (2 * pair.val) (Vector.replicate matrixCount extensionOps.zero))
      (rows.getD (2 * pair.val + 1) (Vector.replicate matrixCount extensionOps.zero))
      challenge

/-- One existing scalar array view, used only for the per-port statement. -/
def portValues {matrixCount : Nat} (rows : Array (Vector K matrixCount))
    (port : Fin matrixCount) : Array K :=
  rows.map fun row => row.get port

/-- Store exactly one row for each adjacent pair, including an odd final row. -/
theorem foldRows_size {matrixCount : Nat} (rows : Array (Vector K matrixCount))
    (challenge : K) : (foldRows rows challenge).size = (rows.size + 1) / 2 := by
  simp only [foldRows, Array.size_ofFn]

private theorem zeroRow_get {matrixCount : Nat} (port : Fin matrixCount) :
    (Vector.replicate matrixCount extensionOps.zero).get port = extensionOps.zero := by
  change (Vector.replicate matrixCount extensionOps.zero)[port.val] = _
  rw [Vector.getElem_replicate]

theorem portValues_getD {matrixCount : Nat}
    (rows : Array (Vector K matrixCount)) (port : Fin matrixCount) (index : Nat) :
    (portValues rows port).getD index extensionOps.zero =
      (rows.getD index (Vector.replicate matrixCount extensionOps.zero)).get port := by
  simp only [portValues, Array.getD_eq_getD_getElem?, Array.getElem?_map]
  cases rows[index]? with
  | none =>
      simp only [Option.map_none, Option.getD_none, zeroRow_get]
  | some row =>
      simp only [Option.map_some, Option.getD_some]

/-- Exact complete-array equality for every port, including an odd tail.
No row-validity, nonzero-value or stored-prefix size premise is required. -/
theorem portValues_foldRows {matrixCount : Nat}
    (rows : Array (Vector K matrixCount)) (challenge : K) (port : Fin matrixCount) :
    portValues (foldRows rows challenge) port =
      PrefixFold.foldOne extensionOps (portValues rows port) challenge := by
  unfold portValues foldRows PrefixFold.foldOne
  rw [Array.map_ofFn]
  apply Array.ext
  · simp only [Array.size_ofFn, Array.size_map]
  · intro index leftBound rightBound
    simp only [Array.getElem_ofFn, Function.comp_apply]
    rw [pairRow_get]
    rw [← portValues_getD rows port (2 * index),
      ← portValues_getD rows port (2 * index + 1)]
    rfl

/-- Per-port MLE meaning follows directly from the existing scalar fold.
Only the table arity decreases; no protocol Shape value is constructed or changed. -/
theorem foldRows_evaluate {matrixCount remaining : Nat}
    (rows : Array (Vector K matrixCount)) (challenge : K) (port : Fin matrixCount)
    (suffix : CubePoint K remaining) :
    (PrefixFold.zeroExtend extensionOps remaining
      (portValues (foldRows rows challenge) port)).evaluate extensionOps suffix =
      (PrefixFold.zeroExtend extensionOps (remaining + 1) (portValues rows port)).evaluate
        extensionOps ⟨challenge :: suffix.coordinates, by simp [suffix.dimension]⟩ := by
  rw [portValues_foldRows]
  exact PrefixFold.foldOne_evaluate extensionOps extensionLaws (portValues rows port) challenge suffix

end NightstreamFPrime.Export.Stage1.PiCCSFreshPrefix
