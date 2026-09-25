import NightstreamFPrime.Layout.ProductionRelation.CcsOpening
import NightstreamFPrime.Layout.ProductionRelation.SparseEvaluation
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PrefixFold

/-!
Owns the fresh scalar prefixes read from one production plan and assignment.
The matrix prefixes execute stored sparse entries; the assignment prefix
includes exact carrier completion. Both are proved equal to the tables of
the actual production-key statement with literal zero running witnesses.
This module does not compute SumCheck round sums or full ring outputs.
-/

namespace NightstreamFPrime.Layout.ProductionRelation.ZeroRunningOracle

open NightstreamFPrime.Spec
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec.Folding.PiCCS
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
open MatrixCoefficientSource UnifiedSources

private abbrev freshIndex : Fin productionShape.freshCount := ⟨0, by decide⟩

/-- The complete fresh carrier prefix comes only from zero completion of the
logical assignment. No Boolean-cube suffix is allocated. -/
def freshPrefix {logicalWidth : Nat}
    (assignment : Fin logicalWidth → F) : Array K :=
  Array.ofFn fun column : Fin (Phi81CarrierLayout.carrierWidth logicalWidth) =>
    K.embed (Phi81CarrierLayout.extendAssignment 0 assignment column)

/-- Each meaningful matrix prefix contains only the live sparse-row images.
Slot 13 is represented by the empty prefix and its implicit zero extension. -/
def matrixPrefix {logicalWidth : Nat} (plan : Plan logicalWidth)
    (assignment : Fin logicalWidth → F)
    (matrix : Fin Spec.ProductionRelation.matrixCount) : Array K :=
  match meaningfulPort? matrix with
  | none => #[]
  | some _ => Array.ofFn fun row : Fin plan.rowCount =>
      K.embed ((plan.portForm row matrix).evalSparse assignment)

/-- The fresh prefix includes every coordinate of the final carrier block. -/
theorem freshPrefix_size {logicalWidth : Nat}
    (assignment : Fin logicalWidth → F) :
    (freshPrefix assignment).size = Phi81CarrierLayout.carrierWidth logicalWidth := by
  simp only [freshPrefix, Array.size_ofFn]

/-- Live matrix rows fit the selected cube; the zero matrix has no entries. -/
theorem matrixPrefix_fits {logicalWidth : Nat} (plan : Plan logicalWidth)
    (assignment : Fin logicalWidth → F)
    (matrix : Fin Spec.ProductionRelation.matrixCount) :
    (matrixPrefix plan assignment matrix).size ≤ 2 ^ cubeVariables := by
  cases selected : meaningfulPort? matrix with
  | none => simp [matrixPrefix, selected]
  | some meaningful =>
      simpa only [matrixPrefix, selected, Array.size_ofFn] using plan.rowCount_le

private theorem embed_literal_zero : K.embed (0 : F) = K.zero :=
  ConcreteCarrier.embed_zero

private theorem embeddedPrefix_getD {arity count : Nat}
    (covered : count ≤ 2 ^ arity) (values : Fin count → F)
    (vertex : BooleanVertex arity) :
    (Array.ofFn fun index => K.embed (values index)).getD
        (NumericBooleanDomain.index vertex) K.zero =
      K.embed ((CanonicalRowLayout.layout arity count covered).paddedValue
        0 values vertex) := by
  by_cases live : NumericBooleanDomain.index vertex < count
  · simp only [Array.getD_eq_getD_getElem?, Array.getElem?_ofFn, dif_pos live,
      Option.getD_some, ColumnLayout.paddedValue, CanonicalRowLayout.layout,
      dif_pos live]
  · simp only [Array.getD_eq_getD_getElem?, Array.getElem?_ofFn, dif_neg live,
      Option.getD_none, ColumnLayout.paddedValue, CanonicalRowLayout.layout,
      dif_neg live]
    exact embed_literal_zero.symm

/-- Every numeric matrix-prefix read equals the authoritative sparse image,
including the zero row suffix and the separate zero matrix slot. -/
theorem matrixPrefix_getD {logicalWidth : Nat} (plan : Plan logicalWidth)
    (assignment : Fin logicalWidth → F)
    (matrix : Fin Spec.ProductionRelation.matrixCount)
    (vertex : BooleanVertex cubeVariables) :
    (matrixPrefix plan assignment matrix).getD
        (NumericBooleanDomain.index vertex) K.zero =
      K.embed (plan.rowImage assignment vertex matrix) := by
  cases selected : meaningfulPort? matrix with
  | none =>
      cases decoded : plan.rowLayout.toColumn? vertex with
      | none =>
          simpa [matrixPrefix, selected, Plan.rowImage, decoded] using
            embed_literal_zero.symm
      | some row =>
          simpa [matrixPrefix, selected, Plan.rowImage, decoded, Plan.portForm,
            SparseForm.empty_eval] using embed_literal_zero.symm
  | some meaningful =>
      rw [matrixPrefix, selected,
        embeddedPrefix_getD plan.rowCount_le
          (fun row => (plan.portForm row matrix).evalSparse assignment) vertex]
      change K.embed (plan.rowLayout.paddedValue 0
          (fun row => (plan.portForm row matrix).evalSparse assignment) vertex) = _
      cases decoded : plan.rowLayout.toColumn? vertex with
      | none => simp only [ColumnLayout.paddedValue, Plan.rowImage, decoded]
      | some row =>
          simp only [ColumnLayout.paddedValue, Plan.rowImage, decoded,
            SparseForm.evalSparse_eq_eval]

/-- The one fresh witness is completed from the same logical assignment.
Every running source is the literal zero assignment. -/
def sourceWitness {logicalWidth : Nat} (assignment : Fin logicalWidth → F) :
    StrongReduction.OutputWitness productionShape
      (Phi81CarrierLayout.carrierWidth logicalWidth) where
  assignments := fun source column =>
    if source.val = 0 then
      Phi81CarrierLayout.extendAssignment 0 assignment column
    else 0

private theorem sourceWitness_running {logicalWidth : Nat}
    (assignment : Fin logicalWidth → F)
    (index : Fin productionShape.runningCount)
    (column : Fin (Phi81CarrierLayout.carrierWidth logicalWidth)) :
    (sourceWitness assignment).assignments (runningSourceIndex index) column = 0 := by
  change (if 1 + index.val = 0 then _ else (0 : F)) = 0
  rw [if_neg (by omega)]

section

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}

/-- Use the existing key statement and its sole witness-attachment function.
There is no separately supplied matrix source, layout, or table family. -/
noncomputable def source (plan : Plan logicalWidth)
    (cubeFits : Phi81CarrierLayout.carrierWidth logicalWidth ≤ 2 ^ cubeVariables)
    (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
    (assignment : Fin logicalWidth → F) :
    ConnectedInputs K productionShape (Phi81CarrierLayout.carrierWidth logicalWidth)
      (Phi81ColumnLayout.blockCount (Phi81CarrierLayout.carrierWidth logicalWidth)) :=
  ((ProductionKey.key (plan.logicalRelation cubeFits) ajtai).statement
    defaultRunning fresh).sourceConnectedInputs (sourceWitness assignment)

/-- The existing statement-derived protocol data for this exact witness. -/
noncomputable def protocol (plan : Plan logicalWidth)
    (cubeFits : Phi81CarrierLayout.carrierWidth logicalWidth ≤ 2 ^ cubeVariables)
    (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
    (assignment : Fin logicalWidth → F) : ProtocolPolynomial.Data K productionShape :=
  ((ProductionKey.key (plan.logicalRelation cubeFits) ajtai).statement
    defaultRunning fresh).sourceProtocolData K.embed (sourceWitness assignment)

/-- The source used by the production statement has literal zero running
assignments, not a carried equality assumption. -/
theorem source_running_zero (plan : Plan logicalWidth)
    (cubeFits : Phi81CarrierLayout.carrierWidth logicalWidth ≤ 2 ^ cubeVariables)
    (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
    (assignment : Fin logicalWidth → F)
    (index : Fin productionShape.runningCount)
    (column : Fin (Phi81CarrierLayout.carrierWidth logicalWidth)) :
    (source plan cubeFits ajtai fresh assignment).assignments
      (runningSourceIndex index) column = 0 := by
  exact sourceWitness_running assignment index column

/-- Zero completion preserves the plan's literal zero matrix-13 slot. -/
theorem source_matrix13_zero (plan : Plan logicalWidth)
    (cubeFits : Phi81CarrierLayout.carrierWidth logicalWidth ≤ 2 ^ cubeVariables)
    (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
    (assignment : Fin logicalWidth → F)
    (vertex : BooleanVertex cubeVariables)
    (column : Fin (Phi81CarrierLayout.carrierWidth logicalWidth)) :
    (source plan cubeFits ajtai fresh assignment).matrixSource.matrices
      Spec.ProductionRelation.zeroPort vertex column = 0 := by
  change Phi81CarrierLayout.extendMatrix 0
    (plan.matrix Spec.ProductionRelation.zeroPort) vertex column = 0
  rw [plan.zeroPort_matrix]
  unfold Phi81CarrierLayout.extendMatrix
  cases Phi81CarrierLayout.logicalColumn? column <;> rfl

/-- The completed assignment array is exactly the fresh source table in the
actual production-key protocol data at every Boolean vertex. -/
theorem freshPrefix_table (plan : Plan logicalWidth)
    (cubeFits : Phi81CarrierLayout.carrierWidth logicalWidth ≤ 2 ^ cubeVariables)
    (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
    (assignment : Fin logicalWidth → F) :
    PrefixFold.zeroExtend extensionOps cubeVariables (freshPrefix assignment) =
      (protocol plan cubeFits ajtai fresh assignment).sourceAssignments
        (freshSourceIndex freshIndex) := by
  change BooleanTable.tabulate (fun vertex =>
      (freshPrefix assignment).getD (NumericBooleanDomain.index vertex) K.zero) =
    BooleanTable.tabulate (fun vertex =>
      K.embed ((CanonicalRowLayout.layout cubeVariables
        (Phi81CarrierLayout.carrierWidth logicalWidth) cubeFits).paddedValue
          0 (Phi81CarrierLayout.extendAssignment 0 assignment) vertex))
  apply congrArg BooleanTable.tabulate
  funext vertex
  exact embeddedPrefix_getD cubeFits
    (Phi81CarrierLayout.extendAssignment 0 assignment) vertex

/-- Sparse entry execution, original scalar matrices, and exact carrier
completion produce the same fresh matrix-image table. The equality includes
all 14 ports; the final port has an empty numeric prefix. -/
theorem matrixPrefix_table (plan : Plan logicalWidth)
    (cubeFits : Phi81CarrierLayout.carrierWidth logicalWidth ≤ 2 ^ cubeVariables)
    (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
    (assignment : Fin logicalWidth → F)
    (matrix : Fin Spec.ProductionRelation.matrixCount) :
    PrefixFold.zeroExtend extensionOps cubeVariables (matrixPrefix plan assignment matrix) =
      (protocol plan cubeFits ajtai fresh assignment).freshMatrixImages freshIndex matrix := by
  change BooleanTable.tabulate (fun vertex =>
      (matrixPrefix plan assignment matrix).getD (NumericBooleanDomain.index vertex) K.zero) =
    BooleanTable.tabulate (fun vertex => K.embed
      (PaperLinearAlgebra.matrixVectorAt baseOps
        (Phi81CarrierLayout.extendMatrix 0 (plan.matrix matrix))
        (Phi81CarrierLayout.extendAssignment 0 assignment) vertex))
  apply congrArg BooleanTable.tabulate
  funext vertex
  rw [matrixPrefix_getD]
  exact congrArg K.embed
    ((Phi81CarrierLayout.matrixVectorAt_extend baseOps baseLaws
      (plan.matrix matrix) assignment vertex).trans
      (Plan.matrixVectorAt_matrix plan assignment vertex matrix)).symm

end

end NightstreamFPrime.Layout.ProductionRelation.ZeroRunningOracle
