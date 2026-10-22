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

def freshPrefix {logicalWidth : Nat}
    (assignment : Fin logicalWidth → F) : Array K :=
  Array.ofFn fun column : Fin (Phi81CarrierLayout.carrierWidth logicalWidth) =>
    K.embed (Phi81CarrierLayout.extendAssignment 0 assignment column)

def matrixPrefix {logicalWidth : Nat} (plan : Plan logicalWidth)
    (assignment : Fin logicalWidth → F)
    (matrix : Fin Spec.ProductionRelation.matrixCount) : Array K :=
  match meaningfulPort? matrix with
  | none => #[]
  | some _ => Array.ofFn fun row : Fin plan.rowCount =>
      K.embed ((plan.portForm row matrix).evalSparse assignment)

theorem freshPrefix_size {logicalWidth : Nat}
    (assignment : Fin logicalWidth → F) :
    (freshPrefix assignment).size = Phi81CarrierLayout.carrierWidth logicalWidth := by
  simp only [freshPrefix, Array.size_ofFn]

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
  · simp only [Array.getD_eq_getD_getElem?, Array.getElem?_ofFn, 
      Option.getD_some, ColumnLayout.paddedValue, CanonicalRowLayout.layout,
      dif_pos live]
  · simp only [Array.getD_eq_getD_getElem?, Array.getElem?_ofFn, 
      Option.getD_none, ColumnLayout.paddedValue, CanonicalRowLayout.layout,
      dif_neg live]
    exact embed_literal_zero.symm

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

noncomputable def source (plan : Plan logicalWidth)
    (cubeFits : Phi81CarrierLayout.carrierWidth logicalWidth ≤ 2 ^ cubeVariables)
    (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
    (assignment : Fin logicalWidth → F) :
    ConnectedInputs K productionShape (Phi81CarrierLayout.carrierWidth logicalWidth)
      (Phi81ColumnLayout.blockCount (Phi81CarrierLayout.carrierWidth logicalWidth)) :=
  ((ProductionKey.key (plan.logicalRelation cubeFits) ajtai).statement
    defaultRunning fresh).sourceConnectedInputs (sourceWitness assignment)

noncomputable def protocol (plan : Plan logicalWidth)
    (cubeFits : Phi81CarrierLayout.carrierWidth logicalWidth ≤ 2 ^ cubeVariables)
    (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
    (assignment : Fin logicalWidth → F) : ProtocolPolynomial.Data K productionShape :=
  ((ProductionKey.key (plan.logicalRelation cubeFits) ajtai).statement
    defaultRunning fresh).sourceProtocolData K.embed (sourceWitness assignment)

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
