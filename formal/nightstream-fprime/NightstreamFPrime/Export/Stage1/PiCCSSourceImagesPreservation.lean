import NightstreamFPrime.Export.Stage1.PiCCSSourceImages
import NightstreamFPrime.Export.Stage1.PiCCSPublicReplay
import NightstreamFPrime.Export.Stage1.PiDECEvaluationSelectedPrefix
import NightstreamFPrime.Export.Stage1.PiDECCanonicalSourceCache

/-! Proof-only transport from the original scalar and coefficient reads to
all four sourceProtocolData endpoint families. Do not import this module
into an executable: its old evaluation-row proofs retain reference plans. -/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiCCSSourceImages

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.MatrixCoefficientSource
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.UnifiedSources
open NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Export.Stage1.PiRLCPartialTrace

private abbrev selectedPlan := PerApplicationFixedPoint.structuralPlan
  Poseidon2HashChainV1Package.application Poseidon2HashChainV1Package.fits
private abbrev selectedRelation := PerApplicationFixedPoint.relation
  Poseidon2HashChainV1Package.application Poseidon2HashChainV1Package.fits
private abbrev selectedProgram := PerApplicationMatrixProgram.matrixProgram
  Poseidon2HashChainV1Package.application
private abbrev selectedSource := PerApplicationCanonicalPackage.sourceRow
  Poseidon2HashChainV1Package.application Poseidon2HashChainV1Package.fits
private abbrev selectedLayout :=
  (Lifecycle.PiRLC.v1_1.InputBinding.relationSource selectedRelation).cubeLayout
private noncomputable abbrev statementFor
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (input : PiCCSPublicReplay.Input) :=
  (ProductionKey.key relation Poseidon2HashChainV1Setup.productionAjtaiKey).statement
    (PiCCSPublicReplay.running input) (PiCCSPublicReplay.fresh input)
private noncomputable abbrev selectedStatement (input : PiCCSPublicReplay.Input) :=
  statementFor selectedRelation input

/-- Instantiation of the old arbitrary-assignment row theorem. This reference
conversion is used only in proofs; no executable definition calls it. -/
private def stored (assignment : Phi81Relation.Assignment shape) :
    Vector (Vector F shape.carrierWidth) productionGlobalParams.k :=
  Vector.replicate productionGlobalParams.k (Vector.ofFn assignment)

private theorem get_replicate {Alpha : Type} {count : Nat}
    (value : Alpha) (index : Fin count) :
    (Vector.replicate count value).get index = value := by
  change (Vector.replicate count value)[index.val] = value
  rw [Vector.getElem_replicate]

private theorem stored_get (assignment : Phi81Relation.Assignment shape)
    (child : Fin productionGlobalParams.k) :
    ((stored assignment).get child).get = assignment := by
  funext column
  change ((Vector.replicate productionGlobalParams.k (Vector.ofFn assignment))[child.val])[column.val] = _
  rw [Vector.getElem_replicate, Vector.getElem_ofFn]

private theorem eval_kernelRead_eq_row {columns : Nat} (form : SparseForm columns)
    (assignment : Phi81Relation.Assignment shape) (output : Fin ringDegree) :
    form.evalSparse (kernelRead assignment output) =
      (PiDECEvaluationRows.row (shape := shape) form (stored assignment) ⟨0, by decide⟩).get output := by
  rw [PiDECEvaluationRows.row, PiDECEvaluationBlockSupport.kernel_eq_evalSparse]
  apply congrArg (fun read : Fin columns → F => form.evalSparse read)
  funext column
  by_cases live : column.val / ringDegree < blockCount
  · simp only [kernelRead, blockAt, dif_pos live,
      PiDECCommitmentFold.childBlocks_value, stored_get]
    rfl
  · simp only [kernelRead, blockAt, dif_neg live, get_replicate,
      PiDECCommitmentFold.zero_value]

/-- The prepared sparse basis reader works on the original block itself,
without any signed-digit or decomposition premise. -/
theorem coefficientRead_value {columns : Nat}
    (assignment : Phi81Relation.Assignment shape) (output : Fin ringDegree)
    (column : Fin columns) :
    (((PiDECParentSparseRead.prepare ()).get
        ⟨column.val % ringDegree, Nat.mod_lt _ (by decide)⟩).get output).evalSparse
        (blockAt assignment (column.val / ringDegree)) =
      kernelRead assignment output column := by
  rw [PiDECParentSparseRead.prepare, FixedArray.get_ofFn, FixedArray.get_ofFn,
    PiDECParentScalarRead.prepare, FixedArray.get_ofFn,
    PiDECParentSparseRead.coefficientForm_value, MaterializedRingF.toRing_ofRing]
  exact (congrFun (CarrierAction.kernelImage_eq_ringFMul
    ⟨column.val % ringDegree, Nat.mod_lt _ (by decide)⟩
    (blockAt assignment (column.val / ringDegree))) output).symm

/-- Preparing the actual basis tables gives the same complete original
assignment read, for every field coefficient and every requested column. -/
theorem preparedRead_eq {columns : Nat} (assignment : Phi81Relation.Assignment shape)
    (output : Fin ringDegree) :
    (preparedRead (PiDECParentSparseRead.prepare ()) assignment output : Fin columns → F) =
      kernelRead assignment output := by
  funext column
  exact coefficientRead_value assignment output column

private theorem numericRow_of_plan {columns : Nat}
    (plan : Plan columns) (program : MatrixProgram.Program)
    (sourceRow : Nat → Option R1CS.Row) (read : Fin columns → F)
    (row : Fin plan.rowCount) (matrix : Fin Spec.ProductionRelation.matrixCount)
    (loaded : program.row? columns sourceRow row.val = some (plan.forms row)) :
    (PiDECMatrixNumericRows.row? program sourceRow read row.val).map
        (fun values => values.get matrix) =
      some ((plan.portForm row matrix).evalSparse read) := by
  rw [PiDECMatrixNumericRows.row?_value, loaded]
  rfl

private theorem canonical_numericRow
    (application : Lifecycle.Stage1.Application.Program)
    (fits : PerApplicationFixedPoint.FitsTwoPow28 application)
    (read : Fin (PerApplicationFixedPoint.logicalWidth application) → F)
    (row : Fin (PerApplicationFixedPoint.structuralPlan application fits).rowCount)
    (matrix : Fin Spec.ProductionRelation.matrixCount) :
    (PiDECMatrixNumericRows.row?
        (PerApplicationMatrixProgram.matrixProgram application)
        (PerApplicationCanonicalPackage.sourceRow application fits) read row.val).map
        (fun values => values.get matrix) =
      some (((PerApplicationFixedPoint.structuralPlan application fits).portForm row matrix).evalSparse read) := by
  exact numericRow_of_plan
    (PerApplicationFixedPoint.structuralPlan application fits)
    (PerApplicationMatrixProgram.matrixProgram application)
    (PerApplicationCanonicalPackage.sourceRow application fits) read row matrix
    (PerApplicationCanonicalPackage.matrixProgram_row? application fits row)

/-- All selected rows and all fourteen ports, including the complete zero
suffix. Matrix/source correspondence is discharged by the selected theorem. -/
theorem matrixImage_value (assignment : Phi81Relation.Assignment shape)
    (output : Fin ringDegree) (vertex : BooleanVertex Lifecycle.cubeVariables)
    (matrix : Fin Spec.ProductionRelation.matrixCount) :
    (matrixImage? selectedProgram selectedSource assignment output vertex).map
        (fun values => values.get matrix) =
      some (PiRLC.rowRing selectedRelation.system assignment matrix vertex output) := by
  have reference := congrFun (PiDECEvaluationSelectedPrefix.matrixRow_value
    (stored assignment) ⟨0, by decide⟩ matrix vertex) output
  rw [stored_get] at reference
  have counts : selectedProgram.rowCount = selectedPlan.rowCount :=
    PerApplicationMatrixProgram.matrixProgram_rowCount_eq_structuralPlan
      Poseidon2HashChainV1Package.application Poseidon2HashChainV1Package.fits
  by_cases live : NumericBooleanDomain.index vertex < selectedPlan.rowCount
  · have programLive : NumericBooleanDomain.index vertex < selectedProgram.rowCount :=
      counts.symm ▸ live
    simp only [matrixImage?, rowValues?, if_pos programLive]
    have loaded := canonical_numericRow Poseidon2HashChainV1Package.application
      Poseidon2HashChainV1Package.fits (kernelRead assignment output)
      ⟨NumericBooleanDomain.index vertex, live⟩ matrix
    have evaluated :
        (selectedPlan.portForm ⟨NumericBooleanDomain.index vertex, live⟩ matrix).evalSparse
            (kernelRead assignment output) =
          PiRLC.rowRing selectedRelation.system assignment matrix vertex output := by
      exact (eval_kernelRead_eq_row
        (selectedPlan.portForm ⟨NumericBooleanDomain.index vertex, live⟩ matrix)
        assignment output).trans (by
          simpa only [PiDECEvaluationSelectedPrefix.matrixRow, dif_pos live] using reference)
    exact loaded.trans (congrArg some evaluated)
  · have programOutside : ¬ NumericBooleanDomain.index vertex < selectedProgram.rowCount :=
      counts.symm ▸ live
    simp only [matrixImage?, rowValues?, if_neg programOutside, Option.map_some, get_replicate]
    rw [PiDECEvaluationSelectedPrefix.matrixRow, dif_neg live,
      PiDECCommitmentFold.zero_value] at reference
    exact congrArg some reference

/-- The existing prepared source HashMap has the same selected authority. -/
theorem cachedMatrixImage_value (assignment : Phi81Relation.Assignment shape)
    (output : Fin ringDegree) (vertex : BooleanVertex Lifecycle.cubeVariables)
    (matrix : Fin Spec.ProductionRelation.matrixCount) :
    (matrixImage? selectedProgram
        (fun index => (PiDECCanonicalSourceCache.stored
          Poseidon2HashChainV1Package.application)[index]?)
        assignment output vertex).map (fun values => values.get matrix) =
      some (PiRLC.rowRing selectedRelation.system assignment matrix vertex output) := by
  have sourceEqual :
      (fun index => (PiDECCanonicalSourceCache.stored
        Poseidon2HashChainV1Package.application)[index]?) = selectedSource := by
    funext index
    exact PiDECCanonicalSourceCache.stored_value Poseidon2HashChainV1Package.application
      Poseidon2HashChainV1Package.fits index
  rw [sourceEqual]
  exact matrixImage_value assignment output vertex matrix

private theorem padImage_eq_form
    (layout : ColumnLayout Lifecycle.cubeVariables shape.carrierWidth)
    (assignment : Phi81Relation.Assignment shape) (output : Fin ringDegree)
    (vertex : BooleanVertex Lifecycle.cubeVariables) :
    padImage layout assignment output vertex =
      (PiDECEvaluationPadBlock.form layout vertex).evalSparse (kernelRead assignment output) := by
  cases decoded : layout.toColumn? vertex <;>
    simp only [padImage, PiDECEvaluationPadBlock.form, decoded]

private theorem padMatrix_eq {width : Nat}
    {fits : ringDegree * PaperAlgebra.publicRingColumns ≤ Phi81CarrierLayout.carrierWidth width}
    (relation : ProductionKey.LogicalRelation width fits) :
    PaperAlgebra.padMatrix (Lifecycle.PiRLC.v1_1.InputBinding.relationSource relation) =
      (Lifecycle.PiRLC.v1_1.InputBinding.relationSource relation).cubeLayout.paddedIdentityEntry
        (0 : F) 1 := rfl

theorem padImage_value (assignment : Phi81Relation.Assignment shape)
    (output : Fin ringDegree) (vertex : BooleanVertex Lifecycle.cubeVariables) :
    padImage selectedLayout assignment output vertex =
      PiRLC.ExplicitMatrix.rowRing selectedRelation.system
        (PaperAlgebra.padMatrix
          (Lifecycle.PiRLC.v1_1.InputBinding.relationSource selectedRelation))
        assignment vertex output := by
  rw [padImage_eq_form, eval_kernelRead_eq_row, padMatrix_eq]
  have reference := congrFun (PiDECEvaluationRows.padRow_value
    selectedRelation.system selectedLayout vertex (stored assignment) ⟨0, by decide⟩) output
  rw [stored_get] at reference
  exact reference

private theorem kernelImage_constant_output (basis : Fin ringDegree) (source : RingF) :
    CarrierAction.kernelImage basis source Phi81CoefficientKernel.constant = source basis := by
  have constant_eq : Phi81CoefficientKernel.phi81Kernel.constant =
      Phi81CoefficientKernel.constant := rfl
  have constant_weight (row column : Fin ringDegree) :
      Phi81CoefficientKernel.phi81Kernel.weight Phi81CoefficientKernel.constant row column =
        if row = column then (1 : F) else 0 := by
    have identity := Phi81CoefficientKernel.phi81ConstantTermLaw.weight row column
    simpa only [constant_eq] using! identity
  rw [PiRLC.kernelImage_apply]
  let term := fun index => if live : index < ringDegree then source ⟨index, live⟩ else 0
  calc
    _ = sumRange baseOps ringDegree (fun index => if index = basis.val then term index else 0) := by
      apply sumRange_congr baseOps ringDegree
      intro index live
      rw [dif_pos live, constant_weight]
      by_cases equal : index = basis.val
      · subst index
        simp [term]
      · have different : basis ≠ (⟨index, live⟩ : Fin ringDegree) := by
          intro found
          exact equal (congrArg Fin.val found).symm
        simp only [if_neg different, if_neg equal, Fin.mul_zero]
    _ = term basis.val := sumRange_select baseOps baseLaws ringDegree basis.val term basis.isLt
    _ = source basis := by simp only [term, dif_pos basis.isLt]

private theorem logicalBlock_bound {width : Nat} (column : Fin width) :
    column.val / ringDegree <
      Phi81ColumnLayout.blockCount (Phi81CarrierLayout.carrierWidth width) :=
  (Phi81ColumnLayout.decode (Phi81CarrierLayout.embedLogical column)).1.isLt

private theorem assignmentBlock_logical {width : Nat}
    (assignment : CarrierAction.CompleteAssignment width) (column : Fin width)
    (live : column.val / ringDegree <
      Phi81ColumnLayout.blockCount (Phi81CarrierLayout.carrierWidth width)) :
    CarrierAction.assignmentBlock assignment ⟨column.val / ringDegree, live⟩
        ⟨column.val % ringDegree, Nat.mod_lt _ (by decide)⟩ =
      assignment (Phi81CarrierLayout.embedLogical column) := by
  change assignment (CarrierAction.carrierColumn
      ⟨column.val / ringDegree, live⟩
      ⟨column.val % ringDegree, Nat.mod_lt _ (by decide)⟩) = _
  apply congrArg assignment
  apply Fin.ext
  change column.val / ringDegree * ringDegree + column.val % ringDegree = column.val
  exact Nat.div_add_mod' column.val ringDegree

/-- Output coefficient zero is the original scalar read, not basis zero. -/
theorem kernelRead_constant (assignment : Phi81Relation.Assignment shape)
    (column : Fin logicalWidth) :
    kernelRead assignment Phi81CoefficientKernel.constant column = plainRead assignment column := by
  have live : column.val / ringDegree < blockCount := logicalBlock_bound column
  rw [kernelRead, blockAt, dif_pos live, kernelImage_constant_output]
  exact assignmentBlock_logical assignment column live

private theorem freshMatrixImage_eq (program : MatrixProgram.Program)
    (sourceRow : Nat → Option R1CS.Row) (assignment : Phi81Relation.Assignment shape)
    (vertex : BooleanVertex Lifecycle.cubeVariables) :
    freshMatrixImage? program sourceRow assignment vertex =
      matrixImage? program sourceRow assignment Phi81CoefficientKernel.constant vertex := by
  have reads : kernelRead assignment Phi81CoefficientKernel.constant = plainRead assignment :=
    funext (kernelRead_constant assignment)
  simp only [freshMatrixImage?, matrixImage?, reads]

private theorem matrix_constant (assignment : Phi81Relation.Assignment shape)
    (matrix : Fin Spec.ProductionRelation.matrixCount)
    (vertex : BooleanVertex Lifecycle.cubeVariables) :
    PiRLC.rowRing selectedRelation.system assignment matrix vertex Phi81CoefficientKernel.constant =
      PaperLinearAlgebra.matrixVectorAt baseOps
        (selectedRelation.system.matrixSource.matrices matrix) assignment vertex := by
  have row_constant {carrier : Phi81Relation.Shape}
      (system : Phi81Relation.Structure carrier) (value : Phi81Relation.Assignment carrier)
      (index : Fin carrier.matrixCount) (atVertex : BooleanVertex carrier.rowVariables) :
      PiRLC.rowRing system value index atVertex Phi81CoefficientKernel.constant =
        PaperLinearAlgebra.matrixVectorAt baseOps
          (system.matrixSource.matrices index) value atVertex := by
    have kernel_eq := Phi81Relation.Structure.matrixSource_kernel_eq system
    have constant_eq : system.matrixSource.kernel.constant =
        Phi81CoefficientKernel.constant := by
      exact (congrArg (fun kernel : CoefficientKernel F ringDegree => kernel.constant)
        kernel_eq).trans (show Phi81CoefficientKernel.phi81Kernel.constant =
          Phi81CoefficientKernel.constant from rfl)
    have constant_law : ConstantTermLaw baseOps system.matrixSource.kernel := by
      rw [kernel_eq]
      exact Phi81CoefficientKernel.phi81ConstantTermLaw
    have matrix_eq := MatrixSource.coefficientMatrix_constant_eq baseOps baseLaws
      system.matrixSource constant_law index
    rw [constant_eq] at matrix_eq
    exact congrArg (fun image =>
      PaperLinearAlgebra.matrixVectorAt baseOps image value atVertex) matrix_eq
  exact row_constant selectedRelation.system assignment matrix vertex

/-- Original scalar CCS images for the fresh source. -/
theorem freshMatrixImage_value (assignment : Phi81Relation.Assignment shape)
    (vertex : BooleanVertex Lifecycle.cubeVariables)
    (matrix : Fin Spec.ProductionRelation.matrixCount) :
    (freshMatrixImage? selectedProgram selectedSource assignment vertex).map
        (fun values => values.get matrix) =
      some (PaperLinearAlgebra.matrixVectorAt baseOps
        (selectedRelation.system.matrixSource.matrices matrix) assignment vertex) := by
  rw [freshMatrixImage_eq, matrixImage_value, matrix_constant]

private theorem statement_matrix_endpoint
    {Commitment PublicInput : Type} {sourceShape : Shape} {columns blocks : Nat}
    (statement : StrongReduction.Statement K Commitment PublicInput sourceShape columns blocks baseOps)
    (witness : StrongReduction.OutputWitness sourceShape columns)
    (coordinate : MatrixCoordinate sourceShape)
    (vertex : BooleanVertex sourceShape.cubeVariables) :
    ((statement.sourceProtocolData K.embed witness).matrixImages coordinate).valueAt vertex =
      K.embed (PaperLinearAlgebra.matrixVectorAt baseOps
        (statement.matrixSource.coefficientMatrix baseOps coordinate.matrix coordinate.coefficient)
        (witness.assignments (runningSourceIndex coordinate.running)) vertex) := by
  simp only [StrongReduction.Statement.sourceProtocolData,
    StrongReduction.Statement.sourceConnectedInputs, ConnectedInputs.toUnifiedInputs,
    ProtocolDataRefinement.toProtocolData, UnifiedInputs.matrixData,
    MatrixSource.coefficientMatrices, MatrixEvaluationResidual.imageTable,
    MatrixEvaluationResidual.imageCoefficientAt, BooleanTable.valueAt_tabulate]

private theorem statement_pad_endpoint
    {Commitment PublicInput : Type} {sourceShape : Shape} {columns blocks : Nat}
    (statement : StrongReduction.Statement K Commitment PublicInput sourceShape columns blocks baseOps)
    (witness : StrongReduction.OutputWitness sourceShape columns)
    (coordinate : PadCoordinate sourceShape)
    (vertex : BooleanVertex sourceShape.cubeVariables) :
    ((statement.sourceProtocolData K.embed witness).padImages coordinate).valueAt vertex =
      K.embed (PaperLinearAlgebra.matrixVectorAt baseOps
        (statement.matrixSource.coefficientMatrixOf baseOps
          (statement.cubeLayout.paddedIdentityEntry (0 : F) 1) coordinate.coefficient)
        (witness.assignments (runningSourceIndex coordinate.running)) vertex) := by
  simp only [StrongReduction.Statement.sourceProtocolData,
    StrongReduction.Statement.sourceConnectedInputs, ConnectedInputs.toUnifiedInputs,
    ProtocolDataRefinement.toProtocolData, UnifiedInputs.padData,
    ConnectedInputs.padCoefficientMatrices,
    PadEvaluationResidual.imageTable, PadEvaluationResidual.imageCoefficientAt,
    BooleanTable.valueAt_tabulate]
  rfl

private theorem statement_fresh_endpoint
    {Commitment PublicInput : Type} {sourceShape : Shape} {columns blocks : Nat}
    (statement : StrongReduction.Statement K Commitment PublicInput sourceShape columns blocks baseOps)
    (witness : StrongReduction.OutputWitness sourceShape columns)
    (source : Fin sourceShape.freshCount) (matrix : Fin sourceShape.matrixCount)
    (vertex : BooleanVertex sourceShape.cubeVariables) :
    ((statement.sourceProtocolData K.embed witness).freshMatrixImages source matrix).valueAt vertex =
      K.embed (PaperLinearAlgebra.matrixVectorAt baseOps (statement.matrixSource.matrices matrix)
        (witness.assignments (freshSourceIndex source)) vertex) := by
  simp only [StrongReduction.Statement.sourceProtocolData,
    StrongReduction.Statement.sourceConnectedInputs, ConnectedInputs.toUnifiedInputs,
    ProtocolDataRefinement.toProtocolData, CCSResidualTable.matrixImagesAt,
    MatrixSource.system, BooleanTable.valueAt_tabulate]

private theorem source_matrices_eq {width : Nat}
    {fits : ringDegree * PaperAlgebra.publicRingColumns ≤ Phi81CarrierLayout.carrierWidth width}
    (system : Phi81Relation.Structure (PaperAlgebra.FullShape width fits)) :
    (PaperAlgebra.matrixSource system).matrices = system.matrixSource.matrices := rfl

private theorem source_coefficientMatrix_eq {width : Nat}
    {fits : ringDegree * PaperAlgebra.publicRingColumns ≤ Phi81CarrierLayout.carrierWidth width}
    (system : Phi81Relation.Structure (PaperAlgebra.FullShape width fits))
    (matrix : Fin productionShape.matrixCount) (coefficient : Fin ringDegree) :
    (PaperAlgebra.matrixSource system).coefficientMatrix baseOps matrix coefficient =
      system.matrixSource.coefficientMatrix baseOps matrix coefficient := rfl

private theorem source_coefficientMatrixOf_eq {width : Nat}
    {fits : ringDegree * PaperAlgebra.publicRingColumns ≤ Phi81CarrierLayout.carrierWidth width}
    (system : Phi81Relation.Structure (PaperAlgebra.FullShape width fits))
    (matrix : PaperLinearAlgebra.BooleanMatrix F Lifecycle.cubeVariables
      (Phi81CarrierLayout.carrierWidth width)) (coefficient : Fin ringDegree) :
    (PaperAlgebra.matrixSource system).coefficientMatrixOf baseOps matrix coefficient =
      system.matrixSource.coefficientMatrixOf baseOps matrix coefficient := rfl

private theorem statementFor_matrixSource
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (input : PiCCSPublicReplay.Input) :
    (statementFor relation input).matrixSource = PaperAlgebra.matrixSource relation.system := rfl

private theorem statementFor_cubeLayout
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (input : PiCCSPublicReplay.Input) :
    (statementFor relation input).cubeLayout =
      (Lifecycle.PiRLC.v1_1.InputBinding.relationSource relation).cubeLayout := rfl

private theorem matrix_endpoint_for
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (input : PiCCSPublicReplay.Input)
    (witness : StrongReduction.OutputWitness productionShape shape.carrierWidth)
    (coordinate : MatrixCoordinate productionShape)
    (vertex : BooleanVertex Lifecycle.cubeVariables) :
    (((statementFor relation input).sourceProtocolData K.embed witness).matrixImages coordinate).valueAt vertex =
      K.embed (PiRLC.rowRing relation.system
        (witness.assignments (runningSourceIndex coordinate.running))
        coordinate.matrix vertex coordinate.coefficient) := by
  erw [statement_matrix_endpoint (statementFor relation input) witness coordinate vertex,
    statementFor_matrixSource, source_coefficientMatrix_eq]
  rfl

private theorem pad_endpoint_for
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (input : PiCCSPublicReplay.Input)
    (witness : StrongReduction.OutputWitness productionShape shape.carrierWidth)
    (coordinate : PadCoordinate productionShape)
    (vertex : BooleanVertex Lifecycle.cubeVariables) :
    (((statementFor relation input).sourceProtocolData K.embed witness).padImages coordinate).valueAt vertex =
      K.embed (PiRLC.ExplicitMatrix.rowRing relation.system
        (PaperAlgebra.padMatrix (Lifecycle.PiRLC.v1_1.InputBinding.relationSource relation))
        (witness.assignments (runningSourceIndex coordinate.running)) vertex coordinate.coefficient) := by
  erw [statement_pad_endpoint (statementFor relation input) witness coordinate vertex,
    statementFor_matrixSource, statementFor_cubeLayout,
    source_coefficientMatrixOf_eq, padMatrix_eq]
  rfl

private theorem fresh_endpoint_for
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (input : PiCCSPublicReplay.Input)
    (witness : StrongReduction.OutputWitness productionShape shape.carrierWidth)
    (source : Fin productionShape.freshCount) (matrix : Fin Spec.ProductionRelation.matrixCount)
    (vertex : BooleanVertex Lifecycle.cubeVariables) :
    (((statementFor relation input).sourceProtocolData K.embed witness).freshMatrixImages source matrix).valueAt vertex =
      K.embed (PaperLinearAlgebra.matrixVectorAt baseOps
        (relation.system.matrixSource.matrices matrix)
        (witness.assignments (freshSourceIndex source)) vertex) := by
  erw [statement_fresh_endpoint (statementFor relation input) witness source matrix vertex,
    statementFor_matrixSource, source_matrices_eq]
  rfl

private theorem assignment_endpoint_for
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (input : PiCCSPublicReplay.Input)
    (witness : StrongReduction.OutputWitness productionShape shape.carrierWidth)
    (source : Fin productionShape.sourceCount)
    (vertex : BooleanVertex Lifecycle.cubeVariables) :
    (((statementFor relation input).sourceProtocolData K.embed witness).sourceAssignments source).valueAt vertex =
      K.embed (assignmentValue (Lifecycle.PiRLC.v1_1.InputBinding.relationSource relation).cubeLayout
        (witness.assignments source) vertex) := by
  change (BooleanTable.tabulate (fun current => K.embed
    ((Lifecycle.PiRLC.v1_1.InputBinding.relationSource relation).cubeLayout.paddedValue
      0 (witness.assignments source) current))).valueAt vertex = _
  exact BooleanTable.valueAt_tabulate _ _

private theorem matrix_endpoint (input : PiCCSPublicReplay.Input)
    (witness : StrongReduction.OutputWitness productionShape shape.carrierWidth)
    (coordinate : MatrixCoordinate productionShape)
    (vertex : BooleanVertex Lifecycle.cubeVariables) :
    (((selectedStatement input).sourceProtocolData K.embed witness).matrixImages coordinate).valueAt vertex =
      K.embed (PiRLC.rowRing selectedRelation.system
        (witness.assignments (runningSourceIndex coordinate.running))
        coordinate.matrix vertex coordinate.coefficient) := by
  exact matrix_endpoint_for selectedRelation input witness coordinate vertex

/-- Every running matrix endpoint is an original assignment image. There is
no matrix-equality, split, norm, row-validity or opening premise. -/
theorem runningMatrix_sourceProtocolData (input : PiCCSPublicReplay.Input)
    (witness : StrongReduction.OutputWitness productionShape shape.carrierWidth)
    (coordinate : MatrixCoordinate productionShape)
    (vertex : BooleanVertex Lifecycle.cubeVariables) :
    (matrixImage? selectedProgram selectedSource
        (witness.assignments (runningSourceIndex coordinate.running))
        coordinate.coefficient vertex).map (fun values => K.embed (values.get coordinate.matrix)) =
      some ((((selectedStatement input).sourceProtocolData K.embed witness).matrixImages coordinate).valueAt vertex) := by
  rw [matrix_endpoint]
  simpa only [Option.map_map, Function.comp_def, Option.map_some] using
    congrArg (Option.map K.embed) (matrixImage_value
      (witness.assignments (runningSourceIndex coordinate.running))
      coordinate.coefficient vertex coordinate.matrix)

/-- Every running Pad endpoint retains the original complete carrier. -/
theorem runningPad_sourceProtocolData (input : PiCCSPublicReplay.Input)
    (witness : StrongReduction.OutputWitness productionShape shape.carrierWidth)
    (coordinate : PadCoordinate productionShape)
    (vertex : BooleanVertex Lifecycle.cubeVariables) :
    K.embed (padImage selectedLayout
        (witness.assignments (runningSourceIndex coordinate.running)) coordinate.coefficient vertex) =
      (((selectedStatement input).sourceProtocolData K.embed witness).padImages coordinate).valueAt vertex := by
  rw [padImage_value]
  exact (pad_endpoint_for selectedRelation input witness coordinate vertex).symm

/-- The fresh source uses original scalar M_j z, including the matrix's
canonical zero completion. It does not use a nonconstant carried coefficient. -/
theorem freshMatrix_sourceProtocolData (input : PiCCSPublicReplay.Input)
    (witness : StrongReduction.OutputWitness productionShape shape.carrierWidth)
    (source : Fin productionShape.freshCount) (matrix : Fin Spec.ProductionRelation.matrixCount)
    (vertex : BooleanVertex Lifecycle.cubeVariables) :
    (freshMatrixImage? selectedProgram selectedSource
        (witness.assignments (freshSourceIndex source)) vertex).map
        (fun values => K.embed (values.get matrix)) =
      some ((((selectedStatement input).sourceProtocolData K.embed witness).freshMatrixImages source matrix).valueAt vertex) := by
  have endpoint :
      (((selectedStatement input).sourceProtocolData K.embed witness).freshMatrixImages source matrix).valueAt vertex =
        K.embed (PaperLinearAlgebra.matrixVectorAt baseOps
          (selectedRelation.system.matrixSource.matrices matrix)
          (witness.assignments (freshSourceIndex source)) vertex) := by
    exact fresh_endpoint_for selectedRelation input witness source matrix vertex
  rw [endpoint]
  simpa only [Option.map_map, Function.comp_def, Option.map_some] using
    congrArg (Option.map K.embed)
      (freshMatrixImage_value (witness.assignments (freshSourceIndex source)) vertex matrix)

/-- All seventeen source assignment endpoints are the padded original read. -/
theorem assignment_sourceProtocolData (input : PiCCSPublicReplay.Input)
    (witness : StrongReduction.OutputWitness productionShape shape.carrierWidth)
    (source : Fin productionShape.sourceCount)
    (vertex : BooleanVertex Lifecycle.cubeVariables) :
    K.embed (assignmentValue selectedLayout (witness.assignments source) vertex) =
      (((selectedStatement input).sourceProtocolData K.embed witness).sourceAssignments source).valueAt vertex := by
  exact (assignment_endpoint_for selectedRelation input witness source vertex).symm

private theorem images_get_ofFn {Alpha : Type} {count : Nat}
    (values : Fin count → Alpha) (index : Fin count) :
    (Vector.ofFn values).get index = values index := by
  change (Vector.ofFn values)[index.val] = values index
  rw [Vector.getElem_ofFn]

private theorem images_ofFnM_some {Alpha : Type} {count : Nat}
    (action : Fin count → Option Alpha) (values : Fin count → Alpha)
    (returned : ∀ index, action index = some (values index)) :
    Vector.ofFnM action = some (Vector.ofFn values) := by
  have same : action = fun index => some (values index) := funext returned
  rw [same]
  exact Vector.ofFnM_pure

private theorem images_preparedPad_eq
    (layout : ColumnLayout Lifecycle.cubeVariables shape.carrierWidth)
    (assignment : Phi81Relation.Assignment shape) (output : Fin ringDegree)
    (vertex : BooleanVertex Lifecycle.cubeVariables) :
    preparedPadImage (PiDECParentSparseRead.prepare ()) layout assignment output vertex =
      padImage layout assignment output vertex := by
  unfold preparedPadImage
  rw [preparedRead_eq]
  cases decoded : layout.toColumn? vertex <;>
    simp only [padImage, decoded, SparseForm.singleton, SparseForm.empty,
      SparseForm.evalSparse, List.foldl_cons, List.foldl_nil, Fin.one_mul, Fin.zero_add]

/-- The executable assembly returns the complete original-source message.
The selected source cache, basis tables and full-carrier layout discharge
all row and image correspondences; no successful-lookup premise is needed. -/
theorem images_sourceProtocolData (input : PiCCSPublicReplay.Input)
    (witness : StrongReduction.OutputWitness productionShape shape.carrierWidth)
    (vertex : BooleanVertex Lifecycle.cubeVariables) :
    images? selectedProgram
        (fun row => (PiDECCanonicalSourceCache.stored
          Poseidon2HashChainV1Package.application)[row]?)
        (PiDECParentSparseRead.prepare ()) selectedLayout witness.assignments vertex =
      some (ProtocolPolynomial.vertexMessage
        ((selectedStatement input).sourceProtocolData K.embed witness) vertex) := by
  classical
  have sourceEqual :
      (fun row => (PiDECCanonicalSourceCache.stored
        Poseidon2HashChainV1Package.application)[row]?) = selectedSource := by
    funext row
    exact PiDECCanonicalSourceCache.stored_value Poseidon2HashChainV1Package.application
      Poseidon2HashChainV1Package.fits row
  rw [sourceEqual]
  obtain ⟨fresh, freshLoaded, _⟩ := Option.map_eq_some_iff.mp
    (freshMatrix_sourceProtocolData input witness ⟨0, by decide⟩ ⟨0, by decide⟩ vertex)
  have runningPresent :
      ∀ (source : Fin productionShape.runningCount) (output : Fin ringDegree),
        ∃ values : Vector F Spec.ProductionRelation.matrixCount,
          rowValues? selectedProgram selectedSource
            (preparedRead (PiDECParentSparseRead.prepare ())
              (witness.assignments (runningSourceIndex source)) output) vertex = some values := by
    intro source output
    obtain ⟨values, returned, _⟩ := Option.map_eq_some_iff.mp
      (runningMatrix_sourceProtocolData input witness
        { running := source, matrix := ⟨0, by decide⟩, coefficient := output } vertex)
    refine ⟨values, ?_⟩
    rw [preparedRead_eq]
    simpa only [matrixImage?] using returned
  choose values returned using runningPresent
  have matricesLoaded :
      (Vector.ofFnM fun source : Fin productionShape.runningCount =>
        Vector.ofFnM fun output : Fin ringDegree =>
          rowValues? selectedProgram selectedSource
            (preparedRead (PiDECParentSparseRead.prepare ())
              (witness.assignments (runningSourceIndex source)) output) vertex) =
        some (Vector.ofFn fun source => Vector.ofFn (values source)) := by
    apply images_ofFnM_some
    intro source
    exact images_ofFnM_some _ _ (returned source)
  simp only [images?, freshLoaded, matricesLoaded, bind, Option.bind]
  apply congrArg some
  apply ProtocolPolynomial.OutputMessage.ext
  · funext source matrix
    have onlyFresh : source = ⟨0, by decide⟩ := by
      apply Fin.ext
      change source.val = 0
      have bound : source.val < 1 := source.isLt
      omega
    subst source
    have endpoint := freshMatrix_sourceProtocolData input witness
      ⟨0, by decide⟩ matrix vertex
    rw [freshLoaded] at endpoint
    exact Option.some.inj endpoint
  · funext source
    simpa only [images_get_ofFn, ProtocolPolynomial.vertexMessage] using
      assignment_sourceProtocolData input witness source vertex
  · funext coordinate
    simp only [images_get_ofFn, ProtocolPolynomial.vertexMessage]
    exact (congrArg K.embed (images_preparedPad_eq selectedLayout
      (witness.assignments (runningSourceIndex coordinate.running))
      coordinate.coefficient vertex)).trans
        (runningPad_sourceProtocolData input witness coordinate vertex)
  · funext coordinate
    simp only [images_get_ofFn, ProtocolPolynomial.vertexMessage]
    have loaded := returned coordinate.running coordinate.coefficient
    rw [preparedRead_eq] at loaded
    have endpoint := runningMatrix_sourceProtocolData input witness coordinate vertex
    change matrixImage? selectedProgram selectedSource
      (witness.assignments (runningSourceIndex coordinate.running))
      coordinate.coefficient vertex = some (values coordinate.running coordinate.coefficient)
      at loaded
    rw [loaded] at endpoint
    exact Option.some.inj endpoint

end NightstreamFPrime.Export.Stage1.PiCCSSourceImages
