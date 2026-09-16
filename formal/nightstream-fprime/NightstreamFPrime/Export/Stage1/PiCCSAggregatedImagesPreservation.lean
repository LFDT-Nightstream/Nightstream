import NightstreamFPrime.Export.Stage1.PiCCSAggregatedImages
import NightstreamFPrime.Export.Stage1.PiCCSGammaAggregation
import NightstreamFPrime.Export.Stage1.PiCCSSourceImagesPreservation

/-! Proof-only transport from nested original-source weighted reads to the
existing optional numeric matrix interpreter. No row lookup is assumed. -/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiCCSAggregatedImages

universe uOuter uInner

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.FiniteSumAlgebra
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.UnifiedSources
open NightstreamFPrime.Export.Stage1.PiRLCPartialTrace
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Lifecycle

/-- Both weighted index levels pass through the existing sparse row.
Optional failures and all repeated indices and entries are preserved. -/
theorem row?_nested {Outer : Type uOuter} {Inner : Type uInner}
    (program : MatrixProgram.Program) {columns : Nat}
    (sourceRow : Nat → Option R1CS.Row)
    (outer : List Outer) (inner : List Inner)
    (outerWeight : Outer → K) (innerWeight : Inner → K)
    (read : Outer → Inner → Fin columns → F)
    (ordinal : Nat) (port : Fin Spec.ProductionRelation.matrixCount) :
    (PiCCSLinearRows.row? program sourceRow (fun column =>
      sumMap extensionOps outer (fun output => extensionOps.mul (outerWeight output)
        (sumMap extensionOps inner (fun source =>
          extensionOps.mul (innerWeight source) (K.embed (read output source column))))))
      ordinal).map (fun values => values.get port) =
    (program.row? columns sourceRow ordinal).map (fun forms =>
      sumMap extensionOps outer (fun output => extensionOps.mul (outerWeight output)
        (sumMap extensionOps inner (fun source => extensionOps.mul (innerWeight source)
          (K.embed ((match meaningfulPort? port with
            | some meaningful => forms meaningful
            | none => SparseForm.empty).evalSparse (read output source))))))) := by
  rw [PiCCSLinearRows.row?_value]
  apply congrArg (Option.map · (program.row? columns sourceRow ordinal))
  funext forms
  let form := match meaningfulPort? port with
    | some meaningful => forms meaningful
    | none => SparseForm.empty
  calc
    _ = sumMap extensionOps outer (fun output =>
        extensionOps.mul (outerWeight output)
          (PiCCSSparseEvaluation.evaluateK form (fun column =>
            sumMap extensionOps inner (fun source =>
              extensionOps.mul (innerWeight source)
                (K.embed (read output source column)))))) :=
      (PiCCSSparseEvaluation.weighted_readsK form outer outerWeight
        (fun output column => sumMap extensionOps inner (fun source =>
          extensionOps.mul (innerWeight source) (K.embed (read output source column))))).symm
    _ = _ := by
      apply sumMap_congr
      intro output _
      apply congrArg (extensionOps.mul (outerWeight output))
      exact (PiCCSSparseEvaluation.weighted_reads form inner innerWeight
        (read output)).symm

/-- Prepared basis coefficients and combined original source blocks give
exactly the nested source/output read, including complete carrier tails. -/
theorem carriedRead_value
    (forms : FixedArray (FixedArray (SparseForm ringDegree) ringDegree) ringDegree)
    (outputWeight : Fin ringDegree → K) (powers : Nat → K)
    (assignments : Fin productionShape.sourceCount →
      Phi81Relation.Assignment PiCCSSourceImages.shape)
    {columns : Nat} (column : Fin columns) :
    PiCCSCarriedRead.read (PiCCSCarriedRead.prepare forms outputWeight)
        (combinedBlock powers assignments) column =
      sumMap extensionOps (canonicalFinIndices ringDegree) (fun output =>
        extensionOps.mul (outputWeight output)
          (sumMap extensionOps (canonicalFinIndices productionShape.runningCount) (fun source =>
            extensionOps.mul (powers source.val) (K.embed
              (PiCCSSourceImages.preparedRead forms
                (assignments (runningSourceIndex source)) output column))))) := by
  exact PiCCSCarriedRead.read_prepare forms outputWeight
    (fun source : Fin productionShape.runningCount => powers source.val)
    (fun source block => PiCCSSourceImages.blockAt
      (assignments (runningSourceIndex source)) block) column

/-- The actual prepared carried row is the same nested weighted family of
original-source numeric images. The reference lookup remains optional. -/
theorem row?_prepared (program : MatrixProgram.Program)
    (sourceRow : Nat → Option R1CS.Row)
    (forms : FixedArray (FixedArray (SparseForm ringDegree) ringDegree) ringDegree)
    (outputWeight : Fin ringDegree → K) (powers : Nat → K)
    (assignments : Fin productionShape.sourceCount →
      Phi81Relation.Assignment PiCCSSourceImages.shape)
    (ordinal : Nat) (port : Fin Spec.ProductionRelation.matrixCount) :
    (PiCCSLinearRows.row? program sourceRow
      (PiCCSCarriedRead.read (PiCCSCarriedRead.prepare forms outputWeight)
        (combinedBlock powers assignments) : Fin PiCCSSourceImages.logicalWidth → K)
      ordinal).map (fun values => values.get port) =
    (program.row? PiCCSSourceImages.logicalWidth sourceRow ordinal).map (fun rowForms =>
      sumMap extensionOps (canonicalFinIndices ringDegree) (fun output =>
        extensionOps.mul (outputWeight output)
          (sumMap extensionOps (canonicalFinIndices productionShape.runningCount) (fun source =>
            extensionOps.mul (powers source.val) (K.embed
              ((match meaningfulPort? port with
                | some meaningful => rowForms meaningful
                | none => SparseForm.empty).evalSparse
                (PiCCSSourceImages.preparedRead forms
                  (assignments (runningSourceIndex source)) output))))))) := by
  have reads :
      (PiCCSCarriedRead.read (PiCCSCarriedRead.prepare forms outputWeight)
        (combinedBlock powers assignments) : Fin PiCCSSourceImages.logicalWidth → K) =
      fun column => sumMap extensionOps (canonicalFinIndices ringDegree) (fun output =>
        extensionOps.mul (outputWeight output)
          (sumMap extensionOps (canonicalFinIndices productionShape.runningCount) (fun source =>
            extensionOps.mul (powers source.val) (K.embed
              (PiCCSSourceImages.preparedRead forms
                (assignments (runningSourceIndex source)) output column))))) := by
    funext column
    exact carriedRead_value forms outputWeight powers assignments column
  rw [reads]
  exact row?_nested program sourceRow
    (canonicalFinIndices ringDegree) (canonicalFinIndices productionShape.runningCount)
    outputWeight (fun source : Fin productionShape.runningCount => powers source.val)
    (fun output source => PiCCSSourceImages.preparedRead forms
      (assignments (runningSourceIndex source)) output) ordinal port

private theorem get_ofFn {Alpha : Type} {count : Nat}
    (values : Fin count → Alpha) (index : Fin count) :
    (Vector.ofFn values).get index = values index := by
  change (Vector.ofFn values)[index.val] = values index
  rw [Vector.getElem_ofFn]

private theorem get_replicate {Alpha : Type} {count : Nat}
    (value : Alpha) (index : Fin count) :
    (Vector.replicate count value).get index = value := by
  change (Vector.replicate count value)[index.val] = value
  rw [Vector.getElem_replicate]

private theorem sparseValues_get {columns : Nat} (read : Fin columns → F)
    (forms : MatrixProgram.RowForms columns) (port : Fin Spec.ProductionRelation.matrixCount) :
    (PiDECMatrixNumericRows.sparseValues read forms).get port =
      (match meaningfulPort? port with
        | some meaningful => forms meaningful
        | none => SparseForm.empty).evalSparse read := by
  exact get_ofFn _ port

private theorem sparseValues_empty {columns : Nat} (read : Fin columns → F) :
    PiDECMatrixNumericRows.sparseValues read (fun _ => SparseForm.empty) =
      Vector.replicate Spec.ProductionRelation.matrixCount 0 := by
  apply Vector.ext
  intro index bounded
  simp only [PiDECMatrixNumericRows.sparseValues, Vector.getElem_ofFn,
    Vector.getElem_replicate]
  cases meaningfulPort? ⟨index, bounded⟩ <;> rfl

-- This optional form occurs only in proof statements. The executable
-- continues to call the numeric interpreter, including in the active case.
private theorem scalarRows_reference (program : MatrixProgram.Program)
    (sourceRow : Nat → Option R1CS.Row)
    (read : Fin PiCCSSourceImages.logicalWidth → F)
    (vertex : BooleanVertex cubeVariables) :
    PiCCSSourceImages.rowValues? program sourceRow read vertex =
      (if NumericBooleanDomain.index vertex < program.rowCount then
        program.row? PiCCSSourceImages.logicalWidth sourceRow (NumericBooleanDomain.index vertex)
       else some (fun _ => SparseForm.empty)).map (PiDECMatrixNumericRows.sparseValues read) := by
  by_cases active : NumericBooleanDomain.index vertex < program.rowCount
  · simpa only [PiCCSSourceImages.rowValues?, if_pos active] using
      PiDECMatrixNumericRows.row?_eq program sourceRow read (NumericBooleanDomain.index vertex)
  · simp only [PiCCSSourceImages.rowValues?, if_neg active, Option.map_some, sparseValues_empty]

private theorem carriedRows_reference (program : MatrixProgram.Program)
    (sourceRow : Nat → Option R1CS.Row)
    (forms : FixedArray (FixedArray (SparseForm ringDegree) ringDegree) ringDegree)
    (outputWeight : Fin ringDegree → K) (powers : Nat → K)
    (assignments : Fin productionShape.sourceCount →
      Phi81Relation.Assignment PiCCSSourceImages.shape)
    (vertex : BooleanVertex cubeVariables) (port : Fin Spec.ProductionRelation.matrixCount) :
    (if NumericBooleanDomain.index vertex < program.rowCount then
      PiCCSLinearRows.row? program sourceRow
        (PiCCSCarriedRead.read (PiCCSCarriedRead.prepare forms outputWeight)
          (combinedBlock powers assignments) : Fin PiCCSSourceImages.logicalWidth → K)
        (NumericBooleanDomain.index vertex)
     else some (Vector.replicate Spec.ProductionRelation.matrixCount K.zero)).map
        (fun values => values.get port) =
    (if NumericBooleanDomain.index vertex < program.rowCount then
      program.row? PiCCSSourceImages.logicalWidth sourceRow (NumericBooleanDomain.index vertex)
     else some (fun _ => SparseForm.empty)).map (fun rowForms =>
      sumMap extensionOps (canonicalFinIndices ringDegree) (fun output =>
        extensionOps.mul (outputWeight output)
          (sumMap extensionOps (canonicalFinIndices productionShape.runningCount) (fun source =>
            extensionOps.mul (powers source.val) (K.embed
              ((match meaningfulPort? port with
                | some meaningful => rowForms meaningful
                | none => SparseForm.empty).evalSparse
                (PiCCSSourceImages.preparedRead forms
                  (assignments (runningSourceIndex source)) output))))))) := by
  by_cases active : NumericBooleanDomain.index vertex < program.rowCount
  · simpa only [if_pos active] using row?_prepared program sourceRow forms outputWeight
      powers assignments (NumericBooleanDomain.index vertex) port
  · simp only [if_neg active, Option.map_some, get_replicate]
    cases meaningfulPort? port <;>
      simp only [SparseForm.evalSparse, SparseForm.empty, List.foldl_nil,
        show K.embed (0 : F) = extensionOps.zero from rfl,
        extensionLaws.mul_zero, sumMap_zero extensionOps extensionLaws] <;> rfl

private theorem option_matrix_eq_some
    (result : Option (Vector K Spec.ProductionRelation.matrixCount))
    (value : Fin Spec.ProductionRelation.matrixCount → K)
    (ports : ∀ port, result.map (fun values => values.get port) = some (value port)) :
    result = some (Vector.ofFn value) := by
  cases result with
  | none => cases ports ⟨0, by decide⟩
  | some result =>
      apply congrArg some
      apply Vector.ext
      intro index bounded
      rw [Vector.getElem_ofFn]
      exact Option.some.inj (ports ⟨index, bounded⟩)

private theorem pad_value
    (forms : FixedArray (FixedArray (SparseForm ringDegree) ringDegree) ringDegree)
    (layout : UnifiedSources.ColumnLayout cubeVariables PiCCSSourceImages.shape.carrierWidth)
    (assignments : Fin productionShape.sourceCount →
      Phi81Relation.Assignment PiCCSSourceImages.shape)
    (powers : Nat → K) (vertex : BooleanVertex cubeVariables) :
    (match layout.toColumn? vertex with
      | some column => PiCCSCarriedRead.read (prepare forms powers).1
          (combinedBlock powers assignments) column
      | none => K.zero) =
      PiCCSGammaAggregation.padTotal extensionOps powers
        (fun coordinate : PadCoordinate productionShape => K.embed
          (PiCCSSourceImages.preparedPadImage forms layout
            (assignments (runningSourceIndex coordinate.running)) coordinate.coefficient vertex)) := by
  cases decoded : layout.toColumn? vertex with
  | none =>
      simp only [decoded, PiCCSGammaAggregation.padTotal,
        PiCCSSourceImages.preparedPadImage,
        show K.embed (0 : F) = extensionOps.zero from rfl,
        extensionLaws.mul_zero, sumMap_zero extensionOps extensionLaws]
      rfl
  | some column =>
      simpa only [decoded, prepare, PiCCSGammaAggregation.padTotal,
        PiCCSSourceImages.preparedPadImage] using carriedRead_value forms
          (fun output => powers (productionShape.runningCount * output.val)) powers assignments column

private theorem ofFnM_some {Alpha : Type} {count : Nat}
    (values : Fin count → Alpha) :
    (Vector.ofFnM (fun index => some (values index)) : Option (Vector Alpha count)) =
      some (Vector.ofFn values) := by
  exact Vector.ofFnM_pure

private theorem images_of_rows (program : MatrixProgram.Program)
    (sourceRow : Nat → Option R1CS.Row)
    (forms : FixedArray (FixedArray (SparseForm ringDegree) ringDegree) ringDegree)
    (layout : UnifiedSources.ColumnLayout cubeVariables PiCCSSourceImages.shape.carrierWidth)
    (assignments : Fin productionShape.sourceCount →
      Phi81Relation.Assignment PiCCSSourceImages.shape)
    (vertex : BooleanVertex cubeVariables)
    (fresh : Vector F Spec.ProductionRelation.matrixCount)
    (values : Fin productionShape.runningCount → Fin ringDegree →
      Vector F Spec.ProductionRelation.matrixCount)
    (freshLoaded : PiCCSSourceImages.freshMatrixImage? program sourceRow
      (assignments (freshSourceIndex ⟨0, by decide⟩)) vertex = some fresh)
    (runningLoaded : ∀ source output, PiCCSSourceImages.rowValues? program sourceRow
      (PiCCSSourceImages.preparedRead forms (assignments (runningSourceIndex source)) output)
        vertex = some (values source output)) :
    PiCCSSourceImages.images? program sourceRow forms layout assignments vertex =
      some {
        freshMatrixImage := fun _ matrix => K.embed (fresh.get matrix)
        sourceAssignment := fun source =>
          K.embed (PiCCSSourceImages.assignmentValue layout (assignments source) vertex)
        padImage := fun coordinate => K.embed (PiCCSSourceImages.preparedPadImage forms layout
          (assignments (runningSourceIndex coordinate.running)) coordinate.coefficient vertex)
        matrixImage := fun coordinate =>
          K.embed ((values coordinate.running coordinate.coefficient).get coordinate.matrix) } := by
  simp only [PiCCSSourceImages.images?, freshLoaded, runningLoaded, ofFnM_some,
    bind, Option.bind, get_ofFn]
  rfl

/-- The aggregated endpoint is a total optional image transformation.
Fresh/norm fields are unchanged; unused carried fields are zero, and the
returned scalars are the complete factored carried sums. Failures agree. -/
theorem endpoint_images (program : MatrixProgram.Program)
    (sourceRow : Nat → Option R1CS.Row)
    (forms : FixedArray (FixedArray (SparseForm ringDegree) ringDegree) ringDegree)
    (layout : UnifiedSources.ColumnLayout cubeVariables PiCCSSourceImages.shape.carrierWidth)
    (assignments : Fin productionShape.sourceCount →
      Phi81Relation.Assignment PiCCSSourceImages.shape)
    (powers : Nat → K) (vertex : BooleanVertex cubeVariables) :
    endpoint? program sourceRow layout assignments (prepare forms powers).1
        (prepare forms powers).2 (combinedBlock powers assignments) powers vertex =
      (PiCCSSourceImages.images? program sourceRow forms layout assignments vertex).map
        (fun message =>
          ({ message with padImage := fun _ => K.zero, matrixImage := fun _ => K.zero },
           PiCCSGammaAggregation.padTotal extensionOps powers message.padImage,
           PiCCSGammaAggregation.matrixTotal extensionOps powers message.matrixImage)) := by
  let reference : Option (MatrixProgram.RowForms PiCCSSourceImages.logicalWidth) :=
    if NumericBooleanDomain.index vertex < program.rowCount then
      program.row? PiCCSSourceImages.logicalWidth sourceRow (NumericBooleanDomain.index vertex)
    else some (fun _ => SparseForm.empty)
  have freshRow := scalarRows_reference program sourceRow
    (PiCCSSourceImages.plainRead (assignments (freshSourceIndex ⟨0, by decide⟩))) vertex
  change PiCCSSourceImages.freshMatrixImage? program sourceRow
      (assignments (freshSourceIndex ⟨0, by decide⟩)) vertex =
    reference.map (PiDECMatrixNumericRows.sparseValues
      (PiCCSSourceImages.plainRead (assignments (freshSourceIndex ⟨0, by decide⟩)))) at freshRow
  cases decoded : reference with
  | none =>
      rw [decoded] at freshRow
      simp only [endpoint?, PiCCSSourceImages.images?, freshRow, bind, Option.bind, Option.map_none]
  | some rowForms =>
      have referenceLoaded :
          (if NumericBooleanDomain.index vertex < program.rowCount then
            program.row? PiCCSSourceImages.logicalWidth sourceRow
              (NumericBooleanDomain.index vertex)
           else some (fun _ => SparseForm.empty)) = some rowForms := decoded
      rw [decoded] at freshRow
      let fresh := PiDECMatrixNumericRows.sparseValues
        (PiCCSSourceImages.plainRead (assignments (freshSourceIndex ⟨0, by decide⟩))) rowForms
      let values := fun (source : Fin productionShape.runningCount) (output : Fin ringDegree) =>
        PiDECMatrixNumericRows.sparseValues
          (PiCCSSourceImages.preparedRead forms (assignments (runningSourceIndex source)) output)
          rowForms
      have freshLoaded : PiCCSSourceImages.freshMatrixImage? program sourceRow
          (assignments (freshSourceIndex ⟨0, by decide⟩)) vertex = some fresh := freshRow
      have runningLoaded : ∀ source output, PiCCSSourceImages.rowValues? program sourceRow
          (PiCCSSourceImages.preparedRead forms (assignments (runningSourceIndex source)) output)
          vertex = some (values source output) := by
        intro source output
        have loaded := scalarRows_reference program sourceRow
          (PiCCSSourceImages.preparedRead forms (assignments (runningSourceIndex source)) output)
          vertex
        rw [referenceLoaded] at loaded
        exact loaded
      let aggregate := fun matrix : Fin Spec.ProductionRelation.matrixCount =>
        sumMap extensionOps (canonicalFinIndices ringDegree) (fun output =>
          extensionOps.mul (powers
            (productionShape.runningCount * productionShape.matrixCount * output.val))
            (sumMap extensionOps (canonicalFinIndices productionShape.runningCount) (fun source =>
              extensionOps.mul (powers source.val) (K.embed ((values source output).get matrix)))))
      have matrixLoaded :
          (if NumericBooleanDomain.index vertex < program.rowCount then
            PiCCSLinearRows.row? program sourceRow
              (PiCCSCarriedRead.read (prepare forms powers).2 (combinedBlock powers assignments) :
                Fin PiCCSSourceImages.logicalWidth → K) (NumericBooleanDomain.index vertex)
           else some (Vector.replicate Spec.ProductionRelation.matrixCount K.zero)) =
            some (Vector.ofFn aggregate) := by
        apply option_matrix_eq_some
        intro matrix
        have loaded := carriedRows_reference program sourceRow forms
          (fun output => powers
            (productionShape.runningCount * productionShape.matrixCount * output.val))
          powers assignments vertex matrix
        rw [referenceLoaded] at loaded
        simpa only [prepare, aggregate, values, sparseValues_get, Option.map_some] using loaded
      rw [images_of_rows program sourceRow forms layout assignments vertex fresh values
        freshLoaded runningLoaded, Option.map_some]
      have joined :
          endpoint? program sourceRow layout assignments (prepare forms powers).1
              (prepare forms powers).2 (combinedBlock powers assignments) powers vertex =
            some (fromRows layout assignments (prepare forms powers).1
              (combinedBlock powers assignments) powers vertex fresh (Vector.ofFn aggregate)) := by
        by_cases active : NumericBooleanDomain.index vertex < program.rowCount
        · simp only [if_pos active] at matrixLoaded
          simp only [endpoint?, freshLoaded, if_pos active, matrixLoaded, bind, Option.bind]
          rfl
        · simp only [if_neg active] at matrixLoaded
          have emptyRows := Option.some.inj matrixLoaded
          simp only [endpoint?, freshLoaded, if_neg active, bind, Option.bind, emptyRows]
          rfl
      rw [joined]
      apply congrArg some
      simp only [fromRows, get_ofFn]
      apply Prod.ext
      · rfl
      · apply Prod.ext
        · exact pad_value forms layout assignments powers vertex
        · rfl

abbrev selectedRelation := PerApplicationFixedPoint.relation
  Poseidon2HashChainV1Package.application Poseidon2HashChainV1Package.fits
abbrev selectedProgram := PerApplicationMatrixProgram.matrixProgram
  Poseidon2HashChainV1Package.application
abbrev selectedSource := fun (row : Nat) =>
  (PiDECCanonicalSourceCache.stored Poseidon2HashChainV1Package.application)[row]?
abbrev selectedLayout :=
  (Lifecycle.PiRLC.v1_1.InputBinding.relationSource selectedRelation).cubeLayout

noncomputable abbrev selectedMessage (input : PiCCSPublicReplay.Input)
    (witness : StrongReduction.OutputWitness productionShape PiCCSSourceImages.shape.carrierWidth)
    (vertex : BooleanVertex cubeVariables) :=
  ProtocolPolynomial.vertexMessage
    (((ProductionKey.key selectedRelation Poseidon2HashChainV1Setup.productionAjtaiKey).statement
      (PiCCSPublicReplay.running input) (PiCCSPublicReplay.fresh input)).sourceProtocolData
        K.embed witness) vertex

/-- The selected endpoint retains the exact nonlinear vertex fields and
returns both complete canonical carried sums. Source rows, the full-carrier
layout and both prepared bases are fixed by their checked owners. The pair
kernel applies the global matrix shift separately, exactly once. -/
theorem endpoint_sourceProtocolData (input : PiCCSPublicReplay.Input)
    (witness : StrongReduction.OutputWitness productionShape PiCCSSourceImages.shape.carrierWidth)
    (gamma : K) (vertex : BooleanVertex cubeVariables) :
    endpoint? selectedProgram selectedSource selectedLayout witness.assignments
        (prepare (PiDECParentSparseRead.prepare ())
          (TargetPolynomial.power extensionOps.toOps gamma)).1
        (prepare (PiDECParentSparseRead.prepare ())
          (TargetPolynomial.power extensionOps.toOps gamma)).2
        (combinedBlock (TargetPolynomial.power extensionOps.toOps gamma) witness.assignments)
        (TargetPolynomial.power extensionOps.toOps gamma) vertex =
      some (
        { (selectedMessage input witness vertex) with
          padImage := fun _ => K.zero, matrixImage := fun _ => K.zero },
        sumMap extensionOps (canonicalPadCoordinates productionShape) (fun coordinate =>
          extensionOps.mul
            (TargetPolynomial.power extensionOps.toOps gamma coordinate.localGammaExponent)
            ((selectedMessage input witness vertex).padImage coordinate)),
        sumMap extensionOps (canonicalMatrixCoordinates productionShape) (fun coordinate =>
          extensionOps.mul
            (TargetPolynomial.power extensionOps.toOps gamma coordinate.localGammaExponent)
            ((selectedMessage input witness vertex).matrixImage coordinate))) := by
  have original :
      PiCCSSourceImages.images? selectedProgram selectedSource (PiDECParentSparseRead.prepare ())
        selectedLayout witness.assignments vertex = some (selectedMessage input witness vertex) := by
    exact PiCCSSourceImages.images_sourceProtocolData input witness vertex
  have result := endpoint_images selectedProgram selectedSource (PiDECParentSparseRead.prepare ())
    selectedLayout witness.assignments (TargetPolynomial.power extensionOps.toOps gamma) vertex
  rw [original, Option.map_some,
    PiCCSGammaAggregation.padTotal_power extensionOps extensionLaws gamma,
    PiCCSGammaAggregation.matrixTotal_power extensionOps extensionLaws gamma] at result
  exact result

end NightstreamFPrime.Export.Stage1.PiCCSAggregatedImages
