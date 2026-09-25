import NightstreamFPrime.Export.Stage1.PiCCSFreshPrefix
import NightstreamFPrime.Export.Stage1.PiCCSSourceImagesPreservation
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PrefixFold

/-! Proof-only fresh-array and prefix-MLE transport. Do not import this module
into an executable. It does not prove an IO cache loop or file decoder. -/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiCCSFreshPrefix

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.UnifiedSources
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation

private abbrev selectedRelation := PerApplicationFixedPoint.relation
  Poseidon2HashChainV1Package.application Poseidon2HashChainV1Package.fits
private abbrev selectedProgram := PerApplicationMatrixProgram.matrixProgram
  Poseidon2HashChainV1Package.application
private abbrev selectedSource := fun (row : Nat) =>
  (PiDECCanonicalSourceCache.stored Poseidon2HashChainV1Package.application)[row]?
private noncomputable abbrev selectedStatement (input : PiCCSPublicReplay.Input) :=
  (ProductionKey.key selectedRelation Poseidon2HashChainV1Setup.productionAjtaiKey).statement
    (PiCCSPublicReplay.running input) (PiCCSPublicReplay.fresh input)

private theorem selectedFits : selectedProgram.rowCount ≤ 2 ^ cubeVariables := by
  rw [PerApplicationMatrixProgram.matrixProgram_rowCount_eq_structuralPlan
    Poseidon2HashChainV1Package.application Poseidon2HashChainV1Package.fits]
  exact PerApplicationFixedPoint.structuralPlan_rowCount_le
    Poseidon2HashChainV1Package.application Poseidon2HashChainV1Package.fits

private theorem get_ofFn {Alpha : Type} {count : Nat}
    (value : Fin count → Alpha) (index : Fin count) :
    (Vector.ofFn value).get index = value index := by
  change (Vector.ofFn value)[index.val] = value index
  rw [Vector.getElem_ofFn]

private theorem selected_value (input : PiCCSPublicReplay.Input)
    (witness : StrongReduction.OutputWitness productionShape PiCCSSourceImages.shape.carrierWidth)
    (source : Fin productionShape.freshCount) (port : Fin Spec.ProductionRelation.matrixCount)
    (vertex : BooleanVertex cubeVariables) :
    (PiCCSSourceImages.freshMatrixImage? selectedProgram selectedSource
      (witness.assignments (freshSourceIndex source)) vertex).map
        (fun values => K.embed (values.get port)) =
      some ((((selectedStatement input).sourceProtocolData K.embed witness).freshMatrixImages
        source port).valueAt vertex) := by
  have cache : selectedSource = PerApplicationCanonicalPackage.sourceRow
      Poseidon2HashChainV1Package.application Poseidon2HashChainV1Package.fits := by
    funext row
    exact PiDECCanonicalSourceCache.stored_value Poseidon2HashChainV1Package.application
      Poseidon2HashChainV1Package.fits row
  rw [cache]
  exact PiCCSSourceImages.freshMatrix_sourceProtocolData input witness source port vertex

private theorem embedded_vector (action : Option (Vector F Spec.ProductionRelation.matrixCount))
    (value : Fin Spec.ProductionRelation.matrixCount → K)
    (ports : ∀ port, action.map (fun values => K.embed (values.get port)) = some (value port)) :
    action.map (fun values => values.map K.embed) = some (Vector.ofFn value) := by
  cases returned : action with
  | none =>
      have impossible := ports ⟨0, by decide⟩
      simp only [returned, Option.map_none, reduceCtorEq] at impossible
  | some values =>
      simp only [Option.map_some]
      apply congrArg some
      apply Vector.ext
      intro index inside
      have equal := ports ⟨index, inside⟩
      simp only [returned, Option.map_some] at equal
      simpa only [Vector.getElem_map, Vector.getElem_ofFn] using! Option.some.inj equal

private def reference {arity ports : Nat} (tables : Fin ports → BooleanTable K arity)
    (count : Nat) (fits : count ≤ 2 ^ arity) : Array (Vector K ports) :=
  Array.ofFn fun row : Fin count => Vector.ofFn fun port =>
    (tables port).valueAt (NumericBooleanDomain.vertex arity ⟨row.val, Nat.lt_of_lt_of_le row.isLt fits⟩)

/-- The optional numeric traversal succeeds for every original witness and
returns all active source-protocol images in numeric row order. -/
theorem rows?_value (input : PiCCSPublicReplay.Input)
    (witness : StrongReduction.OutputWitness productionShape PiCCSSourceImages.shape.carrierWidth)
    (source : Fin productionShape.freshCount) :
    rows? selectedProgram selectedSource (witness.assignments (freshSourceIndex source)) selectedFits =
      some (reference (((selectedStatement input).sourceProtocolData K.embed witness).freshMatrixImages source)
        selectedProgram.rowCount selectedFits) := by
  unfold rows? reference
  have action :
      (fun row : Fin selectedProgram.rowCount =>
        (PiCCSSourceImages.freshMatrixImage? selectedProgram selectedSource
          (witness.assignments (freshSourceIndex source))
          (NumericBooleanDomain.vertex cubeVariables
            ⟨row.val, Nat.lt_of_lt_of_le row.isLt selectedFits⟩)).map
              (fun values => values.map K.embed)) =
      (fun row : Fin selectedProgram.rowCount => some (Vector.ofFn fun port =>
        ((((selectedStatement input).sourceProtocolData K.embed witness).freshMatrixImages source port).valueAt
          (NumericBooleanDomain.vertex cubeVariables
            ⟨row.val, Nat.lt_of_lt_of_le row.isLt selectedFits⟩)))) := by
    funext row
    exact embedded_vector _ _ (fun port => selected_value input witness source port _)
  rw [action]
  exact Array.ofFnM_pure

private theorem reference_size {arity ports : Nat} (tables : Fin ports → BooleanTable K arity)
    (count : Nat) (fits : count ≤ 2 ^ arity) :
    Option.map Array.size (some (reference tables count fits)) = some count := by
  simp only [Option.map_some, reference, Array.size_ofFn]

/-- Exact active-row extent of the successful selected fresh initializer. -/
theorem rows?_size (input : PiCCSPublicReplay.Input)
    (witness : StrongReduction.OutputWitness productionShape PiCCSSourceImages.shape.carrierWidth)
    (source : Fin productionShape.freshCount) :
    (rows? selectedProgram selectedSource (witness.assignments (freshSourceIndex source))
      selectedFits).map Array.size = some selectedProgram.rowCount := by
  exact (congrArg (Option.map Array.size) (rows?_value input witness source)).trans
    (reference_size _ selectedProgram.rowCount selectedFits)

private theorem table_ext {arity : Nat} (left right : BooleanTable K arity)
    (equal : ∀ vertex, left.valueAt vertex = right.valueAt vertex) : left = right := by
  induction arity with
  | zero =>
      cases left with
      | leaf left =>
          cases right with
          | leaf right => exact congrArg BooleanTable.leaf (equal .nil)
  | succ arity ih =>
      cases left with
      | branch low high =>
          cases right with
          | branch otherLow otherHigh =>
              have lowEqual := ih low otherLow (fun vertex => equal (.cons false vertex))
              have highEqual := ih high otherHigh (fun vertex => equal (.cons true vertex))
              rw [lowEqual, highEqual]

private theorem reference_port {arity ports : Nat} (tables : Fin ports → BooleanTable K arity)
    (count : Nat) (fits : count ≤ 2 ^ arity) (port : Fin ports)
    (padding : ∀ vertex, count ≤ NumericBooleanDomain.index vertex →
      (tables port).valueAt vertex = K.zero) :
    PrefixFold.zeroExtend extensionOps arity
      ((reference tables count fits).map (fun row => row.get port)) = tables port := by
  apply table_ext
  intro vertex
  simp only [PrefixFold.zeroExtend, BooleanTable.valueAt_tabulate, reference,
    Array.map_ofFn, Array.getD_eq_getD_getElem?, Array.getElem?_ofFn]
  by_cases inside : NumericBooleanDomain.index vertex < count
  · simp only [dif_pos inside, Option.getD_some, Function.comp_apply, get_ofFn,
      NumericBooleanDomain.vertex_index]
  · simp only [dif_neg inside, Option.getD_none]
    exact (padding vertex (Nat.le_of_not_lt inside)).symm

private theorem freshOutside (program : MatrixProgram.Program)
    (sourceRow : Nat → Option R1CS.Row)
    (assignment : Phi81Relation.Assignment PiCCSSourceImages.shape)
    (vertex : BooleanVertex cubeVariables)
    (outside : program.rowCount ≤ NumericBooleanDomain.index vertex) :
    PiCCSSourceImages.freshMatrixImage? program sourceRow assignment vertex =
      some (Vector.replicate Spec.ProductionRelation.matrixCount (0 : F)) := by
  unfold PiCCSSourceImages.freshMatrixImage? PiCCSSourceImages.rowValues?
  rw [if_neg (Nat.not_lt.mpr outside)]

private theorem embeddedZero {count : Nat} (port : Fin count) :
    K.embed ((Vector.replicate count (0 : F)).get port) = K.zero := by
  change K.embed ((Vector.replicate count (0 : F))[port.val]) = K.zero
  rw [Vector.getElem_replicate]
  rfl

private theorem selected_padding (input : PiCCSPublicReplay.Input)
    (witness : StrongReduction.OutputWitness productionShape PiCCSSourceImages.shape.carrierWidth)
    (source : Fin productionShape.freshCount) (port : Fin Spec.ProductionRelation.matrixCount)
    (vertex : BooleanVertex cubeVariables)
    (outside : selectedProgram.rowCount ≤ NumericBooleanDomain.index vertex) :
    ((((selectedStatement input).sourceProtocolData K.embed witness).freshMatrixImages source port).valueAt vertex) =
      K.zero := by
  have original := selected_value input witness source port vertex
  rw [freshOutside selectedProgram selectedSource _ vertex outside, Option.map_some] at original
  exact (Option.some.inj original).symm.trans (embeddedZero port)

private theorem reference_fold {arity ports remaining : Nat}
    (tables : Fin ports → BooleanTable K arity) (count : Nat) (fits : count ≤ 2 ^ arity)
    (port : Fin ports) (padding : ∀ vertex, count ≤ NumericBooleanDomain.index vertex →
      (tables port).valueAt vertex = K.zero)
    (challenges : List K) (dimension : arity = remaining + challenges.length)
    (suffix : CubePoint K remaining) :
    (PrefixFold.zeroExtend extensionOps remaining
      (PrefixFold.foldPrefix extensionOps ((reference tables count fits).map (fun row => row.get port))
        challenges)).evaluate extensionOps suffix =
      (tables port).evaluate extensionOps
        ⟨challenges ++ suffix.coordinates, by simp [suffix.dimension]; omega⟩ := by
  subst arity
  rw [PrefixFold.foldPrefix_evaluate extensionOps extensionLaws _ challenges suffix
    (by simpa only [reference, Array.size_map, Array.size_ofFn] using fits)]
  rw [reference_port tables count fits port padding]

/-- Every port's executable prefix fold is the original source table MLE at
prefix ++ suffix. All row-success and zero-padding facts are discharged here.
This does not assert that a saved file or an IO cache loop equals rows?. -/
theorem fold_evaluate (input : PiCCSPublicReplay.Input)
    (witness : StrongReduction.OutputWitness productionShape PiCCSSourceImages.shape.carrierWidth)
    (source : Fin productionShape.freshCount) (port : Fin Spec.ProductionRelation.matrixCount)
    (challenges : List K) {remaining : Nat}
    (dimension : cubeVariables = remaining + challenges.length) (suffix : CubePoint K remaining) :
    (rows? selectedProgram selectedSource (witness.assignments (freshSourceIndex source)) selectedFits).map
      (fun rows => (PrefixFold.zeroExtend extensionOps remaining
        (PrefixFold.foldPrefix extensionOps (rows.map (fun row => row.get port)) challenges)).evaluate
          extensionOps suffix) =
      some (((((selectedStatement input).sourceProtocolData K.embed witness).freshMatrixImages source port)).evaluate
        extensionOps ⟨challenges ++ suffix.coordinates, by
          change (challenges ++ suffix.coordinates).length = cubeVariables
          simp only [List.length_append, suffix.dimension]
          omega⟩) := by
  rw [rows?_value]
  apply congrArg some
  exact reference_fold _ selectedProgram.rowCount selectedFits port
    (selected_padding input witness source port) challenges dimension suffix

end NightstreamFPrime.Export.Stage1.PiCCSFreshPrefix
