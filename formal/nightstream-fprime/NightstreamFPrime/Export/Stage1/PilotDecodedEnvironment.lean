import NightstreamFPrime.Export.Stage1.PilotDecodedHashes
import NightstreamFPrime.Export.Stage1.PilotDirectSemantics
import NightstreamFPrime.Export.Stage1.PilotOrdinaryMatrixProgram

/-!
Owns the arbitrary pilot environment. Pilot ordinary locations use their
compiled forms. Other locations retain the PiCCS view, including the two
actual hash preimages. Row support proves that this extension preserves every
ordinary row; input-range exclusion proves that both hash inputs are shared.
-/

namespace NightstreamFPrime.Export.Stage1.PilotDecodedEnvironment

open NightstreamFPrime.Circuit
open NightstreamFPrime.Export.Package
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
open PilotOrdinaryDirectSource

variable {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}

def env (geometry : PilotOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth) : Env :=
  fun column => match PilotOrdinaryDirectPlan.classifyTarget column with
    | some decoded => (decoded.location.form geometry).eval assignment
    | none => PilotDecodedHashes.inputEnv
        (PilotOrdinaryDirectPlan.piCcsGeometry geometry) assignment column

private theorem location_not_input (location : PilotOrdinaryDirectPlan.Location)
    (source : Nat) (input : source < 49393 ∨ 49663 ≤ source ∧ source < 99056) :
    location.sourceColumn ≠ source := by
  intro same
  cases location with
  | priorDigest lane =>
      change 7410252 + lane.val = source at same
      omega
  | priorPublic index =>
      have bound := index.isLt
      change 49393 + index.val = source at same
      omega
  | canonicalLocal index =>
      change Lifecycle.PriorStateHash.hashEnd PilotProduction.priorInterface
        PilotProduction.witnessOffset + index.val = source at same
      unfold Lifecycle.PriorStateHash.hashEnd at same
      rw [PilotProduction.priorHashLogicalLength_eq,
        PilotProduction.witnessOffset_eq] at same
      omega
  | outputState lane =>
      change 14721716 + lane.val = source at same
      omega
  | canonicalFresh index =>
      change 14721724 + index.val = source at same
      omega
  | outputDigest lane =>
      change 99056 + lane.val = source at same
      omega

private theorem classifySource_none_of_input (source : Nat)
    (input : source < 49393 ∨ 49663 ≤ source ∧ source < 99056) :
    PilotOrdinaryDirectPlan.classifySource source = none := by
  cases found : PilotOrdinaryDirectPlan.classifySource source with
  | none => rfl
  | some located =>
      exact False.elim (location_not_input located.location source input located.owns)

private theorem prior_classify_none (index : Fin Data.priorChain.inputLength) :
    PilotOrdinaryDirectPlan.classifyTarget
      (PilotData.priorChain.inputStart + index.val) = none := by
  have bound : index.val < 49393 := index.isLt
  have mapped : PilotSpartan.sourceToSpartan index.val =
      PilotData.priorChain.inputStart + index.val := by
    unfold PilotSpartan.sourceToSpartan
    rw [if_pos (by change index.val < 49393; exact bound)]
    change index.val = 0 + index.val
    omega
  rw [← mapped, PilotOrdinaryDirectPlan.classifyTarget,
    PilotSpartan.spartanToSource_sourceToSpartan index.val (by
      change index.val < 14722512
      omega)]
  simp only [classifySource_none_of_input index.val (Or.inl bound)]

private theorem output_classify_none (index : Fin Data.outputChain.inputLength) :
    PilotOrdinaryDirectPlan.classifyTarget
      (PilotData.outputChain.inputStart + index.val) = none := by
  have bound : index.val < 49393 := index.isLt
  have mapped : PilotSpartan.sourceToSpartan (49663 + index.val) =
      PilotData.outputChain.inputStart + index.val := by
    unfold PilotSpartan.sourceToSpartan
    rw [if_neg (by change ¬49663 + index.val < 49393; omega)]
    rw [if_neg (by change ¬49663 + index.val < 49663; omega)]
    rw [if_pos (by change 49663 + index.val < 99056; omega)]
    change 49393 + ((49663 + index.val) - 49663) = 49393 + index.val
    omega
  rw [← mapped, PilotOrdinaryDirectPlan.classifyTarget,
    PilotSpartan.spartanToSource_sourceToSpartan (49663 + index.val) (by
      change 49663 + index.val < 14722512
      omega)]
  simp only [classifySource_none_of_input (49663 + index.val) (Or.inr (by omega))]

private theorem form_eval_of_target
    (geometry : PilotOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (column : Fin PilotSpartan.spartanColumnCount) (support : Target column.val) :
    ((PilotOrdinaryDirectPlan.sourceMap geometry).form column).eval assignment =
      env geometry assignment column.val := by
  rcases PilotOrdinaryDirectPlan.classifyTarget_complete support with
    ⟨decoded, found, _⟩
  simp only [PilotOrdinaryDirectPlan.sourceMap, env, found]

/-- The decoder gives each declared pilot location its actual compiled value. -/
theorem env_location
    (geometry : PilotOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (location : PilotOrdinaryDirectPlan.Location) :
    env geometry assignment (PilotSpartan.sourceToSpartan location.sourceColumn) =
      (location.form geometry).eval assignment := by
  let column : Fin PilotSpartan.spartanColumnCount :=
    ⟨PilotSpartan.sourceToSpartan location.sourceColumn, location.targetColumn_lt⟩
  have support : Target column.val :=
    ⟨location.sourceColumn, location.physicalSupport, rfl⟩
  have compiled := PilotOrdinaryMatrixProgram.substitution_agrees_on_target
    geometry column support
  have selected := PilotOrdinaryMatrixProgram.substitution_location_form?
    geometry location
  have forms : (PilotOrdinaryDirectPlan.sourceMap geometry).form column =
      location.form geometry := Option.some.inj (compiled.symm.trans selected)
  rw [← form_eval_of_target geometry assignment column support, forms]

/-- The ordinary-row view preserves every actual prior-hash input. -/
theorem priorInputForm_eval
    (geometry : PilotOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (index : Fin Data.priorChain.inputLength) :
    ((PiRLCPoseidonGeometry.priorInputBlock program).form
      (PiRLCPoseidonGeometry.priorInputStart program)
      (PiRLCPoseidonGeometry.priorInputFits
        (PilotDirectSemantics.poseidonGeometry geometry)) index).eval assignment =
      env geometry assignment (PilotData.priorChain.inputStart + index.val) := by
  rw [env, prior_classify_none]
  exact PilotDecodedHashes.priorInputForm_eval
    (PilotOrdinaryDirectPlan.piCcsGeometry geometry) assignment index

/-- The ordinary-row view preserves every actual output-hash input. -/
theorem outputInputForm_eval
    (geometry : PilotOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (index : Fin Data.outputChain.inputLength) :
    ((PiRLCPoseidonGeometry.outputInputBlock program).form
      (PiRLCPoseidonGeometry.outputInputStart program)
      (PiRLCPoseidonGeometry.outputInputFits
        (PilotDirectSemantics.poseidonGeometry geometry)) index).eval assignment =
      env geometry assignment (PilotData.outputChain.inputStart + index.val) := by
  rw [env, output_classify_none]
  exact PilotDecodedHashes.outputInputForm_eval
    (PilotOrdinaryDirectPlan.piCcsGeometry geometry) assignment index

/-- Accepted compiled pilot rows imply all source rows in the same arbitrary
environment that preserves the two actual preimages. -/
theorem rowsZero_implies_sourceRows
    (geometry : PilotOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (one : assignment (PilotOrdinaryDirectPlan.oneColumn geometry) = 1)
    (rows : (PilotOrdinaryDirectPlan.plan geometry).RowsZero assignment) :
    R1CS.RowsHold (env geometry assignment) PilotOrdinaryDirectSource.sourceRows := by
  rw [← PilotOrdinaryDirectSource.programRows_hold_iff_rowsHold]
  intro index
  have scope := PilotOrdinaryDirectSource.sourceRows_varsSatisfy _
    (List.get_mem _ (Fin.cast PilotOrdinaryDirectSource.sourceRows_length.symm index))
  have combinationPreserves (combination : R1CS.LinearCombination)
      (bounded : SourceCompiler.CombinationBounded
        PilotSpartan.spartanColumnCount combination)
      (support : combination.VarsSatisfy Target) :
      OrdinarySourcePlan.SourceMap.PreservesCombination
        (PilotOrdinaryDirectPlan.sourceMap geometry) assignment
        (env geometry assignment) combination bounded := by
    intro term member
    exact form_eval_of_target geometry assignment ⟨term.1, bounded term member⟩
      (support term member)
  have preserves := OrdinarySourcePlan.compileRow_preserves_local
    (PilotOrdinaryDirectPlan.sourceMap geometry)
    (PilotOrdinaryDirectPlan.oneColumn geometry)
    (PilotOrdinaryDirectSource.programRow index)
    (PilotOrdinaryDirectSource.programRow_bounded index) assignment
    (env geometry assignment) one
    ⟨combinationPreserves _ _ scope.1, combinationPreserves _ _ scope.2.1,
      combinationPreserves _ _ scope.2.2⟩
  exact (OrdinaryRow.planOfForms_residual_zero_iff
    (by norm_num [Lifecycle.cubeVariables]) (PilotOrdinaryDirectPlan.rowForms geometry)
    assignment (env geometry assignment) index
    (PilotOrdinaryDirectSource.programRow index) preserves).mp (rows index)

/-- The pilot and PiCCS logical views read the same complete prior preimage. -/
theorem priorWord_agrees
    (geometry : PilotOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (word : Fin PilotProduction.stateHashWords) :
    PilotSpartan.pullback (env geometry assignment)
        (PilotProduction.priorPreimageStart + word.val) =
      Spartan.pullback (PiCCSAssignmentSoundness.decodedEnv
        (PilotOrdinaryDirectPlan.piCcsGeometry geometry) assignment)
        (PilotProduction.priorPreimageStart + word.val) := by
  have pilot := priorInputForm_eval geometry assignment word
  have piCcs := PiCCSAssignmentSoundness.decodedEnv_location
    (PilotOrdinaryDirectPlan.piCcsGeometry geometry) assignment (.priorInput word)
  rw [PiCCSOrdinaryDirectPlan.Location.priorInput_form_eq_pilot] at piCcs
  have mapped : PilotSpartan.sourceToSpartan
      (PilotProduction.priorPreimageStart + word.val) =
        PilotData.priorChain.inputStart + word.val := by
    unfold PilotSpartan.sourceToSpartan
    rw [if_pos (by
      simpa only [PilotProduction.priorPreimageStart, Nat.zero_add,
        PilotSpartan.priorPublicStart] using word.isLt)]
    rfl
  unfold PilotSpartan.pullback Spartan.pullback
  rw [mapped, ← pilot]
  exact piCcs.symm

/-- The pilot and PiCCS logical views read the same complete next preimage. -/
theorem outputWord_agrees
    (geometry : PilotOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (word : Fin PilotProduction.stateHashWords) :
    PilotSpartan.pullback (env geometry assignment)
        (PilotProduction.outputPreimageStart + word.val) =
      Spartan.pullback (PiCCSAssignmentSoundness.decodedEnv
        (PilotOrdinaryDirectPlan.piCcsGeometry geometry) assignment)
        (PilotProduction.outputPreimageStart + word.val) := by
  have pilot := outputInputForm_eval geometry assignment word
  have piCcs := PiCCSAssignmentSoundness.decodedEnv_location
    (PilotOrdinaryDirectPlan.piCcsGeometry geometry) assignment (.outputInput word)
  rw [PiCCSOrdinaryDirectPlan.Location.outputInput_form_eq_pilot] at piCcs
  have bound : word.val < 49393 := word.isLt
  have mapped : PilotSpartan.sourceToSpartan
      (PilotProduction.outputPreimageStart + word.val) =
        PilotData.outputChain.inputStart + word.val := by
    unfold PilotSpartan.sourceToSpartan
    rw [if_neg (by change ¬49663 + word.val < 49393; omega)]
    rw [if_neg (by change ¬49663 + word.val < 49663; omega)]
    rw [if_pos (by change 49663 + word.val < 99056; omega)]
    change 49393 + ((49663 + word.val) - 49663) = 49393 + word.val
    omega
  unfold PilotSpartan.pullback Spartan.pullback
  rw [mapped, ← pilot]
  exact piCcs.symm

/-- The pilot encoded hash instance is the exact fresh public input read by
PiCCS. The equality follows from the shared owned forms, without copy rows. -/
theorem priorPublic_agrees
    (geometry : PilotOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (word : Fin Lifecycle.PriorStateHash.publicWidth) :
    PilotSpartan.pullback (env geometry assignment)
        (PilotProduction.priorPublicInputStart + word.val) =
      Spartan.pullback (PiCCSAssignmentSoundness.decodedEnv
        (PilotOrdinaryDirectPlan.piCcsGeometry geometry) assignment)
        (PilotProduction.priorPublicInputStart + word.val) := by
  calc
    _ = ((PilotOrdinaryDirectPlan.Location.priorPublic word).form geometry
        ).eval assignment := env_location geometry assignment (.priorPublic word)
    _ = ((PiCCSOrdinaryDirectPlan.Location.freshPublicInput word).form
        (PilotOrdinaryDirectPlan.piCcsGeometry geometry)).eval assignment := by rfl
    _ = _ := (PiCCSAssignmentSoundness.decodedEnv_location
      (PilotOrdinaryDirectPlan.piCcsGeometry geometry) assignment
      (.freshPublicInput word)).symm

end NightstreamFPrime.Export.Stage1.PilotDecodedEnvironment
