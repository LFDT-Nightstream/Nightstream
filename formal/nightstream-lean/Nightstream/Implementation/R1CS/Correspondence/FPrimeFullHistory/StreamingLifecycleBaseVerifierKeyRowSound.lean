import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingLifecycleBaseVerifierKeyProgramBridge
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingVariableHashRecipeConstantSound

/-!
Contract: the three exact base verifier-key core recipe traces compute
Poseidon2 on their complete typed preimages.

Owns the semantic consequence of satisfying the formulaic trace rows. It does
not prove source-input authority outside the stated column equalities or that
final selective rows imply these source rows.

Assurance tier: artifact-checked for
`FPRIME-STREAMING-LIFECYCLE-BASE-VERIFIER-KEY-PROVENANCE-V1` and
model-level for `FPRIME-STREAMING-BASE-VERIFIER-KEY-PROGRAM-V1`,
Nightstream b2/k16.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingLifecycleBaseVerifierKeyRowSound

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Poseidon2Sponge
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingVariableHashRecipe.Artifact
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleBaseVerifierKey.Artifact
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingLifecycleBaseVerifierKey
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingLifecycleBaseVerifierKey
open Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingLifecycleBaseVerifierKeyProgramBridge
open Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingVariableHashRecipeConstantSound
open Nightstream.Implementation.Rust.CanonicalConformance.NativeStep
open Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.VerifierKeyProgram

def computedBaseDigest (inputs : Inputs) : DigestFields := fun lane =>
  runValueRounds rawArtifact.baseVerifierKeyHash.recipe.trace.rounds
    (basePreimage productionParameters inputs) (fun _ => 0) lane.val

def computedPolicyDigest (inputs : Inputs) : DigestFields := fun lane =>
  runValueRounds rawArtifact.policyVerifierKeyHash.recipe.trace.rounds
    (policyPreimage (computedBaseDigest inputs)) (fun _ => 0) lane.val

def computedInitialBoundary (inputs : Inputs) : DigestFields := fun lane =>
  runValueRounds rawArtifact.initialBoundaryHash.recipe.trace.rounds
    (initialBoundaryPreimage inputs.structureDigest
      productionParameters.publicInputLength) (fun _ => 0) lane.val

structure InputColumnsBound (assignment : Nat -> Nat) (inputs : Inputs) : Prop where
  structureDigest : readColumns assignment structureDigestColumns =
    digestFields inputs.structureDigest
  piCcsHeader : readColumns assignment piCcsHeaderColumns =
    digestFields inputs.piCcsHeader
  ajtaiPpDigest : readColumns assignment ajtaiPpDigestColumns =
    digestFields inputs.ajtaiPpDigest
  initialSemanticState :
    readColumns assignment initialSemanticStateDigestColumns =
      digestFields inputs.initialSemanticStateDigest

/-- Structural satisfaction of the exact base verifier-key core rows. -/
structure StageSatisfied (assignment : Nat -> Nat) : Prop where
  baseConstants : Satisfies
    (constantRows rawArtifact.baseVerifierKeyHash.recipe) assignment
  baseHash : Satisfies
    rawArtifact.baseVerifierKeyHash.recipe.trace.rows assignment
  policyConstants : Satisfies
    (constantRows rawArtifact.policyVerifierKeyHash.recipe) assignment
  policyHash : Satisfies
    rawArtifact.policyVerifierKeyHash.recipe.trace.rows assignment
  policyDigestBinding : Satisfies
    rawArtifact.policyDigestBinding.rows assignment
  initialBoundaryConstants : Satisfies
    (constantRows rawArtifact.initialBoundaryHash.recipe) assignment
  initialBoundaryHash : Satisfies
    rawArtifact.initialBoundaryHash.recipe.trace.rows assignment
  initialBoundaryBinding : Satisfies
    rawArtifact.initialBoundaryBinding.rows assignment

structure StageOutputs
    (assignment : Nat -> Nat) (inputs : Inputs) : Prop where
  baseVerifierKeyDigest : forall lane : Fin 4,
    assignment
        (rawArtifact.baseVerifierKeyHash.recipe.outputColumns.getD lane.val 0) =
      computedBaseDigest inputs lane
  policyVerifierKeyDigest : forall lane : Fin 4,
    assignment
        (rawArtifact.policyVerifierKeyHash.recipe.outputColumns.getD lane.val 0) =
      computedPolicyDigest inputs lane
  vkFsDigest : forall lane : Fin 4,
    assignment
        (rawArtifact.policyDigestBinding.leftColumns.getD lane.val 0) =
      computedPolicyDigest inputs lane
  initialBoundaryDigest : forall lane : Fin 4,
    assignment
        (rawArtifact.initialBoundaryHash.recipe.outputColumns.getD lane.val 0) =
      computedInitialBoundary inputs lane
  initialBoundary : forall lane : Fin 4,
    assignment
        (rawArtifact.initialBoundaryBinding.leftColumns.getD lane.val 0) =
      computedInitialBoundary inputs lane

private theorem base_constantColumns_exact :
    rawArtifact.baseVerifierKeyHash.recipe.constantColumns =
      baseTagColumns ++ baseParameterColumns := by
  rfl

private theorem policy_constantColumns_exact :
    rawArtifact.policyVerifierKeyHash.recipe.constantColumns =
      policyTagColumns ++ policyFlagColumns := by
  rfl

private theorem initialBoundary_constantColumns_exact :
    rawArtifact.initialBoundaryHash.recipe.constantColumns =
      initialBoundaryTagColumns ++ publicInputLengthColumns := by
  rfl

private theorem base_constantValues_canonical :
    ∀ value ∈ rawArtifact.baseVerifierKeyHash.recipe.constantValues,
      value < goldilocksP := by
  rw [base_constantFields_exact]
  norm_num [baseConstantFields, productionParameters, vkFsTag, u64Halves,
    rawFieldOfNat, optionalUsizeFields, goldilocksModulus, twoPow32,
    goldilocksP]

private theorem policy_constantValues_canonical :
    ∀ value ∈ rawArtifact.policyVerifierKeyHash.recipe.constantValues,
      value < goldilocksP := by
  rw [policy_constantFields_exact]
  norm_num [policyConstantFields, vkFsPolicyTag, goldilocksP]

private theorem initialBoundary_constantValues_canonical :
    ∀ value ∈ rawArtifact.initialBoundaryHash.recipe.constantValues,
      value < goldilocksP := by
  rw [initialBoundary_constantFields_exact]
  norm_num [initialBoundaryConstantFields, productionParameters,
    initialBoundaryTag, optionalUsizeFields, rawFieldOfNat,
    goldilocksModulus, goldilocksP]

private theorem base_fixedSegments
    (assignment : Nat -> Nat)
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : Satisfies
      (constantRows rawArtifact.baseVerifierKeyHash.recipe) assignment) :
    readColumns assignment baseTagColumns = vkFsTag /\
      readColumns assignment baseParameterColumns =
        (baseConstantFields productionParameters).drop 5 := by
  have constants := constantRows_values
    rawArtifact.baseVerifierKeyHash.recipe assignment canonical one
      base_constantValues_canonical satisfied
  rw [base_constantColumns_exact, base_constantFields_exact] at constants
  constructor
  · have headValues := congrArg (List.take 5) constants
    simpa [List.map_append, readColumns, baseTagColumns,
      baseConstantFields, vkFsTag] using headValues
  · have tailValues := congrArg (List.drop 5) constants
    simpa [List.map_append, readColumns, baseTagColumns] using tailValues

private theorem policy_fixedSegments
    (assignment : Nat -> Nat)
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : Satisfies
      (constantRows rawArtifact.policyVerifierKeyHash.recipe) assignment) :
    readColumns assignment policyTagColumns = vkFsPolicyTag /\
      readColumns assignment policyFlagColumns = [1, 1, 1] := by
  have constants := constantRows_values
    rawArtifact.policyVerifierKeyHash.recipe assignment canonical one
      policy_constantValues_canonical satisfied
  rw [policy_constantColumns_exact, policy_constantFields_exact] at constants
  constructor
  · have headValues := congrArg (List.take 6) constants
    simpa [List.map_append, readColumns, policyTagColumns,
      policyConstantFields, vkFsPolicyTag] using headValues
  · have tailValues := congrArg (List.drop 6) constants
    simpa [List.map_append, readColumns, policyTagColumns,
      policyConstantFields, vkFsPolicyTag] using tailValues

private theorem initialBoundary_fixedSegments
    (assignment : Nat -> Nat)
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : Satisfies
      (constantRows rawArtifact.initialBoundaryHash.recipe) assignment) :
    readColumns assignment initialBoundaryTagColumns = initialBoundaryTag /\
      readColumns assignment publicInputLengthColumns =
        optionalUsizeFields productionParameters.publicInputLength := by
  have constants := constantRows_values
    rawArtifact.initialBoundaryHash.recipe assignment canonical one
      initialBoundary_constantValues_canonical satisfied
  rw [initialBoundary_constantColumns_exact,
    initialBoundary_constantFields_exact] at constants
  constructor
  · have headValues := congrArg (List.take 6) constants
    simpa [List.map_append, readColumns, initialBoundaryTagColumns,
      initialBoundaryConstantFields, initialBoundaryTag] using headValues
  · have tailValues := congrArg (List.drop 6) constants
    simpa [List.map_append, readColumns, initialBoundaryTagColumns] using
      tailValues

theorem baseHash_rows_imply_typedPreimage
    (assignment : Nat -> Nat) (inputs : Inputs)
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (constantRowsSatisfied : Satisfies
      (constantRows rawArtifact.baseVerifierKeyHash.recipe) assignment)
    (satisfied : Satisfies
      rawArtifact.baseVerifierKeyHash.recipe.trace.rows assignment)
    (structureValues : readColumns assignment structureDigestColumns =
      digestFields inputs.structureDigest)
    (header : readColumns assignment piCcsHeaderColumns =
      digestFields inputs.piCcsHeader)
    (ajtai : readColumns assignment ajtaiPpDigestColumns =
      digestFields inputs.ajtaiPpDigest)
    (initialSemantic :
      readColumns assignment initialSemanticStateDigestColumns =
        digestFields inputs.initialSemanticStateDigest) :
    forall lane, lane < 4 ->
      assignment
          (rawArtifact.baseVerifierKeyHash.recipe.trace.outputColumns.getD
            lane 0) =
        runValueRounds rawArtifact.baseVerifierKeyHash.recipe.trace.rounds
          (basePreimage productionParameters inputs) (fun _ => 0) lane := by
  have inputValues :
      rawArtifact.baseVerifierKeyHash.recipe.trace.inputColumns.map
          assignment =
        basePreimage productionParameters inputs := by
    have fixed := base_fixedSegments assignment canonical one
      constantRowsSatisfied
    change rawArtifact.baseVerifierKeyHash.recipe.inputColumns.map
      assignment = _
    rw [base_inputColumns_exact]
    exact basePreimage_from_columnProgram assignment inputs fixed.1
      structureValues header ajtai fixed.2 initialSemantic
  intro lane laneLt
  rw [ownedTrace_values_sound baseHash_trace_ownedValid canonical one
    satisfied lane laneLt, inputValues]

theorem policyHash_rows_imply_typedPreimage
    (assignment : Nat -> Nat) (baseDigest : DigestFields)
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (constantRowsSatisfied : Satisfies
      (constantRows rawArtifact.policyVerifierKeyHash.recipe) assignment)
    (satisfied : Satisfies
      rawArtifact.policyVerifierKeyHash.recipe.trace.rows assignment)
    (base : readColumns assignment baseDigestColumns =
      digestFields baseDigest) :
    forall lane, lane < 4 ->
      assignment
          (rawArtifact.policyVerifierKeyHash.recipe.trace.outputColumns.getD
            lane 0) =
        runValueRounds rawArtifact.policyVerifierKeyHash.recipe.trace.rounds
          (policyPreimage baseDigest) (fun _ => 0) lane := by
  have inputValues :
      rawArtifact.policyVerifierKeyHash.recipe.trace.inputColumns.map
          assignment =
        policyPreimage baseDigest := by
    have fixed := policy_fixedSegments assignment canonical one
      constantRowsSatisfied
    change rawArtifact.policyVerifierKeyHash.recipe.inputColumns.map
      assignment = _
    rw [policy_inputColumns_exact]
    exact policyPreimage_from_columnProgram assignment baseDigest fixed.1 base
      fixed.2
  intro lane laneLt
  rw [ownedTrace_values_sound policyHash_trace_ownedValid canonical one
    satisfied lane laneLt, inputValues]

theorem initialBoundaryHash_rows_imply_typedPreimage
    (assignment : Nat -> Nat) (structureDigest : DigestFields)
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (constantRowsSatisfied : Satisfies
      (constantRows rawArtifact.initialBoundaryHash.recipe) assignment)
    (satisfied : Satisfies
      rawArtifact.initialBoundaryHash.recipe.trace.rows assignment)
    (structureValues : readColumns assignment structureDigestColumns =
      digestFields structureDigest) :
    forall lane, lane < 4 ->
      assignment
          (rawArtifact.initialBoundaryHash.recipe.trace.outputColumns.getD
            lane 0) =
        runValueRounds rawArtifact.initialBoundaryHash.recipe.trace.rounds
          (initialBoundaryPreimage structureDigest
            productionParameters.publicInputLength) (fun _ => 0) lane := by
  have inputValues :
      rawArtifact.initialBoundaryHash.recipe.trace.inputColumns.map
          assignment =
        initialBoundaryPreimage structureDigest
          productionParameters.publicInputLength := by
    have fixed := initialBoundary_fixedSegments assignment canonical one
      constantRowsSatisfied
    change rawArtifact.initialBoundaryHash.recipe.inputColumns.map
      assignment = _
    rw [initialBoundary_inputColumns_exact]
    exact initialBoundaryPreimage_from_columnProgram assignment
      structureDigest fixed.1 structureValues fixed.2
  intro lane laneLt
  rw [ownedTrace_values_sound initialBoundaryHash_trace_ownedValid canonical
    one satisfied lane laneLt, inputValues]

private theorem digestBinding_rows_sound
    (assignment : Nat -> Nat)
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1) (binding : DigestBinding)
    (satisfied : Satisfies binding.rows assignment) (lane : Fin 4) :
    assignment (binding.leftColumns.getD lane.val 0) =
      assignment (binding.rightColumns.getD lane.val 0) := by
  have holds : RowHolds assignment (binding.row lane.val) :=
    satisfied _ (List.mem_map.mpr
      ⟨lane.val, List.mem_range.mpr lane.isLt, rfl⟩)
  have leftLt := canonical (binding.leftColumns.getD lane.val 0)
  have rightLt := canonical (binding.rightColumns.getD lane.val 0)
  simp only [DigestBinding.row, RowHolds, lcEval, List.foldl, one,
    goldilocksP] at holds leftLt rightLt
  omega

private theorem readColumns_eq_digestFields
    (assignment : Nat -> Nat) (columns : List Nat) (digest : DigestFields)
    (length : columns.length = 4)
    (pointwise : forall lane : Fin 4,
      assignment (columns.getD lane.val 0) = digest lane) :
    readColumns assignment columns = digestFields digest := by
  apply List.ext_get
  · simp [readColumns, digestFields, length]
  · intro index leftLt rightLt
    simp only [readColumns, List.get_eq_getElem, List.getElem_map,
      digestFields, List.getElem_ofFn]
    have columnLt : index < columns.length := by
      simpa [readColumns] using leftLt
    have indexLt : index < 4 := by omega
    have atIndex := pointwise ⟨index, indexLt⟩
    rw [← List.getElem_eq_getD (l := columns) (h := columnLt) 0] at atIndex
    exact atIndex

/-- Exact source rows derive every base verifier-key core output. The four
dynamic preimages remain tied to verifier-owned columns; neither computed
digest is accepted as independent authority. -/
theorem stage_rows_imply_outputs
    (assignment : Nat -> Nat) (inputs : Inputs)
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (authority : InputColumnsBound assignment inputs)
    (satisfied : StageSatisfied assignment) :
    StageOutputs assignment inputs := by
  have baseSound := baseHash_rows_imply_typedPreimage assignment inputs
    canonical one satisfied.baseConstants satisfied.baseHash
    authority.structureDigest authority.piCcsHeader authority.ajtaiPpDigest
    authority.initialSemanticState
  have baseColumns :
      readColumns assignment baseDigestColumns =
        digestFields (computedBaseDigest inputs) := by
    rw [← base_outputColumns_feed_policy]
    exact readColumns_eq_digestFields assignment
      rawArtifact.baseVerifierKeyHash.recipe.outputColumns
      (computedBaseDigest inputs) baseHash_trace_ownedValid.outputLength
      (fun lane => baseSound lane.val lane.isLt)
  have policySound := policyHash_rows_imply_typedPreimage assignment
    (computedBaseDigest inputs) canonical one satisfied.policyConstants
    satisfied.policyHash baseColumns
  have initialBoundarySound :=
    initialBoundaryHash_rows_imply_typedPreimage assignment
      inputs.structureDigest canonical one satisfied.initialBoundaryConstants
      satisfied.initialBoundaryHash authority.structureDigest
  have policyBinding := digestBinding_rows_sound assignment canonical one
    rawArtifact.policyDigestBinding satisfied.policyDigestBinding
  have initialBoundaryBinding := digestBinding_rows_sound assignment
    canonical one rawArtifact.initialBoundaryBinding
    satisfied.initialBoundaryBinding
  refine {
    baseVerifierKeyDigest := ?_
    policyVerifierKeyDigest := ?_
    vkFsDigest := ?_
    initialBoundaryDigest := ?_
    initialBoundary := ?_
  }
  · intro lane
    exact baseSound lane.val lane.isLt
  · intro lane
    exact policySound lane.val lane.isLt
  · intro lane
    calc
      assignment
          (rawArtifact.policyDigestBinding.leftColumns.getD lane.val 0) =
          assignment
            (rawArtifact.policyDigestBinding.rightColumns.getD lane.val 0) :=
        policyBinding lane
      _ = assignment
          (rawArtifact.policyVerifierKeyHash.recipe.outputColumns.getD
            lane.val 0) := by rw [policyBinding_rightColumns_exact]
      _ = computedPolicyDigest inputs lane := policySound lane.val lane.isLt
  · intro lane
    exact initialBoundarySound lane.val lane.isLt
  · intro lane
    calc
      assignment
          (rawArtifact.initialBoundaryBinding.leftColumns.getD lane.val 0) =
          assignment
            (rawArtifact.initialBoundaryBinding.rightColumns.getD lane.val 0) :=
        initialBoundaryBinding lane
      _ = assignment
          (rawArtifact.initialBoundaryHash.recipe.outputColumns.getD
            lane.val 0) := by rw [initialBoundaryBinding_rightColumns_exact]
      _ = computedInitialBoundary inputs lane :=
        initialBoundarySound lane.val lane.isLt

end Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingLifecycleBaseVerifierKeyRowSound
