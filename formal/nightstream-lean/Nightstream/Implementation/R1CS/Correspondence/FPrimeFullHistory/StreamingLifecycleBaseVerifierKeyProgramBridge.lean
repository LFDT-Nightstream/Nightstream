import Mathlib.Tactic
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingLifecycleBaseVerifierKeyHashValidityCertificate
import Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.VerifierKeyProgram

/-!
Contract: exact value-order bridge from the Rust-emitted base verifier-key
core recipes to the typed native preimage programs.

Owns the concrete Nightstream parameter fields, the three constant schedules,
and the complete ordered source-column programs. It does not prove that rows
hold, that source columns are verifier-owned, or that Poseidon2 is secure.

Assurance tier: artifact-checked for
`FPRIME-STREAMING-LIFECYCLE-BASE-VERIFIER-KEY-PROVENANCE-V1` and
model-level for `FPRIME-STREAMING-BASE-VERIFIER-KEY-PROGRAM-V1`,
Nightstream b2/k16.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingLifecycleBaseVerifierKeyProgramBridge

open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingVariableHashRecipe.Artifact
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingLifecycleBaseVerifierKey
open Nightstream.Implementation.Rust.CanonicalConformance.NativeStep
open Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.VerifierKeyProgram

def productionParameters : Parameters where
  q := 18446744069414584321
  eta := 81
  d := 54
  kappa := 18
  m := 1073741824
  b := 2
  kRho := 16
  bigB := 65536
  t := 216
  extensionDegree := 2
  lambda := 114
  publicInputLength := some 648

theorem productionParameters_profile :
    NightstreamGoldilocksProfile productionParameters := by
  norm_num [NightstreamGoldilocksProfile, productionParameters]

def structureDigestColumns : List Nat :=
  List.range' 36007 4

def piCcsHeaderColumns : List Nat :=
  List.range' 646 4

def ajtaiPpDigestColumns : List Nat :=
  List.range' 36011 4

def initialSemanticStateDigestColumns : List Nat :=
  List.range' 36015 4

def baseDigestColumns : List Nat :=
  List.range' 42671 4

def baseTagColumns : List Nat :=
  List.range' 36019 5

def baseParameterColumns : List Nat :=
  List.range' 36024 16

def policyTagColumns : List Nat :=
  List.range' 42679 6

def policyFlagColumns : List Nat :=
  List.range' 42685 3

def initialBoundaryTagColumns : List Nat :=
  List.range' 45703 6

def publicInputLengthColumns : List Nat :=
  List.range' 45709 2

def baseInputColumnProgram : List Nat :=
  baseTagColumns ++ structureDigestColumns ++ piCcsHeaderColumns ++
    ajtaiPpDigestColumns ++ baseParameterColumns ++
    initialSemanticStateDigestColumns

def policyInputColumnProgram : List Nat :=
  policyTagColumns ++ baseDigestColumns ++ policyFlagColumns

def initialBoundaryInputColumnProgram : List Nat :=
  initialBoundaryTagColumns ++ structureDigestColumns ++
    publicInputLengthColumns

def readColumns (assignment : Nat -> RawField)
    (columns : List Nat) : List RawField :=
  columns.map assignment

theorem base_constantFields_exact :
    rawArtifact.baseVerifierKeyHash.recipe.constantValues =
      baseConstantFields productionParameters := by
  norm_num [rawArtifact, baseConstantFields, productionParameters,
    vkFsTag, u64Halves, rawFieldOfNat, optionalUsizeFields,
    goldilocksModulus, twoPow32]

theorem policy_constantFields_exact :
    rawArtifact.policyVerifierKeyHash.recipe.constantValues =
      policyConstantFields := by
  rfl

theorem initialBoundary_constantFields_exact :
    rawArtifact.initialBoundaryHash.recipe.constantValues =
      initialBoundaryConstantFields productionParameters.publicInputLength := by
  norm_num [rawArtifact, initialBoundaryConstantFields,
    productionParameters, initialBoundaryTag, optionalUsizeFields,
    rawFieldOfNat, goldilocksModulus]

theorem base_inputColumns_exact :
    rawArtifact.baseVerifierKeyHash.recipe.inputColumns =
      baseInputColumnProgram := by
  rfl

theorem policy_inputColumns_exact :
    rawArtifact.policyVerifierKeyHash.recipe.inputColumns =
      policyInputColumnProgram := by
  rfl

theorem initialBoundary_inputColumns_exact :
    rawArtifact.initialBoundaryHash.recipe.inputColumns =
      initialBoundaryInputColumnProgram := by
  rfl

/-- The base hash output is the exact four-column slice consumed by the
policy hash. This is a constant-size artifact leaf. -/
theorem base_outputColumns_feed_policy :
    rawArtifact.baseVerifierKeyHash.recipe.outputColumns =
      baseDigestColumns := by
  change [42671, 42672, 42673, 42674] =
    List.range' 42671 4
  simp [List.range'_eq_map_range, List.range_succ]

theorem policyBinding_rightColumns_exact :
    rawArtifact.policyDigestBinding.rightColumns =
      rawArtifact.policyVerifierKeyHash.recipe.outputColumns := by
  change [45695, 45696, 45697, 45698] =
    [45695, 45696, 45697, 45698]
  rfl

theorem initialBoundaryBinding_rightColumns_exact :
    rawArtifact.initialBoundaryBinding.rightColumns =
      rawArtifact.initialBoundaryHash.recipe.outputColumns := by
  change [48117, 48118, 48119, 48120] =
    [48117, 48118, 48119, 48120]
  rfl

theorem basePreimage_from_columnProgram
    (assignment : Nat -> RawField) (inputs : Inputs)
    (tag : readColumns assignment baseTagColumns = vkFsTag)
    (structureValues : readColumns assignment structureDigestColumns =
      digestFields inputs.structureDigest)
    (header : readColumns assignment piCcsHeaderColumns =
      digestFields inputs.piCcsHeader)
    (ajtai : readColumns assignment ajtaiPpDigestColumns =
      digestFields inputs.ajtaiPpDigest)
    (parameters : readColumns assignment baseParameterColumns =
      (baseConstantFields productionParameters).drop 5)
    (initialSemantic :
      readColumns assignment initialSemanticStateDigestColumns =
        digestFields inputs.initialSemanticStateDigest) :
      readColumns assignment baseInputColumnProgram =
      basePreimage productionParameters inputs := by
  simp only [readColumns] at tag structureValues header ajtai parameters
  simp only [readColumns] at initialSemantic
  simp only [baseInputColumnProgram, readColumns, List.map_append]
  rw [tag, structureValues, header, ajtai, parameters, initialSemantic]
  norm_num [basePreimage, baseConstantFields, productionParameters,
    vkFsTag, u64Halves, rawFieldOfNat, optionalUsizeFields,
    goldilocksModulus, twoPow32]

theorem policyPreimage_from_columnProgram
    (assignment : Nat -> RawField) (baseDigest : DigestFields)
    (tag : readColumns assignment policyTagColumns = vkFsPolicyTag)
    (base : readColumns assignment baseDigestColumns =
      digestFields baseDigest)
    (flags : readColumns assignment policyFlagColumns = [1, 1, 1]) :
    readColumns assignment policyInputColumnProgram =
      policyPreimage baseDigest := by
  simp only [readColumns] at tag base flags
  simp only [policyInputColumnProgram, readColumns, List.map_append]
  rw [tag, base, flags]
  rfl

theorem initialBoundaryPreimage_from_columnProgram
    (assignment : Nat -> RawField) (structureDigest : DigestFields)
    (tag : readColumns assignment initialBoundaryTagColumns =
      initialBoundaryTag)
    (structureValues : readColumns assignment structureDigestColumns =
      digestFields structureDigest)
    (publicInputLength :
      readColumns assignment publicInputLengthColumns =
        optionalUsizeFields productionParameters.publicInputLength) :
    readColumns assignment initialBoundaryInputColumnProgram =
      initialBoundaryPreimage structureDigest
        productionParameters.publicInputLength := by
  simp only [readColumns] at tag structureValues publicInputLength
  simp only [initialBoundaryInputColumnProgram, readColumns,
    List.map_append]
  rw [tag, structureValues, publicInputLength]
  rfl

end Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingLifecycleBaseVerifierKeyProgramBridge
