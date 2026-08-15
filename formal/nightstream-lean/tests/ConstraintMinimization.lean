import Nightstream.Assurance.ConstraintMinimization

/-!
Small positive and negative controls for the constraint-classification bridge.

Assurance tier: artifact-checked for the concrete values in this file only.
No Rust-conformant or security-reduced claim follows from these controls.
-/

namespace tests.ConstraintMinimization

open Nightstream.Assurance
open Nightstream.Assurance.ConstraintMinimization
open Nightstream.SuperNeo.CheckPlan

theorem rustTerminalNativeGuardNamesExact :
    Nightstream.Implementation.R1CS.Artifacts.TerminalVerifierNativeGuards.names =
      terminalNativeGuardNames :=
  rust_terminal_native_guard_names_exact

def zeroRow : Numeric.Row :=
  ⟨[(1, 1)], [(0, 1)], []⟩

def bitRow : Numeric.Row :=
  ⟨[(1, 1)], [(0, 18446744069414584320), (1, 1)], []⟩

def retained : IndexedRow :=
  ⟨0, "retained", zeroRow⟩

def duplicate : IndexedRow :=
  ⟨1, "duplicate", zeroRow⟩

def duplicateArtifact : Artifact where
  schema := "nightstream/r1cs-redundancy-problem/v3"
  profile := "lean-duplicate-control"
  scope := "branch"
  diagnosticDigest := "test-only"
  fieldModulus := "18446744069414584321"
  totalRows := 2
  columnCount := 2
  constantOneColumn := 0
  publicInputCount := 1
  completeFamilies := ["duplicate"]
  rows := [retained, duplicate]

def duplicatePlan : List String := ["retained", "duplicate"]

theorem duplicateArtifactWellFormed : duplicateArtifact.WellFormed := by
  native_decide

def wrongModulusArtifact : Artifact :=
  { duplicateArtifact with fieldModulus := "17" }

theorem wrongModulusArtifactRejected :
    Artifact.ExactValidation wrongModulusArtifact
      wrongModulusArtifact = false := by
  native_decide

def wrongScopeArtifact : Artifact :=
  { duplicateArtifact with scope := "row" }

theorem wrongScopeArtifactRejected :
    Artifact.ExactValidation wrongScopeArtifact wrongScopeArtifact = false := by
  native_decide

def duplicateScalar : ScalarCertificate where
  candidate := duplicate
  support := [{ source := retained, coefficient := 1 }]

def duplicateFamily : FamilyCertificate where
  family := "duplicate"
  certificates := [duplicateScalar]

theorem duplicateCertificateValid :
    duplicateFamily.Valid duplicateArtifact duplicatePlan := by
  constructor
  · simp [duplicateArtifact, duplicateFamily]
  · constructor
    · rfl
    · intro scalar scalarMember
      simp only [duplicateFamily, List.mem_singleton] at scalarMember
      subst scalar
      constructor
      · simp [ScalarCertificate.Valid, duplicateScalar, duplicate, retained,
          scalarCombination]
      · intro support supportMember
        simp only [duplicateScalar, List.mem_singleton] at supportMember
        subst support
        simp [retained, duplicateArtifact, duplicatePlan, duplicateFamily]

theorem duplicateCertificateValidAuto :
    duplicateFamily.Valid duplicateArtifact duplicatePlan := by
  simp [FamilyCertificate.Valid, duplicateFamily, duplicateArtifact,
    duplicatePlan, duplicateScalar, duplicate, retained,
    ScalarCertificate.Valid, scalarCombination, candidateRows]

def doubleZeroRow : Numeric.Row :=
  ⟨[(1, 2)], [(0, 1)], []⟩

def doubled : IndexedRow :=
  ⟨1, "double", doubleZeroRow⟩

def doubleArtifact : Artifact :=
  { duplicateArtifact with
    profile := "lean-double-control"
    completeFamilies := ["double"]
    rows := [retained, doubled] }

def doublePlan : List String := ["retained", "double"]

def doubleFamily : FamilyCertificate where
  family := "double"
  certificates := [{
    candidate := doubled
    support := [{ source := retained, coefficient := 2 }]
  }]

theorem doubleCertificateValidAuto :
    doubleFamily.Valid doubleArtifact doublePlan := by
  simp [FamilyCertificate.Valid, doubleFamily, doubleArtifact,
    doublePlan, doubleZeroRow, zeroRow, doubled, retained,
    ScalarCertificate.Valid, scalarCombination, candidateRows,
    Algebraic.residual, Algebraic.linearPolynomial]
  ring

theorem duplicateRedundant :
    Redundant (FamilyHolds duplicateArtifact) duplicatePlan "duplicate" :=
  duplicateFamily.redundant_of_valid duplicateArtifact duplicatePlan
    duplicateCertificateValid

theorem duplicateNormalizedRedundant :
    Redundant (NormalizedFamilyHolds duplicateArtifact)
      duplicatePlan "duplicate" :=
  normalizedRedundant_of_redundant duplicateArtifact duplicatePlan
    "duplicate" duplicateRedundant

def duplicateBinding : SelectiveBinding where
  branch := "base"
  requestedSourceRows := [0, 1]
  closureSourceRows := [0, 1]
  additionalSourceRows := []
  retainedRows := [
    { sourceRow := 0, emittedRow := 4, stageOccurrence := some 0 },
    { sourceRow := 1, emittedRow := 5, stageOccurrence := some 1 }
  ]
  rewrites := []
  emittedRows := [4, 5]
  finalRows := 6
  finalColumns := 8
  finalPublicInputCount := 2
  finalPlanDigest := "test-plan"
  projectedSliceDigest := "test-slice"
  projectedRows := [
    { emittedRow := 4, runIndex := 1, family := "retained",
      arm := some 0, ports := List.replicate 13 ⟨[], [], []⟩ },
    { emittedRow := 5, runIndex := 1, family := "retained",
      arm := some 0, ports := List.replicate 13 ⟨[], [], []⟩ }
  ]

def duplicateBoundArtifact : BoundArtifact :=
  { source := duplicateArtifact, binding := duplicateBinding }

theorem duplicateBoundArtifactCoherent : duplicateBoundArtifact.Coherent := by
  native_decide

theorem duplicateBoundArtifactExact :
    BoundArtifact.ExactValidation duplicateBoundArtifact
      duplicateBoundArtifact = true := by
  native_decide

def validSeededBlock : FinalSeededBlock where
  rowStart := 4
  wordStarts := [2]
  wordWidth := 1
  kappa := 1
  messageCols := 1
  chunkSize := 1
  chunkSeedsByRow := [[List.replicate 32 0]]
  superneoTransformedColumns := false

def emptyFinalPort : FinalPort := ⟨[], [], []⟩

def validSeededPort : FinalPort where
  explicit := []
  geometricRuns := []
  seededBlocks := [validSeededBlock]

def seededBinding : SelectiveBinding :=
  { duplicateBinding with
    finalRows := 58
    projectedRows := [
      { emittedRow := 4, runIndex := 1, family := "retained",
        arm := some 0,
        ports := validSeededPort :: List.replicate 12 emptyFinalPort },
      { emittedRow := 5, runIndex := 1, family := "retained",
        arm := some 0, ports := List.replicate 13 emptyFinalPort }
    ] }

def seededBoundArtifact : BoundArtifact :=
  { source := duplicateArtifact, binding := seededBinding }

theorem seededBoundArtifactCoherent : seededBoundArtifact.Coherent := by
  native_decide

def malformedSeededBlock : FinalSeededBlock :=
  { validSeededBlock with chunkSeedsByRow := [[[0]]] }

def malformedSeededBinding : SelectiveBinding :=
  { seededBinding with
    projectedRows := [
      { emittedRow := 4, runIndex := 1, family := "retained",
        arm := some 0,
        ports := { validSeededPort with
          seededBlocks := [malformedSeededBlock] } ::
            List.replicate 12 emptyFinalPort },
      { emittedRow := 5, runIndex := 1, family := "retained",
        arm := some 0, ports := List.replicate 13 emptyFinalPort }
    ] }

def malformedSeededBoundArtifact : BoundArtifact :=
  { source := duplicateArtifact, binding := malformedSeededBinding }

theorem malformedSeededBlockRejected :
    ¬ malformedSeededBoundArtifact.Coherent := by
  native_decide

def wrongRowSeededBlock : FinalSeededBlock :=
  { validSeededBlock with rowStart := 5 }

def wrongRowSeededBinding : SelectiveBinding :=
  { seededBinding with
    projectedRows := [
      { emittedRow := 4, runIndex := 1, family := "retained",
        arm := some 0,
        ports := { validSeededPort with
          seededBlocks := [wrongRowSeededBlock] } ::
            List.replicate 12 emptyFinalPort },
      { emittedRow := 5, runIndex := 1, family := "retained",
        arm := some 0, ports := List.replicate 13 emptyFinalPort }
    ] }

def wrongRowSeededBoundArtifact : BoundArtifact :=
  { source := duplicateArtifact, binding := wrongRowSeededBinding }

theorem wrongRowSeededBlockRejected :
    ¬ wrongRowSeededBoundArtifact.Coherent := by
  native_decide

theorem duplicateBoundRedundant :
    Redundant (FamilyHolds duplicateBoundArtifact.source)
      duplicatePlan "duplicate" :=
  duplicateFamily.redundant_of_bound_valid duplicateBoundArtifact
    duplicateBoundArtifact duplicatePlan duplicateBoundArtifactExact
    duplicateCertificateValid

def completeDuplicateArtifact : Artifact :=
  { duplicateArtifact with
    completeFamilies := ["duplicate", "retained"] }

def completeDuplicateBoundArtifact : BoundArtifact :=
  { source := completeDuplicateArtifact, binding := duplicateBinding }

theorem completeDuplicateBoundArtifactCoversFullRelation :
    completeDuplicateBoundArtifact.CoversFullRelation := by
  native_decide

theorem completeDuplicateNormalizedSound :
    Sound (NormalizedFamilyHolds completeDuplicateArtifact)
      (NormalizedTarget completeDuplicateArtifact)
      completeDuplicateArtifact.completeFamilies :=
  completeDuplicateArtifact.normalizedFullPlanSound
    completeDuplicateBoundArtifactCoversFullRelation.2

theorem completeDuplicateNormalizedComplete :
    Complete (NormalizedFamilyHolds completeDuplicateArtifact)
      (NormalizedTarget completeDuplicateArtifact)
      completeDuplicateArtifact.completeFamilies :=
  completeDuplicateArtifact.normalizedFullPlanComplete

theorem completeDuplicateNormalizedExact :
    Exact (NormalizedFamilyHolds completeDuplicateArtifact)
      (NormalizedTarget completeDuplicateArtifact)
      completeDuplicateArtifact.completeFamilies :=
  completeDuplicateArtifact.normalizedFullPlanExact
    completeDuplicateBoundArtifactCoversFullRelation.2

theorem completeDuplicateBoundRedundant :
    Redundant (FamilyHolds completeDuplicateBoundArtifact.source)
      duplicatePlan "duplicate" :=
  duplicateFamily.redundant_of_full_bound_valid
    completeDuplicateBoundArtifact completeDuplicateBoundArtifact
    duplicatePlan completeDuplicateBoundArtifactCoversFullRelation
    (by native_decide) (by
      simp [FamilyCertificate.Valid, duplicateFamily,
        completeDuplicateBoundArtifact, completeDuplicateArtifact,
        duplicateArtifact, duplicatePlan, duplicateScalar, duplicate,
        retained, zeroRow, ScalarCertificate.Valid, scalarCombination,
        candidateRows])

def duplicateTerminalBinding : TerminalBinding where
  requestedSourceRows := [0, 1]
  verifierNativeGuards := terminalNativeGuardNames
  columnLayout := {
    sourcePublicColumns := 1
    sourcePrivateColumns := 1
    spartanPrivateColumns := 2
  }
  projectedRows := [
    { sourceRow := 0, spartanRow := 0,
      row := ⟨[(0, 1)], [(2, 1)], []⟩ },
    { sourceRow := 1, spartanRow := 1,
      row := ⟨[(0, 1)], [(2, 1)], []⟩ }
  ]
  spartanRows := 2
  spartanColumns := 3
  spartanPaddingRows := { start := 2, stop := 2 }
  spartanPrivatePaddingColumns := 1
  diagnosticDigest := "test-terminal"

def duplicateTerminalBoundArtifact : TerminalBoundArtifact :=
  { source := duplicateArtifact, binding := duplicateTerminalBinding }

theorem duplicateTerminalBoundArtifactCoherent :
    duplicateTerminalBoundArtifact.Coherent := by
  native_decide

theorem duplicateTerminalBoundArtifactExact :
    TerminalBoundArtifact.ExactValidation duplicateTerminalBoundArtifact
      duplicateTerminalBoundArtifact = true := by
  native_decide

theorem duplicateTerminalNativeGuardRetained :
    "terminal.context.induction" ∈
      duplicateTerminalBoundArtifact.binding.verifierNativeGuards :=
  TerminalBoundArtifact.accepted_retains_native_guard
    duplicateTerminalBoundArtifactExact (by decide)

theorem duplicateTerminalNativeGuardNotPolynomial :
    "terminal.context.induction" ∉
      duplicateTerminalBoundArtifact.source.completeFamilies :=
  TerminalBoundArtifact.accepted_native_guard_not_polynomial
    duplicateTerminalBoundArtifactExact (by decide)

theorem duplicateTerminalProofGuardsRetained :
    ∀ guard ∈ TerminalProofBoundary.guardNames,
      guard ∈ duplicateTerminalBoundArtifact.binding.verifierNativeGuards :=
  TerminalBoundArtifact.accepted_retains_proof_guards
    duplicateTerminalBoundArtifactExact

theorem duplicateTerminalContextGuardsRetained :
    ∀ guard ∈ TerminalContextBoundary.guardNames,
      guard ∈ duplicateTerminalBoundArtifact.binding.verifierNativeGuards :=
  TerminalBoundArtifact.accepted_retains_context_guards
    duplicateTerminalBoundArtifactExact

theorem duplicateTerminalStatementGuardsRetained :
    ∀ guard ∈ TerminalStatementBoundary.guardNames,
      guard ∈ duplicateTerminalBoundArtifact.binding.verifierNativeGuards :=
  TerminalBoundArtifact.accepted_retains_statement_guards
    duplicateTerminalBoundArtifactExact

theorem duplicateTerminalBoundRedundant :
    Redundant (FamilyHolds duplicateTerminalBoundArtifact.source)
      duplicatePlan "duplicate" :=
  duplicateFamily.redundant_of_terminal_bound_valid
    duplicateTerminalBoundArtifact duplicateTerminalBoundArtifact
    duplicatePlan duplicateTerminalBoundArtifactExact
    duplicateCertificateValid

def completeDuplicateTerminalBoundArtifact : TerminalBoundArtifact :=
  { source := completeDuplicateArtifact,
    binding := duplicateTerminalBinding }

theorem completeDuplicateTerminalBoundArtifactCoversFullRelation :
    completeDuplicateTerminalBoundArtifact.CoversFullRelation := by
  native_decide

theorem completeDuplicateTerminalBoundRedundant :
    Redundant (FamilyHolds completeDuplicateTerminalBoundArtifact.source)
      duplicatePlan "duplicate" :=
  duplicateFamily.redundant_of_full_terminal_bound_valid
    completeDuplicateTerminalBoundArtifact
    completeDuplicateTerminalBoundArtifact duplicatePlan
    completeDuplicateTerminalBoundArtifactCoversFullRelation
    (by native_decide) (by
      simp [FamilyCertificate.Valid, duplicateFamily,
        completeDuplicateTerminalBoundArtifact,
        completeDuplicateArtifact, duplicateArtifact, duplicatePlan,
        duplicateScalar, duplicate, retained, zeroRow,
        ScalarCertificate.Valid, scalarCombination, candidateRows])

def bitness : IndexedRow :=
  ⟨0, "bitness", bitRow⟩

def requiredZero : IndexedRow :=
  ⟨1, "zero", zeroRow⟩

def necessaryArtifact : Artifact where
  schema := "nightstream/r1cs-redundancy-problem/v3"
  profile := "lean-necessary-control"
  scope := "branch"
  diagnosticDigest := "test-only"
  fieldModulus := "18446744069414584321"
  totalRows := 2
  columnCount := 2
  constantOneColumn := 0
  publicInputCount := 1
  completeFamilies := ["bitness", "zero"]
  rows := [bitness, requiredZero]

def necessaryPlan : List String := ["bitness", "zero"]

def zeroCounterexample : RemovalCounterexample where
  removedFamily := "zero"
  values := [1, 1]

theorem zeroCounterexampleValid :
    zeroCounterexample.Valid necessaryArtifact necessaryPlan := by
  decide

theorem zeroNecessary :
    NecessaryForSoundness (FamilyHolds necessaryArtifact)
      (Target necessaryArtifact) necessaryPlan "zero" :=
  zeroCounterexample.necessary_of_valid necessaryArtifact necessaryPlan
    zeroCounterexampleValid

theorem zeroNormalizedNecessary :
    NecessaryForSoundness (NormalizedFamilyHolds necessaryArtifact)
      (NormalizedTarget necessaryArtifact) necessaryPlan "zero" :=
  zeroCounterexample.necessary_normalized_of_valid
    necessaryArtifact necessaryPlan zeroCounterexampleValid

end tests.ConstraintMinimization
