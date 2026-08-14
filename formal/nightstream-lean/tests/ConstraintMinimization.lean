import Nightstream.Assurance.ConstraintMinimization

/-!
Small positive and negative controls for the constraint-classification bridge.

Assurance tier: artifact-checked for the concrete values in this file only.
No Rust-conformant or security-reduced claim follows from these controls.
-/

namespace tests.ConstraintMinimization

open Nightstream.Assurance.ConstraintMinimization
open Nightstream.SuperNeo.CheckPlan

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
  diagnosticDigest := "test-only"
  totalRows := 2
  columnCount := 2
  constantOneColumn := 0
  publicInputCount := 1
  rows := [retained, duplicate]

def duplicatePlan : List String := ["retained", "duplicate"]

def duplicateScalar : ScalarCertificate where
  candidate := duplicate
  support := [{ source := retained, coefficient := 1 }]

def duplicateFamily : FamilyCertificate where
  family := "duplicate"
  certificates := [duplicateScalar]

theorem duplicateCertificateValid :
    duplicateFamily.Valid duplicateArtifact duplicatePlan := by
  constructor
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

theorem duplicateRedundant :
    Redundant (FamilyHolds duplicateArtifact) duplicatePlan "duplicate" :=
  duplicateFamily.redundant_of_valid duplicateArtifact duplicatePlan
    duplicateCertificateValid

def bitness : IndexedRow :=
  ⟨0, "bitness", bitRow⟩

def requiredZero : IndexedRow :=
  ⟨1, "zero", zeroRow⟩

def necessaryArtifact : Artifact where
  schema := "nightstream/r1cs-redundancy-problem/v3"
  profile := "lean-necessary-control"
  diagnosticDigest := "test-only"
  totalRows := 2
  columnCount := 2
  constantOneColumn := 0
  publicInputCount := 1
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

end tests.ConstraintMinimization
