import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.NebulaRecursiveArmSharing

/-!
# Nebula recursive-arm sharing

Assurance tier: artifact-checked for the Rust lifecycle-to-arm mapping and
model-level for duplicate-relation elimination.

This file proves that a logical relation with base, bootstrap-recursive, and
steady-recursive branches has the same accepted language as a physical
relation with base and recursive arms when both recursive branches use the
same predicate. It does not prove any recursive-circuit row semantics.
-/

namespace Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.NebulaRecursiveArmSharing

open Nightstream.Implementation.R1CS.FPrimeFullHistory.NebulaRecursiveArmSharingArtifact

variable {α : Type}

inductive LogicalBranch where
  | base
  | bootstrapRecursive
  | recursive
  deriving DecidableEq

def logicalIndex : LogicalBranch → Nat
  | .base => 0
  | .bootstrapRecursive => 1
  | .recursive => 2

def physicalArm : LogicalBranch → Nat
  | .base => 0
  | .bootstrapRecursive | .recursive => 1

def artifactPhysicalArm (branch : LogicalBranch) : Nat :=
  logicalToPhysical.getD (logicalIndex branch) 2

theorem artifact_mapping_exact : logicalToPhysical = [0, 1, 1] := by
  rfl

theorem artifact_counts_exact :
    logicalArmCount = 3 ∧ physicalArmCount = 2 := by
  constructor <;> rfl

theorem artifactPhysicalArm_eq (branch : LogicalBranch) :
    artifactPhysicalArm branch = physicalArm branch := by
  cases branch <;> rfl

def logicalRelation (base recursive : α → Prop) : LogicalBranch → α → Prop
  | .base => base
  | .bootstrapRecursive | .recursive => recursive

def logicalAccepts (base recursive : α → Prop) (value : α) : Prop :=
  ∃ branch, logicalRelation base recursive branch value

def physicalAccepts (base recursive : α → Prop) (value : α) : Prop :=
  base value ∨ recursive value

theorem logicalAccepts_iff_physicalAccepts
    (base recursive : α → Prop) (value : α) :
    logicalAccepts base recursive value ↔ physicalAccepts base recursive value := by
  constructor
  · rintro ⟨branch, accepted⟩
    cases branch <;> simp_all [logicalRelation, physicalAccepts]
  · intro accepted
    rcases accepted with baseAccepted | recursiveAccepted
    · exact ⟨.base, baseAccepted⟩
    · exact ⟨.recursive, recursiveAccepted⟩

theorem bootstrap_and_steady_use_one_artifact_arm :
    artifactPhysicalArm .bootstrapRecursive = artifactPhysicalArm .recursive := by
  rfl

end Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.NebulaRecursiveArmSharing
