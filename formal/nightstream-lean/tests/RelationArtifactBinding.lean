import Nightstream.Assurance.RelationArtifactBinding

/-! Executable checks for exact verifier-key relation artifact binding. -/

set_option autoImplicit false

namespace tests.RelationArtifactBinding

open Nightstream.Assurance.RelationArtifactBinding

def shape : Shape where
  logicalRows := 9148066
  assignmentFields := 14391108
  paddedRows := selectedPaddedRows
  rowVariables := 24
  publicStartField := 0
  publicFields := selectedPublicFields
  semanticMatrixCount := selectedSemanticMatrixCount
  jointMatrixCount := selectedJointMatrixCount
  polynomialDegree := selectedPolynomialDegree

def authoritative : Artifact Nat (List Nat) Nat String where
  format := artifactFormat
  schema := artifactSchema
  matrixPayloadEncoding := "rust-ccs-structure-serde-json-v1"
  source := "neo-fold-clean/r1cs-fprime-fixed-point-v1"
  shape := shape
  structureDigest := 11
  matrixDigest := 12
  ajtaiPublicParametersDigest := 13
  verifierKeyDigest := 14
  matrices := List.range selectedSemanticMatrixCount
  polynomial := [1, 8]

def changedMatrix : Artifact Nat (List Nat) Nat String :=
  { authoritative with matrices := 99 :: authoritative.matrices.tail }

#guard ExactValidation authoritative authoritative
#guard !ExactValidation authoritative changedMatrix

example : changedMatrix ≠ authoritative := by native_decide

example : SelectedProfile authoritative := by
  unfold SelectedProfile SelectedShape
  native_decide

example :
    ExactValidation authoritative authoritative = true := by
  native_decide

example :
    paddedIdentityWidth authoritative.shape = 14391108 := by
  native_decide

end tests.RelationArtifactBinding
