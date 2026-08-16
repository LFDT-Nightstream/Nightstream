import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiRLCAuthority
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyPublic
import Nightstream.Implementation.R1CS.Canonical.GoldilocksField

/-!
Contract: exact Rust artifact boundary for the PiRLC family public suffix.

Assurance tier: Rust-conformant for property
`FPRIME-STREAMING-PIRLC-FAMILY-FULL-XOUT-ROWS-V2`.

Owns both parity shapes, the exact suffix boundary, complete suffix-row
ownership, the 937-field local states, both 32-field full XOut preimages and
four-field outputs, and transport of every canonical-u64 and Poseidon2 leaf
to its Lean soundness theorem.

Does not own interpretation of the state columns as `FamilyState`, the
state-digest transcript, the source-prefix relation, selective lowering, or
recursive lifecycle integration.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublicArtifact

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.Implementation.R1CS.Canonical.CanonicalU64Recipe
open Nightstream.Implementation.R1CS.Canonical.CanonicalU64RecipeSound
open Nightstream.Implementation.R1CS.Canonical.GoldilocksField
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublic.Artifact
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyPublic
open Nightstream.Implementation.R1CS.Program

inductive ArmKind where
  | even
  | odd
deriving DecidableEq, Repr

def armFor : ArmKind → RawArm
  | .even => evenArm
  | .odd => oddArm

theorem artifact_valid : rawArtifact.Valid :=
  rawArtifact_valid

theorem exact_shape :
    rawArtifact.even.sourceRowCount = 275006 /\
      rawArtifact.even.rowCount = 569886 /\
      rawArtifact.even.columnCount = 570115 /\
      rawArtifact.odd.sourceRowCount = 276206 /\
      rawArtifact.odd.rowCount = 571086 /\
      rawArtifact.odd.columnCount = 571315 /\
      rawArtifact.lowNormRows = 282459 /\
      rawArtifact.lowNormColumns = 2521314 /\
      rawArtifact.lowNormPublicColumns = 648 := by
  native_decide

theorem exact_leaf_counts :
    rawArtifact.even.canonicalCalls.length = 11 /\
      rawArtifact.odd.canonicalCalls.length = 11 /\
      rawArtifact.even.poseidon2Calls.length = 490 /\
      rawArtifact.odd.poseidon2Calls.length = 490 /\
      rawArtifact.even.glueRows.length = 121 /\
      rawArtifact.odd.glueRows.length = 121 := by
  native_decide

/-- The shared public prefix is the after full XOut digest, the before full
XOut digest, the before global cursor, and the after global cursor. -/
theorem exact_public_word_layout :
    rawArtifact.even.publicColumnCount = 641 /\
      rawArtifact.odd.publicColumnCount = 641 /\
      rawArtifact.even.publicWordCallIndices =
        [3, 4, 5, 6, 7, 8, 9, 10, 0, 1] /\
      rawArtifact.odd.publicWordCallIndices =
        [3, 4, 5, 6, 7, 8, 9, 10, 0, 1] /\
      rawArtifact.lowNormPublicColumns = 648 := by
  native_decide

/-- Each serialized side has exactly the complete 937-field `FamilyState`.
Its last source column is the family cursor carried by the source rows. -/
theorem exact_state_column_shape (kind : ArmKind) :
    (armFor kind).beforeStateColumns.length = 937 /\
      (armFor kind).afterStateColumns.length = 937 /\
      (armFor kind).beforeStateColumns.getD 936 0 =
        (armFor kind).beforeFamilyCursorColumn /\
      (armFor kind).afterStateColumns.getD 936 0 =
        (armFor kind).afterFamilyCursorColumn := by
  cases kind <;> native_decide

/-- Each side carries the complete 32-field full-state preimage and its
four-field Poseidon2 output. -/
theorem exact_x_out_column_shape (kind : ArmKind) :
    (armFor kind).afterXOutPreimageColumns.length = 32 /\
      (armFor kind).beforeXOutPreimageColumns.length = 32 /\
      (armFor kind).afterXOutDigestColumns.length = 4 /\
      (armFor kind).beforeXOutDigestColumns.length = 4 := by
  cases kind <;> native_decide

/-- The generated owner chain covers every suffix row once, from the exact
source boundary to the physical row count. -/
theorem exact_suffix_owner_chain (kind : ArmKind) :
    exactOwnerChainFrom (armFor kind) (armFor kind).sourceRowCount
      (armFor kind).owners = true := by
  cases kind <;> native_decide

/-- Every canonical-word leaf in a satisfying suffix has the unique
canonical 64-bit value specified by the Lean-owned recipe. -/
theorem canonical_call_refines
    (arm : RawArm) (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : arm.Satisfied assignment)
    (call :
      FPrimeFullHistoryStreamingClaimReplay.Artifact.CanonicalCall)
    (member : call ∈ arm.canonicalCalls) :
    Refines assignment call.layout := by
  apply CanonicalU64RecipeSound.sound goldilocks_euclidPrime canonical one
  exact satisfied.1 call member

/-- Every suffix Poseidon2 leaf agrees with the exact production permutation
under its Rust-emitted column renaming. -/
theorem poseidon2_call_refines
    (arm : RawArm) (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : arm.Satisfied assignment)
    (call : Poseidon2Call.Call) (member : call ∈ arm.poseidon2Calls) :
    AgreeOn
      (Poseidon2PermutationSound.interpret
        (pullAssignment assignment call.columnMap))
      (pullAssignment assignment call.columnMap)
      (knownAfter Poseidon2Permutation.inputColumns
        Poseidon2Permutation.definitions) := by
  apply Poseidon2PermutationSound.poseidon2Permutation_renamed_sound
    call.columnMap call.columnMap_zero canonical one
  simpa [Poseidon2Call.Call.rows] using satisfied.2.1 call member

/-- The compact artifact stores each non-template suffix row exactly. -/
theorem glue_row_holds
    (arm : RawArm) (assignment : Nat → Nat)
    (satisfied : arm.Satisfied assignment)
    (indexed : FPrimeFullHistoryStreamingClaimReplay.Artifact.IndexedRow)
    (member : indexed ∈ arm.glueRows) :
    RowHolds assignment indexed.row := by
  exact satisfied.2.2 indexed member

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublicArtifact
