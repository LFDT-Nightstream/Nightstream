import Nightstream.Implementation.R1CS.Canonical.GoldilocksField
import Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalU64Honest
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingClaimReplayArtifact

/-!
Contract: redundancy proof for the private canonical-u64 decompositions
removed from the streaming claim-replay relation.

The deleted rows only allocated bits, a high-word flag, and an inverse for a
private field word. No retained row read those auxiliary columns. This module
proves the required local elimination rule: for any canonical source field
before a fresh allocation, the reduced assignment has an honest extension
that satisfies all 69 deleted rows and preserves every earlier column.

It does not remove the ten retained decompositions. Those words are public
program cursors and public state-digest lanes.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayReduction

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.Implementation.R1CS.Canonical.CanonicalU64Recipe
open Nightstream.Implementation.R1CS.Canonical.CanonicalU64RecipeHonest
open Nightstream.Implementation.R1CS.Canonical.GoldilocksField
open Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalU64Honest
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplay.Artifact
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingClaimReplay

/-- One deleted private decomposition, placed after its retained source. -/
def privateLayout (sourceColumn base : Nat) : Layout where
  base := base
  input := [(sourceColumn, 1)]

/-- Honest values for exactly the 66 deleted auxiliary columns. -/
def privateWitness
    (initial : Nat → Nat) (sourceColumn base : Nat) : Nat → Nat :=
  witness goldilocksFieldInverse (sourceOf (initial sourceColumn))
    (privateLayout sourceColumn base) initial

@[simp] theorem privateWitness_before
    (initial : Nat → Nat) (sourceColumn base column : Nat)
    (before : column < base) :
    privateWitness initial sourceColumn base column = initial column := by
  exact witness_before goldilocksFieldInverse
    (sourceOf (initial sourceColumn)) (privateLayout sourceColumn base)
    initial before

theorem privateWitness_canonical
    (initial : Nat → Nat) (sourceColumn base : Nat)
    (canonical : ∀ column, initial column < goldilocksP) :
    ∀ column, privateWitness initial sourceColumn base column < goldilocksP := by
  exact witness_canonical goldilocksFieldInverse
    (sourceOf (initial sourceColumn)) (privateLayout sourceColumn base)
    initial canonical

/-- A canonical private source always has a satisfying extension for all 69
deleted decomposition rows. -/
theorem privateWitness_satisfies
    (initial : Nat → Nat) (sourceColumn base : Nat)
    (positive : 0 < base) (sourceBefore : sourceColumn < base)
    (canonical : ∀ column, initial column < goldilocksP)
    (one : initial 0 = 1) :
    Satisfies (rows (privateLayout sourceColumn base))
      (privateWitness initial sourceColumn base) := by
  apply complete goldilocksFieldInverse (sourceOf (initial sourceColumn))
    (privateLayout sourceColumn base) initial positive one
  · intro column coefficient member
    simp only [privateLayout, List.mem_singleton, Prod.mk.injEq] at member
    exact member.1 ▸ sourceBefore
  · rw [sourceWord_sourceOf (initial sourceColumn) (canonical sourceColumn)]
    simp only [privateLayout, lcEval, List.foldl_cons, List.foldl_nil,
      Nat.zero_add, Nat.one_mul]
    exact Nat.mod_eq_of_lt (canonical sourceColumn)
  · rw [sourceWord_sourceOf (initial sourceColumn) (canonical sourceColumn)]
    exact canonical sourceColumn

/-- A relation reads no column in or after `base`. -/
def DependsOnlyBelow
    (relation : (Nat → Nat) → Prop) (base : Nat) : Prop :=
  ∀ left right,
    (∀ column, column < base → left column = right column) →
      (relation left ↔ relation right)

/-- Adding one private canonical-u64 recipe does not restrict any relation
that reads only earlier columns. The recipe is a pure witness extension. -/
theorem private_decomposition_redundant
    (relation : (Nat → Nat) → Prop)
    (initial : Nat → Nat) (sourceColumn base : Nat)
    (positive : 0 < base) (sourceBefore : sourceColumn < base)
    (canonical : ∀ column, initial column < goldilocksP)
    (one : initial 0 = 1)
    (depends : DependsOnlyBelow relation base) :
    relation initial ↔
      ∃ extended,
        relation extended ∧
          Satisfies (rows (privateLayout sourceColumn base)) extended ∧
          (∀ column, column < base → extended column = initial column) ∧
          ∀ column, extended column < goldilocksP := by
  constructor
  · intro reduced
    refine ⟨privateWitness initial sourceColumn base, ?_, ?_, ?_, ?_⟩
    · exact (depends initial _ fun column before =>
        (privateWitness_before initial sourceColumn base column before).symm).mp
          reduced
    · exact privateWitness_satisfies initial sourceColumn base positive
        sourceBefore canonical one
    · exact privateWitness_before initial sourceColumn base
    · exact privateWitness_canonical initial sourceColumn base canonical
  · rintro ⟨extended, reduced, _satisfies, agrees, _canonical⟩
    exact (depends extended initial agrees).mp reduced

def removedPrivateWords : Nat := 38
def removedRows : Nat := removedPrivateWords * 69
def removedAuxiliaryColumns : Nat := removedPrivateWords * 66
def priorReplayTransitionWords : Nat := 40

/-- Exact reduction census. The current state has two 128-field sides. Of
the prior 40 replay-field decompositions, only the two program cursors
remain. Eight digest-lane decompositions also remain because they define the
public digest bits. The 108 coordinate fields on each side are field values,
so they do not need canonical-u64 decompositions. -/
theorem exact_reduction_census :
    rawArtifact.transitionStateWords = 256 ∧
      rawArtifact.transitionStateWords = 2 * (20 + 108) ∧
      priorReplayTransitionWords = 40 ∧
      removedPrivateWords = 38 ∧
      priorReplayTransitionWords - removedPrivateWords = 2 ∧
      rawArtifact.stateDigestWords = 8 ∧
      rawArtifact.full.canonicalCalls.length = 10 ∧
      rawArtifact.finalChunk.canonicalCalls.length = 10 ∧
      removedRows = 2622 ∧
      removedAuxiliaryColumns = 2508 := by
  native_decide

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayReduction
