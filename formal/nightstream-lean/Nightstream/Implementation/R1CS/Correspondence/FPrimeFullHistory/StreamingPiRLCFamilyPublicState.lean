import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPiRLCFamilyState
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPiRLCFamilyXOutArtifact

/-!
Contract: exact local-state and full-XOut public binding for one PiRLC family
arm.

Owns the two offset rows that derive global program cursors from family
cursors, their canonical public words, local semantic-digest placement in the
full XOut preimages, and exact execution of both full-XOut hashes into their
eight public digest words.

Does not own family arithmetic, overlay links, collision resistance, selective
lowering, or recursive lifecycle integration.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 524288
set_option maxHeartbeats 8000000

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublicState

open Nightstream.Implementation.Nebula.ProductionStreamingPiRlcAuthority
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyDigest
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublic.Artifact
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublicArtifact
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyState
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyXOutArtifact
open Nightstream.Implementation.R1CS.Program

def firstFamilyProgramCursor : Nat := 199

def programCursorColumn (kind : ArmKind) (side : StateSide) : Nat :=
  (publicWordCall kind (cursorPublicWordIndex side)).fieldColumn

def cursorOffsetTerms (side : StateSide) : List (Nat × Nat) :=
  [(cursorColumn side, 1), (0, firstFamilyProgramCursor)]

/-- Normalized form of Rust's exact `derived = family + 199` row. -/
def cursorOffsetRow (kind : ArmKind) (side : StateSide) : Row :=
  ⟨[(0, goldilocksP - firstFamilyProgramCursor),
      (cursorColumn side, goldilocksP - 1),
      (programCursorColumn kind side, 1)],
    [(0, 1)], []⟩

def cursorOffsetRows (kind : ArmKind) : List Row :=
  [cursorOffsetRow kind .before, cursorOffsetRow kind .after]

private theorem cursor_offset_rows_in_glue (kind : ArmKind) :
    rowsIncluded (cursorOffsetRows kind) (glueProgram kind) = true := by
  cases kind <;> native_decide

private theorem cursor_offset_terms_canonical (side : StateSide) :
    CanonicalTerms (cursorOffsetTerms side) := by
  cases side <;> native_decide

private theorem rowHolds_of_operand_perms
    (assignment : Nat → Nat) {source target : Row}
    (a : source.a.Perm target.a)
    (b : source.b.Perm target.b)
    (c : source.c.Perm target.c)
    (holds : RowHolds assignment source) :
    RowHolds assignment target := by
  unfold RowHolds at holds ⊢
  calc
    lcEval assignment target.a * lcEval assignment target.b % goldilocksP =
        lcEval assignment source.a * lcEval assignment source.b %
          goldilocksP := by
      rw [Program.lcEval_eq_of_perm assignment a,
        Program.lcEval_eq_of_perm assignment b]
    _ = lcEval assignment source.c := holds
    _ = lcEval assignment target.c :=
      Program.lcEval_eq_of_perm assignment c

private theorem cursor_offset_row_perms
    (kind : ArmKind) (side : StateSide) :
    (cursorOffsetRow kind side).a.Perm
        (builderLinearRow (programCursorColumn kind side)
          (cursorOffsetTerms side)).a ∧
      (cursorOffsetRow kind side).b.Perm
        (builderLinearRow (programCursorColumn kind side)
          (cursorOffsetTerms side)).b ∧
      (cursorOffsetRow kind side).c.Perm
        (builderLinearRow (programCursorColumn kind side)
          (cursorOffsetTerms side)).c := by
  cases kind <;> cases side <;> native_decide

private theorem glue_satisfies
    (kind : ArmKind) (assignment : Nat → Nat)
    (satisfied : (armFor kind).Satisfied assignment) :
    Satisfies (glueProgram kind) assignment := by
  intro row member
  rcases List.mem_map.mp member with ⟨indexed, indexedMember, rfl⟩
  exact glue_row_holds (armFor kind) assignment satisfied indexed indexedMember

/-- The exact suffix row derives the selected global cursor modulo the
Goldilocks modulus. -/
theorem program_cursor_field_mod
    (kind : ArmKind) (side : StateSide)
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : (armFor kind).Satisfied assignment) :
    assignment (programCursorColumn kind side) =
      (assignment (cursorColumn side) + firstFamilyProgramCursor) %
        goldilocksP := by
  have compactSatisfies : Satisfies (cursorOffsetRows kind) assignment := by
    intro row member
    exact glue_satisfies kind assignment satisfied row
      (rowsIncluded_sound (cursor_offset_rows_in_glue kind) row member)
  have rowHolds := compactSatisfies (cursorOffsetRow kind side) (by
    cases side <;> simp [cursorOffsetRows])
  have builderHolds := rowHolds_of_operand_perms assignment
    (cursor_offset_row_perms kind side).1
    (cursor_offset_row_perms kind side).2.1
    (cursor_offset_row_perms kind side).2.2 rowHolds
  have exact := builderLinearRow_sound canonical one
    (programCursorColumn kind side) (cursorOffsetTerms side)
    (cursor_offset_terms_canonical side) builderHolds
  simpa [cursorOffsetTerms, lcEval, one, Nat.mul_comm] using exact

/-- Public word 8 or 9 is the global cursor derived from the family cursor
inside the same decoded state. -/
theorem cursor_public_word_mod
    (kind : ArmKind) (side : StateSide)
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : (armFor kind).Satisfied assignment) :
    publicWordValue assignment kind (cursorPublicWordIndex side) =
      ((familyStateAt assignment canonical kind side).familyCursor +
        firstFamilyProgramCursor) % goldilocksP := by
  rw [public_word_refines kind (cursorPublicWordIndex side) assignment
    canonical one satisfied]
  change assignment (programCursorColumn kind side) = _
  rw [program_cursor_field_mod kind side assignment canonical one satisfied]
  rfl

/-- The production family range excludes modular wrap, so the public word is
the natural global program cursor. -/
theorem cursor_public_word_exact
    (kind : ArmKind) (side : StateSide)
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : (armFor kind).Satisfied assignment)
    (cursorBound :
      (familyStateAt assignment canonical kind side).familyCursor < 111) :
    publicWordValue assignment kind (cursorPublicWordIndex side) =
      (familyStateAt assignment canonical kind side).familyCursor +
        firstFamilyProgramCursor := by
  rw [cursor_public_word_mod kind side assignment canonical one satisfied,
    Nat.mod_eq_of_lt]
  unfold firstFamilyProgramCursor goldilocksP
  omega

/-- Complete public meaning of one accepted PiRLC family body. -/
structure Binding
    (kind : ArmKind)
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP) : Prop where
  afterPreimage :
    (List.range 937).map (fun index =>
        assignment (stateWordColumnFor kind .after index)) =
      familyStateFields
        (familyStateAt assignment canonical kind .after)
  beforePreimage :
    (List.range 937).map (fun index =>
        assignment (stateWordColumnFor kind .before index)) =
      familyStateFields
        (familyStateAt assignment canonical kind .before)
  afterSemanticDigest : ∀ lane : Fin 4,
    (stateDigest assignment canonical kind .after lane).val =
      assignment (xOutPreimageColumn kind .after (19 + lane.val))
  beforeSemanticDigest : ∀ lane : Fin 4,
    (stateDigest assignment canonical kind .before lane).val =
      assignment (xOutPreimageColumn kind .before (19 + lane.val))
  afterXOutDigest : ∀ lane : Fin 4,
    assignment (xOutDigestColumn kind .after lane) =
      publicWordValue assignment kind
        (xOutPublicWordIndex .after lane)
  beforeXOutDigest : ∀ lane : Fin 4,
    assignment (xOutDigestColumn kind .before lane) =
      publicWordValue assignment kind
        (xOutPublicWordIndex .before lane)
  afterXOutHash : ∀ lane : Fin 4,
    assignment (xOutDigestColumn kind .after lane) =
      Poseidon2Sponge.digest Poseidon2CanonicalConstants.selected
        (xOutChunks assignment kind .after) lane
  beforeXOutHash : ∀ lane : Fin 4,
    assignment (xOutDigestColumn kind .before lane) =
      Poseidon2Sponge.digest Poseidon2CanonicalConstants.selected
        (xOutChunks assignment kind .before) lane
  beforeCursor :
    publicWordValue assignment kind (cursorPublicWordIndex .before) =
      (familyStateAt assignment canonical kind .before).familyCursor +
        firstFamilyProgramCursor
  afterCursor :
    publicWordValue assignment kind (cursorPublicWordIndex .after) =
      (familyStateAt assignment canonical kind .after).familyCursor +
        firstFamilyProgramCursor

/-- The local state digests occupy their exact full-XOut preimage slots, the
eight public words are the physical full-XOut outputs, and the last two words
are the non-wrapping global cursors. -/
theorem shared_public_state_refines
    (kind : ArmKind)
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : (armFor kind).Satisfied assignment)
    (beforeBound :
      (familyStateAt assignment canonical kind .before).familyCursor < 111)
    (afterBound :
      (familyStateAt assignment canonical kind .after).familyCursor < 111) :
    Binding kind assignment canonical where
  afterPreimage :=
    digest_preimage_is_family_state kind .after assignment canonical one
      satisfied
  beforePreimage :=
    digest_preimage_is_family_state kind .before assignment canonical one
      satisfied
  afterSemanticDigest := fun lane =>
    state_digest_x_out_preimage kind .after lane assignment canonical one
      satisfied
  beforeSemanticDigest := fun lane =>
    state_digest_x_out_preimage kind .before lane assignment canonical one
      satisfied
  afterXOutDigest :=
    (shared_x_out_public_words_refine kind assignment canonical one
      satisfied).1
  beforeXOutDigest :=
    (shared_x_out_public_words_refine kind assignment canonical one
      satisfied).2
  afterXOutHash := fun lane =>
    x_out_hash_refines kind .after assignment canonical one satisfied lane
  beforeXOutHash := fun lane =>
    x_out_hash_refines kind .before assignment canonical one satisfied lane
  beforeCursor :=
    cursor_public_word_exact kind .before assignment canonical one satisfied
      beforeBound
  afterCursor :=
    cursor_public_word_exact kind .after assignment canonical one satisfied
      afterBound

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublicState
