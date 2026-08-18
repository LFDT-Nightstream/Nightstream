import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPiRLCFamilyPhysicalState
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Transcript.Operations

/-!
Contract: local semantic-digest continuity between two accepted physical
PiRLC family arms.

Owns the independent Poseidon2 digest of the exact 1,045-field `FamilyState`
encoding, the named collision event for two distinct canonical encodings, and
the reduction from an explicit adjacent local semantic-digest link to equal
semantic states or that collision event. It also recovers exact cursor
continuity from the two public cursor words without a cryptographic
assumption.

Does not own public full-XOut continuity, start or finish circuits, the
110-arm sequence, collision resistance, selective lowering, or recursive
lifecycle integration.

Emits constraints: no.

Assurance tier: security-reduced prototype. Accepted generated rows plus an
explicit private semantic-digest link imply exact state continuity unless the
exact Poseidon2 state-digest function has a collision.
-/

set_option autoImplicit false
set_option maxRecDepth 524288
set_option maxHeartbeats 8000000

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyContinuity

open Nightstream.Implementation.Nebula.ProductionStreamingPiRlcAuthority
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyDigest
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyDigestDomain
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPhysicalState
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublic.Artifact
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublicArtifact
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublicState
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyState
open Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript
open Nightstream.Implementation.R1CS.PiRlcChallenge.TranscriptMachine

/-- Exact field framing used before the 1,045 external state fields. -/
def familyStateFrame : List Nat := [2, 5, 435744240755, 1045]

private def absorbFields (initial : State) (values : List Field) : State :=
  values.foldl absorbElem initial

/-- Independent Poseidon2 input state for one complete semantic family state.
The framing words use transcript word semantics. The state words use canonical
field semantics. -/
def familyStateDigestInput (state : FamilyState) : State :=
  absorbFields
    (absorbFields domainInitialState (familyStateFrame.map wordField))
    ((familyStateFields state).map fieldValue)

/-- Independent four-lane Poseidon2 digest of one exact family-state frame. -/
def familyStateDigest (state : FamilyState) : Fin 4 -> Field :=
  (Nightstream.Implementation.R1CS.PiRlcChallenge.TranscriptMachine.digest
    (familyStateDigestInput state)).2

/-- Witness for a collision in the exact family-state Poseidon2 application
domain. Both preimages are canonical, and their 1,045-field serializations are
different. -/
structure Poseidon2FamilyStateCollisionWitness where
  left : FamilyState
  right : FamilyState
  leftCanonical : ∀ value, value ∈ familyStateFields left -> value < goldilocksP
  rightCanonical : ∀ value, value ∈ familyStateFields right -> value < goldilocksP
  preimagesDifferent : familyStateFields left ≠ familyStateFields right
  digestEqual : familyStateDigest left = familyStateDigest right

/-- Named security-reduction event for the exact Poseidon2 state digest. -/
def Poseidon2FamilyStateCollision : Prop :=
  Nonempty Poseidon2FamilyStateCollisionWitness

/-- Equal local family digests recover the exact 1,045-field family state, or
exhibit a collision in the exact framed Poseidon2 application domain. -/
theorem familyState_eq_or_poseidon2_collision
    (left right : FamilyState)
    (leftCanonical :
      ∀ value, value ∈ familyStateFields left -> value < goldilocksP)
    (rightCanonical :
      ∀ value, value ∈ familyStateFields right -> value < goldilocksP)
    (digestEqual : familyStateDigest left = familyStateDigest right) :
    left = right ∨ Poseidon2FamilyStateCollision := by
  by_cases stateEqual : left = right
  · exact Or.inl stateEqual
  · right
    refine ⟨{
      left := left
      right := right
      leftCanonical := leftCanonical
      rightCanonical := rightCanonical
      preimagesDifferent := ?_
      digestEqual := digestEqual }⟩
    intro preimagesEqual
    exact stateEqual (familyStateFields_injective preimagesEqual)

private theorem fieldAt_eq_fieldValue
    (assignment : Nat -> Nat)
    (canonical : forall column, assignment column < goldilocksP)
    (column : Nat) :
    CallRefinement.fieldAt assignment canonical column =
      fieldValue (assignment column) := by
  apply Fin.ext
  simp [fieldValue, Nat.mod_eq_of_lt (canonical column)]

private theorem semanticExecute_pinned
    (assignment : Nat -> Nat)
    (canonical : ColumnReplay.CanonicalAssignment assignment)
    (run : ColumnReplay.SemanticRun) (values : List Nat) :
    ColumnReplay.semanticExecute assignment canonical run
        (values.map ColumnReplay.Operation.pinned) =
      { run with
        state := absorbFields run.state (values.map wordField) } := by
  induction values generalizing run with
  | nil => rfl
  | cons value values inductionHypothesis =>
      change ColumnReplay.semanticExecute assignment canonical
          { run with state := absorbElem run.state (wordField value) }
          (values.map ColumnReplay.Operation.pinned) = _
      rw [inductionHypothesis]
      rfl

private theorem semanticExecute_external
    (assignment : Nat -> Nat)
    (canonical : ColumnReplay.CanonicalAssignment assignment)
    (run : ColumnReplay.SemanticRun) (columns : List Nat) :
    ColumnReplay.semanticExecute assignment canonical run
        (columns.map ColumnReplay.Operation.external) =
      { run with
        state := absorbFields run.state
          ((columns.map assignment).map fieldValue) } := by
  induction columns generalizing run with
  | nil => rfl
  | cons column columns inductionHypothesis =>
      change ColumnReplay.semanticExecute assignment canonical
          { run with
            state := absorbElem run.state
              (CallRefinement.fieldAt assignment canonical column) }
          (columns.map ColumnReplay.Operation.external) = _
      rw [fieldAt_eq_fieldValue, inductionHypothesis]
      rfl

private theorem digest_operations_eq
    (kind : ArmKind) (side : StateSide) :
    digestOperations kind side =
      familyStateFrame.map ColumnReplay.Operation.pinned ++
        ((List.range 1045).map fun index =>
          ColumnReplay.Operation.external
            (stateWordColumnFor kind side index)) ++
          [ColumnReplay.Operation.digest] := by
  rfl

private theorem semanticExecute_first_digest
    (assignment : Nat -> Nat)
    (canonical : ColumnReplay.CanonicalAssignment assignment)
    (state : State) :
    (ColumnReplay.semanticExecute assignment canonical
        { state := state, digests := [] }
        [ColumnReplay.Operation.digest]).digests.getD 0 zeroDigest =
      (Nightstream.Implementation.R1CS.PiRlcChallenge.TranscriptMachine.digest
        state).2 := by
  rfl

/-- The assignment-dependent physical semantic replay is the independent
digest of the decoded semantic state when all 1,045 preimage fields agree. -/
theorem state_digest_eq_family_state_digest
    (assignment : Nat -> Nat)
    (canonical : forall column, assignment column < goldilocksP)
    (kind : ArmKind) (side : StateSide) (state : FamilyState)
    (preimage :
      (List.range 1045).map (fun index =>
          assignment (stateWordColumnFor kind side index)) =
        familyStateFields state) :
    stateDigest assignment canonical kind side = familyStateDigest state := by
  have assignedColumns :
      (((List.range 1045).map fun index =>
          stateWordColumnFor kind side index).map assignment) =
        familyStateFields state := by
    simpa only [List.map_map, Function.comp_apply] using preimage
  have externalOperations :
      ((List.range 1045).map fun index =>
          ColumnReplay.Operation.external
            (stateWordColumnFor kind side index)) =
        (((List.range 1045).map fun index =>
          stateWordColumnFor kind side index).map
            ColumnReplay.Operation.external) := by
    rw [List.map_map]
    apply List.map_congr_left
    intro index _
    rfl
  unfold stateDigest semanticDigestRun
  rw [digest_operations_eq,
    Operations.semanticExecute_append,
    Operations.semanticExecute_append,
    semanticExecute_pinned,
    externalOperations,
    semanticExecute_external,
    assignedColumns]
  simpa only [initialSemanticRun, familyStateDigest,
    familyStateDigestInput] using
    semanticExecute_first_digest assignment canonical
      (absorbFields
        (absorbFields domainInitialState (familyStateFrame.map wordField))
        ((familyStateFields state).map fieldValue))

private theorem canonical_fields_of_preimage
    (assignment : Nat -> Nat)
    (canonical : forall column, assignment column < goldilocksP)
    (columnAt : Nat -> Nat) (count : Nat) (state : FamilyState)
    (preimage :
      (List.range count).map (fun index => assignment (columnAt index)) =
        familyStateFields state) :
    ∀ value, value ∈ familyStateFields state -> value < goldilocksP := by
  intro value member
  rw [← preimage] at member
  rcases List.mem_map.mp member with ⟨index, _, rfl⟩
  exact canonical (columnAt index)

/-- Every field in the accepted arm's decoded before state is a canonical
Goldilocks representative. -/
theorem accepted_before_state_fields_canonical
    {setup : InputBindingSetup} {family : Family}
    (accepted : AcceptedArm setup family) :
    ∀ value, value ∈ familyStateFields accepted.beforeState ->
      value < goldilocksP := by
  apply canonical_fields_of_preimage accepted.bodyAssignment
    accepted.bodyCanonical
    (fun index => stateWordColumnFor (kindForFamily family) .before index)
    1045 accepted.beforeState
  simpa [AcceptedArm.beforeState] using accepted.publicBinding.beforePreimage

/-- Every field in the accepted arm's decoded after state is a canonical
Goldilocks representative. -/
theorem accepted_after_state_fields_canonical
    {setup : InputBindingSetup} {family : Family}
    (accepted : AcceptedArm setup family) :
    ∀ value, value ∈ familyStateFields accepted.afterState ->
      value < goldilocksP := by
  apply canonical_fields_of_preimage accepted.bodyAssignment
    accepted.bodyCanonical
    (fun index => stateWordColumnFor (kindForFamily family) .after index)
    1045 accepted.afterState
  simpa [AcceptedArm.afterState] using accepted.publicBinding.afterPreimage

/-- Explicit local family-digest equality and exact public cursor equality
between adjacent accepted arms. The local digest fields feed the phase
envelope; they are not XOut fields or public output words. -/
structure SemanticStateContinuous
    {setup : InputBindingSetup} {leftFamily rightFamily : Family}
    (left : AcceptedArm setup leftFamily)
    (right : AcceptedArm setup rightFamily) : Prop where
  localDigest : forall lane : Fin 4,
    left.bodyAssignment
        (phaseEnvelopeLocalSourceColumn (kindForFamily leftFamily) .after
          lane) =
      right.bodyAssignment
        (phaseEnvelopeLocalSourceColumn (kindForFamily rightFamily) .before
          lane)
  cursor :
    publicWordValue left.bodyAssignment (kindForFamily leftFamily)
        (cursorPublicWordIndex .after) =
      publicWordValue right.bodyAssignment (kindForFamily rightFamily)
        (cursorPublicWordIndex .before)

/-- An explicit adjacent local semantic link gives exact family-cursor
continuity and semantic state continuity, except for one named Poseidon2
collision event. -/
theorem accepted_semantic_continuity
    {setup : InputBindingSetup} {leftFamily rightFamily : Family}
    (left : AcceptedArm setup leftFamily)
    (right : AcceptedArm setup rightFamily)
    (continuous : SemanticStateContinuous left right) :
    left.afterState.familyCursor = right.beforeState.familyCursor ∧
      (left.afterState = right.beforeState ∨
        Poseidon2FamilyStateCollision) := by
  have leftBinding := left.publicBinding
  have rightBinding := right.publicBinding
  have cursorEqual :
      left.afterState.familyCursor = right.beforeState.familyCursor := by
    have leftCursorWord :
        publicWordValue left.bodyAssignment (kindForFamily leftFamily)
            (cursorPublicWordIndex .after) =
          left.afterState.familyCursor + firstFamilyProgramCursor := by
      simpa [AcceptedArm.afterState] using leftBinding.afterCursor
    have rightCursorWord :
        publicWordValue right.bodyAssignment (kindForFamily rightFamily)
            (cursorPublicWordIndex .before) =
          right.beforeState.familyCursor + firstFamilyProgramCursor := by
      simpa [AcceptedArm.beforeState] using rightBinding.beforeCursor
    have publicEqual := continuous.cursor
    rw [leftCursorWord, rightCursorWord] at publicEqual
    exact Nat.add_right_cancel publicEqual
  have leftDigest :
      stateDigest left.bodyAssignment left.bodyCanonical
          (kindForFamily leftFamily) .after =
        familyStateDigest left.afterState :=
    state_digest_eq_family_state_digest left.bodyAssignment
      left.bodyCanonical (kindForFamily leftFamily) .after left.afterState
      leftBinding.afterPreimage
  have rightDigest :
      stateDigest right.bodyAssignment right.bodyCanonical
          (kindForFamily rightFamily) .before =
        familyStateDigest right.beforeState :=
    state_digest_eq_family_state_digest right.bodyAssignment
      right.bodyCanonical (kindForFamily rightFamily) .before
      right.beforeState rightBinding.beforePreimage
  have digestEqual :
      familyStateDigest left.afterState =
        familyStateDigest right.beforeState := by
    rw [← leftDigest, ← rightDigest]
    funext lane
    apply Fin.ext
    exact (leftBinding.afterLocalDigestSource lane).trans
      ((continuous.localDigest lane).trans
        (rightBinding.beforeLocalDigestSource lane).symm)
  refine ⟨cursorEqual, ?_⟩
  by_cases stateEqual : left.afterState = right.beforeState
  · exact Or.inl stateEqual
  · right
    refine ⟨{
      left := left.afterState
      right := right.beforeState
      leftCanonical := canonical_fields_of_preimage left.bodyAssignment
        left.bodyCanonical
        (fun index => stateWordColumnFor (kindForFamily leftFamily) .after index)
        1045 left.afterState (by
          simpa [AcceptedArm.afterState] using leftBinding.afterPreimage)
      rightCanonical := canonical_fields_of_preimage right.bodyAssignment
        right.bodyCanonical
        (fun index => stateWordColumnFor (kindForFamily rightFamily) .before index)
        1045 right.beforeState (by
          simpa [AcceptedArm.beforeState] using rightBinding.beforePreimage)
      preimagesDifferent := ?_
      digestEqual := digestEqual }⟩
    intro preimagesEqual
    exact stateEqual (familyStateFields_injective preimagesEqual)

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyContinuity
