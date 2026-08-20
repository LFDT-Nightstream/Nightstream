import Nightstream.Implementation.Nebula.FPrime.State.OutputPoseidonBinding
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPreludeSource
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingLifecycleRelation
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPreludeXOutRowSound

/-!
Contract: exact base-lifecycle binding for the two Rust Prelude XOut hashes.

Owns the reduction from satisfying exact hash rows and verifier-bound public
digest lanes to equality of both complete 32-field lifecycle frames, or one
named outer Poseidon2 collision. The configured abstract XOut hash must be
proved equal to the concrete canonical Poseidon2 computation.

Does not own bounds for lifecycle natural-number fields, Poseidon2 collision
resistance, recursive or terminal rows, or final selective-arm placement.

Assurance tier: artifact-checked conditional adapter for
`FPRIME-STREAMING-PRELUDE-XOUT-ROWS-V1`, Nightstream b2/k16.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPreludeXOutLifecycleBridge

open Nightstream.Implementation.Nebula
open Nightstream.Implementation.Nebula.StateOutputPoseidonBinding
open Nightstream.Implementation.Nebula.StateOutputPoseidonRows
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.Implementation.R1CS.Canonical.CanonicalU64Recipe
open Nightstream.Implementation.R1CS.Canonical.CanonicalU64RecipeSound
open Nightstream.Implementation.R1CS.Canonical.GoldilocksField
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPreludeXOut
open Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPreludeXOutRowSound
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplay.Artifact
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleRelation
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPreludeXOut.Artifact
open Nightstream.Implementation.R1CS.Poseidon2Sponge
open Nightstream.Protocol.FPrime

universe uParams uStructure uRunning uFresh uNifsProof uNebulaOpen

/-- Concrete hash conformance for the state-output branch. This is a
deterministic implementation claim, not a collision-resistance assumption. -/
structure Poseidon2Compatible
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {NifsProof : Type uNifsProof}
    {Nebula : Type}
    {NebulaOpen : Type uNebulaOpen}
    {armCount : Nat}
    (configuration : Configuration Params StructureDigest Running Fresh
      NifsProof Nebula NebulaOpen armCount) : Prop where
  stateOutput :
    ∀ (state : OuterState Running Fresh Nebula) (nebula : Nebula),
      state.nebula = some nebula →
        digestValues
            (XOut.compute configuration.hashSemantics .stateful
              configuration.context state) =
          outerHash (frame configuration state nebula)

/-- Exact missing trust-boundary link from the two XOut output columns to the
single public envelope shared by the lifecycle and selected phase. Generated
public-bit rows must discharge this structure. -/
structure PublicDigestBinding
    (assignment : Nat → Nat) (envelope : PublicEnvelope) : Prop where
  before :
    assignedDigest artifact.beforeXOut assignment =
      digestValues envelope.beforeXOut
  after :
    assignedDigest artifact.afterXOut assignment =
      digestValues envelope.afterXOut

/-- Numeric value of one normalized 64-bit public word. The generated XOut
artifact fixes the word bases and the exact source-to-normalized pullback. -/
def normalizedWordValue
    (assignment : Nat → Nat) (block : HashBlock) (lane : Fin 4) : Nat :=
  (List.range 64).foldl
    (fun value bit => value + 2 ^ bit *
      artifact.sourceAssignment assignment
        (bitColumn (publicCall block lane).layout bit)) 0

/-- Side of the base lifecycle cursor boundary owned by this bridge. -/
inductive CursorSide where
  | before
  | after
deriving DecidableEq, Repr

/-- Exact Rust canonical-u64 call for one base lifecycle cursor word. -/
def cursorCall : CursorSide → CanonicalCall
  | .before =>
      { rowStart := 671261, rowEnd := 671330, fieldColumn := 665426,
        bitBase := 671267, highFlagColumn := 671331,
        inverseColumn := 671332 }
  | .after =>
      { rowStart := 665405, rowEnd := 665474, fieldColumn := 665427,
        bitBase := 665429, highFlagColumn := 665493,
        inverseColumn := 665494 }

/-- First normalized public column for one base lifecycle cursor word. -/
def cursorNormalizedBitBase : CursorSide → Nat
  | .before => 513
  | .after => 577

/-- Numeric value of one cursor word in the normalized public prefix. -/
def normalizedCursorWordValue
    (assignment : Nat → Nat) (side : CursorSide) : Nat :=
  (List.range 64).foldl
    (fun value bit => value + 2 ^ bit *
      assignment (cursorNormalizedBitBase side + bit)) 0

/-- Verifier-owned typed interpretation of the normalized public bits. -/
structure PublicAssignmentBinding
    (assignment : Nat → Nat) (envelope : PublicEnvelope) : Prop where
  beforeEnvelope : ∀ lane,
    normalizedWordValue assignment artifact.beforeXOut lane =
      digestValues envelope.beforeXOut lane
  afterEnvelope : ∀ lane,
    normalizedWordValue assignment artifact.afterXOut lane =
      digestValues envelope.afterXOut lane
  beforeCursor : normalizedCursorWordValue assignment .before =
    envelope.beforeCursor
  afterCursor : normalizedCursorWordValue assignment .after =
    envelope.afterCursor

private theorem cursorCall_member (side : CursorSide) :
    cursorCall side ∈
      Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPreludeSource.artifact.canonicalU64Calls := by
  change cursorCall side ∈
    Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPreludeSource.Generated.FPrimeFullHistoryStreamingPreludeCanonicalCalls.calls
  cases side <;>
    simp [cursorCall,
      Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPreludeSource.Generated.FPrimeFullHistoryStreamingPreludeCanonicalCalls.calls]

private theorem cursorBit_normalizedColumn
    (side : CursorSide) (index : Nat) (bounded : index < 64) :
    artifact.normalizedColumn (bitColumn (cursorCall side).layout index) =
      cursorNormalizedBitBase side + index := by
  cases side <;>
    simp only [cursorCall, cursorNormalizedBitBase, CanonicalCall.layout, bitColumn]
  all_goals
    unfold RawArtifact.normalizedColumn
    rw [artifact_valid.publicSpans]
    simp only [List.foldr_cons, ColumnSpan.mapColumn]
    repeat' first
      | rw [if_neg (by omega)]
      | rw [if_pos (by omega)]
    omega

private theorem foldl_cursorBits_eq_sourceBits
    (assignment : Nat → Nat) (side : CursorSide)
    (indices : List Nat)
    (bounded : ∀ index ∈ indices, index < 64) (initial : Nat) :
    indices.foldl
        (fun value index => value + 2 ^ index *
          artifact.sourceAssignment assignment
            (bitColumn (cursorCall side).layout index)) initial =
      indices.foldl
        (fun value index => value + 2 ^ index *
          assignment (cursorNormalizedBitBase side + index)) initial := by
  induction indices generalizing initial with
  | nil => rfl
  | cons index tail inductionHypothesis =>
      simp only [List.foldl_cons]
      rw [show artifact.sourceAssignment assignment
          (bitColumn (cursorCall side).layout index) =
          assignment (cursorNormalizedBitBase side + index) by
        simp [RawArtifact.sourceAssignment,
          cursorBit_normalizedColumn side index (bounded index (by simp))]]
      apply inductionHypothesis
      intro next member
      exact bounded next (by simp [member])

private theorem cursor_bitsValue_eq_normalizedCursorWordValue
    (assignment : Nat → Nat) (side : CursorSide) :
    CanonicalU64RecipeSound.bitsValue (artifact.sourceAssignment assignment)
        (cursorCall side).layout =
      normalizedCursorWordValue assignment side := by
  exact foldl_cursorBits_eq_sourceBits assignment side (List.range 64)
    (fun index member => List.mem_range.mp member) 0

private theorem cursor_call_input_exact
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied :
      Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPreludeSource.artifact.Satisfied
        (artifact.sourceAssignment assignment))
    (side : CursorSide) :
    artifact.sourceAssignment assignment (cursorCall side).fieldColumn =
      CanonicalU64RecipeSound.bitsValue (artifact.sourceAssignment assignment)
        (cursorCall side).layout := by
  have refined := CanonicalU64RecipeSound.sound goldilocks_euclidPrime
    (fun column => canonical (artifact.normalizedColumn column))
    (by simpa [RawArtifact.sourceAssignment, RawArtifact.normalizedColumn] using one)
    (satisfied.1 _ (cursorCall_member side))
  simpa [CanonicalCall.layout, lcEval,
    Nat.mod_eq_of_lt
      (canonical (artifact.normalizedColumn (cursorCall side).fieldColumn))]
    using refined.input_eq

/-- Exact cursor rows bind both source cursor fields to the same complete
public envelope used by the base lifecycle relation. -/
theorem public_rows_imply_cursor_fields
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied :
      Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPreludeSource.artifact.Satisfied
        (artifact.sourceAssignment assignment))
    (envelope : PublicEnvelope)
    (binding : PublicAssignmentBinding assignment envelope) :
    artifact.sourceAssignment assignment (cursorCall .before).fieldColumn =
        envelope.beforeCursor /\
      artifact.sourceAssignment assignment (cursorCall .after).fieldColumn =
        envelope.afterCursor := by
  constructor
  · calc
      artifact.sourceAssignment assignment (cursorCall .before).fieldColumn =
          CanonicalU64RecipeSound.bitsValue
            (artifact.sourceAssignment assignment)
            (cursorCall .before).layout :=
        cursor_call_input_exact assignment canonical one satisfied .before
      _ = normalizedCursorWordValue assignment .before :=
        cursor_bitsValue_eq_normalizedCursorWordValue assignment .before
      _ = envelope.beforeCursor := binding.beforeCursor
  · calc
      artifact.sourceAssignment assignment (cursorCall .after).fieldColumn =
          CanonicalU64RecipeSound.bitsValue
            (artifact.sourceAssignment assignment)
            (cursorCall .after).layout :=
        cursor_call_input_exact assignment canonical one satisfied .after
      _ = normalizedCursorWordValue assignment .after :=
        cursor_bitsValue_eq_normalizedCursorWordValue assignment .after
      _ = envelope.afterCursor := binding.afterCursor

/-- The exact base cursor fields are zero and one. The values come from the
typed base transition, not from a self-reported public assignment. -/
theorem base_rows_imply_cursor_fields_zero_one
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {NifsProof : Type uNifsProof}
    {Nebula : Type}
    {NebulaOpen : Type uNebulaOpen}
    {armCount : Nat}
    {configuration : Configuration Params StructureDigest Running Fresh
      NifsProof Nebula NebulaOpen armCount}
    (base : Base configuration)
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied :
      Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPreludeSource.artifact.Satisfied
        (artifact.sourceAssignment assignment))
    (binding : PublicAssignmentBinding assignment base.commonPublic) :
    artifact.sourceAssignment assignment (cursorCall .before).fieldColumn = 0 /\
      artifact.sourceAssignment assignment (cursorCall .after).fieldColumn = 1 := by
  have fields := public_rows_imply_cursor_fields assignment canonical one
    satisfied base.commonPublic binding
  have cursors := Base.public_cursors_zero_one base
  exact ⟨fields.1.trans cursors.1, fields.2.trans cursors.2⟩

private theorem bitsValue_eq_normalizedWordValue
    (assignment : Nat → Nat)
    (block : HashBlock) (lane : Fin 4) :
    Nightstream.Implementation.R1CS.Canonical.CanonicalU64RecipeSound.bitsValue
        (artifact.sourceAssignment assignment) (publicCall block lane).layout =
      normalizedWordValue assignment block lane := by
  rfl

private theorem canonical_call_input_exact
    (block : HashBlock) (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : block.SourceSatisfied assignment)
    (lane : Fin 4)
    (member : publicCall block lane ∈ block.canonicalCalls) :
    assignment (publicCall block lane).fieldColumn =
      Nightstream.Implementation.R1CS.Canonical.CanonicalU64RecipeSound.bitsValue
        assignment (publicCall block lane).layout := by
  have refined := CanonicalU64RecipeSound.sound goldilocks_euclidPrime
    canonical one (satisfied.2 _ member)
  simpa [Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplay.Artifact.CanonicalCall.layout,
    lcEval,
    Nat.mod_eq_of_lt (canonical (publicCall block lane).fieldColumn)] using
    refined.input_eq

private theorem block_public_digest_exact
    (block : HashBlock)
    (callMember : ∀ lane,
      publicCall block lane ∈ block.canonicalCalls)
    (fieldColumn : ∀ lane,
      (publicCall block lane).fieldColumn =
        block.recipe.outputColumns.getD lane.val 0)
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : block.SourceSatisfied (artifact.sourceAssignment assignment))
    (target : Fin 4 → Nat)
    (envelope : ∀ lane,
      normalizedWordValue assignment block lane = target lane) :
    assignedDigest block assignment = target := by
  funext lane
  change artifact.sourceAssignment assignment
      (block.recipe.outputColumns.getD lane.val 0) = target lane
  rw [← fieldColumn lane]
  calc
    artifact.sourceAssignment assignment (publicCall block lane).fieldColumn =
        Nightstream.Implementation.R1CS.Canonical.CanonicalU64RecipeSound.bitsValue
          (artifact.sourceAssignment assignment) (publicCall block lane).layout :=
      canonical_call_input_exact block (artifact.sourceAssignment assignment)
        (fun column => canonical (artifact.normalizedColumn column))
        (by simpa [RawArtifact.sourceAssignment, RawArtifact.normalizedColumn] using one)
        satisfied lane (callMember lane)
    _ = normalizedWordValue assignment block lane :=
      bitsValue_eq_normalizedWordValue assignment block lane
    _ = target lane := envelope lane

/-- The exact eight canonical-u64 row blocks and their emitted public-column
map discharge the digest-binding premise used by the lifecycle theorem. -/
theorem public_rows_imply_digest_binding
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : artifact.Satisfied assignment)
    (envelope : PublicEnvelope)
    (binding : PublicAssignmentBinding assignment envelope) :
    PublicDigestBinding assignment envelope where
  before := block_public_digest_exact artifact.beforeXOut
    before_publicCall_member before_publicCall_fieldColumn assignment canonical one satisfied.2
    (digestValues envelope.beforeXOut) binding.beforeEnvelope
  after := block_public_digest_exact artifact.afterXOut
    after_publicCall_member after_publicCall_fieldColumn assignment canonical one satisfied.1
    (digestValues envelope.afterXOut) binding.afterEnvelope

private theorem inputValues_canonical
    (block : HashBlock) (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP) :
    ∀ value ∈ inputValues block assignment, value < goldilocksP := by
  intro value member
  rw [inputValues] at member
  rcases List.mem_map.mp member with ⟨column, _, rfl⟩
  exact canonical (artifact.normalizedColumn column)

private theorem computedDigest_eq_outerHash
    (block : HashBlock) (values : List Nat)
    (schedule :
      valueSchedules block.recipe.trace.rounds = expectedSchedule) :
    computedDigest block values = outerHash values := by
  have sameSchedule :
      valueSchedules block.recipe.trace.rounds =
        valueSchedules representativeRounds :=
    schedule.trans representativeRounds_schedule.symm
  have runExact :=
    runValueRounds_eq_of_schedules sameSchedule values (fun _ => 0)
  funext lane
  exact congrFun runExact lane.val

private theorem rows_bind_frame_or_collision
    (block : HashBlock)
    (inputLength : block.recipe.inputColumns.length = 32)
    (schedule :
      valueSchedules block.recipe.trace.rounds = expectedSchedule)
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (rowSound :
      assignedDigest block assignment =
        computedDigest block (inputValues block assignment))
    (authoritative : List Nat)
    (authoritativeLength : authoritative.length = 32)
    (authoritativeCanonical :
      ∀ value ∈ authoritative, value < goldilocksP)
    (digestBound :
      assignedDigest block assignment = outerHash authoritative) :
    inputValues block assignment = authoritative ∨ OuterCollision := by
  have suppliedLength :
      (inputValues block assignment).length = 32 := by
    simpa [inputValues] using inputLength
  let suppliedFrame : CanonicalFrame :=
    ⟨inputValues block assignment, suppliedLength,
      inputValues_canonical block assignment canonical⟩
  let authoritativeFrame : CanonicalFrame :=
    ⟨authoritative, authoritativeLength, authoritativeCanonical⟩
  have equalHashes :
      outerHash (inputValues block assignment) = outerHash authoritative := by
    calc
      outerHash (inputValues block assignment) =
          computedDigest block (inputValues block assignment) :=
        (computedDigest_eq_outerHash block _ schedule).symm
      _ = assignedDigest block assignment := rowSound.symm
      _ = outerHash authoritative := digestBound
  have framedEqual : digest suppliedFrame = digest authoritativeFrame := by
    simpa [digest, suppliedFrame, authoritativeFrame] using equalHashes
  simpa [suppliedFrame, authoritativeFrame] using
    frame_values_eq_or_outer_collision suppliedFrame authoritativeFrame
      framedEqual

/-- Satisfying both exact Prelude XOut row blocks binds their full ordered
inputs to one verifier-derived base lifecycle invocation. If either supplied
frame differs, the result is the named concrete outer Poseidon2 collision.

`PublicDigestBinding` remains explicit until generated public-bit rows prove
that the four output columns equal the common public envelope. -/
theorem base_frames_exact_or_outer_collision
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {NifsProof : Type uNifsProof}
    {Nebula : Type}
    {NebulaOpen : Type uNebulaOpen}
    {armCount : Nat}
    {configuration : Configuration Params StructureDigest Running Fresh
      NifsProof Nebula NebulaOpen armCount}
    (compatible : Poseidon2Compatible configuration)
    (base : Base configuration)
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : artifact.Satisfied assignment)
    (publicBound : PublicDigestBinding assignment base.commonPublic)
    (priorCanonical :
      ∀ value ∈ frame configuration base.prior base.priorNebula,
        value < goldilocksP)
    (nextCanonical :
      ∀ value ∈ frame configuration base.next base.nextNebula,
        value < goldilocksP) :
    (inputValues artifact.beforeXOut assignment =
          frame configuration base.prior base.priorNebula ∧
        inputValues artifact.afterXOut assignment =
          frame configuration base.next base.nextNebula) ∨
      OuterCollision := by
  have beforeSchedule :
      valueSchedules artifact.beforeXOut.recipe.trace.rounds =
        expectedSchedule := by
    simpa [expectedSchedule] using before_valueSchedules_exact
  have afterSchedule :
      valueSchedules artifact.afterXOut.recipe.trace.rounds =
        expectedSchedule := by
    simpa [expectedSchedule] using after_valueSchedules_exact
  have beforeDigestBound :
      assignedDigest artifact.beforeXOut assignment =
        outerHash (frame configuration base.prior base.priorNebula) := by
    calc
      assignedDigest artifact.beforeXOut assignment =
          digestValues base.commonPublic.beforeXOut := publicBound.before
      _ = digestValues
          (XOut.compute configuration.hashSemantics .stateful
            configuration.context base.prior) := by
        rw [Invocation.before_public_exact base.toInvocation]
      _ = outerHash
          (frame configuration base.prior base.priorNebula) :=
        compatible.stateOutput base.prior base.priorNebula
          base.priorNebulaExact
  have afterDigestBound :
      assignedDigest artifact.afterXOut assignment =
        outerHash (frame configuration base.next base.nextNebula) := by
    calc
      assignedDigest artifact.afterXOut assignment =
          digestValues base.commonPublic.afterXOut := publicBound.after
      _ = digestValues
          (XOut.compute configuration.hashSemantics .stateful
            configuration.context base.next) := by
        rw [Invocation.after_public_exact base.toInvocation]
      _ = outerHash
          (frame configuration base.next base.nextNebula) :=
        compatible.stateOutput base.next base.nextNebula base.nextNebulaExact
  have beforeResult := rows_bind_frame_or_collision
    artifact.beforeXOut before_inputLength_exact beforeSchedule assignment
    canonical
    (before_rows_imply_hash assignment canonical one satisfied)
    (frame configuration base.prior base.priorNebula)
    (frame_length configuration base.prior base.priorNebula)
    priorCanonical beforeDigestBound
  rcases beforeResult with beforeExact | collision
  · have afterResult := rows_bind_frame_or_collision
      artifact.afterXOut after_inputLength_exact afterSchedule assignment
      canonical
      (after_rows_imply_hash assignment canonical one satisfied)
      (frame configuration base.next base.nextNebula)
      (frame_length configuration base.next base.nextNebula)
      nextCanonical afterDigestBound
    rcases afterResult with afterExact | collision
    · exact Or.inl ⟨beforeExact, afterExact⟩
    · exact Or.inr collision
  · exact Or.inr collision

/-- Exact Prelude rows and the verifier-owned public-bit assignment discharge
the intermediate digest binding. Both complete base lifecycle frames are then
equal to the Rust-emitted XOut inputs, or the result is the named outer
Poseidon2 collision. -/
theorem base_rows_and_public_assignment_exact_or_outer_collision
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {NifsProof : Type uNifsProof}
    {Nebula : Type}
    {NebulaOpen : Type uNebulaOpen}
    {armCount : Nat}
    {configuration : Configuration Params StructureDigest Running Fresh
      NifsProof Nebula NebulaOpen armCount}
    (compatible : Poseidon2Compatible configuration)
    (base : Base configuration)
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : artifact.Satisfied assignment)
    (publicAssignment : PublicAssignmentBinding assignment base.commonPublic)
    (priorCanonical :
      ∀ value ∈ frame configuration base.prior base.priorNebula,
        value < goldilocksP)
    (nextCanonical :
      ∀ value ∈ frame configuration base.next base.nextNebula,
        value < goldilocksP) :
    (inputValues artifact.beforeXOut assignment =
          frame configuration base.prior base.priorNebula ∧
        inputValues artifact.afterXOut assignment =
          frame configuration base.next base.nextNebula) ∨
      OuterCollision :=
  base_frames_exact_or_outer_collision compatible base assignment canonical
    one satisfied
    (public_rows_imply_digest_binding assignment canonical one satisfied
      base.commonPublic publicAssignment)
    priorCanonical nextCanonical

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPreludeXOutLifecycleBridge
