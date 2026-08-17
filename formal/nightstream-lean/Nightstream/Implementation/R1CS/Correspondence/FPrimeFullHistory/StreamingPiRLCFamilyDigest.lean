import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPiRLCFamilyDigestDomain
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPiRLCFamilyPublicArtifact
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCPhaseEnvelope
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Transcript.ColumnReplay
import Nightstream.Implementation.R1CS.Canonical.Poseidon2Normalized

/-!
Contract: exact local-state digest and full-XOut public-word placement for one
streaming PiRLC family arm.

Owns the four fixed field-framing words, the exact 1045-field before and after
local preimages, the 263 connected Poseidon2 calls for each local digest, the
placement of those digests in the phase-envelope source columns, and the eight
full-XOut public digest-word roles.

Does not own collision resistance, the full XOut hash execution,
the phase-envelope hash, interpretation of the XOut fields, the two global
cursor words, the source-prefix relation, selective lowering, or recursive
lifecycle integration.

The generated artifact supplies only physical columns and call slices. The
handwritten operation list supplies protocol meaning. A digest is accepted
only when the exact generated rows replay this operation list from the
independently certified application-domain state.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 524288
set_option maxHeartbeats 8000000

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyDigest

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.Implementation.R1CS.Canonical.CanonicalU64RecipeSound
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplay.Artifact
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublic.Artifact
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublicArtifact
open Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript
open Nightstream.Implementation.R1CS.PiRlcChallenge.TranscriptMachine
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyDigestDomain

inductive StateSide where
  | after
  | before
deriving DecidableEq, Repr

/-- Phase-envelope boundary that corresponds to one local-state boundary. -/
def phaseEnvelopeSideFor : StateSide →
    FPrimeFullHistoryStreamingPiRLCPhaseEnvelope.Artifact.StateSide
  | .after => .after
  | .before => .before

/-- Exact generated phase-envelope arm for one public-suffix parity. -/
def phaseEnvelopeArmFor (kind : ArmKind) :
    FPrimeFullHistoryStreamingPiRLCPhaseEnvelope.Artifact.RawArm :=
  match kind with
  | .even =>
      Artifacts.FPrimeFullHistory.StreamingPiRLCPhaseEnvelope.evenArm
  | .odd =>
      Artifacts.FPrimeFullHistory.StreamingPiRLCPhaseEnvelope.oddArm

/-- Exact local-digest source column consumed by the phase envelope. -/
def phaseEnvelopeLocalSourceColumn
    (kind : ArmKind) (side : StateSide) (lane : Fin 4) : Nat :=
  ((phaseEnvelopeArmFor kind).localSourceColumns
    (phaseEnvelopeSideFor side)).getD lane.val 0

/-! ## Exact physical schedule -/

def digestPinValues : List Nat :=
  initialStateValues ++ [2, 5, 435744240755, 1045, 1]

def digestPinColumns (kind : ArmKind) : StateSide → List Nat
  | .after => (armFor kind).afterDigestPinColumns
  | .before => (armFor kind).beforeDigestPinColumns

def digestCallOffset : StateSide → Nat
  | .after => 0
  | .before => 263

def digestTrace (kind : ArmKind) (side : StateSide) :
    TranscriptCertificate.Trace where
  pins := (digestPinColumns kind side).zip digestPinValues
  calls := ((armFor kind).poseidon2Calls.drop
    (digestCallOffset side)).take 263

def digestStart (kind : ArmKind) (side : StateSide) : ColumnReplay.Run where
  cursor := {
    lanes := fun lane => (digestPinColumns kind side).getD lane.val 0
    absorbed := ⟨2, by decide⟩
    nextPin := 8
    nextCall := 0 }
  digests := []

def stateWordColumnFor
    (kind : ArmKind) (side : StateSide) (index : Nat) : Nat :=
  match side with
  | .after => (armFor kind).afterStateColumns.getD index 0
  | .before => (armFor kind).beforeStateColumns.getD index 0

def xOutPreimageColumn
    (kind : ArmKind) (side : StateSide) (index : Nat) : Nat :=
  match side with
  | .after => (armFor kind).afterXOutPreimageColumns.getD index 0
  | .before => (armFor kind).beforeXOutPreimageColumns.getD index 0

def xOutDigestColumn
    (kind : ArmKind) (side : StateSide) (lane : Fin 4) : Nat :=
  match side with
  | .after => (armFor kind).afterXOutDigestColumns.getD lane.val 0
  | .before => (armFor kind).beforeXOutDigestColumns.getD lane.val 0

/-- The digest relation contains the append-fields framing, exactly 1045 state
fields, and one digest gate. -/
def digestOperations (kind : ArmKind) (side : StateSide) :
    List ColumnReplay.Operation :=
  [.pinned 2, .pinned 5, .pinned 435744240755, .pinned 1045] ++
    ((List.range 1045).map fun index =>
      .external (stateWordColumnFor kind side index)) ++
    [.digest]

/-- The operation list has no hidden or prover-selected preimage word. -/
theorem exact_digest_operation_shape (kind : ArmKind) (side : StateSide) :
    (digestOperations kind side).length = 1050 ∧
      ((digestOperations kind side).drop 4).take 1045 =
        ((List.range 1045).map fun index =>
          .external (stateWordColumnFor kind side index)) ∧
      (digestOperations kind side).getD 1049 (.pinned 0) = .digest := by
  simp [digestOperations]

def digestLastCall (kind : ArmKind) (side : StateSide) : Poseidon2Call.Call :=
  (digestTrace kind side).calls.getD 262 default

def digestOutputColumns
    (kind : ArmKind) (side : StateSide) : Fin 4 → Nat := fun lane =>
  ColumnReplay.callOutputColumns (digestLastCall kind side)
    ⟨lane.val, by
      have laneLt := lane.isLt
      change lane.val < 8
      omega⟩

def digestResult (kind : ArmKind) (side : StateSide) : ColumnReplay.Run where
  cursor := {
    lanes := ColumnReplay.callOutputColumns (digestLastCall kind side)
    absorbed := ⟨0, by decide⟩
    nextPin := 13
    nextCall := 263 }
  digests := [digestOutputColumns kind side]

private structure CursorView where
  lanes : List Nat
  absorbed : Nat
  nextPin : Nat
  nextCall : Nat
deriving DecidableEq

private def cursorView (cursor : ColumnReplay.Cursor) : CursorView where
  lanes := List.ofFn cursor.lanes
  absorbed := cursor.absorbed.val
  nextPin := cursor.nextPin
  nextCall := cursor.nextCall

private structure RunView where
  cursor : CursorView
  digests : List (List Nat)
deriving DecidableEq

private def runView (run : ColumnReplay.Run) : RunView where
  cursor := cursorView run.cursor
  digests := run.digests.map List.ofFn

private theorem cursorView_injective : Function.Injective cursorView := by
  intro left right equal
  cases left with
  | mk leftLanes leftAbsorbed leftPin leftCall =>
      cases right with
      | mk rightLanes rightAbsorbed rightPin rightCall =>
          have lanesEqual : leftLanes = rightLanes :=
            List.ofFn_injective (congrArg CursorView.lanes equal)
          have absorbedEqual : leftAbsorbed = rightAbsorbed :=
            Fin.ext (congrArg CursorView.absorbed equal)
          have pinEqual : leftPin = rightPin :=
            congrArg CursorView.nextPin equal
          have callEqual : leftCall = rightCall :=
            congrArg CursorView.nextCall equal
          subst rightLanes
          subst rightAbsorbed
          subst rightPin
          subst rightCall
          rfl

private theorem runView_injective : Function.Injective runView := by
  intro left right equal
  cases left with
  | mk leftCursor leftDigests =>
      cases right with
      | mk rightCursor rightDigests =>
          have cursorEqual : leftCursor = rightCursor :=
            cursorView_injective (congrArg RunView.cursor equal)
          have digestEqual : leftDigests = rightDigests := by
            apply (List.map_injective_iff.mpr fun first second valuesEqual =>
              List.ofFn_injective valuesEqual)
            exact congrArg RunView.digests equal
          subst rightCursor
          subst rightDigests
          rfl

private def executionMatches
    (result : Option ColumnReplay.Run) (expected : ColumnReplay.Run) : Bool :=
  match result with
  | none => false
  | some actual => decide (runView actual = runView expected)

private theorem executionMatches_sound
    {result : Option ColumnReplay.Run} {expected : ColumnReplay.Run}
    (checked : executionMatches result expected = true) :
    result = some expected := by
  cases result with
  | none => simp [executionMatches] at checked
  | some actual =>
      have viewsEqual : runView actual = runView expected := by
        exact of_decide_eq_true (by simpa [executionMatches] using checked)
      rw [runView_injective viewsEqual]

private def digestTail0 (kind : ArmKind) (side : StateSide) :=
  digestOperations kind side

private def digestChunk0 (kind : ArmKind) (side : StateSide) :=
  (digestTail0 kind side).take 256

private def digestTail1 (kind : ArmKind) (side : StateSide) :=
  (digestTail0 kind side).drop 256

private def digestChunk1 (kind : ArmKind) (side : StateSide) :=
  (digestTail1 kind side).take 256

private def digestTail2 (kind : ArmKind) (side : StateSide) :=
  (digestTail1 kind side).drop 256

private def digestChunk2 (kind : ArmKind) (side : StateSide) :=
  (digestTail2 kind side).take 256

private def digestTail3 (kind : ArmKind) (side : StateSide) :=
  (digestTail2 kind side).drop 256

private def digestChunk3 (kind : ArmKind) (side : StateSide) :=
  (digestTail3 kind side).take 256

private def digestTail4 (kind : ArmKind) (side : StateSide) :=
  (digestTail3 kind side).drop 256

/-- Exact replay state after one 256-operation leaf. The first two lanes are
the last two absorbed state columns. The other lanes remain the output of the
last consumed call. -/
private def digestCheckpoint
    (kind : ArmKind) (side : StateSide)
    (leftState rightState callIndex nextCall : Nat) : ColumnReplay.Run where
  cursor := {
    lanes := fun lane =>
      if lane.val = 0 then stateWordColumnFor kind side leftState
      else if lane.val = 1 then stateWordColumnFor kind side rightState
      else ColumnReplay.callOutputColumns
        ((digestTrace kind side).calls.getD callIndex default) lane
    absorbed := ⟨2, by decide⟩
    nextPin := 12
    nextCall := nextCall }
  digests := []

private def digestRun1 (kind : ArmKind) (side : StateSide) : ColumnReplay.Run :=
  digestCheckpoint kind side 250 251 63 64

private def digestRun2 (kind : ArmKind) (side : StateSide) : ColumnReplay.Run :=
  digestCheckpoint kind side 506 507 127 128

private def digestRun3 (kind : ArmKind) (side : StateSide) : ColumnReplay.Run :=
  digestCheckpoint kind side 762 763 191 192

private def digestRun4 (kind : ArmKind) (side : StateSide) : ColumnReplay.Run :=
  digestCheckpoint kind side 1018 1019 255 256

private theorem digestChunk0_execution
    (kind : ArmKind) (side : StateSide) :
    ColumnReplay.execute (digestTrace kind side) (digestStart kind side)
        (digestChunk0 kind side) =
      some (digestRun1 kind side) := by
  cases kind <;> cases side <;> apply executionMatches_sound <;> rfl

private theorem digestChunk1_execution
    (kind : ArmKind) (side : StateSide) :
    ColumnReplay.execute (digestTrace kind side) (digestRun1 kind side)
        (digestChunk1 kind side) =
      some (digestRun2 kind side) := by
  cases kind <;> cases side <;> apply executionMatches_sound <;> rfl

private theorem digestChunk2_execution
    (kind : ArmKind) (side : StateSide) :
    ColumnReplay.execute (digestTrace kind side) (digestRun2 kind side)
        (digestChunk2 kind side) =
      some (digestRun3 kind side) := by
  cases kind <;> cases side <;> apply executionMatches_sound <;> rfl

private theorem digestChunk3_execution
    (kind : ArmKind) (side : StateSide) :
    ColumnReplay.execute (digestTrace kind side) (digestRun3 kind side)
        (digestChunk3 kind side) =
      some (digestRun4 kind side) := by
  cases kind <;> cases side <;> apply executionMatches_sound <;> rfl

private theorem digestTail4_execution
    (kind : ArmKind) (side : StateSide) :
    ColumnReplay.execute (digestTrace kind side) (digestRun4 kind side)
        (digestTail4 kind side) =
      some (digestResult kind side) := by
  cases kind <;> cases side <;> apply executionMatches_sound <;> rfl

private theorem digestTail3_execution
    (kind : ArmKind) (side : StateSide) :
    ColumnReplay.execute (digestTrace kind side) (digestRun3 kind side)
        (digestTail3 kind side) =
      some (digestResult kind side) := by
  rw [← List.take_append_drop 256 (digestTail3 kind side)]
  exact ColumnReplay.execute_append (digestChunk3_execution kind side)
    (digestTail4_execution kind side)

private theorem digestTail2_execution
    (kind : ArmKind) (side : StateSide) :
    ColumnReplay.execute (digestTrace kind side) (digestRun2 kind side)
        (digestTail2 kind side) =
      some (digestResult kind side) := by
  rw [← List.take_append_drop 256 (digestTail2 kind side)]
  exact ColumnReplay.execute_append (digestChunk2_execution kind side)
    (digestTail3_execution kind side)

private theorem digestTail1_execution
    (kind : ArmKind) (side : StateSide) :
    ColumnReplay.execute (digestTrace kind side) (digestRun1 kind side)
        (digestTail1 kind side) =
      some (digestResult kind side) := by
  rw [← List.take_append_drop 256 (digestTail1 kind side)]
  exact ColumnReplay.execute_append (digestChunk1_execution kind side)
    (digestTail2_execution kind side)

/-- Each physical digest path consumes 13 pins and 263 connected Poseidon2
calls and exposes one four-lane digest. -/
theorem digest_execution (kind : ArmKind) (side : StateSide) :
    ColumnReplay.execute (digestTrace kind side) (digestStart kind side)
        (digestOperations kind side) =
      some (digestResult kind side) := by
  change ColumnReplay.execute (digestTrace kind side) (digestStart kind side)
      (digestTail0 kind side) = some (digestResult kind side)
  rw [← List.take_append_drop 256 (digestTail0 kind side)]
  exact ColumnReplay.execute_append (digestChunk0_execution kind side)
    (digestTail1_execution kind side)

def glueProgram (kind : ArmKind) : List Row :=
  (armFor kind).glueRows.map IndexedRow.row

theorem digest_trace_pins_canonical
    (kind : ArmKind) (side : StateSide) :
    ConstantPins.ValuesCanonical (digestTrace kind side).pins := by
  cases kind <;> cases side <;> decide

def normalizedDigestPinRows
    (kind : ArmKind) (side : StateSide) : List Row :=
  Poseidon2Normalized.normalizeProgram
    (ConstantPins.rows (digestTrace kind side).pins)

private theorem normalized_digest_pin_rows_exact
    (kind : ArmKind) (side : StateSide) :
    normalizedDigestPinRows kind side =
      ((glueProgram kind).drop
        (match side with | .after => 6 | .before => 21)).take 13 := by
  cases kind <;> cases side <;> rfl

private theorem digest_pin_row_mem_glue
    (kind : ArmKind) (side : StateSide) (row : Row)
    (member : row ∈ normalizedDigestPinRows kind side) :
    row ∈ glueProgram kind := by
  rw [normalized_digest_pin_rows_exact kind side] at member
  exact List.mem_of_mem_drop (List.mem_of_mem_take member)

private theorem rowsIncluded_self (rows : List Row) :
    rowsIncluded rows rows = true := by
  unfold rowsIncluded
  apply List.all_eq_true.mpr
  intro row member
  exact decide_eq_true member

private theorem glue_satisfies
    (kind : ArmKind) (assignment : Nat → Nat)
    (satisfied : (armFor kind).Satisfied assignment) :
    Satisfies (glueProgram kind) assignment := by
  intro row member
  rcases List.mem_map.mp member with ⟨indexed, indexedMember, rfl⟩
  exact glue_row_holds (armFor kind) assignment satisfied indexed indexedMember

private theorem digest_pin_facts
    (kind : ArmKind) (side : StateSide)
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : (armFor kind).Satisfied assignment) :
    ∀ pin ∈ (digestTrace kind side).pins,
      assignment pin.1 = pin.2 := by
  have normalizedSatisfies :
      Satisfies (normalizedDigestPinRows kind side) assignment := by
    intro row member
    exact glue_satisfies kind assignment satisfied row
      (digest_pin_row_mem_glue kind side row member)
  have pinRowsSatisfy :
      Satisfies (ConstantPins.rows (digestTrace kind side).pins)
        assignment :=
    (Poseidon2Normalized.satisfies_normalizeProgram
      (ConstantPins.rows (digestTrace kind side).pins) assignment).mp
        normalizedSatisfies
  exact ConstantPins.sound
    (programRows := ConstantPins.rows (digestTrace kind side).pins)
    (digest_trace_pins_canonical kind side)
    (rowsIncluded_self _)
    canonical one pinRowsSatisfy

/-- Satisfying exact arm rows accept the isolated state-digest trace. -/
theorem digest_trace_accepted
    (kind : ArmKind) (side : StateSide)
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : (armFor kind).Satisfied assignment) :
    (digestTrace kind side).Accepted assignment := by
  constructor
  · exact digest_pin_facts kind side assignment canonical one satisfied
  · intro call member
    apply poseidon2_call_refines (armFor kind) assignment canonical one
      satisfied call
    apply List.mem_of_mem_drop
    apply List.mem_of_mem_take
    simpa [digestTrace] using member

private theorem getD_mem_of_lt {alpha : Type} [Inhabited alpha]
    {entries : List alpha} {index : Nat} (bounded : index < entries.length) :
    entries.getD index default ∈ entries := by
  have member := List.getElem_mem (l := entries) bounded
  rwa [List.getElem_eq_getD default] at member

private theorem digest_pin_count
    (kind : ArmKind) (side : StateSide) :
    (digestTrace kind side).pins.length = 13 := by
  cases kind <;> cases side <;> rfl

private theorem initial_pin_shape
    (kind : ArmKind) (side : StateSide) :
    ∀ lane : Fin 8,
      (digestTrace kind side).pins.getD lane.val (0, 0) =
        ((digestStart kind side).cursor.lanes lane,
          initialStateValues.getD lane.val 0) := by
  intro lane
  cases kind <;> cases side <;> fin_cases lane <;> rfl

private theorem state_ext {left right : State}
    (lanes : left.lanes = right.lanes)
    (absorbed : left.absorbed = right.absorbed) : left = right := by
  cases left
  cases right
  simp_all

private theorem semantic_run_ext
    {left right : ColumnReplay.SemanticRun}
    (state : left.state = right.state)
    (digests : left.digests = right.digests) : left = right := by
  cases left
  cases right
  simp_all

/-- The accepted initial pin rows decode to the state computed from the exact
PiRLC family application-domain bytes. -/
theorem decoded_digest_start_eq_domain
    (kind : ArmKind) (side : StateSide)
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (accepted : (digestTrace kind side).Accepted assignment) :
    ColumnReplay.decodeRun assignment canonical (digestStart kind side) =
      { state := domainInitialState, digests := [] } := by
  apply semantic_run_ext
  · apply state_ext
    · funext lane
      apply Fin.ext
      have bounded : lane.val < (digestTrace kind side).pins.length := by
        rw [digest_pin_count kind side]
        exact Nat.lt_trans lane.isLt (by decide)
      have pinEqual := accepted.1
        ((digestTrace kind side).pins.getD lane.val default)
        (getD_mem_of_lt bounded)
      have pinShape := initial_pin_shape kind side lane
      have columnShape := congrArg Prod.fst pinShape
      have valueShape := congrArg Prod.snd pinShape
      change assignment ((digestStart kind side).cursor.lanes lane) =
        (domainInitialState.lanes lane).val
      change assignment
          (((digestTrace kind side).pins.getD lane.val (0, 0)).1) =
        ((digestTrace kind side).pins.getD lane.val (0, 0)).2 at pinEqual
      rw [columnShape, valueShape] at pinEqual
      exact pinEqual.trans (domain_initial_state_exact.1 lane).symm
    · apply Fin.ext
      exact domain_initial_state_exact.2.symm
  · rfl

def initialSemanticRun : ColumnReplay.SemanticRun where
  state := domainInitialState
  digests := []

/-- Independent semantic execution whose single output is the digest of the
exact selected 1045-field state preimage. -/
def semanticDigestRun
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (kind : ArmKind) (side : StateSide) : ColumnReplay.SemanticRun :=
  ColumnReplay.semanticExecute assignment canonical initialSemanticRun
    (digestOperations kind side)

def zeroDigest : Fin 4 → Field := fun _ => wordField 0

def stateDigest
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (kind : ArmKind) (side : StateSide) : Fin 4 → Field :=
  (semanticDigestRun assignment canonical kind side).digests.getD 0 zeroDigest

/-- Accepted rows refine the independent digest execution on the same
assignment. -/
theorem digest_execution_refines
    (kind : ArmKind) (side : StateSide)
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : (armFor kind).Satisfied assignment) :
    semanticDigestRun assignment canonical kind side =
      ColumnReplay.decodeRun assignment canonical (digestResult kind side) := by
  have accepted := digest_trace_accepted kind side assignment canonical one
    satisfied
  have refined := ColumnReplay.execute_sound canonical
    (digest_trace_pins_canonical kind side) one accepted
    (digest_execution kind side)
  rw [decoded_digest_start_eq_domain kind side assignment canonical accepted]
    at refined
  exact refined

/-- The four physical output columns equal the four independently computed
state-digest lanes. -/
theorem state_digest_refines
    (kind : ArmKind) (side : StateSide)
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : (armFor kind).Satisfied assignment) :
    stateDigest assignment canonical kind side =
      ColumnReplay.decodeDigest assignment canonical
        (digestOutputColumns kind side) := by
  have refined := digest_execution_refines kind side assignment canonical one
    satisfied
  have digestsEqual := congrArg ColumnReplay.SemanticRun.digests refined
  funext lane
  have selected := congrArg
    (fun digests => (digests.getD 0 zeroDigest) lane) digestsEqual
  simpa [stateDigest, semanticDigestRun, digestResult,
    ColumnReplay.decodeRun] using selected

/-! ## Exact shared-public digest-word projection -/

/-- One of the ten canonical-u64 calls selected by Rust's ordered public
output list. Word indices `0..3` are the after-state digest, `4..7` are the
before-state digest, and `8..9` are the global cursors. -/
def publicWordCall (kind : ArmKind) (word : Fin 10) : CanonicalCall :=
  let arm := armFor kind
  arm.canonicalCalls.getD
    (arm.publicWordCallIndices.getD word.val 0) default

/-- Source assignment column for one bit in the normalized public prefix. -/
def publicBitSourceColumn
    (kind : ArmKind) (word : Fin 10) (bit : Fin 64) : Nat :=
  CanonicalU64Recipe.bitColumn (publicWordCall kind word).layout bit.val

/-- Integer represented by one exact 64-bit public word in the raw arm
assignment. -/
def publicWordValue
    (assignment : Nat → Nat) (kind : ArmKind) (word : Fin 10) : Nat :=
  CanonicalU64RecipeSound.bitsValue assignment
    (publicWordCall kind word).layout

/-- Public word occupied by one full-XOut digest lane. -/
def xOutPublicWordIndex (side : StateSide) (lane : Fin 4) : Fin 10 :=
  match side with
  | .after => ⟨lane.val, by omega⟩
  | .before => ⟨4 + lane.val, by omega⟩

/-- Public word occupied by one global program cursor. -/
def cursorPublicWordIndex : StateSide → Fin 10
  | .before => ⟨8, by decide⟩
  | .after => ⟨9, by decide⟩

private theorem publicWordCall_mem
    (kind : ArmKind) (word : Fin 10) :
    publicWordCall kind word ∈ (armFor kind).canonicalCalls := by
  unfold publicWordCall
  apply getD_mem_of_lt
  cases kind <;> fin_cases word <;> decide

/-- The exact canonical-u64 rows identify each ordered public word with its
Rust-selected source field. -/
theorem public_word_refines
    (kind : ArmKind) (word : Fin 10)
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : (armFor kind).Satisfied assignment) :
    publicWordValue assignment kind word =
      assignment (publicWordCall kind word).fieldColumn := by
  have refined := canonical_call_refines (armFor kind) assignment canonical
    one satisfied (publicWordCall kind word) (publicWordCall_mem kind word)
  unfold publicWordValue
  rw [← refined.input_eq]
  simp [CanonicalCall.layout, lcEval,
    Nat.mod_eq_of_lt (canonical (publicWordCall kind word).fieldColumn)]

/-- Every source coordinate in the 640-bit public projection is Boolean. -/
theorem public_bit_binary
    (kind : ArmKind) (word : Fin 10) (bit : Fin 64)
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : (armFor kind).Satisfied assignment) :
    assignment (publicBitSourceColumn kind word bit) < 2 := by
  have refined := canonical_call_refines (armFor kind) assignment canonical
    one satisfied (publicWordCall kind word) (publicWordCall_mem kind word)
  have bounded := refined.bit bit.val bit.isLt
  simpa [publicBitSourceColumn, bitValue] using
    (Nat.lt_succ_iff.mpr bounded)

private theorem public_x_out_field_column_exact
    (kind : ArmKind) (side : StateSide) (lane : Fin 4) :
    (publicWordCall kind (xOutPublicWordIndex side lane)).fieldColumn =
      xOutDigestColumn kind side lane := by
  cases kind <;> cases side <;> fin_cases lane <;> rfl

/-- One physical full-XOut digest lane is exactly the canonical integer
represented by its ordered public 64-bit word. -/
theorem x_out_digest_public_word
    (kind : ArmKind) (side : StateSide) (lane : Fin 4)
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : (armFor kind).Satisfied assignment) :
    assignment (xOutDigestColumn kind side lane) =
      publicWordValue assignment kind (xOutPublicWordIndex side lane) := by
  rw [public_word_refines kind (xOutPublicWordIndex side lane) assignment
    canonical one satisfied, public_x_out_field_column_exact]

/-- The local family-state digest occupies the exact source columns consumed
by the phase envelope. It is not itself the outer XOut semantic digest. -/
theorem state_digest_phase_envelope_source
    (kind : ArmKind) (side : StateSide) (lane : Fin 4)
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : (armFor kind).Satisfied assignment) :
    (stateDigest assignment canonical kind side lane).val =
      assignment (phaseEnvelopeLocalSourceColumn kind side lane) := by
  have digestEqual := congrFun
    (state_digest_refines kind side assignment canonical one satisfied) lane
  have columnEqual :
      digestOutputColumns kind side lane =
        phaseEnvelopeLocalSourceColumn kind side lane := by
    cases kind <;> cases side <;> fin_cases lane <;> rfl
  exact (congrArg Fin.val digestEqual).trans (congrArg assignment columnEqual)

/-- Complete physical binding of the eight shared full-XOut digest words on
one satisfying raw arm assignment. -/
theorem shared_x_out_public_words_refine
    (kind : ArmKind)
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : (armFor kind).Satisfied assignment) :
    (∀ lane : Fin 4,
      assignment (xOutDigestColumn kind .after lane) =
        publicWordValue assignment kind
          (xOutPublicWordIndex .after lane)) ∧
      (∀ lane : Fin 4,
        assignment (xOutDigestColumn kind .before lane) =
          publicWordValue assignment kind
            (xOutPublicWordIndex .before lane)) := by
  exact ⟨
    fun lane => x_out_digest_public_word kind .after lane assignment canonical
      one satisfied,
    fun lane => x_out_digest_public_word kind .before lane assignment canonical
      one satisfied⟩

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyDigest
