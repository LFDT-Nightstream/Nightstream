import Nightstream.Implementation.R1CS.Artifacts.FPrimeRecursive.PiRlcChallenge.SamplerLayout
import Nightstream.Implementation.R1CS.Correspondence.FPrimeRecursive.PiRlcChallenge.Transcript.Schedule
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Transcript.Operations

/-!
Three-matrix diagnostic PiRLC transcript replay.

Owns: instantiation of the handwritten output-bind plus 15-by-4 operation
schedule against the active 291-pin/78-call trace; kernel-reduced physical
execution; exact post-bind/final boundaries; and composition with the generic
semantic replay theorem.

Does not own: authority for the incoming PiCCS state or four digest fields,
canonical-u64/sampler row semantics, challenge ring assembly, cryptographic
bad-event bounds, costs, or row removal.

Emits constraints: no.

Authority boundary: generated columns instantiate physical inputs only.
Protocol labels, tags, coordinates, counters, and ordering come from the
handwritten `Operations` module. The four digest columns remain explicitly
external until a PiCCS authority theorem binds them to recomputed outputs.

| Stage path | Exact checked result | Assurance |
|---|---|---|
| `nifs.pi_rlc.challenge.transcript.output_bind` | 6 pins, 2 calls, cursor 0 to 2 | artifact-checked physical replay |
| `nifs.pi_rlc.challenge.transcript.rhos` | 285 pins, 76 calls, 60 four-lane digests | artifact-checked physical replay |
| `nifs.pi_rlc.challenge.transcript.field_outputs` | every captured lane is the active sampler field column | artifact-checked cross-layout identity |
| `nifs.pi_rlc.challenge.transcript.input_authority` | incoming state and four digest fields equal verifier-derived values | explicit upstream premise |
-/

namespace Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcChallenge.Transcript.Replay

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.PiRlcChallenge.TranscriptMachine
open Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript.ColumnReplay

namespace Layout

abbrev piCcsOutputDigestInputColumns :=
  FPrimeRecursivePiRlcChallenge.TranscriptLayout.piCcsOutputDigestInputColumns
abbrev initialStateColumns :=
  FPrimeRecursivePiRlcChallenge.TranscriptLayout.initialStateColumns
abbrev postBindStateColumns :=
  FPrimeRecursivePiRlcChallenge.TranscriptLayout.postBindStateColumns
abbrev finalStateColumns :=
  FPrimeRecursivePiRlcChallenge.TranscriptLayout.finalStateColumns

end Layout

namespace Sampler

abbrev fieldColumn := FPrimeRecursivePiRlcChallenge.SamplerLayout.fieldColumn

end Sampler

namespace ProtocolOps

abbrev outputBind := PiRlcChallenge.Transcript.Operations.outputBind
abbrev sampler := PiRlcChallenge.Transcript.Operations.sampler
abbrev full := PiRlcChallenge.Transcript.Operations.full
abbrev semanticSampler := PiRlcChallenge.Transcript.Operations.semanticSampler

end ProtocolOps

/-- Total finite lane view of one generated eight-column boundary. -/
def stateColumns (columns : List Nat) : Fin width → Nat :=
  fun lane => columns.getD lane.val 0

/-- Four separately authoritative PiCCS output-digest columns. -/
def digestInputColumns : Fin 4 → Nat :=
  fun lane => Layout.piCcsOutputDigestInputColumns.getD lane.val 0

def initialCursor : Cursor where
  lanes := stateColumns Layout.initialStateColumns
  absorbed := ⟨0, by decide⟩
  nextPin := 0
  nextCall := 0

def initialRun : Run where
  cursor := initialCursor
  digests := []

def bindOperations : List Operation :=
  ProtocolOps.outputBind digestInputColumns

def samplerOperations : List Operation :=
  ProtocolOps.sampler 15

def activeOperations : List Operation :=
  ProtocolOps.full digestInputColumns 15

/-- Result computed by the physical interpreter. The following theorem proves
that the fallback is unreachable for the active artifact. -/
def bindResult : Run :=
  (execute Schedule.trace initialRun bindOperations).getD initialRun

def activeResult : Run :=
  (execute Schedule.trace initialRun activeOperations).getD initialRun

theorem bind_execution :
    execute Schedule.trace initialRun bindOperations = some bindResult := by
  set_option maxRecDepth 100000 in
    rfl

theorem active_execution :
    execute Schedule.trace initialRun activeOperations = some activeResult := by
  set_option maxRecDepth 100000 in
    rfl

/-- Closed post-bind counts and exact mixed boundary state. -/
theorem bind_result_facts :
    bindResult.cursor.nextPin = 6 ∧
      bindResult.cursor.nextCall = 2 ∧
      bindResult.cursor.absorbed.val = 2 ∧
      bindResult.digests.length = 0 ∧
      (∀ lane : Fin width,
        bindResult.cursor.lanes lane =
          Layout.postBindStateColumns.getD lane.val 0) := by
  set_option maxRecDepth 100000 in
    decide

/-- Closed full-profile counts and final state. -/
theorem active_result_facts :
    activeResult.cursor.nextPin = 291 ∧
      activeResult.cursor.nextCall = 78 ∧
      activeResult.cursor.absorbed.val = 0 ∧
      activeResult.digests.length = 60 ∧
      (∀ lane : Fin width,
        activeResult.cursor.lanes lane =
          Layout.finalStateColumns.getD lane.val 0) := by
  set_option maxRecDepth 100000 in
    decide

/-- Every captured digest lane is exactly the field column consumed by the
active canonical-u64 sampler leaves. This is the cross-layout identity that
the generated facades intentionally do not assume. -/
theorem digest_columns_match_sampler :
    ∀ (rho : Fin 15) (block lane : Fin 4),
      (activeResult.digests.getD (rho.val * 4 + block.val)
        (fun _ => 0)) lane =
        Sampler.fieldColumn rho block lane := by
  set_option maxRecDepth 100000 in
    decide

/-- Semantic entry state decoded from the prior PiCCS transcript boundary. -/
def entryState
    (assignment : Nat → Nat) (canonical : CanonicalAssignment assignment) :
    State :=
  decodeCursor assignment canonical initialCursor

/-- Semantic four-field digest read from the explicit external columns. -/
def inputDigest
    (assignment : Nat → Nat) (canonical : CanonicalAssignment assignment) :
    Fin 4 → Field :=
  decodeDigest assignment canonical digestInputColumns

/-- Semantic state physically reached after output-digest binding. -/
def postBindState
    (assignment : Nat → Nat) (canonical : CanonicalAssignment assignment) :
    State :=
  decodeCursor assignment canonical bindResult.cursor

/-- Semantic final state physically reached after all fifteen scalars. -/
def finalState
    (assignment : Nat → Nat) (canonical : CanonicalAssignment assignment) :
    State :=
  decodeCursor assignment canonical activeResult.cursor

/-- Explicit authority boundary for the state and digest entering the active
PiRLC transcript. This packages upstream conclusions without deriving either
one from generated column identity. -/
structure InputsBound
    (assignment : Nat → Nat) (canonical : CanonicalAssignment assignment)
    (postPiCcsState : State) (recomputedDigest : Fin 4 → Field) : Prop where
  entryState_eq : entryState assignment canonical = postPiCcsState
  inputDigest_eq : inputDigest assignment canonical = recomputedDigest

/-- Assemble the input-authority boundary from separately proved physical
state and digest-lane equalities. The future PiCCS bridge must supply both. -/
theorem inputsBound_of_boundary_equalities
    {assignment : Nat → Nat}
    {canonical : CanonicalAssignment assignment}
    {postPiCcsState : State} {recomputedDigest : Fin 4 → Field}
    (stateBound : entryState assignment canonical = postPiCcsState)
    (digestBound : ∀ lane,
      PiRlcChallenge.Transcript.CallRefinement.fieldAt assignment canonical
          (digestInputColumns lane) =
        recomputedDigest lane) :
    InputsBound assignment canonical postPiCcsState recomputedDigest := by
  refine ⟨stateBound, ?_⟩
  funext lane
  simpa only [inputDigest, decodeDigest] using digestBound lane

/-- Accepted physical rows force the independently specified output binding. -/
theorem accepted_binds_outputDigest
    {assignment : Nat → Nat}
    (canonical : CanonicalAssignment assignment)
    (one : assignment 0 = 1)
    (accepted : Schedule.trace.Accepted assignment) :
    PiRlcChallenge.Transcript.OutputDigestSemantics.appendInputClaimsDigest
        (entryState assignment canonical) (inputDigest assignment canonical) =
      postBindState assignment canonical := by
  have replayed := execute_sound canonical Schedule.pinValuesCanonical one
    accepted bind_execution
  have semantic :=
    PiRlcChallenge.Transcript.Operations.semanticExecute_outputBind
      assignment canonical (decodeRun assignment canonical initialRun)
      digestInputColumns
  have stateEquality :=
    congrArg SemanticRun.state (semantic.symm.trans replayed)
  simpa only [entryState, inputDigest, postBindState, initialRun, decodeRun]
    using stateEquality

/-- Accepted physical rows refine the complete independent binding and
15-scalar semantic execution. -/
theorem accepted_replays_all
    {assignment : Nat → Nat}
    (canonical : CanonicalAssignment assignment)
    (one : assignment 0 = 1)
    (accepted : Schedule.trace.Accepted assignment) :
    PiRlcChallenge.Transcript.Operations.semanticSampler 15
        { decodeRun assignment canonical initialRun with
          state :=
            PiRlcChallenge.Transcript.OutputDigestSemantics.appendInputClaimsDigest
            (entryState assignment canonical) (inputDigest assignment canonical) } =
      decodeRun assignment canonical activeResult := by
  have replayed := execute_sound canonical Schedule.pinValuesCanonical one
    accepted active_execution
  have semantic := PiRlcChallenge.Transcript.Operations.semanticExecute_full
    assignment canonical (decodeRun assignment canonical initialRun)
    digestInputColumns 15
  simpa only [entryState, inputDigest] using semantic.symm.trans replayed

/-- The complete physical digest list equals the independent production
batch's raw Poseidon2 digest list. -/
theorem accepted_batchDigests
    {assignment : Nat → Nat}
    (canonical : CanonicalAssignment assignment)
    (one : assignment 0 = 1)
    (accepted : Schedule.trace.Accepted assignment) :
    PiRlcChallenge.Transcript.Operations.batchDigests
        (postBindState assignment canonical) 15 =
      activeResult.digests.map (decodeDigest assignment canonical) := by
  have replayed := accepted_replays_all canonical one accepted
  have digestsEqual := congrArg SemanticRun.digests replayed
  rw [PiRlcChallenge.Transcript.Operations.semanticSampler_digests]
    at digestsEqual
  have binding := accepted_binds_outputDigest canonical one accepted
  rw [binding] at digestsEqual
  set_option maxRecDepth 100000 in
    change ([] : List (Fin 4 → Field)) ++
        PiRlcChallenge.Transcript.Operations.batchDigests
          (postBindState assignment canonical) 15 =
        activeResult.digests.map (decodeDigest assignment canonical)
      at digestsEqual
  rw [List.nil_append] at digestsEqual
  exact digestsEqual

private def zeroColumns : Fin 4 → Nat := fun _ => 0

private def zeroDigest : Fin 4 → Field :=
  fun _ => ⟨0, by decide⟩

private theorem map_getD_of_lt
    {α β : Type} (values : List α) (transform : α → β)
    (index : Nat) (bounded : index < values.length)
    (fallbackInput : α) (fallbackOutput : β) :
    (values.map transform).getD index fallbackOutput =
      transform (values.getD index fallbackInput) := by
  simp [List.getD_eq_getElem?_getD, bounded]

/-- Each active sampler field lane is the independently replayed raw digest
lane for the same scalar and counter block. -/
theorem accepted_fieldDigest
    {assignment : Nat → Nat}
    (canonical : CanonicalAssignment assignment)
    (one : assignment 0 = 1)
    (accepted : Schedule.trace.Accepted assignment)
    (rho : Fin 15) (block lane : Fin 4) :
    PiRlcChallenge.Transcript.CallRefinement.fieldAt assignment canonical
        (Sampler.fieldColumn rho block lane) =
      PiRlcChallenge.Transcript.Operations.blockDigest
        (enterScalar
          (Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler.stateAt
            specification
            (postBindState assignment canonical) rho.val)
          rho.val)
        rho.val block.val lane := by
  let index := rho.val * 4 + block.val
  have physicalBound : index < activeResult.digests.length := by
    have length := active_result_facts.2.2.2.1
    rw [length]
    have rhoLt := rho.isLt
    have blockLt := block.isLt
    omega
  have batchEquality := accepted_batchDigests canonical one accepted
  have atIndex := congrArg
    (fun digests => digests.getD index zeroDigest) batchEquality
  dsimp only [index] at atIndex
  rw [PiRlcChallenge.Transcript.Operations.batchDigests_getD
      (postBindState assignment canonical) 15 rho.val block.val
      rho.isLt block.isLt zeroDigest,
    map_getD_of_lt activeResult.digests
      (decodeDigest assignment canonical) index physicalBound
      zeroColumns zeroDigest]
    at atIndex
  have atLane := congrArg (fun digest => digest lane) atIndex
  have physicalColumns := digest_columns_match_sampler rho block lane
  have decodedColumn :
      decodeDigest assignment canonical
          (activeResult.digests.getD index zeroColumns) lane =
        PiRlcChallenge.Transcript.CallRefinement.fieldAt assignment canonical
          (Sampler.fieldColumn rho block lane) := by
    exact congrArg
      (PiRlcChallenge.Transcript.CallRefinement.fieldAt assignment canonical)
      physicalColumns
  exact decodedColumn.symm.trans atLane.symm

/-- The final physical boundary is the independently threaded state after all
fifteen scalar sources. -/
theorem accepted_finalState
    {assignment : Nat → Nat}
    (canonical : CanonicalAssignment assignment)
    (one : assignment 0 = 1)
    (accepted : Schedule.trace.Accepted assignment) :
    Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler.stateAt
        specification (postBindState assignment canonical)
        15 =
      finalState assignment canonical := by
  have replayed := accepted_replays_all canonical one accepted
  have stateEquality := congrArg SemanticRun.state replayed
  rw [PiRlcChallenge.Transcript.Operations.semanticSampler_state]
    at stateEquality
  have binding := accepted_binds_outputDigest canonical one accepted
  rw [binding] at stateEquality
  simpa only [finalState, decodeRun] using stateEquality

/-- Complete active transcript refinement, with incoming state/digest authority
left explicit for the PiCCS handoff. -/
structure Refines
    (assignment : Nat → Nat) (canonical : CanonicalAssignment assignment) :
    Prop where
  outputBind :
    PiRlcChallenge.Transcript.OutputDigestSemantics.appendInputClaimsDigest
        (entryState assignment canonical) (inputDigest assignment canonical) =
      postBindState assignment canonical
  fieldDigest : ∀ (rho : Fin 15) (block lane : Fin 4),
    PiRlcChallenge.Transcript.CallRefinement.fieldAt assignment canonical
        (Sampler.fieldColumn rho block lane) =
      PiRlcChallenge.Transcript.Operations.blockDigest
        (enterScalar
          (Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler.stateAt
            specification
            (postBindState assignment canonical) rho.val)
          rho.val)
        rho.val block.val lane
  finalState :
    Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler.stateAt
        specification (postBindState assignment canonical)
        15 =
      Replay.finalState assignment canonical

theorem accepted_refines
    {assignment : Nat → Nat}
    (canonical : CanonicalAssignment assignment)
    (one : assignment 0 = 1)
    (accepted : Schedule.trace.Accepted assignment) :
    Refines assignment canonical :=
  { outputBind := accepted_binds_outputDigest canonical one accepted
    fieldDigest := accepted_fieldDigest canonical one accepted
    finalState := accepted_finalState canonical one accepted }

end Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcChallenge.Transcript.Replay
