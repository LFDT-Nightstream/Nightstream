import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Transcript.ColumnReplay
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Transcript.OutputDigestSemantics

/-!
Handwritten protocol operations for the PiRLC transcript.

Owns: the output-digest binding stream, raw-pair headers, scalar coordinates,
the four counter values `coordinate + block`, squeeze operations, and exact
ascending 15-scalar composition; plus their interpretation by the independent
transcript machine.

Does not own: generated pins/calls/columns, physical row satisfaction,
authority for the PiCCS digest, sampler selection, ring assembly, costs, or row
removal.

Emits constraints: no.

Authority boundary: protocol meaning is written here independently. A
generated trace can satisfy or fail this operation list; it cannot choose the
label, tags, counters, ordering, or challenge count.

| Stage path | Mathematical stream | Multiplicity |
|---|---|---:|
| `nifs.pi_rlc.challenge.transcript.output_bind` | label fields, count `4`, four authoritative digest fields | one |
| `nifs.pi_rlc.challenge.transcript.scalar_domain` | raw pair `[2, 0, coordinate]` | 15 |
| `nifs.pi_rlc.challenge.transcript.digest_block` | raw pair `[2, 1, coordinate + block]`, then `digest32` | 15 x 4 |
-/

namespace Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript.Operations

open Nightstream.Implementation.R1CS.PiRlcChallenge.TranscriptMachine
open Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript.ColumnReplay
open Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler
open Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler.ProductionSchedule

/-- Exact physical operations for `append_fields_raw([first, second])`. -/
def rawPair (first second : Nat) : List Operation :=
  [.pinned 2, .pinned first, .pinned second]

/-- One counter block, including the raw-pair domain and `digest32`. -/
def digestBlock (coordinate block : Nat) : List Operation :=
  rawPair 1 (coordinate + block) ++ [.digest]

/-- First `count` digest blocks in ascending block order. -/
def blocks (coordinate : Nat) : Nat → List Operation
  | 0 => []
  | count + 1 => blocks coordinate count ++ digestBlock coordinate count

/-- One complete scalar schedule: domain separation followed by four blocks. -/
def scalar (coordinate : Nat) : List Operation :=
  rawPair 0 coordinate ++ blocks coordinate 4

/-- First `count` scalar schedules in ascending coordinate order. -/
def sampler : Nat → List Operation
  | 0 => []
  | count + 1 => sampler count ++ scalar count

/-- Exact output-digest binding with independently specified label encoding. -/
def outputBind (digestColumns : Fin 4 → Nat) : List Operation :=
  OutputDigestSemantics.inputClaimsDigestLabelNats.map .pinned ++
    ([.pinned 4] ++ (List.ofFn digestColumns).map .external)

/-- Complete binding plus fixed-size scalar challenge schedule. -/
def full (digestColumns : Fin 4 → Nat) (challengeCount : Nat) : List Operation :=
  outputBind digestColumns ++ sampler challengeCount

/-! ## Independent value-level interpretation -/

theorem semanticExecute_append
    (assignment : Nat → Nat) (canonical : CanonicalAssignment assignment)
    (run : SemanticRun) (left right : List Operation) :
    semanticExecute assignment canonical run (left ++ right) =
      semanticExecute assignment canonical
        (semanticExecute assignment canonical run left) right := by
  induction left generalizing run with
  | nil => rfl
  | cons operation rest induction =>
      change semanticExecute assignment canonical
          (semanticStep assignment canonical run operation) (rest ++ right) = _
      rw [induction]
      rfl

/-- Semantic interpretation of the handwritten raw pair is exactly the
independent transcript machine's `appendRawPair`. -/
theorem semanticExecute_rawPair
    (assignment : Nat → Nat) (canonical : CanonicalAssignment assignment)
    (run : SemanticRun) (first second : Nat) :
    semanticExecute assignment canonical run (rawPair first second) =
      { run with state := appendRawPair run.state first second } := by
  rfl

/-- Value-level execution of one handwritten block. -/
def semanticDigestBlock (run : SemanticRun) (counter : Nat) : SemanticRun :=
  let result := TranscriptMachine.digest
    (appendRawPair run.state 1 counter)
  { state := result.1, digests := run.digests ++ [result.2] }

theorem semanticExecute_digestBlock
    (assignment : Nat → Nat) (canonical : CanonicalAssignment assignment)
    (run : SemanticRun) (coordinate block : Nat) :
    semanticExecute assignment canonical run (digestBlock coordinate block) =
      semanticDigestBlock run (coordinate + block) := by
  rw [digestBlock, semanticExecute_append,
    semanticExecute_rawPair]
  rfl

/-- Value-level execution of the first `count` digest blocks. -/
def semanticBlocks (coordinate : Nat) : Nat → SemanticRun → SemanticRun
  | 0, run => run
  | count + 1, run =>
      semanticDigestBlock (semanticBlocks coordinate count run)
        (coordinate + count)

theorem semanticExecute_blocks
    (assignment : Nat → Nat) (canonical : CanonicalAssignment assignment)
    (run : SemanticRun) (coordinate count : Nat) :
    semanticExecute assignment canonical run (blocks coordinate count) =
      semanticBlocks coordinate count run := by
  induction count with
  | zero => rfl
  | succ count induction =>
      rw [blocks, semanticExecute_append, induction,
        semanticExecute_digestBlock]
      rfl

/-- Raw four-lane digest jointly used to derive the 16 candidates of one
independent production block. -/
def blockDigest (entered : State) (coordinate block : Nat) : Fin 4 → Field :=
  (TranscriptMachine.digest
    (appendRawPair
      (stateBeforeBlock TranscriptMachine.machine entered coordinate block)
      1 (coordinate + block))).2

/-- Ordered raw digest list for the first `count` blocks. -/
def blockDigests (entered : State) (coordinate : Nat) :
    Nat → List (Fin 4 → Field)
  | 0 => []
  | count + 1 =>
      blockDigests entered coordinate count ++
        [blockDigest entered coordinate count]

theorem blockDigests_length
    (entered : State) (coordinate count : Nat) :
    (blockDigests entered coordinate count).length = count := by
  induction count with
  | zero => rfl
  | succ count induction =>
      simp [blockDigests, induction]

theorem blockDigests_getElem?
    (entered : State) (coordinate count block : Nat)
    (bounded : block < count) :
    (blockDigests entered coordinate count)[block]? =
      some (blockDigest entered coordinate block) := by
  induction count with
  | zero => omega
  | succ count induction =>
      rw [blockDigests]
      by_cases earlier : block < count
      · rw [List.getElem?_append_left]
        · exact induction earlier
        · simpa [blockDigests_length] using earlier
      · have last : block = count := by omega
        subst block
        rw [List.getElem?_append_right]
        · simp [blockDigests_length]
        · simp [blockDigests_length]

/-- Block execution reaches exactly the state used by the independent
production candidate schedule. -/
theorem semanticBlocks_state
    (coordinate count : Nat) (run : SemanticRun) :
    (semanticBlocks coordinate count run).state =
      stateBeforeBlock TranscriptMachine.machine run.state coordinate count := by
  induction count with
  | zero => rfl
  | succ count induction =>
      unfold semanticBlocks semanticDigestBlock
      rw [induction]
      rfl

/-- Block execution captures exactly one raw digest per production block. -/
theorem semanticBlocks_digests
    (coordinate count : Nat) (run : SemanticRun) :
    (semanticBlocks coordinate count run).digests =
      run.digests ++ blockDigests run.state coordinate count := by
  induction count with
  | zero => simp [semanticBlocks, blockDigests]
  | succ count induction =>
      unfold semanticBlocks semanticDigestBlock blockDigests
      rw [induction, semanticBlocks_state]
      simp only [blockDigest, List.append_assoc]

/-- Value-level execution of one complete scalar schedule. -/
def semanticScalar (run : SemanticRun) (coordinate : Nat) : SemanticRun :=
  semanticBlocks coordinate 4
    { run with state := enterScalar run.state coordinate }

theorem semanticExecute_scalar
    (assignment : Nat → Nat) (canonical : CanonicalAssignment assignment)
    (run : SemanticRun) (coordinate : Nat) :
    semanticExecute assignment canonical run (scalar coordinate) =
      semanticScalar run coordinate := by
  rw [scalar, semanticExecute_append, semanticExecute_rawPair,
    semanticExecute_blocks]
  rfl

def scalarDigests (initial : State) (coordinate : Nat) :
    List (Fin 4 → Field) :=
  blockDigests
    (enterScalar
      (stateAt TranscriptMachine.specification initial coordinate) coordinate)
    coordinate 4

theorem scalarDigests_length (initial : State) (coordinate : Nat) :
    (scalarDigests initial coordinate).length = 4 := by
  exact blockDigests_length _ _ _

theorem scalarDigests_getElem?
    (initial : State) (coordinate block : Nat) (bounded : block < 4) :
    (scalarDigests initial coordinate)[block]? =
      some (blockDigest
        (enterScalar
          (stateAt TranscriptMachine.specification initial coordinate)
          coordinate)
        coordinate block) := by
  exact blockDigests_getElem? _ _ 4 block bounded

/-- Ordered raw digest list for the first `count` scalar coordinates. -/
def batchDigests (initial : State) : Nat → List (Fin 4 → Field)
  | 0 => []
  | count + 1 =>
      batchDigests initial count ++ scalarDigests initial count

theorem batchDigests_length (initial : State) (count : Nat) :
    (batchDigests initial count).length = count * 4 := by
  induction count with
  | zero => rfl
  | succ count induction =>
      rw [batchDigests, List.length_append, induction,
        scalarDigests_length]
      omega

theorem batchDigests_getElem?
    (initial : State) (count rho block : Nat)
    (rhoBounded : rho < count) (blockBounded : block < 4) :
    (batchDigests initial count)[rho * 4 + block]? =
      some (blockDigest
        (enterScalar
          (stateAt TranscriptMachine.specification initial rho) rho)
        rho block) := by
  induction count with
  | zero => omega
  | succ count induction =>
      rw [batchDigests]
      by_cases earlier : rho < count
      · rw [List.getElem?_append_left]
        · exact induction earlier
        · rw [batchDigests_length]
          omega
      · have last : rho = count := by omega
        subst rho
        have afterPrefix :
            (batchDigests initial count).length ≤ count * 4 + block := by
          rw [batchDigests_length]
          omega
        rw [List.getElem?_append_right afterPrefix]
        have localIndex :
            count * 4 + block - (batchDigests initial count).length =
              block := by
          rw [batchDigests_length]
          omega
        rw [localIndex]
        exact scalarDigests_getElem? initial count block blockBounded

theorem batchDigests_getD
    (initial : State) (count rho block : Nat)
    (rhoBounded : rho < count) (blockBounded : block < 4)
    (fallback : Fin 4 → Field) :
    (batchDigests initial count).getD (rho * 4 + block) fallback =
      blockDigest
        (enterScalar
          (stateAt TranscriptMachine.specification initial rho) rho)
        rho block := by
  rw [List.getD_eq_getElem?_getD,
    batchDigests_getElem? initial count rho block rhoBounded blockBounded]
  rfl

/-- Value-level execution of the first `count` scalar schedules. -/
def semanticSampler : Nat → SemanticRun → SemanticRun
  | 0, run => run
  | count + 1, run => semanticScalar (semanticSampler count run) count

theorem semanticExecute_sampler
    (assignment : Nat → Nat) (canonical : CanonicalAssignment assignment)
    (run : SemanticRun) (count : Nat) :
    semanticExecute assignment canonical run (sampler count) =
      semanticSampler count run := by
  induction count with
  | zero => rfl
  | succ count induction =>
      rw [sampler, semanticExecute_append, induction,
        semanticExecute_scalar]
      rfl

/-- The semantic batch state is exactly the independently threaded production
sampler state. -/
theorem semanticSampler_state
    (count : Nat) (run : SemanticRun) :
    (semanticSampler count run).state =
      stateAt TranscriptMachine.specification run.state count := by
  induction count with
  | zero => rfl
  | succ count induction =>
      unfold semanticSampler semanticScalar
      rw [semanticBlocks_state, induction]
      rfl

/-- The semantic batch captures exactly the raw digests whose chunks feed the
independent production sampler. -/
theorem semanticSampler_digests
    (count : Nat) (run : SemanticRun) :
    (semanticSampler count run).digests =
      run.digests ++ batchDigests run.state count := by
  induction count with
  | zero => simp [semanticSampler, batchDigests]
  | succ count induction =>
      unfold semanticSampler batchDigests semanticScalar
      rw [semanticBlocks_digests, induction, semanticSampler_state]
      simp only [scalarDigests, List.append_assoc]

private theorem semanticExecute_pinnedList
    (assignment : Nat → Nat) (canonical : CanonicalAssignment assignment)
    (run : SemanticRun) (values : List Nat) :
    semanticExecute assignment canonical run (values.map .pinned) =
      { run with
        state := OutputDigestSemantics.absorbAll run.state
          (values.map wordField) } := by
  induction values generalizing run with
  | nil => rfl
  | cons value values induction =>
      change semanticExecute assignment canonical
          { run with state := absorbElem run.state (wordField value) }
          (values.map .pinned) = _
      rw [induction]
      rfl

private theorem semanticExecute_pinnedOne
    (assignment : Nat → Nat) (canonical : CanonicalAssignment assignment)
    (run : SemanticRun) (value : Nat) :
    semanticExecute assignment canonical run [.pinned value] =
      { run with state := absorbElem run.state (wordField value) } := by
  rfl

private theorem semanticExecute_externalList
    (assignment : Nat → Nat) (canonical : CanonicalAssignment assignment)
    (run : SemanticRun) (columns : List Nat) :
    semanticExecute assignment canonical run (columns.map .external) =
      { run with
        state := OutputDigestSemantics.absorbAll run.state
          (columns.map
            (CallRefinement.fieldAt assignment canonical)) } := by
  induction columns generalizing run with
  | nil => rfl
  | cons column columns induction =>
      change semanticExecute assignment canonical
          { run with
            state := absorbElem run.state
              (CallRefinement.fieldAt assignment canonical column) }
          (columns.map .external) = _
      rw [induction]
      rfl

private theorem absorbAll_append
    (state : State) (left right : List Field) :
    OutputDigestSemantics.absorbAll state (left ++ right) =
      OutputDigestSemantics.absorbAll
        (OutputDigestSemantics.absorbAll state left) right := by
  induction left generalizing state with
  | nil => rfl
  | cons value values induction =>
      change OutputDigestSemantics.absorbAll
          (absorbElem state value) (values ++ right) = _
      exact induction (absorbElem state value)

/-- The handwritten binding operations equal the independent label/count/
digest absorption semantics. -/
theorem semanticExecute_outputBind
    (assignment : Nat → Nat) (canonical : CanonicalAssignment assignment)
    (run : SemanticRun) (digestColumns : Fin 4 → Nat) :
    semanticExecute assignment canonical run (outputBind digestColumns) =
      { run with
        state := OutputDigestSemantics.appendInputClaimsDigest run.state
          (decodeDigest assignment canonical digestColumns) } := by
  rw [outputBind, semanticExecute_append, semanticExecute_pinnedList,
    semanticExecute_append, semanticExecute_pinnedOne,
    semanticExecute_externalList]
  unfold OutputDigestSemantics.appendInputClaimsDigest
  rw [OutputDigestSemantics.appendSequence, absorbAll_append,
    absorbAll_append]
  congr 2

/-- Complete independent interpretation of the binding and scalar schedule. -/
theorem semanticExecute_full
    (assignment : Nat → Nat) (canonical : CanonicalAssignment assignment)
    (run : SemanticRun) (digestColumns : Fin 4 → Nat)
    (challengeCount : Nat) :
    semanticExecute assignment canonical run
        (full digestColumns challengeCount) =
      semanticSampler challengeCount
        { run with
          state := OutputDigestSemantics.appendInputClaimsDigest run.state
            (decodeDigest assignment canonical digestColumns) } := by
  rw [full, semanticExecute_append, semanticExecute_outputBind,
    semanticExecute_sampler]

end Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript.Operations
