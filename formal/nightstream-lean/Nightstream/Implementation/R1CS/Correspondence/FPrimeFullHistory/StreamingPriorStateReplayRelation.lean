import Nightstream.Implementation.Nebula.Production.Carrier.StreamingFPrimeProgram
import Nightstream.Implementation.Nebula.Production.Carrier.StreamingSuccessorStateBinding

/-!
Contract: typed meaning of one bounded prior-state replay work item.

Owns the exact 1,024-field full arm, the 522-field final arm with zero
padding, the ten-field Poseidon2 continuation state, and the final comparison
with an explicit verifier-owned prior-state digest. It does not own generated
rows, source placement of that digest, or Poseidon2 collision resistance.

Assurance tier: model-level.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPriorStateReplayRelation

open Nightstream.Implementation.Nebula
open Nightstream.Implementation.Nebula.ProductionStreamingFPrimeProgram
open Nightstream.Implementation.Nebula.ProductionSuccessorStateStreaming
open Nightstream.Implementation.R1CS.Canonical

abbrev State := ProductionSuccessorStateStreaming.State
abbrev ReplayState := ProductionSuccessorStateStreaming.ReplayState
abbrev Digest := Fin 4 -> Nat

/-- The two physical shapes needed by the 95,754-field production frame. -/
inductive ArmKind where
  | full
  | final
  deriving DecidableEq, Inhabited, Repr

def chunkWidth : Nat := 1024

def fullChunks : Nat := 93

def finalFields : Nat := 522

def chunkCount : Nat := fullChunks + 1

def frameFields : Nat := fullChunks * chunkWidth + finalFields

def firstProgramCursor : Nat := 1

def activeFields : ArmKind -> Nat
  | .full => chunkWidth
  | .final => finalFields

def kindAt (index : Nat) : ArmKind :=
  if index + 1 = chunkCount then .final else .full

theorem production_geometry :
    fullChunks = 93 /\ finalFields = 522 /\ chunkCount = 94 /\
      frameFields = 95754 := by
  decide

theorem final_padding_fields : chunkWidth - finalFields = 502 := by
  decide

theorem kindAt_zero : kindAt 0 = .full := by
  decide

theorem kindAt_last : kindAt fullChunks = .final := by
  decide

/-- Four protocol digest lanes projected from a domain-gated duplex state. -/
def outputDigest (state : State) : Digest := fun lane =>
  state.lanes ⟨lane.val, by
    change lane.val < 8
    omega⟩

/-- Concrete profiles instantiate this function with the protocol-bound
Poseidon2 digest of the ten explicit continuation fields. -/
structure Semantics where
  stateDigest : List Nat -> Digest

/-- The shared public values used by this selected work item. -/
structure PublicEnvelope where
  beforeLocalStateDigest : Digest
  afterLocalStateDigest : Digest
  beforeProgramCursor : Nat
  afterProgramCursor : Nat

/-- The final arm alone closes the complete prior-state replay. The target is
an explicit verifier-owned value; this relation does not accept a self-chosen
digest as authority. -/
noncomputable def FinalChecks
    (kind : ArmKind) (after : ReplayState) (chunk : List Nat)
    (target : Digest) : Prop :=
  match kind with
  | .full => True
  | .final =>
      chunk.drop finalFields =
          List.replicate (chunkWidth - finalFields) 0 /\
        after.cursor = frameFields /\
        outputDigest
            (Poseidon2Duplex.gate ProductPoseidon2.constants
              after.transcript) = target

/-- Exact local relation for one verifier-selected prior-state replay item. -/
noncomputable def Holds
    (semantics : Semantics) (kind : ArmKind) (item : WorkItem)
    (before after : ReplayState) (chunk : List Nat) (target : Digest)
    (envelope : PublicEnvelope) : Prop :=
  item.phase = .priorStateReplay /\
    item.index < chunkCount /\
    kind = kindAt item.index /\
    chunk.length = chunkWidth /\
    before.transcript.absorbed = 0 /\
    before.cursor = item.index * chunkWidth /\
    after = before.advance (chunk.take (activeFields kind)) /\
    envelope.beforeLocalStateDigest =
      semantics.stateDigest (persistentFields before) /\
    envelope.afterLocalStateDigest =
      semantics.stateDigest (persistentFields after) /\
    envelope.beforeProgramCursor = firstProgramCursor + item.index /\
    envelope.afterProgramCursor = envelope.beforeProgramCursor + 1 /\
    FinalChecks kind after chunk target

theorem workItem_exact
    {semantics : Semantics} {kind : ArmKind} {item : WorkItem}
    {before after : ReplayState} {chunk : List Nat} {target : Digest}
    {envelope : PublicEnvelope}
    (holds : Holds semantics kind item before after chunk target envelope) :
    item.phase = .priorStateReplay /\ item.index < chunkCount /\
      kind = kindAt item.index :=
  ⟨holds.1, holds.2.1, holds.2.2.1⟩

theorem cursor_exact
    {semantics : Semantics} {kind : ArmKind} {item : WorkItem}
    {before after : ReplayState} {chunk : List Nat} {target : Digest}
    {envelope : PublicEnvelope}
    (holds : Holds semantics kind item before after chunk target envelope) :
    before.cursor = item.index * chunkWidth /\
      envelope.beforeProgramCursor = firstProgramCursor + item.index /\
      envelope.afterProgramCursor = firstProgramCursor + item.index + 1 := by
  rcases holds with
    ⟨_, _, _, _, _, beforeCursor, _, _, _, beforeProgram, afterProgram, _⟩
  refine ⟨beforeCursor, beforeProgram, ?_⟩
  omega

theorem transition_exact
    {semantics : Semantics} {kind : ArmKind} {item : WorkItem}
    {before after : ReplayState} {chunk : List Nat} {target : Digest}
    {envelope : PublicEnvelope}
    (holds : Holds semantics kind item before after chunk target envelope) :
    after = before.advance (chunk.take (activeFields kind)) :=
  holds.2.2.2.2.2.2.1

theorem public_digests_exact
    {semantics : Semantics} {kind : ArmKind} {item : WorkItem}
    {before after : ReplayState} {chunk : List Nat} {target : Digest}
    {envelope : PublicEnvelope}
    (holds : Holds semantics kind item before after chunk target envelope) :
    envelope.beforeLocalStateDigest =
        semantics.stateDigest (persistentFields before) /\
      envelope.afterLocalStateDigest =
        semantics.stateDigest (persistentFields after) :=
  ⟨holds.2.2.2.2.2.2.2.1, holds.2.2.2.2.2.2.2.2.1⟩

theorem final_closes_target
    {semantics : Semantics} {item : WorkItem}
    {before after : ReplayState} {chunk : List Nat} {target : Digest}
    {envelope : PublicEnvelope}
    (holds : Holds semantics .final item before after chunk target envelope) :
    chunk.drop finalFields =
        List.replicate (chunkWidth - finalFields) 0 /\
      after.cursor = frameFields /\
      outputDigest
          (Poseidon2Duplex.gate ProductPoseidon2.constants
            after.transcript) = target := by
  exact holds.2.2.2.2.2.2.2.2.2.2.2

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPriorStateReplayRelation
