import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.Primitives

/-!
Verifier-authority binding prefix for the production-shaped `Pi_CCS`
transcript.

Assurance tier: executable implementation semantics. The five raw messages
are stated as typed mathematical inputs and exact domain-tagged field lists;
no generated column or Rust helper defines their order here.

Owns: the header bundle, recomputed instance digest, running-input count,
checked-parent handle, their five exact raw messages, and the deterministic
state after that prefix.

Does not own: derivation of the header bundle from public parameters,
derivation of the instance digest from accepted claims, validation of the
checked parent, the incoming outer transcript state, later challenges,
SumCheck, R1CS lowering, cost totals, or row removal.

Emits constraints: no.

Authority boundary: the fields in `Input` are obligations supplied by earlier
semantic layers. This module only proves what their exact transcript binding
means. A later refinement theorem must derive each field from verifier-owned
inputs before this prefix can authorize challenges.

| Protocol | Phase | Constraint family | Exact mathematical message |
|---|---|---|---|
| `Pi_CCS` | public header | `headerFields` | `[11, header[0..4]]` |
| `Pi_CCS` | public instance | `instanceFields` | `[12, instanceDigest[0..4]]` |
| `Pi_CCS` | running domain | `runningDomainFields` | `[4]` |
| `Pi_CCS` | running count | `runningCountFields` | `[5, count]` |
| `Pi_CCS` | parent authority | `parentHandleFields` | `[13, checkedParent[0..4]]` |
-/

namespace Nightstream.Implementation.R1CS.PiCcsTranscript.Binding

open Nightstream.Implementation.R1CS.PiRlcChallenge.TranscriptMachine
open Nightstream.Implementation.R1CS.PiCcsTranscript.Primitives

set_option maxHeartbeats 1000000

/-- Verifier-visible values bound before any `Pi_CCS` challenge is sampled. -/
structure Input where
  headerBundle : Fin 4 -> Field
  instanceDigest : Fin 4 -> Field
  runningCount : Nat
  checkedParentHandle : Fin 4 -> Field

private def fourFields (values : Fin 4 -> Field) : List Field :=
  [values ⟨0, by decide⟩, values ⟨1, by decide⟩,
   values ⟨2, by decide⟩, values ⟨3, by decide⟩]

/-- Header-bundle raw payload with its independent domain tag. -/
def headerFields (input : Input) : List Field :=
  wordField 11 :: fourFields input.headerBundle

/-- Public-instance digest raw payload with its independent domain tag. -/
def instanceFields (input : Input) : List Field :=
  wordField 12 :: fourFields input.instanceDigest

/-- Running-input domain separator. -/
def runningDomainFields : List Field :=
  [wordField 4]

/-- Running-input cardinality header. -/
def runningCountFields (input : Input) : List Field :=
  [wordField 5, wordField input.runningCount]

/-- Checked-parent authority handle. -/
def parentHandleFields (input : Input) : List Field :=
  wordField 13 :: fourFields input.checkedParentHandle

/-- State after the header bundle. Each phase is named so later R1CS
refinement can stop at a meaningful boundary without unfolding Poseidon2. -/
def afterHeader (initial : State) (input : Input) : State :=
  appendRaw initial (headerFields input)

/-- State after the public-instance digest. -/
def afterInstance (initial : State) (input : Input) : State :=
  appendRaw (afterHeader initial input) (instanceFields input)

/-- State after the running-input domain separator. -/
def afterRunningDomain (initial : State) (input : Input) : State :=
  appendRaw (afterInstance initial input) runningDomainFields

/-- State after the running-input count. -/
def afterRunningCount (initial : State) (input : Input) : State :=
  appendRaw (afterRunningDomain initial input) (runningCountFields input)

/-- Execute the complete five-message verifier-binding prefix. -/
def run (initial : State) (input : Input) : State :=
  appendRaw (afterRunningCount initial input) (parentHandleFields input)

/-- Named phase trace used by later refinement. It makes every state boundary
available without allowing callers to supply one independently. -/
structure Trace where
  afterHeader : State
  afterInstance : State
  afterRunningDomain : State
  afterRunningCount : State
  afterParentHandle : State

/-- Compute every prefix boundary from one initial state and one typed input. -/
def trace (initial : State) (input : Input) : Trace :=
  { afterHeader := afterHeader initial input
    afterInstance := afterInstance initial input
    afterRunningDomain := afterRunningDomain initial input
    afterRunningCount := afterRunningCount initial input
    afterParentHandle := run initial input }

@[simp] theorem trace_afterParentHandle (initial : State) (input : Input) :
    (trace initial input).afterParentHandle = run initial input := by
  simp only [trace]

/-- The checked-parent handle is the final authority-bearing payload before
challenge derivation. This equality exposes the exact predecessor state for
row-level schedule refinement. -/
theorem run_eq_appendParentHandle (initial : State) (input : Input) :
    run initial input =
      appendRaw (trace initial input).afterRunningCount
        (parentHandleFields input) := by
  simp only [trace, run]

end Nightstream.Implementation.R1CS.PiCcsTranscript.Binding
