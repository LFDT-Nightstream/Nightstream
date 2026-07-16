import Nightstream.Implementation.R1CS.Correspondence.PiCcsOutputDigest.Poseidon.EnvelopeSemantics
import Nightstream.Implementation.R1CS.Correspondence.PiCcsOutputDigest.Poseidon.Schedule
import Nightstream.Implementation.R1CS.Correspondence.PiCcsOutputDigest.Sis.ProductionBinding
import Nightstream.Implementation.R1CS.Core.ConstantPins

/-!
Exact terminal refinement from the typed `Pi_CCS` output serialization through
both SIS maps and the final Poseidon2 envelope.

Assurance tier: implementation/R1CS correspondence. Accepted owner pieces are
used only to reconstruct exact prefix pins, SIS equations, and the isolated
Poseidon2 sponge. The conclusion recomputes every digest lane.

Owns: semantic meaning of the ten envelope-prefix columns; equality of the 64
absorbed values with the independent envelope; exact 10,266-row sponge
soundness; and composition with the already-proved primary/compression SIS
maps.

Does not own: authority of dynamic `Pi_CCS` output columns; public-seed-to-map
coefficient conformance; native Rust/ChaCha parity; native Poseidon2 parity;
collision resistance; transcript placement; row necessity; row removal; or
cost totals.

Emits constraints: no.

Authority boundary: the four result columns are conclusions of accepted
constant, SIS, absorb, and permutation equations. They are never accepted as
a prover-carried digest. Upstream message authority remains an explicit open
premise in `SourceLayout`.

| Protocol | Phase | Constraint family | Theorem | Exact guarantee |
|---|---|---|---|---|
| `Pi_CCS` | output digest | envelope constants | `accepted_prefixValues` | ten columns equal the independent domain/shape prefix |
| `Pi_CCS` | output digest | absorbed input | `accepted_traceInputValues` | all 64 trace inputs equal prefix plus compression outputs |
| `Pi_CCS` | output digest | Poseidon2 sponge | `accepted_digestEnvelope` | four lanes equal the pure 17-round sponge execution |
| `Pi_CCS` | output digest | full composition | `accepted_composedDigest` | sponge input is the typed serialization after both exact SIS maps |
-/

namespace Nightstream.Implementation.R1CS.PiCcsOutputDigest.Poseidon.Refinement

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.OwnerCertificate

set_option maxRecDepth 1048576
set_option maxHeartbeats 8000000

def envelopePins : List (Nat × Nat) :=
  [(2543165, 40),
   (2543166, 30521782141150574),
   (2543167, 31069335676202596),
   (2543168, 33052923221205295),
   (2543169, 32421790864400748),
   (2543170, 28542674997834601),
   (2543171, 225321120883),
   (2543172, 5785229152774737749),
   (2543173, 6683),
   (2543174, 2)]

def prefixColumns : List Nat := envelopePins.map Prod.fst
def prefixValues : List Nat := envelopePins.map Prod.snd

theorem envelopePins_canonical : ConstantPins.ValuesCanonical envelopePins := by
  decide

theorem envelopePinRows_in_initialPiece :
    rowsIncluded (ConstantPins.rows envelopePins)
      Schedule.initialPiece.rows = true := by
  decide

theorem prefixColumns_eq_schedule :
    prefixColumns = Schedule.prefixColumns := by
  decide

theorem compressionColumns_eq_schedule :
    Sis.ProductionBinding.compressionBlock.outputColumns =
      Schedule.compressionColumns := by
  decide

theorem prefixValues_eq_semantics :
    prefixValues = EnvelopeSemantics.envelopePrefix := by
  rw [EnvelopeSemantics.envelopePrefix_eq]
  decide

private theorem initialPiece_satisfies
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (accepted :
      FPrimeFullHistoryTerminalPiCcsOutputMessageHashes.Accepted assignment) :
    Satisfies Schedule.initialPiece.rows assignment := by
  exact Payload.complete canonical one
    (accepted Schedule.initialPiece Schedule.initialPiece_mem_owner)

/-- Accepted prefix equations give the independent ten-field envelope prefix. -/
theorem accepted_prefixValues
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (accepted :
      FPrimeFullHistoryTerminalPiCcsOutputMessageHashes.Accepted assignment) :
    prefixColumns.map assignment = EnvelopeSemantics.envelopePrefix := by
  have facts := ConstantPins.sound envelopePins_canonical
    envelopePinRows_in_initialPiece canonical one
    (initialPiece_satisfies canonical one accepted)
  calc
    prefixColumns.map assignment = prefixValues := by
      unfold prefixColumns prefixValues
      rw [List.map_map]
      apply List.map_congr_left
      intro pin pinMember
      exact facts pin pinMember
    _ = EnvelopeSemantics.envelopePrefix := prefixValues_eq_semantics

private theorem initialHashRows_satisfy
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (accepted :
      FPrimeFullHistoryTerminalPiCcsOutputMessageHashes.Accepted assignment) :
    Satisfies (Schedule.initialPiece.rows.drop 10) assignment := by
  have full := initialPiece_satisfies canonical one accepted
  intro row member
  exact full row (List.mem_of_mem_drop member)

private theorem tailHashRows_satisfy
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (accepted :
      FPrimeFullHistoryTerminalPiCcsOutputMessageHashes.Accepted assignment) :
    Satisfies
      ((Schedule.tailPieces.map Piece.rows).flatten) assignment := by
  apply (satisfies_flatten_iff
    (Schedule.tailPieces.map Piece.rows) assignment).mpr
  intro rows rowsMember
  rcases List.mem_map.mp rowsMember with ⟨piece, pieceMember, rfl⟩
  exact Payload.complete canonical one
    (accepted piece (Schedule.tailPiece_mem_owner pieceMember))

private theorem hashRows_satisfy
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (accepted :
      FPrimeFullHistoryTerminalPiCcsOutputMessageHashes.Accepted assignment) :
    Satisfies Schedule.hashRows assignment := by
  intro row member
  rw [Schedule.hashRows, List.mem_append] at member
  rcases member with initial | tail
  · exact initialHashRows_satisfy canonical one accepted row initial
  · exact tailHashRows_satisfy canonical one accepted row tail

/-- The production trace absorbs exactly the independent envelope instantiated
with the 54 production compression-output values. -/
theorem accepted_traceInputValues
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (accepted :
      FPrimeFullHistoryTerminalPiCcsOutputMessageHashes.Accepted assignment) :
    Schedule.trace.inputColumns.map assignment =
      EnvelopeSemantics.envelope
        (Sis.ProductionBinding.compressionBlock.outputColumns.map assignment) := by
  rw [Schedule.trace_inputColumns,
    Schedule.envelopeColumns, List.map_append]
  rw [← prefixColumns_eq_schedule,
    ← compressionColumns_eq_schedule,
    accepted_prefixValues canonical one accepted]
  rfl

/-- Every accepted digest column is recomputed by the exact pure sponge over
the independent envelope values. -/
theorem accepted_digestEnvelope
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (accepted :
      FPrimeFullHistoryTerminalPiCcsOutputMessageHashes.Accepted assignment) :
    ∀ lane, lane < 4 →
      assignment (Schedule.digestColumns.getD lane 0) =
        Poseidon2Sponge.runValueRounds Schedule.trace.rounds
          (EnvelopeSemantics.envelope
            (Sis.ProductionBinding.compressionBlock.outputColumns.map assignment))
          (fun _ => 0) lane := by
  have traceValues := Poseidon2Sponge.trace_values_sound
    Schedule.trace_valid canonical one
    (hashRows_satisfy canonical one accepted)
  intro lane laneLt
  have value := traceValues lane laneLt
  rw [accepted_traceInputValues canonical one accepted] at value
  simpa [Schedule.trace_outputColumns] using value

/-- End-to-end equality from the independently typed terminal-output
serialization, through both exact SIS maps, into every recomputed digest lane.
The coefficient maps remain explicit until seed/Rust conformance is proved. -/
theorem accepted_composedDigest
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (accepted :
      FPrimeFullHistoryTerminalPiCcsOutputMessageHashes.Accepted assignment) :
    ∀ lane, lane < 4 →
      assignment (Schedule.digestColumns.getD lane 0) =
        Poseidon2Sponge.runValueRounds Schedule.trace.rounds
          (EnvelopeSemantics.envelope
            (Sis.Semantics.apply
              (Sis.Refinement.mapOfBlock Sis.ProductionBinding.compressionBlock)
              (Sis.Semantics.apply
                (Sis.Refinement.mapOfBlock Sis.ProductionBinding.primaryBlock)
                (Sis.ProductionBinding.serializedValues assignment canonical))))
          (fun _ => 0) lane := by
  have digest := accepted_digestEnvelope canonical one accepted
  have composed := Sis.ProductionBinding.accepted_composedOutputs
    prime canonical one accepted
  rw [composed] at digest
  exact digest

end Nightstream.Implementation.R1CS.PiCcsOutputDigest.Poseidon.Refinement
