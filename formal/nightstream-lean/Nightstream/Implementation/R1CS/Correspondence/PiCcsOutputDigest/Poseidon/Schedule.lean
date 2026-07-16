import Nightstream.Implementation.R1CS.Core.Poseidon2Sponge
import Nightstream.Implementation.R1CS.Ownership.FPrimeFullHistory.FPrimeFullHistoryTerminalPiCcsOutputMessageHashesArtifact

/-!
Exact terminal-profile structure of the `Pi_CCS` output-digest Poseidon2
envelope.

Assurance tier: implementation/R1CS structural correspondence. This file
separates the final sponge from the preceding 6,683 canonical encodings and
two SIS maps, then checks its exact production row slice.

Owns: the ten envelope-prefix columns; 54 compression-output columns; sixteen
four-field absorb rounds; one padding round; local row/column geometry; the
four output columns; and an exact structural certificate for the 10,266 rows.

Does not own: prefix constant meaning, compression-output meaning, Poseidon2
native parity, upstream `Pi_CCS` authority, transcript placement, collision
resistance, row necessity, row removal, or cost totals.

Emits constraints: no; it classifies existing emitted constraints.

Authority boundary: generated owner pieces identify exact rows only. No
column receives message or digest meaning in this module.

| Protocol | Phase | Constraint family | Exact structural obligation |
|---|---|---|---|
| `Pi_CCS` | output digest | envelope prefix | ten fixed-source columns precede all compression coordinates |
| `Pi_CCS` | output digest | absorb rounds | 64 fields are consumed by sixteen rate-four rounds |
| `Pi_CCS` | output digest | padding | one `state[0] += 1` row and one final permutation |
| `Pi_CCS` | output digest | output lanes | first four final-state lanes are columns `2553433..2553436` |
-/

namespace Nightstream.Implementation.R1CS.PiCcsOutputDigest.Poseidon.Schedule

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.OwnerCertificate
open Nightstream.Implementation.R1CS.Poseidon2Sponge

set_option maxRecDepth 1048576
set_option maxHeartbeats 8000000

def rangeFrom (start count : Nat) : List Nat :=
  (List.range count).map (start + ·)

def prefixColumns : List Nat := rangeFrom 2543165 10
def compressionColumns : List Nat := rangeFrom 2529935 54
def envelopeColumns : List Nat := prefixColumns ++ compressionColumns
def digestColumns : List Nat := rangeFrom 2553433 4
def zeroColumn : Nat := 2543175

/-- The sixteen uniform rate-four absorb rounds. Local row positions are
relative to the zero-definition row, not to the whole F-prime relation. -/
def absorbRound (index : Fin 16) : Round :=
  let firstAllocated := 2543180 + 604 * index.val
  let stateBefore :=
    if index.val = 0 then List.replicate 8 zeroColumn
    else rangeFrom (firstAllocated - 12) 8
  let permutationInput :=
    if index.val = 0 then
      rangeFrom 2543176 4 ++ List.replicate 4 zeroColumn
    else
      rangeFrom (firstAllocated - 4) 4 ++
        rangeFrom (firstAllocated - 8) 4
  { kind := .absorb ((envelopeColumns.drop (4 * index.val)).take 4)
    stateBeforeColumns := stateBefore
    permutationInputColumns := permutationInput
    permutationOutputColumns := rangeFrom (firstAllocated + 592) 8
    definingRows := rangeFrom (1 + 604 * index.val) 4
    call :=
      { rowStart := 5 + 604 * index.val
        rowEnd := 605 + 604 * index.val
        inputColumns := permutationInput
        firstAllocatedColumn := firstAllocated } }

/-- Final production padding round. -/
def padRound : Round :=
  { kind := .pad
    stateBeforeColumns := rangeFrom 2552832 8
    permutationInputColumns := [2552840] ++ rangeFrom 2552833 7
    permutationOutputColumns := rangeFrom 2553433 8
    definingRows := [9665]
    call :=
      { rowStart := 9666
        rowEnd := 10266
        inputColumns := [2552840] ++ rangeFrom 2552833 7
        firstAllocatedColumn := 2552841 } }

def rounds : List Round := List.ofFn absorbRound ++ [padRound]

/-- The owner piece containing ten prefix pins, the zero definition, and the
first four absorb definitions. -/
def initialPieceIndex :
    Fin FPrimeFullHistoryTerminalPiCcsOutputMessageHashes.Generated.pieces14.length :=
  ⟨2, by decide⟩

def initialPiece : Piece :=
  FPrimeFullHistoryTerminalPiCcsOutputMessageHashes.Generated.pieces14.get
    initialPieceIndex

/-- Remaining alternating call/definition pieces, ending at the padding
permutation and excluding the four post-hash copy rows. -/
def tailPieces : List Piece :=
  (FPrimeFullHistoryTerminalPiCcsOutputMessageHashes.Generated.pieces14.drop 3).take 33

/-- Exact isolated row list for the final sponge. -/
def hashRows : List Row :=
  initialPiece.rows.drop 10 ++ (tailPieces.map Piece.rows).flatten

def trace : Trace where
  inputColumns := envelopeColumns
  zeroColumn := zeroColumn
  zeroRow := 0
  rounds := rounds
  outputColumns := digestColumns

theorem prefixColumns_length : prefixColumns.length = 10 := by decide
theorem compressionColumns_length : compressionColumns.length = 54 := by decide
theorem envelopeColumns_length : envelopeColumns.length = 64 := by decide
theorem rounds_length : rounds.length = 17 := by decide
theorem digestColumns_length : digestColumns.length = 4 := by decide
theorem hashRows_length : hashRows.length = 10266 := by decide

/-- Kernel evaluation checks every absorb definition, exact renamed 600-row
permutation, link, input order, final output, and terminal padding condition. -/
theorem trace_valid : trace.Valid hashRows := by
  decide

theorem initialPiece_mem_shard :
    initialPiece ∈
      FPrimeFullHistoryTerminalPiCcsOutputMessageHashes.Generated.pieces14 :=
  List.get_mem _ _

theorem tailPiece_mem_shard {piece : Piece} (member : piece ∈ tailPieces) :
    piece ∈
      FPrimeFullHistoryTerminalPiCcsOutputMessageHashes.Generated.pieces14 := by
  exact List.mem_of_mem_drop (List.mem_of_mem_take member)

private theorem shardPiece_mem_owner {piece : Piece}
    (member : piece ∈
      FPrimeFullHistoryTerminalPiCcsOutputMessageHashes.Generated.pieces14) :
    piece ∈ FPrimeFullHistoryTerminalPiCcsOutputMessageHashes.owner.pieces := by
  change piece ∈ FPrimeFullHistoryTerminalPiCcsOutputMessageHashes.pieces
  unfold FPrimeFullHistoryTerminalPiCcsOutputMessageHashes.pieces
  apply List.mem_append_right
  exact member

theorem initialPiece_mem_owner :
    initialPiece ∈
      FPrimeFullHistoryTerminalPiCcsOutputMessageHashes.owner.pieces :=
  shardPiece_mem_owner initialPiece_mem_shard

theorem tailPiece_mem_owner {piece : Piece} (member : piece ∈ tailPieces) :
    piece ∈ FPrimeFullHistoryTerminalPiCcsOutputMessageHashes.owner.pieces :=
  shardPiece_mem_owner (tailPiece_mem_shard member)

theorem trace_inputColumns : trace.inputColumns = envelopeColumns := rfl
theorem trace_outputColumns : trace.outputColumns = digestColumns := rfl

end Nightstream.Implementation.R1CS.PiCcsOutputDigest.Poseidon.Schedule
