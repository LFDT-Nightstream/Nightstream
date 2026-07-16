import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.SelectionRows
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Transcript.ChunkOrder
import Nightstream.Implementation.R1CS.Ownership.FPrimeFullHistory.FPrimeFullHistoryTerminalPiRlcTranscriptRhosArtifact

/-!
Exact protocol-to-leaf ownership tree for all fifteen terminal-profile
`Pi_RLC` scalar samplers.

Assurance tier: implementation/R1CS correspondence. This file proves where
the terminal owner places each sampler leaf. It does not use that owner, its
row count, or the current Rust circuit as a semantic specification.

Owns: the indexed `rho -> digest block -> lane` layout; exact canonical-u64,
four-candidate lane, zero-prefix, and 54-of-64 tail pieces; and owner-acceptance
projections for those leaves.

Does not own: Poseidon2 transcript semantics, first-accepted correctness,
coefficient-to-ring assembly, paper-level `Pi_RLC` soundness, Rust trace
conformance, constraint necessity, row removal, or cost totals.

Emits constraints: no.

Authority boundary: generated pieces are implementation objects. The closed
tree equalities below check their placement against independently named affine
layout formulas. Mathematical authority remains in the separate transcript,
sampler, and NIFS semantics.

| Protocol | Phase | Constraint family | Indexed leaf | Exact obligation |
|---|---|---|---|---|
| `Pi_RLC` | terminal challenge | scalar batch | `rho : Fin 15` | exactly fifteen independently addressed scalar samplers |
| `Pi_RLC` | digest decomposition | canonical lane | `rho/block/lane` | one exact canonical-u64 piece binds a digest field lane to 64 bits |
| `Pi_RLC` | sampler chunk | four candidates | `rho/block/lane` | one exact 104-row leaf consumes the lane bits and prior count |
| `Pi_RLC` | sampler initialization | zero prefix | `rho` | the accepted-prefix chain starts at integer zero |
| `Pi_RLC` | sampler selection | 54-of-64 tail | `rho` | one exact 2,599-row tail consumes the sixteen lane leaves |
-/

namespace Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.Terminal.ScalarRows

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.OwnerCertificate
open Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript

set_option maxRecDepth 1000000
set_option maxHeartbeats 4000000

/-- Terminal `Pi_RLC` has one challenge per one of the exact fifteen outputs. -/
def scalarCount : Nat := 15

/-- The generated terminal owner contains four prefix pieces followed by one
44-piece first scalar and fourteen 43-piece successor scalars. -/
def pieceCount : Nat := 650

theorem ownerPieces_length :
    FPrimeFullHistoryTerminalPiRlcTranscriptRhos.owner.pieces.length =
      pieceCount := by
  decide

/-- Total, proof-carrying index into the exact terminal owner. -/
def ownerIndex (index : Fin pieceCount) :
    Fin FPrimeFullHistoryTerminalPiRlcTranscriptRhos.owner.pieces.length :=
  ⟨index.val, by
    rw [ownerPieces_length]
    exact index.isLt⟩

def pieceAt (index : Fin pieceCount) : Piece :=
  FPrimeFullHistoryTerminalPiRlcTranscriptRhos.owner.pieces.get
    (ownerIndex index)

theorem pieceAt_mem (index : Fin pieceCount) :
    pieceAt index ∈
      FPrimeFullHistoryTerminalPiRlcTranscriptRhos.owner.pieces := by
  exact List.get_mem _ _

/-! ## Independent affine layout formulas -/

/-- First accepted-prefix column of scalar `rho`. -/
def initialCountColumn (rho : Fin scalarCount) : Nat :=
  2555254 + 7984 * rho.val

/-- Digest output field column at one block/lane leaf. -/
def fieldColumn (rho : Fin scalarCount) (block lane : Fin 4) : Nat :=
  2556451 + 7984 * rho.val + 1236 * block.val + lane.val

/-- First Boolean column of one canonical digest lane. -/
def bitStart (rho : Fin scalarCount) (block lane : Fin 4) : Nat :=
  2556459 + 7984 * rho.val + 1236 * block.val + 158 * lane.val

/-- Accepted-prefix predecessor consumed by one four-candidate lane. -/
def cumulativePredecessor
    (rho : Fin scalarCount) (block lane : Fin 4) : Nat :=
  if block.val = 0 ∧ lane.val = 0 then initialCountColumn rho
  else
    let previous := 4 * block.val + lane.val - 1
    2556459 + 7984 * rho.val +
      1236 * (previous / 4) + 158 * (previous % 4) + 157

/-- First newly allocated column of the complete selection tail. -/
def tailFirstAllocated (rho : Fin scalarCount) : Nat :=
  2560799 + 7984 * rho.val

/-- Sixteen lane-bit bases in block-major, lane-major order. -/
def tailBitStarts (rho : Fin scalarCount) : List Nat :=
  [ bitStart rho ⟨0, by decide⟩ ⟨0, by decide⟩,
    bitStart rho ⟨0, by decide⟩ ⟨1, by decide⟩,
    bitStart rho ⟨0, by decide⟩ ⟨2, by decide⟩,
    bitStart rho ⟨0, by decide⟩ ⟨3, by decide⟩,
    bitStart rho ⟨1, by decide⟩ ⟨0, by decide⟩,
    bitStart rho ⟨1, by decide⟩ ⟨1, by decide⟩,
    bitStart rho ⟨1, by decide⟩ ⟨2, by decide⟩,
    bitStart rho ⟨1, by decide⟩ ⟨3, by decide⟩,
    bitStart rho ⟨2, by decide⟩ ⟨0, by decide⟩,
    bitStart rho ⟨2, by decide⟩ ⟨1, by decide⟩,
    bitStart rho ⟨2, by decide⟩ ⟨2, by decide⟩,
    bitStart rho ⟨2, by decide⟩ ⟨3, by decide⟩,
    bitStart rho ⟨3, by decide⟩ ⟨0, by decide⟩,
    bitStart rho ⟨3, by decide⟩ ⟨1, by decide⟩,
    bitStart rho ⟨3, by decide⟩ ⟨2, by decide⟩,
    bitStart rho ⟨3, by decide⟩ ⟨3, by decide⟩ ]

/-- Scalar zero owns a separate five-row initialization piece. Every later
scalar carries the same zero row in its six-row domain-transition piece. -/
def initializationPieceIndex (rho : Fin scalarCount) : Fin pieceCount :=
  ⟨if rho.val = 0 then 6 else 5 + 43 * rho.val, by
    have rhoLt := rho.isLt
    simp only [scalarCount, pieceCount] at rhoLt ⊢
    split <;> omega⟩

def initializationPiece (rho : Fin scalarCount) : Piece :=
  pieceAt (initializationPieceIndex rho)

/-! ## Closed exact-tree checks

The closed equalities are deliberately checked as whole nested trees. This
avoids fifteen copied scalar proofs and makes a missing, duplicated, or shifted
leaf fail at its protocol/phase/family address.
-/

/-- Flat size of the protocol `rho -> block -> lane` tree. -/
def laneLeafCount : Nat := scalarCount * 16

/-- Independent block-major/lane-major address of one flattened leaf. -/
def flatAddress
    (rho : Fin scalarCount) (block lane : Fin 4) : Fin laneLeafCount :=
  ⟨16 * rho.val + 4 * block.val + lane.val, by
    have rhoLt := rho.isLt
    have blockLt := block.isLt
    have laneLt := lane.isLt
    simp only [scalarCount, laneLeafCount] at rhoLt ⊢
    omega⟩

def addressRho (index : Fin laneLeafCount) : Fin scalarCount :=
  ⟨index.val / 16, by
    have indexLt := index.isLt
    simp only [laneLeafCount, scalarCount] at indexLt ⊢
    omega⟩

def addressBlock (index : Fin laneLeafCount) : Fin 4 :=
  ⟨(index.val % 16) / 4, by
    have remainderLt := Nat.mod_lt index.val (by decide : 0 < 16)
    omega⟩

def addressLane (index : Fin laneLeafCount) : Fin 4 :=
  ⟨index.val % 4, Nat.mod_lt _ (by decide)⟩

theorem addressRho_flat
    (rho : Fin scalarCount) (block lane : Fin 4) :
    addressRho (flatAddress rho block lane) = rho := by
  apply Fin.ext
  have blockLt := block.isLt
  have laneLt := lane.isLt
  simp only [addressRho, flatAddress]
  omega

theorem addressBlock_flat
    (rho : Fin scalarCount) (block lane : Fin 4) :
    addressBlock (flatAddress rho block lane) = block := by
  apply Fin.ext
  have blockLt := block.isLt
  have laneLt := lane.isLt
  simp only [addressBlock, flatAddress]
  omega

theorem addressLane_flat
    (rho : Fin scalarCount) (block lane : Fin 4) :
    addressLane (flatAddress rho block lane) = lane := by
  apply Fin.ext
  have laneLt := lane.isLt
  simp only [addressLane, flatAddress]
  omega

private def rho0 : Fin scalarCount := ⟨0, by decide⟩
private def rho1 : Fin scalarCount := ⟨1, by decide⟩
private def rho2 : Fin scalarCount := ⟨2, by decide⟩
private def rho3 : Fin scalarCount := ⟨3, by decide⟩
private def rho4 : Fin scalarCount := ⟨4, by decide⟩
private def rho5 : Fin scalarCount := ⟨5, by decide⟩
private def rho6 : Fin scalarCount := ⟨6, by decide⟩
private def rho7 : Fin scalarCount := ⟨7, by decide⟩
private def rho8 : Fin scalarCount := ⟨8, by decide⟩
private def rho9 : Fin scalarCount := ⟨9, by decide⟩
private def rho10 : Fin scalarCount := ⟨10, by decide⟩
private def rho11 : Fin scalarCount := ⟨11, by decide⟩
private def rho12 : Fin scalarCount := ⟨12, by decide⟩
private def rho13 : Fin scalarCount := ⟨13, by decide⟩
private def rho14 : Fin scalarCount := ⟨14, by decide⟩

private theorem rho_cases (rho : Fin scalarCount) :
    rho = rho0 ∨ rho = rho1 ∨ rho = rho2 ∨ rho = rho3 ∨
    rho = rho4 ∨ rho = rho5 ∨ rho = rho6 ∨ rho = rho7 ∨
    rho = rho8 ∨ rho = rho9 ∨ rho = rho10 ∨ rho = rho11 ∨
    rho = rho12 ∨ rho = rho13 ∨ rho = rho14 := by
  have rhoLt := rho.isLt
  have values : rho.val = 0 ∨ rho.val = 1 ∨ rho.val = 2 ∨
      rho.val = 3 ∨ rho.val = 4 ∨ rho.val = 5 ∨ rho.val = 6 ∨
      rho.val = 7 ∨ rho.val = 8 ∨ rho.val = 9 ∨ rho.val = 10 ∨
      rho.val = 11 ∨ rho.val = 12 ∨ rho.val = 13 ∨ rho.val = 14 := by
    simp only [scalarCount] at rhoLt
    omega
  rcases values with value | value | value | value | value | value | value |
      value | value | value | value | value | value | value | value
  all_goals first
    | exact Or.inl (Fin.ext value)
    | exact Or.inr (Or.inl (Fin.ext value))
    | exact Or.inr (Or.inr (Or.inl (Fin.ext value)))
    | exact Or.inr (Or.inr (Or.inr (Or.inl (Fin.ext value))))
    | exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inl (Fin.ext value)))))
    | exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl (Fin.ext value))))))
    | exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl (Fin.ext value)))))))
    | exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl (Fin.ext value))))))))
    | exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl (Fin.ext value)))))))))
    | exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl (Fin.ext value))))))))))
    | exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl (Fin.ext value)))))))))))
    | exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl (Fin.ext value))))))))))))
    | exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl (Fin.ext value)))))))))))))
    | exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl (Fin.ext value))))))))))))))
    | exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Fin.ext value))))))))))))))

/-- Exact generated-list address of one canonical-u64 leaf. -/
def canonicalPieceIndex
    (rho : Fin scalarCount) (block lane : Fin 4) : Fin pieceCount :=
  ⟨9 + 43 * rho.val + 10 * block.val + 2 * lane.val, by
    have rhoLt := rho.isLt
    have blockLt := block.isLt
    have laneLt := lane.isLt
    simp only [scalarCount, pieceCount] at rhoLt ⊢
    omega⟩

/-- Exact generated-list address of one four-candidate lane leaf. -/
def lanePieceIndex
    (rho : Fin scalarCount) (block lane : Fin 4) : Fin pieceCount :=
  ⟨10 + 43 * rho.val + 10 * block.val + 2 * lane.val, by
    have rhoLt := rho.isLt
    have blockLt := block.isLt
    have laneLt := lane.isLt
    simp only [scalarCount, pieceCount] at rhoLt ⊢
    omega⟩

def tailPieceIndex (rho : Fin scalarCount) : Fin pieceCount :=
  ⟨47 + 43 * rho.val, by
    have rhoLt := rho.isLt
    simp only [scalarCount, pieceCount] at rhoLt ⊢
    omega⟩

def canonicalPiece (rho : Fin scalarCount) (block lane : Fin 4) : Piece :=
  pieceAt (canonicalPieceIndex rho block lane)

def lanePiece (rho : Fin scalarCount) (block lane : Fin 4) : Piece :=
  pieceAt (lanePieceIndex rho block lane)

def tailPiece (rho : Fin scalarCount) : Piece :=
  pieceAt (tailPieceIndex rho)

def expectedCanonicalPiece
    (rho : Fin scalarCount) (block lane : Fin 4) : Piece :=
  { rowStart := 2729120 + 8387 * rho.val +
        1296 * block.val + 173 * lane.val
    rowEnd := 2729189 + 8387 * rho.val +
        1296 * block.val + 173 * lane.val
    payload := .canonicalU64
      (fieldColumn rho block lane) (bitStart rho block lane) }

def expectedLanePiece
    (rho : Fin scalarCount) (block lane : Fin 4) : Piece :=
  { rowStart := 2729189 + 8387 * rho.val +
        1296 * block.val + 173 * lane.val
    rowEnd := 2729293 + 8387 * rho.val +
        1296 * block.val + 173 * lane.val
    payload := .ordinary
      (AlphabetSamplingResidualTemplate.laneRows
        (bitStart rho block lane)
        (cumulativePredecessor rho block lane)) }

def expectedTailPiece (rho : Fin scalarCount) : Piece :=
  { rowStart := 2733700 + 8387 * rho.val
    rowEnd := 2736299 + 8387 * rho.val
    payload := .ordinary
      (AlphabetSamplingResidualTemplate.tailRows
        (tailBitStarts rho) (tailFirstAllocated rho)) }

private def index0 : Fin 4 := ⟨0, by decide⟩
private def index1 : Fin 4 := ⟨1, by decide⟩
private def index2 : Fin 4 := ⟨2, by decide⟩
private def index3 : Fin 4 := ⟨3, by decide⟩

private theorem fin4_cases (index : Fin 4) :
    index = index0 ∨ index = index1 ∨ index = index2 ∨ index = index3 := by
  have indexLt := index.isLt
  have values : index.val = 0 ∨ index.val = 1 ∨
      index.val = 2 ∨ index.val = 3 := by omega
  rcases values with value | value | value | value
  · exact Or.inl (Fin.ext value)
  · exact Or.inr (Or.inl (Fin.ext value))
  · exact Or.inr (Or.inr (Or.inl (Fin.ext value)))
  · exact Or.inr (Or.inr (Or.inr (Fin.ext value)))

/-- The whole terminal `rho -> block -> lane` canonical tree is definitionally
the independently named affine layout. No row-list equality procedure runs. -/
theorem canonicalPiece_eq
    (rho : Fin scalarCount) (block lane : Fin 4) :
    canonicalPiece rho block lane = expectedCanonicalPiece rho block lane := by
  rcases rho_cases rho with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl |
      rfl | rfl | rfl | rfl | rfl | rfl | rfl <;>
    rcases fin4_cases block with rfl | rfl | rfl | rfl <;>
    rcases fin4_cases lane with rfl | rfl | rfl | rfl <;> rfl

/-- The whole terminal `rho -> block -> lane` four-candidate tree is
definitionally the independently named affine layout. -/
theorem lanePiece_eq
    (rho : Fin scalarCount) (block lane : Fin 4) :
    lanePiece rho block lane = expectedLanePiece rho block lane := by
  rcases rho_cases rho with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl |
      rfl | rfl | rfl | rfl | rfl | rfl | rfl <;>
    rcases fin4_cases block with rfl | rfl | rfl | rfl <;>
    rcases fin4_cases lane with rfl | rfl | rfl | rfl <;> rfl

theorem tailPiece_eq (rho : Fin scalarCount) :
    tailPiece rho = expectedTailPiece rho := by
  rcases rho_cases rho with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl |
      rfl | rfl | rfl | rfl | rfl | rfl | rfl <;> rfl

theorem canonicalPiece_mem
    (rho : Fin scalarCount) (block lane : Fin 4) :
    canonicalPiece rho block lane ∈
      FPrimeFullHistoryTerminalPiRlcTranscriptRhos.owner.pieces :=
  pieceAt_mem _

theorem lanePiece_mem
    (rho : Fin scalarCount) (block lane : Fin 4) :
    lanePiece rho block lane ∈
      FPrimeFullHistoryTerminalPiRlcTranscriptRhos.owner.pieces :=
  pieceAt_mem _

theorem tailPiece_mem (rho : Fin scalarCount) :
    tailPiece rho ∈
      FPrimeFullHistoryTerminalPiRlcTranscriptRhos.owner.pieces :=
  pieceAt_mem _

theorem initializationPiece_mem (rho : Fin scalarCount) :
    initializationPiece rho ∈
      FPrimeFullHistoryTerminalPiRlcTranscriptRhos.owner.pieces :=
  pieceAt_mem _

/-! ## Accepted-owner projections -/

def initializationRows (rho : Fin scalarCount) : List Row :=
  match (initializationPiece rho).payload with
  | .ordinary rows => rows
  | _ => []

def initializationEquation (rho : Fin scalarCount) : Row :=
  ⟨[(initialCountColumn rho, 1)], [(0, 1)], []⟩

theorem initializationPiece_payload (rho : Fin scalarCount) :
    (initializationPiece rho).payload = .ordinary (initializationRows rho) := by
  rcases rho_cases rho with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl |
      rfl | rfl | rfl | rfl | rfl | rfl | rfl <;> decide

theorem initializationEquation_mem (rho : Fin scalarCount) :
    initializationEquation rho ∈ initializationRows rho := by
  rcases rho_cases rho with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl |
      rfl | rfl | rfl | rfl | rfl | rfl | rfl <;> decide

/-- Exact owner acceptance exposes one canonical-u64 leaf, without assigning
transcript meaning to its field value. -/
theorem accepted_canonicalRows
    {assignment : Nat → Nat}
    (accepted :
      FPrimeFullHistoryTerminalPiRlcTranscriptRhos.Accepted assignment)
    (rho : Fin scalarCount) (block lane : Fin 4) :
    Satisfies CanonicalU64.rows
      (ChunkOrder.laneSource assignment
        (fieldColumn rho block lane) (bitStart rho block lane)) := by
  have pieceAccepted := accepted (canonicalPiece rho block lane)
    (canonicalPiece_mem rho block lane)
  rw [Piece.Accepted, canonicalPiece_eq, expectedCanonicalPiece,
    Payload.Accepted] at pieceAccepted
  exact (Relabel.satisfies_mapped_iff CanonicalU64.rows
    (canonicalU64ColumnMap
      (fieldColumn rho block lane) (bitStart rho block lane)) assignment).mp
        pieceAccepted

/-- The exact terminal canonical-u64 leaf has the independent integer and
four-chunk meaning. This closes decomposition semantics only: the field column
is not yet proved to be the output of the verifier-owned transcript schedule. -/
theorem accepted_canonicalLane_refines
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat → Nat}
    (canonical : ChunkOrder.CanonicalAssignment assignment)
    (one : assignment 0 = 1)
    (accepted :
      FPrimeFullHistoryTerminalPiRlcTranscriptRhos.Accepted assignment)
    (rho : Fin scalarCount) (block lane : Fin 4) :
    ChunkOrder.LaneRefines assignment canonical
      (fieldColumn rho block lane) (bitStart rho block lane) := by
  exact ChunkOrder.satisfyingLane_refines prime canonical one
    (fieldColumn rho block lane) (bitStart rho block lane)
    (accepted_canonicalRows accepted rho block lane)

/-- Exact owner acceptance exposes one readable four-candidate lane leaf. -/
theorem accepted_laneRows
    {assignment : Nat → Nat}
    (accepted :
      FPrimeFullHistoryTerminalPiRlcTranscriptRhos.Accepted assignment)
    (rho : Fin scalarCount) (block lane : Fin 4) :
    Satisfies
      (AlphabetSamplingResidualTemplate.laneRows
        (bitStart rho block lane) (cumulativePredecessor rho block lane))
      assignment := by
  have pieceAccepted := accepted (lanePiece rho block lane)
    (lanePiece_mem rho block lane)
  rw [Piece.Accepted, lanePiece_eq, expectedLanePiece,
    Payload.Accepted] at pieceAccepted
  exact pieceAccepted

/-- Exact owner acceptance exposes the complete mapped selection tail for one
terminal rho coordinate. -/
theorem accepted_tailRows
    {assignment : Nat → Nat}
    (accepted :
      FPrimeFullHistoryTerminalPiRlcTranscriptRhos.Accepted assignment)
    (rho : Fin scalarCount) :
    Satisfies
      (AlphabetSamplingResidualTemplate.tailRows
        (tailBitStarts rho) (tailFirstAllocated rho)) assignment := by
  have pieceAccepted := accepted (tailPiece rho) (tailPiece_mem rho)
  rw [Piece.Accepted, tailPiece_eq, expectedTailPiece,
    Payload.Accepted] at pieceAccepted
  exact pieceAccepted

/-- The independently addressed initialization equation and canonical integer
representation force every terminal scalar's prefix counter to start at zero. -/
theorem accepted_initialCount_zero
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (accepted :
      FPrimeFullHistoryTerminalPiRlcTranscriptRhos.Accepted assignment)
    (rho : Fin scalarCount) :
    assignment (initialCountColumn rho) = 0 := by
  have pieceAccepted := accepted (initializationPiece rho)
    (initializationPiece_mem rho)
  rw [Piece.Accepted, initializationPiece_payload, Payload.Accepted]
    at pieceAccepted
  have rowHolds := pieceAccepted (initializationEquation rho)
    (initializationEquation_mem rho)
  have valueCanonical := canonical (initialCountColumn rho)
  simpa [initializationEquation, EquationHolds, lcEval, one,
    Nat.mod_eq_of_lt valueCanonical] using rowHolds

end Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.Terminal.ScalarRows
