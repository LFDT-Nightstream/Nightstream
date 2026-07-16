import Nightstream.Implementation.R1CS.Correspondence.PiCcsOutputDigest.Semantics
import Nightstream.Implementation.R1CS.Correspondence.ShiftedTernary.CanonicalWord
import Nightstream.Implementation.R1CS.Ownership.FPrimeFullHistory.FPrimeFullHistoryTerminalPiCcsOutputMessageHashesArtifact

/-!
Exact production schedule for the two canonical-word/SIS layers of the
terminal `Pi_CCS` output digest.

Assurance tier: implementation/R1CS correspondence. The independent message
serialization remains in `Semantics`; this file only proves how generated
pieces are arranged and what accepted pieces imply. Neither generated row
ranges nor legacy constraint totals are used as semantic premises.

Owns: the 6,683 primary canonical-word leaves; their exact connection to the
rank-2 seeded-Phi81 block; the 108 compression canonical-word leaves; their
exact connection from rank-2 outputs to the rank-1 block; and accepted-owner
projections for all three constraint families.

Does not own: identification of the 6,683 source columns with typed `Pi_CCS`
outputs; uniqueness of each canonical digit vector as a function of its source
field; seeded-matrix conformance to Rust; the Poseidon2 envelope; transcript
placement; cryptographic binding; row necessity; row removal; or cost totals.

Emits constraints: no.

Authority boundary: generated pieces are implementation objects. They acquire
meaning only through the independent shifted-ternary and seeded-Phi81
semantics projected below. In particular, this module does not call the four
eventual digest columns authoritative.

| Protocol | Phase | Constraint family | Leaf/group | Exact obligation |
|---|---|---|---|---|
| `Pi_CCS` | output digest | source encoding | `mainPieces[i]` | one canonical shifted-ternary opening for serialized field `i` |
| `Pi_CCS` | output digest | primary SIS | `mainCommitmentPiece` | rank-2 seeded-Phi81 block consumes exactly the 6,683 digit words |
| `Pi_CCS` | output digest | compression encoding | `compressionPieces[i]` | one canonical opening for primary SIS output coordinate `i` |
| `Pi_CCS` | output digest | compression SIS | `compressionCommitmentPiece` | rank-1 seeded-Phi81 block consumes exactly those 108 digit words |
-/

namespace Nightstream.Implementation.R1CS.PiCcsOutputDigest.EncodingSchedule

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.OwnerCertificate

set_option maxRecDepth 1048576
set_option maxHeartbeats 8000000

abbrev artifactOwner : Owner :=
  FPrimeFullHistoryTerminalPiCcsOutputMessageHashes.owner

abbrev ArtifactAccepted (assignment : Nat -> Nat) : Prop :=
  FPrimeFullHistoryTerminalPiCcsOutputMessageHashes.Accepted assignment

/-- One field leaf for every field in the independently fixed terminal
serialization. -/
def mainFieldCount : Nat := 6683

/-- Rank two times the 54-dimensional Phi81 target. -/
def compressionFieldCount : Nat := 108

/-- The artifact starts with one ordinary constant/shape piece. The next
6,683 pieces are the primary canonical encodings. -/
def mainPieces : List Piece :=
  (artifactOwner.pieces.drop 1).take mainFieldCount

/-- After the primary leaves, primary block, and one ordinary shape piece,
the next 108 pieces canonically encode the primary commitment outputs. -/
def compressionPieces : List Piece :=
  (artifactOwner.pieces.drop 6686).take compressionFieldCount

def isShifted (piece : Piece) : Bool :=
  match piece.payload with
  | .shiftedTernary _ _ => true
  | _ => false

def shiftedFieldColumn (piece : Piece) : Nat :=
  match piece.payload with
  | .shiftedTernary fieldColumn _ => fieldColumn
  | _ => 0

def shiftedDigitStart (piece : Piece) : Nat :=
  match piece.payload with
  | .shiftedTernary _ digitStart => digitStart
  | _ => 0

def mainFieldColumns : List Nat := mainPieces.map shiftedFieldColumn
def mainDigitStarts : List Nat := mainPieces.map shiftedDigitStart
def compressionFieldColumns : List Nat :=
  compressionPieces.map shiftedFieldColumn
def compressionDigitStarts : List Nat :=
  compressionPieces.map shiftedDigitStart

theorem mainPieces_length : mainPieces.length = mainFieldCount := by
  decide

theorem compressionPieces_length :
    compressionPieces.length = compressionFieldCount := by
  decide

theorem mainPieces_all_shifted : mainPieces.all isShifted = true := by
  decide

theorem compressionPieces_all_shifted :
    compressionPieces.all isShifted = true := by
  decide

theorem mainFieldColumns_length : mainFieldColumns.length = mainFieldCount := by
  simp [mainFieldColumns, mainPieces_length]

theorem mainDigitStarts_length : mainDigitStarts.length = mainFieldCount := by
  simp [mainDigitStarts, mainPieces_length]

theorem compressionFieldColumns_length :
    compressionFieldColumns.length = compressionFieldCount := by
  simp [compressionFieldColumns, compressionPieces_length]

theorem compressionDigitStarts_length :
    compressionDigitStarts.length = compressionFieldCount := by
  simp [compressionDigitStarts, compressionPieces_length]

/-- Every primary digit word is consumed by the exact rank-2 block, in the
same order as the independent 6,683-field serialization. -/
theorem mainDigitStarts_eq_primaryWordStarts :
    mainDigitStarts = FPrimeFullHistorySeededPhi81.block8.wordStarts := by
  decide

/-- The exact 108 primary commitment coordinates are the source fields of the
compression encodings. -/
theorem compressionFieldColumns_eq_primaryOutputs :
    compressionFieldColumns =
      FPrimeFullHistorySeededPhi81.block8.outputColumns := by
  decide

/-- Every compression digit word is consumed by the exact rank-1 block. -/
theorem compressionDigitStarts_eq_compressionWordStarts :
    compressionDigitStarts =
      FPrimeFullHistorySeededPhi81.block9.wordStarts := by
  decide

theorem primaryWordWidth :
    FPrimeFullHistorySeededPhi81.block8.wordWidth = 41 := by
  decide

theorem primaryKappa :
    FPrimeFullHistorySeededPhi81.block8.kappa = 2 := by
  decide

theorem compressionWordWidth :
    FPrimeFullHistorySeededPhi81.block9.wordWidth = 41 := by
  decide

theorem compressionKappa :
    FPrimeFullHistorySeededPhi81.block9.kappa = 1 := by
  decide

private theorem shifted_shape_of_true {piece : Piece}
    (isTrue : isShifted piece = true) :
    exists fieldColumn digitStart,
      piece.payload = .shiftedTernary fieldColumn digitStart := by
  cases payloadShape : piece.payload <;> simp [isShifted, payloadShape] at isTrue
  case shiftedTernary fieldColumn digitStart =>
    exact ⟨fieldColumn, digitStart, rfl⟩

theorem mainPiece_mem_owner {piece : Piece} (member : piece ∈ mainPieces) :
    piece ∈ artifactOwner.pieces := by
  exact List.mem_of_mem_drop (List.mem_of_mem_take member)

theorem compressionPiece_mem_owner {piece : Piece}
    (member : piece ∈ compressionPieces) :
    piece ∈ artifactOwner.pieces := by
  exact List.mem_of_mem_drop (List.mem_of_mem_take member)

theorem mainPiece_shape {piece : Piece} (member : piece ∈ mainPieces) :
    exists fieldColumn digitStart,
      piece.payload = .shiftedTernary fieldColumn digitStart := by
  apply shifted_shape_of_true
  exact (List.all_eq_true.mp mainPieces_all_shifted) piece member

theorem compressionPiece_shape {piece : Piece}
    (member : piece ∈ compressionPieces) :
    exists fieldColumn digitStart,
      piece.payload = .shiftedTernary fieldColumn digitStart := by
  apply shifted_shape_of_true
  exact (List.all_eq_true.mp compressionPieces_all_shifted) piece member

/-- An accepted shifted-ternary owner leaf has the independently proved
canonical-opening meaning after exact column relabeling. -/
theorem canonicalOpening_of_shiftedPiece
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (accepted : ArtifactAccepted assignment)
    {piece : Piece} (member : piece ∈ artifactOwner.pieces)
    {fieldColumn digitStart : Nat}
    (shape : piece.payload = .shiftedTernary fieldColumn digitStart) :
    ShiftedTernaryCompiler.CanonicalOpening
      (Relabel.assignment
        (shiftedTernaryColumnMap fieldColumn digitStart) assignment) := by
  have pieceAccepted := accepted piece member
  rw [Piece.Accepted, shape, Payload.Accepted] at pieceAccepted
  apply ShiftedTernarySound.canonicalOpening_of_canonicalRows prime
  · exact Relabel.canonical canonical
  · apply Relabel.constantOne
    · simp [shiftedTernaryColumnMap, Relabel.column]
    · exact one
  · apply (Relabel.satisfies_mapped_iff
      ShiftedTernaryCompiler.canonicalRows
      (shiftedTernaryColumnMap fieldColumn digitStart) assignment).mp
    simpa [Payload.rows, EquationsAccepted, EquationHolds, Satisfies,
      RowHolds] using pieceAccepted

/-- Every primary field leaf has an exact source column, digit-word start, and
canonical-opening proof. This still does not identify the source column with
one typed `Pi_CCS` output position. -/
theorem accepted_mainPiece_opening
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (accepted : ArtifactAccepted assignment)
    {piece : Piece} (member : piece ∈ mainPieces) :
    exists fieldColumn digitStart,
      piece.payload = .shiftedTernary fieldColumn digitStart /\
      ShiftedTernaryCompiler.CanonicalOpening
        (Relabel.assignment
          (shiftedTernaryColumnMap fieldColumn digitStart) assignment) := by
  rcases mainPiece_shape member with ⟨fieldColumn, digitStart, shape⟩
  exact ⟨fieldColumn, digitStart, shape,
    canonicalOpening_of_shiftedPiece prime canonical one accepted
      (mainPiece_mem_owner member) shape⟩

/-- Every primary SIS input coordinate is the unique native digit determined
by that leaf's source field. This rules out alternate accepted digit witnesses
before the seeded linear map is interpreted. -/
theorem accepted_mainPiece_word
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (accepted : ArtifactAccepted assignment)
    {piece : Piece} (member : piece ∈ mainPieces) :
    exists fieldColumn digitStart,
      piece.payload = .shiftedTernary fieldColumn digitStart /\
      forall index, index < ShiftedTernaryCompiler.digitCount ->
        assignment (digitStart + index) =
          ShiftedTernaryComplete.nativeDigit
            (ShiftedTernaryCanonicalWord.localAssignment
              assignment fieldColumn digitStart) index := by
  rcases accepted_mainPiece_opening prime canonical one accepted member with
    ⟨fieldColumn, digitStart, shape, opening⟩
  refine ⟨fieldColumn, digitStart, shape, ?_⟩
  intro index indexLt
  apply ShiftedTernaryCanonicalWord.productionDigit_eq_native
    (fieldColumn := fieldColumn) (digitStart := digitStart) (indexLt := indexLt)
  simpa [ShiftedTernaryCanonicalWord.localAssignment] using opening

/-- Index-addressable form of the primary word theorem. This is the bridge
used by the abstract SIS map: neither a generated digit witness nor a block
output appears on the right-hand side. -/
theorem accepted_mainWordAt
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (accepted : ArtifactAccepted assignment)
    (index digit : Nat) (indexLt : index < mainFieldCount)
    (digitLt : digit < ShiftedTernaryCompiler.digitCount) :
    assignment (mainDigitStarts.getD index 0 + digit) =
      ShiftedTernaryCanonicalWord.canonicalDigit
        (assignment (mainFieldColumns.getD index 0)) digit := by
  have pieceIndexLt : index < mainPieces.length := by
    simpa [mainPieces_length] using indexLt
  let piece := mainPieces.get ⟨index, pieceIndexLt⟩
  have pieceMember : piece ∈ mainPieces := List.get_mem _ _
  rcases accepted_mainPiece_opening prime canonical one accepted pieceMember with
    ⟨fieldColumn, digitStart, shape, opening⟩
  have fieldColumnEq : mainFieldColumns.getD index 0 = fieldColumn := by
    change (mainPieces.map shiftedFieldColumn).getD index 0 = fieldColumn
    rw [List.getD_eq_getElem?_getD, List.getElem?_map,
      List.getElem?_eq_getElem pieceIndexLt]
    change shiftedFieldColumn piece = fieldColumn
    simp [shiftedFieldColumn, shape]
  have digitStartEq : mainDigitStarts.getD index 0 = digitStart := by
    change (mainPieces.map shiftedDigitStart).getD index 0 = digitStart
    rw [List.getD_eq_getElem?_getD, List.getElem?_map,
      List.getElem?_eq_getElem pieceIndexLt]
    change shiftedDigitStart piece = digitStart
    simp [shiftedDigitStart, shape]
  rw [fieldColumnEq, digitStartEq]
  exact ShiftedTernaryCanonicalWord.productionDigit_eq_canonicalDigit
    opening digit digitLt

/-- Every compression leaf canonically encodes one exact primary SIS output
coordinate. -/
theorem accepted_compressionPiece_opening
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (accepted : ArtifactAccepted assignment)
    {piece : Piece} (member : piece ∈ compressionPieces) :
    exists fieldColumn digitStart,
      piece.payload = .shiftedTernary fieldColumn digitStart /\
      ShiftedTernaryCompiler.CanonicalOpening
        (Relabel.assignment
          (shiftedTernaryColumnMap fieldColumn digitStart) assignment) := by
  rcases compressionPiece_shape member with
    ⟨fieldColumn, digitStart, shape⟩
  exact ⟨fieldColumn, digitStart, shape,
    canonicalOpening_of_shiftedPiece prime canonical one accepted
      (compressionPiece_mem_owner member) shape⟩

/-- Compression SIS input words are likewise deterministic functions of the
108 primary commitment output fields. -/
theorem accepted_compressionPiece_word
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (accepted : ArtifactAccepted assignment)
    {piece : Piece} (member : piece ∈ compressionPieces) :
    exists fieldColumn digitStart,
      piece.payload = .shiftedTernary fieldColumn digitStart /\
      forall index, index < ShiftedTernaryCompiler.digitCount ->
        assignment (digitStart + index) =
          ShiftedTernaryComplete.nativeDigit
            (ShiftedTernaryCanonicalWord.localAssignment
              assignment fieldColumn digitStart) index := by
  rcases accepted_compressionPiece_opening prime canonical one accepted member
    with ⟨fieldColumn, digitStart, shape, opening⟩
  refine ⟨fieldColumn, digitStart, shape, ?_⟩
  intro index indexLt
  apply ShiftedTernaryCanonicalWord.productionDigit_eq_native
    (fieldColumn := fieldColumn) (digitStart := digitStart) (indexLt := indexLt)
  simpa [ShiftedTernaryCanonicalWord.localAssignment] using opening

/-- Index-addressable deterministic word theorem for the compression map. -/
theorem accepted_compressionWordAt
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (accepted : ArtifactAccepted assignment)
    (index digit : Nat) (indexLt : index < compressionFieldCount)
    (digitLt : digit < ShiftedTernaryCompiler.digitCount) :
    assignment (compressionDigitStarts.getD index 0 + digit) =
      ShiftedTernaryCanonicalWord.canonicalDigit
        (assignment (compressionFieldColumns.getD index 0)) digit := by
  have pieceIndexLt : index < compressionPieces.length := by
    simpa [compressionPieces_length] using indexLt
  let piece := compressionPieces.get ⟨index, pieceIndexLt⟩
  have pieceMember : piece ∈ compressionPieces := List.get_mem _ _
  rcases accepted_compressionPiece_opening prime canonical one accepted pieceMember with
    ⟨fieldColumn, digitStart, shape, opening⟩
  have fieldColumnEq :
      compressionFieldColumns.getD index 0 = fieldColumn := by
    change (compressionPieces.map shiftedFieldColumn).getD index 0 = fieldColumn
    rw [List.getD_eq_getElem?_getD, List.getElem?_map,
      List.getElem?_eq_getElem pieceIndexLt]
    change shiftedFieldColumn piece = fieldColumn
    simp [shiftedFieldColumn, shape]
  have digitStartEq :
      compressionDigitStarts.getD index 0 = digitStart := by
    change (compressionPieces.map shiftedDigitStart).getD index 0 = digitStart
    rw [List.getD_eq_getElem?_getD, List.getElem?_map,
      List.getElem?_eq_getElem pieceIndexLt]
    change shiftedDigitStart piece = digitStart
    simp [shiftedDigitStart, shape]
  rw [fieldColumnEq, digitStartEq]
  exact ShiftedTernaryCanonicalWord.productionDigit_eq_canonicalDigit
    opening digit digitLt

def mainCommitmentPiece : Piece :=
  artifactOwner.pieces.get ⟨6684, by decide⟩

def compressionCommitmentPiece : Piece :=
  artifactOwner.pieces.get ⟨6794, by decide⟩

theorem mainCommitmentPiece_mem :
    mainCommitmentPiece ∈ artifactOwner.pieces := by
  exact List.get_mem _ _

theorem compressionCommitmentPiece_mem :
    compressionCommitmentPiece ∈ artifactOwner.pieces := by
  exact List.get_mem _ _

theorem mainCommitmentPiece_eq :
    mainCommitmentPiece =
      { rowStart := 2702270
        rowEnd := 2702378
        payload := .seededPhi81 FPrimeFullHistorySeededPhi81.block8 } := by
  decide

theorem compressionCommitmentPiece_eq :
    compressionCommitmentPiece =
      { rowStart := 2715772
        rowEnd := 2715826
        payload := .seededPhi81 FPrimeFullHistorySeededPhi81.block9 } := by
  decide

/-- Owner acceptance exposes the independently executable rank-2 seeded
linear map; no digest or commitment claim is assumed. -/
theorem accepted_primaryCommitment
    {assignment : Nat -> Nat}
    (accepted : ArtifactAccepted assignment) :
    FPrimeFullHistorySeededPhi81.block8.Holds assignment := by
  have pieceAccepted := accepted mainCommitmentPiece
    mainCommitmentPiece_mem
  rw [Piece.Accepted, mainCommitmentPiece_eq, Payload.Accepted] at pieceAccepted
  exact pieceAccepted

/-- Owner acceptance exposes the independently executable rank-1 compression
map; no Poseidon2 digest claim is assumed. -/
theorem accepted_compressionCommitment
    {assignment : Nat -> Nat}
    (accepted : ArtifactAccepted assignment) :
    FPrimeFullHistorySeededPhi81.block9.Holds assignment := by
  have pieceAccepted := accepted compressionCommitmentPiece
    compressionCommitmentPiece_mem
  rw [Piece.Accepted, compressionCommitmentPiece_eq, Payload.Accepted]
    at pieceAccepted
  exact pieceAccepted

end Nightstream.Implementation.R1CS.PiCcsOutputDigest.EncodingSchedule
