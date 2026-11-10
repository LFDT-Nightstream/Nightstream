import NightstreamFPrime.Export.Generic.CircuitRelation
import NightstreamFPrime.Export.StreamingIdentity
import NightstreamFPrime.Export.Generic.SlotLayout
import NightstreamFPrime.Export.Generic.PieceRows

/-! The package of a bounded-depth step circuit (CW-5.9). It lists the circuit's pieces in the
relation's order: the lowered R1CS rows of each constraint, each Π_RLC ring product with its fresh
quotient columns, and each packed Poseidon2 block. Rust expands the package into the relation's
rows; `relationRows_packageOf` shows the expansion is `CircuitRelation.relationRows` when the
blocks pack. Every format round-trips (`Format.decode_encode`), so the encoded package determines
the relation. -/

namespace NightstreamFPrime.Export.BoundedDepth.StepPackage

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Layout
open NightstreamFPrime.Export
open NightstreamFPrime.Export.Codec
open NightstreamFPrime.Export.Package
open NightstreamFPrime.Export.Generic
open NightstreamFPrime.Export.Generic.Pieces

/-! ### Formats -/

def fieldFormat : Format F where
  encode := fun value => .atom value.val
  decode
    | .atom value =>
        if bounded : value < goldilocksModulus then .ok ⟨value, bounded⟩
        else .error "field element out of range"
    | .array _ => .error "expected a field atom"
  decode_encode := by
    intro value
    simp [value.isLt]

def combinationFormat : Format R1CS.LinearCombination where
  encode := fun value => .array [fieldFormat.encode value.constant,
    (list (pair nat fieldFormat)).encode value.terms]
  decode
    | .array [constant, terms] => do
        pure ⟨← fieldFormat.decode constant, ← (list (pair nat fieldFormat)).decode terms⟩
    | _ => .error "invalid linear combination"
  decode_encode := by
    intro value
    cases value
    simp [Format.decode_encode]
    rfl

def rowFormat : Format R1CS.Row where
  encode := fun value => .array [combinationFormat.encode value.a,
    combinationFormat.encode value.b, combinationFormat.encode value.c]
  decode
    | .array [a, b, c] => do
        pure ⟨← combinationFormat.decode a, ← combinationFormat.decode b,
          ← combinationFormat.decode c⟩
    | _ => .error "invalid row"
  decode_encode := by
    intro value
    cases value
    simp [Format.decode_encode]
    rfl

/-! ### Package pieces -/

/-- A piece of the package: the lowered rows of a constraint, a ring product with its fresh
quotient columns from `quotient`, or a packed Poseidon2 block. -/
inductive PackagePiece
  | rows (rows : List R1CS.Row)
  | ring (cells start quotient : Nat) (lanes : List Expr)
  | block (start : Nat) (lanes : List Expr)

def PackagePiece.format : Format PackagePiece where
  encode
    | .rows chunk => .array [.atom 0, (list rowFormat).encode chunk]
    | .ring cells start quotient lanes =>
        .array [.atom 1, .atom cells, .atom start, .atom quotient,
          (list exprFormat).encode lanes]
    | .block start lanes => .array [.atom 2, .atom start, (list exprFormat).encode lanes]
  decode
    | .array [.atom 0, chunk] => do pure (.rows (← (list rowFormat).decode chunk))
    | .array [.atom 1, .atom cells, .atom start, .atom quotient, lanes] => do
        pure (.ring cells start quotient (← (list exprFormat).decode lanes))
    | .array [.atom 2, .atom start, lanes] => do
        pure (.block start (← (list exprFormat).decode lanes))
    | _ => .error "invalid package piece"
  decode_encode := by
    intro value
    cases value <;> simp [Format.decode_encode]

/-- The relation rows a piece contributes before the binding rows. -/
def PackagePiece.relationRows : PackagePiece → List R1CS.Row
  | .rows chunk => chunk
  | .ring cells start quotient lanes => PieceRows.ringRows cells start quotient lanes
  | .block .. => []

def PackagePiece.block? : PackagePiece → Option Frame.Block
  | .block start lanes => some ⟨start, lanes⟩
  | _ => none

/-- The package piece of `piece` with fresh columns from `next`, and the next fresh column. -/
def pieceOf (piece : Piece) (next : Nat) : PackagePiece × Nat :=
  match piece with
  | .permutation start lanes => (.block start lanes, next)
  | .ringProduct cells start lanes =>
      (.ring cells start next lanes, next + ringDegree * cells)
  | .constraint expression =>
      let lowered := PieceRows.lowerPiece (PilotData.canonicalConstraints ())
        (.constraint expression) next
      (.rows lowered.1, lowered.2)

/-- The package pieces, tail recursive with the pieces reversed. -/
def packageAux : List Piece → Nat → List PackagePiece → List PackagePiece
  | [], _, reversed => reversed.reverse
  | piece :: rest, next, reversed =>
      let current := pieceOf piece next
      packageAux rest current.2 (current.1 :: reversed)

def packagePieces (pieces : List Piece) (next : Nat) : List PackagePiece :=
  packageAux pieces next []

/-- The first fresh column after the pieces' fresh columns from `next`. -/
def freshAfter : List Piece → Nat → Nat
  | [], next => next
  | piece :: rest, next => freshAfter rest (pieceOf piece next).2

/-! ### Expansion -/

/-- The lowering of the pieces outside the blocks, without an accumulator. -/
def lowerSpec : List Piece → Nat → List R1CS.Row × Nat
  | [], next => ([], next)
  | piece :: rest, next =>
      let lowered := PieceRows.lowerPiece (PilotData.canonicalConstraints ()) piece next
      let tail := lowerSpec rest lowered.2
      (lowered.1 ++ tail.1, tail.2)

/-- The package pieces without an accumulator. -/
def packageSpec : List Piece → Nat → List PackagePiece
  | [], _ => []
  | piece :: rest, next => (pieceOf piece next).1 :: packageSpec rest (pieceOf piece next).2

theorem lowerPiecesAux_eq (canonical : List Expr) (canonicalEq :
    canonical = PilotData.canonicalConstraints ()) :
    ∀ (pieces : List Piece) (next : Nat) (reversed : List R1CS.Row),
      PieceRows.lowerPiecesAux canonical pieces next reversed =
        (reversed.reverse ++ (lowerSpec pieces next).1, (lowerSpec pieces next).2)
  | [], next, reversed => by simp [PieceRows.lowerPiecesAux, lowerSpec]
  | piece :: rest, next, reversed => by
      subst canonicalEq
      simp only [PieceRows.lowerPiecesAux, lowerSpec]
      rw [lowerPiecesAux_eq _ rfl rest]
      simp [List.reverseAux_eq]

theorem packageAux_eq :
    ∀ (pieces : List Piece) (next : Nat) (reversed : List PackagePiece),
      packageAux pieces next reversed = reversed.reverse ++ packageSpec pieces next
  | [], _, reversed => by simp [packageAux, packageSpec]
  | piece :: rest, next, reversed => by
      simp only [packageAux, packageSpec]
      rw [packageAux_eq rest]
      simp

/-- The pieces' rows before the binding rows are the lowering of the pieces outside the blocks. -/
theorem flatMap_packageSpec :
    ∀ (pieces : List Piece) (next : Nat),
      (packageSpec pieces next).flatMap PackagePiece.relationRows =
        (lowerSpec (CircuitRelation.otherPieces pieces) next).1
  | [], _ => by simp [packageSpec, lowerSpec, CircuitRelation.otherPieces]
  | piece :: rest, next => by
      have tail := flatMap_packageSpec rest
      cases piece with
      | permutation start lanes =>
          simp only [packageSpec, pieceOf, List.flatMap_cons, PackagePiece.relationRows,
            List.nil_append]
          exact tail next
      | ringProduct cells start lanes =>
          simp only [packageSpec, pieceOf, List.flatMap_cons, PackagePiece.relationRows]
          rw [tail]
          rfl
      | constraint expression =>
          simp only [packageSpec, pieceOf, List.flatMap_cons, PackagePiece.relationRows]
          rw [tail]
          rfl

theorem blocks_packageSpec :
    ∀ (pieces : List Piece) (next : Nat),
      (packageSpec pieces next).filterMap PackagePiece.block? = CircuitRelation.blocksOf pieces
  | [], _ => rfl
  | .permutation start lanes :: rest, next => by
      show List.filterMap PackagePiece.block? (.block start lanes :: packageSpec rest next) =
        ⟨start, lanes⟩ :: CircuitRelation.blocksOf rest
      simp only [List.filterMap_cons, PackagePiece.block?, blocks_packageSpec rest]
  | .ringProduct cells start lanes :: rest, next => by
      show List.filterMap PackagePiece.block?
          (.ring cells start next lanes :: packageSpec rest (next + ringDegree * cells)) =
        CircuitRelation.blocksOf rest
      simp only [List.filterMap_cons, PackagePiece.block?, blocks_packageSpec rest]
  | .constraint expression :: rest, next => by
      show List.filterMap PackagePiece.block? ((pieceOf (.constraint expression) next).1 ::
          packageSpec rest (pieceOf (.constraint expression) next).2) =
        CircuitRelation.blocksOf rest
      simp only [List.filterMap_cons, pieceOf, PackagePiece.block?, blocks_packageSpec rest]

/-! ### The package -/

/-! ### The witness program -/

/-- The column ranges that Rust fills natively: each block's 1,096 locals (outputs by the
permutation, the rest unused) and each ring piece's outputs (prior plus challenge times value). -/
def nativeRanges (pieces : List Piece) : List (Nat × Nat) :=
  pieces.filterMap fun
    | .permutation start _ => some (start, start + 1096)
    | .ringProduct cells start _ => some (start, start + ringDegree * cells)
    | .constraint _ => none

abbrev Entry := Nat × (Expr ⊕ Hint)

def batchEntries (batch : WitnessBatch) : List Entry :=
  (batch.recipes.mapIdx fun index recipe => (batch.start + index, Sum.inl recipe)) ++
    batch.hints.mapIdx fun index hint =>
      (batch.start + batch.recipes.length + index, Sum.inr hint)

/-- Drops the entries inside `ranges`; both lists ascend by column. -/
def trimAux : List (Nat × Nat) → List Entry → List Entry → List Entry
  | _, [], kept => kept.reverse
  | ranges, (column, entry) :: rest, kept =>
      match ranges.dropWhile fun range => range.2 ≤ column with
      | range :: more =>
          if range.1 ≤ column then trimAux (range :: more) rest kept
          else trimAux (range :: more) rest ((column, entry) :: kept)
      | [] => trimAux [] rest ((column, entry) :: kept)

/-- Regroups entries into batches of consecutive columns of one kind. -/
def regroupAux : List Entry → List WitnessBatch → List WitnessBatch
  | [], batches => batches.reverse
  | (column, entry) :: rest, batches =>
      let extended := match batches, entry with
        | current :: earlier, Sum.inl recipe =>
            if current.hints.isEmpty ∧ current.start + current.recipes.length = column then
              some ({ current with recipes := current.recipes ++ [recipe] } :: earlier)
            else none
        | current :: earlier, Sum.inr hint =>
            if current.recipes.isEmpty ∧ current.start + current.hints.length = column then
              some ({ current with hints := current.hints ++ [hint] } :: earlier)
            else none
        | [], _ => none
      match extended with
      | some grown => regroupAux rest grown
      | none =>
          let fresh : WitnessBatch := match entry with
            | Sum.inl recipe => WitnessBatch.arithmetic column [recipe]
            | Sum.inr hint => WitnessBatch.hinted column [hint]
          regroupAux rest (fresh :: batches)

/-- The circuit's witness batches without the native ranges, in column order. -/
def witnessOf (batches : List WitnessBatch) (pieces : List Piece) : List WitnessBatch :=
  regroupAux (trimAux (nativeRanges pieces) (batches.flatMap batchEntries) []) []

/-! ### The package -/

/-- A step circuit's package. Rust fills the columns in this order: the shared source columns
from the step's inputs; the circuit's locals by column, `witness` batches and the native ranges
(a block's outputs by the permutation of its lanes, a ring piece's outputs by the ring product);
then, piece by piece, each fresh lowering column `c := a * b` from its row, each ring piece's
quotients, and each block's S-box outputs from `sboxStart`. The relation reads the columns through
the slots' layout (`SlotLayout.layout (SlotLayout.runSlots width slots)`). The application's
region is the first part of each column range: its witness word count, its first local, its local
count, its pieces from the first piece and its fresh columns from `firstFresh` (`application`), so
Rust can put a native application in its place. -/
structure Package where
  circuit : Nat
  publicColumns : Nat
  width : Nat
  firstFresh : Nat
  sboxStart : Nat
  pieces : List PackagePiece
  witness : List WitnessBatch
  slots : List (Nat × Nat × Nat)
  application : Nat × Nat × Nat × Nat × Nat

def Package.format : Format Package where
  encode := fun value => .array [.atom value.circuit, .atom value.publicColumns,
    .atom value.width, .atom value.firstFresh, .atom value.sboxStart,
    (list PackagePiece.format).encode value.pieces, (list WitnessBatch.format).encode value.witness,
    (list (pair nat (pair nat nat))).encode value.slots,
    (pair nat (pair nat (pair nat (pair nat nat)))).encode value.application]
  decode
    | .array [.atom circuit, .atom publicColumns, .atom width, .atom firstFresh,
        .atom sboxStart, pieces, witness, slots, application] => do
        pure ⟨circuit, publicColumns, width, firstFresh, sboxStart,
          ← (list PackagePiece.format).decode pieces, ← (list WitnessBatch.format).decode witness,
          ← (list (pair nat (pair nat nat))).decode slots,
          ← (pair nat (pair nat (pair nat (pair nat nat)))).decode application⟩
    | _ => .error "invalid step package"
  decode_encode := by
    intro value
    cases value
    simp [Format.decode_encode]
    rfl

def Package.blocks (package : Package) : List Frame.Block :=
  package.pieces.filterMap PackagePiece.block?

/-- Rust's expansion: the pieces' rows, then one binding row per block output lane. The S-box rows
are `PoseidonSboxFamilyPlan` on the blocks from `sboxStart`. -/
def Package.relationRows (package : Package) : List R1CS.Row :=
  package.pieces.flatMap PackagePiece.relationRows ++
    CircuitRelation.bindingRows package.width package.blocks package.sboxStart

/-! ### The slots -/

/-- The columns that the relation reads from a piece: the terms of a rows piece; a ring piece's
outputs, quotients and lanes; a block's output locals and lanes, which its binding rows and its
S-box rows read. -/
def PackagePiece.reads : PackagePiece → List Nat
  | .rows chunk => chunk.flatMap SlotLayout.rowColumns
  | .ring cells start quotient lanes =>
      List.range' start (ringDegree * cells) ++ List.range' quotient (ringDegree * cells) ++
        lanes.flatMap SlotLayout.exprColumns
  | .block start lanes => List.range' (start + 1080) 16 ++ lanes.flatMap SlotLayout.exprColumns

/-- The columns that the rows of a rows piece force to 0 or 1 (`SlotLayout.forcedColumn?`). -/
def PackagePiece.forced : PackagePiece → List Nat
  | .rows chunk => chunk.filterMap SlotLayout.forcedColumn?
  | _ => []

/-- The columns that the `bit` hints of `batches` fill. -/
def bitColumns (batches : List WitnessBatch) : List Nat :=
  batches.flatMap fun batch => (batch.hints.zipIdx).filterMap fun (hint, index) => match hint with
    | .bit .. => some (batch.start + batch.recipes.length + index)
    | _ => none

/-- The kind code of a column (`SlotLayout.kindOf`), or 3 for no slot. A column that a row reads
and each S-box column has a slot; the slot is a bit when the column is a bit candidate and a row
forces it to 0 or 1, and a field element otherwise. -/
def slotCode (sboxStart : Nat) (reads bits forced : Array Bool) (column : Nat) : Nat :=
  if (reads.getD column false || decide (sboxStart ≤ column)) = true then
    if (bits.getD column false && forced.getD column false) = true then 0 else 2
  else 3

/-- One column of the run fold: no run for code 3, else extend the last run or start one. -/
def runStep (code : Nat → Nat) (runs : List (Nat × Nat × Nat)) (column : Nat) :
    List (Nat × Nat × Nat) :=
  if code column = 3 then runs else
    match runs with
    | (start, count, kind) :: rest =>
        if kind = code column ∧ start + count = column then (start, count + 1, kind) :: rest
        else (column, 1, code column) :: runs
    | [] => [(column, 1, code column)]

/-- A package's slot runs `(start, count, kind)` from the public columns to the last S-box
column: a column has a slot only if a row of the relation reads it (`PackagePiece.reads`, and the
S-box rows read each S-box column), and the slot is a bit only if `bits` names the column and a row
forces it to 0 or 1 (`PackagePiece.forced`). Every other slot is a field element. A block's dead
locals have no slot, because no row reads them. -/
def runsOf (publicColumns sboxStart : Nat) (pieces : List PackagePiece) (bits : List Nat) :
    List (Nat × Nat × Nat) :=
  let size := sboxStart + 150 * (pieces.filterMap PackagePiece.block?).length
  let empty := Array.replicate size false
  let code := slotCode sboxStart
    (pieces.foldl (fun mask piece => SlotLayout.markColumns mask piece.reads) empty)
    (SlotLayout.markColumns empty bits)
    (pieces.foldl (fun mask piece => SlotLayout.markColumns mask piece.forced) empty)
  (((List.range size).drop publicColumns).foldl (runStep code) []).reverse

/-- A step circuit's slot runs: `runsOf` on its package pieces, with the columns of its `bit`
hints as the bit candidates. -/
def slotRuns (publicColumns : Nat) (pieces : List Piece) (next : Nat)
    (batches : List WitnessBatch) : List (Nat × Nat × Nat) :=
  runsOf publicColumns (CircuitRelation.sboxStart pieces next) (packagePieces pieces next)
    (bitColumns batches)

def packageOf (circuit publicColumns : Nat) (pieces : List Piece) (next width : Nat)
    (batches : List WitnessBatch) (application : Nat × Nat × Nat × Nat × Nat) : Package :=
  let sboxStart := CircuitRelation.sboxStart pieces next
  let package := packagePieces pieces next
  ⟨circuit, publicColumns, width, next, sboxStart, package, witnessOf batches pieces,
    runsOf publicColumns sboxStart package (bitColumns batches), application⟩

theorem relationRows_packageOf (circuit publicColumns : Nat) (pieces : List Piece)
    (next width : Nat) (batches : List WitnessBatch) (application : Nat × Nat × Nat × Nat × Nat)
    (packed : CircuitRelation.packable publicColumns pieces = true) :
    (packageOf circuit publicColumns pieces next width batches application).relationRows =
      CircuitRelation.relationRows publicColumns pieces next width := by
  simp only [Package.relationRows, Package.blocks, packageOf, packagePieces, packageAux_eq,
    List.reverse_nil, List.nil_append, blocks_packageSpec, CircuitRelation.relationRows, packed,
    ite_true, PieceRows.lowerPieces, lowerPiecesAux_eq _ rfl, flatMap_packageSpec]

/-- The rows of the package's pieces, before the binding rows, are the lowering of the pieces
outside the blocks. -/
theorem pieceRows_packageOf (circuit publicColumns : Nat) (pieces : List Piece) (next width : Nat)
    (batches : List WitnessBatch) (application : Nat × Nat × Nat × Nat × Nat) :
    (packageOf circuit publicColumns pieces next width batches application).pieces.flatMap
        PackagePiece.relationRows =
      (PieceRows.lowerPieces (PilotData.canonicalConstraints ()) (CircuitRelation.otherPieces pieces)
        next).1 := by
  simp only [packageOf, packagePieces, packageAux_eq, List.reverse_nil, List.nil_append,
    flatMap_packageSpec, PieceRows.lowerPieces, lowerPiecesAux_eq _ rfl]

theorem blocks_packageOf (circuit publicColumns : Nat) (pieces : List Piece) (next width : Nat)
    (batches : List WitnessBatch) (application : Nat × Nat × Nat × Nat × Nat) :
    (packageOf circuit publicColumns pieces next width batches application).blocks =
      CircuitRelation.blocksOf pieces := by
  simp only [Package.blocks, packageOf, packagePieces, packageAux_eq, List.reverse_nil,
    List.nil_append, blocks_packageSpec]

/-! ### One identity for both packages -/

/-- "Nightstream/OptionB/packages/v1". -/
def boundedDepthDomain : List F :=
  [78, 105, 103, 104, 116, 115, 116, 114, 101, 97, 109, 47, 79, 112, 116, 105, 111, 110, 66, 47,
    112, 97, 99, 107, 97, 103, 101, 115, 47, 118, 49]

/-- The identity of the light and refresh packages together: the Poseidon2 hash of their encoding,
as `Package.relationIdentifier`. -/
def identity (light refresh : Package) : List F :=
  Poseidon2.hash (boundedDepthDomain ++
    valuePreimage ((pair Package.format Package.format).encode (light, refresh)))

def identityValue (light refresh : Package) : Value :=
  (pair Package.format Package.format).encode (light, refresh)

/-- The same hash, streamed over the encoding (`StreamingIdentity`). -/
def identityFast (light refresh : Package) : List F :=
  StreamingIdentity.finalize (StreamingIdentity.processValue (identityValue light refresh)
    (StreamingIdentity.prefixState 2 boundedDepthDomain))

theorem identityFast_eq (light refresh : Package) :
    identityFast light refresh = identity light refresh := by
  have absorbed := StreamingIdentity.prefixStreamed_eq 2 boundedDepthDomain
    (identityValue light refresh) (by decide) (by decide) (by decide)
  rw [identity, Poseidon2.hash_eq_hashFast, ← StreamingIdentity.canonicalWords_eq_valuePreimage]
  unfold identityFast StreamingIdentity.finalize Poseidon2.hashFast
  dsimp only
  rw [← identityValue, absorbed]

end NightstreamFPrime.Export.BoundedDepth.StepPackage
