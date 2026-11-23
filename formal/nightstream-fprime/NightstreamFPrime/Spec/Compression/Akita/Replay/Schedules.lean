import NightstreamFPrime.Spec.Compression.Akita.Replay.Layout
import NightstreamFPrime.Spec.Compression.Akita.Key20Opening
import NightstreamFPrime.Spec.Compression.Akita.Key30Opening

/-! The layouts of the replay for the rows of `key20.aks` and `key30.aks`, from the Lean numbers
of `Key20.lean`, `Key30.lean`, `Rows.lean` and the families of `Key20Opening.lean` and
`Key30Opening.lean`, which `tests/AkitaSchedules.lean` compares with the files. Each count of a
message is the count of the pinned code (Akita revision `d50f700`):

- the opening payload of a level is the image of `D` after its compression maps, `deg·rank` of
  the second map, or the `openRank` ring elements of degree `openDegree` without them;
- the next witness of a level is the outer commitment of the witness group of the next level, in
  the same form with `B`, and at the last level the blocks of `t` of the terminal without its
  last block (`TailSegmentGroupLayout::t_field_elems`);
- a fold group of a level has one fold coordinate for each live block, in the ring of the
  carrier (the subring of the packing, or `R_{q,d_A}` at a trace level);
- the tensor reduction of a trace level and of the terminal has one round less than the count of
  variables of the level before (`canonical_extension_opening_reduction_shape`, split 1);
- the rows of the relation of a level (`relation`) have the ranks of `A`, `B` and `D` and the
  degrees of the compression maps of the shapes of the level, with the groups in the order
  `groupOrder` of the specification (`RowBatch.lean`): the last group first. `layout20With` and
  `layout30With` take the order as an argument, for the test of another order.

The schedule files hold the most bytes of the payload of `z` and the code computes it from a
cap that Lean does not have, so `layout20` and `layout30` take it as `zMax`. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

open NightstreamFPrime.Spec.Compression.Akita

structure Numbers where
  groups : List FoldGroup
  payload : ℕ
  outer : ℕ
  vars : ℕ
  low : ℕ
  batch : ℕ
  logBasis : ℕ
  trace : Option ℕ
  euclidean : Bool
  limit : ℕ
  compressed : Bool
  setup : Bool
  rhs : List RhsRun × ℕ

/-- The count of field elements of an image of `B` or `D`: after the compression maps, or
`rank` ring elements of degree `degree`. -/
def image (chain : Option ChainShape) (rank degree : ℕ) : ℕ :=
  match chain with
  | some c => c.secondDegree * c.secondRank
  | none => rank * degree

def outer (S : Shape) : ℕ := image S.chain S.outerRank S.outerDegree

/-- The ring degree of the last compression map. -/
def lastDegree : Option ChainShape → ℕ
  | some c => c.secondDegree
  | none => 0

/-- The rows of `y` with public coefficients and the count of rows of the relation of a fold
(`RelationRhsLayout::row_families`, `assemble_relation_rhs`, `assemble_compressed_relation_rhs`,
`scalar_opening_row_index`, `evaluation_trace_row_index`). `order` has for each group, in the order
of the rows (`root_group_order`), its commitment shape and the source of its commitment rows. A
group has its consistency row, the rows of `A` and, for
each slice, the rows of `B`; then `D` has its rows; a compressed fold has then two compression maps
with one row for each group and one for `D`. The public rows are those of `B` and `D`, or those of
the last map when the fold is compressed. -/
def relation (order : List (Shape × RhsSource)) (openRank openDegree : ℕ)
    (openChain : Option ChainShape) (compressed : Bool) : List RhsRun × ℕ :=
  let (runs, start) := order.foldl (fun (runs, start) (S, source) =>
    (runs ++ [⟨start + 1 + S.innerRank, S.outerDegree, S.slices * S.outerRank, source⟩],
      start + 1 + S.innerRank + S.slices * S.outerRank)) ([], 0)
  let base := start + openRank
  if compressed then
    ((order.zipIdx.map fun ((S, source), k) =>
        ⟨base + order.length + 1 + k, lastDegree S.chain, 1, source⟩) ++
      [⟨base + 2 * order.length + 1, lastDegree openChain, 1, .payload⟩],
      base + 2 * (order.length + 1))
  else (runs ++ [⟨start, openDegree, openRank, .payload⟩], base)

def assemble (levels : List Numbers) (setupVars : ℕ) (terminal : Shape) (family : ℕ × ℕ)
    (zMax : ℕ) : Layout :=
  let nexts := (levels.drop 1).map some ++ [none]
  let previous := none :: levels.map some
  let folds := ((levels.zip nexts).zip previous).map fun ((L, next), before) =>
    { groups := L.groups
      trace := L.trace.map fun degree => ((before.map (·.vars)).getD 0 - 1, degree)
      payload := L.payload
      witness := match next with
        | some N => N.outer
        | none => (terminal.blocks - 1) * terminal.innerRank * terminal.innerDegree
      last := next.isNone
      vars := L.vars
      low := L.low
      batch := L.batch
      logBasis := L.logBasis
      euclidean := L.euclidean
      limit := L.limit
      compressed := L.compressed
      stage3 := if (next.map (·.setup)).getD false then some setupVars else none
      rhs := L.rhs.1
      openingRow := L.rhs.2 : Fold }
  { folds
    terminal := {
      rounds := (levels.getLast?.map (·.vars)).getD 0 - 1
      degree := terminal.innerDegree
      e := (terminal.blocks - 1) * terminal.innerDegree
      zMax
      group := ⟨terminal.blocks, terminal.innerDegree, terminal.innerDegree, family.1, family.2,
        true⟩ } }

/-- The numbers of a fold with opening groups: the root (`root`, the commitments of the
statement) or a level whose groups are the setup prefix and the next witness of the level
before. -/
def ofBatch {n : ℕ} (order : Equiv.Perm (Fin n)) (B : Batch (Fin n)) (R : Level) (pairs : ℕ → ℕ × ℕ) (vars batch : ℕ)
    (base : ℕ) (root : Bool) : Numbers where
  groups := (List.finRange n).map fun g =>
    ⟨(B.shape g).blocks, (B.shape g).innerDegree, B.carrierDegree g, (pairs g).1, (pairs g).2,
      false⟩
  payload := image B.chain B.openRank B.openDegree
  outer := (List.finRange n).getLast?.map (fun g => outer (B.shape g)) |>.getD 0
  vars
  low := B.low
  batch
  logBasis := Nat.log 2 base
  trace := none
  euclidean := false
  limit := 0
  compressed := R.compressed
  setup := (R.groups.head?.map (·.setup)).getD false
  rhs := relation ((List.finRange n).map fun k => (B.shape (order k),
      if root then .root (order k).val
      else if ((R.groups[(order k).val]?).map (·.setup)).getD false then .setup else .witness))
    B.openRank B.openDegree B.chain R.compressed

def ofLevel (L : Key20.Level) (R : Level) (pair : ℕ × ℕ) (norm : Option Key20.Squared := none) :
    Numbers :=
  let trace := (R.groups.head?.map (·.opening)) == some .trace
  let euclidean := (R.groups.head?.map (·.norm)) == some .euclidean
  { groups := [⟨L.shape.blocks, L.shape.innerDegree, L.opening.carrierDegree, pair.1, pair.2,
      euclidean⟩]
    payload := image L.opening.chain L.opening.openRank L.opening.openDegree
    outer := outer L.shape
    vars := L.vars
    low := L.opening.low
    batch := L.batch
    logBasis := Nat.log 2 L.shape.openingBase
    trace := if trace then some L.shape.innerDegree else none
    euclidean
    limit := (norm.map (·.limit)).getD 0
    compressed := R.compressed
    setup := false
    rhs := relation [(L.shape, .witness)] L.opening.openRank L.opening.openDegree L.opening.chain
      R.compressed }

def rowLevel (row : Row) (i : ℕ) : Level := row.levels.getD i ⟨[], false, .quotientFree, 0, 0, 0⟩

def layout20With (order : ∀ n, Equiv.Perm (Fin n)) (zMax : ℕ) : Layout :=
  open Key20 in
  let R := rowLevel Akita.key20
  assemble [ofBatch (order 3) root (R 0) (family 0) rootVars rootBatch (root.shape 0).openingBase true,
      ofLevel level₁ (R 1) (family 1 0), ofLevel level₂ (R 2) (family 2 0),
      ofLevel level₃ (R 3) (family 3 0) norm₃, ofLevel level₄ (R 4) (family 4 0) norm₄,
      ofLevel level₅ (R 5) (family 5 0) norm₅]
    0 terminal (family 6 0) zMax

/-- The layout of `key20.aks`. -/
def layout20 (zMax : ℕ) : Layout := layout20With groupOrder zMax

def layout30With (order : ∀ n, Equiv.Perm (Fin n)) (r : Key30.Row) (zMax : ℕ) : Layout :=
  open Key30 in
  let R := rowLevel (if r.vars = 27 then Akita.key30W27 else Akita.key30W26)
  assemble [ofBatch (order 3) (root r) (R 0) (family 0) rootVars rootBatch
        ((root r).shape 0).openingBase true,
      ofBatch (order 2) (level₁ r) (R 1) (family 1) vars₁ batch₁ ((level₁ r).shape 0).openingBase false,
      ofLevel (level₂ r) (R 2) (family 2 0), ofLevel level₃ (R 3) (family 3 0),
      ofLevel level₄ (R 4) (family 4 0), ofLevel level₅ (R 5) (family 5 0) norm₅,
      ofLevel level₆ (R 6) (family 6 0) norm₆]
    setupVars terminal (family 7 0) zMax

/-- The layout of a row of `key30.aks`. -/
def layout30 (r : Key30.Row) (zMax : ℕ) : Layout := layout30With groupOrder r zMax

end NightstreamFPrime.Spec.Compression.Akita.Replay
