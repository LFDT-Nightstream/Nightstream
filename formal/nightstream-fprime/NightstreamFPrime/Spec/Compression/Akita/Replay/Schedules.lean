import NightstreamFPrime.Spec.Compression.Akita.Replay.Rows
import NightstreamFPrime.Spec.Compression.Akita.Replay.Stage3Weight
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
- the rows of the relation of a level (`relation` of `Replay/Rows.lean`) have the ranks and the
  degrees of the shapes of the level, with the groups in the order `groupOrder` of the
  specification (`RowBatch.lean`): the last group first. `layout20With` and `layout30With` take
  the order as an argument, for the test of another order.

The schedule files hold the most bytes of the payload of `z` and the code computes it from a
cap that Lean does not have, so `layout20` and `layout30` take it as `zMax`, and the Rice
parameter of the payload as `rice`. The terminal has the shape `Key20.terminal`
(`Key30.terminal`) and the cap `normTerminal.limit`. -/

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
  fused : FusedShape
  plan : Option SetupPlan := none

/-- The count of field elements of an image of `B` or `D`: after the compression maps, or
`rank` ring elements of degree `degree`. -/
def image (chain : Option ChainShape) (rank degree : ℕ) : ℕ :=
  match chain with
  | some c => c.secondDegree * c.secondRank
  | none => rank * degree

def outer (S : Shape) : ℕ := image S.chain S.outerRank S.outerDegree

def assemble (levels : List Numbers) (setupVars : ℕ) (terminal : Shape) (family : ℕ × ℕ)
    (limit zMax rice : ℕ) : Layout :=
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
      openingRow := L.rhs.2
      setup := if (next.map (·.setup)).getD false then L.plan else none
      fused := L.fused : Fold }
  { folds
    terminal := {
      rounds := (levels.getLast?.map (·.vars)).getD 0 - 1
      degree := terminal.innerDegree
      e := (terminal.blocks - 1) * terminal.innerDegree
      zMax
      group := ⟨terminal.blocks, terminal.innerDegree, terminal.innerDegree, family.1, family.2,
        true⟩
      shape := terminal
      limit
      rice } }

/-- The source of the commitment rows of the group at the place `k` of a fold with opening groups
in the order `order`: the commitment of a root group, or the setup prefix or the next witness of
the level before. -/
def batchSource {n : ℕ} (order : Equiv.Perm (Fin n)) (R : Level) (root : Bool) (k : Fin n) :
    RhsSource :=
  if root then .root (order k).val
  else if ((R.groups[(order k).val]?).map (·.setup)).getD false then .setup else .witness

/-- The public rows of the relation of a fold with one group, the next witness of the level
before. -/
def levelRelation (L : Key20.Level) : List RhsRun × ℕ :=
  relation (fun _ : Fin 1 => L.shape) L.opening.openRank L.opening.openDegree L.opening.chain
    fun _ => .witness

/-- The numbers of a fold with opening groups: the root (`root`, the commitments of the
statement) or a level whose groups are the setup prefix and the next witness of the level
before, with `digits` quotient digits. -/
def ofBatch {n : ℕ} (order : Equiv.Perm (Fin n)) (B : Batch (Fin n)) (R : Level) (pairs : ℕ → ℕ × ℕ) (vars batch : ℕ)
    (base : ℕ) (root : Bool) (digits : ℕ) : Numbers where
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
  rhs := relation (fun k => B.shape (order k)) B.openRank B.openDegree B.chain
    (batchSource order R root)
  fused := .packing ⟨(List.finRange n).map fun k =>
      ⟨(order k).val, B.shape (order k), B.opening (order k), B.fold (order k)⟩,
    ⟨B.openDegree, B.openRank, B.chain⟩, digits⟩
  plan := some (SetupPlan.ofBatch B order B.low (Nat.log 2 base))

/-- The numbers of a fold with one group; a level with coefficient packing has `digits` quotient
digits. -/
def ofLevel (L : Key20.Level) (R : Level) (pair : ℕ × ℕ) (norm : Option Key20.Squared := none)
    (digits : ℕ := 0) : Numbers :=
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
    rhs := levelRelation L
    fused := if trace then .trace ⟨L.shape, L.opening, L.fold⟩ else
      .packing ⟨[⟨0, L.shape, L.opening, L.fold⟩],
        ⟨L.opening.openDegree, L.opening.openRank, L.opening.chain⟩, digits⟩ }

def rowLevel (row : Row) (i : ℕ) : Level := row.levels.getD i ⟨[], false, .quotientFree, 0, 0, 0⟩

def layout20With (order : ∀ n, Equiv.Perm (Fin n)) (zMax rice : ℕ) : Layout :=
  open Key20 in
  let R := rowLevel Akita.key20
  assemble [ofBatch (order 3) root (R 0) (family 0) rootVars rootBatch (root.shape 0).openingBase true
        quotientDigits,
      ofLevel level₁ (R 1) (family 1 0) (digits := quotientDigits), ofLevel level₂ (R 2) (family 2 0),
      ofLevel level₃ (R 3) (family 3 0) norm₃, ofLevel level₄ (R 4) (family 4 0) norm₄,
      ofLevel level₅ (R 5) (family 5 0) norm₅]
    0 terminal (family 6 0) normTerminal.limit zMax rice

/-- The layout of `key20.aks`. -/
def layout20 (zMax rice : ℕ) : Layout := layout20With groupOrder zMax rice

def layout30With (order : ∀ n, Equiv.Perm (Fin n)) (r : Key30.Row) (zMax rice : ℕ) : Layout :=
  open Key30 in
  let R := rowLevel (if r.vars = 27 then Akita.key30W27 else Akita.key30W26)
  assemble [ofBatch (order 3) (root r) (R 0) (family 0) rootVars rootBatch
        ((root r).shape 0).openingBase true quotientDigits₀,
      ofBatch (order 2) (level₁ r) (R 1) (family 1) vars₁ batch₁ ((level₁ r).shape 0).openingBase false
        quotientDigits,
      ofLevel (level₂ r) (R 2) (family 2 0) (digits := quotientDigits),
      ofLevel (level₃ r) (R 3) (family 3 0) (digits := quotientDigits),
      ofLevel level₄ (R 4) (family 4 0), ofLevel level₅ (R 5) (family 5 0) (norm₅ r),
      ofLevel level₆ (R 6) (family 6 0) norm₆]
    setupVars terminal (family 7 0) normTerminal.limit zMax rice

/-- The layout of a row of `key30.aks`. -/
def layout30 (r : Key30.Row) (zMax rice : ℕ) : Layout := layout30With groupOrder r zMax rice

/-- The public rows of each fold of `layout20` are `relation` of the shapes of the reduction of the
claims at the level: `Batch.liftedReduction` of `Key20.root` (`batch_fused`), `liftedReduction` of
level 1 (`packing_fused`) and `traceReduction` of levels 2 to 5 (`trace_fused`). -/
theorem layout20_relations (zMax rice : ℕ) :
    (layout20 zMax rice).folds.map (fun F => (F.rhs, F.openingRow)) =
      [relation (fun k => Key20.root.shape (groupOrder 3 k)) Key20.root.openRank
          Key20.root.openDegree Key20.root.chain
          (batchSource (groupOrder 3) (rowLevel Akita.key20 0) true),
        levelRelation Key20.level₁, levelRelation Key20.level₂, levelRelation Key20.level₃,
        levelRelation Key20.level₄, levelRelation Key20.level₅] :=
  rfl

/-- The public rows of each fold of `layout30` are `relation` of the shapes of the reduction of the
claims at the level: `Batch.liftedReduction` of the root and of level 1 (`batch_fused`),
`liftedReduction` of levels 2 and 3 (`packing_fused`) and `traceReduction` of levels 4 to 6
(`trace_fused`). -/
theorem layout30_relations (r : Key30.Row) (zMax rice : ℕ) :
    (layout30 r zMax rice).folds.map (fun F => (F.rhs, F.openingRow)) =
      let R := rowLevel (if r.vars = 27 then Akita.key30W27 else Akita.key30W26)
      [relation (fun k => (Key30.root r).shape (groupOrder 3 k)) (Key30.root r).openRank
          (Key30.root r).openDegree (Key30.root r).chain (batchSource (groupOrder 3) (R 0) true),
        relation (fun k => (Key30.level₁ r).shape (groupOrder 2 k)) (Key30.level₁ r).openRank
          (Key30.level₁ r).openDegree (Key30.level₁ r).chain
          (batchSource (groupOrder 2) (R 1) false),
        levelRelation (Key30.level₂ r), levelRelation (Key30.level₃ r), levelRelation Key30.level₄,
        levelRelation Key30.level₅, levelRelation Key30.level₆] :=
  rfl

end NightstreamFPrime.Spec.Compression.Akita.Replay
