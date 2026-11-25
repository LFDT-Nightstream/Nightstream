import NightstreamFPrime.Spec.Compression.Akita.Replay.Field
import NightstreamFPrime.Spec.Compression.Akita.Replay.Keccak
import NightstreamFPrime.Spec.Compression.Akita.Opening

/-! The public input of the replay of an Akita opening proof of the pinned code (Akita
revision `d50f700`, `crates/akita-verifier/src/fold/`): the numbers of each level of the schedule
that fix the order and the length of each message, the grinding plan, the session bytes, the
descriptor, the statement and the commitment rows of the setup prefix.

- `FoldGroup` is the family of the fold challenges of one group: the challenges are in the ring
  of degree `degree` (the carrier of the group, in `R_{q,ring}`) with `pm1` coefficients `±1` and `pm2` coefficients `±2`, and the
  operator-norm filter of the Euclidean route applies when `filtered`
  (`crates/akita-challenges/src/sampler/stream.rs`). `FoldGroup.messageBytes` is the length of
  the squeezed message of the group: `SparseChallengeShell::stream` with `pool_len`.
- `Fold` has the numbers of one level before the terminal and `Terminal` those of the terminal
  (`fold/mod.rs`, `fold/terminal.rs`). `Layout` is the list of levels of one schedule row.
- `Run` is a run of the grinding plan (`GrindingPlan::runs`), with the site in the text of the
  code. The descriptor binds the plan, so the plan is public. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

/-- The fold challenges of one group, and the ring degree `ring` of its matrix `A`. -/
structure FoldGroup where
  coordinates : ℕ
  ring : ℕ
  degree : ℕ
  pm1 : ℕ
  pm2 : ℕ
  filtered : Bool
deriving DecidableEq, Repr

namespace FoldGroup

/-- `C(n, k)` by the product rule of `binomial` in `stream.rs`. -/
def binomial (n k : ℕ) : ℕ := (List.range k).foldl (fun value i => value * (n - i) / (i + 1)) 1

theorem binomial_eq_choose (n k : ℕ) : binomial n k = n.choose k := by
  suffices step : ∀ j, (List.range j).foldl (fun value i => value * (n - i) / (i + 1)) 1 =
      n.choose j from step k
  intro j
  induction j with
  | zero => simp
  | succ j ih =>
      rw [List.range_succ, List.foldl_append, ih, List.foldl_cons, List.foldl_nil,
        ← Nat.choose_succ_right_eq, Nat.mul_div_cancel _ (Nat.succ_pos j)]

/-- The count of positions of the nonzero coefficients: `C(d, pm1)·C(d − pm1, pm2)`. -/
def rankSpace (G : FoldGroup) : ℕ := binomial G.degree G.pm1 * binomial (G.degree - G.pm1) G.pm2

/-- The bits of one candidate: the rank, `⌊log_2⌋` of `rankSpace` without the filter and
`⌈log_2⌉` with it, and one sign bit for each nonzero coefficient. -/
def candidateBits (G : FoldGroup) : ℕ :=
  (if G.filtered then Nat.size (G.rankSpace - 1) else Nat.log 2 G.rankSpace) + G.pm1 + G.pm2

/-- The rates `(u, v)` and `(r_num, r_den)` of `operator_norm_rates` for the filter in the ring
of degree 64 and in the ring of degree 128. -/
def rates (degree : ℕ) : (ℕ × ℕ) × (ℕ × ℕ) :=
  if degree = 64 then ((3611, 10000), (64, 100)) else ((5999, 10000), (132, 100))

/-- `B·u > W·v` and `(B·u − W·v)² ≥ 224·B·u·v`. -/
def chernoff (u v w b : ℕ) : Bool := w * v < b * u && 224 * b * u * v ≤ (b * u - w * v) ^ 2

def climb (u v w : ℕ) : ℕ → ℕ → ℕ
  | 0, b => b
  | fuel + 1, b => if chernoff u v w b then b else climb u v w fuel (b + 1)

def descend (u v w : ℕ) : ℕ → ℕ → ℕ
  | 0, b => b
  | fuel + 1, b => if w < b ∧ chernoff u v w (b - 1) then descend u v w fuel (b - 1) else b

/-- `pool_len` of `stream.rs` for `w` challenges: the least pool of the Chernoff condition from
its larger root, and then the least pool of the union bound in each class of bit length below
it. -/
def poolLength (degree w : ℕ) : ℕ :=
  let ((u, v), (num, den)) := rates degree
  let root := v * (w + 112 + Nat.sqrt (112 * (2 * w + 112))) / u
  let least := climb u v w (16 * w + 4096) (descend u v w (16 * w + 4096) (max root w))
  if w = 0 then least else
    let classes := (List.range (Nat.size (least - 1) + 1)).filter (Nat.size w ≤ ·)
    let first (len : ℕ) : Option ℕ :=
      let low := max w (2 ^ (len - 1))
      let high := min (2 ^ len - 1) (least - 1)
      let tail := ((w - 1) * len + 161) * den
      let start := max (w - 1 + (tail + num - 1) / num) low
      if start ≤ high then some start else none
    (classes.findSome? first).getD least

def candidates (G : FoldGroup) : ℕ :=
  if G.filtered then poolLength G.degree G.coordinates else G.coordinates

/-- The length of the squeezed message of the group: whole words of 64 bits. -/
def messageBytes (G : FoldGroup) : ℕ := (G.candidates * G.candidateBits + 63) / 64 * 8

end FoldGroup

/-- Where the verifier gets the coefficients of public rows of the right-hand side `y` of the
relation of a fold: the commitment rows of a root group (the last coefficients of its commitment
bytes), the commitment rows of the setup prefix, the next witness of the level before, or the
opening payload of the level. -/
inductive RhsSource
  | root (group : ℕ)
  | setup
  | witness
  | payload
deriving DecidableEq, Repr

/-- `count` rows of `y` from the row `row`, each one element of `R_{q,degree}`, with the
coefficients of `source` in order. -/
structure RhsRun where
  row : ℕ
  degree : ℕ
  count : ℕ
  source : RhsSource
deriving DecidableEq, Repr

/-- The numbers of one group of a fold (`SetupContributionGroupPlan`): the degrees `d_A` and
`d_B`, the rank `n_A`, the positions `M` of a block, the digits of the witness, of the fold, of
`t̂` and of `ê`, the live blocks, the slices of `B`, the rank of `B` and the width
`k·d_car` of an element of `ê`. One claim. -/
structure SetupGroup where
  innerDegree : ℕ
  outerDegree : ℕ
  innerRank : ℕ
  positions : ℕ
  witnessDigits : ℕ
  foldDigits : ℕ
  commitDigits : ℕ
  openDigits : ℕ
  blocks : ℕ
  slices : ℕ
  outerRank : ℕ
  openWidth : ℕ
deriving DecidableEq, Repr

/-- The groups of a fold in the order of the code and the numbers of `D`: its degree and rank,
the relation block `2^κ` and the bits of the base of the fold gadget. -/
structure SetupPlan where
  groups : List SetupGroup
  openDegree : ℕ
  openRank : ℕ
  block : ℕ
  logBasis : ℕ
deriving DecidableEq, Repr

/-- The Lean shapes of a level with evaluation trace: the shape, the opening shape and the
descriptor of its fold (`Key20.Level`). -/
structure TraceShape where
  shape : Shape
  opening : OpeningShape shape
  fold : Descriptor

/-- One group of a level with coefficient packing: its index in the fold (the index of its claim,
of its fold challenges and of its point), its shape, its opening shape and its descriptor. -/
structure PackSlot where
  index : ℕ
  shape : Shape
  opening : OpeningShape shape
  fold : Descriptor

/-- The numbers of the shared path of `D` and `H`: its degree, its rank and its compression maps. -/
structure OpenPath where
  degree : ℕ
  rank : ℕ
  chain : Option ChainShape

/-- The Lean shapes of a level with coefficient packing: its groups in the order of the cells of
the code (`root_group_order`), the path of `D` and the count `δ_quot` of quotient digits. -/
structure PackShape where
  slots : List PackSlot
  path : OpenPath
  digits : ℕ

/-- The Lean shapes of the end check of the fused sum-check of a level: `traceFused` at a level with
evaluation trace, `foldFused` at a level with coefficient packing. -/
inductive FusedShape
  | trace (T : TraceShape)
  | packing (P : PackShape)

instance : Repr FusedShape := ⟨fun
  | .trace T, _ => s!"trace (d_A = {T.shape.innerDegree})"
  | .packing P, _ => s!"packing ({P.slots.length} groups, δ_quot = {P.digits})"⟩

/-- The numbers of one level before the terminal (`fold/mod.rs`, `verify_fold`):

- `groups`: one claim for each group, in the order of the claims;
- `trace`: at a level with evaluation trace, the rounds of the tensor reduction and the ring
  degree of `A`, which fixes the point of the level (`ring_subfield_packed_extension_opening_point`);
- `payload`, `witness`: the counts of field elements of the opening payload and of the next
  witness, which at the last level is the `t` of the terminal without its last block (`last`);
- `vars` and `batch`: the widths of `τ0` and `τ1`; `vars` is the count of rounds of each sum-check
  of Stage 1 and Stage 2; `low`: the count of the last coordinates of `τ0` that the range proof
  puts first (`relation_coefficient_variable_count`);
- `logBasis`: the opening base of the range proof (`DigitRangePlan`);
- `euclidean`: the norm check of the Euclidean route, with the cap `limit` of the squared norm of
  the response (`response_l2_sq_cap`); `compressed`: the compression digits;
- `stage3`: the rounds of Stage 3 when the next level opens the setup prefix, and `setup` the
  plan of its weight (`SetupPlan.ofBatch`);
- `rhs`: the rows of `y` with public coefficients (each other row of `y` is zero), and
  `openingRow`: the count of rows of `y`, which is the index of the opening row;
- `fused`: the Lean shapes of the end check of the fused sum-check. -/
structure Fold where
  groups : List FoldGroup
  trace : Option (ℕ × ℕ)
  payload : ℕ
  witness : ℕ
  last : Bool
  vars : ℕ
  low : ℕ
  batch : ℕ
  logBasis : ℕ
  euclidean : Bool
  limit : ℕ
  compressed : Bool
  stage3 : Option ℕ
  rhs : List RhsRun
  openingRow : ℕ
  setup : Option SetupPlan := none
  fused : FusedShape
deriving Repr

/-- The numbers of the terminal (`fold/terminal.rs`): the rounds of its tensor reduction, the ring
degree of `A`, the count of field elements of `e` without its last block, the most bytes of the
payload of `z`, its fold challenges, its shape, the cap `S_max` of `‖z‖_2²`
(`response_l2_sq_cap`) and the Rice parameter of the payload of `z` (`z_rice_low_bits`). -/
structure Terminal where
  rounds : ℕ
  degree : ℕ
  e : ℕ
  zMax : ℕ
  group : FoldGroup
  shape : Shape
  limit : ℕ
  rice : ℕ

/-- `z_coords`: the coefficients of `z`, `M·δ·d` (`TailSegmentGroupLayout`). -/
def Terminal.zCoords (T : Terminal) : ℕ :=
  T.shape.blockLength * T.shape.sourceDigits * T.shape.innerDegree

/-- The elements of the setup prefix that the rows of `A` read: `n_A·M·δ·d`. -/
def Terminal.setupCount (T : Terminal) : ℕ := T.shape.innerRank * T.zCoords

structure Layout where
  folds : List Fold
  terminal : Terminal

/-- The product stages of the range proof for the opening base `2^logBasis`
(`DigitRangePlan::product_stage_arities`) and the count of child claims after each. -/
def productStages (logBasis : ℕ) : List (ℕ × ℕ) :=
  match logBasis with
  | 4 => [(2, 2)]
  | 5 => [(4, 4)]
  | 6 => [(2, 2), (4, 8)]
  | _ => []

/-- The degree of the leaf polynomial of the range proof (`DigitRangePlan::leaf_degree`). -/
def leafDegree (logBasis : ℕ) : ℕ := if logBasis = 2 then 2 else 4

structure Run where
  site : String
  loss : ℕ
  bits : ℕ
  nonceBits : ℕ
  count : ℕ
  setBits : ℕ
deriving DecidableEq, Repr, Inhabited

structure Statement where
  commitment : ByteArray
  point : Array Ext
  value : Array Ext

/-- The shape of one group of the public statement of a schedule row: the variables of its
polynomial (the length of its point), the profile header of its commitment
(`akita-types/src/proof/commitment.rs`, `write_group`) and the count of the coefficients of its
commitment rows after the header. -/
structure RootGroup where
  vars : ℕ
  header : ByteArray
  rows : ℕ

structure Public where
  layout : Layout
  plan : Array Run
  session : ByteArray
  descriptor : ByteArray
  groups : Array Statement
  slot : ByteArray
  setupRows : Array Fq
  seed : ByteArray
  /-- The prefix of the flat setup vector of `seed` (`Setup.expand`), when it is computed before
  the replay; the replay expands a longer prefix from `seed` when it needs one. `verifyProof`
  does not read it (`Public.sealed`). -/
  setup : Array Fq := #[]
  /-- The shape of each group of the statement of the schedule row. -/
  root : Array RootGroup := #[]
  /-- The sponge before the channel absorbs the session: `Duplex.empty` in the code. -/
  sponge : Duplex := Duplex.empty

/-- The public input without the setup prefix that a caller computed. -/
def Public.sealed (pub : Public) : Public := { pub with setup := #[] }

/-- The checks of the statement of the code before the first message of the proof
(`ns-compress` `akita/verifier.rs`, `Verifier::verify`; `PolynomialGroupClaims::new`;
`PreparedSubringCoefficientPackingPoint::new`; `fold/root.rs`, `verify_root`): the count of the
groups is that of the schedule row, each group has one value, a point of the length of its
variables, and a commitment that is the profile header of the row and then the canonical
coefficients of its rows. -/
def Public.shaped (pub : Public) : Bool :=
  pub.groups.size = pub.root.size &&
    (pub.groups.zip pub.root).all fun (g, r) =>
      g.value.size = 1 && g.point.size = r.vars &&
        g.commitment.size = r.header.size + 8 * r.rows &&
        (g.commitment.extract 0 r.header.size).data = r.header.data &&
        (List.range r.rows).all fun j =>
          (Fq.ofWord? (word g.commitment (r.header.size + 8 * j))).isSome
where
  word (bytes : ByteArray) (at_ : ℕ) : UInt64 :=
    (List.range 8).foldr (fun i acc => (acc <<< 8) ||| (bytes.get! (at_ + i)).toUInt64) 0

end NightstreamFPrime.Spec.Compression.Akita.Replay
