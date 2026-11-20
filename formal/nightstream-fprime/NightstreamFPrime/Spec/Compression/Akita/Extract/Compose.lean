import NightstreamFPrime.Spec.Compression.Akita.Extract.BatchLevel
import NightstreamFPrime.Spec.Compression.Akita.Extract.SplitGame

/-! The compression argument over an Akita opening of three groups as two checkpoint segments
(Akita, ePrint 2026/1983, revision of 2026-10-15, Definition 10.16 and Theorem 10.17; SPEC.md
section 6 and Algorithm 10).

The groups of the opening are 0 for the key polynomial, 1 for `W` and 2 for `V`. The prover
commits to `V` after the lookup challenges `τ`. So the argument has two checkpoints.

- `Compose` has the data: the stage `start` of the draw of `τ`, the reader of the messages, the
  claims of the first message and the commitments `bound` that a stage input fixes.
- `Compose.front` is the checkpoint at the stage `start`. Its candidate is the tables of the key
  polynomial and of `W`, its game is `Verifier.front`, and its test is `Verifier.Relation`. A
  result `τ` is true for a candidate when some table `V` has no error. The table `V` is not a
  part of the candidate, and the game does not read the commitment of `V`.
- `Compose.back` is the checkpoint at the stage `start + 1`. Its candidate is the three tables,
  its game is `Verifier.back`, and its test is "no error for `(W, V)`". A result is true for a
  candidate when the opening claims are true for its tables.
- `OpensAt` is the relation of the first certificates of the Akita opening at the stage `root`:
  a certified decommitment of each commitment, and each claim is true for the table of its group.
- `Compose.Ran` has what the verifier checks on an accepting transcript before the Akita
  opening.
- `Compose.suffix` puts the two segments before a suffix of the Akita opening. A first
  certificate has certified decommitments of the commitments of the key polynomial and of `W`,
  and `Relation` holds for its table of `W` when its table of the key polynomial is the table of
  `Params.key` (`Compose.Extracted`). Two certificates with different candidates give a solution
  of Module-SIS by `tableCollision` (`Decommits.collide_isSome`).
- `Compose.extract_sound`, `Compose.extraction_le` and `Compose.runs_le` are Akita Theorem 10.17
  for these stages.
- `Compose.Extracted.relation_or_collision`: with a certified decommitment of the key commitment
  to the table of `Params.key`, a first certificate gives `Relation` or a solution of
  Module-SIS. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Extract

open scoped BigOperators
open NightstreamFPrime.Spec.Compression.Lib (mle)
open NightstreamFPrime.Spec.Compression.Algebra (keyTable)
open NightstreamFPrime.Spec.Compression (Game)
open NightstreamFPrime.Spec.Checkpoint
open NightstreamFPrime.Spec.RandomOracle (OracleComp)

local instance : Fact (Nat.Prime fieldModulus) := ⟨FieldPrime.fieldModulus_natPrime⟩

attribute [local instance low] Classical.propDecidable

variable {E : Type} [Field E] [Fintype E] [DecidableEq E] [Algebra Zq E]

/-- The key polynomial of `Params.key` as a table of `Zq`. -/
def keyBase (P : Verifier.Params Zq E) : Table :=
  keyTable P.lanes P.keyBlockVars P.rows P.key

omit [Fintype E] [DecidableEq E] in
theorem lifted_keyBase (P : Verifier.Params Zq E) : lifted E (keyBase P) = Verifier.keyCells P := by
  funext x
  show algebraMap Zq E (if _ then _ else 0) = if _ then _ else 0
  split
  · rfl
  · exact map_zero _

/-- The claim of a group: 0 is the key polynomial, 1 is `W` and 2 is `V`. -/
def claimOf (claims : Verifier.Finals E) : Fin 3 → List E × E :=
  ![claims.key, claims.w, claims.v]

section Opening

variable {count : ℕ} {B : Batch (Fin count)} {Wire Coord : Type} {total : ℕ}

/-- Certified decommitments of the commitments `payload` of the groups in `groups`, with
`κ̄_{1,g} = 2ω_g` and `β̄_{∞,g} = Δ^cert_{f,g}`, to the tables that the source blocks decode
to. -/
def Decommits (key : (g : Fin count) → Key (B.shape g)) (mass : Fin count → ℕ)
    (groups : Fin count → Prop) (payload : (g : Fin count) → (B.shape g).Payload)
    (opening : Option B.Weak) : Prop :=
  ∃ ω, opening = some ω ∧ ∀ g, groups g → (ω.certified g).Opens (2 * mass g)
    ((B.fold g).sourceBound (B.shape g)) (key g) (payload g) (Batch.tablesOf opening g)

/-- `Coll_ℓ` of a checkpoint (Akita Definition 10.16 (vi), Corollary 10.25): two weak openings
with certified decommitments of one commitment of a group to different tables give a solution of
Module-SIS for a matrix of this group. -/
theorem Decommits.collide_isSome {key : (g : Fin count) → Key (B.shape g)} (openKey : B.OpenKey)
    {mass : Fin count → ℕ} {groups : Fin count → Prop}
    {payload : (g : Fin count) → (B.shape g).Payload} {opening other : Option B.Weak}
    (left : Decommits key mass groups payload opening)
    (right : Decommits key mass groups payload other) {g : Fin count} (member : groups g)
    (different : Batch.tablesOf opening g ≠ Batch.tablesOf other g) :
    (Batch.collide key openKey (fun g => 2 * (2 * mass g) * (B.fold g).sourceBound (B.shape g))
      opening other).isSome := by
  obtain ⟨ω, rfl, opens⟩ := left
  obtain ⟨ω', rfl, opens'⟩ := right
  show (Finset.univ.toList.findSome? _).isSome
  refine List.findSome?_isSome_iff.mpr ⟨g, Finset.mem_toList.mpr (Finset.mem_univ g), ?_⟩
  rw [Option.isSome_map]
  exact tableCollision_isSome (opens g member) (opens' g member) (le_of_eq (by ring)) different

/-- The relation of a first certificate of an Akita opening with groups whose batching root is
at the stage `start` (Akita (173) with `φ = 1`; `RootOpens` is the case `start = 0`). The message
before the stage `start` has the commitment `commit g`, the point `place g` and the value
`value g` of each group. The opening has a certified decommitment of each commitment, and each
claim holds for the table of its group. -/
def OpensAt (start : ℕ) (commit : Wire → (g : Fin count) → (B.shape g).Payload)
    (value : Wire → Fin count → E) (place : Wire → Fin count → List E)
    (key : (g : Fin count) → Key (B.shape g)) (mass : Fin count → ℕ)
    (point : Prefix Wire (Coord → E) total) (opening : Option B.Weak) : Prop :=
  (∃ ω, opening = some ω ∧ ∀ g, (ω.certified g).Opens (2 * mass g)
    ((B.fold g).sourceBound (B.shape g)) (key g) (commit (point.wire start) g)
    ((B.shape g).decode (ω.certified g).source)) ∧
  (fun g => mle (place ((point.take start).wire start) g)
    (lifted E (Batch.tablesOf opening g))) = value ((point.take start).wire start)

end Opening

/-- The data of the compression argument over an Akita opening of the groups `B`. The stage
`start` has the lookup challenges `τ`. `read` decodes the messages of the proof. `statement` is
the claims of the first message. `bound anchor g` is the commitment of the group `g` that the
stage input `anchor` fixes. -/
structure Compose (P : Verifier.Params Zq E) (B : Batch (Fin 3)) (Wire Coord : Type)
    (total : ℕ) where
  start : ℕ
  read : Reader Wire
  statement : Wire → Verifier.Statement P
  bound : Prefix Wire (Coord → E) total → (g : Fin 3) → (B.shape g).Payload

namespace Compose

variable {P : Verifier.Params Zq E} {B : Batch (Fin 3)} {Wire Coord : Type} {total : ℕ}
  [Fintype Coord] [DecidableEq Coord] (C : Compose P B Wire Coord total)
  (vf : Verifier.VerifierData P)

omit [Fintype E] [DecidableEq E] [Fintype Coord] [DecidableEq Coord] in
/-- The first stage of the Akita opening: the stage after the draws of `Verifier.back`. -/
noncomputable def root : ℕ := C.start + 1 + (Verifier.backCosts vf).length

/-- The lookup challenges of a transcript: the challenges that the answer of the stage `start`
gives. -/
noncomputable def tau (point : Prefix Wire (Coord → E) total) : Fin P.lookups → Zq :=
  ((point.reply C.start).map (sample (Coord → E) (Fin P.lookups → Zq))).getD 0

omit [Field E] [DecidableEq E] [Algebra Zq E] in
theorem tau_take (point : Prefix Wire (Coord → E) total) :
    C.tau (point.take (C.start + 1)) = C.tau point := by
  unfold tau
  rw [Prefix.reply_take (Nat.lt_succ_self _)]

/-- The checkpoint of `W` (Akita Definition 10.16 (v) to (vii)). A candidate is the tables of
the key polynomial and of `W`. The game is the draw of `τ`. The test of a candidate is
`Relation` for its `W`, and a result `τ` is true for a candidate when `τ` has no table value and
some table `V` has no error. Each condition holds when the table of the key polynomial is not
the table of `Params.key`. -/
noncomputable def front :
    Segment Wire Coord E (Fin P.lookups → Zq) (Table × Table) total where
  start := C.start
  costs := [Verifier.tauError P]
  read := C.read
  check _ tables := Verifier.front P tables.2
  passes anchor tables τ := tables.1 = keyBase P →
    Verifier.Apart P τ ∧ ∃ V, ¬ Verifier.Fails P (C.statement (anchor.wire 0)) tables.2 V τ
  test anchor tables := tables.1 = keyBase P →
    Verifier.Relation P (C.statement (anchor.wire 0)) tables.2

/-- The checkpoint of `(W, V)` at the stage after `τ`. A candidate is the three tables. The game
is `Verifier.back` for the lookup challenges of the stage input. The test of a candidate is "`τ`
has no table value and `(W, V)` has no error", and a result is true for a candidate when it is
the opening claims and they are true for the tables. Each condition holds when the table of the
key polynomial is not the table of `Params.key`. -/
noncomputable def back :
    Segment Wire Coord E (Option (Verifier.Finals E)) (Fin 3 → Table) total where
  start := C.start + 1
  costs := Verifier.backCosts vf
  read := C.read
  check anchor tables :=
    Verifier.back vf (C.statement (anchor.wire 0)) (tables 1) (tables 2) (C.tau anchor)
  passes _ tables result := ∃ claims, result = some claims ∧ (tables 0 = keyBase P →
    Verifier.Opened (Verifier.keyCells P) (lifted E (tables 1)) (lifted E (tables 2)) claims)
  test anchor tables := tables 0 = keyBase P → Verifier.Apart P (C.tau anchor) ∧
    ¬ Verifier.Fails P (C.statement (anchor.wire 0)) (tables 1) (tables 2) (C.tau anchor)

/-- The conditions on the parameters. `wf` to `key_fits` are the hypotheses of
`Verifier.game_avoids`. `lookups`, `challenges` and `column`: an answer of the oracle gives
uniform lookup challenges, uniform challenges after `V` and a uniform column point. `early`: the
commitments of the key polynomial and of `W` are fixed by the stage input of the stage `start`.
`fits`: the stages before the Akita opening are stages of the transcript. -/
structure Sized : Prop where
  wf : Verifier.WellFormed P
  counts : Verifier.CountsNonzero P
  matrix : Verifier.MatrixSpec vf
  columns : P.lookups ≤ 2 ^ vf.columnVars
  rows : P.rows ≤ 2 ^ P.rowVars
  key_fits : ∀ c, (P.shape c).witness.vars - P.laneVars ≤ P.keyBlockVars
  lookups : ∃ decode : (Coord → E) → Fin P.lookups → Zq, Fair decode
  challenges : ∃ decode : (Coord → E) → Verifier.Challenges P, Fair decode
  column : vf.columnVars ≤ Fintype.card Coord
  early : ∀ anchor g, g ≠ 2 → C.bound anchor g = C.bound (anchor.take C.start) g
  fits : C.root vf ≤ total

variable {C vf}

omit [DecidableEq E] in
/-- Definition 10.16 (vii)(a) for the checkpoint of `W`: for a table `W` that does not satisfy
`Relation`, outside the bad set of `τ` no table `V` is without error. -/
theorem front_valid (sized : C.Sized vf) : C.front.Valid where
  sound anchor tables failed :=
    (Verifier.front_avoids sized.counts (C.statement (anchor.wire 0)) tables.2
      fun related => failed fun _ => related).mono
      fun _ none passed => none (passed (not_imp.mp failed).1)
  staged _ tables := Verifier.front_staged sized.lookups tables.2
  nonnegative cost member := by
    rw [List.mem_singleton.mp member]
    exact Verifier.tauError_nonneg
  draws := List.cons_ne_nil _ _

/-- Definition 10.16 (vii)(a) for the checkpoint of `(W, V)`: for tables with an error, on each
path that avoids the bad sets an opening claim is false. -/
theorem back_valid [Nonempty Coord] (sized : C.Sized vf) : (C.back vf).Valid where
  sound anchor tables failed :=
    (Verifier.back_avoids vf sized.wf sized.matrix sized.columns sized.rows sized.key_fits
      (C.statement (anchor.wire 0)) (tables 1) (tables 2) (C.tau anchor)).mono
      fun _ sound passed => by
        obtain ⟨claims, ended, opened⟩ := passed
        exact (not_imp.mp failed).2 (sound claims ended (opened (not_imp.mp failed).1))
  staged anchor tables := Verifier.back_staged vf sized.challenges exists_fair_slot
    (exists_fair_slots sized.column) sized.wf _ _ _ _
  nonnegative := Verifier.backCosts_nonnegative vf
  draws := List.cons_ne_nil _ _

omit [DecidableEq E] in
/-- The game of the checkpoint of `W` on a transcript with the stage `start` ends in the lookup
challenges of the transcript. -/
theorem front_game (tables : Table × Table) {point : Prefix Wire (Coord → E) total}
    (long : C.start < point.rest.length) : C.front.game tables point = .pure (C.tau point) := by
  obtain ⟨move, found⟩ : ∃ move, point.rest[C.start]? = some move :=
    ⟨_, List.getElem?_eq_getElem long⟩
  have moves : (point.rest.take C.front.finish).drop C.front.start = [move] := by
    show (point.rest.take (C.start + 1)).drop C.start = [move]
    rw [List.take_add_one, found, List.drop_append_of_le_length (by
        rw [List.length_take]
        omega),
      List.drop_eq_nil_of_le (by
        rw [List.length_take]
        omega)]
    rfl
  unfold Segment.game
  rw [moves]
  simp [position, settle, advance, Compose.front, Verifier.front, Compose.tau, Prefix.reply, found]

variable {Inner BreakT CandT KeyT : Type} {accept : Prefix Wire (Coord → E) total → Prop}
  {body : Structure (Prefix Wire (Coord → E) total) Coord E (ℕ → Wire) Inner BreakT CandT KeyT
    (Option B.Weak)}
  {key : (g : Fin 3) → Key (B.shape g)} {openKey : B.OpenKey} {mass : Fin 3 → ℕ}
  {commit : Wire → (g : Fin 3) → (B.shape g).Payload} {value : Wire → Fin 3 → E}
  {place : Wire → Fin 3 → List E} {error : ℝ} {loops tree : ℕ}

variable (C vf)

/-- What the verifier checks on an accepting transcript before the Akita opening. `long`: the
transcript has the stage `start`. `bound`: the commitments of the Akita opening are the
commitments that the stage input of the stage `start + 1` fixes. `ended`: `Verifier.back` ends in
the opening claims, and the points and the values of the Akita opening are these claims. The
game does not read the tables (`Verifier.back_bare`), so the tables are zero here. -/
structure Ran (commit : Wire → (g : Fin 3) → (B.shape g).Payload) (value : Wire → Fin 3 → E)
    (place : Wire → Fin 3 → List E) (point : Prefix Wire (Coord → E) total) : Prop where
  long : C.start < point.rest.length
  bound : commit (point.wire (C.root vf)) = C.bound (point.take (C.start + 1))
  ended : ∃ claims, (C.back vf).game (fun _ _ => 0) point = .pure (some claims) ∧
    place (point.wire (C.root vf)) = (fun g => (claimOf claims g).1) ∧
    value (point.wire (C.root vf)) = fun g => (claimOf claims g).2

/-- The relation of a first certificate of the stages from `start + 1`: certified decommitments
of the three commitments of the stage input, and the test of the checkpoint of `(W, V)`. -/
def Middle (key : (g : Fin 3) → Key (B.shape g)) (mass : Fin 3 → ℕ)
    (point : Prefix Wire (Coord → E) total) (opening : Option B.Weak) : Prop :=
  Decommits key mass (fun _ => True) (C.bound (point.take (C.start + 1))) opening ∧
    (C.back vf).test (point.take (C.start + 1)) (Batch.tablesOf opening)

omit [Fintype E] [DecidableEq E] [Fintype Coord] [DecidableEq Coord] in
/-- The relation of a first certificate of the stages from `start`: certified decommitments of
the commitments of the key polynomial and of `W` of the stage input, and `Relation` for the
table of `W` when the table of the key polynomial is the table of `Params.key`. -/
def Extracted (key : (g : Fin 3) → Key (B.shape g)) (mass : Fin 3 → ℕ)
    (point : Prefix Wire (Coord → E) total) (opening : Option B.Weak) : Prop :=
  Decommits key mass (· ≠ 2) (C.bound (point.take C.start)) opening ∧
    (Batch.tablesOf opening 0 = keyBase P →
      Verifier.Relation P (C.statement (point.wire 0)) (Batch.tablesOf opening 1))

/-- The checkpoint structure of the stages from `start + 1`: the checkpoint of `(W, V)` over the
structure `body` of the Akita opening. -/
noncomputable def backChain (accept : Prefix Wire (Coord → E) total → Prop)
    (body : Structure (Prefix Wire (Coord → E) total) Coord E (ℕ → Wire) Inner BreakT CandT KeyT
      (Option B.Weak))
    (key : (g : Fin 3) → Key (B.shape g)) (openKey : B.OpenKey) (mass : Fin 3 → ℕ)
    (embed : B.LevelBreak key openKey
      (fun g => 2 * (2 * mass g) * (B.fold g).sourceBound (B.shape g)) → BreakT) :=
  (C.back vf).over accept body (fun inner => Batch.tablesOf (body.witness inner))
    (fun inner other => embed (Batch.collide key openKey _ (body.witness inner)
      (body.witness other)))
    (C.Middle vf key mass)

/-- The checkpoint structure of the compression argument: the checkpoint of `W` over the
checkpoint of `(W, V)` over the structure `body` of the Akita opening. A witness of `R_bind` is
a witness of `R_bind` of `body`, and `embed` maps a solution of Module-SIS for a matrix of a
group to it. -/
noncomputable def chain (accept : Prefix Wire (Coord → E) total → Prop)
    (body : Structure (Prefix Wire (Coord → E) total) Coord E (ℕ → Wire) Inner BreakT CandT KeyT
      (Option B.Weak))
    (key : (g : Fin 3) → Key (B.shape g)) (openKey : B.OpenKey) (mass : Fin 3 → ℕ)
    (embed : B.LevelBreak key openKey
      (fun g => 2 * (2 * mass g) * (B.fold g).sourceBound (B.shape g)) → BreakT) :=
  C.front.over accept (C.backChain vf accept body key openKey mass embed)
    (fun inner => (Batch.tablesOf ((C.backChain vf accept body key openKey mass embed).witness
        inner) 0,
      Batch.tablesOf ((C.backChain vf accept body key openKey mass embed).witness inner) 1))
    (fun inner other => embed (Batch.collide key openKey _
      ((C.backChain vf accept body key openKey mass embed).witness inner)
      ((C.backChain vf accept body key openKey mass embed).witness other)))
    (C.Extracted key mass)

variable {C vf}

/-- The two checkpoints of the compression argument before a suffix of the Akita opening are a
suffix (Akita Theorem 10.45 for two segments). The stage errors add `runCost`, the sum of the
bounds of the draws of one run of the verifier game, the retry loops add 2, and the tree size is
multiplied by 4. `ran`: the verifier accepts only with the conditions `Ran`. `embedded`: `embed`
maps a solution of Module-SIS for a matrix of a group to a witness of `R_bind` of the Akita
opening. `later`: the stages of the Akita opening are a suffix whose first certificates satisfy
`OpensAt`.

Definition 10.16 (vii)(c) for the checkpoint of `W`: a first certificate of the stages from
`start + 1` passed the test of the checkpoint of `(W, V)`, so its own table `V` has no error for
the lookup challenges of its transcript. (vi) for both checkpoints: `Decommits.collide_isSome`. -/
theorem suffix [Nonempty Coord] (sized : C.Sized vf)
    (ran : ∀ point, accept point → C.Ran vf commit value place point)
    (embed : B.LevelBreak key openKey
      (fun g => 2 * (2 * mass g) * (B.fold g).sourceBound (B.shape g)) → BreakT)
    (embedded : ∀ solution, solution.isSome → body.breaks (embed solution))
    (later : Tier.Suffix accept (C.root vf) (OpensAt (C.root vf) commit value place key mass)
      body error loops tree) :
    Tier.Suffix accept C.start (C.Extracted key mass)
      (C.chain vf accept body key openKey mass embed) (Verifier.runCost vf + error) (2 + loops)
      (4 * tree) := by
  have decommits : ∀ point opening, accept point →
      OpensAt (C.root vf) commit value place key mass point opening →
      Decommits key mass (fun _ => True) (C.bound (point.take (C.start + 1))) opening := by
    rintro point _ accepted ⟨⟨ω, rfl, certified⟩, _⟩
    refine ⟨ω, rfl, fun g _ => ?_⟩
    rw [← (ran point accepted).bound]
    exact certified g
  have early : ∀ point opening, C.Middle vf key mass point opening →
      Decommits key mass (· ≠ 2) (C.bound (point.take C.start)) opening := by
    rintro point _ ⟨⟨ω, rfl, certified⟩, _⟩
    refine ⟨ω, rfl, fun g small => ?_⟩
    have group := certified g trivial
    rwa [sized.early _ g small, Prefix.take_take (Nat.le_succ _)] at group
  have middle := (C.back vf).suffix_before (back_valid sized) later sized.fits Batch.tablesOf
    (fun opening other => embed (Batch.collide key openKey _ opening other))
    (C.Middle vf key mass)
    (fun point opening opens => by
      show Decommits key mass _ (C.bound ((point.take (C.start + 1)).take (C.start + 1))) opening ∧
        (C.back vf).test ((point.take (C.start + 1)).take (C.start + 1)) (Batch.tablesOf opening)
      rw [Prefix.take_take le_rfl]
      exact opens)
    (fun point opening accepted opens good => ⟨decommits point opening accepted opens, good⟩)
    (fun point opening accepted opens => by
      obtain ⟨claims, ended, places, values⟩ := (ran point accepted).ended
      refine ⟨some claims, ended_of_bare C.read (Verifier.back_bare vf _ _ _ _ _ _) ended,
        claims, rfl, fun honest => ?_⟩
      have claimed := opens.2
      rw [Prefix.wire_take le_rfl, places, values] at claimed
      have keyClaim : mle claims.key.1 (lifted E (Batch.tablesOf opening 0)) = claims.key.2 :=
        congrFun claimed 0
      rw [honest, lifted_keyBase] at keyClaim
      exact ⟨keyClaim.symm, (congrFun claimed 1).symm, (congrFun claimed 2).symm⟩)
    (fun point other opening opening' accepted accepted' opens opens' same different => by
      have same : point.take (C.start + 1) = other.take (C.start + 1) := same
      have right := decommits other opening' accepted' opens'
      rw [← same] at right
      obtain ⟨g, differs⟩ := Function.ne_iff.mp different
      exact embedded _ ((decommits point opening accepted opens).collide_isSome openKey right
        trivial differs))
  have result := C.front.suffix_before (front_valid sized) middle
    (show C.start + 1 ≤ total from (Nat.le_add_right _ _).trans sized.fits)
    (fun opening : Option B.Weak => (Batch.tablesOf opening 0, Batch.tablesOf opening 1))
    (fun opening other => embed (Batch.collide key openKey _ opening other))
    (C.Extracted key mass)
    (fun point opening opens => by
      show C.Extracted key mass (point.take C.start) opening
      exact ⟨by
        rw [Prefix.take_take le_rfl]
        exact opens.1, by
        rw [Prefix.wire_take (Nat.zero_le _)]
        exact opens.2⟩)
    (fun point opening accepted opens good => ⟨early point opening opens, by
      have good : Batch.tablesOf opening 0 = keyBase P →
        Verifier.Relation P (C.statement ((point.take C.start).wire 0))
          (Batch.tablesOf opening 1) := good
      rwa [Prefix.wire_take (Nat.zero_le _)] at good⟩)
    (fun point opening accepted opens => ⟨C.tau point,
      front_game _ (ran point accepted).long, fun honest => by
        have tested : Verifier.Apart P (C.tau (point.take (C.start + 1))) ∧
            ¬ Verifier.Fails P (C.statement ((point.take (C.start + 1)).wire 0))
              (Batch.tablesOf opening 1) (Batch.tablesOf opening 2)
              (C.tau (point.take (C.start + 1))) := opens.2 honest
        rw [C.tau_take, Prefix.wire_take (Nat.zero_le _)] at tested
        show Verifier.Apart P (C.tau point) ∧ ∃ V, ¬ Verifier.Fails P
          (C.statement ((point.take C.start).wire 0)) (Batch.tablesOf opening 1) V (C.tau point)
        rw [Prefix.wire_take (Nat.zero_le _)]
        exact ⟨tested.1, _, tested.2⟩⟩)
    (fun point other opening opening' accepted accepted' opens opens' same different => by
      have same : point.take C.start = other.take C.start := same
      have right := early other opening' opens'
      rw [← same] at right
      by_cases zero : Batch.tablesOf opening 0 = Batch.tablesOf opening' 0
      · exact embedded _ ((early point opening opens).collide_isSome openKey right (g := 1)
          (by decide) fun one => different (Prod.ext zero one))
      · exact embedded _ ((early point opening opens).collide_isSome openKey right (g := 0)
          (by decide) zero))
  have costs : [Verifier.tauError P].sum + ((Verifier.backCosts vf).sum + error) =
      Verifier.runCost vf + error := by
    rw [List.sum_singleton, ← add_assoc, Verifier.backCosts_sum]
  have count : 1 + (1 + loops) = 2 + loops := by omega
  have size : 2 * (2 * tree) = 4 * tree := by omega
  rw [← costs, ← count, ← size]
  exact result

omit [Fintype E] [DecidableEq E] [Fintype Coord] [DecidableEq Coord] in
/-- A first certificate with a certified decommitment `honest` of the key commitment to the
table of `Params.key` gives `Relation` for its table of `W`, or its decommitment of the key
commitment and `honest` give a solution of Module-SIS for a matrix of the key group
(Akita Corollary 10.25). -/
theorem Extracted.relation_or_collision {point : Prefix Wire (Coord → E) total}
    {opening : Option B.Weak} (extracted : C.Extracted key mass point opening)
    {honest : Certified (B.shape 0)} {factorBound sourceBound innerBound : ℕ}
    (registered : honest.Opens factorBound sourceBound (key 0) (C.bound (point.take C.start) 0)
      (keyBase P))
    (covers : factorBound * (B.fold 0).sourceBound (B.shape 0) + 2 * mass 0 * sourceBound ≤
      innerBound) :
    Verifier.Relation P (C.statement (point.wire 0)) (Batch.tablesOf opening 1) ∨
      ∃ ω, opening = some ω ∧
        (tableCollision (key 0) innerBound (ω.certified 0) honest).isSome := by
  obtain ⟨⟨ω, rfl, certified⟩, related⟩ := extracted
  by_cases same : Batch.tablesOf (some ω) 0 = keyBase P
  · exact Or.inl (related same)
  · exact Or.inr ⟨ω, rfl, tableCollision_isSome (certified 0 (by decide)) registered covers same⟩

section Extraction

variable [Nonempty Coord] (sized : C.Sized vf)
  (ran : ∀ point, accept point → C.Ran vf commit value place point)
  (embed : B.LevelBreak key openKey
    (fun g => 2 * (2 * mass g) * (B.fold g).sourceBound (B.shape g)) → BreakT)
  (embedded : ∀ solution, solution.isSome → body.breaks (embed solution))
  (later : Tier.Suffix accept (C.root vf) (OpensAt (C.root vf) commit value place key mass) body
    error loops tree)
  (first : C.start = 0)
  (prover : OracleComp (Prefix Wire (Coord → E) total) (Coord → E) (ℕ → Wire))

local notation "whole" => chain C vf accept body key openKey mass embed

include sized ran embedded later first

/-- An output of the extractor for the compression argument over an Akita opening is correct
(Akita Theorem 10.17): the verifier accepts the transcript of the prover, and the output is a
first certificate with `Extracted` for the first message of the prover, or a witness of `R_bind`
of the Akita opening. -/
theorem extract_sound (budget : ℕ) (table : Prefix Wire (Coord → E) total → Coord → E)
    (tape : Tape (Prefix Wire (Coord → E) total) Coord E budget (Structure.protocol whole).stages)
    (out : Inner ⊕ BreakT) (result : Checkpoint.extract whole prover budget table tape = some out) :
    accept ((Structure.protocol whole).final prover table) ∧
      Sum.elim
        (fun certificate => C.Extracted key mass ((Structure.protocol whole).input prover table 0)
          (body.witness certificate))
        body.breaks out :=
  (suffix sized ran embed embedded later).extract_sound first prover budget table tape out result

variable [Finite Wire]

/-- Akita Theorem 10.17 for the compression argument over an Akita opening. For each prover with
at most `Q` queries to the oracle, the probability that the verifier accepts and the extractor
returns no output is at most `(Q + 1)·(runCost + error) + (Q + 1)·(2 + loops)/budget`. `runCost`
is the sum of the bounds of the draws of `Verifier.front` and `Verifier.back`, and `error` and
`loops` are the stage errors and the retry loops of the Akita opening. -/
theorem extraction_le {queries : ℕ} (bounded : prover.QueryBound queries) {budget : ℕ}
    (positive : 0 < budget) :
    𝔼 table, 𝔼 tape : Tape (Prefix Wire (Coord → E) total) Coord E budget
          (Structure.protocol whole).stages,
        (if accept ((Structure.protocol whole).final prover table) ∧
          Checkpoint.extract whole prover budget table tape = none then (1 : ℝ) else 0) ≤
      (queries + 1) * (Verifier.runCost vf + error) +
        (queries + 1) * (2 + loops : ℕ) / budget :=
  (suffix sized ran embed embedded later).extraction_le first prover bounded positive

/-- The expected number of runs of the prover in the extraction is at most
`K̂ + (Q + 1)·(K̂ − 1)` with `K̂ = 4·tree` for the tree size `tree` of the Akita opening (Akita
Theorem 10.17, the cost). -/
theorem runs_le {queries : ℕ} (bounded : prover.QueryBound queries) (budget : ℕ) :
    𝔼 table, 𝔼 tape : Tape (Prefix Wire (Coord → E) total) Coord E budget
          (Structure.protocol whole).stages,
        (extractRuns whole prover budget table tape : ℝ) ≤
      (4 * tree : ℕ) + (queries + 1) * ((4 * tree : ℕ) - 1) :=
  (suffix sized ran embed embedded later).runs_le first prover bounded budget

end Extraction

end Compose

end NightstreamFPrime.Spec.Compression.Akita.Extract
