import NightstreamFPrime.Spec.Compression.Akita.Extract.BatchLevel
import NightstreamFPrime.Spec.Compression.Akita.Extract.SplitGame

/-! The compression argument over an Akita opening of three groups as two checkpoint segments
(Akita, ePrint 2026/1983, revision of 2026-10-15, Definition 10.16 and Theorem 10.17; SPEC.md
section 6 and Algorithm 10).

The groups of the opening are 0 for the key polynomial, 1 for `W` and 2 for `V`. The prover
commits to `V` after the lookup challenges `τ`. So the argument has two checkpoints.

- `Compose` has the data: the first stage `start`, the reader of the messages, the game `lead`
  before the lookup challenges with its bounds, the claims of the first message and of a value
  of `lead`, the test `good` of a table `W`, and the commitments `bound` that a stage input
  fixes. For the final verifier, `lead` is Π_CCS on the fresh instance and `good` is the
  relation of the final statement.
- `Compose.front` is the checkpoint at the stage `start`. Its candidate is the tables of the key
  polynomial and of `W`, its game is `Verifier.led`: `lead` and then the draw of `τ`. Its test
  is `good`. A result is true for a candidate when some table `V` has no error. The table `V`
  is not a part of the candidate, and the game does not read the commitment of `V`.
- `Compose.back` is the checkpoint at the stage after `τ`. Its candidate is the three tables,
  its game is `Verifier.back` for the result `Compose.ran` of the stages before it, and its test
  is "no error for `(W, V)`". A result is true for a candidate when the opening claims are true
  for its tables.
- `OpensAt` is the relation of the first certificates of the Akita opening at the stage `root`:
  a certified decommitment of each commitment, and each claim is true for the table of its group.
- `Compose.Ran` has what the verifier checks on an accepting transcript before the Akita
  opening.
- `Compose.suffix` puts the two segments before a suffix of the Akita opening. A first
  certificate has certified decommitments of the commitments of the key polynomial and of `W`,
  and `good` holds for its table of `W` when its table of the key polynomial is the table of
  `Params.key` (`Compose.Extracted`). Two certificates with different candidates give a solution
  of Module-SIS by `tableCollision` (`Decommits.collide_isSome`).
- `Compose.extract_sound`, `Compose.extraction_le` and `Compose.runs_le` are Akita Theorem 10.17
  for these stages.
- `Compose.Extracted.relation_or_collision`: with a certified decommitment of the key commitment
  to the table of `Params.key`, a first certificate gives `good` or a solution of
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

/-- The result of a game that has ended. -/
def finished {α : Type} : Game (Option α) → Option α
  | .pure result => result
  | _ => none

/-- The data of the compression argument over an Akita opening of the groups `B`. The stages
from `start` have the draws of `lead` and then the lookup challenges `τ`. `read` decodes the
messages of the proof. `lead wire W` is the game before the lookup challenges for the first
message `wire` and a table `W`, with a value of the type `Lead` at its end, and `leadCosts` has
the bounds of its draws. `statement wire value` is the claims of the argument. `good wire W` is
the test of a table `W`. `bound anchor g` is the commitment of the group `g` that the stage
input `anchor` fixes. -/
structure Compose (P : Verifier.Params Zq E) (B : Batch (Fin 3)) (Wire Coord Lead : Type)
    (total : ℕ) where
  start : ℕ
  read : Reader Wire
  leadCosts : List ℝ
  lead : Wire → Table → Game (Option Lead)
  statement : Wire → Lead → Verifier.Statement P
  good : Wire → Table → Prop
  bound : Prefix Wire (Coord → E) total → (g : Fin 3) → (B.shape g).Payload

namespace Compose

variable {P : Verifier.Params Zq E} {B : Batch (Fin 3)} {Wire Coord Lead : Type} {total : ℕ}
  [Fintype Coord] [DecidableEq Coord] (C : Compose P B Wire Coord Lead total)
  (vf : Verifier.VerifierData P)

/-- The checkpoint of `W` (Akita Definition 10.16 (v) to (vii)). A candidate is the tables of
the key polynomial and of `W`. The game is `lead` and then the draw of `τ`. The test of a
candidate is `good` for its `W`, and a result is true for a candidate when `lead` ends in a
value, `τ` has no table value and some table `V` has no error for the claims of this value. Each
condition holds when the table of the key polynomial is not the table of `Params.key`. -/
noncomputable def front :
    Segment Wire Coord E (Option (Lead × (Fin P.lookups → Zq))) (Table × Table) total where
  start := C.start
  costs := C.leadCosts ++ [Verifier.tauError P]
  read := C.read
  check anchor tables := Verifier.led P (C.lead (anchor.wire 0) tables.2) tables.2
  passes anchor tables result := tables.1 = keyBase P → ∃ value τ, result = some (value, τ) ∧
    Verifier.Apart P τ ∧
      ∃ V, ¬ Verifier.Fails P (C.statement (anchor.wire 0) value) tables.2 V τ
  test anchor tables := tables.1 = keyBase P → C.good (anchor.wire 0) tables.2

/-- The value of `lead` and the lookup challenges of a transcript: the result of the game of the
checkpoint of `W`. The game does not read the tables (`Sized.lead_bare`, `Verifier.front_bare`),
so the tables are zero here. -/
noncomputable def ran (point : Prefix Wire (Coord → E) total) :
    Option (Lead × (Fin P.lookups → Zq)) :=
  finished (C.front.game 0 point)

omit [DecidableEq E] in
theorem ran_take (point : Prefix Wire (Coord → E) total) :
    C.ran (point.take C.front.finish) = C.ran point := by
  unfold ran
  rw [Segment.game_take_finish]

/-- The first stage of the Akita opening: the stage after the draws of `Verifier.back`. -/
noncomputable def root : ℕ := C.front.finish + (Verifier.backCosts vf).length

/-- The game of the checkpoint of `(W, V)` for a result of the stages before it. -/
noncomputable def backCheck (wire : Wire) (tables : Fin 3 → Table) :
    Option (Lead × (Fin P.lookups → Zq)) → Game (Option (Verifier.Finals E))
  | none => .pure none
  | some (value, τ) => Verifier.back vf (C.statement wire value) (tables 1) (tables 2) τ

omit [Fintype Coord] [DecidableEq Coord] in
theorem backCheck_bare (wire : Wire) (tables other : Fin 3 → Table)
    (result : Option (Lead × (Fin P.lookups → Zq))) :
    bare (C.backCheck vf wire tables result) = bare (C.backCheck vf wire other result) := by
  rcases result with _ | ⟨value, τ⟩
  · rfl
  · exact Verifier.back_bare vf _ _ _ _ _ _

/-- The checkpoint of `(W, V)` at the stage after `τ`. A candidate is the three tables. The game
is `Verifier.back` for the value of `lead` and the lookup challenges of the stage input. The test
of a candidate is "`τ` has no table value and `(W, V)` has no error", and a result is true for a
candidate when it is the opening claims and they are true for the tables. Each condition holds
when the table of the key polynomial is not the table of `Params.key`. -/
noncomputable def back :
    Segment Wire Coord E (Option (Verifier.Finals E)) (Fin 3 → Table) total where
  start := C.front.finish
  costs := Verifier.backCosts vf
  read := C.read
  check anchor tables := C.backCheck vf (anchor.wire 0) tables (C.ran anchor)
  passes _ tables result := ∃ claims, result = some claims ∧ (tables 0 = keyBase P →
    Verifier.Opened (Verifier.keyCells P) (lifted E (tables 1)) (lifted E (tables 2)) claims)
  test anchor tables := tables 0 = keyBase P → ∃ value τ, C.ran anchor = some (value, τ) ∧
    Verifier.Apart P τ ∧
      ¬ Verifier.Fails P (C.statement (anchor.wire 0) value) (tables 1) (tables 2) τ

/-- The conditions on the parameters. `wf` to `key_fits` are the hypotheses of
`Verifier.game_avoids`. `lookups`, `challenges` and `column`: an answer of the oracle gives
uniform lookup challenges, uniform challenges after `V` and a uniform column point.
`lead_sound`: for a table `W` that fails the test `good`, on each path of `lead` that avoids the
bad sets `W` does not satisfy `Relation` for the claims of the value. `lead_staged`: each draw of
`lead` has its bound of `leadCosts`, and a path that ends in a value has all the draws.
`lead_bare`: `lead` does not read the table. `early`: the commitments of the key polynomial and
of `W` are fixed by the stage input of the stage `start`. `fits`: the stages before the Akita
opening are stages of the transcript. -/
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
  lead_sound : ∀ wire W, ¬ C.good wire W → (C.lead wire W).Avoids fun result =>
    ∀ value, result = some value → ¬ Verifier.Relation P (C.statement wire value) W
  lead_staged : ∀ wire W, StagedTo (Coord → E) (C.lead wire W)
    (C.leadCosts ++ [Verifier.tauError P]) fun result rest =>
      ∀ value, result = some value → rest = [Verifier.tauError P]
  lead_bare : ∀ wire W other, bare (C.lead wire W) = bare (C.lead wire other)
  lead_nonnegative : ∀ cost ∈ C.leadCosts, 0 ≤ cost
  early : ∀ anchor g, g ≠ 2 → C.bound anchor g = C.bound (anchor.take C.start) g
  fits : C.root vf ≤ total

variable {C vf}

omit [DecidableEq E] in
/-- Definition 10.16 (vii)(a) for the checkpoint of `W`: for a table `W` that fails the test
`good`, on each path that avoids the bad sets of `lead` and of `τ` no table `V` is without
error. -/
theorem front_valid (sized : C.Sized vf) : C.front.Valid where
  sound anchor tables failed :=
    (Verifier.led_avoids sized.counts (C.statement (anchor.wire 0)) tables.2
      (sized.lead_sound _ _ (not_imp.mp failed).2)).mono
      fun _ none passed => by
        obtain ⟨value, τ, ended, apart, clean⟩ := passed (not_imp.mp failed).1
        exact none value τ ended ⟨apart, clean⟩
  staged _ tables := Verifier.led_staged sized.lookups tables.2 (sized.lead_staged _ _)
  nonnegative cost member := by
    rcases List.mem_append.mp member with early | last
    · exact sized.lead_nonnegative cost early
    · rw [List.mem_singleton.mp last]
      exact Verifier.tauError_nonneg
  draws := List.append_ne_nil_of_right_ne_nil _ (List.cons_ne_nil _ _)

/-- Definition 10.16 (vii)(a) for the checkpoint of `(W, V)`: for tables with an error, on each
path that avoids the bad sets an opening claim is false. -/
theorem back_valid [Nonempty Coord] (sized : C.Sized vf) : (C.back vf).Valid where
  sound anchor tables failed := by
    show (C.backCheck vf (anchor.wire 0) tables (C.ran anchor)).Avoids _
    rcases found : C.ran anchor with _ | ⟨value, τ⟩
    · rintro ⟨claims, ended, -⟩
      cases ended
    · exact (Verifier.back_avoids vf sized.wf sized.matrix sized.columns sized.rows
        sized.key_fits (C.statement (anchor.wire 0) value) (tables 1) (tables 2) τ).mono
        fun _ sound passed => by
          obtain ⟨claims, ended, opened⟩ := passed
          have clean := sound claims ended (opened (not_imp.mp failed).1)
          exact (not_imp.mp failed).2 ⟨value, τ, found, clean.1, clean.2⟩
  staged anchor tables := by
    show Staged _ (C.backCheck vf (anchor.wire 0) tables (C.ran anchor)) _
    rcases C.ran anchor with _ | ⟨value, τ⟩
    · trivial
    · exact Verifier.back_staged vf sized.challenges exists_fair_slot
        (exists_fair_slots sized.column) sized.wf _ _ _ _
  nonnegative := Verifier.backCosts_nonnegative vf
  draws := List.cons_ne_nil _ _

omit [DecidableEq E] in
/-- The game of the checkpoint of `W` ends in the same result for each candidate. -/
theorem front_game (sized : C.Sized vf) (tables : Table × Table)
    {point : Prefix Wire (Coord → E) total} {result : Option (Lead × (Fin P.lookups → Zq))}
    (ended : C.front.game 0 point = .pure result) : C.front.game tables point = .pure result :=
  ended_of_bare C.read (Verifier.led_bare (sized.lead_bare _ _ _) _ _) ended

/-- The game of the checkpoint of `(W, V)` ends in the same result for each candidate. -/
theorem back_game (tables other : Fin 3 → Table) {point : Prefix Wire (Coord → E) total}
    {result : Option (Verifier.Finals E)} (ended : (C.back vf).game other point = .pure result) :
    (C.back vf).game tables point = .pure result :=
  ended_of_bare C.read (C.backCheck_bare vf _ _ _ _) ended

variable {Inner BreakT CandT KeyT : Type} {accept : Prefix Wire (Coord → E) total → Prop}
  {body : Structure (Prefix Wire (Coord → E) total) Coord E (ℕ → Wire) Inner BreakT CandT KeyT
    (Option B.Weak)}
  {key : (g : Fin 3) → Key (B.shape g)} {openKey : B.OpenKey} {mass : Fin 3 → ℕ}
  {commit : Wire → (g : Fin 3) → (B.shape g).Payload} {value : Wire → Fin 3 → E}
  {place : Wire → Fin 3 → List E} {error : ℝ} {loops tree : ℕ}

variable (C vf)

/-- What the verifier checks on an accepting transcript before the Akita opening. `led`: `lead`
ends in a value, and the transcript has the lookup challenges. `bound`: the commitments of the
Akita opening are the commitments that the stage input of the stage after `τ` fixes. `ended`:
`Verifier.back` ends in the opening claims, and the points and the values of the Akita opening
are these claims. The games do not read the tables (`Verifier.back_bare`), so the tables are zero
here. -/
structure Ran (commit : Wire → (g : Fin 3) → (B.shape g).Payload) (value : Wire → Fin 3 → E)
    (place : Wire → Fin 3 → List E) (point : Prefix Wire (Coord → E) total) : Prop where
  led : ∃ value τ, C.front.game 0 point = .pure (some (value, τ))
  bound : commit (point.wire (C.root vf)) = C.bound (point.take C.front.finish)
  ended : ∃ claims, (C.back vf).game (fun _ _ => 0) point = .pure (some claims) ∧
    place (point.wire (C.root vf)) = (fun g => (claimOf claims g).1) ∧
    value (point.wire (C.root vf)) = fun g => (claimOf claims g).2

/-- The relation of a first certificate of the stages after `τ`: certified decommitments of the
three commitments of the stage input, and the test of the checkpoint of `(W, V)`. -/
def Middle (key : (g : Fin 3) → Key (B.shape g)) (mass : Fin 3 → ℕ)
    (point : Prefix Wire (Coord → E) total) (opening : Option B.Weak) : Prop :=
  Decommits key mass (fun _ => True) (C.bound (point.take C.front.finish)) opening ∧
    (C.back vf).test (point.take C.front.finish) (Batch.tablesOf opening)

omit [Fintype E] [DecidableEq E] [Fintype Coord] [DecidableEq Coord] in
/-- The relation of a first certificate of the stages from `start`: certified decommitments of
the commitments of the key polynomial and of `W` of the stage input, and `good` for the table of
`W` when the table of the key polynomial is the table of `Params.key`. -/
def Extracted (key : (g : Fin 3) → Key (B.shape g)) (mass : Fin 3 → ℕ)
    (point : Prefix Wire (Coord → E) total) (opening : Option B.Weak) : Prop :=
  Decommits key mass (· ≠ 2) (C.bound (point.take C.start)) opening ∧
    (Batch.tablesOf opening 0 = keyBase P → C.good (point.wire 0) (Batch.tablesOf opening 1))

/-- The checkpoint structure of the stages after `τ`: the checkpoint of `(W, V)` over the
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
suffix (Akita Theorem 10.45 for two segments). The stage errors add the bounds `leadCosts` of the
draws of `lead` and `runCost`, the sum of the bounds of the draws of one run of the verifier
game, the retry loops add 2, and the tree size is multiplied by 4. `ran`: the verifier accepts
only with the conditions `Ran`. `embedded`: `embed` maps a solution of Module-SIS for a matrix of
a group to a witness of `R_bind` of the Akita opening. `later`: the stages of the Akita opening are a suffix whose first certificates satisfy
`OpensAt`.

Definition 10.16 (vii)(c) for the checkpoint of `W`: a first certificate of the stages after
`τ` passed the test of the checkpoint of `(W, V)`, so its own table `V` has no error for
the lookup challenges of its transcript. (vi) for both checkpoints: `Decommits.collide_isSome`. -/
theorem suffix [Nonempty Coord] (sized : C.Sized vf)
    (ran : ∀ point, accept point → C.Ran vf commit value place point)
    (embed : B.LevelBreak key openKey
      (fun g => 2 * (2 * mass g) * (B.fold g).sourceBound (B.shape g)) → BreakT)
    (embedded : ∀ solution, solution.isSome → body.breaks (embed solution))
    (later : Tier.Suffix accept (C.root vf) (OpensAt (C.root vf) commit value place key mass)
      body error loops tree) :
    Tier.Suffix accept C.start (C.Extracted key mass)
      (C.chain vf accept body key openKey mass embed)
      (C.leadCosts.sum + Verifier.runCost vf + error) (2 + loops) (4 * tree) := by
  have decommits : ∀ point opening, accept point →
      OpensAt (C.root vf) commit value place key mass point opening →
      Decommits key mass (fun _ => True) (C.bound (point.take C.front.finish)) opening := by
    rintro point _ accepted ⟨⟨ω, rfl, certified⟩, _⟩
    refine ⟨ω, rfl, fun g _ => ?_⟩
    rw [← (ran point accepted).bound]
    exact certified g
  have early : ∀ point opening, C.Middle vf key mass point opening →
      Decommits key mass (· ≠ 2) (C.bound (point.take C.start)) opening := by
    rintro point _ ⟨⟨ω, rfl, certified⟩, _⟩
    refine ⟨ω, rfl, fun g small => ?_⟩
    have group := certified g trivial
    rwa [sized.early _ g small,
      Prefix.take_take (show C.start ≤ C.front.finish from Nat.le_add_right _ _)] at group
  have middle := (C.back vf).suffix_before (back_valid sized) later sized.fits Batch.tablesOf
    (fun opening other => embed (Batch.collide key openKey _ opening other))
    (C.Middle vf key mass)
    (fun point opening opens => by
      show Decommits key mass _ (C.bound ((point.take C.front.finish).take C.front.finish))
          opening ∧
        (C.back vf).test ((point.take C.front.finish).take C.front.finish) (Batch.tablesOf opening)
      rw [Prefix.take_take le_rfl]
      exact opens)
    (fun point opening accepted opens good => ⟨decommits point opening accepted opens, good⟩)
    (fun point opening accepted opens => by
      obtain ⟨claims, ended, places, values⟩ := (ran point accepted).ended
      refine ⟨some claims, back_game _ _ ended, claims, rfl, fun honest => ?_⟩
      have claimed := opens.2
      rw [Prefix.wire_take le_rfl, places, values] at claimed
      have keyClaim : mle claims.key.1 (lifted E (Batch.tablesOf opening 0)) = claims.key.2 :=
        congrFun claimed 0
      rw [honest, lifted_keyBase] at keyClaim
      exact ⟨keyClaim.symm, (congrFun claimed 1).symm, (congrFun claimed 2).symm⟩)
    (fun point other opening opening' accepted accepted' opens opens' same different => by
      have same : point.take C.front.finish = other.take C.front.finish := same
      have right := decommits other opening' accepted' opens'
      rw [← same] at right
      obtain ⟨g, differs⟩ := Function.ne_iff.mp different
      exact embedded _ ((decommits point opening accepted opens).collide_isSome openKey right
        trivial differs))
  have result := C.front.suffix_before (front_valid sized) middle
    (show C.front.finish ≤ total from (Nat.le_add_right _ _).trans sized.fits)
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
        C.good ((point.take C.start).wire 0) (Batch.tablesOf opening 1) := good
      rwa [Prefix.wire_take (Nat.zero_le _)] at good⟩)
    (fun point opening accepted opens => by
      obtain ⟨value, τ, ended⟩ := (ran point accepted).led
      refine ⟨some (value, τ), front_game sized _ ended, fun honest => ?_⟩
      obtain ⟨value', τ', found, apart, clean⟩ := opens.2 honest
      have same : some (value', τ') = some (value, τ) := by
        rw [← found, C.ran_take]
        unfold Compose.ran
        rw [ended]
        rfl
      cases same
      rw [Prefix.wire_take (Nat.zero_le _)] at clean
      refine ⟨value, τ, rfl, apart, Batch.tablesOf opening 2, ?_⟩
      show ¬ Verifier.Fails P (C.statement ((point.take C.start).wire 0) value) _ _ τ
      rw [Prefix.wire_take (Nat.zero_le _)]
      exact clean)
    (fun point other opening opening' accepted accepted' opens opens' same different => by
      have same : point.take C.start = other.take C.start := same
      have right := early other opening' opens'
      rw [← same] at right
      by_cases zero : Batch.tablesOf opening 0 = Batch.tablesOf opening' 0
      · exact embedded _ ((early point opening opens).collide_isSome openKey right (g := 1)
          (by decide) fun one => different (Prod.ext zero one))
      · exact embedded _ ((early point opening opens).collide_isSome openKey right (g := 0)
          (by decide) zero))
  have costs : (C.leadCosts ++ [Verifier.tauError P]).sum +
      ((Verifier.backCosts vf).sum + error) = C.leadCosts.sum + Verifier.runCost vf + error := by
    rw [List.sum_append, List.sum_singleton, ← Verifier.backCosts_sum]
    ring
  have count : 1 + (1 + loops) = 2 + loops := by omega
  have size : 2 * (2 * tree) = 4 * tree := by omega
  rw [← costs, ← count, ← size]
  exact result

omit [Field E] [Algebra Zq E] [Fintype E] [DecidableEq E] [Fintype Coord] [DecidableEq Coord] in
/-- A first certificate with a certified decommitment `honest` of the key commitment to the
table of `Params.key` gives `good` for its table of `W`, or its decommitment of the key
commitment and `honest` give a solution of Module-SIS for a matrix of the key group
(Akita Corollary 10.25). -/
theorem Extracted.relation_or_collision {point : Prefix Wire (Coord → E) total}
    {opening : Option B.Weak} (extracted : C.Extracted key mass point opening)
    {honest : Certified (B.shape 0)} {factorBound sourceBound innerBound : ℕ}
    (registered : honest.Opens factorBound sourceBound (key 0) (C.bound (point.take C.start) 0)
      (keyBase P))
    (covers : factorBound * (B.fold 0).sourceBound (B.shape 0) + 2 * mass 0 * sourceBound ≤
      innerBound) :
    C.good (point.wire 0) (Batch.tablesOf opening 1) ∨
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
returns no output is at most
`(Q + 1)·(Σ leadCosts + runCost + error) + (Q + 1)·(2 + loops)/budget`. `runCost` is the sum of
the bounds of the draws of `Verifier.front` and `Verifier.back`, and `error` and `loops` are the
stage errors and the retry loops of the Akita opening. -/
theorem extraction_le {queries : ℕ} (bounded : prover.QueryBound queries) {budget : ℕ}
    (positive : 0 < budget) :
    𝔼 table, 𝔼 tape : Tape (Prefix Wire (Coord → E) total) Coord E budget
          (Structure.protocol whole).stages,
        (if accept ((Structure.protocol whole).final prover table) ∧
          Checkpoint.extract whole prover budget table tape = none then (1 : ℝ) else 0) ≤
      (queries + 1) * (C.leadCosts.sum + Verifier.runCost vf + error) +
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
