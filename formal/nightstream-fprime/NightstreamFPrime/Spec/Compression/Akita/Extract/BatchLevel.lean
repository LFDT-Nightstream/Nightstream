import NightstreamFPrime.Spec.Compression.Akita.Batch
import NightstreamFPrime.Spec.Compression.Akita.Extract.Recursion
import NightstreamFPrime.Spec.Compression.Akita.Extract.Root

/-! A fold with opening groups as a level of the opening suffix of Akita, and the batching root
before it (ePrint 2026/1983, revision of 2026-10-15, Corollary 10.32, Proposition 10.44 and
Theorem 10.46, "Batching root" and "Root claims").

- `Batch.method` is the opening method `Method` of a fold with opening groups and coefficient
  packing. Its fold coordinates are the pairs of a group and a source block, and the coordinate
  `(g, i)` has the challenge ring of the group `g`. `Batch.method_sound` is
  `Batch.extract_opens_or_collision` for it. A `Tier` with this method is a level with groups, and
  `Tier.suffix_before` is its step of the suffix.
- `Batch.root` is the batching root for the claims `f̃_g(r_g) = v_g` of the groups: the candidate
  has one table for each group.
- `Batch.root_suffix` puts the batching root before a suffix whose first level is a fold with
  groups: a first certificate has a certified decommitment of the commitment of each group, and
  each claim holds for the table of its group (Akita (173) with `φ = 1`). -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita

open scoped BigOperators
open NightstreamFPrime.Spec.Compression.Lib (mle)
open NightstreamFPrime.Spec.Checkpoint

attribute [local instance low] Classical.propDecidable

namespace Batch

variable {G : Type} [Fintype G] {B : Batch G} {E : Type}

/-- The claim of one level with opening groups (Akita Definition 10.22, "The fixed folding
statement", without the matrices): the payloads `u_pub,g` and `v_pub`, the weights of the opening
point of each group, the weights `ϑ_g` of the scalar claims and the target `v_root`. -/
structure Claim (B : Batch G) (E : Type) where
  payload : (g : G) → (B.shape g).Payload
  openPayload : B.Payload
  point : (g : G) → Point (B.opening g) E
  weight : G → E
  value : E

omit [Fintype G] in
/-- The folding statement of a claim for the matrices of the level. -/
@[reducible] def Claim.statement (key : (g : G) → Key (B.shape g)) (openKey : B.OpenKey)
    (claim : B.Claim E) : B.Statement E :=
  ⟨key, openKey, claim.payload, claim.openPayload, claim.point, claim.weight, claim.value⟩

/-- A witness of `R_bind` for the matrices of one level with opening groups, or `none`: a solution
of Module-SIS for a matrix `A_g` with the bound `innerBound g`, for a matrix `B_g` with the bound
`b⋆_g − 1`, for `D` with the bound `max_g (b⋆_g − 1)`, or for a compression map with the bound 1
(Akita Table 3). -/
abbrev LevelBreak (key : (g : G) → Key (B.shape g)) (openKey : B.OpenKey) (innerBound : G → ℕ) :
    Type :=
  Option ((Σ g, ShortKernel (key g).inner (innerBound g)) ⊕
    ((Σ g, OuterBreak (key g)) ⊕ B.OpenBreak openKey))

omit [Fintype G] in
/-- The children of the fold coordinates as a `Batch.Fork`: the challenge of the coordinate
`(g, i)` is the challenge `c_{g,i}`. -/
def pair (first second : B.Block → ((b : B.Block) → Rq (B.carrierDegree b.1)) × B.Witness) :
    B.Fork :=
  ⟨fun b => ⟨fun g i => (first b).1 ⟨g, i⟩, (first b).2⟩,
    fun b => ⟨fun g i => (second b).1 ⟨g, i⟩, (second b).2⟩⟩

variable [CommRing E] [Module Zq E]

/-- The method of a fold with opening groups and coefficient packing, with one claim in each
group (Akita Sections 5.1 and 5.4): `Batch.Witness.Satisfies`, the weak opening
`Batch.Weak.Opens` with `κ̄_{1,g} = 2ω_g` and `β̄_{∞,g} = Δ^cert_{f,g}` to the tables that the
source blocks of the groups decode to, and `Batch.forkCollision` with `Batch.extract`. The fold
coordinate `(g, i)` has the challenge ring `R_{q,d_car,g}`. `mass g` is the norm `ω_g = ‖c‖_1` of
a fold challenge of the group `g`. -/
noncomputable def method (β : ∀ g, Module.Basis (Fin (B.planes g)) Zq E)
    (key : (g : G) → Key (B.shape g)) (openKey : B.OpenKey) (mass : G → ℕ) :
    Extract.Method B.Block (fun b => Rq (B.carrierDegree b.1)) (B.Claim E)
      (B.LevelBreak key openKey fun g => 2 * (2 * mass g) * (B.fold g).sourceBound (B.shape g))
      B.Witness B.Weak where
  Satisfies w claim challenge :=
    w.Satisfies β (claim.statement key openKey) fun g i => challenge ⟨g, i⟩
  Opens ω claim :=
    ω.Opens β (fun g => 2 * mass g) (fun g => (B.fold g).sourceBound (B.shape g))
      (claim.statement key openKey) fun g => (B.shape g).decode (ω.certified g).source
  merge claim first second w :=
    match B.forkCollision (claim.statement key openKey) (pair first second) w.toCarried with
    | some solution => .inr (some (.inr solution))
    | none => .inl (extract (pair first second) w.toCarried)
  broken solution := solution.isSome

/-- Akita Corollary 10.32 for the method `Batch.method`. The challenge family of each group has
units as differences (Akita Definition 9.2, E1) and members of the norm `ω_g`, and
`b_{z,g} ≥ 2`. -/
theorem method_sound {β : ∀ g, Module.Basis (Fin (B.planes g)) Zq E}
    {key : (g : G) → Key (B.shape g)} {openKey : B.OpenKey} {mass peak : G → ℕ}
    {family : (g : G) → Set (Rq (B.carrierDegree g))} (units : ∀ g, UnitDifferences (family g))
    (sparse : ∀ g, ∀ challenge ∈ family g, Sparse (mass g) (peak g) challenge)
    (base : ∀ g, 2 ≤ (B.fold g).responseBase) :
    (method β key openKey mass).Sound fun b => family b.1 := by
  intro claim first second w challenge firstMember secondMember different agree firstSatisfies
    secondSatisfies good
  have result := extract_opens_or_collision units sparse base (fork := pair first second)
    (st := claim.statement key openKey)
    ⟨fun g i => firstMember ⟨g, i⟩, fun g i => secondMember ⟨g, i⟩, fun g i => different ⟨g, i⟩,
      fun b g j other => agree b ⟨g, j⟩ other, firstSatisfies, secondSatisfies⟩
    (Witness.Satisfies.carried good)
  unfold method
  rcases collision : B.forkCollision (claim.statement key openKey) (pair first second)
      w.toCarried with _ | solution
  · rw [collision] at result
    simp only [collision]
    exact result.resolve_left (by simp)
  · simp only [collision]
    rfl

omit [Fintype G] [CommRing E] [Module Zq E] in
/-- The tables of the successor source of a weak opening: for each group the table that its
source blocks decode to (Akita (201)). -/
noncomputable def tablesOf : Option B.Weak → G → Table
  | some ω => fun g => (B.shape g).decode (ω.certified g).source
  | none => fun _ _ => 0

omit [CommRing E] [Module Zq E] in
/-- `Coll_ℓ` for two weak openings of a level with opening groups: the solution of
`tableCollision` for the first group whose certified decommitments give one. -/
noncomputable def collide (key : (g : G) → Key (B.shape g)) (openKey : B.OpenKey)
    (innerBound : G → ℕ) (opening other : Option B.Weak) : B.LevelBreak key openKey innerBound :=
  match opening, other with
  | some ω, some ω' =>
      Finset.univ.toList.findSome? fun g =>
        (tableCollision (key g) (innerBound g) (ω.certified g) (ω'.certified g)).map fun
          | .inl solution => .inl ⟨g, solution⟩
          | .inr solution => .inr (.inl ⟨g, solution⟩)
  | _, _ => none

end Batch

namespace Extract

open NightstreamFPrime.Spec.Compression (Game)

variable {count : ℕ} {B : Batch (Fin count)} {E : Type} [Field E] [Fintype E] [DecidableEq E]
  [Algebra Zq E] {Wire Coord Slot : Type} {total : ℕ}

/-- The batching root for the claims of the opening groups (Akita (173) with `φ = 1`, and
Theorem 10.46, "Batching root": the test `R_app`). The message before the stage `start` has the
claimed value `value g` and the opening point `place g` of each group. A candidate has one table
for each group, and its value for the group `g` is the multilinear evaluation of its table at the
point of the group. -/
noncomputable def claimsRoot (start : ℕ) (read : Reader Wire) (value : Wire → Fin count → E)
    (place : Wire → Fin count → List E) :
    Root E Wire Coord Slot (Fin count → Table) count total where
  start := start
  read := read
  claimed anchor := value (anchor.wire start)
  actual anchor tables g := mle (place (anchor.wire start) g) (lifted E (tables g))

variable [Fintype Slot] [Nonempty Slot] [Fintype Coord] [DecidableEq Coord]
  {β : ∀ g, Module.Basis (Fin (B.planes g)) Zq E} {key : (g : Fin count) → Key (B.shape g)}
  {openKey : B.OpenKey} {mass : Fin count → ℕ} {Result Source : Type}
  (T : Tier B.Block (fun b => Rq (B.carrierDegree b.1)) (B.Claim E)
    (B.LevelBreak key openKey fun g => 2 * (2 * mass g) * (B.fold g).sourceBound (B.shape g))
    B.Witness B.Weak Wire Coord Slot Result Source)

omit [Fintype E] [DecidableEq E] [Fintype Slot] [Nonempty Slot] [Fintype Coord]
  [DecidableEq Coord] in
/-- The link of the verifier from the batching root at the stage `start` to the root fold `T` on
a transcript (Akita Section 5.4, "Combining the scalar claims"). The payload of the claim of `T`
for a group is the commitment `commit` of the message before the stage `start`. The weights of the
claim of `T` are `weights`, and its target is `Σ_g ϑ_g·v_g` (Akita (72) with `φ = 1`). The opening
point of the group `g` has the segments `(r_pack, r_tail, r_pos, r_blk)` of the lengths of the
shape of the group. -/
def RootLinked (start : ℕ) (commit : Wire → (g : Fin count) → (B.shape g).Payload)
    (value : Wire → Fin count → E) (place : Wire → Fin count → List E) (weights : Fin count → E)
    (point : Prefix Wire (Coord → Slot) total) : Prop :=
  (T.claim (point.wire T.start)).payload = commit (point.wire start) ∧
    (T.claim (point.wire T.start)).weight = weights ∧
    (T.claim (point.wire T.start)).value = ∑ g, weights g * value (point.wire start) g ∧
    ∀ g, ∃ block position pack tail,
      (B.shape g).blocks = 2 ^ block.length ∧ (B.shape g).blockLength = 2 ^ position.length ∧
      (B.opening g).stride = 2 ^ pack.length ∧ B.carrierDegree g = 2 ^ tail.length ∧
      (T.claim (point.wire T.start)).point g =
        Point.ofSegments (B.opening g) block position pack tail ∧
      place (point.wire start) g = pack ++ (tail ++ (position ++ block))

variable {T} {Inner BreakT CandT KeyT : Type} {accept : Prefix Wire (Coord → Slot) total → Prop}
  {body : Structure (Prefix Wire (Coord → Slot) total) Coord Slot (ℕ → Wire) Inner BreakT CandT
    KeyT (Option B.Weak)}
  {error : ℝ} {loops tree : ℕ}

/-- The batching root before a suffix whose first level `T` is a fold with opening groups and
coefficient packing (Akita Theorem 10.46, "Batching root" and "Root claims", for independent
weights). A first certificate of the result has, for each group, a certified decommitment of the
commitment of the group to the table that its source blocks decode to, and the claim
`f̃_g(r_g) = v_g` of the group holds for this table: the two conjuncts of Akita (173) with `φ = 1`.
The value of a group is the multilinear evaluation of its table by the partials of the weak
opening (`scalar_contract_eq_mle`). The stage errors add `1/|E|`, the retry loops add 1, and the
tree size is multiplied by 2. `fair`: an answer of the oracle gives a uniform vector of one weight
for each group. `packed`: the method of `T` is `Batch.method`. `linked`: the verifier accepts
only a transcript with the stage `start` and the link `RootLinked` for the weights of the
transcript. `embedded`: `embed` maps a solution of Module-SIS for a matrix of `T` to a witness of
`R_bind` of the later stages. Two certificates with different tables, after one stage input of the
stage `start`, give a solution for a matrix `A_g`, `B_g` or a map `F_a` of a group
(Akita Corollary 10.25). -/
theorem root_suffix (start : ℕ) (read : Reader Wire) (value : Wire → Fin count → E)
    (place : Wire → Fin count → List E)
    (fair : ∃ decode : (Coord → Slot) → Fin count → E, Fair decode)
    (packed : T.method = Batch.method β key openKey mass) (next : T.start = start + 1)
    (fits : start + 1 ≤ total) (commit : Wire → (g : Fin count) → (B.shape g).Payload)
    (linked : ∀ point, accept point → start < point.rest.length ∧
      RootLinked T start commit value place
        ((claimsRoot (Coord := Coord) (Slot := Slot) (total := total) start read value
          place).weights point) point)
    (embed : B.LevelBreak key openKey
      (fun g => 2 * (2 * mass g) * (B.fold g).sourceBound (B.shape g)) → BreakT)
    (embedded : ∀ solution, solution.isSome → body.breaks (embed solution))
    (later : Tier.Suffix accept T.start T.Opens body error loops tree) :
    Tier.Suffix accept start
      (fun point opening =>
        (∃ ω, opening = some ω ∧ ∀ g, (ω.certified g).Opens (2 * mass g)
          ((B.fold g).sourceBound (B.shape g)) (key g) (commit (point.wire start) g)
          ((B.shape g).decode (ω.certified g).source)) ∧
        (fun g => mle (place ((point.take start).wire start) g)
          (lifted E (Batch.tablesOf opening g))) = value ((point.take start).wire start))
      ((claimsRoot start read value place).segment.over accept body
        (fun inner => Batch.tablesOf (body.witness inner))
        (fun inner other => embed (Batch.collide key openKey _ (body.witness inner)
          (body.witness other)))
        fun point opening =>
          (∃ ω, opening = some ω ∧ ∀ g, (ω.certified g).Opens (2 * mass g)
            ((B.fold g).sourceBound (B.shape g)) (key g) (commit (point.wire start) g)
            ((B.shape g).decode (ω.certified g).source)) ∧
          (fun g => mle (place ((point.take start).wire start) g)
            (lifted E (Batch.tablesOf opening g))) = value ((point.take start).wire start))
      (1 / Fintype.card E + error) (1 + loops) (2 * tree) := by
  have opened : ∀ point opening, accept point → T.Opens point opening →
      ∃ ω, opening = some ω ∧
        ω.Opens β (fun g => 2 * mass g) (fun g => (B.fold g).sourceBound (B.shape g))
          ((T.claim (point.wire T.start)).statement key openKey)
          (fun g => (B.shape g).decode (ω.certified g).source) := by
    rintro point _ _ ⟨ω, rfl, opens⟩
    rw [packed] at opens
    exact ⟨ω, rfl, opens⟩
  have certified : ∀ point opening, accept point → T.Opens point opening →
      ∃ ω, opening = some ω ∧ ∀ g, (ω.certified g).Opens (2 * mass g)
        ((B.fold g).sourceBound (B.shape g)) (key g) (commit (point.wire start) g)
        ((B.shape g).decode (ω.certified g).source) := by
    intro point opening accepted opens
    obtain ⟨ω, rfl, weak⟩ := opened point opening accepted opens
    obtain ⟨_, payload, _⟩ := linked point accepted
    refine ⟨ω, rfl, fun g => ?_⟩
    have group := (weak.groups g).certified
    have same : (T.claim (point.wire T.start)).payload g = commit (point.wire start) g :=
      congrFun payload g
    rw [← same]
    exact group
  refine (claimsRoot start read value place).suffix_before fair (next ▸ later) fits
    Batch.tablesOf
    (fun opening other => embed (Batch.collide key openKey _ opening other))
    (fun point opening => ∃ ω, opening = some ω ∧ ∀ g, (ω.certified g).Opens (2 * mass g)
      ((B.fold g).sourceBound (B.shape g)) (key g) (commit (point.wire start) g)
      ((B.shape g).decode (ω.certified g).source))
    (fun point opening kept => by
      have same : (point.take start).wire start = point.wire start :=
        Prefix.wire_take le_rfl point
      show ∃ ω, opening = some ω ∧ ∀ g, (ω.certified g).Opens _ _ (key g)
        (commit ((point.take start).wire start) g) _
      rw [same]
      exact kept)
    (fun point opening accepted opens => ?_) ?_
  · obtain ⟨long, _, weight, target, places⟩ := linked point accepted
    refine ⟨long, certified point opening accepted opens, ?_⟩
    obtain ⟨ω, rfl, weak⟩ := opened point opening accepted opens
    have row := weak.value
    unfold Batch.scalar at row
    rw [show ((T.claim (point.wire T.start)).statement key openKey).value =
      (T.claim (point.wire T.start)).value from rfl, target] at row
    have wire : (point.take start).wire start = point.wire start := Prefix.wire_take le_rfl point
    show ∑ g, _ * mle (place ((point.take start).wire start) g)
      (lifted E ((B.shape g).decode (ω.certified g).source)) =
        ∑ g, _ * value ((point.take start).wire start) g
    rw [wire, ← row]
    refine Finset.sum_congr rfl fun g _ => ?_
    obtain ⟨block, position, pack, tail, blocks, length, stride, degree, placed, listed⟩ :=
      places g
    have planes : (B.opening g).partialPlane (fun column => ω.partials ⟨g, column⟩) =
        fun i t => contract (β g) ((T.claim (point.wire T.start)).point g) t
          (gadget (B.shape g).sourceBase ((ω.certified g).source i)) :=
      funext fun i => funext fun t => (weak.groups g).planes i t
    show _ = ((T.claim (point.wire T.start)).weight g) * Akita.scalar (β g)
      ((T.claim (point.wire T.start)).point g)
      ((B.opening g).partialPlane fun column => ω.partials ⟨g, column⟩)
    rw [planes, placed, scalar_contract_eq_mle (β g) block position pack tail blocks length stride
      degree, listed, weight]
    rfl
  · rintro point other opening opening' accepted accepted' opens opens' same different
    obtain ⟨ω, rfl, left⟩ := certified point opening accepted opens
    obtain ⟨ω', rfl, right⟩ := certified other opening' accepted' opens'
    have sameWire : other.wire start = point.wire start := by
      have taken : (other.take start).wire start = (point.take start).wire start :=
        congrArg (fun anchor : Prefix Wire (Coord → Slot) total => anchor.wire start) same.symm
      rwa [Prefix.wire_take le_rfl, Prefix.wire_take le_rfl] at taken
    obtain ⟨g, differs⟩ := Function.ne_iff.mp different
    refine embedded _ ?_
    show (Finset.univ.toList.findSome? _).isSome
    refine List.findSome?_isSome_iff.mpr ⟨g, Finset.mem_toList.mpr (Finset.mem_univ g), ?_⟩
    rw [Option.isSome_map]
    have rightAt := right g
    rw [sameWire] at rightAt
    exact tableCollision_isSome (left g) rightAt (le_of_eq (by ring)) differs

end Extract

end NightstreamFPrime.Spec.Compression.Akita
