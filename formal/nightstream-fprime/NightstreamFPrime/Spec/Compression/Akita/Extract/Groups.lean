import NightstreamFPrime.Spec.Compression.Akita.Binding
import NightstreamFPrime.Spec.Compression.Akita.Extract.BatchLevel

/-! A level of the opening suffix of Akita whose successor is a fold with opening groups, and the
setup-prefix group of an offloaded fold (ePrint 2026/1983, revision of 2026-10-15, Section 9 with
(136), Lemma 9.1 with its proof in Appendix D.2, Corollary 10.25 with (162) to (164), and
Proposition 10.44).

- `Method.pinned` is a method with one more condition `Pin` on a weak opening: `merge` returns
  the witness `fix` of `R_bind` for a weak opening without `Pin`. `Method.pinned_sound` keeps Akita
  Corollary 10.32, and `Tier.pinned` is a level with this method.
- `Batch.Pinned` is the condition of Akita Lemma 9.1 for the group `g` of a fold with groups: when
  the payload of the group is the commitment `registry` of the verifier setup, the source of the
  group decodes to the table `table`. `Batch.pinBreak` is the reduction `collision` of Akita
  Corollary 10.25 for the certified decommitment of the group and the reference decommitment
  (162), and `Batch.pin_fixed` is the proof of Lemma 9.1: an opening without `Pinned` gives a
  solution of Module-SIS for a matrix of the group.
- `root_suffix_pinned` is `root_suffix` for a first level with a condition `Pin` on its
  openings: the first certificates of the batching stage keep `Pin`.
- `Tier.Offloaded` is the link of the verifier from an offloaded level to the two groups of the
  next level (Akita (136)): the group 0 has the commitment `registry` of the setup prefix and the
  claim `(ρ_S, s_{ρ_S})`, and the group 1 has the commitment of the proof and the claim
  `(r_2, v_W)`. `Tier.suffix_push_offloaded` is the step of the suffix for it: the offloaded
  level over the batching stage of the two claims. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita

open scoped BigOperators
open NightstreamFPrime.Spec.Compression.Lib (mle)
open NightstreamFPrime.Spec.Checkpoint

attribute [local instance low] Classical.propDecidable

namespace Extract

section Pinned

variable {Block : Type} {Ring : Block → Type} {Stmt Brk Wit Weak : Type}

/-- The method `M` with the condition `Pin` on its weak openings. `merge` returns the weak opening
of `M` when it has `Pin`, and the witness `fix` of `R_bind` when it does not. -/
noncomputable def Method.pinned (M : Method Block Ring Stmt Brk Wit Weak)
    (Pin : Weak → Stmt → Prop) (fix : Weak → Stmt → Brk) :
    Method Block Ring Stmt Brk Wit Weak where
  Satisfies := M.Satisfies
  Opens ω claim := M.Opens ω claim ∧ Pin ω claim
  merge claim first second w :=
    match M.merge claim first second w with
    | .inl ω => if Pin ω claim then .inl ω else .inr (fix ω claim)
    | .inr solution => .inr solution
  broken := M.broken

/-- Akita Corollary 10.32 for a method with a condition on its weak openings. `fixed`: a weak
opening of `M` without `Pin` gives a true witness `fix` of `R_bind`. -/
theorem Method.pinned_sound {M : Method Block Ring Stmt Brk Wit Weak}
    {Pin : Weak → Stmt → Prop} {fix : Weak → Stmt → Brk} {family : (b : Block) → Set (Ring b)}
    (sound : M.Sound family)
    (fixed : ∀ ω claim, M.Opens ω claim → ¬ Pin ω claim → M.broken (fix ω claim)) :
    (M.pinned Pin fix).Sound family := by
  intro claim first second w challenge firstMember secondMember different agree firstSatisfies
    secondSatisfies good
  have result := sound claim first second w challenge firstMember secondMember different agree
    firstSatisfies secondSatisfies good
  show Sum.elim _ _ (match M.merge claim first second w with
    | .inl ω => if Pin ω claim then .inl ω else .inr (fix ω claim)
    | .inr solution => .inr solution)
  generalize M.merge claim first second w = merged at result ⊢
  rcases merged with ω | solution
  · show Sum.elim _ _ (if Pin ω claim then Sum.inl ω else Sum.inr (fix ω claim))
    by_cases pin : Pin ω claim
    · rw [ite_eq_left pin]
      exact ⟨result, pin⟩
    · rw [ite_eq_right pin]
      exact fixed ω claim result pin
  · exact result

variable {Wire Coord Slot Result Source : Type}

/-- The level `T` with the condition `Pin` on its weak openings. -/
noncomputable def Tier.pinned (T : Tier Block Ring Stmt Brk Wit Weak Wire Coord Slot Result Source)
    (Pin : Weak → Stmt → Prop) (fix : Weak → Stmt → Brk) :
    Tier Block Ring Stmt Brk Wit Weak Wire Coord Slot Result Source :=
  { T with method := T.method.pinned Pin fix }

/-- A level with a condition on its weak openings satisfies the conditions of `Tier.Valid`. -/
theorem Tier.pinned_valid [∀ b, Zero (Ring b)] [Fintype Slot] [Nonempty Slot] [Fintype Coord]
    [DecidableEq Coord] {T : Tier Block Ring Stmt Brk Wit Weak Wire Coord Slot Result Source}
    {Pin : Weak → Stmt → Prop} {fix : Weak → Stmt → Brk} (valid : T.Valid)
    (fixed : ∀ ω claim, T.method.Opens ω claim → ¬ Pin ω claim → T.method.broken (fix ω claim)) :
    (T.pinned Pin fix).Valid where
  injective := valid.injective
  avoid := valid.avoid
  merges := Method.pinned_sound valid.merges fixed
  sound := valid.sound
  staged := valid.staged
  nonnegative := valid.nonnegative
  draws := valid.draws
  aligned := valid.aligned

end Pinned

end Extract

namespace Batch

variable {G : Type} [Fintype G] {B : Batch G} {E : Type} [CommRing E] [Module Zq E]

omit [Fintype G] [CommRing E] [Module Zq E] in
/-- The condition of Akita Lemma 9.1 on a weak opening for the group `g`: when the payload of the
group is the commitment `registry`, its source decodes to `table`. -/
def Pinned (g : G) (registry : (B.shape g).Payload) (table : Table) (ω : B.Weak)
    (claim : B.Claim E) : Prop :=
  claim.payload g = registry → (B.shape g).decode (ω.certified g).source = table

omit [Fintype G] [CommRing E] [Module Zq E] in
/-- A solution of Module-SIS for `A_g`, `B_g` or a compression map of the group `g` as a witness
of `R_bind` of the level. -/
def groupBreak (key : (g : G) → Key (B.shape g)) (openKey : B.OpenKey) (innerBound : G → ℕ)
    (g : G) (solution : Option (ShortKernel (key g).inner (innerBound g) ⊕ OuterBreak (key g))) :
    B.LevelBreak key openKey innerBound :=
  solution.map fun
    | .inl inner => .inl ⟨g, inner⟩
    | .inr outer => .inr (.inl ⟨g, outer⟩)

omit [Fintype G] [CommRing E] [Module Zq E] in
/-- The reduction of Akita Corollary 10.25 for the certified decommitment of the group `g` of a
weak opening and a reference decommitment (Akita (162)): a solution of Module-SIS for `A_g`,
`B_g` or a compression map of the group. -/
noncomputable def pinBreak (key : (g : G) → Key (B.shape g)) (openKey : B.OpenKey)
    (innerBound : G → ℕ) (g : G) (reference : Decommitment (B.shape g)) (ω : B.Weak) :
    B.LevelBreak key openKey innerBound :=
  groupBreak key openKey innerBound g (collision (key g) (innerBound g) (ω.certified g) reference)

/-- The proof of Akita Lemma 9.1 (Appendix D.2): a weak opening of `Batch.method` whose group `g`
has the payload `registry` and does not decode to `table` gives a solution of Module-SIS for a
matrix of the group. `registered`: `reference` is an algebraic decommitment of `registry` to
`table` (Akita (162)). `short`: its source has `‖s⁰_i‖_∞ ≤ ρ`. `covers`: `β̄_∞ + κ̄_1·ρ ≤ η_A`
(Akita (164)). -/
theorem pin_fixed {β : ∀ g, Module.Basis (Fin (B.planes g)) Zq E}
    {key : (g : G) → Key (B.shape g)} {openKey : B.OpenKey} {mass : G → ℕ} {g : G}
    {registry : (B.shape g).Payload} {table : Table} {reference : Decommitment (B.shape g)}
    {referenceBound : ℕ} (registered : reference.Opens (key g) registry table)
    (short : ∀ i, vecNormInf (reference.source i) ≤ referenceBound)
    (covers : (B.fold g).sourceBound (B.shape g) + 2 * mass g * referenceBound ≤
      2 * (2 * mass g) * (B.fold g).sourceBound (B.shape g))
    (ω : B.Weak) (claim : B.Claim E) (opens : (method β key openKey mass).Opens ω claim)
    (unpinned : ¬ Pinned g registry table ω claim) :
    (method β key openKey mass).broken (pinBreak key openKey _ g reference ω) := by
  obtain ⟨payload, different⟩ := Classical.not_imp.mp unpinned
  have certified : (ω.certified g).Opens (2 * mass g) ((B.fold g).sourceBound (B.shape g))
      (key g) (claim.payload g) ((B.shape g).decode (ω.certified g).source) :=
    (Weak.Opens.groups opens g).certified
  rw [payload] at certified
  show ((collision (key g) _ (ω.certified g) reference).map _).isSome
  rw [Option.isSome_map]
  exact collision_isSome certified registered short covers different

omit [Fintype G] [CommRing E] [Module Zq E] in
/-- The premise of Akita Lemma 9.1, "correct preprocessing": the commitment `registry` of the
verifier setup has the reference decommitment `reference` to the table `flat` of the setup prefix
(Akita (162)), and each coefficient of its source is a balanced digit of the base `b`. -/
structure Registered {S : Shape} (key : Key S) (registry : S.Payload) (flat : Table)
    (reference : Decommitment S) : Prop where
  opens : reference.Opens key registry flat
  short : ∀ i, vecNormInf (reference.source i) ≤ S.sourceBase / 2

omit [Fintype G] [CommRing E] [Module Zq E] in
/-- The commitment algorithm gives the premise `Registered` for its payload and the padded
table. -/
theorem registered_commit {S : Shape} (honest : S.Honest) (key : Key S) (table : Table) :
    Registered key (commit key table).1 (S.pad table) (commit key table).2 where
  opens := commit_opens honest key table
  short i := by
    have bound := (commit_certified honest key table le_rfl le_rfl).sourceNorm i
    rwa [one_smul] at bound

end Batch

namespace Extract

open NightstreamFPrime.Spec.Compression (Game)

section Root

variable {count : ℕ} {B : Batch (Fin count)} {E : Type} [Field E] [Fintype E] [DecidableEq E]
  [Algebra Zq E] {Wire Coord Slot : Type} {total : ℕ}
  [Fintype Slot] [Nonempty Slot] [Fintype Coord] [DecidableEq Coord]
  {β : ∀ g, Module.Basis (Fin (B.planes g)) Zq E} {key : (g : Fin count) → Key (B.shape g)}
  {openKey : B.OpenKey} {mass : Fin count → ℕ} {Result Source : Type}
  {T : Tier B.Block (fun b => Rq (B.carrierDegree b.1)) (B.Claim E)
    (B.LevelBreak key openKey fun g => 2 * (2 * mass g) * (B.fold g).sourceBound (B.shape g))
    B.Witness B.Weak Wire Coord Slot Result Source}
  {Inner BreakT CandT KeyT : Type} {accept : Prefix Wire (Option Coord → Slot) total → Prop}
  {body : Structure (Prefix Wire (Option Coord → Slot) total) (Option Coord) Slot (ℕ → Wire) Inner
    BreakT CandT
    KeyT (Option B.Weak)}
  {error : ℝ} {loops tree : ℕ}

/-- `root_suffix` for a first level `T` with a condition on its weak openings (Akita Theorem
10.46, "Batching root", and Section 9, (137) for independent weights): a first certificate of the
result has the certified decommitments and the claims of `root_suffix`, and the condition `Pin`.
`opened`: an opening of `T` is a weak opening of `Batch.method` with `Pin`. The other hypotheses
are those of `root_suffix`. -/
theorem root_suffix_pinned (start : ℕ) (read : Reader Wire) (value : Wire → Fin count → E)
    (place : Wire → Fin count → List E)
    (fair : ∃ decode : (Coord → Slot) → Fin count → E, Fair decode)
    (next : T.start = start + 1) (fits : start + 1 ≤ total)
    (commit : Wire → (g : Fin count) → (B.shape g).Payload) (Pin : Option B.Weak → Prop)
    (opened : ∀ point opening, accept point → T.Opens point opening →
      (∃ ω, opening = some ω ∧
        ω.Opens β (fun g => 2 * mass g) (fun g => (B.fold g).sourceBound (B.shape g))
          ((T.claim (point.wire T.start)).statement key openKey)
          (fun g => (B.shape g).decode (ω.certified g).source)) ∧ Pin opening)
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
        ((∃ ω, opening = some ω ∧ ∀ g, (ω.certified g).Opens (2 * mass g)
          ((B.fold g).sourceBound (B.shape g)) (key g) (commit (point.wire start) g)
          ((B.shape g).decode (ω.certified g).source)) ∧ Pin opening) ∧
        (fun g => mle (place ((point.take start).wire start) g)
          (lifted E (Batch.tablesOf opening g))) = value ((point.take start).wire start))
      ((claimsRoot start read value place).segment.over accept body
        (fun inner => Batch.tablesOf (body.witness inner))
        (fun inner other => embed (Batch.collide key openKey _ (body.witness inner)
          (body.witness other)))
        fun point opening =>
          ((∃ ω, opening = some ω ∧ ∀ g, (ω.certified g).Opens (2 * mass g)
            ((B.fold g).sourceBound (B.shape g)) (key g) (commit (point.wire start) g)
            ((B.shape g).decode (ω.certified g).source)) ∧ Pin opening) ∧
          (fun g => mle (place ((point.take start).wire start) g)
            (lifted E (Batch.tablesOf opening g))) = value ((point.take start).wire start))
      (1 / Fintype.card E / 2 ^ Root.rootBits count + error) (1 + loops) (2 * tree) := by
  have certified : ∀ point opening, accept point → T.Opens point opening →
      ∃ ω, opening = some ω ∧ ∀ g, (ω.certified g).Opens (2 * mass g)
        ((B.fold g).sourceBound (B.shape g)) (key g) (commit (point.wire start) g)
        ((B.shape g).decode (ω.certified g).source) := by
    intro point opening accepted opens
    obtain ⟨⟨ω, rfl, weak⟩, _⟩ := opened point opening accepted opens
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
    (fun point opening => (∃ ω, opening = some ω ∧ ∀ g, (ω.certified g).Opens (2 * mass g)
      ((B.fold g).sourceBound (B.shape g)) (key g) (commit (point.wire start) g)
      ((B.shape g).decode (ω.certified g).source)) ∧ Pin opening)
    (fun point opening kept => by
      have same : (point.take start).wire start = point.wire start :=
        Prefix.wire_take le_rfl point
      show (∃ ω, opening = some ω ∧ ∀ g, (ω.certified g).Opens _ _ (key g)
        (commit ((point.take start).wire start) g) _) ∧ Pin opening
      rw [same]
      exact kept)
    (fun point opening accepted opens => ?_) ?_
  · obtain ⟨long, _, weight, target, places, worked⟩ := linked point accepted
    refine ⟨long, ⟨certified point opening accepted opens,
      (opened point opening accepted opens).2⟩, ?_, worked⟩
    obtain ⟨⟨ω, rfl, weak⟩, _⟩ := opened point opening accepted opens
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
        congrArg (fun anchor : Prefix Wire (Option Coord → Slot) total => anchor.wire start)
          same.symm
      rwa [Prefix.wire_take le_rfl, Prefix.wire_take le_rfl] at taken
    obtain ⟨g, differs⟩ := Function.ne_iff.mp different
    refine embedded _ ?_
    show (Finset.univ.toList.findSome? _).isSome
    refine List.findSome?_isSome_iff.mpr ⟨g, Finset.mem_toList.mpr (Finset.mem_univ g), ?_⟩
    rw [Option.isSome_map]
    have rightAt := right g
    rw [sameWire] at rightAt
    exact tableCollision_isSome (left g) rightAt (le_of_eq (by ring)) differs

end Root

section Offloaded

variable {B : Batch (Fin 2)} {Block : Type} {Ring : Block → Type}
  {E Stmt Brk Wit Wk Wire Coord Slot : Type} [∀ b, Zero (Ring b)] [Fintype Block] [Field E]
  [Algebra Zq E]
  (T : Tier Block Ring Stmt Brk Wit Wk Wire Coord Slot (Option ((List E × E) × (List E × E)))
    Table)
  {total : ℕ} [Fintype Slot] [Nonempty Slot] [Fintype Coord] [DecidableEq Coord]
  {Inner BreakT CandT KeyT : Type}

/-- The link of the verifier from an offloaded level `T` to the two groups of the next level on a
transcript (Akita (136)). The transcript has the stages of `T`, each fold challenge decodes, and
the checks of `T` end in the claims `(r_2, v_W)` and `(ρ_S, s_{ρ_S})`. The message after the
checks has the claims of the next level: the group 0 has the commitment `registry` of the
verifier setup, the point `ρ_S` and the value `s_{ρ_S}`, and the group 1 has the commitment
`commit` of the message after the fold challenge of `T`, the point `r_2` and the value `v_W`. -/
def Tier.Offloaded (commits : Wire → (g : Fin 2) → (B.shape g).Payload) (value : Wire → Fin 2 → E)
    (place : Wire → Fin 2 → List E) (registry : (B.shape 0).Payload)
    (commit : Wire → (B.shape 1).Payload) (point : Prefix Wire (Option Coord → Slot) total) : Prop
      :=
  T.finish ≤ point.rest.length ∧ T.Decodes point ∧
    ∃ r2 v ρ s, T.game (fun _ => 0) point = .pure (some ((r2, v), (ρ, s))) ∧
      place (point.wire T.finish) = ![ρ, r2] ∧ value (point.wire T.finish) = ![s, v] ∧
      commits (point.wire T.finish) 0 = registry ∧
      commits (point.wire T.finish) 1 = commit (point.wire (T.start + 1))

/-- `Coll_ℓ` for two weak openings of a fold with groups and the group `g`: the solution of
`tableCollision` for the certified decommitments of the group. -/
noncomputable def groupCollide {G : Type} {B : Batch G} (g : G) (key : Key (B.shape g))
    (innerBound : ℕ) (opening other : Option B.Weak) :
    Option (ShortKernel key.inner innerBound ⊕ OuterBreak key) :=
  match opening, other with
  | some ω, some ω' => tableCollision key innerBound (ω.certified g) (ω'.certified g)
  | _, _ => none

variable {T} {accept : Prefix Wire (Option Coord → Slot) total → Prop}
  {tail : Structure (Prefix Wire (Option Coord → Slot) total) (Option Coord) Slot (ℕ → Wire) Inner
    BreakT CandT
    KeyT (Option B.Weak)}
  {error : ℝ} {loops tree : ℕ}

/-- Akita Proposition 10.44 for an offloaded level over the batching stage of the two claims of
Akita (136). `later` is the suffix of `root_suffix_pinned` at the stage after the checks of `T`:
a first certificate has a certified decommitment of each commitment, the table `flat` for the
group 0 of the setup prefix, and the two claims for its tables. The successor source of `T` is the
table of the group 1. The claim `(r_2, v_W)` is true by the claim of the group 1, and the claim
`(ρ_S, s_{ρ_S})` is true for `flat` by the claim of the group 0. Two sources after one stage
input of the checkpoint give a solution of Module-SIS for a matrix of the group 1 (Akita
Corollary 10.25). `uniform`: the verifier of the checks does not read the source. `passes`: the
final claim is true for a table `w` when `v_W = w̃(r_2)` and `s_{ρ_S}` is the value of `flat` at
`ρ_S`. `linked`: the verifier accepts only with the link `Tier.Offloaded`. -/
theorem Tier.suffix_push_offloaded {blocks : ℕ} (count : Fintype.card Block = blocks)
    (valid : T.Valid) (fits : T.finish ≤ total) (flat : Table)
    (uniform : ∀ wire challenge table other,
      bare (T.check wire challenge table) = bare (T.check wire challenge other))
    (passes : ∀ (table : Table) r2 v ρ s, v = mle r2 (lifted E table) →
      s = mle ρ (lifted E flat) → T.passes table (some ((r2, v), (ρ, s))))
    (commits : Wire → (g : Fin 2) → (B.shape g).Payload) (value : Wire → Fin 2 → E)
    (place : Wire → Fin 2 → List E) (registry : (B.shape 0).Payload)
    (commit : Wire → (B.shape 1).Payload)
    (linked : ∀ point, accept point → T.Offloaded commits value place registry commit point)
    (key : (g : Fin 2) → Key (B.shape g)) (mass : Fin 2 → ℕ)
    (embed : Option (ShortKernel (key 1).inner
      (2 * (2 * mass 1) * (B.fold 1).sourceBound (B.shape 1)) ⊕ OuterBreak (key 1)) → BreakT)
    (embedded : ∀ solution, solution.isSome → tail.breaks (embed solution))
    (later : Tier.Suffix accept T.finish
      (fun point opening =>
        ((∃ ω, opening = some ω ∧ ∀ g, (ω.certified g).Opens (2 * mass g)
          ((B.fold g).sourceBound (B.shape g)) (key g) (commits (point.wire T.finish) g)
          ((B.shape g).decode (ω.certified g).source)) ∧ Batch.tablesOf opening 0 = flat) ∧
        (fun g => mle (place ((point.take T.finish).wire T.finish) g)
          (lifted E (Batch.tablesOf opening g))) = value ((point.take T.finish).wire T.finish))
      tail error loops tree) :
    Tier.Suffix accept T.start T.Opens
      (T.over accept tail (fun inner => Batch.tablesOf (tail.witness inner) 1)
        (fun inner other => embed (groupCollide 1 (key 1) _ (tail.witness inner)
          (tail.witness other))))
      (T.levelError + error) (blocks + 1 + loops)
      (2 * (blocks + 1) * tree) := by
  refine Tier.suffix_before count valid later fits (fun opening => Batch.tablesOf opening 1)
    (fun opening other => embed (groupCollide 1 (key 1) _ opening other))
    (fun point accepted => (linked point accepted).2.1) ?_ ?_
  · rintro point _ accepted ⟨⟨⟨ω, rfl, _⟩, pin⟩, claims⟩
    obtain ⟨long, decodes, r2, v, ρ, s, ended, places, values, _, _⟩ := linked point accepted
    have wire : (point.take T.finish).wire T.finish = point.wire T.finish :=
      Prefix.wire_take le_rfl point
    rw [wire, places, values] at claims
    have witness := congrFun claims 1
    have setup := congrFun claims 0
    rw [pin] at setup
    exact ⟨long, decodes, _, ended_of_bare T.read (uniform _ _ _ _) ended,
      passes _ r2 v ρ s witness.symm setup.symm⟩
  · rintro point other _ _ accepted acceptedOther ⟨⟨⟨ω, rfl, left⟩, _⟩, _⟩
      ⟨⟨⟨ω', rfl, right⟩, _⟩, _⟩ same different
    obtain ⟨_, _, _, _, _, _, _, _, _, _, committed⟩ := linked point accepted
    obtain ⟨_, _, _, _, _, _, _, _, _, _, committed'⟩ := linked other acceptedOther
    have sameWire : other.wire (T.start + 1) = point.wire (T.start + 1) := by
      rw [← Prefix.wire_take le_rfl other, ← same, Prefix.wire_take le_rfl]
    have first := left 1
    have second := right 1
    rw [committed] at first
    rw [committed', sameWire] at second
    exact embedded _ (tableCollision_isSome first second (le_of_eq (by ring)) different)

end Offloaded

end Extract

end NightstreamFPrime.Spec.Compression.Akita
