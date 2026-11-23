import NightstreamFPrime.Spec.Compression.Akita.Extract.ZeroFit

/-! The honest prover of a fold with opening groups whose groups have their own commitment
states, and the batching root for a claim of the weights (Akita, ePrint 2026/1983, revision of
2026-11-23, Theorem 10.9 with Lemma 10.7, Section 5.4 with (69) to (72), Lemma 7.1, item 1,
Section 4.5 and Section 9 with (136)).

The groups of a fold need not have the same commitment algorithm: the group of the setup prefix
of an offloaded fold has the full-field source of `commit`, and the group of the witness has the
bounded source of `commitSource`.

- `Batch.pairOf` is a dependent function on `Fin 2` from its two values.
- `Batch.stateClaim` is the claim of the honest prover for a payload and a commitment state of
  each group, and `Batch.stateClaim_satisfies` states that the witness `Batch.honest` satisfies
  `Batch.Witness.Satisfies` for it when each state is an algebraic decommitment of its payload and
  each response is encodable. `Batch.honestClaim` is `Batch.stateClaim` for the states of
  `commit` (`Batch.stateClaim_commit`).
- `stateNext` is the table that the next level commits to, `StateFit` has the fit conditions,
  and `stateLevel_wins` gives the honest prover of the checks of the level `liftBatch`.
- `rootLink` is the batching root: the honest prover sends a claim for the weights of the
  transcript, and each transcript with this message has the link `RootLinked`.
- `Batch.honest_flat_zero_of` and `stateFit_zero`: for states with zero digit vectors, the flat
  witness of the honest prover is the zero table, and the fit conditions hold when `after` holds
  for the zero table. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita

open scoped BigOperators
open NightstreamFPrime.Spec.Compression.Lib (mle)

local instance : Fact (Nat.Prime fieldModulus) := ⟨FieldPrime.fieldModulus_natPrime⟩

namespace Batch

/-- The dependent function on `Fin 2` with the values `first` and `second`. -/
def pairOf {F : Fin 2 → Type*} (first : F 0) (second : F 1) : (g : Fin 2) → F g :=
  Fin.cases first fun g => Fin.cases second (fun h => h.elim0) g

theorem pairOf_zero {F : Fin 2 → Type*} (first : F 0) (second : F 1) :
    pairOf (F := F) first second 0 = first := rfl

theorem pairOf_one {F : Fin 2 → Type*} (first : F 0) (second : F 1) :
    pairOf (F := F) first second 1 = second := rfl

variable {n : ℕ} {B : Batch (Fin n)} {E : Type} [Field E] [Algebra Zq E]

/-- The claim of the honest prover of a fold with opening groups for the payload `(state g).1`
and the commitment state `(state g).2` of each group, the opening points and the weights `ϑ_g`
(Akita Figure 4, step 1, and (72)): the payload of each group, the payload `v_pub` of the shared
path, and the target `Σ_g ϑ_g·v_g` for the values (59) of the partials. -/
noncomputable def stateClaim (β : ∀ g, Module.Basis (Fin (B.planes g)) Zq E)
    (openKey : B.OpenKey)
    (state : (g : Fin n) → (B.shape g).Payload × Decommitment (B.shape g))
    (point : (g : Fin n) → Point (B.opening g) E) (weight : Fin n → E) : B.Claim E where
  payload g := (state g).1
  openPayload := (honestOpening openKey
    (honestPartials β point fun g => (state g).2.source)).2
  point := point
  weight := weight
  value := ∑ g, weight g * Akita.scalar (O := B.opening g) (β g) (point g) fun i t =>
    contract (β g) (point g) t (gadget (B.shape g).sourceBase ((state g).2.source i))

variable {β : ∀ g, Module.Basis (Fin (B.planes g)) Zq E} {key : (g : Fin n) → Key (B.shape g)}
  {openKey : B.OpenKey}

theorem stateClaim_commit (table : Fin n → Table) (point : (g : Fin n) → Point (B.opening g) E)
    (weight : Fin n → E) :
    stateClaim β openKey (fun g => commit (key g) (table g)) point weight =
      honestClaim β key openKey table point weight := rfl

/-- The deterministic part of Akita Lemma 10.7 for a fold with opening groups and a commitment
state of each group: when each state is an algebraic decommitment of its payload (`opens`) and
the response of each group is admissible, the honest witness satisfies the branch relation
`Batch.Witness.Satisfies` of the claim `stateClaim`. `opening`, `bounded`, `radix`, `certified`:
the conditions of `honest_satisfies` on each group. `path`: the condition `PathFits` of the shared
maps `H_1`, `H_2`. The source of a state has no condition. -/
theorem stateClaim_satisfies
    (state : (g : Fin n) → (B.shape g).Payload × Decommitment (B.shape g))
    (point : (g : Fin n) → Point (B.opening g) E) (weight : Fin n → E) (challenge : B.Challenge)
    (opening : ∀ g, Base (B.shape g).openingBase (B.shape g).openingDigits)
    (bounded : ∀ g, (B.shape g).openingBase ≤ (B.shape g).certifiedBase)
    (radix : ∀ g, Radix (B.fold g).responseBase)
    (certified : ∀ g, (B.fold g).responseBase ≤ (B.shape g).certifiedBase)
    (path : PathFits B.openDegree (1 * B.openRank) B.chain)
    (opens : ∀ g, ∃ table, (state g).2.Opens (key g) (state g).1 table)
    (encodable : ∀ g, Encodable (B.fold g).responseBase (B.fold g).responseDigits
      (response (B.opening g) (challenge g) (state g).2.source)) :
    (honest β openKey point challenge fun g => (state g).2).Satisfies β
      ((stateClaim β openKey state point weight).statement key openKey) challenge := by
  have rows := pathCommit_rows path openKey.chain (stack openKey.matrix fun _ : Fin 1 =>
    honestPartials β point fun g => (state g).2.source)
  refine ⟨fun g => ?_, rows.1, rows.2, ?_⟩
  · obtain ⟨table, opened⟩ := opens g
    have full := honest_satisfies (O := B.opening g) (fold := B.fold g) (β := β g)
      (st := ⟨key g, openKey.group g, (state g).1, (Akita.honestOpening (openKey.group g)
        (Akita.honestPartials (β g) (point g) (state g).2.source)).2, point g,
        Akita.scalar (β g) (point g) fun i t =>
          contract (β g) (point g) t (gadget (B.shape g).sourceBase ((state g).2.source i))⟩)
      (challenge g) (opening g) (bounded g) (radix g) (certified g) path opened (encodable g) rfl
      rfl
    exact ⟨⟨full.carried.rows, full.carried.innerDigits, full.carried.chainDigits,
      full.carried.partialDigits⟩, full.evaluation, full.innerRows, full.responseDigits⟩
  · have decoded : ∀ g, (B.opening g).partialPlane ((B.opening g).encode fun i t =>
        contract (β g) (point g) t
          (gadget (B.shape g).sourceBase ((state g).2.source i))) =
        fun i t => contract (β g) (point g) t
          (gadget (B.shape g).sourceBase ((state g).2.source i)) :=
      fun g => funext fun i => funext fun t =>
        (B.opening g).partialPlane_encode (opening g) _ i t
    exact Finset.sum_congr rfl fun g _ =>
      congrArg (fun planes => weight g * Akita.scalar (O := B.opening g) (β g) (point g) planes)
        (decoded g)

/-- The flat witness of the honest prover of a fold with opening groups is the zero table when
the source blocks, the digits `t̂` and the digits of the `F` maps of each state are 0. -/
theorem honest_flat_zero_of (openings : ∀ g, 2 ≤ (B.shape g).openingBase)
    (radix : ∀ g, 2 ≤ (B.fold g).responseBase) (β : ∀ g, Module.Basis (Fin (B.planes g)) Zq E)
    (openKey : B.OpenKey) (point : (g : Fin n) → Point (B.opening g) E) (c : B.Challenge)
    {state : (g : Fin n) → Decommitment (B.shape g)}
    (zero : ∀ g, (state g).source = 0 ∧ (state g).inner = 0 ∧
      PathDigits.firstTable (B.shape g).chain (state g).chain = 0 ∧
      PathDigits.secondTable (B.shape g).chain (state g).chain = 0) :
    (honest β openKey point c state).flat = 0 := by
  have partials : honestPartials β point (fun g => (state g).source) = 0 :=
    funext fun column => by
      show Akita.honestPartials (O := B.opening column.1) (β column.1) (point column.1)
        (state column.1).source column.2 = 0
      rw [(zero column.1).1, honestPartials_zero (openings column.1)]
      rfl
  have opened : PathDigits.firstTable B.chain (honestOpening openKey
        (honestPartials β point fun g => (state g).source)).1 = 0 ∧
      PathDigits.secondTable B.chain (honestOpening openKey
        (honestPartials β point fun g => (state g).source)).1 = 0 := by
    have image : stack openKey.matrix (fun _ : Fin 1 => (0 : B.Partials)) = 0 :=
      stack_zero openKey.matrix
    unfold honestOpening
    rw [partials, image]
    exact pathCommit_zero _ _
  have group : (fun g => ((honest β openKey point c state).group g).flat) = fun _ => 0 :=
    funext fun g => Akita.Witness.flat_eq_zero
      (by
        show decompose (B.fold g).responseBase
          (response (B.opening g) (c g) (state g).source) = 0
        rw [(zero g).1, response_zero, decompose_zero (radix g)])
      (funext fun column => congrFun partials ⟨g, column⟩) (zero g).2.1 (zero g).2.2.1
      (zero g).2.2.2 opened.1 opened.2
  have first : (fun g => PathDigits.firstTable (B.shape g).chain (state g).chain) =
      fun _ => 0 := funext fun g => (zero g).2.2.1
  have second : (fun g => PathDigits.secondTable (B.shape g).chain (state g).chain) =
      fun _ => 0 := funext fun g => (zero g).2.2.2
  funext x
  unfold Witness.flat
  rw [group]
  show join _ _ (join _ (concat _ fun g => PathDigits.firstTable (B.shape g).chain
    (state g).chain) (join _ (PathDigits.firstTable B.chain (honestOpening openKey
      (honestPartials β point fun g => (state g).source)).1) (join _ (concat _ fun g =>
        PathDigits.secondTable (B.shape g).chain (state g).chain) (join _
          (PathDigits.secondTable B.chain (honestOpening openKey
            (honestPartials β point fun g => (state g).source)).1) _)))) x = _
  rw [first, second, opened.1, opened.2, concat_zero, concat_zero, concat_zero]
  simp only [join]
  split_ifs <;> rfl

end Batch

namespace Extract

open NightstreamFPrime.Spec.Checkpoint

attribute [local instance low] Classical.propDecidable

section Groups

variable {n : ℕ} {B : Batch (Fin n)} {E : Type} [Field E] [Algebra Zq E]

/-- The table that the next level commits to after a fold with opening groups and a commitment
state of each group on the quotient-lift route: the flat witness of the honest prover with the
digits of the quotients of its ring rows at the places of Akita (76) (Figure 4, step 3). -/
noncomputable def stateNext (β : ∀ g, Module.Basis (Fin (B.planes g)) Zq E)
    (key : (g : Fin n) → Key (B.shape g)) (openKey : B.OpenKey) (μ : ℕ) (cells : B.cells ≤ 2 ^ μ)
    (base digits : ℕ) (state : (g : Fin n) → (B.shape g).Payload × Decommitment (B.shape g))
    (point : (g : Fin n) → Point (B.opening g) E) (weight : Fin n → E) (challenge : B.Challenge) :
    Table :=
  (B.flatRelation β ((Batch.stateClaim β openKey state point weight).statement key openKey)
    challenge μ cells).completion (B.gaps digits) base digits
    (Batch.honest β openKey point challenge fun g => (state g).2).flat

/-- The fit conditions of the honest prover of a fold with opening groups for the states `state`
(Akita Definition 10.5 and Lemma 10.7): for each opening point and weight of the groups and each
fold challenge vector of the families, the response `z_g = Σ_i ι(c_{g,i})·s_{g,i}` of each group
satisfies `Enc`, and `after` holds for the table `stateNext`. -/
def StateFit (β : ∀ g, Module.Basis (Fin (B.planes g)) Zq E)
    (key : (g : Fin n) → Key (B.shape g)) (openKey : B.OpenKey) (μ : ℕ) (cells : B.cells ≤ 2 ^ μ)
    (base digits : ℕ) (decode : (g : Fin n) → E → Option (Rq (B.carrierDegree g)))
    (state : (g : Fin n) → (B.shape g).Payload × Decommitment (B.shape g))
    (after : Table → Prop) : Prop :=
  ∀ (point : (g : Fin n) → Point (B.opening g) E) (weight : Fin n → E) (challenge : B.Challenge),
    (∀ g i, ∃ slot, decode g slot = some (challenge g i)) →
    (∀ g, Encodable (B.fold g).responseBase (B.fold g).responseDigits
      (response (B.opening g) (challenge g) (state g).2.source)) ∧
      after (stateNext β key openKey μ cells base digits state point weight challenge)

variable {β : ∀ g, Module.Basis (Fin (B.planes g)) Zq E} {key : (g : Fin n) → Key (B.shape g)}
  {openKey : B.OpenKey}

/-- The fit conditions `StateFit` hold for states with zero digit vectors, when `after` holds
for the zero table. `opening`, `bounded`, `radix`, `certified`, `path`, `opens`: the conditions
of `Batch.stateClaim_satisfies`. `two`: `2 ≤ base`. -/
theorem stateFit_zero (opening : ∀ g, Base (B.shape g).openingBase (B.shape g).openingDigits)
    (bounded : ∀ g, (B.shape g).openingBase ≤ (B.shape g).certifiedBase)
    (radix : ∀ g, Radix (B.fold g).responseBase)
    (certified : ∀ g, (B.fold g).responseBase ≤ (B.shape g).certifiedBase)
    (path : PathFits B.openDegree (1 * B.openRank) B.chain)
    {state : (g : Fin n) → (B.shape g).Payload × Decommitment (B.shape g)}
    (opens : ∀ g, ∃ table, (state g).2.Opens (key g) (state g).1 table)
    (zero : ∀ g, (state g).2.source = 0 ∧ (state g).2.inner = 0 ∧
      PathDigits.firstTable (B.shape g).chain (state g).2.chain = 0 ∧
      PathDigits.secondTable (B.shape g).chain (state g).2.chain = 0)
    {base : ℕ} (two : 2 ≤ base) {μ : ℕ} {cells : B.cells ≤ 2 ^ μ} {digits : ℕ}
    {decode : (g : Fin n) → E → Option (Rq (B.carrierDegree g))} {after : Table → Prop}
    (holds : after 0) :
    StateFit β key openKey μ cells base digits decode state after := by
  intro point weight c _
  have encodable : ∀ g, Encodable (B.fold g).responseBase (B.fold g).responseDigits
      (response (B.opening g) (c g) (state g).2.source) := fun g => by
    rw [(zero g).1, response_zero]
    exact encodable_zero (radix g).two _
  refine ⟨encodable, ?_⟩
  have satisfies := Batch.stateClaim_satisfies (β := β) (key := key) (openKey := openKey) state
    point weight c opening bounded radix certified path opens encodable
  have rows := satisfies.flat μ cells
  have flat := Batch.honest_flat_zero_of (fun g => (opening g).two) (fun g => (radix g).two) β
    openKey point c (state := fun g => (state g).2) zero
  unfold stateNext
  rw [flat] at rows ⊢
  rw [FlatRelation.completion_zero _ _ _ two rows.1]
  exact holds

/-- The fit conditions `StateFit` hold for each family of states when each fold challenge of the
decoders is 0 (`quiet`) and `after` holds for the table of the zero challenges. -/
theorem stateFit_quiet (radix : ∀ g, 2 ≤ (B.fold g).responseBase) {base μ : ℕ}
    {cells : B.cells ≤ 2 ^ μ} {digits : ℕ}
    {decode : (g : Fin n) → E → Option (Rq (B.carrierDegree g))}
    (quiet : ∀ g slot c, decode g slot = some c → c = 0)
    {state : (g : Fin n) → (B.shape g).Payload × Decommitment (B.shape g)} {after : Table → Prop}
    (holds : ∀ point weight,
      after (stateNext β key openKey μ cells base digits state point weight fun _ _ => 0)) :
    StateFit β key openKey μ cells base digits decode state after := by
  intro point weight c decodable
  obtain rfl : c = fun _ _ => 0 := funext fun g => funext fun i => by
    obtain ⟨slot, decoded⟩ := decodable g i
    exact quiet g slot _ decoded
  exact ⟨fun g => by rw [response_quiet]; exact encodable_zero (radix g) _, holds point weight⟩

variable [Fintype E] [DecidableEq E] {Wire Coord : Type} {claim : Wire → B.Claim E} {μ : ℕ}
  {cells : B.cells ≤ 2 ^ μ} {base digits : ℕ} {room : B.quotientRoom digits ≤ 2 ^ μ} {batch : ℕ}
  {rowsFit : B.rowCount + 1 ≤ 2 ^ batch} {block : B.Block → Coord}
  {decode : (g : Fin n) → E → Option (Rq (B.carrierDegree g))} {read : Reader Wire}
  {mass : Fin n → ℕ} {start : ℕ}

/-- The honest prover of the checks of a fold with opening groups and a commitment state of each
group on the quotient-lift route (Akita Lemma 10.7 and Lemma 7.1, item 1, with Theorem 10.9). The
message `wire` has the claim `Batch.stateClaim`, and `fit` has the fit conditions. Then for each
fold challenge vector of the families, `after` holds for the table `stateNext`, this table is 0
from the end `Batch.quotientRoom` of the quotient digits, and the prover has a message for each
draw sequence of the checks for it. `opening`, `bounded`, `radix`, `certified`, `path`, `opens`:
the conditions of `Batch.stateClaim_satisfies`. `uniform`, `digit`, `covers`: each group has the
certified base `base`, which has the conditions `Radix`, and each element of `Z_q` has `digits`
balanced digits. -/
theorem stateLevel_wins {sendable : (M : Type) → M → Prop} {slots width cap : ℕ}
    (sends : Sends sendable E width cap) (range : rangeWidth base ≤ width) (four : 4 ≤ width)
    (positive : 0 < slots)
    (opening : ∀ g, Base (B.shape g).openingBase (B.shape g).openingDigits)
    (bounded : ∀ g, (B.shape g).openingBase ≤ (B.shape g).certifiedBase)
    (radix : ∀ g, Radix (B.fold g).responseBase)
    (certified : ∀ g, (B.fold g).responseBase ≤ (B.shape g).certifiedBase)
    (path : PathFits B.openDegree (1 * B.openRank) B.chain)
    (uniform : ∀ g, (B.shape g).certifiedBase = base) (digit : Radix base)
    (covers : fieldModulus ≤ base ^ digits)
    {state : (g : Fin n) → (B.shape g).Payload × Decommitment (B.shape g)}
    (opens : ∀ g, ∃ table, (state g).2.Opens (key g) (state g).1 table)
    {point : (g : Fin n) → Point (B.opening g) E} {weight : Fin n → E}
    {wire : Wire} (ready : claim wire = Batch.stateClaim β openKey state point weight)
    {after : Table → Prop} (fit : StateFit β key openKey μ cells base digits decode state after)
    {c : (b : B.Block) → Rq (B.carrierDegree b.1)}
    (member : ∀ b, ∃ slot, decode b.1 slot = some (c b)) :
    after (stateNext β key openKey μ cells base digits state point weight fun g i => c ⟨g, i⟩) ∧
      (∀ x, B.quotientRoom digits ≤ x →
        stateNext β key openKey μ cells base digits state point weight (fun g i => c ⟨g, i⟩) x =
          0) ∧
      WinsAt sendable slots
        ((liftBatch β key openKey claim μ cells base digits room batch rowsFit block decode read
          mass start).check wire c
          (stateNext β key openKey μ cells base digits state point weight fun g i => c ⟨g, i⟩))
        (foldDraws μ base B.compressed) 0 fun result _ place => place ≤ 1 ∧
          ∃ r2 : List E, r2.length = μ ∧ result = some (r2, mle r2 (lifted E
            (stateNext β key openKey μ cells base digits state point weight
              fun g i => c ⟨g, i⟩))) := by
  obtain ⟨encodable, later⟩ := fit point weight (fun g i => c ⟨g, i⟩) fun g i => member ⟨g, i⟩
  have satisfies := Batch.stateClaim_satisfies (β := β) (key := key) (openKey := openKey) state
    point weight (fun g i => c ⟨g, i⟩) opening bounded radix certified path opens encodable
  rw [← ready] at satisfies
  have wins := liftBatch_winsAt (cells := cells) (room := room) (batch := batch)
    (rowsFit := rowsFit) (block := block) (decode := decode) (read := read) (mass := mass)
    (start := start) sends range four uniform digit covers positive satisfies
  rw [ready] at wins
  exact ⟨later, fun x large => FlatRelation.completion_of_le _
    (Batch.flatRelation_counts β _ _ μ cells digits) _
    ((Batch.flatRelation_quotientEnd β _ _ μ cells digits).trans_le large), wins⟩

end Groups

section Root

variable {count : ℕ} {B : Batch (Fin count)} {E : Type} [Field E] [Fintype E] [DecidableEq E]
  [Algebra Zq E] {Wire Coord : Type} [Fintype Coord] [DecidableEq Coord] {total : ℕ}
  {accept : Prefix Wire (Option Coord → E) total → Prop}
  {key : (g : Fin count) → Key (B.shape g)} {openKey : B.OpenKey} {mass : Fin count → ℕ}
  {Result Source : Type}
  {T : Tier B.Block (fun b => Rq (B.carrierDegree b.1)) (B.Claim E)
    (B.LevelBreak key openKey fun g => 2 * (2 * mass g) * (B.fold g).sourceBound (B.shape g))
    B.Witness B.Weak Wire Coord E Result Source}
  {sendable : (M : Type) → M → Prop} {slots : ℕ} {read : Reader Wire}
  {commit : Wire → (g : Fin count) → (B.shape g).Payload} {value : Wire → Fin count → E}
  {place : Wire → Fin count → List E}

omit [DecidableEq E] in
/-- The batching root of the honest prover (Akita Section 5.4, "Combining the scalar claims", and
Theorem 10.46, "Batching root"). `early` is a transcript prefix of `start` moves on the path of
the table of the oracle, and its last message has the commitments `commit`, the values `value` and
the points `place` of the groups. The answer of the stage `start` gives the weights `ϑ`, and the
honest prover sends the claim `target ϑ` of the fold `T` that starts after this stage, with a
nonce of the response search below `responseTries`. `paid`,
`weighed`, `valued`, `pointed`: for each weight vector, the claim has the commitments of the
message, the weights, the target `Σ_g ϑ_g·v_g`, and a point with the segments of the shape of
each group whose concatenation is the point of the message. Then each transcript with this
message has the stage `start` and the link `RootLinked` for its weights. -/
theorem rootLink (carries : Carries sendable slots read T.claim) {start : ℕ}
    (begins : T.start = start + 1)
    (table : Prefix Wire (Option Coord → E) total → Option Coord → E)
    (worked : ∀ input, table input none ∈ workZeros E 57)
    {first : Wire} {early : List ((Option Coord → E) × Wire)} (on : OnPath accept table first early)
    (starts : early.length = start) (rooted : start + 1 ≤ total)
    (target : (Fin count → E) → B.Claim E)
    (paid : ∀ ϑ, (target ϑ).payload = commit (lastWire first early))
    (weighed : ∀ ϑ, (target ϑ).weight = ϑ)
    (valued : ∀ ϑ, (target ϑ).value = ∑ g, ϑ g * value (lastWire first early) g)
    (pointed : ∀ ϑ g, ∃ block position pack tail,
      (B.shape g).blocks ≤ 2 ^ block.length ∧ (B.shape g).blockLength = 2 ^ position.length ∧
      (B.opening g).stride = 2 ^ pack.length ∧ B.carrierDegree g = 2 ^ tail.length ∧
      (target ϑ).point g = Point.ofSegments (B.opening g) block position pack tail ∧
      place (lastWire first early) g = pack ++ (tail ++ (position ++ block))) :
    ∃ (ϑ : Fin count → E) (wire : Wire), T.claim wire = target ϑ ∧
      read.nonce wire < responseTries ∧
      OnPath accept table first (early ++ [(answer table first early, wire)]) ∧
      ∀ stage : Prefix Wire (Option Coord → E) total, stage.first = first →
        (∃ rest, stage.rest = early ++ (answer table first early, wire) :: rest) →
        start < stage.rest.length ∧
          RootLinked T start commit value place
            ((claimsRoot (Coord := Coord) (Slot := E) (total := total) start read value
              place).weights stage) stage := by
  obtain ⟨wire, claimed, tried, _⟩ := carries (target (sample (Coord → E) (Fin count → E)
    (view (answer table first early)))) [] (Nat.zero_le _) (by simp)
  refine ⟨_, wire, claimed, tried, on.snoc (by rw [starts]; exact rooted) wire,
    fun stage rooted' ⟨rest, shape⟩ => ?_⟩
  have before : stage.wire start = lastWire first early := by
    rw [← starts, ← rooted']
    exact Prefix.wire_length shape
  have after : stage.wire T.start = wire := by
    rw [begins]
    exact stage.wire_of shape (congrArg (· + 1) starts.symm)
  have drawn : (claimsRoot (Coord := Coord) (Slot := E) (total := total) start read value
      place).weights stage =
        sample (Coord → E) (Fin count → E) (view (answer table first early)) := by
    show ((stage.reply start).map fun reply =>
      sample (Coord → E) (Fin count → E) (view reply)).getD 0 = _
    rw [stage.reply_of shape starts.symm]
    rfl
  have zeroRoot : answer table first early none ∈ workZeros E (Root.rootBits count) := by
    rw [answer_eq table first (by omega)]
    exact workZeros_anti E (Root.rootBits_le count) (worked _)
  refine ⟨by rw [shape, List.length_append, List.length_cons, starts]; omega, ?_, ?_, ?_,
    fun g => ?_, answer table first early, stage.reply_of shape starts.symm, zeroRoot⟩
  · rw [after, claimed, before, paid]
  · rw [after, claimed, drawn, weighed]
  · rw [after, claimed, drawn, before, valued]
  · obtain ⟨block, position, pack, tail, blocks, length, stride, degree, point, joined⟩ :=
      pointed (sample (Coord → E) (Fin count → E) (view (answer table first early))) g
    exact ⟨block, position, pack, tail, blocks, length, stride, degree,
      by rw [after, claimed]; exact point, by rw [before]; exact joined⟩

end Root

end Extract

end NightstreamFPrime.Spec.Compression.Akita
