import NightstreamFPrime.Spec.Compression.Akita.Extract.LiftComplete

/-! The honest prover of the root fold of Akita with opening groups, and the verifier `towerLinks`
of the opening (ePrint 2026/1983, revision of 2026-10-15, Theorem 10.9 with Lemma 10.7,
Section 5.4 with (69) to (72), Lemma 7.1, item 1, Figure 4 and Section 8.2). The theorems
`tower_suffix` and `mixedTower_suffix` bound an event of the verifier `towerLinks`. This file
shows that this verifier accepts the honest prover.

- `Batch.honest` is the witness of the honest prover of a fold with opening groups: the digit
  vectors of each group, and the digits `ê = concat(ê_0, …, ê_{G−1})` with the digits of the
  shared maps `H_1`, `H_2`. `Batch.honestClaim` is its claim for a table of each group, and
  `Batch.honestClaim_satisfies` states that it satisfies `Batch.Witness.Satisfies` when each
  response is encodable.
- `batchNext` is the table that the level 1 commits to, `BatchFit` has the fit conditions of the
  root fold, and `batchLevel_wins` gives the honest prover of the checks of the level
  `liftBatch`.
- `packLink` gives the transcript prefix of the honest prover through the stages of a level whose
  next level is a packing level with a bounded source: the first message has the commitment to
  the table, the last message has the claim `sourceClaim`, and each transcript with the prefix has
  the link `Tier.Linked`.
- `tower_complete` gives, for a transcript prefix on the path of the table of the oracle whose
  last message has the commitments, the points and the values of the three claims, the later
  messages of a proof such that the complete transcript satisfies `towerLinks`.

The fit conditions `BatchFit`, `LiftFit` and `TraceFit` are hypotheses (Akita Definition 10.5).
The file has no bound for the probability `ξ_h` of Akita (155). -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita

open scoped BigOperators
open NightstreamFPrime.Spec.Compression.Lib (mle)

local instance : Fact (Nat.Prime fieldModulus) := ⟨FieldPrime.fieldModulus_natPrime⟩

namespace Batch

variable {n : ℕ} {B : Batch (Fin n)} {E : Type} [Field E] [Algebra Zq E]

/-- The matrices of the opening path for the group `g`: the columns of `D` of the group, and the
shared maps `H_1`, `H_2`. -/
def OpenKey.group (openKey : B.OpenKey) (g : Fin n) : OpeningKey (B.opening g) :=
  ⟨fun row column => openKey.matrix row ⟨g, column⟩, openKey.chain⟩

/-- The digits `ê = concat(ê_0, …, ê_{G−1})` of the honest prover: the digits `honestPartials` of
each group (Akita (69)). -/
noncomputable def honestPartials (β : ∀ g, Module.Basis (Fin (B.planes g)) Zq E)
    (point : (g : Fin n) → Point (B.opening g) E)
    (source : (g : Fin n) → Fin (B.shape g).blocks → (B.shape g).Source) : B.Partials :=
  fun column =>
    Akita.honestPartials (O := B.opening column.1) (β column.1) (point column.1) (source column.1)
      column.2

/-- The digits of `H_1`, `H_2` and the payload `v_pub` of the shared opening path for the digits
`ê` (Akita (70)). -/
noncomputable def honestOpening (openKey : B.OpenKey) (partials : B.Partials) :
    PathDigits B.chain × B.Payload :=
  pathCommit B.chain openKey.chain (stack openKey.matrix fun _ : Fin 1 => partials)

/-- The witness of the honest prover of one fold with opening groups for the commitment states
`state g` and the challenges `c_{g,i}` (Akita Figure 4, steps 1 and 2, and Section 5.4). -/
noncomputable def honest (β : ∀ g, Module.Basis (Fin (B.planes g)) Zq E) (openKey : B.OpenKey)
    (point : (g : Fin n) → Point (B.opening g) E) (challenge : B.Challenge)
    (state : (g : Fin n) → Decommitment (B.shape g)) : B.Witness where
  inner g := (state g).inner
  chain g := (state g).chain
  partials := honestPartials β point fun g => (state g).source
  openChain := (honestOpening openKey (honestPartials β point fun g => (state g).source)).1
  response g :=
    decompose (B.fold g).responseBase (response (B.opening g) (challenge g) (state g).source)

/-- The claim of the honest prover of a fold with opening groups for a table of each group, the
opening points and the weights `ϑ_g` (Akita Figure 4, step 1, and (72)): the payload `u_pub,g` of
`commit` for each group, the payload `v_pub` of the shared path, and the target
`Σ_g ϑ_g·v_g` for the values (59) of the partials. -/
noncomputable def honestClaim (β : ∀ g, Module.Basis (Fin (B.planes g)) Zq E)
    (key : (g : Fin n) → Key (B.shape g)) (openKey : B.OpenKey) (table : Fin n → Table)
    (point : (g : Fin n) → Point (B.opening g) E) (weight : Fin n → E) : B.Claim E where
  payload g := (commit (key g) (table g)).1
  openPayload := (honestOpening openKey
    (honestPartials β point fun g => (commit (key g) (table g)).2.source)).2
  point := point
  weight := weight
  value := ∑ g, weight g * Akita.scalar (O := B.opening g) (β g) (point g) fun i t =>
    contract (β g) (point g) t
      (gadget (B.shape g).sourceBase ((commit (key g) (table g)).2.source i))

variable {β : ∀ g, Module.Basis (Fin (B.planes g)) Zq E} {key : (g : Fin n) → Key (B.shape g)}
  {openKey : B.OpenKey}

/-- The deterministic part of Akita Lemma 10.7 for a fold with opening groups: when the response
of each group is admissible, the honest witness satisfies the branch relation
`Batch.Witness.Satisfies` of the claim `honestClaim`. Each group has the conditions of
`honest_satisfies`, and `path` is the condition `PathFits` of the shared maps `H_1`, `H_2`. -/
theorem honestClaim_satisfies (table : Fin n → Table)
    (point : (g : Fin n) → Point (B.opening g) E) (weight : Fin n → E) (challenge : B.Challenge)
    (shape : ∀ g, (B.shape g).Honest) (radix : ∀ g, Radix (B.fold g).responseBase)
    (certified : ∀ g, (B.fold g).responseBase ≤ (B.shape g).certifiedBase)
    (path : PathFits B.openDegree (1 * B.openRank) B.chain)
    (encodable : ∀ g, Encodable (B.fold g).responseBase (B.fold g).responseDigits
      (response (B.opening g) (challenge g) (commit (key g) (table g)).2.source)) :
    (honest β openKey point challenge fun g => (commit (key g) (table g)).2).Satisfies β
      ((honestClaim β key openKey table point weight).statement key openKey) challenge := by
  have rows := pathCommit_rows path openKey.chain (stack openKey.matrix fun _ : Fin 1 =>
    honestPartials β point fun g => (commit (key g) (table g)).2.source)
  refine ⟨fun g => ?_, rows.1, rows.2, ?_⟩
  · have full := Extract.honestClaim_satisfies (O := B.opening g) (fold := B.fold g) (β := β g)
      (key := key g) (openKey := openKey.group g) (table g) (point g) (challenge g) (shape g)
      (radix g) (certified g) path (encodable g)
    exact ⟨⟨full.carried.rows, full.carried.innerDigits, full.carried.chainDigits,
      full.carried.partialDigits⟩, full.evaluation, full.innerRows, full.responseDigits⟩
  · have decoded : ∀ g, (B.opening g).partialPlane ((B.opening g).encode fun i t =>
        contract (β g) (point g) t
          (gadget (B.shape g).sourceBase ((commit (key g) (table g)).2.source i))) =
        fun i t => contract (β g) (point g) t
          (gadget (B.shape g).sourceBase ((commit (key g) (table g)).2.source i)) :=
      fun g => funext fun i => funext fun t =>
        (B.opening g).partialPlane_encode (shape g).opening _ i t
    exact Finset.sum_congr rfl fun g _ =>
      congrArg (fun planes => weight g * Akita.scalar (O := B.opening g) (β g) (point g) planes)
        (decoded g)

end Batch

namespace Extract

open NightstreamFPrime.Spec.Checkpoint

attribute [local instance low] Classical.propDecidable

section Groups

variable {n : ℕ} {B : Batch (Fin n)} {E : Type} [Field E] [Algebra Zq E]

/-- The table that the level 1 commits to after a fold with opening groups on the quotient-lift
route: the flat witness of the honest prover for the tables, the points, the weights and the
fold challenges, and then the digits of the quotients of its ring rows (Akita (76) and Figure 4,
step 3). -/
noncomputable def batchNext (β : ∀ g, Module.Basis (Fin (B.planes g)) Zq E)
    (key : (g : Fin n) → Key (B.shape g)) (openKey : B.OpenKey) (μ : ℕ) (cells : B.cells < 2 ^ μ)
    (base digits : ℕ) (table : Fin n → Table) (point : (g : Fin n) → Point (B.opening g) E)
    (weight : Fin n → E) (challenge : B.Challenge) : Table :=
  (B.lockedRelation β ((Batch.honestClaim β key openKey table point weight).statement key openKey)
    challenge μ cells).completion base digits (B.cells + 1)
    (Batch.honest β openKey point challenge fun g => (commit (key g) (table g)).2).flat

/-- The fit conditions of the honest prover of a fold with opening groups for the tables `table`
(Akita Definition 10.5 and Lemma 10.7): for each opening point and weight of the groups and each
fold challenge vector of the families, the response `z_g = Σ_i ι(c_{g,i})·s_{g,i}` of each group
satisfies `Enc`, and `after` holds for the table `batchNext`. Akita Theorem 10.9 has the
probability that a response does not fit in the term `ξ_h` of (155). -/
def BatchFit (β : ∀ g, Module.Basis (Fin (B.planes g)) Zq E)
    (key : (g : Fin n) → Key (B.shape g)) (openKey : B.OpenKey) (μ : ℕ) (cells : B.cells < 2 ^ μ)
    (base digits : ℕ) (decode : (g : Fin n) → E → Option (Rq (B.carrierDegree g)))
    (table : Fin n → Table) (after : Table → Prop) : Prop :=
  ∀ (point : (g : Fin n) → Point (B.opening g) E) (weight : Fin n → E) (challenge : B.Challenge),
    (∀ g i, ∃ slot, decode g slot = some (challenge g i)) →
    (∀ g, Encodable (B.fold g).responseBase (B.fold g).responseDigits
      (response (B.opening g) (challenge g) (commit (key g) (table g)).2.source)) ∧
      after (batchNext β key openKey μ cells base digits table point weight challenge)

variable [Fintype E] [DecidableEq E] {Wire Coord : Type}
  {β : ∀ g, Module.Basis (Fin (B.planes g)) Zq E} {key : (g : Fin n) → Key (B.shape g)}
  {openKey : B.OpenKey} {claim : Wire → B.Claim E} {μ : ℕ} {cells : B.cells < 2 ^ μ}
  {base digits : ℕ} {room : B.quotientRoom digits ≤ 2 ^ μ} {batch : ℕ}
  {rowsFit : B.rowCount + 1 + 1 ≤ 2 ^ batch} {block : B.Block → Coord}
  {decode : (g : Fin n) → E → Option (Rq (B.carrierDegree g))} {read : Reader Wire}
  {mass : Fin n → ℕ} {start : ℕ}

/-- The honest prover of the checks of the root fold with opening groups on the quotient-lift
route (Akita Lemma 10.7 and Lemma 7.1, item 1, with Theorem 10.9). The message `wire` has the
claim `Batch.honestClaim`, and `fit` has the fit conditions. Then for each fold challenge vector
of the families, `after` holds for the table `batchNext`, this table is 0 from the end
`Batch.quotientRoom` of the quotient digits, and the prover has a message for each draw sequence
of the checks for it. `shape`, `radix`, `certified`, `path`: the conditions of
`Batch.honestClaim_satisfies`. `uniform`, `digit`, `covers`: each group has the certified base
`base`, which has the conditions `Radix`, and each element of `Z_q` has `digits` balanced
digits. -/
theorem batchLevel_wins {sendable : (M : Type) → M → Prop} {slots width cap : ℕ}
    (sends : Sends sendable E width cap) (range : rangeWidth base ≤ width) (four : 4 ≤ width)
    (positive : 0 < slots) (shape : ∀ g, (B.shape g).Honest)
    (radix : ∀ g, Radix (B.fold g).responseBase)
    (certified : ∀ g, (B.fold g).responseBase ≤ (B.shape g).certifiedBase)
    (path : PathFits B.openDegree (1 * B.openRank) B.chain)
    (uniform : ∀ g, (B.shape g).certifiedBase = base) (digit : Radix base)
    (covers : ∀ x : Zq, -((base / 2 : ℕ) : ℤ) * ∑ u ∈ Finset.range digits, (base : ℤ) ^ u ≤
        x.valMinAbs ∧
      x.valMinAbs ≤ (((base / 2 : ℕ) : ℤ) - 1) * ∑ u ∈ Finset.range digits, (base : ℤ) ^ u)
    {table : Fin n → Table} {point : (g : Fin n) → Point (B.opening g) E} {weight : Fin n → E}
    {wire : Wire} (ready : claim wire = Batch.honestClaim β key openKey table point weight)
    {after : Table → Prop} (fit : BatchFit β key openKey μ cells base digits decode table after)
    {c : (b : B.Block) → Rq (B.carrierDegree b.1)}
    (member : ∀ b, ∃ slot, decode b.1 slot = some (c b)) :
    after (batchNext β key openKey μ cells base digits table point weight fun g i => c ⟨g, i⟩) ∧
      (∀ x, B.quotientRoom digits ≤ x →
        batchNext β key openKey μ cells base digits table point weight (fun g i => c ⟨g, i⟩) x =
          0) ∧
      WinsAt sendable slots
        ((liftBatch β key openKey claim μ cells base digits room batch rowsFit block decode read
          mass start).check wire c
          (batchNext β key openKey μ cells base digits table point weight fun g i => c ⟨g, i⟩))
        (foldDraws μ base) 0 fun result _ place => place ≤ 1 ∧
          ∃ r2 : List E, r2.length = μ ∧ result = some (r2, mle r2 (lifted E
            (batchNext β key openKey μ cells base digits table point weight
              fun g i => c ⟨g, i⟩))) := by
  obtain ⟨encodable, later⟩ := fit point weight (fun g i => c ⟨g, i⟩) fun g i => member ⟨g, i⟩
  have satisfies := Batch.honestClaim_satisfies (β := β) (key := key) (openKey := openKey) table
    point weight (fun g i => c ⟨g, i⟩) shape radix certified path encodable
  rw [← ready] at satisfies
  have wins := liftBatch_winsAt (cells := cells) (room := room) (batch := batch)
    (rowsFit := rowsFit) (block := block) (decode := decode) (read := read) (mass := mass)
    (start := start) sends range four uniform digit covers positive satisfies
  rw [ready] at wins
  exact ⟨later, fun x large => FlatRelation.completion_of_le _ _
    ((Batch.lockedRelation_quotientEnd β _ _ μ cells digits).trans_le large), wins⟩

end Groups

section Link

variable {E : Type} [Field E] [Fintype E] [DecidableEq E] [Algebra Zq E] {Block : Type}
  {Ring : Block → Type} {Stmt Brk Wit Wk Wire Coord : Type} [∀ b, Zero (Ring b)]
  [Fintype Coord] [DecidableEq Coord] {total : ℕ}
  {accept : Prefix Wire (Coord → E) total → Prop}
  {table : Prefix Wire (Coord → E) total → Coord → E}
  {T : Tier Block Ring Stmt Brk Wit Wk Wire Coord E (Option (List E × E)) Table}
  {S' : Shape} {O' : OpeningShape S'} {fold' : Descriptor} {Result' Source' : Type}
  {key' : Key S'} {openKey' : OpeningKey O'} {weight' : ℕ}
  {Next : Tier (Fin S'.blocks) (fun _ => Rq O'.carrierDegree) (Claim O' E)
    (LevelBreak key' openKey' (2 * (2 * weight') * fold'.sourceBound S')) (Witness O' fold')
    (Weak O') Wire Coord E Result' Source'}
  {commit' : Wire → S'.Payload} {sendable : (M : Type) → M → Prop} {slots : ℕ}

omit [DecidableEq E] in
/-- The transcript prefix of the honest prover through the stages of a level `T` whose next level
`Next` is a packing level with a bounded source (Akita Figure 4, steps 1, 3 and 5, and
Section 4.5, "Bounded source"). `wins`: the prover has a message for each draw sequence of the
checks of `T` for the table `source`, which end in a true claim `(r_2, w̃(r_2))` of `μ`
coordinates. The honest prover commits to the bounded source blocks of `source` in the message
after the fold challenges, and its last message has the claim `sourceClaim` of `Next` for the
point `r_2`. `p`, `t` and `s` are the lengths of the segments `r_pack`, `r_tail` and `r_pos` of
`r_2` for the shape of `Next`, and `r_blk` has `k` coordinates or more. `live`: the live blocks of
`Next` have each cell of `source` that is not 0 (Akita Section 4.5, (40) to (42)). Each
transcript with the prefix has the link `Tier.Linked` to `Next`. -/
theorem packLink
    (carries : Carries sendable slots T.read fun wire => (commit' wire, Next.claim wire))
    {first : Wire} {early : List ((Coord → E) × Wire)} (on : OnPath accept table first early)
    (starts : early.length = T.start) (fits : T.finish ≤ total)
    (decodes : ∀ input b, (T.decode b (table input (T.block b))).isSome)
    (uniform : ∀ wire challenge table other,
      bare (T.check wire challenge table) = bare (T.check wire challenge other))
    {draws μ : ℕ} (count : T.costs.length = draws) (source : Table)
    (wins : WinsAt sendable slots
      (T.check (lastWire first early) (T.drawn table first early) source) draws 0
      fun result _ _ => ∃ r2 : List E, r2.length = μ ∧
        result = some (r2, mle r2 (lifted E source)))
    (next : Next.start = T.finish) (shape' : S'.Bounded)
    (β' : Module.Basis (Fin O'.planes) Zq E) {p t s k : ℕ} (stride' : O'.stride = 2 ^ p)
    (degree' : O'.carrierDegree = 2 ^ t) (length' : S'.blockLength = 2 ^ s)
    (blocks' : S'.blocks ≤ 2 ^ k) (vars : μ = p + (t + (s + k)))
    (live : ∀ cell, S'.blocks * (S'.blockLength * S'.innerDegree) ≤ cell → source cell = 0) :
    ∃ (later : List ((Coord → E) × Wire)) (point : Point O' E),
      OnPath accept table first (early ++ later) ∧ (early ++ later).length = Next.start ∧
      Next.claim (lastWire first (early ++ later)) =
        sourceClaim β' key' openKey' (fun i => S'.embed (S'.pack source i)) point ∧
      ∀ stage : Prefix Wire (Coord → E) total, stage.first = first →
        (∃ rest, stage.rest = (early ++ later) ++ rest) → T.Linked Next commit' stage := by
  subst count
  obtain ⟨pointOf, pointed⟩ : ∃ pointOf : List E → Point O' E, pointOf = fun r =>
      Point.ofSegments O' (((r.drop p).drop t).drop s) (((r.drop p).drop t).take s) (r.take p)
        ((r.drop p).take t) := ⟨_, rfl⟩
  obtain ⟨wire, moves, result, counted, ⟨r2, long, rfl⟩, on', held, last, ended, facts⟩ :=
    Tier.run (T := T) carries on starts fits decodes source wins fun _ game =>
      ((commitSource key' fun i => S'.embed (S'.pack source i)).1,
        match game with
        | .pure (some (r, _)) =>
            sourceClaim β' key' openKey' (fun i => S'.embed (S'.pack source i)) (pointOf r)
        | _ => Next.claim first)
  have claimed : Next.claim (lastWire wire moves) =
      sourceClaim β' key' openKey' (fun i => S'.embed (S'.pack source i)) (pointOf r2) :=
    congrArg Prod.snd last
  have paid : commit' wire = (commitSource key' fun i => S'.embed (S'.pack source i)).1 :=
    congrArg Prod.fst held
  have whole : (early ++ (answer table first early, wire) :: moves).length = T.finish := by
    rw [List.length_append, List.length_cons, counted, starts]
    exact T.finish_eq.symm
  have joined : r2.take p ++ ((r2.drop p).take t ++ (((r2.drop p).drop t).take s ++
      ((r2.drop p).drop t).drop s)) = r2 := by simp
  have blocked : S'.blocks ≤ 2 ^ (((r2.drop p).drop t).drop s).length :=
    blocks'.trans (Nat.pow_le_pow_right (by decide) (by simp; omega))
  have lengthed : S'.blockLength = 2 ^ (((r2.drop p).drop t).take s).length := by
    rw [length']; congr 1; simp; omega
  have strided : O'.stride = 2 ^ (r2.take p).length := by
    rw [stride']; congr 1; simp; omega
  have degreed : O'.carrierDegree = 2 ^ ((r2.drop p).take t).length := by
    rw [degree']; congr 1; simp; omega
  refine ⟨(answer table first early, wire) :: moves, pointOf r2, on', whole.trans next.symm,
    by rw [lastWire_append_cons]; exact claimed, fun stage rooted extended => ?_⟩
  obtain ⟨long', decoded, _, opened, closed, game⟩ := facts stage rooted extended
  refine ⟨long', decoded, ((r2.drop p).drop t).drop s, ((r2.drop p).drop t).take s, r2.take p,
    (r2.drop p).take t, mle r2 (lifted E source), ?_, blocked, lengthed, strided, degreed, ?_, ?_,
    ?_⟩
  · rw [joined]
    exact (game _).trans (ended_of_bare T.read (uniform _ _ source fun _ => 0) ended)
  · rw [next, closed, claimed, opened, paid]
    rfl
  · rw [next, closed, claimed, pointed]
    rfl
  · rw [next, closed, claimed, pointed]
    exact (sourceClaim_value shape' source _ _ _ _ blocked lengthed strided degreed live).trans
      (congrArg (fun r => mle r (lifted E source)) joined)

end Link

section Tower

variable {E : Type} [Field E] [Fintype E] [DecidableEq E] [Algebra Zq E] [Degree2 E]
  {count : ℕ} {B : Batch (Fin count)} {S₁ : Shape} {O₁ : OpeningShape S₁} {fold₁ : Descriptor}
  {S₂ : Shape} {O₂ : OpeningShape S₂} {fold₂ : Descriptor} {S₃ : Shape} {O₃ : OpeningShape S₃}
  {fold₃ : Descriptor} {S₄ : Shape} {O₄ : OpeningShape S₄} {fold₄ : Descriptor} {S₅ : Shape}
  {O₅ : OpeningShape S₅} {fold₅ : Descriptor} {S₆ : Shape}
  {Wire Coord : Type} [Fintype Coord] [DecidableEq Coord] {total : ℕ}
  {accept : Prefix Wire (Coord → E) total → Prop} {read : Reader Wire} {start : ℕ}
  {β : ∀ g, Module.Basis (Fin (B.planes g)) Zq E} {key : (g : Fin count) → Key (B.shape g)}
  {openKey : B.OpenKey} {claim : Wire → B.Claim E} {μ : ℕ} {fits : B.cells < 2 ^ μ}
  {base digits : ℕ} {room₀ : B.quotientRoom digits ≤ 2 ^ μ} {batch : ℕ}
  {rowsFit : B.rowCount + 1 + 1 ≤ 2 ^ batch} {block : B.Block → Coord}
  {decode : (g : Fin count) → E → Option (Rq (B.carrierDegree g))} {mass : Fin count → ℕ}
  {β₁ : Module.Basis (Fin O₁.planes) Zq E} {key₁ : Key S₁} {openKey₁ : OpeningKey O₁}
  {claim₁ : Wire → Claim O₁ E} {μ₁ : ℕ} {fits₁ : O₁.cells fold₁ < 2 ^ μ₁} {digits₁ : ℕ}
  {room₁ : O₁.quotientRoom fold₁ digits₁ ≤ 2 ^ μ₁} {batch₁ : ℕ}
  {rowsFit₁ : O₁.rowCount + 1 + 1 ≤ 2 ^ batch₁} {block₁ : Fin S₁.blocks → Coord}
  {decode₁ : E → Option (Rq O₁.carrierDegree)} {weight₁ ν₁ : ℕ}
  {key₂ : Key S₂} {openKey₂ : OpeningKey O₂} {claim₂ : Wire → TraceClaim O₂ E}
  {μ₂ : ℕ} {fits₂ : O₂.cells fold₂ < 2 ^ μ₂} {batch₂ : ℕ}
  {rowsFit₂ : O₂.traceRowCount + 1 + 1 ≤ 2 ^ batch₂} {block₂ : Fin S₂.blocks → Coord}
  {decode₂ : E → Option (Rq S₂.innerDegree)} {weight₂ ν₂ m₂ : ℕ}
  {key₃ : Key S₃} {openKey₃ : OpeningKey O₃} {claim₃ : Wire → TraceClaim O₃ E}
  {μ₃ : ℕ} {fits₃ : O₃.cells fold₃ < 2 ^ μ₃} {batch₃ : ℕ}
  {rowsFit₃ : O₃.traceRowCount + 1 + 1 ≤ 2 ^ batch₃} {block₃ : Fin S₃.blocks → Coord}
  {decode₃ : E → Option (Rq S₃.innerDegree)} {weight₃ ν₃ m₃ : ℕ}
  {key₄ : Key S₄} {openKey₄ : OpeningKey O₄} {claim₄ : Wire → TraceClaim O₄ E}
  {μ₄ : ℕ} {fits₄ : O₄.cells fold₄ < 2 ^ μ₄} {batch₄ : ℕ}
  {rowsFit₄ : O₄.traceRowCount + 1 + 1 ≤ 2 ^ batch₄} {block₄ : Fin S₄.blocks → Coord}
  {decode₄ : E → Option (Rq S₄.innerDegree)} {weight₄ ν₄ m₄ : ℕ}
  {key₅ : Key S₅} {openKey₅ : OpeningKey O₅} {claim₅ : Wire → TraceClaim O₅ E}
  {μ₅ : ℕ} {fits₅ : O₅.cells fold₅ < 2 ^ μ₅} {batch₅ : ℕ}
  {rowsFit₅ : O₅.traceRowCount + 1 + 1 ≤ 2 ^ batch₅} {block₅ : Fin S₅.blocks → Coord}
  {decode₅ : E → Option (Rq S₅.innerDegree)} {weight₅ ν₅ m₅ : ℕ}
  {operator₃ limit₃ operator₄ limit₄ operator₅ limit₅ operatorL limitL : ℕ}
  {L : Terminal S₆ E Wire Coord E total (.euclidean operatorL limitL)} {m₆ : ℕ}
  {commit : Wire → (g : Fin count) → (B.shape g).Payload} {value : Wire → Fin count → E}
  {place : Wire → Fin count → List E} {commit₁ : Wire → S₁.Payload} {commit₂ : Wire → S₂.Payload}
  {commit₃ : Wire → S₃.Payload} {commit₄ : Wire → S₄.Payload} {commit₅ : Wire → S₅.Payload}
  {stateOf : Wire → Fin S₆.blocks → Fin S₆.innerRank → Rq S₆.innerDegree}
  {statementOf : Wire → TerminalStatement S₆ E}
  {partialsOf : Wire → Fin S₆.blocks → Rq S₆.innerDegree}

local notation "zeroth" =>
  liftBatch β key openKey claim μ fits base digits room₀ batch rowsFit block decode read mass
    (start + 1)

local notation "first" =>
  Tier.tensored (liftFold β₁ key₁ openKey₁ claim₁ μ₁ fits₁ digits₁ room₁ batch₁ rowsFit₁ block₁
    decode₁ read weight₁ (Tier.finish zeroth)) ν₁

local notation "second" =>
  Tier.tensored (committedTrace (fold := fold₂) key₂ openKey₂ claim₂ μ₂ fits₂ batch₂ rowsFit₂
    block₂ decode₂ read weight₂ (Tier.finish first)) ν₂

local notation "third" =>
  Tier.tensored (committedTraceSq (fold := fold₃) key₃ openKey₃ claim₃ μ₃ fits₃ batch₃ rowsFit₃
    block₃ decode₃ read weight₃ operator₃ limit₃ (Tier.finish second)) ν₃

local notation "fourth" =>
  Tier.tensored (committedTraceSq (fold := fold₄) key₄ openKey₄ claim₄ μ₄ fits₄ batch₄ rowsFit₄
    block₄ decode₄ read weight₄ operator₄ limit₄ (Tier.finish third)) ν₄

local notation "fifth" =>
  Tier.tensored (committedTraceSq (fold := fold₅) key₅ openKey₅ claim₅ μ₅ fits₅ batch₅ rowsFit₅
    block₅ decode₅ read weight₅ operator₅ limit₅ (Tier.finish fourth)) ν₅

/-- Akita Theorem 10.9 for the opening with a batching root, a fold with opening groups, a packing
level, four trace levels and the terminal, without the probability that a response does not fit and
without the term of (155) for a zero tensor factor, because this verifier has no rejection at a zero
factor: the verifier `towerLinks` of `mixedTower_suffix` accepts the honest prover. `early` is a
transcript prefix of `start` moves on the path of the table of the oracle. Its last message has the
commitment `commit` of the table `tables g` of each group, a point `place g` with the segments
`(r_pack, r_tail, r_pos, r_blk)` of the shape of the group, and the value of the table at this point
(`committed`, `placed`, `valued`). Then later messages exist such that the complete transcript
satisfies `towerLinks`: the link `RootLinked` of the batching root for the weights of the
transcript, the link `Tier.Linked` from the root fold to the level 1, and the links and the checks
of `packTower_complete`. `carriesR` and `carries₀`: a wire message can have the claim of the root
fold, and the commitment and the claim of the level 1. `shape₀`, `radix₀`, `certified₀`, `path₀`,
`uniform₀`, `digit₀`, `covers₀`, `range₀`: the conditions of `batchLevel_wins` on the root fold.
`blocks₀`, `lengths₀`, `strides₀`, `degrees₀`, `live₀`: the segments of the point of each group have
the lengths of its shape, and its table is 0 after the cells of its blocks. `stride₁`, `degree₁`,
`length₁`, `blocks₁`, `point₀`, `covered₀`: the final point of the checks of the root fold has the
segments of the shape of the level 1, and the live blocks of the level 1 have each cell before the
end of the quotient digits of the root fold (Akita Section 4.5, (40) to (42)). `decodes₀`: each slot
of the table of the oracle decodes to a fold challenge of the root fold. `fit`: the fit conditions
`BatchFit` of the root fold, `LiftFit` of the level 1, `TraceFit` of the levels 2 to 5, and `‖z‖_2²
≤ S_max,term` at the terminal. The other hypotheses are those of `packTower_complete`. -/
theorem tower_complete {sendable : (M : Type) → M → Prop} {slots width cap : ℕ}
    (sends : Sends sendable E width cap) (four : 4 ≤ width) (room : 3 < slots)
    (carriesR : Carries sendable slots read claim)
    (carries₀ : Carries sendable slots read fun wire => (commit₁ wire, claim₁ wire))
    (carries₁ : Carries sendable slots read fun wire => (commit₂ wire, claim₂ wire))
    (carries₂ : Carries sendable slots read fun wire => (commit₃ wire, claim₃ wire))
    (carries₃ : Carries sendable slots read fun wire => (commit₄ wire, claim₄ wire))
    (carries₄ : Carries sendable slots read fun wire => (commit₅ wire, claim₅ wire))
    (carries₅ : Carries sendable slots read fun wire =>
      (stateOf wire, statementOf wire, partialsOf wire))
    (carriesL : Carries sendable slots read L.response)
    {p₁ t₁ s₁ k₁ p₂ s₂ k₂ p₃ s₃ k₃ p₄ s₄ k₄ p₅ s₅ k₅ p₆ s₆ k₆ : ℕ}
    (shape₀ : ∀ g, (B.shape g).Honest) (radix₀ : ∀ g, Radix (B.fold g).responseBase)
    (certified₀ : ∀ g, (B.fold g).responseBase ≤ (B.shape g).certifiedBase)
    (path₀ : PathFits B.openDegree (1 * B.openRank) B.chain)
    (uniform₀ : ∀ g, (B.shape g).certifiedBase = base) (digit₀ : Radix base)
    (covers₀ : ∀ x : Zq, -((base / 2 : ℕ) : ℤ) * ∑ u ∈ Finset.range digits, (base : ℤ) ^ u ≤
        x.valMinAbs ∧
      x.valMinAbs ≤ (((base / 2 : ℕ) : ℤ) - 1) * ∑ u ∈ Finset.range digits, (base : ℤ) ^ u)
    (range₀ : rangeWidth base ≤ width)
    (shape₁ : S₁.Bounded) (radix₁ : Radix fold₁.responseBase)
    (certified₁ : fold₁.responseBase ≤ S₁.certifiedBase)
    (path₁ : PathFits O₁.openDegree (1 * O₁.openRank) O₁.chain)
    (digit₁ : Radix S₁.certifiedBase)
    (covers₁ : ∀ x : Zq, -((S₁.certifiedBase / 2 : ℕ) : ℤ) *
        ∑ u ∈ Finset.range digits₁, (S₁.certifiedBase : ℤ) ^ u ≤ x.valMinAbs ∧
      x.valMinAbs ≤ (((S₁.certifiedBase / 2 : ℕ) : ℤ) - 1) *
        ∑ u ∈ Finset.range digits₁, (S₁.certifiedBase : ℤ) ^ u)
    (range₁ : rangeWidth S₁.certifiedBase ≤ width)
    (quarter₂ : 4 * m₂ = S₂.innerDegree) (quarter₃ : 4 * m₃ = S₃.innerDegree)
    (quarter₄ : 4 * m₄ = S₄.innerDegree) (quarter₅ : 4 * m₅ = S₅.innerDegree)
    (quarter₆ : 4 * m₆ = S₆.innerDegree)
    (source₂ : SourceSized S₂ m₂ p₂ s₂ k₂) (source₃ : SourceSized S₃ m₃ p₃ s₃ k₃)
    (source₄ : SourceSized S₄ m₄ p₄ s₄ k₄) (source₅ : SourceSized S₅ m₅ p₅ s₅ k₅)
    (source₆ : SourceSized S₆ m₆ p₆ s₆ k₆)
    (sized₂ : FoldSized S₂ O₂ fold₂ width) (sized₃ : FoldSized S₃ O₃ fold₃ width)
    (sized₄ : FoldSized S₄ O₄ fold₄ width) (sized₅ : FoldSized S₅ O₅ fold₅ width)
    (stride₁ : O₁.stride = 2 ^ p₁) (degree₁ : O₁.carrierDegree = 2 ^ t₁)
    (length₁ : S₁.blockLength = 2 ^ s₁) (blocks₁ : S₁.blocks ≤ 2 ^ k₁)
    (point₀ : μ = p₁ + (t₁ + (s₁ + k₁)))
    (vars₁ : μ₁ = ν₁ + 1) (vars₂ : μ₂ = ν₂ + 1) (vars₃ : μ₃ = ν₃ + 1) (vars₄ : μ₄ = ν₄ + 1)
    (vars₅ : μ₅ = ν₅ + 1)
    (point₁ : ν₁ = p₂ + (s₂ + k₂)) (point₂ : ν₂ = p₃ + (s₃ + k₃))
    (point₃ : ν₃ = p₄ + (s₄ + k₄)) (point₄ : ν₄ = p₅ + (s₅ + k₅))
    (point₅ : ν₅ = p₆ + (s₆ + k₆))
    (covered₀ : B.quotientRoom digits ≤ S₁.blocks * (S₁.blockLength * S₁.innerDegree))
    (covered₁ : O₁.quotientRoom fold₁ digits₁ ≤ 2 * (S₂.blocks * (S₂.blockLength * (2 * m₂))))
    (covered₂ : O₂.cells fold₂ ≤ 2 * (S₃.blocks * (S₃.blockLength * (2 * m₃))))
    (covered₃ : O₃.cells fold₃ ≤ 2 * (S₄.blocks * (S₄.blockLength * (2 * m₄))))
    (covered₄ : O₄.cells fold₄ ≤ 2 * (S₅.blocks * (S₅.blockLength * (2 * m₅))))
    (covered₅ : O₅.cells fold₅ ≤ 2 * (S₆.blocks * (S₆.blockLength * (2 * m₆))))
    (normRoom₃ : S₃.blockLength * S₃.sourceDigits * S₃.innerDegree ≤ 2 ^ μ₃)
    (normRoom₄ : S₄.blockLength * S₄.sourceDigits * S₄.innerDegree ≤ 2 ^ μ₄)
    (normRoom₅ : S₅.blockLength * S₅.sourceDigits * S₅.innerDegree ≤ 2 ^ μ₅)
    (wide₃ : rangeWidthSq S₃.certifiedBase ≤ width) (wide₄ : rangeWidthSq S₄.certifiedBase ≤ width)
    (wide₅ : rangeWidthSq S₅.certifiedBase ≤ width)
    (capped₃ : limit₃ ≤ cap) (capped₄ : limit₄ ≤ cap) (capped₅ : limit₅ ≤ cap)
    (next : L.start = Tier.finish fifth) (inside : L.start < total)
    (reads : ∀ point, L.read point = statementOf (point.wire L.start))
    (clears : ∀ point, L.partials point = partialsOf (point.wire L.start))
    (table : Prefix Wire (Coord → E) total → Coord → E)
    (decodes₀ : ∀ input b, (decode b.1 (table input (block b))).isSome)
    (decodes₁ : ∀ input i, (decode₁ (table input (block₁ i))).isSome)
    (decodes₂ : ∀ input i, (decode₂ (table input (block₂ i))).isSome)
    (decodes₃ : ∀ input i, (decode₃ (table input (block₃ i))).isSome)
    (decodes₄ : ∀ input i, (decode₄ (table input (block₄ i))).isSome)
    (decodes₅ : ∀ input i, (decode₅ (table input (block₅ i))).isSome)
    (decodesL : ∀ input i, (L.decode (table input (L.block i))).isSome)
    {wire₀ : Wire} {early : List ((Coord → E) × Wire)} (on : OnPath accept table wire₀ early)
    (starts : early.length = start) {tables : Fin count → Table}
    {blockOf positionOf packOf tailOf : Fin count → List E}
    (blocks₀ : ∀ g, (B.shape g).blocks ≤ 2 ^ (blockOf g).length)
    (lengths₀ : ∀ g, (B.shape g).blockLength = 2 ^ (positionOf g).length)
    (strides₀ : ∀ g, (B.opening g).stride = 2 ^ (packOf g).length)
    (degrees₀ : ∀ g, B.carrierDegree g = 2 ^ (tailOf g).length)
    (live₀ : ∀ g cell, (B.shape g).blockLength * (B.shape g).innerDegree * (B.shape g).blocks ≤
      cell → tables g cell = 0)
    (committed : commit (lastWire wire₀ early) = fun g => (Akita.commit (key g) (tables g)).1)
    (placed : place (lastWire wire₀ early) =
      fun g => packOf g ++ (tailOf g ++ (positionOf g ++ blockOf g)))
    (valued : value (lastWire wire₀ early) =
      fun g => mle (place (lastWire wire₀ early) g) (lifted E (tables g)))
    (fit : BatchFit β key openKey μ fits base digits decode tables fun w₀ =>
      LiftFit β₁ key₁ openKey₁ fold₁ μ₁ fits₁ digits₁ decode₁ (fun i => S₁.embed (S₁.pack w₀ i))
        fun w₁ =>
      TraceFit quarter₂ key₂ openKey₂ fold₂ (fun _ => decode₂) (fun _ => True) (packed w₁)
        fun w₂ =>
      TraceFit quarter₃ key₃ openKey₃ fold₃ (fun _ => decode₃) (fun z => vecNormSq z ≤ limit₃)
        (packed w₂) fun w₃ =>
      TraceFit quarter₄ key₄ openKey₄ fold₄ (fun _ => decode₄) (fun z => vecNormSq z ≤ limit₄)
        (packed w₃) fun w₄ =>
      TraceFit quarter₅ key₅ openKey₅ fold₅ (fun _ => decode₅) (fun z => vecNormSq z ≤ limit₅)
        (packed w₄) fun w₅ =>
      ∀ c : Fin S₆.blocks → Rq S₆.innerDegree, (∀ i, ∃ slot, L.decode slot = some (c i)) →
        vecNormSq (traceResponse c (traceSource quarter₆ (packed w₅ : ℕ → E))) ≤ limitL) :
    ∃ later, OnPath accept table wire₀ (early ++ later) ∧ (early ++ later).length = total ∧
      ∀ stage : Prefix Wire (Coord → E) total, stage.first = wire₀ →
        stage.rest = early ++ later →
        towerLinks (weight₃ := weight₃) (weight₄ := weight₄) (weight₅ := weight₅) .coefficient
          (.euclidean operator₃ limit₃) (.euclidean operator₄ limit₄)
          (.euclidean operator₅ limit₅) zeroth first second third fourth fifth L start read commit
          value place quarter₂ quarter₃ quarter₄ quarter₅ quarter₆ commit₁ commit₂ commit₃ commit₄
          commit₅ stateOf stage := by
  have ends₀ := Tier.finish_eq zeroth
  have ends₁ := Tier.finish_eq first
  have ends₂ := Tier.finish_eq second
  have ends₃ := Tier.finish_eq third
  have ends₄ := Tier.finish_eq fourth
  have ends₅ := Tier.finish_eq fifth
  have fin₀ : Tier.finish zeroth ≤ total :=
    (show Tier.finish zeroth ≤ Tier.finish first from
      (Nat.le_add_right _ _).trans_eq ends₁.symm).trans
    ((show Tier.finish first ≤ Tier.finish second from
      (Nat.le_add_right _ _).trans_eq ends₂.symm).trans
    ((show Tier.finish second ≤ Tier.finish third from
      (Nat.le_add_right _ _).trans_eq ends₃.symm).trans
    ((show Tier.finish third ≤ Tier.finish fourth from
      (Nat.le_add_right _ _).trans_eq ends₄.symm).trans
    ((show Tier.finish fourth ≤ Tier.finish fifth from
      (Nat.le_add_right _ _).trans_eq ends₅.symm).trans (next ▸ inside.le)))))
  have rooted₀ : start + 1 ≤ total :=
    (show start + 1 ≤ Tier.finish zeroth from (Nat.le_add_right _ _).trans_eq ends₀.symm).trans
      fin₀
  obtain ⟨reply, answered⟩ : ∃ reply, reply = answer table wire₀ early := ⟨_, rfl⟩
  obtain ⟨weights, weighed⟩ : ∃ weights : Fin count → E,
      weights = sample (Coord → E) (Fin count → E) reply := ⟨_, rfl⟩
  obtain ⟨points, pointed⟩ : ∃ points : (g : Fin count) → Point (B.opening g) E,
      points = fun g => Point.ofSegments (B.opening g) (blockOf g) (positionOf g) (packOf g)
        (tailOf g) := ⟨_, rfl⟩
  obtain ⟨wireR, claimed, _⟩ := carriesR (Batch.honestClaim β key openKey tables points weights)
    [] (Nat.zero_le _) (by simp)
  have onR := on.snoc (by rw [starts]; exact rooted₀) wireR
  rw [← answered] at onR
  have lenR : (early ++ [(reply, wireR)]).length = start + 1 := by
    rw [List.length_append, List.length_singleton, starts]
  have member₀ : ∀ b, ∃ slot, decode b.1 slot =
      some (Tier.drawn zeroth table wire₀ (early ++ [(reply, wireR)]) b) :=
    fun b => ⟨_, Tier.drawn_mem zeroth table (lenR.le.trans rooted₀) decodes₀ b⟩
  obtain ⟨fit₁, zero₀, wins₀⟩ := batchLevel_wins (claim := claim) (room := room₀)
    (batch := batch) (rowsFit := rowsFit) (block := block) (read := read) (mass := mass)
    (start := start + 1) sends range₀ four (Nat.zero_lt_of_lt room) shape₀ radix₀ certified₀
    path₀ uniform₀ digit₀ covers₀
    (wire := lastWire wire₀ (early ++ [(reply, wireR)]))
    (by rw [lastWire_concat]; exact claimed) fit member₀
  obtain ⟨later₀, place₁, on₀, len₀, ready₁, linked₀⟩ :=
    packLink (T := zeroth) (Next := first) (commit' := commit₁) carries₀ onR lenR fin₀ decodes₀
      (fun _ _ table other => foldCheck_bare _ _ _ table other) (foldCosts_length E _ _ _ _) _
      (wins₀.mono fun _ _ _ kept => kept.2) rfl shape₁ β₁ stride₁ degree₁ length₁ blocks₁ point₀
      fun cell large => zero₀ cell (covered₀.trans large)
  obtain ⟨later, onL, whole, each⟩ := packTower_complete (start := Tier.finish zeroth) sends four
    room carries₁ carries₂ carries₃ carries₄ carries₅ carriesL shape₁ radix₁ certified₁ path₁
    digit₁ covers₁ range₁ quarter₂ quarter₃ quarter₄ quarter₅ quarter₆ source₂ source₃ source₄
    source₅ source₆ sized₂ sized₃ sized₄ sized₅ vars₁ vars₂ vars₃ vars₄ vars₅ point₁ point₂
    point₃ point₄ point₅ covered₁ covered₂ covered₃ covered₄ covered₅ normRoom₃ normRoom₄
    normRoom₅ wide₃ wide₄ wide₅ capped₃ capped₄ capped₅ next inside reads clears table decodes₁ decodes₂ decodes₃
    decodes₄ decodes₅ decodesL on₀ len₀ ready₁ fit₁
  have joined : early ++ (reply, wireR) :: (later₀ ++ later) =
      early ++ [(reply, wireR)] ++ later₀ ++ later := by
    simp only [List.append_assoc, List.cons_append, List.nil_append]
  refine ⟨(reply, wireR) :: (later₀ ++ later), joined ▸ onL, joined ▸ whole,
    fun stage rooted shape => ?_⟩
  have before : stage.wire start = lastWire wire₀ early := by
    rw [← starts, ← rooted]
    exact Prefix.wire_length shape
  have after : stage.wire (start + 1) = wireR :=
    stage.wire_of shape (congrArg (· + 1) starts.symm)
  have drawn : (claimsRoot (Coord := Coord) (Slot := E) (total := total) start read value
      place).weights stage = weights := by
    show ((stage.reply start).map (sample (Coord → E) (Fin count → E))).getD 0 = weights
    rw [stage.reply_of shape starts.symm, weighed]
    rfl
  refine ⟨by rw [shape, List.length_append, List.length_cons, starts]; omega, ⟨?_, ?_, ?_,
    fun g => ⟨blockOf g, positionOf g, packOf g, tailOf g, blocks₀ g, lengths₀ g, strides₀ g,
      degrees₀ g, ?_, ?_⟩⟩, linked₀ stage rooted ⟨later, shape.trans joined⟩,
    each stage rooted (shape.trans joined)⟩
  · show (claim (stage.wire (start + 1))).payload = commit (stage.wire start)
    rw [after, claimed, before, committed]
    rfl
  · show (claim (stage.wire (start + 1))).weight = _
    rw [after, claimed, drawn]
    rfl
  · show (claim (stage.wire (start + 1))).value = _
    rw [after, claimed, drawn, before, valued, placed, pointed]
    show ∑ g, weights g * _ = ∑ g, weights g * _
    exact Finset.sum_congr rfl fun g _ => congrArg (weights g * ·)
      (honestClaim_value (O := B.opening g) (β := β g) (key := key g)
        (openKey := openKey.group g) (shape₀ g) (tables g) (blockOf g) (positionOf g) (packOf g)
        (tailOf g) (blocks₀ g) (lengths₀ g) (strides₀ g) (degrees₀ g) (live₀ g))
  · show (claim (stage.wire (start + 1))).point g = _
    rw [after, claimed, pointed]
    rfl
  · rw [before, placed]

end Tower

end Extract

end NightstreamFPrime.Spec.Compression.Akita
