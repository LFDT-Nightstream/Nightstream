import NightstreamFPrime.Spec.Compression.Akita.TraceProver
import NightstreamFPrime.Spec.Compression.Akita.Extract.CheckComplete

/-! The honest prover of a level of Akita with the tensor reduction on a Fiat–Shamir transcript,
and the links of the verifier to a next trace level (Akita, ePrint 2026/1983, revision of
2026-10-15, Theorem 10.9 with Lemma 10.7, Figure 4, steps 0, 1, 3 and 5).

- `honestTraceClaim` is the claim of the honest prover of a trace level for a packed polynomial
  `f̄` and a point `(ρ_pack, ρ_pos, ρ_blk)` with the factor `θ`, and `honestTraceClaim_satisfies`
  is `honestTrace_satisfies` for it.
- `traceNext` is the flat witness of the honest prover of a trace level, and `TraceFit` states
  that for each fold challenge vector of the family the response of the level is admissible and
  a condition `after` holds for this flat witness. `traceLevel_wins` gives the honest prover of
  the checks of a trace level from the claim `honestTraceClaim` and `TraceFit`.
- `tensorStep` gives the transcript prefix of the honest prover through the stages of a level
  with the tensor reduction: the first message of the level and its last message have the data
  that the prover selects, and the checks end in a claim `(ρ, θ, θ·f̄_w(ρ))`.
- `traceLink` is `tensorStep` for a next trace level: the first message has the commitment to
  `ψ(f̄_w)`, the last message has the claim `honestTraceClaim` of the next level, and each
  transcript with this prefix has the link `Tier.TraceLinked`.
- `terminalLink` is `tensorStep` for the terminal as the next level: the messages have the state
  `t_i = A_term·s_i`, the statement and the clear partials of the terminal, and each transcript
  with the prefix has the link `Terminal.Linked`.
- `OnPath.fill` and `OnPath.accepts` give the complete transcript of a proof from a stage input
  on the path.
- `terminal_passed` gives the last messages of the honest prover: the clear response
  `z = Σ_i c_i·s_i` after the fold challenges of the terminal, with the checks `Terminal.Passed`
  on the complete transcript. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Extract

open scoped BigOperators
open NightstreamFPrime.Spec.Compression.Lib NightstreamFPrime.Spec.Compression.Game
open NightstreamFPrime.Spec.Compression (Game)
open NightstreamFPrime.Spec.Checkpoint

local instance : Fact (Nat.Prime fieldModulus) := ⟨FieldPrime.fieldModulus_natPrime⟩

attribute [local instance low] Classical.propDecidable

section Sized

/-- The conditions on the commitment shape of a trace level or of the terminal for the honest
prover, with the lengths `p`, `s` and `k` of the segments of its point. `shape`: the conditions
`Shape.Bounded`. `unit`: `d_A/k = 2·m` is a unit of `Z_q`. `pack`, `length`, `blocks`:
`d_A/k = 2^p`, `M = 2^s` and `B ≤ 2^k`. -/
structure SourceSized (S : Shape) (m p s k : ℕ) : Prop where
  shape : S.Bounded
  unit : IsUnit ((2 * m : ℕ) : Zq)
  pack : 2 * m = 2 ^ p
  length : S.blockLength = 2 ^ s
  blocks : S.blocks ≤ 2 ^ k

/-- The conditions on the fold of a trace level for the honest prover. `radix`: `b_z` is even and
`2 ≤ b_z < q`. `certified`: `b_z ≤ b⋆`. `path`: the inputs of `H_1` and `H_2` have a position for
each digit. `wide`: the view of `D` has the `d_A` coordinates of a partial. `two` and `small`:
`2 ≤ b⋆ < q`. `range`: a message can have each message of the range tree. -/
structure FoldSized (S : Shape) (O : OpeningShape S) (fold : Descriptor) (width : ℕ) : Prop where
  radix : Radix fold.responseBase
  certified : fold.responseBase ≤ S.certifiedBase
  path : PathFits O.openDegree (1 * O.openRank) O.chain
  wide : S.innerDegree ≤ O.subcolumns * O.openDegree
  two : 2 ≤ S.certifiedBase
  small : S.certifiedBase < fieldModulus
  range : rangeWidth S.certifiedBase ≤ width

theorem FoldSized.mono {S : Shape} {O : OpeningShape S} {fold : Descriptor} {width wider : ℕ}
    (sized : FoldSized S O fold width) (le : width ≤ wider) : FoldSized S O fold wider :=
  { sized with range := sized.range.trans le }

end Sized

section Claim

variable {S : Shape} {O : OpeningShape S} {fold : Descriptor} {E : Type} [Field E]
  [Algebra Zq E] [Degree2 E] {m : ℕ}

/-- The claim of the honest prover of a trace level for a packed polynomial `f̄`, the point
`ρ = (ρ_pack, ρ_pos, ρ_blk)` and the factor `θ` (Akita Figure 4, steps 0 and 1): the payload
`u_pub` of the commitment to the ring elements `ψ(f̄)`, the payload `v_pub` of the opening path
for the digits `ê`, and the value `θ·f̄(ρ)`. -/
noncomputable def honestTraceClaim (quarter : 4 * m = S.innerDegree) (key : Key S)
    (openKey : OpeningKey O) (g : ℕ → E) (block position pack : List E) (θ : E) :
    TraceClaim O E :=
  TraceClaim.ofPoint quarter (commitSource key (traceSource quarter g)).1
    (honestOpening openKey (tracePartials O (subfield quarter).toRingHom
      (fun x : Fin S.blockLength => eqAt position x) (traceSource quarter g))).2
    block position pack θ (θ * mle (pack ++ (position ++ block)) g)

/-- `honestTrace_satisfies` for the claim `honestTraceClaim` and the state of `commitSource`
(Akita Lemma 10.7 for a trace level). `unit`: `d_A/k` is a unit of `Z_q`. `blocks`, `length`,
`width`, `live`: the point has segments for `B ≤ 2^{r_blk}`, `M` and `d_A/k`, and `f̄` is 0 after
the values of the live blocks. The other hypotheses are those of `honestTrace_satisfies`. -/
theorem honestTraceClaim_satisfies (quarter : 4 * m = S.innerDegree) (shape : S.Bounded)
    (unit : IsUnit ((2 * m : ℕ) : Zq)) (radix : Radix fold.responseBase)
    (certified : fold.responseBase ≤ S.certifiedBase)
    (path : PathFits O.openDegree (1 * O.openRank) O.chain)
    (wide : S.innerDegree ≤ O.subcolumns * O.openDegree) {g : ℕ → E}
    {block position pack : List E} (blocks : S.blocks ≤ 2 ^ block.length)
    (length : S.blockLength = 2 ^ position.length) (width : 2 * m = 2 ^ pack.length)
    (live : ∀ cell, S.blocks * (S.blockLength * (2 * m)) ≤ cell → g cell = 0) (θ : E)
    (key : Key S) (openKey : OpeningKey O) (challenge : Fin S.blocks → Rq S.innerDegree)
    (encodable : Encodable fold.responseBase fold.responseDigits
      (traceResponse challenge (traceSource quarter g))) :
    (honestTrace openKey (subfield quarter).toRingHom
      (fun x : Fin S.blockLength => eqAt position x) fold challenge
      (commitSource key (traceSource quarter g)).2).TraceSatisfies
      ((honestTraceClaim quarter key openKey g block position pack θ).statement key openKey)
      challenge :=
  honestTrace_satisfies
    (st := (honestTraceClaim quarter key openKey g block position pack θ).statement key openKey)
    challenge shape radix certified path wide (commitSource_opens shape key _) encodable rfl
    (congrArg (θ * ·)
      (traceScalar_source quarter shape unit block position pack blocks length width live).symm)

/-- The flat witness of the honest prover of a trace level for the packed polynomial `f̄`, the
weights of `ρ_pos` and the fold challenges: the table that the next level commits to (Akita (76)
and Figure 4, step 3). -/
noncomputable def traceNext (quarter : 4 * m = S.innerDegree) (key : Key S)
    (openKey : OpeningKey O) (fold : Descriptor) (g : ℕ → E) (position : List E)
    (challenge : Fin S.blocks → Rq S.innerDegree) : Table :=
  (honestTrace openKey (subfield quarter).toRingHom
    (fun x : Fin S.blockLength => eqAt position x) fold challenge
    (commitSource key (traceSource quarter g)).2).flat

/-- The fit conditions of the honest prover of a trace level for the packed polynomial `f̄`
(Akita Definition 10.5 and Lemma 10.7): for each fold challenge vector of the family, the
response `z = Σ_i c_i·s_i` satisfies `Enc` and the condition `admissible` of the norm route, and
`after` holds for the flat witness of the level, for each position segment `ρ_pos`. Akita
Theorem 10.9 has the probability that a response does not fit in the term `ξ_h` of (155). -/
def TraceFit (quarter : 4 * m = S.innerDegree) (key : Key S) (openKey : OpeningKey O)
    (fold : Descriptor) (decode : Fin S.blocks → E → Option (Rq S.innerDegree))
    (admissible : S.Source → Prop) (g : ℕ → E) (after : Table → Prop) : Prop :=
  ∀ (position : List E) (challenge : Fin S.blocks → Rq S.innerDegree),
    (∀ i, ∃ slot, decode i slot = some (challenge i)) →
    Encodable fold.responseBase fold.responseDigits
        (traceResponse challenge (traceSource quarter g)) ∧
      admissible (traceResponse challenge (traceSource quarter g)) ∧
      after (traceNext quarter key openKey fold g position challenge)

/-- The packed polynomial of a table is 0 at each value after the last cell of the table that is
not 0. -/
theorem packed_of_le {w : Table} {bound : ℕ} (zero : ∀ x, bound ≤ x → w x = 0) {cell : ℕ}
    (large : bound ≤ 2 * cell) : (packed w cell : E) = 0 := by
  unfold packed
  rw [zero _ large, zero _ (by omega)]
  exact Degree2.ext (by rw [Degree2.re_pair, Degree2.re_zero])
    (by rw [Degree2.im_pair, Degree2.im_zero])

/-- The honest prover of the checks of a trace level (Akita Lemma 10.7 with Theorem 10.9). The
message `wire` has the claim `honestTraceClaim`, and `fit` has the fit conditions. `level`: the
checks of the level have an honest prover for digit vectors with the exact branch relation and a
response with the condition of the norm route. Then for each fold challenge vector of the family,
`after` holds for the flat witness `traceNext`, this table is 0 from the cell `cells`, and the
prover has a message for each draw sequence of the checks for it. -/
theorem traceLevel_wins {Brk Wire Coord : Type}
    {B : Tier (Fin S.blocks) (fun _ => Rq S.innerDegree) (TraceClaim O E) Brk (Witness O fold)
      (Weak O) Wire Coord E (Option (List E × E)) Table}
    {key : Key S} {openKey : OpeningKey O} {admissible : S.Source → Prop} {draws ν slots : ℕ}
    {sendable : (M : Type) → M → Prop}
    (level : ∀ (wire : Wire) (c : Fin S.blocks → Rq S.innerDegree) (w : Witness O fold),
      w.TraceSatisfies ((B.claim wire).statement key openKey) c →
      admissible (gadget fold.responseBase w.response) →
      WinsAt sendable slots (B.check wire c w.flat) draws 0 fun result _ place =>
        place ≤ 1 ∧ ∃ r2 : List E, r2.length = ν + 1 ∧
          result = some (r2, mle r2 (lifted E w.flat)))
    (quarter : 4 * m = S.innerDegree) (shape : S.Bounded) (unit : IsUnit ((2 * m : ℕ) : Zq))
    (radix : Radix fold.responseBase) (certified : fold.responseBase ≤ S.certifiedBase)
    (path : PathFits O.openDegree (1 * O.openRank) O.chain)
    (wide : S.innerDegree ≤ O.subcolumns * O.openDegree) {g : ℕ → E}
    {block position pack : List E} (blocks : S.blocks ≤ 2 ^ block.length)
    (length : S.blockLength = 2 ^ position.length) (width : 2 * m = 2 ^ pack.length)
    (live : ∀ cell, S.blocks * (S.blockLength * (2 * m)) ≤ cell → g cell = 0) {θ : E}
    {wire : Wire}
    (ready : B.claim wire = honestTraceClaim quarter key openKey g block position pack θ)
    {after : Table → Prop} (fit : TraceFit quarter key openKey fold B.decode admissible g after)
    {c : Fin S.blocks → Rq S.innerDegree} (member : ∀ i, ∃ slot, B.decode i slot = some (c i)) :
    after (traceNext quarter key openKey fold g position c) ∧
      (∀ x, O.cells fold ≤ x → traceNext quarter key openKey fold g position c x = 0) ∧
      WinsAt sendable slots (B.check wire c (traceNext quarter key openKey fold g position c))
        draws 0 fun result _ place => place ≤ 1 ∧ ∃ r2 : List E, r2.length = ν + 1 ∧
          result = some (r2, mle r2
            (lifted E (traceNext quarter key openKey fold g position c))) := by
  obtain ⟨encodable, admitted, later⟩ := fit position c member
  have satisfies := honestTraceClaim_satisfies (fold := fold) quarter shape unit radix certified
    path wide blocks length width live θ key openKey c encodable
  rw [← ready] at satisfies
  refine ⟨later, fun x large => Witness.flat_of_le _ large, level wire c _ satisfies ?_⟩
  show admissible (gadget fold.responseBase
    (decompose fold.responseBase (traceResponse c (traceSource quarter g))))
  rw [gadget_decompose_of_encodable radix encodable]
  exact admitted

end Claim

section Step

variable {E Data : Type}

/-- The data of the wire messages of the honest prover of a level with the tensor reduction:
`first` for the message after the fold challenges, and `last ρ θ` for the message after an end
`(ρ, θ, ṽ)` of the checks. -/
noncomputable def stepData (first : Data) (last : List E → E → Data) (index : ℕ) :
    Game (Option (List E × E × E)) → Data
  | .pure (some (ρ, θ, _)) => if index = 0 then first else last ρ θ
  | _ => first

theorem stepData_zero (first : Data) (last : List E → E → Data)
    (game : Game (Option (List E × E × E))) : stepData first last 0 game = first := by
  unfold stepData
  split
  · exact ite_eq_left rfl
  · rfl

end Step

section Tensor

variable {S : Shape} {O : OpeningShape S} {fold : Descriptor} {E : Type} [Field E] [Fintype E]
  [DecidableEq E] [Algebra Zq E] [Degree2 E] {Ring Stmt Brk Wire Coord Data : Type} [Zero Ring]
  [Fintype Coord] [DecidableEq Coord] {total : ℕ}
  {accept : Prefix Wire (Option Coord → E) total → Prop}
  {table : Prefix Wire (Option Coord → E) total → Option Coord → E}
  {B : Tier (Fin S.blocks) (fun _ => Ring) Stmt Brk (Witness O fold) (Weak O) Wire Coord E
    (Option (List E × E)) Table}
  {ν : ℕ} {sendable : (M : Type) → M → Prop} {slots width cap : ℕ}

/-- The transcript prefix of the honest prover through the stages of a level with the tensor
reduction (Akita Theorem 10.9 for the checks of one level and Theorem 5.1, item 1). `wins`: the
prover has a message for each draw sequence of the checks of the level for the table `source`,
which end in a true claim `(r_2, w̃(r_2))` of `ν + 1` coordinates. `uniform`: the verifier of the
checks does not read the table. Then a message `wire` with the data `dFirst` and moves exist
such that the prefix is on the path of the table of the oracle, the last message has the data
`dLast ρ θ`, and on each transcript with this prefix the checks of the verifier end in the claim
`(ρ, θ, θ·f̄_w(ρ))` for the packed polynomial of `source`. -/
theorem tensorStep {dataOf : Wire → Data} (carries : Carries sendable slots B.read dataOf)
    (sends : Sends sendable E width cap) (wide : 3 ≤ width) {first : Wire}
    {early : List ((Option Coord → E) × Wire)} (on : OnPath accept table first early)
    (starts : early.length = B.start) (fits : (B.tensored ν).finish ≤ total)
    (decodes : ∀ input b, (B.decode b (table input (B.block b))).isSome)
    (worked : ∀ input, table input none ∈ workZeros E 57) (capped : B.Capped)
    (uniform : ∀ wire challenge table other,
      bare (B.check wire challenge table) = bare (B.check wire challenge other))
    {draws bound : ℕ} (count : B.costs.length = draws) (room : bound < slots) (source : Table)
    (wins : WinsAt sendable slots
      (B.check (lastWire first early) (B.drawn table first early) source) draws 0
      fun result _ place => place ≤ bound ∧ ∃ r2 : List E, r2.length = ν + 1 ∧
        result = some (r2, mle r2 (lifted E source)))
    (dFirst : Data) (dLast : List E → E → Data) :
    ∃ (wire : Wire) (moves : List ((Option Coord → E) × Wire)) (ρ : List E) (θ : E), ρ.length = ν ∧
      OnPath accept table first (early ++ (answer table first early, wire) :: moves) ∧
      (early ++ (answer table first early, wire) :: moves).length = (B.tensored ν).finish ∧
      dataOf wire = dFirst ∧ dataOf (lastWire wire moves) = dLast ρ θ ∧
      ∀ stage : Prefix Wire (Option Coord → E) total, stage.first = first →
        (∃ later, stage.rest = (early ++ (answer table first early, wire) :: moves) ++ later) →
        (B.tensored ν).finish ≤ stage.rest.length ∧ (B.tensored ν).Decodes stage ∧
        stage.wire (B.start + 1) = wire ∧
        stage.wire (B.tensored ν).finish = lastWire wire moves ∧
        (B.tensored ν).game (fun _ => 0) stage =
          .pure (some (ρ, θ, θ * mle ρ (packed source))) := by
  have won := Tier.tensored_winsAt (ν := ν) sends wide wins room count
  obtain ⟨wire, moves, result, counted, ⟨ρ, θ, long, rfl⟩, on', held, last, ended, facts⟩ :=
    Tier.run (T := B.tensored ν) carries on starts fits decodes worked
      ⟨capped.1, fun g member => (List.mem_append.mp member).elim (capped.2 g)
        (map_workBits_le _ g)⟩ source won (stepData dFirst dLast)
  have positive : moves.length ≠ 0 := by
    rw [counted]
    show (B.costs ++ tensorCosts E ν).length ≠ 0
    rw [List.length_append, tensorCosts, List.length_cons]
    omega
  have ends := (B.tensored ν).finish_eq
  refine ⟨wire, moves, ρ, θ, long, on', ?_, held.trans (stepData_zero _ _ _),
    last.trans (ite_eq_right positive), fun stage rooted extended => ?_⟩
  · rw [List.length_append, List.length_cons, counted, starts, ends]
    rfl
  · obtain ⟨long', decoded, _, opened, closed, game⟩ := facts stage rooted extended
    exact ⟨long', decoded, opened, closed, (game _).trans
      (ended_of_bare _ (Tier.tensored_uniform uniform _ _ source fun _ => 0) ended)⟩

variable {S' : Shape} {O' : OpeningShape S'} {fold' : Descriptor} {Brk' Result' Source' : Type}
  {Next : Tier (Fin S'.blocks) (fun _ => Rq S'.innerDegree) (TraceClaim O' E) Brk'
    (Witness O' fold') (Weak O') Wire Coord E Result' Source'}
  {m' : ℕ} {commit' : Wire → S'.Payload}

/-- `tensorStep` for a next level `Next` with evaluation trace (Akita Figure 4, steps 0, 1, 3
and 5). The honest prover commits to the ring elements `ψ(f̄_w)` of the packed polynomial of the
table `source` in the message after the fold challenges, and its last message has the claim
`honestTraceClaim` of `Next` for the point `ρ` and the factor `θ` of the tensor reduction. `p`
and `s` are the lengths of the segments `ρ_pack` and `ρ_pos` of `ρ` for the shape of `Next`, and
`ρ_blk` has `k` coordinates or more. Each transcript with the prefix has the link
`Tier.TraceLinked` to `Next`. -/
theorem traceLink
    (carries : Carries sendable slots B.read fun wire => (commit' wire, Next.claim wire))
    (sends : Sends sendable E width cap) (wide : 3 ≤ width) {first : Wire}
    {early : List ((Option Coord → E) × Wire)} (on : OnPath accept table first early)
    (starts : early.length = B.start) (fits : (B.tensored ν).finish ≤ total)
    (decodes : ∀ input b, (B.decode b (table input (B.block b))).isSome)
    (worked : ∀ input, table input none ∈ workZeros E 57) (capped : B.Capped)
    (uniform : ∀ wire challenge table other,
      bare (B.check wire challenge table) = bare (B.check wire challenge other))
    {draws bound : ℕ} (count : B.costs.length = draws) (room : bound < slots) (source : Table)
    (wins : WinsAt sendable slots
      (B.check (lastWire first early) (B.drawn table first early) source) draws 0
      fun result _ place => place ≤ bound ∧ ∃ r2 : List E, r2.length = ν + 1 ∧
        result = some (r2, mle r2 (lifted E source)))
    (next : Next.start = (B.tensored ν).finish) (quarter' : 4 * m' = S'.innerDegree)
    (key' : Key S') (openKey' : OpeningKey O') {p s k : ℕ} (width' : 2 * m' = 2 ^ p)
    (length' : S'.blockLength = 2 ^ s) (blocks' : S'.blocks ≤ 2 ^ k) (vars : ν = p + (s + k)) :
    ∃ (later : List ((Option Coord → E) × Wire)) (block position pack : List E) (θ : E),
      pack.length = p ∧ position.length = s ∧ k ≤ block.length ∧
      OnPath accept table first (early ++ later) ∧ (early ++ later).length = Next.start ∧
      Next.claim (lastWire first (early ++ later)) =
        honestTraceClaim quarter' key' openKey' (packed source) block position pack θ ∧
      ∀ stage : Prefix Wire (Option Coord → E) total, stage.first = first →
        (∃ rest, stage.rest = (early ++ later) ++ rest) →
        (B.tensored ν).TraceLinked Next quarter' commit' stage := by
  obtain ⟨wire, moves, ρ, θ, long, on', whole, held, last, facts⟩ :=
    tensorStep carries sends wide on starts fits decodes worked capped uniform count room source
      wins
      ((commitSource key' (traceSource quarter' (packed source : ℕ → E))).1, Next.claim first)
      fun ρ θ => ((commitSource key' (traceSource quarter' (packed source : ℕ → E))).1,
        honestTraceClaim quarter' key' openKey' (packed source) ((ρ.drop p).drop s)
          ((ρ.drop p).take s) (ρ.take p) θ)
  have claimed : Next.claim (lastWire wire moves) =
      honestTraceClaim quarter' key' openKey' (packed source) ((ρ.drop p).drop s)
        ((ρ.drop p).take s) (ρ.take p) θ := congrArg Prod.snd last
  have paid : commit' wire = (commitSource key' (traceSource quarter' (packed source : ℕ → E))).1 :=
    congrArg Prod.fst held
  have joined : ρ.take p ++ ((ρ.drop p).take s ++ (ρ.drop p).drop s) = ρ := by
    rw [List.take_append_drop, List.take_append_drop]
  refine ⟨(answer table first early, wire) :: moves, (ρ.drop p).drop s, (ρ.drop p).take s,
    ρ.take p, θ, by rw [List.length_take]; omega,
    by rw [List.length_take, List.length_drop]; omega,
    by rw [List.length_drop, List.length_drop]; omega, on', whole.trans next.symm,
    by rw [lastWire_append_cons]; exact claimed, fun stage rooted extended => ?_⟩
  obtain ⟨long', decoded, opened, closed, game⟩ := facts stage rooted extended
  refine ⟨long', decoded, (ρ.drop p).drop s, (ρ.drop p).take s, ρ.take p, θ,
    θ * mle ρ (packed source),
    (honestOpening openKey' (tracePartials O' (subfield (E := E) quarter').toRingHom
      (fun x : Fin S'.blockLength => eqAt ((ρ.drop p).take s) x)
      (traceSource quarter' (packed source : ℕ → E)))).2, ?_,
    blocks'.trans (le_of_eq (by congr 1; simp; omega)), ?_, ?_, ?_⟩
  · rw [joined]
    exact game
  · rw [length']
    congr 1
    simp
    omega
  · rw [width']
    congr 1
    simp
    omega
  · have opened' : stage.wire ((B.tensored ν).start + 1) = wire := opened
    rw [next, closed, claimed, opened', paid, honestTraceClaim, joined]

variable {route : Route} {L : Terminal S' E Wire Coord E total route}
  {stateOf : Wire → Fin S'.blocks → Fin S'.innerRank → Rq S'.innerDegree}
  {statementOf : Wire → TerminalStatement S' E}
  {partialsOf : Wire → Fin S'.blocks → Rq S'.innerDegree}

/-- `tensorStep` for the terminal as the next level (Akita Section 8.2). The honest prover sends
the state `t_i = A_term·s_i` of the bounded source blocks of `ψ(f̄_w)` in the message after the
fold challenges, and its last message has the statement of the terminal for the point `ρ` and
the factor `θ` of the tensor reduction, and the clear partials `e_i`. `reads`: the statement of
the terminal is the statement `statementOf` of the message before its fold challenges. Each
transcript with the prefix has the link `Terminal.Linked`. -/
theorem terminalLink
    (carries : Carries sendable slots B.read fun wire =>
      (stateOf wire, statementOf wire, partialsOf wire))
    (sends : Sends sendable E width cap) (wide : 3 ≤ width) {first : Wire}
    {early : List ((Option Coord → E) × Wire)} (on : OnPath accept table first early)
    (starts : early.length = B.start) (fits : (B.tensored ν).finish ≤ total)
    (decodes : ∀ input b, (B.decode b (table input (B.block b))).isSome)
    (worked : ∀ input, table input none ∈ workZeros E 57) (capped : B.Capped)
    (uniform : ∀ wire challenge table other,
      bare (B.check wire challenge table) = bare (B.check wire challenge other))
    {draws bound : ℕ} (count : B.costs.length = draws) (room : bound < slots) (source : Table)
    (wins : WinsAt sendable slots
      (B.check (lastWire first early) (B.drawn table first early) source) draws 0
      fun result _ place => place ≤ bound ∧ ∃ r2 : List E, r2.length = ν + 1 ∧
        result = some (r2, mle r2 (lifted E source)))
    (next : L.start = (B.tensored ν).finish)
    (reads : ∀ point, L.read point = statementOf (point.wire L.start))
    (quarter' : 4 * m' = S'.innerDegree) {p s k : ℕ} (width' : 2 * m' = 2 ^ p)
    (length' : S'.blockLength = 2 ^ s) (blocks' : S'.blocks ≤ 2 ^ k) (vars : ν = p + (s + k)) :
    ∃ (later : List ((Option Coord → E) × Wire)) (block position pack : List E) (θ : E),
      pack.length = p ∧ position.length = s ∧ k ≤ block.length ∧
      OnPath accept table first (early ++ later) ∧ (early ++ later).length = L.start ∧
      statementOf (lastWire first (early ++ later)) =
        TerminalStatement.ofPoint quarter' L.matrix
          (fun i => L.matrix.mulVec (traceSource quarter' (packed source : ℕ → E) i))
          block position pack θ (θ * mle (pack ++ (position ++ block)) (packed source)) ∧
      partialsOf (lastWire first (early ++ later)) =
        (fun i => traceContract (subfield (E := E) quarter').toRingHom
          (fun x : Fin S'.blockLength => eqAt position x)
          (gadget S'.sourceBase (traceSource quarter' (packed source : ℕ → E) i))) ∧
      ∀ stage : Prefix Wire (Option Coord → E) total, stage.first = first →
        (∃ rest, stage.rest = (early ++ later) ++ rest) →
        Terminal.Linked (B.tensored ν) L quarter' stateOf stage := by
  obtain ⟨wire, moves, ρ, θ, long, on', whole, held, last, facts⟩ :=
    tensorStep carries sends wide on starts fits decodes worked capped uniform count room source
      wins
      ((fun i => L.matrix.mulVec (traceSource quarter' (packed source : ℕ → E) i)),
        statementOf first, partialsOf first)
      fun ρ θ => ((fun i => L.matrix.mulVec (traceSource quarter' (packed source : ℕ → E) i)),
        TerminalStatement.ofPoint quarter' L.matrix
          (fun i => L.matrix.mulVec (traceSource quarter' (packed source : ℕ → E) i))
          ((ρ.drop p).drop s) ((ρ.drop p).take s) (ρ.take p) θ
          (θ * mle (ρ.take p ++ ((ρ.drop p).take s ++ (ρ.drop p).drop s)) (packed source)),
        fun i => traceContract (subfield (E := E) quarter').toRingHom
          (fun x : Fin S'.blockLength => eqAt ((ρ.drop p).take s) x)
          (gadget S'.sourceBase (traceSource quarter' (packed source : ℕ → E) i)))
  have stated := congrArg (fun data => data.2.1) last
  have cleared := congrArg (fun data => data.2.2) last
  have paid : stateOf wire =
      fun i => L.matrix.mulVec (traceSource quarter' (packed source : ℕ → E) i) :=
    congrArg Prod.fst held
  have joined : ρ.take p ++ ((ρ.drop p).take s ++ (ρ.drop p).drop s) = ρ := by
    rw [List.take_append_drop, List.take_append_drop]
  refine ⟨(answer table first early, wire) :: moves, (ρ.drop p).drop s, (ρ.drop p).take s,
    ρ.take p, θ, by rw [List.length_take]; omega,
    by rw [List.length_take, List.length_drop]; omega,
    by rw [List.length_drop, List.length_drop]; omega, on', whole.trans next.symm,
    by rw [lastWire_append_cons]; exact stated, by rw [lastWire_append_cons]; exact cleared,
    fun stage rooted extended => ?_⟩
  obtain ⟨long', decoded, opened, closed, game⟩ := facts stage rooted extended
  have opened' : stage.wire ((B.tensored ν).start + 1) = wire := opened
  refine ⟨long', decoded, (ρ.drop p).drop s, (ρ.drop p).take s, ρ.take p, θ,
    θ * mle (ρ.take p ++ ((ρ.drop p).take s ++ (ρ.drop p).drop s)) (packed source), ?_,
    blocks'.trans (le_of_eq (by congr 1; simp; omega)), ?_, ?_, ?_⟩
  · rw [joined]
    exact game
  · rw [length']
    congr 1
    simp
    omega
  · rw [width']
    congr 1
    simp
    omega
  · show ({ L.read (stage.take L.start) with matrix := L.matrix } : TerminalStatement S' E) = _
    rw [reads, Prefix.wire_take le_rfl, next, closed, opened', paid]
    exact congrArg (fun st : TerminalStatement S' E => { st with matrix := L.matrix }) stated

end Tensor

section Path

variable {Wire Reply : Type} [Nonempty Reply] {total : ℕ}
  {accept : Prefix Wire Reply total → Prop} {table : Prefix Wire Reply total → Reply}

/-- A stage input on the path has an extension on the path with each stage of the transcript:
the later messages are copies of the first message. -/
theorem OnPath.fill {first : Wire} {moves : List (Reply × Wire)}
    (on : OnPath accept table first moves) (short : moves.length ≤ total) :
    ∃ later, OnPath accept table first (moves ++ later) ∧ (moves ++ later).length = total := by
  obtain ⟨gap, sized⟩ : ∃ gap, total = moves.length + gap := Nat.exists_eq_add_of_le short
  induction gap generalizing moves with
  | zero => exact ⟨[], by rwa [List.append_nil], by rw [List.append_nil]; exact sized.symm⟩
  | succ gap step =>
      obtain ⟨later, filled, whole⟩ := step (on.snoc (by omega) first)
        (by rw [List.length_append, List.length_singleton]; omega)
        (by rw [List.length_append, List.length_singleton]; omega)
      exact ⟨(answer table first moves, first) :: later,
        by rwa [List.append_assoc, List.singleton_append] at filled,
        by rwa [List.append_assoc, List.singleton_append] at whole⟩

/-- A property of each transcript with the moves of a stage input of the last stage on the path
holds for the complete transcript of a proof with the first message `first`. -/
theorem OnPath.accepts {holds : Prefix Wire Reply total → Prop} {first : Wire}
    {moves : List (Reply × Wire)} (on : OnPath accept table first moves)
    (whole : moves.length = total)
    (each : ∀ stage : Prefix Wire Reply total, stage.first = first → stage.rest = moves →
      holds stage) :
    ∃ proof : ℕ → Wire, proof 0 = first ∧
      holds ((transcript total accept).final (.done proof) table) := by
  obtain ⟨proof, bounded, zero, complete⟩ := on.final whole
  exact ⟨proof, zero, complete ▸ each ⟨first, moves, bounded⟩ rfl rfl⟩

end Path

section Last

variable {S : Shape} {E : Type} [Field E] [Fintype E] [Algebra Zq E] [Degree2 E] {Wire Coord : Type}
  {total : ℕ} {accept : Prefix Wire (Option Coord → E) total → Prop}
  {table : Prefix Wire (Option Coord → E) total → Option Coord → E} {route : Route}
  {L : Terminal S E Wire Coord E total route} {statementOf : Wire → TerminalStatement S E}
  {partialsOf : Wire → Fin S.blocks → Rq S.innerDegree} {m : ℕ}
  {sendable : (M : Type) → M → Prop} {slots : ℕ}

/-- The last messages of the honest prover (Akita Section 8.2 and Theorem 10.9 at the terminal).
`early` has the moves before the fold challenges of the terminal, and its last message has the
statement of the terminal for the bounded source blocks of `ψ(f̄)` and the clear partials. The
prover sends the clear response `z = Σ_i c_i·s_i` for the fold challenges of the table of the
oracle. `short`: for each fold challenge vector of the family, `z` has the bound of the norm
route. Then the complete transcript passes the checks `Terminal.Passed`: (128), (129), (130) and
the norm bound. `reads` and `clears`: the statement and the clear partials of the terminal are
those of the message before its fold challenges. -/
theorem terminal_passed {read : Reader Wire} (carries : Carries sendable slots read L.response)
    {first : Wire} {early : List ((Option Coord → E) × Wire)} (on : OnPath accept table first early)
    (starts : early.length = L.start) (inside : L.start < total)
    (reads : ∀ point, L.read point = statementOf (point.wire L.start))
    (clears : ∀ point, L.partials point = partialsOf (point.wire L.start))
    (decodes : ∀ input i, (L.decode (table input (L.slot i))).isSome)
    (worked : ∀ input, table input none ∈ workZeros E 57) (quarter : 4 * m = S.innerDegree) (shape
      : S.Bounded) (unit : IsUnit ((2 * m : ℕ) : Zq))
    {g : ℕ → E} {block position pack : List E} (blocks : S.blocks ≤ 2 ^ block.length)
    (length : S.blockLength = 2 ^ position.length) (width : 2 * m = 2 ^ pack.length)
    (live : ∀ cell, S.blocks * (S.blockLength * (2 * m)) ≤ cell → g cell = 0) {θ : E}
    (stated : statementOf (lastWire first early) =
      TerminalStatement.ofPoint quarter L.matrix
        (fun i => L.matrix.mulVec (traceSource quarter g i)) block position pack θ
        (θ * mle (pack ++ (position ++ block)) g))
    (cleared : partialsOf (lastWire first early) =
      fun i => traceContract (subfield (E := E) quarter).toRingHom
        (fun x : Fin S.blockLength => eqAt position x)
        (gadget S.sourceBase (traceSource quarter g i)))
    (short : ∀ c : Fin S.blocks → Rq S.innerDegree, (∀ i, ∃ slot, L.decode slot = some (c i)) →
      route.Short L.bound (traceResponse c (traceSource quarter g))) :
    ∃ later, OnPath accept table first (early ++ later) ∧ (early ++ later).length = total ∧
      ∀ stage : Prefix Wire (Option Coord → E) total, stage.first = first →
        stage.rest = early ++ later → L.Passed stage := by
  obtain ⟨reply, answered⟩ : ∃ reply, reply = answer table first early := ⟨_, rfl⟩
  have room : early.length ≤ total := by omega
  have decoded : ∀ i,
      L.decode (reply (L.slot i)) = some ((L.decode (reply (L.slot i))).getD 0) :=
    fun i => Extract.decoded (by
      rw [answered, answer_eq table first room]
      exact decodes _ i)
  obtain ⟨c, challenged⟩ : ∃ c : Fin S.blocks → Rq S.innerDegree,
      c = fun i => (L.decode (reply (L.slot i))).getD 0 := ⟨_, rfl⟩
  obtain ⟨wire, sent, _⟩ := carries (traceResponse c (traceSource quarter g)) []
    (Nat.zero_le _) (by simp)
  have on' := on.snoc (by omega) wire
  rw [← answered] at on'
  obtain ⟨rest, filled, whole⟩ := on'.fill
    (by rw [List.length_append, List.length_singleton]; omega)
  rw [List.append_assoc, List.singleton_append] at filled whole
  refine ⟨(reply, wire) :: rest, filled, whole, fun stage rooted head => ?_⟩
  have replied : stage.reply L.start = some reply := stage.reply_of head starts.symm
  have responded : stage.wire (L.start + 1) = wire :=
    stage.wire_of head (congrArg (· + 1) starts.symm)
  have claimed : (stage.take L.start).wire L.start = lastWire first early := by
    rw [Prefix.wire_take le_rfl, ← starts, ← rooted]
    exact Prefix.wire_length head
  have fold : L.challenge stage = c := by
    funext i
    show ((stage.reply L.start).bind fun reply => L.decode (reply (L.slot i))).getD 0 = c i
    rw [replied, challenged]
    rfl
  have statement : L.statement stage = TerminalStatement.ofPoint quarter L.matrix
      (fun i => L.matrix.mulVec (traceSource quarter g i)) block position pack θ
      (θ * mle (pack ++ (position ++ block)) g) := by
    show ({ L.read (stage.take L.start) with matrix := L.matrix } : TerminalStatement S E) = _
    rw [reads, claimed, stated]
    rfl
  have clear : L.clear stage = fun i => traceContract (subfield (E := E) quarter).toRingHom
      (fun x : Fin S.blockLength => eqAt position x)
      (gadget S.sourceBase (traceSource quarter g i)) := by
    show L.partials (stage.take L.start) = _
    rw [clears, claimed, cleared]
  have zero : reply none ∈ workZeros E L.foldBits :=
    Tier.path_worked filled (later := []) (by rw [head, List.append_nil]) worked
      (index := L.start) (by rw [whole]; omega) (foldWorkBits_le _ _) reply replied
  refine ⟨⟨reply, replied, zero, fun i => Option.isSome_iff_exists.mpr ⟨_, decoded i⟩⟩, ?_, ?_⟩
  · rw [statement, clear, fold, responded, sent]
    exact terminal_accepts
      (st := TerminalStatement.ofPoint quarter L.matrix
        (fun i => L.matrix.mulVec (traceSource quarter g i)) block position pack θ
        (θ * mle (pack ++ (position ++ block)) g))
      (traceSource quarter g) c (fun _ => rfl) (congrArg (θ * ·)
        (traceScalar_source quarter shape unit block position pack blocks length width live))
  · rw [responded, sent]
    exact short c fun i => ⟨_, by rw [challenged]; exact decoded i⟩

end Last

end NightstreamFPrime.Spec.Compression.Akita.Extract
