import NightstreamFPrime.Spec.Compression.Akita.Extract.Groups
import NightstreamFPrime.Spec.Compression.Akita.Extract.OffloadComplete
import NightstreamFPrime.Spec.Compression.Akita.Extract.GroupComplete

/-! The honest prover of an offloaded root fold of Akita and its link to the two claims of the
next level (ePrint 2026/1983, revision of 2026-11-23, Theorem 10.9 with Lemma 10.7, Section 9
with (136), Lemma 7.1, item 1, and Figure 4).

- `offLevel_wins` gives the honest prover of the checks of the level `offBatch` for the claim
  `Batch.honestClaim`: the checks end in the claim `(r_2, w̃(r_2))` for the table `batchNext` and
  in the claim `(ρ_S, S̃^♭(ρ_S))` for the setup prefix.
- `offLink` gives the transcript prefix of the honest prover through the stages of an offloaded
  level: the first message has the commitment of the group 1 of the next level, the last message
  has the commitments, the values and the points of the two claims of Akita (136), and each
  transcript with the prefix has the link `Tier.Offloaded`.

The fit conditions `BatchFit` are hypotheses (Akita Definition 10.5). The file has no bound for
the probability `ξ_h` of Akita (155). -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Extract

open scoped BigOperators
open NightstreamFPrime.Spec.Compression.Lib (mle)
open NightstreamFPrime.Spec.Checkpoint

local instance : Fact (Nat.Prime fieldModulus) := ⟨FieldPrime.fieldModulus_natPrime⟩

attribute [local instance low] Classical.propDecidable

section Groups

variable {n : ℕ} {B : Batch (Fin n)} {E : Type} [Field E] [Fintype E] [DecidableEq E]
  [Algebra Zq E] {Wire Coord : Type}
  {β : ∀ g, Module.Basis (Fin (B.planes g)) Zq E} {key : (g : Fin n) → Key (B.shape g)}
  {openKey : B.OpenKey} {claim : Wire → B.Claim E} {μ : ℕ} {cells : B.cells ≤ 2 ^ μ}
  {base digits : ℕ} {room : B.quotientRoom digits ≤ 2 ^ μ} {batch : ℕ}
  {rowsFit : B.rowCount + 1 ≤ 2 ^ batch} {block : B.Block → Coord}
  {decode : (g : Fin n) → E → Option (Rq (B.carrierDegree g))} {read : Reader Wire}
  {mass : Fin n → ℕ} {start ν : ℕ} {flat : Table}
  {scan : ∀ (wire : Wire) (challenge : B.Challenge),
    SetupScan (Batch.liftedReduction β ((claim wire).statement key openKey) challenge μ cells base
      digits room batch rowsFit) ν flat}

/-- The honest prover of the checks of an offloaded root fold with opening groups (Akita
Lemma 10.7, Lemma 7.1, item 1, Section 9 and Theorem 10.9). The message `wire` has the claim
`Batch.honestClaim`, and `fit` has the fit conditions. Then for each fold challenge vector of the
families, `after` holds for the table `batchNext`, this table is 0 from the end
`Batch.quotientRoom` of the quotient digits, and the prover has a message for each draw sequence
of the checks of `offBatch` for it. The checks end in the claims `(r_2, v_W)` for this table and
`(ρ_S, s_{ρ_S})` for the setup prefix `flat`. The hypotheses are those of `batchLevel_wins`, and
`space`: a wire message has room for three messages of the checks. -/
theorem offLevel_wins {sendable : (M : Type) → M → Prop} {slots width cap : ℕ}
    (sends : Sends sendable E width cap) (range : rangeWidth base ≤ width) (four : 4 ≤ width)
    (space : 2 < slots) (shape : ∀ g, (B.shape g).Honest)
    (radix : ∀ g, Radix (B.fold g).responseBase)
    (certified : ∀ g, (B.fold g).responseBase ≤ (B.shape g).certifiedBase)
    (path : PathFits B.openDegree (1 * B.openRank) B.chain)
    (uniform : ∀ g, (B.shape g).certifiedBase = base) (digit : Radix base)
    (covers : fieldModulus ≤ base ^ digits)
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
        ((offBatch β key openKey claim μ cells base digits room batch rowsFit block decode read
          mass start ν flat scan).check wire c
          (batchNext β key openKey μ cells base digits table point weight fun g i => c ⟨g, i⟩))
        (offloadDraws μ base B.compressed ν) 0 fun result _ _ =>
          ∃ r2 ρ : List E, r2.length = μ ∧ ρ.length = ν ∧
            result = some ((r2, mle r2 (lifted E
              (batchNext β key openKey μ cells base digits table point weight
                fun g i => c ⟨g, i⟩))), (ρ, mle ρ (lifted E flat))) := by
  obtain ⟨encodable, later⟩ := fit point weight (fun g i => c ⟨g, i⟩) fun g i => member ⟨g, i⟩
  have satisfies := Batch.honestClaim_satisfies (β := β) (key := key) (openKey := openKey) table
    point weight (fun g i => c ⟨g, i⟩) shape radix certified path encodable
  rw [← ready] at satisfies
  have wins := offBatch_winsAt (fits := cells) (room := room) (batch := batch)
    (rowsFit := rowsFit) (block := block) (decode := decode) (read := read) (mass := mass)
    (start := start) (ν := ν) (flat := flat) (scan := scan) sends range four uniform digit covers
    space satisfies
  rw [ready] at wins
  exact ⟨later, fun x large => FlatRelation.completion_of_le _
    (Batch.flatRelation_counts β _ _ μ cells digits) _
    ((Batch.flatRelation_quotientEnd β _ _ μ cells digits).trans_le large), wins⟩

end Groups

section Link

variable {E : Type} [Field E] [Fintype E] [DecidableEq E] [Algebra Zq E] {Block : Type}
  {Ring : Block → Type} {Stmt Brk Wit Wk Wire Coord : Type} [∀ b, Zero (Ring b)]
  [Fintype Coord] [DecidableEq Coord] {total : ℕ}
  {accept : Prefix Wire (Option Coord → E) total → Prop}
  {table : Prefix Wire (Option Coord → E) total → Option Coord → E}
  {T : Tier Block Ring Stmt Brk Wit Wk Wire Coord E (Option ((List E × E) × (List E × E))) Table}
  {B : Batch (Fin 2)} {commits : Wire → (g : Fin 2) → (B.shape g).Payload}
  {values : Wire → Fin 2 → E} {places : Wire → Fin 2 → List E}
  {commit : Wire → (B.shape 1).Payload} {sendable : (M : Type) → M → Prop} {slots : ℕ}

omit [DecidableEq E] in
/-- The transcript prefix of the honest prover through the stages of an offloaded level `T`
(Akita Section 9 with (136), and Figure 4, steps 1, 3 and 5). `wins`: the prover has a message
for each draw sequence of the checks of `T` for the table `source`, which end in the true claims
`(r_2, w̃(r_2))` of `μ` coordinates and `(ρ_S, S̃^♭(ρ_S))` of `ν` coordinates for the setup
prefix `flat`. The honest prover sends the commitment `payload` of the group 1 of the next level
in the message after the fold challenges, and its last message has the commitments
`(registry, payload)`, the values `(s_{ρ_S}, v_W)` and the points `(ρ_S, r_2)` of the two groups.
`tried`: the nonce of the response search in the last message of `early` is below
`responseTries`. The last message of the longer prefix has a nonce below `responseTries` too.
Each transcript with the prefix has the link `Tier.Offloaded`. -/
theorem offLink
    (carries : Carries sendable slots T.read fun wire =>
      (commit wire, commits wire, values wire, places wire))
    {first : Wire} {early : List ((Option Coord → E) × Wire)} (on : OnPath accept table first early)
    (starts : early.length = T.start) (fits : T.finish ≤ total)
    (decodes : ∀ input b, (T.decode b (table input (T.block b))).isSome)
    (worked : ∀ input, table input none ∈ workZeros E 57) (capped : T.Capped)
    (tried : T.read.nonce (lastWire first early) < responseTries)
    (uniform : ∀ wire challenge table other,
      bare (T.check wire challenge table) = bare (T.check wire challenge other))
    {draws μ ν : ℕ} (count : T.costs.length = draws) (source flat : Table)
    (wins : WinsAt sendable slots
      (T.check (lastWire first early) (T.drawn table first early) source) draws 0
      fun result _ _ => ∃ r2 ρ : List E, r2.length = μ ∧ ρ.length = ν ∧
        result = some ((r2, mle r2 (lifted E source)), (ρ, mle ρ (lifted E flat))))
    (registry : (B.shape 0).Payload) (payload : (B.shape 1).Payload) :
    ∃ (later : List ((Option Coord → E) × Wire)) (r2 ρ : List E), r2.length = μ ∧
      ρ.length = ν ∧ OnPath accept table first (early ++ later) ∧
      (early ++ later).length = T.finish ∧
      T.read.nonce (lastWire first (early ++ later)) < responseTries ∧
      commits (lastWire first (early ++ later)) =
        Batch.pairOf (F := fun g => (B.shape g).Payload) registry payload ∧
      values (lastWire first (early ++ later)) =
        ![mle ρ (lifted E flat), mle r2 (lifted E source)] ∧
      places (lastWire first (early ++ later)) = ![ρ, r2] ∧
      ∀ stage : Prefix Wire (Option Coord → E) total, stage.first = first →
        (∃ rest, stage.rest = (early ++ later) ++ rest) →
        T.Offloaded commits values places registry commit stage := by
  subst count
  obtain ⟨wire, moves, result, counted, ⟨r2, ρ, long, long', rfl⟩, fresh, on', held, last,
      ended, facts⟩ :=
    Tier.run (T := T) carries on starts fits decodes worked capped tried source wins fun _ game =>
      (payload, Batch.pairOf (F := fun g => (B.shape g).Payload) registry payload,
        match game with
        | .pure (some ((r, v), (σ, s))) => (![s, v], ![σ, r])
        | _ => (0, fun _ => []))
  have paid : commit wire = payload := congrArg Prod.fst held
  have committed : commits (lastWire wire moves) =
      Batch.pairOf (F := fun g => (B.shape g).Payload) registry payload :=
    congrArg (fun data => data.2.1) last
  have valued : values (lastWire wire moves) =
      ![mle ρ (lifted E flat), mle r2 (lifted E source)] :=
    congrArg (fun data => data.2.2.1) last
  have placed : places (lastWire wire moves) = ![ρ, r2] :=
    congrArg (fun data => data.2.2.2) last
  have whole : (early ++ (answer table first early, wire) :: moves).length = T.finish := by
    rw [List.length_append, List.length_cons, counted, starts]
    exact T.finish_eq.symm
  refine ⟨(answer table first early, wire) :: moves, r2, ρ, long, long', on', whole,
    by rw [lastWire_append_cons]; exact fresh, by rw [lastWire_append_cons]; exact committed,
    by rw [lastWire_append_cons]; exact valued,
    by rw [lastWire_append_cons]; exact placed, fun stage rooted extended => ?_⟩
  obtain ⟨long'', decoded, _, opened, closed, game⟩ := facts stage rooted extended
  refine ⟨long'', decoded, r2, _, ρ, _,
    (game _).trans (ended_of_bare T.read (uniform _ _ source fun _ => 0) ended), ?_, ?_, ?_, ?_⟩
  · rw [closed, placed]
  · rw [closed, valued]
  · rw [closed, committed]
    rfl
  · rw [closed, committed, opened, paid]
    rfl

end Link

end NightstreamFPrime.Spec.Compression.Akita.Extract
