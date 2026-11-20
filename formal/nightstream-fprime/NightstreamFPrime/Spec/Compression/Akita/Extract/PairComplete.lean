import NightstreamFPrime.Spec.Compression.Akita.Extract.Honest

/-! The verifier `pairAccepts` of two packing folds accepts the proofs of the honest prover
(Akita, ePrint 2026/1983, revision of 2026-10-15, Theorem 10.9 with Lemma 10.7 and Figure 4).
The extraction theorems `packingPair_extract_sound`, `packingPair_extraction_le` and
`packingPair_runs_le` bound an event of this verifier, so they are not empty statements.

- `honestClaim` is the claim of the honest prover for a table and an opening point, and
  `honestClaim_satisfies` is `honest_satisfies` for it.
- `Full` states that a wire message can have each claim, each commitment, each family of digit
  vectors and each message of the checks.
- `packingPair_complete` gives, for each table of the oracle whose slots decode to fold
  challenges, a proof whose transcript the verifier accepts. The honest prover commits the flat
  witness of the first level with `commit` for the second level.

The probability that a response is not admissible (`ξ_h` of Akita (155)) is not in this file:
the hypotheses `encodable` and `encodable'` state that each response is admissible. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Extract

open scoped BigOperators
open NightstreamFPrime.Spec.Compression.Lib NightstreamFPrime.Spec.Compression.Game
open NightstreamFPrime.Spec.Compression (Game)
open NightstreamFPrime.Spec.Checkpoint
open NightstreamFPrime.Spec.RandomOracle (OracleComp)

local instance : Fact (Nat.Prime fieldModulus) := ⟨FieldPrime.fieldModulus_natPrime⟩

attribute [local instance low] Classical.propDecidable

section Prefix

variable {Wire Reply : Type} {total : ℕ} (point : Prefix Wire Reply total)
  {before after : List (Reply × Wire)} {move : Reply × Wire} {index : ℕ}

theorem Prefix.reply_of (shape : point.rest = before ++ move :: after)
    (place : index = before.length) : point.reply index = some move.1 := by
  simp [Prefix.reply, shape, place]

theorem Prefix.wire_of (shape : point.rest = before ++ move :: after)
    (place : index = before.length + 1) : point.wire index = move.2 := by
  simp [Prefix.wire, shape, place]

end Prefix

theorem decoded {Slot Ring : Type} [Zero Ring] {decode : Slot → Option Ring} {slot : Slot}
    (some : (decode slot).isSome) : decode slot = .some ((decode slot).getD 0) := by
  obtain ⟨value, found⟩ := Option.isSome_iff_exists.mp some
  rw [found]
  rfl

section Claim

variable {S : Shape} {O : OpeningShape S} {fold : Descriptor} {E : Type} [Field E]
  [Algebra Zq E]

/-- The claim of the honest prover for a table and an opening point (Akita Figure 4, step 1):
the payload `u_pub` of `commit`, the payload `v_pub` of the opening path for the digits `ê`, and
the value (59) of the partials. -/
noncomputable def honestClaim (β : Module.Basis (Fin O.planes) Zq E) (key : Key S)
    (openKey : OpeningKey O) (table : Table) (point : Point O E) : Claim O E :=
  ⟨(commit key table).1,
    (honestOpening openKey (honestPartials β point (commit key table).2.source)).2, point,
    scalar β point fun i t =>
      contract β point t (gadget S.sourceBase ((commit key table).2.source i))⟩

variable {β : Module.Basis (Fin O.planes) Zq E} {key : Key S} {openKey : OpeningKey O}

/-- `honest_satisfies` for the claim `honestClaim` and the commitment state of `commit`. -/
theorem honestClaim_satisfies (table : Table) (point : Point O E)
    (challenge : Fin S.blocks → Rq O.carrierDegree) (shape : S.Honest)
    (radix : Radix fold.responseBase) (certified : fold.responseBase ≤ S.certifiedBase)
    (path : PathFits O.openDegree (1 * O.openRank) O.chain)
    (encodable : Encodable fold.responseBase fold.responseDigits
      (response O challenge (commit key table).2.source)) :
    (honest β openKey point fold challenge (commit key table).2).Satisfies β
      ((honestClaim β key openKey table point).statement key openKey) challenge :=
  honest_satisfies (st := (honestClaim β key openKey table point).statement key openKey)
    challenge shape radix certified path (commit_opens shape key table) encodable rfl rfl

/-- The value of the claim `honestClaim` at a point with four segments is the multilinear
evaluation of the table (Akita Section 5.3), for a table of `B·M·d_A` cells. -/
theorem honestClaim_value (shape : S.Honest) (table : Table) (block position pack tail : List E)
    (blocks : S.blocks = 2 ^ block.length) (length : S.blockLength = 2 ^ position.length)
    (stride : O.stride = 2 ^ pack.length) (degree : O.carrierDegree = 2 ^ tail.length) :
    (honestClaim β key openKey table (Point.ofSegments O block position pack tail)).value =
      mle (pack ++ (tail ++ (position ++ block))) (lifted E table) := by
  have total : 2 ^ (pack ++ (tail ++ (position ++ block))).length =
      S.blockLength * S.innerDegree * S.blocks := by
    rw [blocks, length, ← O.stride_tower, stride, degree]
    simp only [List.length_append, pow_add]
    ring
  show scalar β _ (fun i t => contract β _ t (gadget S.sourceBase
    (decompose S.sourceBase (S.pack table i)))) = _
  rw [scalar_contract_eq_mle β block position pack tail blocks length stride degree,
    decode_decompose_pack shape table]
  refine mle_congr _ fun cell small => ?_
  rw [total] at small
  show algebraMap Zq E (S.pad table cell) = algebraMap Zq E (table cell)
  rw [Shape.pad, ite_eq_left (Nat.div_lt_of_lt_mul small)]

end Claim

section Pair

variable {S : Shape} {O : OpeningShape S} {fold : Descriptor} {S' : Shape} {O' : OpeningShape S'}
  {fold' : Descriptor} {E : Type} [Field E] [Fintype E] [DecidableEq E] [Algebra Zq E]
  {Wire Coord : Type} [Fintype Coord] [DecidableEq Coord]

/-- A wire message can have each claim of the first level, and with each message of the checks
of a type of `fits`: each commitment of the next level, each claim of the next level, or each
family of digit vectors of the next level. -/
structure Full (fits : Type → Prop) (claim : Wire → Claim O E) (claim' : Wire → Claim O' E)
    (payloadOf : Wire → S'.Payload) (witnessOf : Wire → Witness O' fold') (read : Reader Wire) :
    Prop where
  root : ∀ value, ∃ wire, claim wire = value
  message : ∀ (M : Type) [Nonempty M] (message : M), fits M → ∃ wire, read M wire 0 = message
  commitment : ∀ value (M : Type) [Nonempty M] (message : M), fits M →
    ∃ wire, payloadOf wire = value ∧ read M wire 0 = message
  next : ∀ value (M : Type) [Nonempty M] (message : M), fits M →
    ∃ wire, claim' wire = value ∧ read M wire 0 = message
  digits : ∀ value (M : Type) [Nonempty M] (message : M), fits M →
    ∃ wire, witnessOf wire = value ∧ read M wire 0 = message

variable {β : Module.Basis (Fin O.planes) Zq E} {key : Key S} {openKey : OpeningKey O}
  {claim : Wire → Claim O E} {μ : ℕ} {cells : O.cells fold < 2 ^ μ} {batch : ℕ}
  {rowsFit : O.rowCount + 1 + 1 ≤ 2 ^ batch} {block : Fin S.blocks → Coord}
  {decode : E → Option (Rq O.carrierDegree)} {read : Reader Wire} {weight : ℕ}
  {β' : Module.Basis (Fin O'.planes) Zq E} {key' : Key S'} {openKey' : OpeningKey O'}
  {claim' : Wire → Claim O' E} {μ' : ℕ} {cells' : O'.cells fold' < 2 ^ μ'} {batch' : ℕ}
  {rowsFit' : O'.rowCount + 1 ≤ 2 ^ batch'} {block' : Fin S'.blocks → Coord}
  {decode' : E → Option (Rq O'.carrierDegree)} {weight' : ℕ} {payloadOf : Wire → S'.Payload}
  {witnessOf : Wire → Witness O' fold'}

local notation "first" =>
  committedFold β key openKey claim μ cells batch rowsFit block decode read weight 0

local notation "second" =>
  clearFold β' key' openKey' claim' μ' cells' batch' rowsFit' block' decode' read weight'
    (Tier.finish first)

local notation "chain" => pair first second payloadOf witnessOf

/-- Akita Theorem 10.9 for a committed packing fold over a last packing fold, without the
probability that a response is not admissible: the verifier `pairAccepts` of
`packingPair_extraction_le` accepts the honest prover. For each table `source`, each opening
point and each table of the oracle whose slots decode to fold challenges, a proof exists such
that its first message has the claim `honestClaim` and the verifier accepts its transcript.

The hypotheses are: `full`, `scalar`, `cubic`, `range`, `range'`: a wire message can have the
data of the prover. `shape`, `radix`, `certified`, `path`, `two`, `small` and the same for the
second level: the conditions of `commit_opens` and `honest_satisfies`, with `2 ≤ b⋆ < q`.
`stride`, `degree`, `length`, `blocks`, `vars`: the table of the second level has the `2^μ` cells
of the cube of the first level, in segments of powers of two. `encodable`, `encodable'`: the
response of each level is admissible (Akita Definition 10.5) for each fold challenge of the
family. -/
theorem packingPair_complete (fits : Type → Prop)
    (full : Full fits claim claim' payloadOf witnessOf read) (scalar : fits E)
    (cubic : fits (Fin (3 + 1) → E)) (range : fits (Fin (S.certifiedBase / 2 + 1 + 1) → E))
    (range' : fits (Fin (S'.certifiedBase / 2 + 1 + 1) → E)) (shape : S.Honest)
    (radix : Radix fold.responseBase) (certified : fold.responseBase ≤ S.certifiedBase)
    (path : PathFits O.openDegree (1 * O.openRank) O.chain) (two : 2 ≤ S.certifiedBase)
    (small : S.certifiedBase < fieldModulus) (shape' : S'.Honest)
    (radix' : Radix fold'.responseBase) (certified' : fold'.responseBase ≤ S'.certifiedBase)
    (path' : PathFits O'.openDegree (1 * O'.openRank) O'.chain) (two' : 2 ≤ S'.certifiedBase)
    (small' : S'.certifiedBase < fieldModulus) {p t s k : ℕ} (stride : O'.stride = 2 ^ p)
    (degree : O'.carrierDegree = 2 ^ t) (length : S'.blockLength = 2 ^ s)
    (blocks : S'.blocks = 2 ^ k) (vars : μ = p + (t + (s + k))) (source : Table)
    (point : Point O E)
    (encodable : ∀ challenge, (∀ i, ∃ slot, decode slot = some (challenge i)) →
      Encodable fold.responseBase fold.responseDigits
        (response O challenge (commit key source).2.source))
    (encodable' : ∀ challenge challenge', (∀ i, ∃ slot, decode slot = some (challenge i)) →
      (∀ i, ∃ slot, decode' slot = some (challenge' i)) →
      Encodable fold'.responseBase fold'.responseDigits
        (response O' challenge' (commit key'
          (honest β openKey point fold challenge (commit key source).2).flat).2.source))
    (table : Prefix Wire (Coord → E) (Tier.finish second) → Coord → E)
    (decodes : ∀ input i, (decode (table input (block i))).isSome)
    (decodes' : ∀ input i, (decode' (table input (block' i))).isSome) :
    ∃ proof : ℕ → Wire, claim (proof 0) = honestClaim β key openKey source point ∧
      pairAccepts first second payloadOf witnessOf
        ((Structure.protocol chain).final (.done proof) table) := by
  have stages : Tier.finish second =
      0 + μ + 1 + 1 + μ + 1 + 1 + 1 + 1 + (0 + μ' + 1 + 1 + μ' + 1 + 1 + 1 + 1) := by
    show 0 + ((foldCosts E _ μ batch _).length + 1) + ((foldCosts E _ μ' batch' _).length + 1) = _
    rw [foldCosts_length, foldCosts_length]
    omega
  obtain ⟨accept, accepts⟩ : ∃ accept, accept = pairAccepts first second payloadOf witnessOf :=
    ⟨_, rfl⟩
  obtain ⟨wire0, root⟩ := full.root (honestClaim β key openKey source point)
  obtain ⟨reply0, answered0⟩ : ∃ reply, reply = answer table wire0 [] := ⟨_, rfl⟩
  have decoded0 : ∀ i, decode (reply0 (block i)) = some ((decode (reply0 (block i))).getD 0) :=
    fun i => decoded (by
      rw [answered0, answer_eq table wire0 (Nat.zero_le _)]
      exact decodes _ i)
  obtain ⟨c, challenged⟩ : ∃ c : Fin S.blocks → Rq O.carrierDegree,
      c = fun i => (decode (reply0 (block i))).getD 0 := ⟨_, rfl⟩
  have member : ∀ i, ∃ slot, decode slot = some (c i) := fun i =>
    ⟨_, by rw [challenged]; exact decoded0 i⟩
  have satisfies : (honest β openKey point fold c (commit key source).2).Satisfies β
      ((claim wire0).statement key openKey) c := by
    rw [root]
    exact honestClaim_satisfies source point c shape radix certified path (encodable c member)
  obtain ⟨flat, flattened⟩ : ∃ flat : Table,
      flat = (honest β openKey point fold c (commit key source).2).flat := ⟨_, rfl⟩
  obtain ⟨next, nexts⟩ : ∃ next : List E → Claim O' E, next = fun r =>
      honestClaim β' key' openKey' flat (Point.ofSegments O' (((r.drop p).drop t).drop s)
        (((r.drop p).drop t).take s) (r.take p) ((r.drop p).take t)) := ⟨_, rfl⟩
  have wins := lockedFold_winsTo (cells := cells) fits scalar range cubic two small batch rowsFit
    satisfies
  rw [← flattened] at wins
  have free : ∀ (index : ℕ) (M : Type) [Nonempty M] (message : M)
      (after : Game (Option (List E × E))), fits M →
      ∃ wire, ((index = 0 → payloadOf wire = (commit key' flat).1) ∧
        (index ≠ 0 → ∀ r v, after = Game.pure (some (r, v)) → claim' wire = next r)) ∧
        read M wire 0 = message := by
    intro index M _ message after sendable
    by_cases start : index = 0
    · obtain ⟨wire, paid, sent⟩ := full.commitment (commit key' flat).1 M message sendable
      exact ⟨wire, ⟨fun _ => paid, fun later => absurd start later⟩, sent⟩
    by_cases ended : ∃ r v, after = Game.pure (some (r, v))
    · obtain ⟨r, v, rfl⟩ := ended
      obtain ⟨wire, linked, sent⟩ := full.next (next r) M message sendable
      refine ⟨wire, ⟨fun zero => absurd zero start, fun _ r' v' same => ?_⟩, sent⟩
      cases same
      exact linked
    · obtain ⟨wire, sent⟩ := full.message M message sendable
      exact ⟨wire, ⟨fun zero => absurd zero start, fun _ r v same => absurd ⟨r, v, same⟩ ended⟩,
        sent⟩
  obtain ⟨wire1, moves, _, counted, ⟨paid, on, last⟩, ended, r2, measured, rfl⟩ :=
    play read fits wins
      (fun index after wire => (index = 0 → payloadOf wire = (commit key' flat).1) ∧
        (index ≠ 0 → ∀ r v, after = Game.pure (some (r, v)) → claim' wire = next r))
      (fun index after => (free index E 0 after scalar).imp fun _ kept => kept.1) free
      (fun wire moves => answer table wire0 ((reply0, wire) :: moves))
      (fun wire moves => payloadOf wire = (commit key' flat).1 ∧
        OnPath accept table wire0 ((reply0, wire) :: moves) ∧
        ∀ r v, moves ≠ [] →
          position read (Tier.check first wire0 c flat) wire moves = Game.pure (some (r, v)) →
          ∃ init final, moves = init ++ [final] ∧ claim' final.2 = next r)
      (fun wire kept => ⟨kept.1 rfl, by
        have on := (onPath_nil accept table wire0).snoc
          (by rw [stages, List.length_nil]; omega) wire
        rw [← answered0] at on
        exact on, fun _ _ empty => absurd rfl empty⟩)
      (fun wire moves later room kept linked => ⟨kept.1,
        kept.2.1.snoc (by rw [List.length_cons]; omega) later, fun r v _ final =>
          ⟨moves, _, rfl, linked.2 (Nat.succ_ne_zero _) r v final⟩⟩)
  obtain ⟨init, final, rfl, linked⟩ := last r2 _ (by
    rintro rfl
    exact absurd counted.symm (Nat.succ_ne_zero _)) ended
  rw [List.length_append, List.length_singleton] at counted
  obtain ⟨early, earlier⟩ : ∃ early : List ((Coord → E) × Wire),
      early = (reply0, wire1) :: (init ++ [final]) := ⟨_, rfl⟩
  rw [← earlier] at on
  have long : early.length = 0 + μ + 1 + 1 + μ + 1 + 1 + 1 + 1 := by
    rw [earlier, List.length_cons, List.length_append, List.length_singleton, counted]
  have short : early.length ≤ Tier.finish second := by omega
  obtain ⟨reply1, answered1⟩ : ∃ reply, reply = answer table wire0 early := ⟨_, rfl⟩
  have decoded1 : ∀ i,
      decode' (reply1 (block' i)) = some ((decode' (reply1 (block' i))).getD 0) :=
    fun i => decoded (by
      rw [answered1, answer_eq table wire0 short]
      exact decodes' _ i)
  obtain ⟨c', challenged'⟩ : ∃ c' : Fin S'.blocks → Rq O'.carrierDegree,
      c' = fun i => (decode' (reply1 (block' i))).getD 0 := ⟨_, rfl⟩
  have member' : ∀ i, ∃ slot, decode' slot = some (c' i) := fun i =>
    ⟨_, by rw [challenged']; exact decoded1 i⟩
  obtain ⟨digits, digited⟩ : ∃ digits : Witness O' fold',
      digits = honest β' openKey' (next r2).point fold' c' (commit key' flat).2 := ⟨_, rfl⟩
  have satisfies' : digits.Satisfies β' ((claim' final.2).statement key' openKey') c' := by
    rw [linked, digited, nexts]
    refine honestClaim_satisfies flat _ c' shape' radix' certified' path' ?_
    rw [flattened]
    exact encodable' c c' member member'
  have wins' := flatFold_winsTo (cells := cells') fits scalar range' cubic two' small' batch'
    rowsFit' satisfies'
  obtain ⟨wire2, moves', _, counted', ⟨held, on'⟩, ended', r3, _, rfl⟩ :=
    play read fits wins'
      (fun index _ wire => index = 0 → witnessOf wire = digits)
      (fun index after => by
        obtain ⟨wire, held, _⟩ := full.digits digits E 0 scalar
        exact ⟨wire, fun _ => held⟩)
      (fun index M _ message after sendable => by
        obtain ⟨wire, held, sent⟩ := full.digits digits M message sendable
        exact ⟨wire, fun _ => held, sent⟩)
      (fun wire moves => answer table wire0 (early ++ (reply1, wire) :: moves))
      (fun wire moves => witnessOf wire = digits ∧
        OnPath accept table wire0 (early ++ (reply1, wire) :: moves))
      (fun wire kept => ⟨kept rfl, by
        have on := on.snoc (by omega) wire
        rw [← answered1] at on
        exact on⟩)
      (fun wire moves later room kept _ => ⟨kept.1, by
        have on := kept.2.snoc (by
          rw [List.length_append, List.length_cons]
          omega) later
        rwa [List.append_assoc, List.cons_append] at on⟩)
  have whole : (early ++ (reply1, wire2) :: moves').length = Tier.finish second := by
    rw [List.length_append, List.length_cons]
    omega
  obtain ⟨proof, bounded, zero, complete⟩ := on'.final whole
  refine ⟨proof, zero ▸ root, ?_⟩
  show pairAccepts first second payloadOf witnessOf
    ((transcript (Tier.finish second) (pairAccepts first second payloadOf witnessOf)).final
      (.done proof) table)
  rw [← accepts, complete]
  obtain ⟨stage, staged⟩ : ∃ stage : Prefix Wire (Coord → E) (Tier.finish second),
      stage = ⟨wire0, early ++ (reply1, wire2) :: moves', bounded⟩ := ⟨_, rfl⟩
  rw [← staged]
  have rest : stage.rest = early ++ (reply1, wire2) :: moves' := by rw [staged]
  have head : stage.rest = [] ++ (reply0, wire1) ::
      (init ++ final :: (reply1, wire2) :: moves') := by
    rw [rest, earlier]
    simp
  have middle : stage.rest = ((reply0, wire1) :: init) ++ final :: (reply1, wire2) :: moves' := by
    rw [rest, earlier]
    simp
  have finished : Tier.finish first = early.length := by
    show 0 + ((foldCosts E _ μ batch _).length + 1) = _
    rw [foldCosts_length, long]
    omega
  have rooted : stage.wire 0 = wire0 := by rw [staged]; rfl
  have game : ∀ candidate, Tier.game first candidate stage =
      position read (Tier.check first wire0 c candidate) wire1 (init ++ [final]) := by
    intro candidate
    have moved : (stage.rest.take (Tier.finish first)).drop (0 + 1) = init ++ [final] := by
      rw [rest, List.take_left' finished.symm, earlier]
      rfl
    have fold : Tier.challenge first stage = c := by
      funext i
      show ((stage.reply 0).bind fun reply => decode (reply (block i))).getD 0 = c i
      rw [stage.reply_of (index := 0) head rfl, challenged]
      rfl
    show position read (Tier.check first (stage.wire 0) (Tier.challenge first stage) candidate)
      (stage.wire (0 + 1)) ((stage.rest.take (Tier.finish first)).drop (0 + 1)) = _
    rw [moved, fold, rooted, stage.wire_of (index := 0 + 1) head rfl]
  have linkWire : stage.wire (Tier.finish first) = final.2 := by
    refine stage.wire_of middle ?_
    rw [finished, earlier]
    simp
  have blocked : S'.blocks = 2 ^ (((r2.drop p).drop t).drop s).length := by
    rw [blocks]; congr 1; simp; omega
  have lengthed : S'.blockLength = 2 ^ (((r2.drop p).drop t).take s).length := by
    rw [length]; congr 1; simp; omega
  have strided : O'.stride = 2 ^ (r2.take p).length := by
    rw [stride]; congr 1; simp; omega
  have degreed : O'.carrierDegree = 2 ^ ((r2.drop p).take t).length := by
    rw [degree]; congr 1; simp; omega
  have value := honestClaim_value (β := β') (key := key') (openKey := openKey') shape' flat
    (((r2.drop p).drop t).drop s) (((r2.drop p).drop t).take s) (r2.take p) ((r2.drop p).take t)
    blocked lengthed strided degreed
  have joined : r2.take p ++ ((r2.drop p).take t ++ (((r2.drop p).drop t).take s ++
      ((r2.drop p).drop t).drop s)) = r2 := by simp
  rw [accepts]
  refine ⟨⟨by rw [rest, finished, List.length_append]; omega,
    ⟨reply0, stage.reply_of (index := 0) head rfl, fun i =>
      Option.isSome_iff_exists.mpr ⟨_, decoded0 i⟩⟩,
    ((r2.drop p).drop t).drop s, ((r2.drop p).drop t).take s, r2.take p, (r2.drop p).take t,
    mle r2 (lifted E flat), ?_, blocked, lengthed, strided, degreed, ?_, ?_, ?_⟩, ?_⟩
  · rw [game, joined]
    exact ended_of_bare read (committedFold_uniform wire0 c flat fun _ => 0) ended
  · show (claim' (stage.wire (Tier.finish first))).payload = payloadOf (stage.wire (0 + 1))
    rw [linkWire, linked, stage.wire_of (index := 0 + 1) head rfl, paid, nexts]
    rfl
  · show (claim' (stage.wire (Tier.finish first))).point = _
    rw [linkWire, linked, nexts]
    rfl
  · show (claim' (stage.wire (Tier.finish first))).value = _
    rw [linkWire, linked, nexts]
    exact value.trans (congrArg (fun r => mle r (lifted E flat)) joined)
  · have answered : stage.reply (Tier.finish first) = some reply1 :=
      stage.reply_of rest finished
    have fold : Tier.challenge second stage = c' := by
      funext i
      show ((stage.reply (Tier.finish first)).bind fun reply =>
        decode' (reply (block' i))).getD 0 = c' i
      rw [answered, challenged']
      rfl
    have moved : (stage.rest.take (Tier.finish second)).drop (Tier.finish first + 1) =
        moves' := by
      rw [List.take_of_length_le (by rw [rest, whole]), rest, List.append_cons]
      exact List.drop_left' (by rw [List.length_append, List.length_singleton, finished])
    refine ⟨by rw [rest, whole], ⟨reply1, answered, fun i =>
      Option.isSome_iff_exists.mpr ⟨_, decoded1 i⟩⟩,
      some (r3, mle r3 (lifted E (witnessOf (stage.wire (Tier.finish first + 1))).flat)), ?_,
      r3, _, rfl, rfl⟩
    have game' : position read (Tier.check second final.2 c' digits) wire2 moves' =
        Game.pure (some (r3, mle r3 (lifted E digits.flat))) := ended'
    show position read (Tier.check second (stage.wire (Tier.finish first))
      (Tier.challenge second stage) (witnessOf (stage.wire (Tier.finish first + 1))))
      (stage.wire (Tier.finish first + 1))
      ((stage.rest.take (Tier.finish second)).drop (Tier.finish first + 1)) = _
    rw [moved, fold, linkWire, stage.wire_of rest (congrArg (· + 1) finished), held]
    exact game'

end Pair

end NightstreamFPrime.Spec.Compression.Akita.Extract
