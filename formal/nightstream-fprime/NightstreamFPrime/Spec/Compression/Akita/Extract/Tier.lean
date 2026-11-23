import NightstreamFPrime.Spec.Compression.Akita.Fold
import NightstreamFPrime.Spec.Compression.Akita.Extract.Segment

/-! One level of the opening suffix of Akita as a part of a checkpoint structure (ePrint 2026/1983,
revision of 2026-11-23, Proposition 10.44 with its proof, and Lemma 10.42).

- `Method` has the exact branch relation, the weak opening and the reduction of Akita
  Corollary 10.32 of one opening method. Its fold coordinates have the index type `Block`, and the
  coordinate `b` has the challenge ring `Ring b`: a level with several opening groups has one
  coordinate for each group and source block. `packing` is the method of a fold with coefficient
  packing for one group, and `packing_sound` is `extract_opens_or_collision` for it.
- `Tier` has the data of one fold whose stages start at the stage `start` of a longer
  transcript: the stage `start` has the fold challenges, and the next stages have the draws of the
  checks. The message before the stage `start` has the claim of the level. `Source` is the type of
  the successor source `σ`: the data that the next level, or the proof, gives for the flat witness
  of this fold. `digits` is the decoder of the schedule from `σ` to the digit vectors. `bits`
  has the work bits of each draw of the checks and `foldBits` the work bits of the fold
  challenges (Akita (146), `FoldChallengeWork` of the Akita code). `Tier.Decodes` requires the work
  slot of each of these stages to have its work bits zero.
- `Tier.fork` is the indexed stage of the fold challenges over a structure `mid` of the later
  stages, with `Comb_r` of Akita Corollary 10.32. `Tier.segment` is the checkpoint segment of the
  checks, with the candidate `source`.
- `Tier.over` is the checkpoint structure of the level over the structure `tail` of the later
  stages: `Tier.fork` over `Segment.over` of `Tier.segment`.
- `Tier.Fed` is the interface that the later stages give to the level: their conditions of
  Definition 10.16, the successor source of each of their first certificates with the true final
  claim of the checks (Lemma 10.42 (ii)), and a witness of `R_bind` for two sources after one
  stage input of the checkpoint.
- `Tier.over_lawfulFrom` states that the structure satisfies the conditions of Definition 10.16
  from the stage `start`, and `Tier.over_opened` states that its first certificates have a weak
  opening of the claim of the level (Proposition 10.44 (i)). -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Extract

open NightstreamFPrime.Spec.Checkpoint
open NightstreamFPrime.Spec.Compression (Game)
open NightstreamFPrime.Spec.RandomOracle (mass mass_nonnegative mass_mono)

attribute [local instance low] Classical.propDecidable

section Keys

variable {Slot First Second : Type} [Fintype Slot]

/-- The answer law of an indexed stage for the keys of the first part of a sum. -/
theorem mass_inl_le (key : Slot → Option First) (value : First ⊕ Second) {bound : ℝ}
    (nonnegative : 0 ≤ bound) (each : ∀ first, mass {slot | key slot = some first} ≤ bound) :
    mass {slot | (key slot).map Sum.inl = some value} ≤ bound := by
  rcases value with first | second
  · refine (mass_mono fun slot decoded => ?_).trans (each first)
    obtain ⟨other, found, same⟩ := Option.map_eq_some_iff.mp
      (decoded : (key slot).map Sum.inl = some (Sum.inl first))
    exact found.trans (congrArg some (Sum.inl.inj same))
  · refine (mass_mono (large := ∅) fun slot decoded => ?_).trans
      ((le_of_eq (by simp [mass])).trans nonnegative)
    obtain ⟨other, _, same⟩ := Option.map_eq_some_iff.mp
      (decoded : (key slot).map Sum.inl = some (Sum.inr second))
    exact absurd same (by simp)

/-- The answer law of an indexed stage for the keys of the second part of a sum. -/
theorem mass_inr_le (key : Slot → Option Second) (value : First ⊕ Second) {bound : ℝ}
    (nonnegative : 0 ≤ bound) (each : ∀ second, mass {slot | key slot = some second} ≤ bound) :
    mass {slot | (key slot).map Sum.inr = some value} ≤ bound := by
  rcases value with first | second
  · refine (mass_mono (large := ∅) fun slot decoded => ?_).trans
      ((le_of_eq (by simp [mass])).trans nonnegative)
    obtain ⟨other, _, same⟩ := Option.map_eq_some_iff.mp
      (decoded : (key slot).map Sum.inr = some (Sum.inl first))
    exact absurd same (by simp)
  · refine (mass_mono fun slot decoded => ?_).trans (each second)
    obtain ⟨other, found, same⟩ := Option.map_eq_some_iff.mp
      (decoded : (key slot).map Sum.inr = some (Sum.inr second))
    exact found.trans (congrArg some (Sum.inr.inj same))

end Keys

section Challenge

variable {Wire Coord Slot Block : Type} {Ring : Block → Type} {total : ℕ}
  (decode : (b : Block) → Slot → Option (Ring b)) (block : Block → Option Coord) (start : ℕ)

/-- The fold challenges of a stage input, from the answer of the stage `start`. `decode b` gives
the challenge of the coordinate `b` from a slot, and `block b` is the coordinate of the answer
that has this slot. -/
noncomputable def challengeAt [∀ b, Zero (Ring b)] (point : Prefix Wire (Option Coord → Slot) total)
    (b : Block) : Ring b :=
  ((point.reply start).bind fun reply => decode b (reply (block b))).getD 0

/-- The verifier accepts the answer of the stage `start`: its work slot has `bits` zero work bits
(Akita (146), `FoldChallengeWork` of the code), and it gives a fold challenge for each
coordinate. -/
def DecodesAt [Fintype Slot] (bits : ℕ) (point : Prefix Wire (Option Coord → Slot) total) : Prop :=
  ∃ reply, point.reply start = some reply ∧ reply none ∈ workZeros Slot bits ∧
    ∀ b, (decode b (reply (block b))).isSome

variable {decode block start} in
theorem DecodesAt.challenge [∀ b, Zero (Ring b)] [Fintype Slot] {bits : ℕ}
    {point : Prefix Wire (Option Coord → Slot) total}
    {reply : Option Coord → Slot} (answered : point.reply start = some reply)
    (decodes : DecodesAt decode block start bits point) (b : Block) :
    decode b (reply (block b)) = some (challengeAt decode block start point b) := by
  obtain ⟨other, same, _, each⟩ := decodes
  obtain rfl := Option.some.inj (same.symm.trans answered)
  obtain ⟨value, decoded⟩ := Option.isSome_iff_exists.mp (each b)
  simp [challengeAt, answered, decoded]

theorem challengeAt_take [∀ b, Zero (Ring b)] {level : ℕ} (above : start < level)
    (point : Prefix Wire (Option Coord → Slot) total) :
    challengeAt decode block start (point.take level) = challengeAt decode block start point := by
  funext b
  simp only [challengeAt, Prefix.reply_take above]

end Challenge

section Chance

variable {Slot Ring : Type} (decode : Slot → Option Ring) [Fintype Slot] [Nonempty Slot]

/-- The largest probability of one fold challenge for a uniform slot. It is `1/|C|` for a decoder
with the uniform law on the challenge set `C`. -/
noncomputable def slotChance : ℝ :=
  Finset.univ.sup' Finset.univ_nonempty fun slot : Slot =>
    if (decode slot).isSome then mass {other | decode other = decode slot} else 0

theorem slotChance_nonnegative : 0 ≤ slotChance decode := by
  obtain ⟨slot⟩ := ‹Nonempty Slot›
  refine le_trans ?_ (Finset.le_sup' _ (Finset.mem_univ slot))
  split
  · exact mass_nonnegative _
  · exact le_rfl

theorem mass_slot_le (value : Ring) :
    mass {slot | decode slot = some value} ≤ slotChance decode := by
  by_cases reached : ∃ slot, decode slot = some value
  · obtain ⟨slot, decoded⟩ := reached
    refine le_trans (le_of_eq ?_) (Finset.le_sup' _ (Finset.mem_univ slot))
    rw [ite_eq_left (by rw [decoded]; rfl), decoded]
  · have empty : {slot | decode slot = some value} = (∅ : Set Slot) :=
      Set.eq_empty_of_forall_notMem fun slot decoded => reached ⟨slot, decoded⟩
    rw [empty]
    exact (le_of_eq (by simp [mass])).trans (slotChance_nonnegative decode)

end Chance

/-- The claim of one level (Akita Definition 10.22, "The fixed folding statement", without the
matrices): the payloads `u_pub` and `v_pub`, the weights of the opening point and the claimed
value `v`. -/
structure Claim {S : Shape} (O : OpeningShape S) (E : Type) where
  payload : S.Payload
  openPayload : O.Payload
  point : Point O E
  value : E

/-- The folding statement of a claim for the matrices of the level. -/
@[reducible] def Claim.statement {S : Shape} {O : OpeningShape S} {E : Type} (key : Key S)
    (openKey : OpeningKey O) (claim : Claim O E) : Statement O E :=
  ⟨key, openKey, claim.payload, claim.openPayload, claim.point, claim.value⟩

/-- A certificate of a level: a certificate of the later stages, and after the extraction at the
fold challenge a weak opening. -/
structure Opened (Weak Inner : Type) where
  inner : Inner
  opening : Option Weak

/-- A witness of `R_bind` for the matrices of one level, or `none`: a solution of Module-SIS for
`A` with the bound `innerBound`, for `B` or `D` with the bound `b⋆ − 1`, or for a compression map
with the bound 1 (Akita Table 3). -/
abbrev LevelBreak {S : Shape} {O : OpeningShape S} (key : Key S) (openKey : OpeningKey O)
    (innerBound : ℕ) : Type :=
  Option (ShortKernel key.inner innerBound ⊕ (OuterBreak key ⊕ OpenBreak openKey))

/-- One opening method of a fold. `Block` is the index of the fold coordinates (Akita
Definition 10.30 (ii): one coordinate for each opening group and source block), and `Ring b` is
the challenge ring of the coordinate `b`. `Stmt` is the type of the claims, `Wit` of the digit
vectors, `Weak` of the weak openings, and `Brk` of the witnesses of `R_bind`. `Satisfies` is the
exact branch relation of the digit vectors for a claim and the fold challenges (Akita Section 10,
before Corollary 10.32). `Opens` is the weak opening of a claim (Akita Definition 10.22). `merge`
is the reduction of Akita Corollary 10.32 for the two children `(c, ŵ)` of each coordinate and the
digit vectors of the reference child. `broken` is the truth of a witness of `R_bind`. -/
structure Method (Block : Type) (Ring : Block → Type) (Stmt Brk Wit Weak : Type) where
  Satisfies : Wit → Stmt → ((b : Block) → Ring b) → Prop
  Opens : Weak → Stmt → Prop
  merge : Stmt → (Block → ((b : Block) → Ring b) × Wit) →
    (Block → ((b : Block) → Ring b) × Wit) → Wit → Weak ⊕ Brk
  broken : Brk → Prop

/-- Akita Corollary 10.32 for a method and the challenge families of its coordinates. The two
children of the coordinate `b` have challenges of the family at `b` that differ, and their
challenges agree at each other coordinate. The digit vectors of each child, and the digit vectors
`w`, satisfy their exact branch relations for one claim. Then `merge` for `w` returns a weak
opening of the claim or a true witness of `R_bind`. -/
def Method.Sound {Block : Type} {Ring : Block → Type} {Stmt Brk Wit Weak : Type}
    (M : Method Block Ring Stmt Brk Wit Weak) (family : (b : Block) → Set (Ring b)) : Prop :=
  ∀ (claim : Stmt) (first second : Block → ((b : Block) → Ring b) × Wit) (w : Wit)
    (challenge : (b : Block) → Ring b),
    (∀ b, (first b).1 b ∈ family b) → (∀ b, (second b).1 b ∈ family b) →
    (∀ b, (first b).1 b ≠ (second b).1 b) → (∀ b b', b' ≠ b → (first b).1 b' = (second b).1 b') →
    (∀ b, M.Satisfies (first b).2 claim (first b).1) →
    (∀ b, M.Satisfies (second b).2 claim (second b).1) → M.Satisfies w claim challenge →
    Sum.elim (M.Opens · claim) M.broken (M.merge claim first second w)

section Packing

variable {S : Shape} {O : OpeningShape S} {fold : Descriptor} {E : Type} [CommRing E]
  [Module Zq E]

/-- The method of a fold with coefficient packing for one claim in one group (Akita Sections 5.3
and 7.2): `Witness.Satisfies`, the weak opening `Weak.Opens` with `κ̄_1 = 2ω` and
`β̄_∞ = Δ^cert_f` to the table that its source blocks decode to, and `forkCollision` with
`extract`. The fold coordinates are the source blocks, with one challenge ring. `weight` is the
norm `ω = ‖c‖_1` of a fold challenge. A witness of `R_bind` for `A` has the bound
`η_A = 2·κ̄_1·Δ^cert_f` of Akita (50). -/
noncomputable def packing (β : Module.Basis (Fin O.planes) Zq E) (key : Key S)
    (openKey : OpeningKey O) (weight : ℕ) :
    Method (Fin S.blocks) (fun _ => Rq O.carrierDegree) (Claim O E)
      (LevelBreak key openKey (2 * (2 * weight) * fold.sourceBound S)) (Witness O fold)
      (Weak O) where
  Satisfies w claim challenge := w.Satisfies β (claim.statement key openKey) challenge
  Opens ω claim :=
    ω.Opens β (2 * weight) (fold.sourceBound S) (claim.statement key openKey) (S.decode ω.source)
  merge claim first second w :=
    match forkCollision (claim.statement key openKey)
        ⟨fun i => ⟨(first i).1, (first i).2⟩, fun i => ⟨(second i).1, (second i).2⟩⟩
        w.toCarried with
    | some solution => .inr (some (.inr solution))
    | none => .inl (extract
        ⟨fun i => ⟨(first i).1, (first i).2⟩, fun i => ⟨(second i).1, (second i).2⟩⟩ w.toCarried)
  broken solution := solution.isSome

/-- Akita Corollary 10.32 for the method `packing`. The challenge family has units as differences
(Akita Definition 9.2, E1) and members of the norm `ω`, and `b_z ≥ 2`. -/
theorem packing_sound {β : Module.Basis (Fin O.planes) Zq E} {key : Key S}
    {openKey : OpeningKey O} {weight peak : ℕ} {family : Set (Rq O.carrierDegree)}
    (units : UnitDifferences family) (sparse : ∀ challenge ∈ family, Sparse weight peak challenge)
    (base : 2 ≤ fold.responseBase) :
    (packing (fold := fold) β key openKey weight).Sound fun _ => family := by
  intro claim first second w challenge firstMember secondMember different agree firstSatisfies
    secondSatisfies good
  have result := extract_opens_or_collision units sparse base
    (fork := ⟨fun i => ⟨(first i).1, (first i).2⟩, fun i => ⟨(second i).1, (second i).2⟩⟩)
    ⟨firstMember, secondMember, different, agree, firstSatisfies, secondSatisfies⟩
    (Witness.Satisfies.carried good)
  unfold packing
  rcases collision : forkCollision (claim.statement key openKey)
      (⟨fun i => ⟨(first i).1, (first i).2⟩, fun i => ⟨(second i).1, (second i).2⟩⟩ :
        Fork O fold) w.toCarried with _ | solution
  · rw [collision] at result
    simp only [collision]
    exact result.resolve_left (by simp)
  · simp only [collision]
    rfl

end Packing

/-- The data of one fold, with its stages from the stage `start` of a transcript. `method` is the
opening method, and `claim` reads the claim of the level from the message before the stage
`start`. `block b` is the coordinate of the answer of the oracle that has the slot of the fold
challenge of the coordinate `b`, and `decode b` gives the challenge of a slot, or `none` for a
slot that the verifier rejects. `foldBits` has the work bits of the fold challenges
(`FoldChallengeWork` of the Akita code), and `bits` has the work bits of each draw of the
checks (Akita (146)). `read` decodes the messages of the proof. `digits` maps a
successor source to the digit vectors of the fold. `check wire c σ` is the game of the checks of
the verifier for the claim of `wire`, the fold challenges `c` and the source `σ`, `passes σ` is the
truth of its final claim for `σ`, and `costs` has one bound for each of its draws. -/
structure Tier (Block : Type) (Ring : Block → Type)
    (Stmt Brk Wit Weak Wire Coord Slot Result Source : Type) where
  method : Method Block Ring Stmt Brk Wit Weak
  claim : Wire → Stmt
  block : Block → Option Coord
  decode : (b : Block) → Slot → Option (Ring b)
  read : Reader Wire
  digits : Source → Wit
  check : Wire → ((b : Block) → Ring b) → Source → Game Result
  passes : Source → Result → Prop
  costs : List ℝ
  bits : List ℕ
  foldBits : ℕ
  start : ℕ

namespace Tier

variable {Block : Type} {Ring : Block → Type}
  {Stmt Brk Wit Weak Wire Coord Slot Result Source : Type} [∀ b, Zero (Ring b)]
  (T : Tier Block Ring Stmt Brk Wit Weak Wire Coord Slot Result Source) {total : ℕ}

omit [∀ b, Zero (Ring b)] in
/-- The first stage after the stages of the level: the fold challenge, and one stage for each
draw of the checks. -/
def finish : ℕ := T.start + 1 + T.costs.length

omit [∀ b, Zero (Ring b)] in
theorem finish_eq : T.finish = T.start + (T.costs.length + 1) :=
  Nat.add_right_comm _ _ _ |>.trans (Nat.add_assoc _ _ _)

omit [∀ b, Zero (Ring b)] in
/-- The set of the fold challenges of a coordinate: the values of its decoder. -/
def family (b : Block) : Set (Ring b) := {challenge | ∃ slot, T.decode b slot = some challenge}

/-- The fold challenges of a stage input, from the answer of the stage `start`. -/
noncomputable def challenge (point : Prefix Wire (Option Coord → Slot) total) : (b : Block) → Ring
  b :=
  challengeAt T.decode T.block T.start point

omit [∀ b, Zero (Ring b)] in
/-- The verifier accepts the work slots and the fold challenges of the level: the answer of the
stage `start` has zero work bits and a fold challenge for each coordinate, the answer of each
stage of the checks has zero work bits (Akita (146)), and the nonce of the response search in the
message before the stage `start` is below `responseTries` (Akita Section 5.4 and Figure 4, step 2:
the verifier rejects a nonce outside the set; `FoldResponse` of the Akita code). -/
def Decodes [Fintype Slot] (point : Prefix Wire (Option Coord → Slot) total) : Prop :=
  DecodesAt T.decode T.block T.start T.foldBits point ∧
    (∀ index < T.costs.length, ∀ reply, point.reply (T.start + 1 + index) = some reply →
      reply none ∈ workZeros Slot (T.bits.getD index 0)) ∧
    T.read.nonce (point.wire T.start) < responseTries

omit [∀ b, Zero (Ring b)] in
/-- The relation of the extraction: a weak opening of the claim of the level (Akita
Definition 10.22) for the method of the level. -/
def Opens (point : Prefix Wire (Option Coord → Slot) total) (opening : Option Weak) : Prop :=
  ∃ ω, opening = some ω ∧ T.method.Opens ω (T.claim (point.wire T.start))

variable {T} in
theorem Decodes.challenge [Fintype Slot] {point : Prefix Wire (Option Coord → Slot) total}
    {reply : Option Coord → Slot}
    (answered : point.reply T.start = some reply) (decodes : T.Decodes point) (b : Block) :
    T.decode b (reply (T.block b)) = some (T.challenge point b) :=
  DecodesAt.challenge answered decodes.1 b

theorem challenge_take {level : ℕ} (above : T.start < level)
    (point : Prefix Wire (Option Coord → Slot) total) :
    T.challenge (point.take level) = T.challenge point :=
  challengeAt_take T.decode T.block T.start above point

/-- The checkpoint segment of the checks of the level, at the stage after the fold challenges.
The candidate is a successor source, and its test is the exact branch relation of its digit
vectors (Akita Proposition 10.44, "Checkpoints"). -/
noncomputable def segment : Segment Wire Coord Slot Result Source total where
  start := T.start + 1
  costs := T.costs
  bits := T.bits
  read := T.read
  check anchor source := T.check (anchor.wire T.start) (T.challenge anchor) source
  passes _ := T.passes
  test anchor source :=
    T.method.Satisfies (T.digits source) (T.claim (anchor.wire T.start)) (T.challenge anchor)

section Random

variable [Fintype Slot] [Nonempty Slot]

omit [∀ b, Zero (Ring b)] in
/-- The largest probability of one fold challenge of a coordinate for a uniform slot. -/
noncomputable def chance (b : Block) : ℝ := slotChance (T.decode b)

variable [Fintype Coord] [DecidableEq Coord]

/-- The game of the checks for the source `source`, after the answers and the messages of a
stage input through the last stage of the level. -/
noncomputable def game (source : Source) (point : Prefix Wire (Option Coord → Slot) total) :
    Game Result :=
  position T.read (T.check (point.wire T.start) (T.challenge point) source)
    (point.wire (T.start + 1))
    (((point.rest.take T.finish).drop (T.start + 1)).map fun move => (view move.1, move.2))

omit [Nonempty Slot] in
theorem segment_game (source : Source) (point : Prefix Wire (Option Coord → Slot) total) :
    T.segment.game source point = T.game source point := by
  show position T.read (T.check ((point.take (T.start + 1)).wire T.start)
    (T.challenge (point.take (T.start + 1))) source) _ _ = _
  rw [Prefix.wire_take (Nat.le_succ _), T.challenge_take (Nat.lt_succ_self _)]
  rfl

/-- The condition of Akita Lemma 10.42 (ii) on a transcript and a source: the transcript has the
stages of the level, the verifier accepts its work slots and fold challenges, and the game of the
checks for the source ends in a final claim that is true for the source. -/
def Passed (source : Source) (point : Prefix Wire (Option Coord → Slot) total) : Prop :=
  T.finish ≤ point.rest.length ∧ T.Decodes point ∧
    ∃ result, T.game source point = .pure result ∧ T.passes source result

/-- The conditions on the data of one level. `injective`: the coordinates of the fold challenges
are different. `avoid`: no fold challenge is in the work slot. `merges`: Akita Corollary 10.32 for
the method and the challenge families. `sound`: on a path of the game that avoids the bad sets, a
true final claim gives the exact branch relation of the digit vectors of the source. `staged` and
`nonnegative`: the bounds `costs` of the draws. `draws`: the game has a draw. `aligned`: each draw
has its work bits. -/
structure Valid : Prop where
  injective : Function.Injective T.block
  avoid : ∀ b, T.block b ≠ none
  merges : T.method.Sound T.family
  sound : ∀ wire challenge source, (T.check wire challenge source).Avoids fun result =>
    T.passes source result → T.method.Satisfies (T.digits source) (T.claim wire) challenge
  staged : ∀ wire challenge source, Staged (Coord → Slot) (T.check wire challenge source) T.costs
  nonnegative : ∀ cost ∈ T.costs, 0 ≤ cost
  draws : T.costs ≠ []
  aligned : T.bits.length = T.costs.length

variable {T} in
omit [Nonempty Slot] in
theorem Valid.segment (valid : T.Valid) : (T.segment (total := total)).Valid where
  sound _ _ failed := (valid.sound _ _ _).mono fun _ implies passed => failed (implies passed)
  staged _ _ := valid.staged _ _ _
  nonnegative := valid.nonnegative
  draws := valid.draws
  aligned := valid.aligned

section Fork

variable [Fintype Block] {Inner BreakM CandM KeyM WitM : Type}
  (accept : Prefix Wire (Option Coord → Slot) total → Prop)
  (mid : Structure (Address Wire total) (Prefix Wire (Option Coord → Slot) total) (Option Coord) Slot (ℕ → Wire) Inner
    BreakM CandM
    KeyM WitM)
  (source : Inner → Source)

omit [Fintype Slot] [Nonempty Slot] [Fintype Coord] [DecidableEq Coord] [Fintype Block] in
/-- `Comb_r` at the stage `start`: the reduction `merge` of the method (Akita Corollary 10.32)
for the children of Akita Definition 10.30 (ii). The reference transcript is the first child of
each coordinate, and the transcript of the coordinate `b` is its second child. The digit vectors
of a child are those of its successor source. -/
noncomputable def combine (reference : Opened Weak Inner)
    (children : Option Coord → Opened Weak Inner) :
    Opened Weak Inner ⊕ (Brk ⊕ BreakM) :=
  (T.method.merge (T.claim ((mid.point reference.inner).wire T.start))
    (fun _ => (T.challenge (mid.point reference.inner), T.digits (source reference.inner)))
    (fun b => (T.challenge (mid.point (children (T.block b)).inner),
      T.digits (source (children (T.block b)).inner)))
    (T.digits (source reference.inner))).map
    (fun ω => ⟨reference.inner, some ω⟩) Sum.inl

/-- The indexed stage of the fold challenges at the stage `start`, over the structure `mid` of
the later stages (Akita Definition 10.16 (iv) with Corollary 10.32). The stage has one coordinate
for each fold coordinate. A certificate at the stage `start` is an accepting transcript with a
weak opening of the claim of the level. -/
noncomputable def fork :
    Structure (Address Wire total) (Prefix Wire (Option Coord → Slot) total) (Option Coord) Slot (ℕ → Wire) (Opened Weak
      Inner)
      (Brk ⊕ BreakM) CandM ((Σ b, Ring b) ⊕ KeyM) (Option Weak) where
  protocol := transcript total accept
  kind level :=
    if level ≤ T.start then .indexed (Finset.univ.toList.map T.block) else mid.kind level
  key level coordinate slot :=
    if level ≤ T.start then
      (Function.extend T.block
        (fun b slot => ((T.decode b slot).map (Sigma.mk b) : Option (Σ b, Ring b)))
        (fun _ _ => none) coordinate slot).map Sum.inl
    else (mid.key level coordinate slot).map Sum.inr
  weight level coordinate :=
    if level ≤ T.start then Function.extend T.block T.chance (fun _ => 0) coordinate
    else mid.weight level coordinate
  work level :=
    if level ≤ T.start then {reply | reply none ∈ workZeros Slot T.foldBits} else mid.work level
  chance level := if level ≤ T.start then 1 / 2 ^ T.foldBits else mid.chance level
  error := mid.error
  leaf point := ⟨mid.leaf point, none⟩
  point certificate := mid.point certificate.inner
  holds level certificate :=
    if level ≤ T.start then
      accept (mid.point certificate.inner) ∧
        T.Opens (mid.point certificate.inner) certificate.opening
    else mid.holds level certificate.inner
  breaks := Sum.elim T.method.broken mid.breaks
  combine level reference children :=
    if level ≤ T.start then T.combine mid source reference children
    else Sum.map (fun inner => ⟨inner, none⟩) Sum.inr
      (mid.combine level reference.inner fun coordinate => (children coordinate).inner)
  candidate level certificate := mid.candidate level certificate.inner
  good level certificate := mid.good level certificate.inner
  collide level certificate other := .inr (mid.collide level certificate.inner other.inner)
  refuted := mid.refuted
  bad := mid.bad
  witness := Opened.opening
  relation root opening := T.Opens root opening

/-- The interface of the later stages to the indexed stage of the fold challenges. `fits`: the
stage `start` is a stage of the transcript. `decodes`: the verifier rejects an answer of the stage
`start` without a fold challenge for each coordinate. `lawful`: the later stages satisfy the
conditions of Definition 10.16. `checked`: a first certificate of the later stages has an
accepting transcript, and the digit vectors of its successor source satisfy the exact branch
relation of the transcript. -/
structure Forked : Prop where
  fits : T.start < total
  decodes : ∀ point, accept point → T.Decodes point
  lawful : mid.LawfulFrom (transcript total accept) (T.start + 1)
  checked : ∀ inner, mid.holds (T.start + 1) inner → accept (mid.point inner) ∧
    T.method.Satisfies (T.digits (source inner)) (T.claim ((mid.point inner).wire T.start))
      (T.challenge (mid.point inner))

variable {accept mid source}

omit [Fintype Coord] [DecidableEq Coord] in
theorem fork_kind (level : ℕ) :
    (T.fork accept mid source).kind level =
      if level ≤ T.start then .indexed (Finset.univ.toList.map T.block) else mid.kind level :=
  rfl

omit [Fintype Coord] [DecidableEq Coord] in
theorem fork_holds (level : ℕ) (certificate : Opened Weak Inner) :
    (T.fork accept mid source).holds level certificate =
      if level ≤ T.start then
        accept (mid.point certificate.inner) ∧
          T.Opens (mid.point certificate.inner) certificate.opening
      else mid.holds level certificate.inner :=
  rfl

omit [Fintype Coord] [DecidableEq Coord] in
theorem fork_combine (level : ℕ) (reference : Opened Weak Inner)
    (children : Option Coord → Opened Weak Inner) :
    (T.fork accept mid source).combine level reference children =
      if level ≤ T.start then T.combine mid source reference children
      else Sum.map (fun inner => ⟨inner, none⟩) Sum.inr
        (mid.combine level reference.inner fun coordinate => (children coordinate).inner) :=
  rfl

omit [Fintype Coord] [DecidableEq Coord] in
theorem fork_holds_above {level : ℕ} (above : T.start < level)
    (certificate : Opened Weak Inner) :
    (T.fork accept mid source).holds level certificate ↔ mid.holds level certificate.inner :=
  Iff.of_eq (ite_eq_right (Nat.not_le.mpr above))

omit [Fintype Coord] [DecidableEq Coord] in
theorem fork_key_block (injective : Function.Injective T.block) (b : Block) (slot : Slot) :
    (T.fork accept mid source).key T.start (T.block b) slot =
      ((T.decode b slot).map (Sigma.mk b)).map Sum.inl := by
  show (if T.start ≤ T.start then
    (Function.extend T.block
      (fun b slot => ((T.decode b slot).map (Sigma.mk b) : Option (Σ b, Ring b)))
      (fun _ _ => none) (T.block b) slot).map Sum.inl
    else (mid.key T.start (T.block b) slot).map Sum.inr) = _
  rw [ite_eq_left le_rfl, injective.extend_apply]

omit [Fintype Coord] [DecidableEq Coord] in
theorem fork_key_above {level : ℕ} (above : T.start < level) (coordinate : Option Coord)
    (slot : Slot) :
    (T.fork accept mid source).key level coordinate slot =
      (mid.key level coordinate slot).map Sum.inr :=
  ite_eq_right (Nat.not_le.mpr above)

omit [Fintype Coord] [DecidableEq Coord] in
theorem fork_weight_block (injective : Function.Injective T.block) (b : Block) :
    (T.fork accept mid source).weight T.start (T.block b) = T.chance b := by
  show (if T.start ≤ T.start then Function.extend T.block T.chance (fun _ => 0) (T.block b)
    else mid.weight T.start (T.block b)) = _
  rw [ite_eq_left le_rfl, injective.extend_apply]

omit [Fintype Coord] [DecidableEq Coord] in
theorem fork_weight_above {level : ℕ} (above : T.start < level) (coordinate : Option Coord) :
    (T.fork accept mid source).weight level coordinate = mid.weight level coordinate :=
  ite_eq_right (Nat.not_le.mpr above)

omit [Fintype Coord] [DecidableEq Coord] in
theorem fork_kind_above {level : ℕ} (above : T.start < level) :
    (T.fork accept mid source).kind level = mid.kind level :=
  ite_eq_right (Nat.not_le.mpr above)

omit [Fintype Coord] [DecidableEq Coord] in
theorem fork_kind_start :
    (T.fork accept mid source).kind T.start = .indexed (Finset.univ.toList.map T.block) :=
  ite_eq_left le_rfl

omit [Fintype Coord] [DecidableEq Coord] in
theorem fork_kind_indexed {level : ℕ} {coordinates : List (Option Coord)} (low : T.start ≤ level)
    (isIndexed : (T.fork accept mid source).kind level = .indexed coordinates) :
    (level = T.start ∧ coordinates = Finset.univ.toList.map T.block) ∨
      (T.start < level ∧ mid.kind level = .indexed coordinates) := by
  rcases Nat.eq_or_lt_of_le low with rfl | above
  · rw [T.fork_kind_start] at isIndexed
    exact Or.inl ⟨rfl, (Kind.indexed.inj isIndexed).symm⟩
  · rw [T.fork_kind_above above] at isIndexed
    exact Or.inr ⟨above, isIndexed⟩

omit [Fintype Coord] [DecidableEq Coord] in
theorem fork_kind_checkpoint {level span : ℕ}
    (isCheckpoint : (T.fork accept mid source).kind level = .checkpoint span) :
    T.start < level ∧ mid.kind level = .checkpoint span := by
  rw [fork_kind] at isCheckpoint
  by_cases below : level ≤ T.start
  · rw [ite_eq_left below] at isCheckpoint
    exact absurd isCheckpoint (by simp)
  · rw [ite_eq_right below] at isCheckpoint
    exact ⟨Nat.lt_of_not_le below, isCheckpoint⟩

omit [Fintype Coord] [DecidableEq Coord] in
theorem fork_kind_plain {level : ℕ}
    (isPlain : (T.fork accept mid source).kind level = .plain) :
    T.start < level ∧ mid.kind level = .plain := by
  rw [fork_kind] at isPlain
  by_cases below : level ≤ T.start
  · rw [ite_eq_left below] at isPlain
    exact absurd isPlain (by simp)
  · rw [ite_eq_right below] at isPlain
    exact ⟨Nat.lt_of_not_le below, isPlain⟩

omit [Fintype Block] in
/-- `Comb_r` at the stage `start` is correct (Definition 10.16 (iv) with Akita
Corollary 10.32). -/
theorem combine_opens (injective : Function.Injective T.block)
    (merges : T.method.Sound T.family) (fed : T.Forked accept mid source)
    (reference : Opened Weak Inner) (family : Option Coord → Opened Weak Inner)
    (anchor deeper : Prefix Wire (Option Coord → Slot) total)
    (held : mid.holds (T.start + 1) reference.inner)
    (found : (transcript total accept).cut T.start (mid.point reference.inner) = some anchor)
    (foundDeeper :
      (transcript total accept).cut (T.start + 1) (mid.point reference.inner) = some deeper)
    (each : ∀ b, mid.holds (T.start + 1) (family (T.block b)).inner ∧
      (transcript total accept).cut T.start (mid.point (family (T.block b)).inner) =
        some anchor ∧
      ∃ other slot,
        (transcript total accept).cut (T.start + 1) (mid.point (family (T.block b)).inner) =
          some other ∧
        (transcript total accept).last other =
          Function.update ((transcript total accept).last deeper) (T.block b) slot ∧
        T.decode b slot ≠ T.decode b ((transcript total accept).last deeper (T.block b))) :
    Sum.elim
      (fun certificate : Opened Weak Inner =>
        (accept (mid.point certificate.inner) ∧
          T.Opens (mid.point certificate.inner) certificate.opening) ∧
        ∀ earlier ≤ T.start,
          (transcript total accept).cut earlier (mid.point certificate.inner) =
            (transcript total accept).cut earlier (mid.point reference.inner))
      (Sum.elim T.method.broken mid.breaks)
      (T.combine mid source reference family) := by
  obtain ⟨accepted, good⟩ := fed.checked reference.inner held
  have decodes := fed.decodes _ accepted
  obtain ⟨_, rfl⟩ := transcript_cut found
  obtain ⟨_, rfl⟩ := transcript_cut foundDeeper
  obtain ⟨reply, answered, _⟩ := id decodes.1
  have base := last_take_succ (accept := accept) answered
  have child : ∀ b : Block,
      (T.Decodes (mid.point (family (T.block b)).inner) ∧
        T.method.Satisfies (T.digits (source (family (T.block b)).inner))
          (T.claim ((mid.point reference.inner).wire T.start))
          (T.challenge (mid.point (family (T.block b)).inner))) ∧
      ∃ slot, (mid.point (family (T.block b)).inner).reply T.start =
          some (Function.update reply (T.block b) slot) ∧
        T.decode b slot ≠ T.decode b (reply (T.block b)) := by
    intro b
    obtain ⟨heldChild, foundChild, other, slot, foundOther, last, differs⟩ := each b
    obtain ⟨acceptedChild, goodChild⟩ := fed.checked _ heldChild
    obtain ⟨_, sameAnchor⟩ := transcript_cut foundChild
    have sameWire : (mid.point (family (T.block b)).inner).wire T.start =
        (mid.point reference.inner).wire T.start := by
      rw [← Prefix.wire_take le_rfl (mid.point (family (T.block b)).inner), ← sameAnchor,
        Prefix.wire_take le_rfl]
    rw [sameWire] at goodChild
    refine ⟨⟨fed.decodes _ acceptedChild, goodChild⟩, slot, ?_, ?_⟩
    · obtain ⟨_, rfl⟩ := transcript_cut foundOther
      obtain ⟨replyChild, answeredChild, _⟩ := (fed.decodes _ acceptedChild).1
      rw [last_take_succ answeredChild, base] at last
      rw [answeredChild, last]
    · rw [base] at differs
      exact differs
  have result := merges (T.claim ((mid.point reference.inner).wire T.start))
    (fun _ => (T.challenge (mid.point reference.inner), T.digits (source reference.inner)))
    (fun b => (T.challenge (mid.point (family (T.block b)).inner),
      T.digits (source (family (T.block b)).inner)))
    (T.digits (source reference.inner)) (T.challenge (mid.point reference.inner))
    (fun b => ⟨_, decodes.challenge answered b⟩)
    (fun b => by
      obtain ⟨⟨decodesChild, _⟩, slot, answeredChild, _⟩ := child b
      exact ⟨_, decodesChild.challenge answeredChild b⟩)
    (fun b same => by
      obtain ⟨⟨decodesChild, _⟩, slot, answeredChild, differs⟩ := child b
      have second := decodesChild.challenge answeredChild b
      rw [Function.update_self] at second
      refine differs (second.trans ?_)
      rw [decodes.challenge answered b]
      exact congrArg some same.symm)
    (fun b b' other => by
      obtain ⟨⟨decodesChild, _⟩, slot, answeredChild, _⟩ := child b
      have second := decodesChild.challenge answeredChild b'
      rw [Function.update_of_ne (fun same => other (injective same)),
        decodes.challenge answered b'] at second
      exact Option.some.inj second)
    (fun _ => good) (fun b => (child b).1.2) good
  unfold Tier.combine
  generalize T.method.merge _ _ _ _ = merged at result ⊢
  rcases merged with ω | solution
  · exact ⟨⟨accepted, ω, rfl, result⟩, fun _ _ => rfl⟩
  · exact result

/-- The indexed stage of the fold challenges over the later stages satisfies the conditions of
Akita Definition 10.16 from the stage `start`. -/
theorem fork_lawfulFrom (injective : Function.Injective T.block) (avoid : ∀ b, T.block b ≠ none)
    (merges : T.method.Sound T.family) (fed : T.Forked accept mid source) :
    (T.fork accept mid source).LawfulFrom (transcript total accept) T.start := by
  have fits := fed.fits
  refine {
    segment := ?_
    key_some := ?_
    key_mass := ?_
    weight_nonnegative := ?_
    work_accept := ?_
    work_keep := ?_
    work_mass := ?_
    leaf_point := fun point => fed.lawful.leaf_point point
    leaf_holds := fun point accepted =>
      (T.fork_holds_above fits _).mpr (fed.lawful.leaf_holds point accepted)
    plain_holds := ?_
    combine_holds := ?_
    good_holds := ?_
    collide_breaks := ?_
    refuted_start := ?_
    refuted_step := ?_
    refuted_end := ?_
    bad_mass := ?_
    error_nonnegative := fed.lawful.error_nonnegative }
  · intro level span low isCheckpoint
    obtain ⟨above, isMid⟩ := T.fork_kind_checkpoint isCheckpoint
    obtain ⟨short, plains⟩ := fed.lawful.segment level span above isMid
    refine ⟨short, fun later after below => ?_⟩
    rw [T.fork_kind_above (by omega)]
    exact plains later after below
  · intro level coordinates coordinate point deeper low isIndexed member accepted found
    rcases T.fork_kind_indexed low isIndexed with ⟨rfl, rfl⟩ | ⟨above, isMid⟩
    · obtain ⟨b, _, rfl⟩ := List.mem_map.mp member
      obtain ⟨_, rfl⟩ := transcript_cut found
      obtain ⟨⟨reply, answered, _, each⟩, _⟩ := fed.decodes point accepted
      rw [T.fork_key_block injective, last_take_succ answered, Option.isSome_map,
        Option.isSome_map]
      exact each b
    · rw [T.fork_key_above above, Option.isSome_map]
      exact fed.lawful.key_some level coordinates coordinate point deeper above isMid member
        accepted found
  · intro level coordinates coordinate value low isIndexed member
    rcases T.fork_kind_indexed low isIndexed with ⟨rfl, rfl⟩ | ⟨above, isMid⟩
    · obtain ⟨b, _, rfl⟩ := List.mem_map.mp member
      simp only [T.fork_key_block injective, T.fork_weight_block injective]
      refine mass_inl_le _ value (slotChance_nonnegative (T.decode b)) fun first => ?_
      rcases first with ⟨other, challenge⟩
      by_cases same : other = b
      · subst same
        refine (mass_mono fun slot decoded => ?_).trans (mass_slot_le (T.decode other) challenge)
        obtain ⟨found, decodes, equal⟩ := Option.map_eq_some_iff.mp decoded
        exact decodes.trans (congrArg some (sigma_mk_injective equal))
      · refine (mass_mono (large := ∅) fun slot decoded => ?_).trans
          ((le_of_eq (by simp [mass])).trans (slotChance_nonnegative (T.decode b)))
        obtain ⟨found, _, equal⟩ := Option.map_eq_some_iff.mp decoded
        exact absurd (congrArg Sigma.fst equal).symm same
    · simp only [T.fork_key_above above, T.fork_weight_above above]
      exact mass_inr_le _ value (fed.lawful.weight_nonnegative level coordinate) fun second =>
        fed.lawful.key_mass level coordinates coordinate second above isMid member
  · intro level coordinate
    by_cases below : level ≤ T.start
    · show 0 ≤ (if level ≤ T.start then Function.extend T.block T.chance (fun _ => 0) coordinate
        else mid.weight level coordinate)
      rw [ite_eq_left below]
      by_cases reached : ∃ b, T.block b = coordinate
      · obtain ⟨b, rfl⟩ := reached
        rw [injective.extend_apply]
        exact slotChance_nonnegative (T.decode b)
      · rw [Function.extend_apply' _ _ _ reached]
    · rw [T.fork_weight_above (Nat.lt_of_not_le below)]
      exact fed.lawful.weight_nonnegative level coordinate
  · intro level coordinates point deeper low isIndexed accepted found
    rcases T.fork_kind_indexed low isIndexed with ⟨rfl, rfl⟩ | ⟨above, isMid⟩
    · show _ ∈ (if T.start ≤ T.start then {reply : Option Coord → Slot |
        reply none ∈ workZeros Slot T.foldBits} else mid.work T.start)
      rw [ite_eq_left le_rfl]
      obtain ⟨_, rfl⟩ := transcript_cut found
      obtain ⟨⟨reply, answered, zero, _⟩, _⟩ := fed.decodes point accepted
      show (transcript total accept).last (point.take (T.start + 1)) none ∈ workZeros Slot
        T.foldBits
      rw [last_take_succ answered]
      exact zero
    · show _ ∈ (if level ≤ T.start then {reply : Option Coord → Slot |
        reply none ∈ workZeros Slot T.foldBits} else mid.work level)
      rw [ite_eq_right (Nat.not_le.mpr above)]
      exact fed.lawful.work_accept level coordinates point deeper above isMid accepted found
  · intro level coordinates coordinate reply slot low isIndexed member
    rcases T.fork_kind_indexed low isIndexed with ⟨rfl, rfl⟩ | ⟨above, isMid⟩
    · show _ ∈ (if T.start ≤ T.start then {reply : Option Coord → Slot |
          reply none ∈ workZeros Slot T.foldBits} else mid.work T.start) ↔
        _ ∈ (if T.start ≤ T.start then {reply : Option Coord → Slot |
          reply none ∈ workZeros Slot T.foldBits} else mid.work T.start)
      rw [ite_eq_left le_rfl]
      obtain ⟨b, _, rfl⟩ := List.mem_map.mp member
      show Function.update reply (T.block b) slot none ∈ workZeros Slot T.foldBits ↔
        reply none ∈ workZeros Slot T.foldBits
      rw [Function.update_of_ne (avoid b).symm]
    · show _ ∈ (if level ≤ T.start then {reply : Option Coord → Slot |
          reply none ∈ workZeros Slot T.foldBits} else mid.work level) ↔
        _ ∈ (if level ≤ T.start then {reply : Option Coord → Slot |
          reply none ∈ workZeros Slot T.foldBits} else mid.work level)
      rw [ite_eq_right (Nat.not_le.mpr above)]
      exact fed.lawful.work_keep level coordinates coordinate reply slot above isMid member
  · intro level coordinates low isIndexed
    show mass (if level ≤ T.start then {reply : Option Coord → Slot |
        reply none ∈ workZeros Slot T.foldBits} else mid.work level) ≤
      (if level ≤ T.start then 1 / 2 ^ T.foldBits else mid.chance level)
    rcases T.fork_kind_indexed low isIndexed with ⟨rfl, rfl⟩ | ⟨above, isMid⟩
    · rw [ite_eq_left le_rfl, ite_eq_left le_rfl]
      exact mass_work_le T.foldBits
    · rw [ite_eq_right (Nat.not_le.mpr above), ite_eq_right (Nat.not_le.mpr above)]
      exact fed.lawful.work_mass level coordinates above isMid
  · intro level certificate low isPlain held
    obtain ⟨above, isMid⟩ := T.fork_kind_plain isPlain
    exact (T.fork_holds_above above certificate).mpr (fed.lawful.plain_holds level
      certificate.inner above isMid ((T.fork_holds_above (by omega) certificate).mp held))
  · intro level coordinates reference family anchor deeper low isIndexed held found foundDeeper
      each
    rcases T.fork_kind_indexed low isIndexed with ⟨rfl, rfl⟩ | ⟨above, isMid⟩
    · have combined := T.combine_opens injective merges fed reference family anchor deeper
        ((T.fork_holds_above (Nat.lt_succ_self _) reference).mp held) found foundDeeper fun b => by
          obtain ⟨heldChild, foundChild, other, slot, foundOther, last, differs⟩ :=
            each (T.block b) (List.mem_map.mpr ⟨b, Finset.mem_toList.mpr (Finset.mem_univ b), rfl⟩)
          refine ⟨(T.fork_holds_above (Nat.lt_succ_self _) _).mp heldChild, foundChild, other,
            slot, foundOther, last, fun same => differs ?_⟩
          rw [T.fork_key_block injective, T.fork_key_block injective, same]
      rw [T.fork_combine, ite_eq_left le_rfl]
      generalize T.combine mid source reference family = result at combined ⊢
      rcases result with certificate | broken
      · exact ⟨(Iff.of_eq (ite_eq_left le_rfl)).mpr combined.1, combined.2⟩
      · exact combined
    · have combined := fed.lawful.combine_holds level coordinates reference.inner
        (fun coordinate => (family coordinate).inner) anchor deeper above isMid
        ((T.fork_holds_above (by omega) reference).mp held) found foundDeeper
        fun coordinate member => by
          obtain ⟨heldChild, foundChild, other, slot, foundOther, last, differs⟩ :=
            each coordinate member
          refine ⟨(T.fork_holds_above (by omega) (family coordinate)).mp heldChild, foundChild,
            other, slot, foundOther, last, fun same => differs ?_⟩
          rw [T.fork_key_above above, T.fork_key_above above, same]
      rw [T.fork_combine, ite_eq_right (Nat.not_le.mpr above)]
      generalize mid.combine level reference.inner
        (fun coordinate => (family coordinate).inner) = result at combined ⊢
      rcases result with inner | broken
      · exact ⟨(T.fork_holds_above above _).mpr combined.1, combined.2⟩
      · exact combined
  · intro level span certificate low isCheckpoint held isGood
    obtain ⟨above, isMid⟩ := T.fork_kind_checkpoint isCheckpoint
    exact (T.fork_holds_above above certificate).mpr (fed.lawful.good_holds level span
      certificate.inner above isMid ((T.fork_holds_above (by omega) certificate).mp held) isGood)
  · intro level span certificate other anchor low isCheckpoint held heldOther found foundOther
      different
    obtain ⟨above, isMid⟩ := T.fork_kind_checkpoint isCheckpoint
    exact fed.lawful.collide_breaks level span certificate.inner other.inner anchor above isMid
      ((T.fork_holds_above (by omega) certificate).mp held)
      ((T.fork_holds_above (by omega) other).mp heldOther) found foundOther different
  · intro level span certificate anchor low isCheckpoint held notGood found
    obtain ⟨above, isMid⟩ := T.fork_kind_checkpoint isCheckpoint
    exact fed.lawful.refuted_start level span certificate.inner anchor above isMid
      ((T.fork_holds_above (by omega) certificate).mp held) notGood found
  · intro level span input candidate reply proof low isCheckpoint after before refuted outside
    obtain ⟨above, isMid⟩ := T.fork_kind_checkpoint isCheckpoint
    exact fed.lawful.refuted_step level span input candidate reply proof above isMid after before
      refuted outside
  · intro level span certificate input low isCheckpoint held found refuted
    obtain ⟨above, isMid⟩ := T.fork_kind_checkpoint isCheckpoint
    exact fed.lawful.refuted_end level span certificate.inner input above isMid
      ((T.fork_holds_above (by omega) certificate).mp held) found refuted
  · intro level span input candidate low isCheckpoint after before
    obtain ⟨above, isMid⟩ := T.fork_kind_checkpoint isCheckpoint
    exact fed.lawful.bad_mass level span input candidate above isMid after before

end Fork

variable [Fintype Block] {Inner BreakT CandT KeyT WitT : Type}
  (accept : Prefix Wire (Option Coord → Slot) total → Prop)
  (tail : Structure (Address Wire total) (Prefix Wire (Option Coord → Slot) total) (Option Coord) Slot (ℕ → Wire) Inner
    BreakT CandT
    KeyT WitT)
  (source : Inner → Source) (clash : Inner → Inner → BreakT)

/-- The checkpoint structure of the level over the structure `tail` of the later stages (Akita
Definition 10.16 for the stages from `start`): the indexed stage `fork` of the fold challenges
over the checkpoint segment `segment` of the checks. The candidate of the checkpoint is the
successor source `source` of a certificate of the later stages, and `clash` is `Coll_ℓ`. -/
noncomputable def over :
    Structure (Address Wire total) (Prefix Wire (Option Coord → Slot) total) (Option Coord) Slot (ℕ → Wire) (Opened Weak
      Inner)
      (Brk ⊕ BreakT) (Source ⊕ CandT) ((Σ b, Ring b) ⊕ KeyT) (Option Weak) :=
  T.fork accept (T.segment.over accept tail source clash fun _ _ => True) source

/-- The interface of the later stages to the level. `fits`: the stages of the level are stages of
the transcript. `decodes`: the verifier rejects an answer of the stage `start` without a fold
challenge for each coordinate. `lawful`: the later stages satisfy the conditions of
Definition 10.16. `accepted` and `passed`: the transcript of a first certificate of the later
stages is accepting, and the final claim of the checks of the level is true for its successor
source (Akita Lemma 10.42 (ii)). `clash`: two first certificates of the later stages with one
stage input of the checkpoint and different sources give a witness of `R_bind`
(Definition 10.16 (vi)). -/
structure Fed : Prop where
  fits : T.finish ≤ total
  decodes : ∀ point, accept point → T.Decodes point
  lawful : tail.LawfulFrom (transcript total accept) T.finish
  accepted : ∀ inner, tail.holds T.finish inner → accept (tail.point inner)
  passed : ∀ inner, tail.holds T.finish inner → T.Passed (source inner) (tail.point inner)
  clash : ∀ inner other anchor, tail.holds T.finish inner → tail.holds T.finish other →
    (transcript total accept).cut (T.start + 1) (tail.point inner) = some anchor →
    (transcript total accept).cut (T.start + 1) (tail.point other) = some anchor →
    source inner ≠ source other → tail.breaks (clash inner other)

variable {accept tail source clash}

variable {T} in
omit [Fintype Block] in
theorem Fed.segment (fed : T.Fed accept tail source clash) :
    T.segment.Fed accept tail source clash where
  fits := fed.fits
  lawful := fed.lawful
  passed inner held := by
    obtain ⟨_, decodes, result, ended, passed⟩ := fed.passed inner held
    exact ⟨⟨result, (T.segment_game _ _).trans ended, passed⟩, decodes.2.1⟩
  clash := fed.clash

variable {T} in
omit [Fintype Block] in
theorem Fed.forked (valid : T.Valid) (fed : T.Fed accept tail source clash) :
    T.Forked accept (T.segment.over accept tail source clash fun _ _ => True) source where
  fits := by
    have ends := T.finish_eq
    have fits := fed.fits
    omega
  decodes := fed.decodes
  lawful := T.segment.over_lawfulFrom valid.segment fed.segment
  checked inner held := by
    have ends := T.finish_eq
    have long : 0 < T.costs.length := List.length_pos_iff.mpr valid.draws
    obtain ⟨heldTail, good⟩ := (T.segment.holds_inside (show T.start + 1 < T.finish by omega)
      inner).mp held
    have good : T.method.Satisfies (T.digits (source inner))
        (T.claim (((tail.point inner).take (T.start + 1)).wire T.start))
        (T.challenge ((tail.point inner).take (T.start + 1))) := good le_rfl
    rw [Prefix.wire_take (Nat.le_succ _), T.challenge_take (Nat.lt_succ_self _)] at good
    exact ⟨fed.accepted inner heldTail, good⟩

/-- The level over the later stages satisfies the conditions of Akita Definition 10.16 from its
first stage (Akita Proposition 10.44 for the stages from one level). -/
theorem over_lawfulFrom (valid : T.Valid) (fed : T.Fed accept tail source clash) :
    (T.over accept tail source clash).LawfulFrom (transcript total accept) T.start :=
  T.fork_lawfulFrom valid.injective valid.avoid valid.merges (fed.forked valid)

/-- Akita Proposition 10.44 (i) for the stages from one level: a first certificate of the level
has an accepting transcript and a weak opening of the claim of the level. -/
theorem over_opened (certificate : Opened Weak Inner)
    (held : (T.over accept tail source clash).holds T.start certificate) :
    accept ((T.over accept tail source clash).point certificate) ∧
      T.Opens ((T.over accept tail source clash).point certificate)
        ((T.over accept tail source clash).witness certificate) :=
  (Iff.of_eq (ite_eq_left le_rfl)).mp held

end Random

end Tier

end NightstreamFPrime.Spec.Compression.Akita.Extract
