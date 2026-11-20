import NightstreamFPrime.Spec.Compression.Akita.Fold
import NightstreamFPrime.Spec.Compression.Akita.Extract.Staged
import NightstreamFPrime.Spec.Compression.Akita.Extract.Suffix
import NightstreamFPrime.Spec.Compression.Akita.Extract.Transcript

/-! One level of the opening suffix of Akita as a part of a checkpoint structure (ePrint 2026/1983,
revision of 2026-10-15, Proposition 10.44 with its proof, and Lemma 10.42).

- `Method` has the exact branch relation, the weak opening and the reduction of Akita
  Corollary 10.32 of one opening method. `packing` is the method of a fold with coefficient
  packing, and `packing_sound` is `extract_opens_or_collision` for it.
- `Tier` has the data of one fold whose stages start at the stage `start` of a longer
  transcript: the stage `start` has the fold challenges, and the next stages have the draws of the
  checks. The message before the stage `start` has the claim of the level. `Source` is the type of
  the successor source `σ`: the data that the next level, or the proof, gives for the flat witness
  of this fold. `digits` is the decoder of the schedule from `σ` to the digit vectors.
- `Tier.over` is the checkpoint structure of the level over the structure `tail` of the later
  stages. The stage `start` is an indexed stage with `Comb_r` of Akita Corollary 10.32, the next
  stage is the checkpoint of the checks with the candidate `source`, and `clash` is `Coll_ℓ`.
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

section Prefix

variable {Wire Reply : Type} {stages : ℕ}

/-- The stage input with one more answer and one more message. -/
def Prefix.snoc (point : Prefix Wire Reply stages) (move : Reply × Wire)
    (room : point.rest.length < stages) : Prefix Wire Reply stages :=
  ⟨point.first, point.rest ++ [move], by
    rw [List.length_append, List.length_singleton]
    exact room⟩

theorem Prefix.reply_snoc (point : Prefix Wire Reply stages) (move : Reply × Wire)
    (room : point.rest.length < stages) {index : ℕ} (below : index < point.rest.length) :
    (point.snoc move room).reply index = point.reply index := by
  simp only [Prefix.snoc, Prefix.reply, List.getElem?_append_left below]

theorem Prefix.wire_snoc (point : Prefix Wire Reply stages) (move : Reply × Wire)
    (room : point.rest.length < stages) {index : ℕ} (below : index ≤ point.rest.length) :
    (point.snoc move room).wire index = point.wire index := by
  rcases index with _ | index
  · rfl
  · simp only [Prefix.snoc, Prefix.wire, List.getElem?_append_left (Nat.lt_of_succ_le below)]

variable [Nonempty Reply] {accept : Prefix Wire Reply stages → Prop}

theorem transcript_step (proof : ℕ → Wire) (point : Prefix Wire Reply stages) (reply : Reply)
    (room : point.rest.length < stages) :
    (transcript stages accept).step proof point reply =
      point.snoc (reply, proof (point.rest.length + 1)) room :=
  dite_eq_left room

/-- The last answer of the stage input after the stage `index`. -/
theorem last_take_succ {point : Prefix Wire Reply stages} {index : ℕ} {reply : Reply}
    (answered : point.reply index = some reply) :
    (transcript stages accept).last (point.take (index + 1)) = reply := by
  obtain ⟨move, found, rfl⟩ : ∃ move, point.rest[index]? = some move ∧ move.1 = reply :=
    Option.map_eq_some_iff.mp answered
  show (((point.rest.take (index + 1)).getLast?.map Prod.fst).getD _) = move.1
  rw [List.take_add_one, found]
  simp

end Prefix

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

variable {Wire Coord Slot Ring : Type} {blocks total : ℕ} (decode : Slot → Option Ring)
  (block : Fin blocks → Coord) (start : ℕ)

/-- The fold challenges `c_i` of a stage input, from the answer of the stage `start`. `decode`
gives the challenge of a slot, and `block i` is the coordinate of the slot of `c_i`. -/
noncomputable def challengeAt [Zero Ring] (point : Prefix Wire (Coord → Slot) total)
    (i : Fin blocks) : Ring :=
  ((point.reply start).bind fun reply => decode (reply (block i))).getD 0

/-- The answer of the stage `start` gives a fold challenge for each block. -/
def DecodesAt (point : Prefix Wire (Coord → Slot) total) : Prop :=
  ∃ reply, point.reply start = some reply ∧ ∀ i, (decode (reply (block i))).isSome

variable {decode block start} in
theorem DecodesAt.challenge [Zero Ring] {point : Prefix Wire (Coord → Slot) total}
    {reply : Coord → Slot} (answered : point.reply start = some reply)
    (decodes : DecodesAt decode block start point) (i : Fin blocks) :
    decode (reply (block i)) = some (challengeAt decode block start point i) := by
  obtain ⟨other, same, each⟩ := decodes
  obtain rfl := Option.some.inj (same.symm.trans answered)
  obtain ⟨value, decoded⟩ := Option.isSome_iff_exists.mp (each i)
  simp [challengeAt, answered, decoded]

theorem challengeAt_take [Zero Ring] {level : ℕ} (above : start < level)
    (point : Prefix Wire (Coord → Slot) total) :
    challengeAt decode block start (point.take level) = challengeAt decode block start point := by
  funext i
  simp only [challengeAt, Prefix.reply_take above]

variable [Fintype Slot] [Nonempty Slot]

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

end Challenge

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
structure Opened {S : Shape} (O : OpeningShape S) (Inner : Type) where
  inner : Inner
  opening : Option (Weak O)

/-- A witness of `R_bind` for the matrices of one level, or `none`: a solution of Module-SIS for
`A` with the bound `innerBound`, for `B` or `D` with the bound `b⋆ − 1`, or for a compression map
with the bound 1 (Akita Table 3). -/
abbrev LevelBreak {S : Shape} {O : OpeningShape S} (key : Key S) (openKey : OpeningKey O)
    (innerBound : ℕ) : Type :=
  Option (ShortKernel key.inner innerBound ⊕ (OuterBreak key ⊕ OpenBreak openKey))

/-- One opening method of a fold for the challenge ring `Ring`, the claims `Stmt` and the
witnesses `Brk` of `R_bind`. `Satisfies` is the exact branch relation of the digit vectors for a
claim and the fold challenges (Akita Section 10, before Corollary 10.32). `Opens` is the weak
opening of a claim (Akita Definition 10.22). `merge` is the reduction of Akita Corollary 10.32 for
the two children `(c, ŵ)` of each block and the carried digits of the reference child. `broken` is
the truth of a witness of `R_bind`. -/
structure Method {S : Shape} (O : OpeningShape S) (fold : Descriptor) (Ring Stmt Brk : Type) where
  Satisfies : Witness O fold → Stmt → (Fin S.blocks → Ring) → Prop
  Opens : Weak O → Stmt → Prop
  merge : Stmt → (Fin S.blocks → (Fin S.blocks → Ring) × Witness O fold) →
    (Fin S.blocks → (Fin S.blocks → Ring) × Witness O fold) → Carried O → Weak O ⊕ Brk
  broken : Brk → Prop

/-- Akita Corollary 10.32 for a method and a challenge family. The two children of block `i` have
challenges of the family at `i` that differ, and their challenges agree at each other block. The
digit vectors of each child, and the digit vectors `w`, satisfy their exact branch relations for
one claim. Then `merge` for the carried digits of `w` returns a weak opening of the claim or a
true witness of `R_bind`. -/
def Method.Sound {S : Shape} {O : OpeningShape S} {fold : Descriptor} {Ring Stmt Brk : Type}
    (M : Method O fold Ring Stmt Brk) (family : Set Ring) : Prop :=
  ∀ (claim : Stmt) (first second : Fin S.blocks → (Fin S.blocks → Ring) × Witness O fold)
    (w : Witness O fold) (challenge : Fin S.blocks → Ring),
    (∀ i, (first i).1 i ∈ family) → (∀ i, (second i).1 i ∈ family) →
    (∀ i, (first i).1 i ≠ (second i).1 i) → (∀ i j, j ≠ i → (first i).1 j = (second i).1 j) →
    (∀ i, M.Satisfies (first i).2 claim (first i).1) →
    (∀ i, M.Satisfies (second i).2 claim (second i).1) → M.Satisfies w claim challenge →
    Sum.elim (M.Opens · claim) M.broken (M.merge claim first second w.toCarried)

section Packing

variable {S : Shape} {O : OpeningShape S} {fold : Descriptor} {E : Type} [CommRing E]
  [Module Zq E]

/-- The method of a fold with coefficient packing for one claim (Akita Sections 5.3 and 7.2):
`Witness.Satisfies`, the weak opening `Weak.Opens` with `κ̄_1 = 2ω` and `β̄_∞ = Δ^cert_f` to the
table that its source blocks decode to, and `forkCollision` with `extract`. `weight` is the norm
`ω = ‖c‖_1` of a fold challenge. A witness of `R_bind` for `A` has the bound
`η_A = 2·κ̄_1·Δ^cert_f` of Akita (50). -/
noncomputable def packing (β : Module.Basis (Fin O.planes) Zq E) (key : Key S)
    (openKey : OpeningKey O) (weight : ℕ) :
    Method O fold (Rq O.carrierDegree) (Claim O E)
      (LevelBreak key openKey (2 * (2 * weight) * fold.sourceBound S)) where
  Satisfies w claim challenge := w.Satisfies β (claim.statement key openKey) challenge
  Opens ω claim :=
    ω.Opens β (2 * weight) (fold.sourceBound S) (claim.statement key openKey) (S.decode ω.source)
  merge claim first second digits :=
    match forkCollision (claim.statement key openKey)
        ⟨fun i => ⟨(first i).1, (first i).2⟩, fun i => ⟨(second i).1, (second i).2⟩⟩ digits with
    | some solution => .inr (some (.inr solution))
    | none => .inl (extract
        ⟨fun i => ⟨(first i).1, (first i).2⟩, fun i => ⟨(second i).1, (second i).2⟩⟩ digits)
  broken solution := solution.isSome

/-- Akita Corollary 10.32 for the method `packing`. The challenge family has units as differences
(Akita Definition 9.2, E1) and members of the norm `ω`, and `b_z ≥ 2`. -/
theorem packing_sound {β : Module.Basis (Fin O.planes) Zq E} {key : Key S}
    {openKey : OpeningKey O} {weight peak : ℕ} {family : Set (Rq O.carrierDegree)}
    (units : UnitDifferences family) (sparse : ∀ challenge ∈ family, Sparse weight peak challenge)
    (base : 2 ≤ fold.responseBase) :
    (packing (fold := fold) β key openKey weight).Sound family := by
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

/-- The data of one fold for one claim, with its stages from the stage `start` of a transcript.
`method` is the opening method, and `claim` reads the claim of the level from the message before
the stage `start`. `block i` is the coordinate of the answer of the oracle that has the slot of
the fold challenge `c_i`, and `decode` gives the challenge of a slot, or `none` for a slot that
the verifier rejects. `read` decodes the messages of the proof. `digits` maps a successor source
to the digit vectors of the fold. `check wire c σ` is the game of the checks of the verifier for
the claim of `wire`, the fold challenges `c` and the source `σ`, `passes σ` is the truth of its
final claim for `σ`, and `costs` has one bound for each of its draws. -/
structure Tier {S : Shape} (O : OpeningShape S) (fold : Descriptor)
    (Ring Stmt Brk Wire Coord Slot Result Source : Type) where
  method : Method O fold Ring Stmt Brk
  claim : Wire → Stmt
  block : Fin S.blocks → Coord
  decode : Slot → Option Ring
  read : Reader Wire
  digits : Source → Witness O fold
  check : Wire → (Fin S.blocks → Ring) → Source → Game Result
  passes : Source → Result → Prop
  costs : List ℝ
  start : ℕ

namespace Tier

variable {S : Shape} {O : OpeningShape S} {fold : Descriptor}
  {Ring Stmt Brk Wire Coord Slot Result Source : Type} [Zero Ring]
  (T : Tier O fold Ring Stmt Brk Wire Coord Slot Result Source) {total : ℕ}

/-- The first stage after the stages of the level: the fold challenge, and one stage for each
draw of the checks. -/
def finish : ℕ := T.start + (T.costs.length + 1)

omit [Zero Ring] in
theorem finish_eq : T.finish = T.start + (T.costs.length + 1) := rfl

omit [Zero Ring] in
/-- The set of the fold challenges: the values of the decoder. -/
def family : Set Ring := {challenge | ∃ slot, T.decode slot = some challenge}

/-- The fold challenges `c_i` of a stage input, from the answer of the stage `start`. -/
noncomputable def challenge (point : Prefix Wire (Coord → Slot) total) : Fin S.blocks → Ring :=
  challengeAt T.decode T.block T.start point

omit [Zero Ring] in
/-- The answer of the stage `start` gives a fold challenge for each block. -/
def Decodes (point : Prefix Wire (Coord → Slot) total) : Prop :=
  DecodesAt T.decode T.block T.start point

omit [Zero Ring] in
/-- The relation of the extraction: a weak opening of the claim of the level (Akita
Definition 10.22) for the method of the level. -/
def Opens (point : Prefix Wire (Coord → Slot) total) (opening : Option (Weak O)) : Prop :=
  ∃ ω, opening = some ω ∧ T.method.Opens ω (T.claim (point.wire T.start))

omit [Zero Ring] in
/-- Definition 10.16 (i) for the stages of the level: the stage `start` is indexed by the
blocks, the next stage is the checkpoint of the checks, and its segment has the other stages. -/
def kind (level : ℕ) : Kind Coord :=
  if level ≤ T.start then .indexed ((List.finRange S.blocks).map T.block)
  else if level = T.start + 1 then .checkpoint (T.costs.length - 1)
  else .plain

omit [Zero Ring] in
theorem costs_getD_nonnegative (nonnegative : ∀ cost ∈ T.costs, 0 ≤ cost) (index : ℕ) :
    0 ≤ T.costs.getD index 0 := by
  rw [List.getD_eq_getElem?_getD]
  rcases entry : T.costs[index]? with _ | cost
  · exact le_rfl
  · exact nonnegative cost (List.mem_of_getElem? entry)

variable {T} in
theorem Decodes.challenge {point : Prefix Wire (Coord → Slot) total} {reply : Coord → Slot}
    (answered : point.reply T.start = some reply) (decodes : T.Decodes point) (i : Fin S.blocks) :
    T.decode (reply (T.block i)) = some (T.challenge point i) :=
  DecodesAt.challenge answered decodes i

theorem challenge_take {level : ℕ} (above : T.start < level)
    (point : Prefix Wire (Coord → Slot) total) :
    T.challenge (point.take level) = T.challenge point :=
  challengeAt_take T.decode T.block T.start above point

theorem challenge_snoc (point : Prefix Wire (Coord → Slot) total) (move : (Coord → Slot) × Wire)
    (room : point.rest.length < total) (below : T.start < point.rest.length) :
    T.challenge (point.snoc move room) = T.challenge point := by
  funext i
  simp only [Tier.challenge, challengeAt, Prefix.reply_snoc point move room below]

section Random

variable [Fintype Slot] [Nonempty Slot]

omit [Zero Ring] in
/-- The largest probability of one fold challenge of the level for a uniform slot. -/
noncomputable def chance : ℝ := slotChance T.decode

variable [Fintype Coord] [DecidableEq Coord]

/-- The game of the checks for the source `source`, after the answers and the messages of a
stage input through the last stage of the level. -/
noncomputable def game (source : Source) (point : Prefix Wire (Coord → Slot) total) :
    Game Result :=
  position T.read (T.check (point.wire T.start) (T.challenge point) source)
    (point.wire (T.start + 1)) ((point.rest.take T.finish).drop (T.start + 1))

/-- The condition of Akita Lemma 10.42 (ii) on a transcript and a source: the transcript has the
stages of the level, each fold challenge decodes, and the game of the checks for the source ends
in a final claim that is true for the source. -/
def Passed (source : Source) (point : Prefix Wire (Coord → Slot) total) : Prop :=
  T.finish ≤ point.rest.length ∧ T.Decodes point ∧
    ∃ result, T.game source point = .pure result ∧ T.passes source result

/-- The conditions on the data of one level. `injective`: the blocks have different coordinates.
`merges`: Akita Corollary 10.32 for the method and the challenge family. `sound`: on a path of
the game that avoids the bad sets, a true final claim gives the exact branch relation of the
digit vectors of the source. `staged` and `nonnegative`: the bounds `costs` of the draws.
`draws`: the game has a draw. -/
structure Valid : Prop where
  injective : Function.Injective T.block
  merges : T.method.Sound T.family
  sound : ∀ wire challenge source, (T.check wire challenge source).Avoids fun result =>
    T.passes source result → T.method.Satisfies (T.digits source) (T.claim wire) challenge
  staged : ∀ wire challenge source, Staged (Coord → Slot) (T.check wire challenge source) T.costs
  nonnegative : ∀ cost ∈ T.costs, 0 ≤ cost
  draws : T.costs ≠ []

variable {Inner BreakT CandT KeyT WitT : Type}
  (accept : Prefix Wire (Coord → Slot) total → Prop)
  (tail : Structure (Prefix Wire (Coord → Slot) total) Coord Slot (ℕ → Wire) Inner BreakT CandT
    KeyT WitT)
  (source : Inner → Source) (clash : Inner → Inner → BreakT)

/-- `Comb_r` at the stage `start`: the reduction `merge` of the method (Akita Corollary 10.32)
for the children of Akita Definition 10.30 (ii). The reference transcript is the first child of
each block, and the transcript of the coordinate of block `i` is its second child. The digit
vectors of a child are those of its successor source. -/
noncomputable def combine (reference : Opened O Inner) (children : Coord → Opened O Inner) :
    Opened O Inner ⊕ (Brk ⊕ BreakT) :=
  (T.method.merge (T.claim ((tail.point reference.inner).wire T.start))
    (fun _ => (T.challenge (tail.point reference.inner), T.digits (source reference.inner)))
    (fun i => (T.challenge (tail.point (children (T.block i)).inner),
      T.digits (source (children (T.block i)).inner)))
    (T.digits (source reference.inner)).toCarried).map
    (fun ω => ⟨reference.inner, some ω⟩) Sum.inl

/-- `Good_ℓ` of the checkpoint: the digit vectors of the successor source satisfy the exact
branch relation of the transcript (Akita Proposition 10.44, "Checkpoints"). -/
def Good (certificate : Opened O Inner) : Prop :=
  T.method.Satisfies (T.digits (source certificate.inner))
    (T.claim ((tail.point certificate.inner).wire T.start))
    (T.challenge (tail.point certificate.inner))

/-- The certificate sets `Y_level`: at the stage `start` an accepting transcript with a weak
opening of the claim of the level, at the checkpoint a first certificate of the later stages
whose source passes the test `Good`, at a later stage of the level a first certificate of the
later stages, and after the level a certificate of the later stages. -/
def holds (level : ℕ) (certificate : Opened O Inner) : Prop :=
  if T.finish ≤ level then tail.holds level certificate.inner
  else if level ≤ T.start then
    accept (tail.point certificate.inner) ∧
      T.Opens (tail.point certificate.inner) certificate.opening
  else if level = T.start + 1 then
    tail.holds T.finish certificate.inner ∧ T.Good tail source certificate
  else tail.holds T.finish certificate.inner

/-- The checkpoint structure of the level over the structure `tail` of the later stages (Akita
Definition 10.16 for the stages from `start`). The candidate of the checkpoint is the successor
source `source` of a certificate of the later stages, and `clash` is `Coll_ℓ`. The claims of a
stage input are refuted for a source when each path of the game from this stage input that avoids
the bad sets ends in a final claim that is false for the source. -/
noncomputable def over :
    Structure (Prefix Wire (Coord → Slot) total) Coord Slot (ℕ → Wire) (Opened O Inner)
      (Brk ⊕ BreakT) (Source ⊕ CandT) (Ring ⊕ KeyT) (Option (Weak O)) where
  protocol := transcript total accept
  kind level := if level < T.finish then T.kind level else tail.kind level
  key level coordinate slot :=
    if level < T.finish then (T.decode slot).map Sum.inl
    else (tail.key level coordinate slot).map Sum.inr
  weight level coordinate := if level < T.finish then T.chance else tail.weight level coordinate
  error level :=
    if level < T.finish then T.costs.getD (level - T.start - 1) 0 else tail.error level
  leaf point := ⟨tail.leaf point, none⟩
  point certificate := tail.point certificate.inner
  holds := T.holds accept tail source
  breaks := Sum.elim T.method.broken tail.breaks
  combine level reference children :=
    if level < T.finish then T.combine tail source reference children
    else Sum.map (fun inner => ⟨inner, none⟩) Sum.inr
      (tail.combine level reference.inner fun coordinate => (children coordinate).inner)
  candidate level certificate :=
    if level < T.finish then .inl (source certificate.inner)
    else .inr (tail.candidate level certificate.inner)
  good level certificate :=
    if level < T.finish then T.Good tail source certificate
    else tail.good level certificate.inner
  collide level certificate other :=
    if level < T.finish then .inr (clash certificate.inner other.inner)
    else .inr (tail.collide level certificate.inner other.inner)
  refuted level input candidate :=
    match candidate with
    | .inl source =>
        level < T.finish ∧ (T.game source input).Avoids fun result => ¬ T.passes source result
    | .inr candidate => T.finish ≤ level ∧ tail.refuted level input candidate
  bad level input candidate :=
    match candidate with
    | .inl source =>
        if level < T.finish then badReplies (Coord → Slot) (T.game source input) else ∅
    | .inr candidate => if level < T.finish then ∅ else tail.bad level input candidate
  witness := Opened.opening
  relation root opening := T.Opens root opening

/-- The interface of the later stages to the level. `fits`: the stages of the level are stages of
the transcript. `decodes`: the verifier rejects an answer of the stage `start` without a fold
challenge for each block. `lawful`: the later stages satisfy the conditions of Definition 10.16.
`accepted` and `passed`: the transcript of a first certificate of the later stages is accepting,
and the final claim of the checks of the level is true for its successor source (Akita
Lemma 10.42 (ii)). `clash`: two first certificates of the later stages with one stage input of the
checkpoint and different sources give a witness of `R_bind` (Definition 10.16 (vi)). -/
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

theorem over_kind (level : ℕ) :
    (T.over accept tail source clash).kind level =
      if level < T.finish then T.kind level else tail.kind level :=
  rfl

theorem over_key (level : ℕ) (coordinate : Coord) (slot : Slot) :
    (T.over accept tail source clash).key level coordinate slot =
      if level < T.finish then (T.decode slot).map Sum.inl
      else (tail.key level coordinate slot).map Sum.inr :=
  rfl

theorem over_weight (level : ℕ) (coordinate : Coord) :
    (T.over accept tail source clash).weight level coordinate =
      if level < T.finish then T.chance else tail.weight level coordinate :=
  rfl

theorem over_error (level : ℕ) :
    (T.over accept tail source clash).error level =
      if level < T.finish then T.costs.getD (level - T.start - 1) 0 else tail.error level :=
  rfl

theorem over_combine (level : ℕ) (reference : Opened O Inner)
    (children : Coord → Opened O Inner) :
    (T.over accept tail source clash).combine level reference children =
      if level < T.finish then T.combine tail source reference children
      else Sum.map (fun inner => ⟨inner, none⟩) Sum.inr
        (tail.combine level reference.inner fun coordinate => (children coordinate).inner) :=
  rfl

theorem over_candidate (level : ℕ) (certificate : Opened O Inner) :
    (T.over accept tail source clash).candidate level certificate =
      if level < T.finish then .inl (source certificate.inner)
      else .inr (tail.candidate level certificate.inner) :=
  rfl

theorem over_good (level : ℕ) (certificate : Opened O Inner) :
    (T.over accept tail source clash).good level certificate =
      if level < T.finish then T.Good tail source certificate
      else tail.good level certificate.inner :=
  rfl

theorem over_collide (level : ℕ) (certificate other : Opened O Inner) :
    (T.over accept tail source clash).collide level certificate other =
      if level < T.finish then .inr (clash certificate.inner other.inner)
      else .inr (tail.collide level certificate.inner other.inner) :=
  rfl

theorem over_bad_inl (level : ℕ) (input : Prefix Wire (Coord → Slot) total) (candidate : Source) :
    (T.over accept tail source clash).bad level input (.inl candidate) =
      if level < T.finish then badReplies (Coord → Slot) (T.game candidate input) else ∅ :=
  rfl

theorem over_bad_inr (level : ℕ) (input : Prefix Wire (Coord → Slot) total) (candidate : CandT) :
    (T.over accept tail source clash).bad level input (.inr candidate) =
      if level < T.finish then ∅ else tail.bad level input candidate :=
  rfl

theorem over_kind_indexed {level : ℕ} {coordinates : List Coord} (low : T.start ≤ level)
    (isIndexed : (T.over accept tail source clash).kind level = .indexed coordinates) :
    (level = T.start ∧ coordinates = (List.finRange S.blocks).map T.block ∧ level < T.finish) ∨
      (T.finish ≤ level ∧ tail.kind level = .indexed coordinates) := by
  rw [over_kind] at isIndexed
  by_cases inside : level < T.finish
  · rw [ite_eq_left inside] at isIndexed
    unfold Tier.kind at isIndexed
    split at isIndexed
    · rename_i high
      exact Or.inl ⟨Nat.le_antisymm high low, (Kind.indexed.inj isIndexed).symm, inside⟩
    · split at isIndexed
      · exact absurd isIndexed (by simp)
      · exact absurd isIndexed (by simp)
  · rw [ite_eq_right inside] at isIndexed
    exact Or.inr ⟨Nat.le_of_not_lt inside, isIndexed⟩

theorem over_kind_checkpoint {level span : ℕ}
    (isCheckpoint : (T.over accept tail source clash).kind level = .checkpoint span) :
    (level = T.start + 1 ∧ span = T.costs.length - 1 ∧ level < T.finish) ∨
      (T.finish ≤ level ∧ tail.kind level = .checkpoint span) := by
  rw [over_kind] at isCheckpoint
  by_cases inside : level < T.finish
  · rw [ite_eq_left inside] at isCheckpoint
    unfold Tier.kind at isCheckpoint
    split at isCheckpoint
    · exact absurd isCheckpoint (by simp)
    · split at isCheckpoint
      · rename_i one
        exact Or.inl ⟨one, (Kind.checkpoint.inj isCheckpoint).symm, inside⟩
      · exact absurd isCheckpoint (by simp)
  · rw [ite_eq_right inside] at isCheckpoint
    exact Or.inr ⟨Nat.le_of_not_lt inside, isCheckpoint⟩

theorem over_kind_plain {level : ℕ}
    (isPlain : (T.over accept tail source clash).kind level = .plain) :
    (T.start + 2 ≤ level ∧ level < T.finish) ∨ (T.finish ≤ level ∧ tail.kind level = .plain) := by
  rw [over_kind] at isPlain
  by_cases inside : level < T.finish
  · rw [ite_eq_left inside] at isPlain
    unfold Tier.kind at isPlain
    split at isPlain
    · exact absurd isPlain (by simp)
    · split at isPlain
      · exact absurd isPlain (by simp)
      · exact Or.inl ⟨by omega, inside⟩
  · rw [ite_eq_right inside] at isPlain
    exact Or.inr ⟨Nat.le_of_not_lt inside, isPlain⟩

omit [Fintype Slot] [Nonempty Slot] [Fintype Coord] [DecidableEq Coord] in
theorem holds_tail {level : ℕ} (outside : T.finish ≤ level) (certificate : Opened O Inner) :
    T.holds accept tail source level certificate ↔ tail.holds level certificate.inner := by
  unfold holds
  rw [ite_eq_left outside]

omit [Fintype Slot] [Nonempty Slot] [Fintype Coord] [DecidableEq Coord] in
theorem holds_start (certificate : Opened O Inner) :
    T.holds accept tail source T.start certificate ↔
      accept (tail.point certificate.inner) ∧
        T.Opens (tail.point certificate.inner) certificate.opening := by
  have ends := T.finish_eq
  unfold holds
  rw [ite_eq_right (by omega), ite_eq_left le_rfl]

omit [Fintype Slot] [Nonempty Slot] [Fintype Coord] [DecidableEq Coord] in
theorem holds_check (draws : T.costs ≠ []) (certificate : Opened O Inner) :
    T.holds accept tail source (T.start + 1) certificate ↔
      tail.holds T.finish certificate.inner ∧ T.Good tail source certificate := by
  have ends := T.finish_eq
  have long : 0 < T.costs.length := List.length_pos_iff.mpr draws
  unfold holds
  rw [ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_left rfl]

omit [Fintype Slot] [Nonempty Slot] [Fintype Coord] [DecidableEq Coord] in
theorem holds_mid {level : ℕ} (low : T.start + 2 ≤ level) (high : level ≤ T.finish)
    (certificate : Opened O Inner) :
    T.holds accept tail source level certificate ↔ tail.holds T.finish certificate.inner := by
  rcases Nat.lt_or_eq_of_le high with inside | rfl
  · unfold holds
    rw [ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_right (by omega)]
  · exact T.holds_tail le_rfl certificate

omit [Nonempty Slot] in
/-- The game at the stage input of the checkpoint has no answer of a draw. -/
theorem game_take_succ (candidate : Source) (point : Prefix Wire (Coord → Slot) total) :
    T.game candidate (point.take (T.start + 1)) =
      settle T.read (point.wire (T.start + 1))
        (T.check (point.wire T.start) (T.challenge point) candidate) 0 := by
  have none : ((point.take (T.start + 1)).rest.take T.finish).drop (T.start + 1) = [] :=
    List.drop_eq_nil_of_le (by
      show ((point.rest.take (T.start + 1)).take T.finish).length ≤ T.start + 1
      rw [List.length_take, List.length_take]
      omega)
  unfold Tier.game
  rw [none, T.challenge_take (Nat.lt_succ_self _), Prefix.wire_take (Nat.le_succ _),
    Prefix.wire_take le_rfl]
  rfl

omit [Nonempty Slot] in
/-- The game reads the stages of the level only. -/
theorem game_take_finish (candidate : Source) (point : Prefix Wire (Coord → Slot) total) :
    T.game candidate (point.take T.finish) = T.game candidate point := by
  have ends := T.finish_eq
  have same : ((point.take T.finish).rest.take T.finish) = point.rest.take T.finish := by
    show (point.rest.take T.finish).take T.finish = _
    rw [List.take_take, Nat.min_self]
  unfold Tier.game
  rw [same, T.challenge_take (by omega), Prefix.wire_take (by omega), Prefix.wire_take (by omega)]

/-- The game after one more answer and one more message. -/
theorem game_step (candidate : Source) (proof : ℕ → Wire)
    {input : Prefix Wire (Coord → Slot) total} (reply : Coord → Slot)
    (after : T.start + 1 ≤ input.rest.length) (before : input.rest.length < T.finish)
    (room : input.rest.length < total) :
    T.game candidate ((transcript total accept).step proof input reply) =
      settle T.read (proof (input.rest.length + 1)) (advance reply (T.game candidate input)) 0 := by
  have moves : ((input.snoc (reply, proof (input.rest.length + 1)) room).rest.take T.finish).drop
      (T.start + 1) =
        (input.rest.take T.finish).drop (T.start + 1) ++
          [(reply, proof (input.rest.length + 1))] := by
    show ((input.rest ++ [(reply, proof (input.rest.length + 1))]).take T.finish).drop
      (T.start + 1) = _
    rw [List.take_of_length_le (by
        rw [List.length_append, List.length_singleton]
        omega),
      List.take_of_length_le (Nat.le_of_lt before), List.drop_append_of_le_length after]
  rw [transcript_step proof input reply room]
  unfold Tier.game
  rw [moves, T.challenge_snoc input _ room (by omega), Prefix.wire_snoc input _ room (by omega),
    Prefix.wire_snoc input _ room (by omega)]
  exact position_append T.read _ _ _ _ reply

/-- `Comb_r` at the stage `start` is correct (Definition 10.16 (iv) with Akita
Corollary 10.32). -/
theorem combine_opens (valid : T.Valid) (fed : T.Fed accept tail source clash)
    (reference : Opened O Inner) (family : Coord → Opened O Inner)
    (anchor deeper : Prefix Wire (Coord → Slot) total)
    (held : T.holds accept tail source (T.start + 1) reference)
    (found : (transcript total accept).cut T.start (tail.point reference.inner) = some anchor)
    (foundDeeper :
      (transcript total accept).cut (T.start + 1) (tail.point reference.inner) = some deeper)
    (each : ∀ coordinate ∈ (List.finRange S.blocks).map T.block,
      T.holds accept tail source (T.start + 1) (family coordinate) ∧
        (transcript total accept).cut T.start (tail.point (family coordinate).inner) =
          some anchor ∧
        ∃ other slot,
          (transcript total accept).cut (T.start + 1) (tail.point (family coordinate).inner) =
            some other ∧
          (transcript total accept).last other =
            Function.update ((transcript total accept).last deeper) coordinate slot ∧
          T.decode slot ≠ T.decode ((transcript total accept).last deeper coordinate)) :
    Sum.elim
      (fun certificate : Opened O Inner =>
        (accept (tail.point certificate.inner) ∧
          T.Opens (tail.point certificate.inner) certificate.opening) ∧
        ∀ earlier ≤ T.start,
          (transcript total accept).cut earlier (tail.point certificate.inner) =
            (transcript total accept).cut earlier (tail.point reference.inner))
      (Sum.elim T.method.broken tail.breaks)
      (T.combine tail source reference family) := by
  obtain ⟨heldTail, good⟩ := (T.holds_check valid.draws reference).mp held
  have good : T.method.Satisfies (T.digits (source reference.inner))
      (T.claim ((tail.point reference.inner).wire T.start))
      (T.challenge (tail.point reference.inner)) := good
  have passed := fed.passed reference.inner heldTail
  obtain ⟨_, rfl⟩ := transcript_cut found
  obtain ⟨_, rfl⟩ := transcript_cut foundDeeper
  obtain ⟨reply, answered, _⟩ := passed.2.1
  have base := last_take_succ (accept := accept) answered
  have child : ∀ i : Fin S.blocks,
      (tail.holds T.finish (family (T.block i)).inner ∧
        T.method.Satisfies (T.digits (source (family (T.block i)).inner))
          (T.claim ((tail.point reference.inner).wire T.start))
          (T.challenge (tail.point (family (T.block i)).inner))) ∧
      ∃ slot, (tail.point (family (T.block i)).inner).reply T.start =
          some (Function.update reply (T.block i) slot) ∧
        T.decode slot ≠ T.decode (reply (T.block i)) := by
    intro i
    obtain ⟨heldChild, foundChild, other, slot, foundOther, last, differs⟩ :=
      each (T.block i) (List.mem_map.mpr ⟨i, List.mem_finRange i, rfl⟩)
    obtain ⟨heldInner, goodChild⟩ := (T.holds_check valid.draws (family (T.block i))).mp heldChild
    have goodChild : T.method.Satisfies (T.digits (source (family (T.block i)).inner))
        (T.claim ((tail.point (family (T.block i)).inner).wire T.start))
        (T.challenge (tail.point (family (T.block i)).inner)) := goodChild
    obtain ⟨_, sameAnchor⟩ := transcript_cut foundChild
    have sameWire : (tail.point (family (T.block i)).inner).wire T.start =
        (tail.point reference.inner).wire T.start := by
      rw [← Prefix.wire_take le_rfl (tail.point (family (T.block i)).inner), ← sameAnchor,
        Prefix.wire_take le_rfl]
    rw [sameWire] at goodChild
    refine ⟨⟨heldInner, goodChild⟩, slot, ?_, ?_⟩
    · obtain ⟨_, rfl⟩ := transcript_cut foundOther
      obtain ⟨replyChild, answeredChild, _⟩ := (fed.passed _ heldInner).2.1
      rw [last_take_succ answeredChild, base] at last
      rw [answeredChild, last]
    · rw [base] at differs
      exact differs
  have result := valid.merges (T.claim ((tail.point reference.inner).wire T.start))
    (fun _ => (T.challenge (tail.point reference.inner), T.digits (source reference.inner)))
    (fun i => (T.challenge (tail.point (family (T.block i)).inner),
      T.digits (source (family (T.block i)).inner)))
    (T.digits (source reference.inner)) (T.challenge (tail.point reference.inner))
    (fun i => ⟨_, passed.2.1.challenge answered i⟩)
    (fun i => by
      obtain ⟨⟨heldInner, _⟩, slot, answeredChild, _⟩ := child i
      exact ⟨_, (fed.passed _ heldInner).2.1.challenge answeredChild i⟩)
    (fun i same => by
      obtain ⟨⟨heldInner, _⟩, slot, answeredChild, differs⟩ := child i
      have second := (fed.passed _ heldInner).2.1.challenge answeredChild i
      rw [Function.update_self] at second
      refine differs (second.trans ?_)
      rw [passed.2.1.challenge answered i]
      exact congrArg some same.symm)
    (fun i j other => by
      obtain ⟨⟨heldInner, _⟩, slot, answeredChild, _⟩ := child i
      have second := (fed.passed _ heldInner).2.1.challenge answeredChild j
      rw [Function.update_of_ne (fun same => other (valid.injective same)),
        passed.2.1.challenge answered j] at second
      exact Option.some.inj second)
    (fun _ => good) (fun i => (child i).1.2) good
  unfold Tier.combine
  generalize T.method.merge _ _ _ _ = merged at result ⊢
  rcases merged with ω | solution
  · exact ⟨⟨fed.accepted _ heldTail, ω, rfl, result⟩, fun _ _ => rfl⟩
  · exact result

/-- The level over the later stages satisfies the conditions of Akita Definition 10.16 from its
first stage (Akita Proposition 10.44 for the stages from one level). -/
theorem over_lawfulFrom (valid : T.Valid)
    (fed : T.Fed accept tail source clash) :
    (T.over accept tail source clash).LawfulFrom (transcript total accept) T.start := by
  have ends := T.finish_eq
  have long : 0 < T.costs.length := List.length_pos_iff.mpr valid.draws
  have fits := fed.fits
  refine {
    segment := ?_
    key_some := ?_
    key_mass := ?_
    weight_nonnegative := ?_
    leaf_point := fun point => fed.lawful.leaf_point point
    leaf_holds := fun point accepted =>
      (T.holds_tail fits _).mpr (fed.lawful.leaf_holds point accepted)
    plain_holds := ?_
    combine_holds := ?_
    good_holds := ?_
    collide_breaks := ?_
    refuted_start := ?_
    refuted_step := ?_
    refuted_end := ?_
    bad_mass := ?_
    error_nonnegative := ?_ }
  · intro level span low isCheckpoint
    rcases T.over_kind_checkpoint isCheckpoint with ⟨rfl, rfl, _⟩ | ⟨outside, isTail⟩
    · refine ⟨by
        show T.start + 1 + (T.costs.length - 1) < total
        omega, fun later above below => ?_⟩
      rw [over_kind, ite_eq_left (by omega)]
      unfold Tier.kind
      rw [ite_eq_right (by omega), ite_eq_right (by omega)]
    · obtain ⟨short, plains⟩ := fed.lawful.segment level span outside isTail
      refine ⟨short, fun later above below => ?_⟩
      rw [over_kind, ite_eq_right (by omega)]
      exact plains later above below
  · intro level coordinates coordinate point deeper low isIndexed member accepted found
    rcases T.over_kind_indexed low isIndexed with ⟨rfl, rfl, inside⟩ | ⟨outside, isTail⟩
    · obtain ⟨i, _, rfl⟩ := List.mem_map.mp member
      obtain ⟨_, rfl⟩ := transcript_cut found
      obtain ⟨reply, answered, each⟩ := fed.decodes point accepted
      rw [over_key, ite_eq_left inside, last_take_succ answered, Option.isSome_map]
      exact each i
    · rw [over_key, ite_eq_right (by omega), Option.isSome_map]
      exact fed.lawful.key_some level coordinates coordinate point deeper outside isTail member
        accepted found
  · intro level coordinates coordinate value low isIndexed member
    rcases T.over_kind_indexed low isIndexed with ⟨rfl, rfl, inside⟩ | ⟨outside, isTail⟩
    · simp only [over_key, over_weight, ite_eq_left inside]
      exact mass_inl_le T.decode value (slotChance_nonnegative T.decode) (mass_slot_le T.decode)
    · simp only [over_key, over_weight, ite_eq_right (Nat.not_lt.mpr outside)]
      exact mass_inr_le _ value (fed.lawful.weight_nonnegative level coordinate) fun second =>
        fed.lawful.key_mass level coordinates coordinate second outside isTail member
  · intro level coordinate
    rw [over_weight]
    split
    · exact slotChance_nonnegative T.decode
    · exact fed.lawful.weight_nonnegative level coordinate
  · intro level certificate low isPlain held
    rcases T.over_kind_plain isPlain with ⟨above, inside⟩ | ⟨outside, isTail⟩
    · exact (T.holds_mid above (Nat.le_of_lt inside) certificate).mpr
        ((T.holds_mid (by omega) (by omega) certificate).mp held)
    · exact (T.holds_tail outside certificate).mpr (fed.lawful.plain_holds level certificate.inner
        outside isTail ((T.holds_tail (by omega) certificate).mp held))
  · intro level coordinates reference family anchor deeper low isIndexed held found foundDeeper
      each
    rcases T.over_kind_indexed low isIndexed with ⟨rfl, rfl, inside⟩ | ⟨outside, isTail⟩
    · have combined := T.combine_opens valid fed reference family anchor deeper held found
        foundDeeper fun coordinate member => by
          obtain ⟨heldChild, foundChild, other, slot, foundOther, last, differs⟩ :=
            each coordinate member
          refine ⟨heldChild, foundChild, other, slot, foundOther, last, fun same => differs ?_⟩
          rw [over_key, over_key, ite_eq_left inside, ite_eq_left inside, same]
      rw [over_combine, ite_eq_left inside]
      generalize T.combine tail source reference family = result at combined ⊢
      rcases result with certificate | broken
      · exact ⟨(T.holds_start certificate).mpr combined.1, combined.2⟩
      · exact combined
    · have combined := fed.lawful.combine_holds level coordinates reference.inner
        (fun coordinate => (family coordinate).inner) anchor deeper outside isTail
        ((T.holds_tail (by omega) reference).mp held) found foundDeeper
        fun coordinate member => by
          obtain ⟨heldChild, foundChild, other, slot, foundOther, last, differs⟩ :=
            each coordinate member
          refine ⟨(T.holds_tail (by omega) (family coordinate)).mp heldChild, foundChild, other,
            slot, foundOther, last, fun same => differs ?_⟩
          rw [over_key, over_key, ite_eq_right (by omega), ite_eq_right (by omega), same]
      rw [over_combine, ite_eq_right (by omega)]
      generalize tail.combine level reference.inner
        (fun coordinate => (family coordinate).inner) = result at combined ⊢
      rcases result with inner | broken
      · exact ⟨(T.holds_tail outside _).mpr combined.1, combined.2⟩
      · exact combined
  · intro level span certificate low isCheckpoint held isGood
    rcases T.over_kind_checkpoint isCheckpoint with ⟨rfl, rfl, inside⟩ | ⟨outside, isTail⟩
    · rw [over_good, ite_eq_left inside] at isGood
      exact (T.holds_check valid.draws certificate).mpr
        ⟨(T.holds_mid le_rfl (by omega) certificate).mp held, isGood⟩
    · rw [over_good, ite_eq_right (by omega)] at isGood
      exact (T.holds_tail outside certificate).mpr (fed.lawful.good_holds level span
        certificate.inner outside isTail ((T.holds_tail (by omega) certificate).mp held) isGood)
  · intro level span certificate other anchor low isCheckpoint held heldOther found foundOther
      different
    rcases T.over_kind_checkpoint isCheckpoint with ⟨rfl, rfl, inside⟩ | ⟨outside, isTail⟩
    · rw [over_candidate, over_candidate, ite_eq_left inside, ite_eq_left inside] at different
      rw [over_collide, ite_eq_left inside]
      exact fed.clash certificate.inner other.inner anchor
        ((T.holds_mid le_rfl (by omega) certificate).mp held)
        ((T.holds_mid le_rfl (by omega) other).mp heldOther) found foundOther
        fun same => different (congrArg Sum.inl same)
    · rw [over_candidate, over_candidate, ite_eq_right (by omega), ite_eq_right (by omega)]
        at different
      rw [over_collide, ite_eq_right (by omega)]
      exact fed.lawful.collide_breaks level span certificate.inner other.inner anchor outside
        isTail ((T.holds_tail (by omega) certificate).mp held)
        ((T.holds_tail (by omega) other).mp heldOther) found foundOther
        fun same => different (congrArg Sum.inr same)
  · intro level span certificate anchor low isCheckpoint held notGood found
    rcases T.over_kind_checkpoint isCheckpoint with ⟨rfl, rfl, inside⟩ | ⟨outside, isTail⟩
    · rw [over_good, ite_eq_left inside] at notGood
      rw [over_candidate, ite_eq_left inside]
      obtain ⟨_, rfl⟩ := transcript_cut found
      refine ⟨inside, ?_⟩
      rw [T.game_take_succ]
      exact avoids_settle T.read _ ((valid.sound _ _ _).mono fun result implies passed =>
        notGood (implies passed)) 0
    · rw [over_good, ite_eq_right (by omega)] at notGood
      rw [over_candidate, ite_eq_right (by omega)]
      exact ⟨outside, fed.lawful.refuted_start level span certificate.inner anchor outside isTail
        ((T.holds_tail (by omega) certificate).mp held) notGood found⟩
  · intro level span input candidate reply proof low isCheckpoint after before refuted outside
    rcases candidate with candidate | candidate
    · obtain ⟨inside, refuted⟩ := refuted
      rcases T.over_kind_checkpoint isCheckpoint with ⟨rfl, rfl, _⟩ | ⟨out, _⟩
      · have after : T.start + 1 ≤ input.rest.length := after
        have before : input.rest.length ≤ T.start + 1 + (T.costs.length - 1) := before
        rw [over_bad_inl, ite_eq_left inside] at outside
        refine ⟨inside, ?_⟩
        rw [T.game_step candidate proof reply after (by omega) (by omega)]
        exact avoids_settle T.read _ (avoids_advance refuted outside) 0
      · exact absurd inside (by omega)
    · obtain ⟨out, refuted⟩ := refuted
      rcases T.over_kind_checkpoint isCheckpoint with ⟨rfl, rfl, inside⟩ | ⟨_, isTail⟩
      · exact absurd out (by omega)
      · rw [over_bad_inr, ite_eq_right (by omega)] at outside
        exact ⟨out, fed.lawful.refuted_step level span input candidate reply proof out isTail
          after before refuted outside⟩
  · intro level span certificate input low isCheckpoint held found refuted
    rcases T.over_kind_checkpoint isCheckpoint with ⟨rfl, rfl, inside⟩ | ⟨outside, isTail⟩
    · rw [over_candidate, ite_eq_left inside] at refuted
      obtain ⟨_, avoided⟩ := refuted
      obtain ⟨_, _, result, ended, passed⟩ := fed.passed certificate.inner
        ((T.holds_mid le_rfl (by omega) certificate).mp held)
      obtain ⟨_, rfl⟩ := transcript_cut found
      have avoided : (T.game (source certificate.inner) ((tail.point certificate.inner).take
          (T.start + 1 + (T.costs.length - 1) + 1))).Avoids
            fun result => ¬ T.passes (source certificate.inner) result := avoided
      have whole : T.start + 1 + (T.costs.length - 1) + 1 = T.finish := by omega
      rw [whole, T.game_take_finish, ended] at avoided
      exact avoided passed
    · rw [over_candidate, ite_eq_right (by omega)] at refuted
      exact fed.lawful.refuted_end level span certificate.inner input outside isTail
        ((T.holds_tail (by omega) certificate).mp held) found refuted.2
  · intro level span input candidate low isCheckpoint after before
    rcases T.over_kind_checkpoint isCheckpoint with ⟨rfl, rfl, inside⟩ | ⟨outside, isTail⟩
    · have after : T.start + 1 ≤ input.rest.length := after
      have before : input.rest.length ≤ T.start + 1 + (T.costs.length - 1) := before
      show mass ((T.over accept tail source clash).bad _ input candidate) ≤
        (T.over accept tail source clash).error input.rest.length
      rw [over_error, ite_eq_left (by omega)]
      rcases candidate with candidate | candidate
      · rw [over_bad_inl, ite_eq_left inside]
        have staged := position_staged T.read
          (valid.staged (input.wire T.start) (T.challenge input) candidate)
          (input.wire (T.start + 1)) ((input.rest.take T.finish).drop (T.start + 1))
        have count : ((input.rest.take T.finish).drop (T.start + 1)).length =
            input.rest.length - T.start - 1 := by
          rw [List.length_drop, List.length_take]
          omega
        have head : (T.costs.drop ((input.rest.take T.finish).drop (T.start + 1)).length).headD 0 =
            T.costs.getD (input.rest.length - T.start - 1) 0 := by
          rw [count, List.getD_eq_getElem?_getD, List.headD_eq_head?_getD, List.head?_drop]
        have bound := mass_badReplies_le (by
          rw [head]
          exact T.costs_getD_nonnegative valid.nonnegative _) staged
        rw [head] at bound
        exact bound
      · rw [over_bad_inr, ite_eq_left inside]
        exact (le_of_eq (by simp [mass])).trans (T.costs_getD_nonnegative valid.nonnegative _)
    · have after : level ≤ input.rest.length := after
      show mass ((T.over accept tail source clash).bad _ input candidate) ≤
        (T.over accept tail source clash).error input.rest.length
      rw [over_error, ite_eq_right (by omega)]
      rcases candidate with candidate | candidate
      · rw [over_bad_inl, ite_eq_right (by omega)]
        exact (le_of_eq (by simp [mass])).trans (fed.lawful.error_nonnegative _)
      · rw [over_bad_inr, ite_eq_right (by omega)]
        exact fed.lawful.bad_mass level span input candidate outside isTail after before
  · intro level
    rw [over_error]
    split
    · exact T.costs_getD_nonnegative valid.nonnegative _
    · exact fed.lawful.error_nonnegative level

/-- Akita Proposition 10.44 (i) for the stages from one level: a first certificate of the level
has an accepting transcript and a weak opening of the claim of the level. -/
theorem over_opened (certificate : Opened O Inner)
    (held : (T.over accept tail source clash).holds T.start certificate) :
    accept ((T.over accept tail source clash).point certificate) ∧
      T.Opens ((T.over accept tail source clash).point certificate)
        ((T.over accept tail source clash).witness certificate) :=
  (T.holds_start certificate).mp held

end Random

end Tier

end NightstreamFPrime.Spec.Compression.Akita.Extract
