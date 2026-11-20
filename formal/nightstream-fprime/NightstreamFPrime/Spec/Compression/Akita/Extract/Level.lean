import NightstreamFPrime.Spec.Checkpoint.Structure
import NightstreamFPrime.Spec.Compression.Akita.Fold
import NightstreamFPrime.Spec.Compression.Akita.Extract.Staged
import NightstreamFPrime.Spec.Compression.Akita.Extract.Transcript

/-! One packing fold of Akita as a checkpoint structure (ePrint 2026/1983, revision of 2026-10-15,
Definition 10.16, with Corollary 10.32 as `Comb_r` of the fold challenge and the checks after the
fold challenge as one checkpoint segment).

- `Level` has the data of one fold: the fixed statement, the decoder of a fold challenge from one
  slot of the answer of the oracle, the decoders of the messages of the proof, and the game of the
  checks after the fold challenge with the bounds `costs` of its draws.
- The transcript has the stage 0 for the fold challenges `c_i`, one for each block, and one stage
  for each draw of the game. After the answer of the stage 0 the proof has the digit vectors of
  the fold in the clear. The paper has a commitment to them and a claim for the next level. This
  is the model of a last level, as in the terminal of Akita Section 5.6.
- `Level.extraction` is the checkpoint structure: the stage 0 is an indexed stage, the stage 1 is
  a checkpoint, and the later stages are the plain stages of its segment.
- `Level.lawful` states that it satisfies the conditions of Akita Definition 10.16. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Extract

open NightstreamFPrime.Spec.Checkpoint
open NightstreamFPrime.Spec.Compression (Game)
open NightstreamFPrime.Spec.RandomOracle (mass mass_nonnegative)

attribute [local instance low] Classical.propDecidable

/-- The data of one packing fold for one claim. `basis` and `statement` are the fixed folding
statement of Akita Definition 10.22. `block i` is the coordinate of the answer of the oracle that
has the slot of the fold challenge `c_i`, and `decode` gives the challenge of a slot, or `none`
for a slot that the verifier rejects. `witnessOf` and `read` decode the messages of the proof.
`check c w` is the game of the checks of the verifier for the fold challenges `c` and the digit
vectors `w`, `passes w` is the acceptance of its result, and `costs` has one bound for each of
its draws. `weight` is the norm `ω = ‖c‖_1` of a fold challenge. -/
structure Level {S : Shape} (O : OpeningShape S) (fold : Descriptor)
    (E Wire Coord Slot Result : Type) [CommRing E] [Module Zq E] where
  basis : Module.Basis (Fin O.planes) Zq E
  statement : Statement O E
  block : Fin S.blocks → Coord
  decode : Slot → Option (Rq O.carrierDegree)
  witnessOf : Wire → Witness O fold
  read : Reader Wire
  check : (Fin S.blocks → Rq O.carrierDegree) → Witness O fold → Game Result
  passes : Witness O fold → Result → Prop
  costs : List ℝ
  weight : ℕ

namespace Level

variable {S : Shape} {O : OpeningShape S} {fold : Descriptor} {E Wire Coord Slot Result : Type}
  [CommRing E] [Module Zq E] (L : Level O fold E Wire Coord Slot Result)

/-- The number of stages: the fold challenge, and one stage for each draw of the checks. -/
abbrev stages : ℕ := L.costs.length + 1

/-- A stage input of the fold. -/
abbrev Stage : Type := Prefix Wire (Coord → Slot) L.stages

/-- The set of the fold challenges: the values of the decoder. -/
def family : Set (Rq O.carrierDegree) := {challenge | ∃ slot, L.decode slot = some challenge}

/-- The fold challenges `c_i` of a stage input, from the answer of the stage 0. -/
noncomputable def challenge (point : L.Stage) (i : Fin S.blocks) : Rq O.carrierDegree :=
  ((point.reply 0).bind fun reply => L.decode (reply (L.block i))).getD 0

/-- The answer of the stage 0 gives a fold challenge for each block. -/
def Decodes (point : L.Stage) : Prop :=
  ∃ reply, point.reply 0 = some reply ∧ ∀ i, (L.decode (reply (L.block i))).isSome

/-- The digit vectors that the proof has after the fold challenge. -/
def witness (point : L.Stage) : Witness O fold := L.witnessOf (point.wire 1)

/-- A certificate: a transcript prefix, and after the extraction at the stage 0 a weak opening. -/
structure Cert where
  point : L.Stage
  opening : Option (Weak O)

/-- The relation of the extraction: a weak opening of the statement (Akita Definition 10.22) with
`κ̄_1 = 2ω` and `β̄_∞ = Δ^cert_f`, to the table that its source blocks decode to. -/
def Opens (opening : Option (Weak O)) : Prop :=
  ∃ ω, opening = some ω ∧
    ω.Opens L.basis (2 * L.weight) (fold.sourceBound S) L.statement (S.decode ω.source)

/-- A witness of `R_bind`: a solution of Module-SIS for `B` or `D` with the bound `b⋆ − 1`, or for
a compression map with the bound 1 (Akita Table 3). -/
abbrev Break : Type := Option (OuterBreak L.statement.key ⊕ OpenBreak L.statement.openKey)

/-- Definition 10.16 (i): the stage 0 is indexed by the blocks, the stage 1 is the checkpoint of
the checks, and its segment has the other stages. -/
def kind : ℕ → Kind Coord
  | 0 => .indexed ((List.finRange S.blocks).map L.block)
  | 1 => .checkpoint (L.costs.length - 1)
  | _ => .plain

/-- The children of Akita Definition 10.30 (ii) for the extraction at the stage 0: the reference
transcript is the first child of each block, and the transcript of the coordinate of block `i` is
its second child. -/
noncomputable def fork (reference : L.Cert) (children : Coord → L.Cert) : Fork O fold where
  first _ := ⟨L.challenge reference.point, L.witness reference.point⟩
  second i := ⟨L.challenge (children (L.block i)).point, L.witness (children (L.block i)).point⟩

/-- `Comb_0`: the reduction of Akita Corollary 10.32. It returns a solution of Module-SIS when a
child has other carried digits than the reference, and else the weak opening `extract`. -/
noncomputable def combine (reference : L.Cert) (children : Coord → L.Cert) : L.Cert ⊕ L.Break :=
  match forkCollision L.statement (L.fork reference children)
      (L.witness reference.point).toCarried with
  | some solution => .inr (some solution)
  | none => .inl ⟨reference.point,
      some (extract (L.fork reference children) (L.witness reference.point).toCarried)⟩

theorem costs_getD_nonnegative (nonnegative : ∀ cost ∈ L.costs, 0 ≤ cost) (index : ℕ) :
    0 ≤ L.costs.getD index 0 := by
  rw [List.getD_eq_getElem?_getD]
  rcases entry : L.costs[index]? with _ | cost
  · exact le_rfl
  · exact nonnegative cost (List.mem_of_getElem? entry)

variable {L} in
theorem Decodes.challenge {point : L.Stage} {reply : Coord → Slot}
    (answered : point.reply 0 = some reply) (decodes : L.Decodes point) (i : Fin S.blocks) :
    L.decode (reply (L.block i)) = some (L.challenge point i) := by
  obtain ⟨other, same, each⟩ := decodes
  obtain rfl := Option.some.inj (same.symm.trans answered)
  obtain ⟨value, decoded⟩ := Option.isSome_iff_exists.mp (each i)
  simp [Level.challenge, answered, decoded]

variable {L} in
theorem challenge_take {level : ℕ} (positive : 0 < level) (point : L.Stage) :
    L.challenge (point.take level) = L.challenge point := by
  funext i
  simp only [Level.challenge, Prefix.reply_take positive]

variable {L} in
theorem witness_take {level : ℕ} (positive : 0 < level) (point : L.Stage) :
    L.witness (point.take level) = L.witness point := by
  simp only [Level.witness, Prefix.wire_take positive]

section Random

variable [Fintype Slot] [Nonempty Slot]

/-- The largest probability of one fold challenge for a uniform slot. It is `1/|C|` for a decoder
with the uniform law on the challenge set `C`. -/
noncomputable def chance : ℝ :=
  Finset.univ.sup' Finset.univ_nonempty fun slot : Slot =>
    if (L.decode slot).isSome then mass {other | L.decode other = L.decode slot} else 0

theorem chance_nonnegative : 0 ≤ L.chance := by
  obtain ⟨slot⟩ := ‹Nonempty Slot›
  refine le_trans ?_ (Finset.le_sup' _ (Finset.mem_univ slot))
  split
  · exact mass_nonnegative _
  · exact le_rfl

theorem mass_decode_le (value : Rq O.carrierDegree) :
    mass {slot | L.decode slot = some value} ≤ L.chance := by
  by_cases reached : ∃ slot, L.decode slot = some value
  · obtain ⟨slot, decoded⟩ := reached
    refine le_trans (le_of_eq ?_) (Finset.le_sup' _ (Finset.mem_univ slot))
    rw [ite_eq_left (by rw [decoded]; rfl), decoded]
  · have empty : {slot | L.decode slot = some value} = (∅ : Set Slot) :=
      Set.eq_empty_of_forall_notMem fun slot decoded => reached ⟨slot, decoded⟩
    rw [empty]
    exact (le_of_eq (by simp [mass])).trans L.chance_nonnegative

variable [Fintype Coord] [DecidableEq Coord]

/-- The game of the checks for the digit vectors `candidate`, after the answers and the messages
of a stage input. -/
noncomputable def game (candidate : Witness O fold) (point : L.Stage) : Game Result :=
  position L.read (L.check (L.challenge point) candidate) (point.wire 1) (point.rest.drop 1)

/-- The verifier accepts a complete transcript: each fold challenge decodes, and the game of the
checks for the digit vectors of the proof ends in an accepted result. -/
def Accepts (point : L.Stage) : Prop :=
  point.rest.length = L.stages ∧ L.Decodes point ∧
    ∃ result, L.game (L.witness point) point = .pure result ∧ L.passes (L.witness point) result

/-- The certificate sets `Y_level`: a weak opening at the stage 0, an accepting transcript whose
digit vectors satisfy the exact branch relation at the stage 1, and an accepting transcript at a
later stage. -/
def holds : ℕ → L.Cert → Prop
  | 0, certificate => L.Opens certificate.opening
  | 1, certificate => L.Accepts certificate.point ∧
      (L.witness certificate.point).Satisfies L.basis L.statement (L.challenge certificate.point)
  | _, certificate => L.Accepts certificate.point

/-- The checkpoint structure of one fold (Akita Definition 10.16). The candidate of the checkpoint
is the family of the digit vectors of the proof. The claims of a stage input are refuted for a
candidate when each path of the game from this stage input that avoids the bad sets ends in a
result that is not accepted. -/
noncomputable def extraction :
    Structure L.Stage Coord Slot (ℕ → Wire) L.Cert L.Break (Witness O fold)
      (Rq O.carrierDegree) (Option (Weak O)) where
  protocol := transcript L.stages L.Accepts
  kind := L.kind
  key _ _ := L.decode
  weight _ _ := L.chance
  error level := L.costs.getD (level - 1) 0
  leaf point := ⟨point, none⟩
  point := Cert.point
  holds := L.holds
  breaks solution := solution.isSome
  combine _ := L.combine
  candidate _ certificate := L.witness certificate.point
  good _ certificate :=
    (L.witness certificate.point).Satisfies L.basis L.statement (L.challenge certificate.point)
  collide _ _ _ := none
  refuted _ input candidate :=
    (L.game candidate input).Avoids fun result => ¬ L.passes candidate result
  bad _ input candidate := badReplies (Coord → Slot) (L.game candidate input)
  witness := Cert.opening
  relation _ := L.Opens

/-- The conditions on the data of one fold. `injective`: the blocks have different coordinates.
`units` and `sparse`: the challenge family has units as differences (Akita Definition 9.2, E1) and
members of the norm `ω`. `base`: `b_z ≥ 2`. `sound`: on a path of the game that avoids the bad
sets, an accepted result gives the exact branch relation. `staged` and `nonnegative`: the bounds
`costs` of the draws. `draws`: the game has a draw. -/
structure Valid (peak : ℕ) : Prop where
  injective : Function.Injective L.block
  units : UnitDifferences L.family
  sparse : ∀ challenge ∈ L.family, Sparse L.weight peak challenge
  base : 2 ≤ fold.responseBase
  sound : ∀ challenge candidate, (L.check challenge candidate).Avoids fun result =>
    L.passes candidate result → candidate.Satisfies L.basis L.statement challenge
  staged : ∀ challenge candidate, Staged (Coord → Slot) (L.check challenge candidate) L.costs
  nonnegative : ∀ cost ∈ L.costs, 0 ≤ cost
  draws : L.costs ≠ []

variable {L}

theorem last_take_one {point : L.Stage} {reply : Coord → Slot}
    (answered : point.reply 0 = some reply) :
    L.extraction.protocol.last (point.take 1) = reply := by
  obtain ⟨first, rest, short⟩ := point
  rcases rest with _ | ⟨move, rest⟩
  · simp [Prefix.reply] at answered
  · simp only [Prefix.reply, List.getElem?_cons_zero, Option.map_some, Option.some.injEq]
      at answered
    simp [extraction, transcript, Prefix.take, answered]

/-- The game after one more answer and one more message. -/
theorem game_step (candidate : Witness O fold) (proof : ℕ → Wire) {input : L.Stage}
    (reply : Coord → Slot) (positive : 0 < input.rest.length)
    (room : input.rest.length < L.stages) :
    L.game candidate (L.extraction.protocol.step proof input reply) =
      settle L.read (proof (input.rest.length + 1))
        (advance reply (L.game candidate input)) 0 := by
  have stepped : L.extraction.protocol.step proof input reply =
      ⟨input.first, input.rest ++ [(reply, proof (input.rest.length + 1))], by
        rw [List.length_append, List.length_singleton]
        exact room⟩ := dite_eq_left room
  have first : (input.rest ++ [(reply, proof (input.rest.length + 1))])[0]? = input.rest[0]? :=
    List.getElem?_append_left positive
  rw [stepped]
  unfold Level.game
  have sameChallenge : L.challenge (⟨input.first,
      input.rest ++ [(reply, proof (input.rest.length + 1))], by
        rw [List.length_append, List.length_singleton]
        exact room⟩ : L.Stage) = L.challenge input := by
    funext i
    simp only [Level.challenge, Prefix.reply, first]
  have sameWire : (⟨input.first, input.rest ++ [(reply, proof (input.rest.length + 1))], by
        rw [List.length_append, List.length_singleton]
        exact room⟩ : L.Stage).wire 1 = input.wire 1 := by
    simp only [Prefix.wire, first]
  rw [sameChallenge, sameWire]
  simp only [List.drop_append_of_le_length (Nat.succ_le_of_lt positive)]
  exact position_append L.read _ _ _ _ reply

variable (L) in
/-- One packing fold satisfies the conditions of Akita Definition 10.16. -/
theorem lawful {peak : ℕ} (valid : L.Valid peak) : L.extraction.Lawful := by
  have long : 0 < L.costs.length := List.length_pos_iff.mpr valid.draws
  have accepts : ∀ level certificate, 2 ≤ level →
      (L.holds level certificate ↔ L.Accepts certificate.point) := by
    intro level certificate large
    obtain ⟨level, rfl⟩ : ∃ smaller, level = smaller + 2 := ⟨level - 2, by omega⟩
    exact Iff.rfl
  have checkpoint : ∀ level span, L.kind level = .checkpoint span →
      level = 1 ∧ span = L.costs.length - 1 := by
    intro level span isCheckpoint
    rcases level with _ | _ | level
    · exact absurd isCheckpoint (by simp [Level.kind])
    · simp only [Level.kind, Kind.checkpoint.injEq] at isCheckpoint
      exact ⟨rfl, isCheckpoint.symm⟩
    · exact absurd isCheckpoint (by simp [Level.kind])
  have indexed : ∀ level coordinates, L.kind level = .indexed coordinates →
      level = 0 ∧ coordinates = (List.finRange S.blocks).map L.block := by
    intro level coordinates isIndexed
    rcases level with _ | _ | level
    · simp only [Level.kind, Kind.indexed.injEq] at isIndexed
      exact ⟨rfl, isIndexed.symm⟩
    · exact absurd isIndexed (by simp [Level.kind])
    · exact absurd isIndexed (by simp [Level.kind])
  refine {
    segment := ?_
    key_some := ?_
    key_mass := fun _ _ _ value _ _ => L.mass_decode_le value
    weight_nonnegative := fun _ _ => L.chance_nonnegative
    leaf_point := fun _ => rfl
    leaf_holds := fun point accepted => (accepts _ _ (by
      show 2 ≤ L.costs.length + 1
      omega)).mpr accepted
    plain_holds := ?_
    combine_holds := ?_
    good_holds := ?_
    collide_breaks := ?_
    refuted_start := ?_
    refuted_step := ?_
    refuted_end := ?_
    bad_mass := ?_
    error_nonnegative := fun level => L.costs_getD_nonnegative valid.nonnegative (level - 1)
    witness_holds := fun _ _ held _ => held }
  · intro level span isCheckpoint
    obtain ⟨rfl, rfl⟩ := checkpoint level span isCheckpoint
    refine ⟨by
      show 1 + (L.costs.length - 1) < L.costs.length + 1
      omega, fun later low _ => ?_⟩
    obtain ⟨later, rfl⟩ : ∃ smaller, later = smaller + 2 := ⟨later - 2, by omega⟩
    rfl
  · intro level coordinates coordinate point deeper isIndexed member accepted found
    obtain ⟨rfl, rfl⟩ := indexed level coordinates isIndexed
    obtain ⟨i, _, rfl⟩ := List.mem_map.mp member
    have found : L.extraction.protocol.cut 1 point = some deeper := found
    obtain ⟨_, rfl⟩ := transcript_cut found
    obtain ⟨reply, answered, each⟩ := accepted.2.1
    show (L.decode (L.extraction.protocol.last (point.take 1) (L.block i))).isSome
    rw [last_take_one answered]
    exact each i
  · intro level certificate isPlain held
    have isPlain : L.kind level = .plain := isPlain
    have large : 2 ≤ level := by
      rcases level with _ | _ | level
      · exact absurd isPlain (by simp [Level.kind])
      · exact absurd isPlain (by simp [Level.kind])
      · omega
    exact (accepts level certificate large).mpr
      ((accepts (level + 1) certificate (by omega)).mp held)
  · intro level coordinates reference children anchor deeper isIndexed held _ found each
    obtain ⟨rfl, rfl⟩ := indexed level coordinates isIndexed
    obtain ⟨accepted, satisfied⟩ : L.Accepts reference.point ∧
        (L.witness reference.point).Satisfies L.basis L.statement (L.challenge reference.point) :=
      held
    have found : L.extraction.protocol.cut 1 reference.point = some deeper := found
    have each : ∀ coordinate ∈ (List.finRange S.blocks).map L.block,
        (L.Accepts (children coordinate).point ∧
          (L.witness (children coordinate).point).Satisfies L.basis L.statement
            (L.challenge (children coordinate).point)) ∧
        L.extraction.protocol.cut 0 (children coordinate).point = some anchor ∧
        ∃ other slot, L.extraction.protocol.cut 1 (children coordinate).point = some other ∧
          L.extraction.protocol.last other =
            Function.update (L.extraction.protocol.last deeper) coordinate slot ∧
          L.decode slot ≠ L.decode (L.extraction.protocol.last deeper coordinate) := each
    obtain ⟨_, rfl⟩ := transcript_cut found
    obtain ⟨reply, answered, _⟩ := accepted.2.1
    have base := last_take_one answered
    have child : ∀ i : Fin S.blocks,
        (L.Accepts (children (L.block i)).point ∧
          (L.witness (children (L.block i)).point).Satisfies L.basis L.statement
            (L.challenge (children (L.block i)).point)) ∧
        ∃ slot, (children (L.block i)).point.reply 0 =
            some (Function.update reply (L.block i) slot) ∧
          L.decode slot ≠ L.decode (reply (L.block i)) := by
      intro i
      obtain ⟨heldChild, _, other, slot, foundChild, last, differs⟩ :=
        each (L.block i) (List.mem_map.mpr ⟨i, List.mem_finRange i, rfl⟩)
      refine ⟨heldChild, slot, ?_, ?_⟩
      · obtain ⟨_, rfl⟩ := transcript_cut foundChild
        obtain ⟨replyChild, answeredChild, _⟩ := heldChild.1.2.1
        rw [last_take_one answeredChild, base] at last
        rw [answeredChild, last]
      · rw [base] at differs
        exact differs
    have forked : (L.fork reference children).Accepts L.basis L.family L.statement := {
      firstMember := fun i => ⟨_, accepted.2.1.challenge answered i⟩
      secondMember := fun i => by
        obtain ⟨⟨acceptedChild, _⟩, slot, answeredChild, _⟩ := child i
        exact ⟨_, acceptedChild.2.1.challenge answeredChild i⟩
      different := fun i same => by
        obtain ⟨⟨acceptedChild, _⟩, slot, answeredChild, differs⟩ := child i
        have second := acceptedChild.2.1.challenge answeredChild i
        rw [Function.update_self] at second
        refine differs (second.trans ?_)
        rw [accepted.2.1.challenge answered i]
        exact congrArg some same.symm
      agree := fun i j other => by
        obtain ⟨⟨acceptedChild, _⟩, slot, answeredChild, _⟩ := child i
        have second := acceptedChild.2.1.challenge answeredChild j
        rw [Function.update_of_ne (fun same => other (valid.injective same)),
          accepted.2.1.challenge answered j] at second
        exact Option.some.inj second
      firstSatisfies := fun _ => satisfied
      secondSatisfies := fun i => (child i).1.2 }
    have result := extract_opens_or_collision valid.units valid.sparse valid.base forked
      satisfied.carried
    show Sum.elim _ _ (L.combine reference children)
    unfold Level.combine
    rcases collision : forkCollision L.statement (L.fork reference children)
        (L.witness reference.point).toCarried with _ | solution
    · rw [collision] at result
      exact ⟨⟨_, rfl, result.resolve_left (by simp)⟩, fun _ _ => rfl⟩
    · rfl
  · intro level span certificate isCheckpoint held isGood
    obtain ⟨rfl, rfl⟩ := checkpoint level span isCheckpoint
    exact ⟨(accepts 2 certificate le_rfl).mp held, isGood⟩
  · intro level span certificate other anchor isCheckpoint _ _ found foundOther different
    obtain ⟨rfl, rfl⟩ := checkpoint level span isCheckpoint
    obtain ⟨_, rfl⟩ := transcript_cut found
    obtain ⟨_, same⟩ := transcript_cut foundOther
    have same : certificate.point.take 1 = other.point.take 1 := same
    refine absurd ?_ different
    show L.witness certificate.point = L.witness other.point
    rw [← witness_take Nat.one_pos certificate.point, same, witness_take Nat.one_pos]
  · intro level span certificate anchor isCheckpoint _ notGood found
    obtain ⟨rfl, rfl⟩ := checkpoint level span isCheckpoint
    obtain ⟨_, rfl⟩ := transcript_cut found
    show (L.game (L.witness certificate.point) (certificate.point.take 1)).Avoids _
    unfold Level.game
    have none : (certificate.point.take 1).rest.drop 1 = [] :=
      List.drop_eq_nil_of_le (List.length_take_le 1 _)
    rw [none, challenge_take Nat.one_pos]
    exact avoids_settle L.read _ ((valid.sound _ _).mono fun result implies passed =>
      notGood (implies passed)) 0
  · intro level span input candidate reply proof isCheckpoint low high refuted outside
    obtain ⟨rfl, rfl⟩ := checkpoint level span isCheckpoint
    have positive : 0 < input.rest.length := low
    have room : input.rest.length < L.stages := by
      have : input.rest.length ≤ 1 + (L.costs.length - 1) := high
      show input.rest.length < L.costs.length + 1
      omega
    show (L.game candidate (L.extraction.protocol.step proof input reply)).Avoids _
    rw [game_step candidate proof reply positive room]
    exact avoids_settle L.read _ (avoids_advance refuted outside) 0
  · intro level span certificate input isCheckpoint held found refuted
    obtain ⟨rfl, rfl⟩ := checkpoint level span isCheckpoint
    obtain ⟨full, _, result, ended, passed⟩ := (accepts 2 certificate le_rfl).mp held
    obtain ⟨_, rfl⟩ := transcript_cut found
    have whole : certificate.point.take (1 + (L.costs.length - 1) + 1) = certificate.point := by
      have length : 1 + (L.costs.length - 1) + 1 = certificate.point.rest.length := by
        rw [full]
        show 1 + (L.costs.length - 1) + 1 = L.costs.length + 1
        omega
      rw [length]
      exact certificate.point.take_length
    have avoided : (L.game (L.witness certificate.point) (certificate.point.take
        (1 + (L.costs.length - 1) + 1))).Avoids
          fun result => ¬ L.passes (L.witness certificate.point) result :=
      refuted
    rw [whole, ended] at avoided
    exact avoided passed
  · intro level span input candidate isCheckpoint low high
    obtain ⟨rfl, rfl⟩ := checkpoint level span isCheckpoint
    have staged := position_staged L.read (valid.staged (L.challenge input) candidate)
      (input.wire 1) (input.rest.drop 1)
    have head : (L.costs.drop (input.rest.drop 1).length).headD 0 =
        L.costs.getD (input.rest.length - 1) 0 := by
      rw [List.length_drop, List.getD_eq_getElem?_getD, List.headD_eq_head?_getD, List.head?_drop]
    have bound := mass_badReplies_le (by
      rw [head]
      exact L.costs_getD_nonnegative valid.nonnegative _) staged
    rw [head] at bound
    exact bound

end Random

end Level

end NightstreamFPrime.Spec.Compression.Akita.Extract
