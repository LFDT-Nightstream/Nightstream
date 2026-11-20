import NightstreamFPrime.Spec.Compression.Akita.Terminal
import NightstreamFPrime.Spec.Compression.Akita.Extract.Recursion

/-! The evaluation-trace terminal of Akita as the last stages of the opening suffix (ePrint
2026/1983, revision of 2026-10-15, Section 8.2, Proposition 10.29, and Proposition 10.44 with its
proof, "Certificates", "Terminal fold" and "Checkpoints").

- `Terminal` has the data of the terminal with its fold challenge at the stage `start` of a
  transcript. The stage input of the stage `start` gives the public data and the clear partials
  `(e_i)_i`, and the message after the fold challenge has the clear response `z`.
- `Terminal.Passed` is the checks (128), (129) and (130) with the bound of `z`, and
  `Terminal.Sourced` is the certificate of the terminal at its fold challenge: two accepting
  children for each block. `Terminal.Sourced.source` gives the source blocks `s⋆_i` with
  `A_term·s⋆_i = t_i`, the clear partials `e_i = a_term^⊤·G·s⋆_i` and the row (130) of these blocks.
- `Terminal.body` is the checkpoint structure of the terminal: the stage `start` is an indexed
  stage, and each later stage is plain. `Terminal.suffix` states that it is a suffix with the
  stage error `ε_term = B_term·p` of (183), `B_term` retry loops and the tree size `1 + B_term`.
- `Terminal.collide` is `Coll_ℓ` for a level before the terminal, and `Terminal.suffix_after` is
  the suffix of a level over the terminal: two certificates of the terminal with different source
  blocks give the solution `terminalCollision` for `A_term` with the radius (181). -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Extract

open scoped BigOperators
open NightstreamFPrime.Spec.Checkpoint

attribute [local instance low] Classical.propDecidable

theorem Prefix.take_take_self {Wire Reply : Type} {stages : ℕ} (level : ℕ)
    (point : Prefix Wire Reply stages) : (point.take level).take level = point.take level :=
  Prefix.ext rfl (by simp [Prefix.take, List.take_take])

/-- The data of the terminal with its fold challenge at the stage `start` (Akita Section 8.2).
`matrix` is `A_term`. `read` gives the other public data from the stage input of the stage
`start`: the state `(t_i)_i` of the predecessor, the weights and the pair `(θ_term, v_tensor)`.
`partials` gives the clear partials `(e_i)_i` of this stage input, and `response` gives the clear
response `z` of the message after the fold challenge. `block i` is the coordinate of the slot of
the fold challenge `c_i`, and `decode` gives the challenge of a slot. `weight` is `κ_{1,term}` and
`bound` is `Z_{∞,term}`. -/
structure Terminal (S : Shape) (E Wire Coord Slot : Type) [CommRing E] [Module Zq E]
    (total : ℕ) where
  matrix : Matrix (Fin S.innerRank) (Fin S.blockLength × Fin S.sourceDigits) (Rq S.innerDegree)
  read : Prefix Wire (Coord → Slot) total → TerminalStatement S E
  partials : Prefix Wire (Coord → Slot) total → Fin S.blocks → Rq S.innerDegree
  response : Wire → S.Source
  block : Fin S.blocks → Coord
  decode : Slot → Option (Rq S.innerDegree)
  weight : ℕ
  bound : ℕ
  start : ℕ

namespace Terminal

variable {S : Shape} {E Wire Coord Slot : Type} [CommRing E] [Module Zq E] {total : ℕ}
  (L : Terminal S E Wire Coord Slot total)

/-- The set of the fold challenges `C^acc_term`: the values of the decoder. -/
def family : Set (Rq S.innerDegree) := {challenge | ∃ slot, L.decode slot = some challenge}

/-- The statement of the terminal for a transcript: the matrix `A_term`, and the public data of
the stage input of the stage `start`. -/
def statement (point : Prefix Wire (Coord → Slot) total) : TerminalStatement S E :=
  { L.read (point.take L.start) with matrix := L.matrix }

/-- The clear partials `(e_i)_i` of a transcript. -/
def clear (point : Prefix Wire (Coord → Slot) total) : Fin S.blocks → Rq S.innerDegree :=
  L.partials (point.take L.start)

/-- The fold challenges `c_i` of a transcript. -/
noncomputable def challenge (point : Prefix Wire (Coord → Slot) total) :
    Fin S.blocks → Rq S.innerDegree :=
  challengeAt L.decode L.block L.start point

/-- The checks of the verifier of the terminal on a transcript (Akita Section 8.2): each fold
challenge decodes, the checks (128), (129) and (130) hold for the clear partials, the fold
challenges and the clear response `z`, and `‖z‖_∞ ≤ Z_{∞,term}`. -/
structure Passed (point : Prefix Wire (Coord → Slot) total) : Prop where
  decodes : DecodesAt L.decode L.block L.start point
  accepts : (L.statement point).Accepts (L.clear point) (L.challenge point)
    (L.response (point.wire (L.start + 1)))
  short : vecNormInf (L.response (point.wire (L.start + 1))) ≤ L.bound

/-- The certificate of the terminal at its fold challenge (Akita Proposition 10.44,
"Certificates"): two children for each block that pass the checks of the terminal for the clear
partials of the transcript, with responses of the norm `Z_{∞,term}` at most, and the row (130) of
the clear partials. The source blocks are `TerminalFork.source`, and the children give their unit
challenge differences and bounded products. -/
def Sourced (point : Prefix Wire (Coord → Slot) total) (witness : Option (TerminalFork S)) :
    Prop :=
  ∃ fork, witness = some fork ∧ fork.Accepts L.family (L.statement point) (L.clear point) ∧
    (∀ i, vecNormInf (fork.first i).response ≤ L.bound ∧
      vecNormInf (fork.second i).response ≤ L.bound) ∧
    (L.statement point).scale * ∑ i, (L.statement point).block i *
      (L.statement point).functional (L.clear point i) = (L.statement point).value

variable {L} in
/-- Akita Proposition 10.29 for a certificate of the terminal: the source blocks `s⋆_i` have
`A_term·s⋆_i = t_i`, each clear partial is `e_i = a_term^⊤·G_{b_term,M_term}·s⋆_i`, and the row (130)
holds with the partials of the source blocks, which is the condition of Akita Lemma 10.42 (ii) at
the terminal. -/
theorem Sourced.source {point : Prefix Wire (Coord → Slot) total} {fork : TerminalFork S}
    (units : UnitDifferences L.family) (sourced : L.Sourced point (some fork)) :
    (∀ i, L.matrix.mulVec (fork.source i) = (L.statement point).state i) ∧
      (∀ i, L.clear point i = traceContract (L.statement point).subfield
        (L.statement point).position (gadget S.sourceBase (fork.source i))) ∧
      (L.statement point).scale * ∑ i, (L.statement point).block i *
        (L.statement point).functional (traceContract (L.statement point).subfield
          (L.statement point).position (gadget S.sourceBase (fork.source i))) =
        (L.statement point).value := by
  obtain ⟨other, same, accepts, _, value⟩ := sourced
  obtain rfl := Option.some.inj same
  have partials := fun i => (terminal_partial units accepts i).symm
  refine ⟨terminal_inner units accepts, fun i => (partials i).symm, ?_⟩
  simp only [partials]
  exact value

/-- The source blocks of a certificate of the terminal. -/
noncomputable def blocksOf : Option (TerminalFork S) → Fin S.blocks → S.Source
  | some fork => fork.source
  | none => 0

/-- `Coll_ℓ` for a level before the terminal (Akita Proposition 10.44, "Checkpoints"): the
solution `z_A` of `terminalCollision` for the first block where it is a solution for `A_term`
with the radius `8·κ_{1,term}·Z_{∞,term}` of Akita (181). -/
noncomputable def collide (witness other : Option (TerminalFork S)) :
    Option (ShortKernel L.matrix (8 * L.weight * L.bound)) :=
  match witness, other with
  | some fork, some fork' => (List.finRange S.blocks).findSome? fun i =>
      ShortKernel.ofCollision? L.matrix (8 * L.weight * L.bound)
        (fork'.difference i • fork.responseDifference i)
        (fork.difference i • fork'.responseDifference i)
  | _, _ => none

/-- The children of the extraction at the fold challenge of the terminal: the reference
transcript is the first child of each block, and the transcript of the coordinate of block `i` is
its second child (Akita Proposition 10.29, proof). -/
noncomputable def fork (reference : Prefix Wire (Coord → Slot) total)
    (children : Coord → Prefix Wire (Coord → Slot) total) : TerminalFork S where
  first _ := ⟨L.challenge reference, L.response (reference.wire (L.start + 1))⟩
  second i := ⟨L.challenge (children (L.block i)),
    L.response ((children (L.block i)).wire (L.start + 1))⟩

variable [Fintype Slot] [Nonempty Slot] (accept : Prefix Wire (Coord → Slot) total → Prop)

/-- The checkpoint structure of the terminal (Akita Definition 10.16 for the stages from
`start`). The stage `start` is indexed by the blocks, and `Comb_r` is the subtraction step of
Akita Proposition 10.29, which returns the certificate `Sourced` of the reference transcript. Each
later stage is plain, and a certificate after the stage `start` is an accepting transcript. A
witness of `R_bind` is a solution of Module-SIS for `A_term`. -/
noncomputable def body :
    Structure (Prefix Wire (Coord → Slot) total) Coord Slot (ℕ → Wire)
      (Prefix Wire (Coord → Slot) total × Option (TerminalFork S))
      (Option (ShortKernel L.matrix (8 * L.weight * L.bound))) Unit (Rq S.innerDegree)
      (Option (TerminalFork S)) where
  protocol := transcript total accept
  kind level :=
    if level ≤ L.start then .indexed ((List.finRange S.blocks).map L.block) else .plain
  key _ _ slot := L.decode slot
  weight _ _ := slotChance L.decode
  error _ := 0
  leaf point := (point, none)
  point certificate := certificate.1
  holds level certificate :=
    accept certificate.1 ∧ (level ≤ L.start → L.Sourced certificate.1 certificate.2)
  breaks solution := solution.isSome
  combine _ reference children :=
    .inl (reference.1, some (L.fork reference.1 fun coordinate => (children coordinate).1))
  candidate _ _ := ()
  good _ _ := True
  collide _ _ _ := none
  refuted _ _ _ := False
  bad _ _ _ := ∅
  witness certificate := certificate.2
  relation := L.Sourced

/-- The conditions on the terminal. `injective`: the blocks have different coordinates. `fits`:
the stage `start` is a stage of the transcript. `checked`: the verifier accepts only a transcript
that passes the checks of the terminal. -/
structure Valid : Prop where
  injective : Function.Injective L.block
  fits : L.start < total
  checked : ∀ point, accept point → L.Passed point

variable {accept}

theorem body_kind_start :
    (L.body accept).kind L.start = .indexed ((List.finRange S.blocks).map L.block) :=
  ite_eq_left le_rfl

theorem body_kind_after {level : ℕ} (above : L.start < level) :
    (L.body accept).kind level = .plain :=
  ite_eq_right (Nat.not_le.mpr above)

variable [Fintype Coord] [DecidableEq Coord]

/-- The structure of the terminal satisfies the conditions of Akita Definition 10.16 from the
stage `start` (Akita Proposition 10.44, "Terminal fold", with Proposition 10.29). -/
theorem body_lawfulFrom (valid : L.Valid accept) :
    (L.body accept).LawfulFrom (transcript total accept) L.start := by
  have indexed : ∀ {level : ℕ} {coordinates : List Coord}, L.start ≤ level →
      (L.body accept).kind level = .indexed coordinates →
      level = L.start ∧ coordinates = (List.finRange S.blocks).map L.block := by
    intro level coordinates low isIndexed
    by_cases first : level ≤ L.start
    · have value : (L.body accept).kind level =
          .indexed ((List.finRange S.blocks).map L.block) := ite_eq_left first
      exact ⟨Nat.le_antisymm first low, Kind.indexed.inj (isIndexed.symm.trans value)⟩
    · have value : (L.body accept).kind level = .plain := ite_eq_right first
      exact absurd (value.symm.trans isIndexed) (by simp)
  have unchecked : ∀ {level span : ℕ}, (L.body accept).kind level ≠ .checkpoint span := by
    intro level span isCheckpoint
    have value : (if level ≤ L.start then Kind.indexed ((List.finRange S.blocks).map L.block)
        else Kind.plain) = .checkpoint span := isCheckpoint
    split at value <;> exact absurd value (by simp)
  refine {
    segment := fun _ _ _ isCheckpoint => absurd isCheckpoint unchecked
    key_some := ?_
    key_mass := fun _ _ _ value _ _ _ => mass_slot_le L.decode value
    weight_nonnegative := fun _ _ => slotChance_nonnegative L.decode
    leaf_point := fun _ => rfl
    leaf_holds := fun _ accepted =>
      ⟨accepted, fun (high : total ≤ L.start) => absurd high (Nat.not_le.mpr valid.fits)⟩
    plain_holds := fun level certificate _ isPlain held => ⟨held.1, fun first =>
      absurd ((ite_eq_left first : (L.body accept).kind level = _).symm.trans isPlain) (by simp)⟩
    combine_holds := ?_
    good_holds := fun _ _ _ _ isCheckpoint => absurd isCheckpoint unchecked
    collide_breaks := fun _ _ _ _ _ _ isCheckpoint => absurd isCheckpoint unchecked
    refuted_start := fun _ _ _ _ _ isCheckpoint => absurd isCheckpoint unchecked
    refuted_step := fun _ _ _ _ _ _ _ isCheckpoint => absurd isCheckpoint unchecked
    refuted_end := fun _ _ _ _ _ isCheckpoint => absurd isCheckpoint unchecked
    bad_mass := fun _ _ _ _ _ isCheckpoint => absurd isCheckpoint unchecked
    error_nonnegative := fun _ => le_rfl }
  · intro level coordinates coordinate point deeper low isIndexed member accepted found
    obtain ⟨rfl, rfl⟩ := indexed low isIndexed
    obtain ⟨i, _, rfl⟩ := List.mem_map.mp member
    obtain ⟨_, rfl⟩ := transcript_cut found
    obtain ⟨reply, answered, each⟩ := (valid.checked point accepted).decodes
    show (L.decode ((transcript total accept).last (point.take (L.start + 1)) (L.block i))).isSome
    rw [last_take_succ answered]
    exact each i
  · intro level coordinates reference family anchor deeper low isIndexed held found foundDeeper
      each
    obtain ⟨rfl, rfl⟩ := indexed low isIndexed
    have accepted : accept reference.1 := held.1
    have passed := valid.checked reference.1 accepted
    obtain ⟨_, rfl⟩ := transcript_cut found
    obtain ⟨_, rfl⟩ := transcript_cut foundDeeper
    obtain ⟨reply, answered, _⟩ := passed.decodes
    have base := last_take_succ (accept := accept) answered
    have child : ∀ i : Fin S.blocks,
        (accept (family (L.block i)).1 ∧
          (family (L.block i)).1.take L.start = reference.1.take L.start) ∧
        ∃ slot, (family (L.block i)).1.reply L.start =
            some (Function.update reply (L.block i) slot) ∧
          L.decode slot ≠ L.decode (reply (L.block i)) := by
      intro i
      obtain ⟨heldChild, foundChild, other, slot, foundOther, last, differs⟩ :=
        each (L.block i) (List.mem_map.mpr ⟨i, List.mem_finRange i, rfl⟩)
      have acceptedChild : accept (family (L.block i)).1 := heldChild.1
      obtain ⟨_, sameAnchor⟩ := transcript_cut foundChild
      refine ⟨⟨acceptedChild, sameAnchor.symm⟩, slot, ?_, ?_⟩
      · obtain ⟨_, rfl⟩ := transcript_cut foundOther
        obtain ⟨replyChild, answeredChild, _⟩ := (valid.checked _ acceptedChild).decodes
        have last : (transcript total accept).last ((family (L.block i)).1.take (L.start + 1)) =
            Function.update ((transcript total accept).last (reference.1.take (L.start + 1)))
              (L.block i) slot := last
        rw [last_take_succ answeredChild, base] at last
        rw [answeredChild, last]
      · have differs : L.decode slot ≠ L.decode
            ((transcript total accept).last (reference.1.take (L.start + 1)) (L.block i)) :=
          differs
        rw [base] at differs
        exact differs
    have same : ∀ i, L.statement (family (L.block i)).1 = L.statement reference.1 ∧
        L.clear (family (L.block i)).1 = L.clear reference.1 := fun i => by
      unfold Terminal.statement Terminal.clear
      rw [(child i).1.2]
      exact ⟨rfl, rfl⟩
    refine ⟨⟨accepted, fun _ => ⟨_, rfl, {
      firstMember := fun i => ⟨_, passed.decodes.challenge answered i⟩
      secondMember := fun i => by
        obtain ⟨⟨acceptedChild, _⟩, slot, answeredChild, _⟩ := child i
        exact ⟨_, (valid.checked _ acceptedChild).decodes.challenge answeredChild i⟩
      different := fun i same => by
        obtain ⟨⟨acceptedChild, _⟩, slot, answeredChild, differs⟩ := child i
        have second := (valid.checked _ acceptedChild).decodes.challenge answeredChild i
        rw [Function.update_self] at second
        refine differs (second.trans ?_)
        rw [passed.decodes.challenge answered i]
        exact congrArg some same.symm
      agree := fun i j other => by
        obtain ⟨⟨acceptedChild, _⟩, slot, answeredChild, _⟩ := child i
        have second := (valid.checked _ acceptedChild).decodes.challenge answeredChild j
        rw [Function.update_of_ne (fun same => other (valid.injective same)),
          passed.decodes.challenge answered j] at second
        exact Option.some.inj second
      firstAccepts := fun _ => passed.accepts
      secondAccepts := fun i => by
        have accepts := (valid.checked _ (child i).1.1).accepts
        rw [(same i).1, (same i).2] at accepts
        exact accepts },
      fun i => ⟨passed.short, (valid.checked _ (child i).1.1).short⟩, passed.accepts.value⟩⟩,
      fun _ _ => rfl⟩

omit [Fintype Coord] [DecidableEq Coord] in
/-- The stage error of the terminal: `ε_term = B_term·p` of Akita (183), for the largest
probability `p` of one fold challenge, which is `1/|C^acc_term|` for a uniform decoder. -/
theorem body_chargeFrom (fits : L.start < total) :
    (L.body accept).chargeFrom L.start = S.blocks * slotChance L.decode := by
  have first : (L.body accept).charge L.start = S.blocks * slotChance L.decode := by
    unfold Structure.charge
    rw [ite_eq_left (show L.start < (L.body accept).protocol.stages from fits), L.body_kind_start]
    show (((List.finRange S.blocks).map L.block).map fun _ => slotChance L.decode).sum = _
    simp [Function.comp_def]
  have rest : ∀ level ∈ Finset.Ico (L.start + 1) total, (L.body accept).charge level = 0 := by
    intro level member
    unfold Structure.charge
    rw [L.body_kind_after (Nat.lt_of_succ_le (Finset.mem_Ico.mp member).1)]
    split <;> rfl
  unfold Structure.chargeFrom
  show ∑ later ∈ Finset.Ico L.start total, (L.body accept).charge later = _
  rw [Finset.sum_eq_sum_Ico_succ_bot fits, first, Finset.sum_eq_zero rest, add_zero]

omit [Fintype Coord] [DecidableEq Coord] in
/-- The extractor has one retry loop for each block of the terminal. -/
theorem body_loopsFrom (fits : L.start < total) : (L.body accept).loopsFrom L.start = S.blocks := by
  have first : (L.body accept).loops L.start = S.blocks := by
    unfold Structure.loops
    rw [L.body_kind_start]
    show ((List.finRange S.blocks).map L.block).length = _
    simp
  have rest : (L.body accept).loopsFrom (L.start + 1) = 0 :=
    Finset.sum_eq_zero fun level member => by
      unfold Structure.loops
      rw [L.body_kind_after (Nat.lt_of_succ_le (Finset.mem_Ico.mp member).1)]
  rw [Structure.loopsFrom_succ _ (show L.start < (L.body accept).protocol.stages from fits),
    first, rest, Nat.add_zero]

omit [Fintype Coord] [DecidableEq Coord] in
/-- The tree size of the terminal: the fold factor `1 + B_term` of Akita (197). -/
theorem body_treeFrom (fits : L.start < total) :
    (L.body accept).treeFrom L.start = S.blocks + 1 := by
  have first : (L.body accept).arity L.start = 1 + S.blocks := by
    unfold Structure.arity
    rw [L.body_kind_start]
    show 1 + ((List.finRange S.blocks).map L.block).length = _
    simp
  have rest : (L.body accept).treeFrom (L.start + 1) = 1 :=
    Finset.prod_eq_one fun level member => by
      unfold Structure.arity
      rw [L.body_kind_after (Nat.lt_of_succ_le (Finset.mem_Ico.mp member).1)]
  rw [Structure.treeFrom_succ _ (show L.start < (L.body accept).protocol.stages from fits),
    first, rest, Nat.mul_one, Nat.add_comm]

/-- The terminal is a suffix (Akita Proposition 10.44 for the stages from the fold challenge of
the terminal): a first certificate has an accepting transcript and the certificate `Sourced`, the
stage error is `B_term·p`, the extractor has `B_term` retry loops, and the tree size is
`1 + B_term`. -/
theorem suffix (valid : L.Valid accept) :
    Tier.Suffix accept L.start L.Sourced (L.body accept) (S.blocks * slotChance L.decode)
      S.blocks (S.blocks + 1) where
  opened _ held := ⟨held.1, held.2 le_rfl⟩
  rooted point witness sourced := by
    unfold Sourced Terminal.statement Terminal.clear at sourced ⊢
    rwa [Prefix.take_take_self]
  lawfulFrom := L.body_lawfulFrom valid
  protocol := rfl
  related _ _ := Iff.rfl
  errorSum := L.body_chargeFrom valid.fits
  loopCount := L.body_loopsFrom valid.fits
  treeSize := L.body_treeFrom valid.fits

variable {L}

/-- The suffix of a level over the terminal (Akita Proposition 10.44 for the last nonterminal
level). The successor source of a certificate of the terminal is `decoder` of its source blocks.
Two certificates with different sources, for two accepting transcripts with one stage input of
the checkpoint, have one state `(t_i)_i` (`bound`), and `terminalCollision_isSome` gives a solution
of Module-SIS for `A_term` with the radius `8·κ_{1,term}·Z_{∞,term}` of Akita (181). `units` and
`sparse`: the challenge family of the terminal has units as differences and members of the norm
`κ_{1,term}`. `decodes`: the verifier rejects an answer without a fold challenge for each block of
the level. `passed`: the final claim of the checks of the level is true for the source of a
certificate of the terminal (Akita Lemma 10.42 (ii)). The stage errors are
`r·p + Σ_j ε_j + B_term·p_term`, the retry loops are `r + 1 + B_term`, and the tree size is
`2·(r + 1)·(1 + B_term)`, for `r` blocks of the level. -/
theorem suffix_after {S' : Shape} {O : OpeningShape S'} {fold : Descriptor}
    {Ring Stmt Brk Result Source : Type} [Zero Ring]
    {T : Tier O fold Ring Stmt Brk Wire Coord Slot Result Source} (valid : T.Valid)
    (terminal : L.Valid accept) (next : L.start = T.finish) {peak : ℕ}
    (units : UnitDifferences L.family)
    (sparse : ∀ challenge ∈ L.family, Sparse L.weight peak challenge)
    (decoder : (Fin S.blocks → S.Source) → Source)
    (bound : ∀ point other : Prefix Wire (Coord → Slot) total, accept point → accept other →
      point.take (T.start + 1) = other.take (T.start + 1) →
      (L.read (other.take L.start)).state = (L.read (point.take L.start)).state)
    (decodes : ∀ point, accept point → T.Decodes point)
    (passed : ∀ point fork, accept point → L.Sourced point (some fork) →
      T.Passed (decoder fork.source) point) :
    Tier.Suffix accept T.start T.Opens
      (T.over accept (L.body accept)
        (fun inner => decoder (blocksOf ((L.body accept).witness inner)))
        (fun inner other =>
          L.collide ((L.body accept).witness inner) ((L.body accept).witness other)))
      (S'.blocks * T.chance + T.costs.sum + S.blocks * slotChance L.decode)
      (S'.blocks + 1 + S.blocks) (2 * (S'.blocks + 1) * (S.blocks + 1)) := by
  refine Tier.suffix_before valid (next ▸ L.suffix terminal)
    (next ▸ Nat.le_of_lt terminal.fits) (fun witness => decoder (blocksOf witness))
    L.collide decodes ?_ ?_
  · intro point witness accepted sourced
    obtain ⟨fork, rfl, _⟩ := id sourced
    exact passed point fork accepted sourced
  · rintro point other _ _ accepted accepted' ⟨fork, rfl, accepts, short, _⟩
      ⟨fork', rfl, accepts', short', _⟩ same different
    obtain ⟨i, differs⟩ := Function.ne_iff.mp fun equal : fork.source = fork'.source =>
      different (congrArg decoder equal)
    exact List.findSome?_isSome_iff.mpr ⟨i, List.mem_finRange i,
      terminalCollision_isSome units sparse accepts accepts' rfl
        (bound point other accepted accepted' same) short
        short' differs⟩

end Terminal

end NightstreamFPrime.Spec.Compression.Akita.Extract
