import NightstreamFPrime.Spec.Compression.Akita.Replay.Sound20Closed
import NightstreamFPrime.Spec.Compression.Akita.Replay.Sound20Digest
import NightstreamFPrime.Spec.Compression.Akita.Replay.StageDigest

/-! The digest positions of a sponge key of `key20.aks`, and the record facts that follow from
"no bad event" (Akita, ePrint 2026/1983, revision of 2026-11-23, Lemma 10.47, the paragraphs
"Simulation" and "Acceptance", with (213); `docs/akita-work/w22/design.md`, the section "The
assembly"; `Akita/Digest.lean`, `Replay/Resolve.lean`, `Replay/SpongeFree.lean`,
`Replay/Sound20Game.lean`, `Replay/Sound20Closed.lean`, `Replay/Sound20Digest.lean`).

- `Simulate.recorded_of_unique`: with no collision, the record of all the digest queries has one
  input for the answer of each digest query.
- `Simulate.recorded_of_fresh`: with no bad event, a digest query whose answer is in a digest
  position of a sponge query is in the record of the digest queries before this sponge query,
  and the record has no other input with this answer.
- `positions20 pub fs`: the digest positions of a query for the table `fs` of the sponge. A key
  with the events of a stage `k` has the digest of the last block of `t` for `379 ≤ k` and the
  digest of the last block of `e` for `472 ≤ k`. `length_positions20`: at most two strings.
  `positions20_local` is the hypothesis `local_` of `Digest.Run.closed_alike`, and
  `positions20_alike` is the hypothesis `alike` of `Simulate.oracleDeps_closed_bad_le`
  (`Replay/Sound20Count.lean`).
- `stage20_of_accepted`, `digestAt20_hash`, `positions20_hit`: for an accepted proof, the key
  of each stage has the stage of the proof, and the digests of the table `hash` for the last
  blocks `Last20` are in its digest positions.
- `record20_of_fresh`: the hypotheses `tListed` and `eListed` of `resolveWith20_eq` for the
  record of the digest queries before the query of the key of a stage. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay.Simulate

open NightstreamFPrime.Spec.Compression.Akita

variable {Fs Y Msg Dig Output : Type}

theorem mem_recorded {hash : Msg → Dig} {queries : List (Fs ⊕ Msg)} {message : Msg}
    {digest : Dig} :
    (message, digest) ∈ recorded hash queries ↔
      Sum.inr message ∈ queries ∧ digest = hash message := by
  unfold recorded
  rw [List.mem_reverse, List.mem_filterMap]
  constructor
  · rintro ⟨query, member, mapped⟩
    cases query with
    | inl point => simp at mapped
    | inr asked =>
        have pair : (asked, hash asked) = (message, digest) := Option.some.inj mapped
        obtain ⟨first, second⟩ := Prod.mk.inj pair
        rw [← first, ← second]
        exact ⟨member, rfl⟩
  · rintro ⟨member, value⟩
    rw [value]
    exact ⟨Sum.inr message, member, rfl⟩

/-- With no collision, the record of all the digest queries has one input for the answer of each
digest query. -/
theorem recorded_of_unique [DecidableEq Msg] {fs : Fs → Y} {hash : Msg → Dig}
    {run : Digest.Run Fs Y Msg Dig Output} (unique : ¬ Digest.Run.Collided fs hash run)
    {message : Msg} (opened : Sum.inr message ∈ run.trace fs hash) :
    (message, hash message) ∈ recorded hash (run.trace fs hash) ∧
      ∀ other, (other, hash message) ∈ recorded hash (run.trace fs hash) → other = message := by
  refine ⟨mem_recorded.mpr ⟨opened, rfl⟩, fun other listed => ?_⟩
  obtain ⟨member, same⟩ := mem_recorded.mp listed
  by_contra different
  exact unique ⟨other, member, message, opened, different, same.symm⟩

/-- With no bad event, a digest query `message` whose answer is in a digest position of a query
`point` to `H_FS` is in the record of the digest queries before `point`, and this record has no
other input with this answer (Akita Lemma 10.47, the paragraph "Acceptance"). -/
theorem recorded_of_fresh [DecidableEq Msg] {fs : Fs → Y} {hash : Msg → Dig}
    {run : Digest.Run Fs Y Msg Dig Output} (positions : Fs ⊕ Msg → List Dig)
    (unique : ¬ Digest.Run.Collided fs hash run)
    (fresh : ¬ Digest.Run.Guessed fs positions hash run) {message : Msg}
    (opened : Sum.inr message ∈ run.trace fs hash) {pre post : List (Fs ⊕ Msg)} {point : Fs}
    (parts : run.trace fs hash = pre ++ Sum.inl point :: post)
    (hit : hash message ∈ positions (.inl point)) :
    (message, hash message) ∈ recorded hash pre ∧
      ∀ other, (other, hash message) ∈ recorded hash pre → other = message := by
  obtain ⟨single, later, split, asked, clean⟩ :=
    Digest.Run.preimage fs positions unique fresh opened
  have early : run.upTo fs hash message <+: pre := by
    refine List.prefix_of_prefix_length_le ⟨later, split.symm⟩ ⟨_, parts.symm⟩ ?_
    by_contra long
    have next : pre ++ [Sum.inl point] <+: run.upTo fs hash message :=
      List.prefix_of_prefix_length_le
        ⟨post, by rw [parts, List.append_assoc, List.singleton_append]⟩ ⟨later, split.symm⟩
        (by rw [List.length_append, List.length_singleton]; omega)
    exact clean (Sum.inl point) (next.subset (by simp)) hit
  refine ⟨mem_recorded.mpr ⟨early.subset asked, rfl⟩, fun other listed => ?_⟩
  obtain ⟨member, same⟩ := mem_recorded.mp listed
  exact single other (by rw [parts]; exact List.mem_append_left _ member) same.symm

end NightstreamFPrime.Spec.Compression.Akita.Replay.Simulate

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

open NightstreamFPrime.Spec.Compression.Lib
open NightstreamFPrime.Spec.Compression.Akita
open Sponge (answerOf spongeOf)
open Replayer

section Positions

variable {L B C S D : ℕ} {pub : Public} {proof : ByteArray} {messages : Array Message}

open scoped Classical in
/-- The string of 32 bytes with the bytes `digest`, and no string when `digest` does not have
32 bytes. -/
noncomputable def bytes32 (digest : ByteArray) : List (Bytes 32) :=
  if found : ∃ u : Bytes 32, Bytes.array u = digest then [Classical.choose found] else []

theorem bytes32_array (u : Bytes 32) : bytes32 (Bytes.array u) = [u] := by
  have found : ∃ v : Bytes 32, Bytes.array v = Bytes.array u := ⟨u, rfl⟩
  unfold bytes32
  rw [dite_eq_left found]
  exact congrArg (fun v => [v]) (Bytes.array_injective (Classical.choose_spec found))

theorem length_bytes32 (digest : ByteArray) : (bytes32 digest).length ≤ 1 := by
  unfold bytes32
  split
  · exact Nat.le_refl 1
  · exact Nat.zero_le 1

/-- The strings in the digest positions of the key of the stage `k` of a proof: the digest of
the last block of `t` for `379 ≤ k`, then the digest of the last block of `e` for `472 ≤ k`. -/
noncomputable def stageDigests20 (pub : Public) (proof : ByteArray) (messages : Array Message)
    (k : ℕ) : List (Bytes 32) :=
  (if 379 ≤ k then bytes32 (tDigestAt pub proof messages k) else []) ++
    if 472 ≤ k then bytes32 (eDigestAt pub proof messages k) else []

theorem length_stageDigests20 (pub : Public) (proof : ByteArray) (messages : Array Message)
    (k : ℕ) : (stageDigests20 pub proof messages k).length ≤ 2 := by
  have first :
      (if 379 ≤ k then bytes32 (tDigestAt pub proof messages k) else []).length ≤ 1 := by
    split
    · exact length_bytes32 _
    · exact Nat.zero_le 1
  have second :
      (if 472 ≤ k then bytes32 (eDigestAt pub proof messages k) else []).length ≤ 1 := by
    split
    · exact length_bytes32 _
    · exact Nat.zero_le 1
  unfold stageDigests20
  rw [List.length_append]
  omega

open scoped Classical in
/-- The strings in the digest positions of the key `base` for the table `fs`: those of a chosen
stage and proof with the events of `base`, and no string when no such stage exists. -/
noncomputable def keyDigests20 (pub : Public) (fs : Sponge.Base L B C → Sponge.Y S C)
    (base : Sponge.Base L B C) : List (Bytes 32) :=
  if found : ∃ stage : ℕ × ByteArray × Array Message,
      Stage20 (pub.reideal (some (answerOf fs))) base.events stage.1 stage.2.1 stage.2.2 then
    stageDigests20 (pub.reideal (some (answerOf fs))) (Classical.choose found).2.1
      (Classical.choose found).2.2 (Classical.choose found).1
  else []

/-- The strings in the digest positions of a query, for the table `fs` of the sponge (Akita
Lemma 10.47, `V_o = 2` in (213)): `keyDigests20` for a key, and no string for a digest
query. -/
noncomputable def positions20 (pub : Public) (fs : Sponge.Base L B C → Sponge.Y S C) :
    Sponge.Base L B C ⊕ Sponge.Str D → List (Bytes 32) :=
  Sum.elim (keyDigests20 pub fs) fun _ => []

theorem positions20_inr (pub : Public) (fs : Sponge.Base L B C → Sponge.Y S C)
    (message : Sponge.Str D) : positions20 pub fs (.inr message) = [] := rfl

theorem length_positions20 (pub : Public) (fs : Sponge.Base L B C → Sponge.Y S C)
    (base : Sponge.Base L B C) : (positions20 (D := D) pub fs (.inl base)).length ≤ 2 := by
  show (keyDigests20 pub fs base).length ≤ 2
  unfold keyDigests20
  split
  · exact length_stageDigests20 _ _ _ _
  · exact Nat.zero_le 2

/-- With a work bit or more in each stage, the digest positions of a key with the events of a
stage of a proof have the digests of this stage and this proof. -/
theorem positions20_stage (bits : Bits20 pub) {fs : Sponge.Base L B C → Sponge.Y S C}
    {base : Sponge.Base L B C} {k : ℕ}
    (stage : Stage20 (pub.reideal (some (answerOf fs))) base.events k proof messages) :
    positions20 (D := D) pub fs (.inl base) =
      stageDigests20 (pub.reideal (some (answerOf fs))) proof messages k := by
  have found : ∃ stage : ℕ × ByteArray × Array Message,
      Stage20 (pub.reideal (some (answerOf fs))) base.events stage.1 stage.2.1 stage.2.2 :=
    ⟨(k, proof, messages), stage⟩
  have chosen := Classical.choose_spec found
  have bits' : Bits20 (pub.reideal (some (answerOf fs))) := bits
  have index : (Classical.choose found).1 = k := stage20_unique bits' chosen stage
  show keyDigests20 pub fs base = _
  unfold keyDigests20
  rw [dite_eq_left found]
  generalize Classical.choose found = w at chosen index ⊢
  obtain ⟨k', proof', messages'⟩ := w
  dsimp only at chosen index ⊢
  subst index
  obtain ⟨t, e⟩ := digestAt_causal chosen stage
  unfold stageDigests20
  rw [t, e]

/-- The digest positions of a key read the table of the sponge only at the keys of `deps20`. -/
theorem positions20_local (bits : Bits20 pub) {fs other : Sponge.Base L B C → Sponge.Y S C}
    {base : Sponge.Base L B C} (same : ∀ key ∈ deps20 S base, fs key = other key) :
    positions20 (D := D) pub fs (.inl base) = positions20 (D := D) pub other (.inl base) := by
  have agree : AgreeOn base.events (answerOf fs) (answerOf other) :=
    agreeOn_answerOf fun before count key sub inside found =>
      same key (mem_depsOf S inside found)
  have ideal : (pub.reideal (some (answerOf fs))).sponge.ideal = some (answerOf fs) := rfl
  by_cases found : ∃ stage : ℕ × ByteArray × Array Message,
      Stage20 (pub.reideal (some (answerOf fs))) base.events stage.1 stage.2.1 stage.2.2
  · obtain ⟨⟨k, proof, messages⟩, staged⟩ := found
    have stage : Stage20 (pub.reideal (some (answerOf fs))) base.events k proof messages := staged
    have moved : Stage20 (pub.reideal (some (answerOf other))) base.events k proof messages :=
      stage20_reideal (pub := pub.reideal (some (answerOf fs))) ideal agree stage
    rw [positions20_stage bits stage, positions20_stage bits moved]
    obtain ⟨inside, -, ran, events⟩ := stage
    have suffix := stateBefore_suffix20 (pub.reideal (some (answerOf fs))) proof messages
      (Nat.le_refl k) inside ran
    rw [events] at suffix
    have digests := digestAt_reideal (pub := pub.reideal (some (answerOf fs))) ideal ran
      (agree.suffix suffix)
    have t : tDigestAt (pub.reideal (some (answerOf other))) proof messages k =
        tDigestAt (pub.reideal (some (answerOf fs))) proof messages k := digests.1
    have e : eDigestAt (pub.reideal (some (answerOf other))) proof messages k =
        eDigestAt (pub.reideal (some (answerOf fs))) proof messages k := digests.2
    unfold stageDigests20
    rw [t, e]
  · have missing : ¬∃ stage : ℕ × ByteArray × Array Message,
        Stage20 (pub.reideal (some (answerOf other))) base.events stage.1 stage.2.1 stage.2.2 := by
      rintro ⟨⟨k, proof, messages⟩, stage⟩
      exact found ⟨(k, proof, messages),
        stage20_of_reideal (pub := pub.reideal (some (answerOf fs))) ideal agree stage⟩
    show keyDigests20 pub fs base = keyDigests20 pub other base
    unfold keyDigests20
    rw [dite_eq_right found, dite_eq_right missing]

/-- The hypothesis `alike` of `Simulate.oracleDeps_closed_bad_le` for `positions20` and each
closed run. -/
theorem positions20_alike {Output : Type} (bits : Bits20 pub)
    (run : Digest.Run (Sponge.Base L B C) (Sponge.Y S C) (Sponge.Str D) (Bytes 32) Output) :
    ∀ (fs other : Sponge.Base L B C → Sponge.Y S C) (hash : Sponge.Str D → Bytes 32),
      (run.closed (deps20 S)).answered fs hash = (run.closed (deps20 S)).answered other hash →
      ∀ query ∈ (run.closed (deps20 S)).trace fs hash,
        positions20 pub fs query = positions20 pub other query :=
  fun fs other hash same =>
    Digest.Run.closed_alike (deps20 S) (deps20_closed S) (positions20 pub)
      (fun _ _ _ equal => positions20_local bits equal) (fun _ _ => rfl) run fs other hash same

end Positions

section Hit

variable {E : Type} [Field E] [Algebra Zq E] [Degree2 E] {L B C S D : ℕ} {zMax rice : ℕ}
  {pubV : Public} {proof : ByteArray} {messages : Array Message}
  {fs : Sponge.Base L B C → Sponge.Y S C} {hash : Sponge.Str D → Bytes 32}
  {tLast : ℕ → Rq Key20.terminal.innerDegree} {eLast : Rq Key20.terminal.innerDegree}

/-- For an accepted proof of `layout20`: the public input of `resolveAt` has the work bits of
the model, and the events before the squeeze of the work of each stage `k < 473` of the run of
the verifier are the events of the stage `k` of the proof for the answers of the table `fs`. -/
theorem stage20_of_accepted (layout : pubV.layout = layout20 zMax rice)
    (accepted : verifyProof (pubV.withOracles fs hash) proof = .verdict .accept) {rest : ℕ}
    (decoded : decode (pubV.withOracles fs hash).sealed.ops proof = .ok (messages, rest))
    {k : ℕ} (inside : k < 473) :
    Bits20 (pubV.withSponge Duplex.empty).sealed ∧
      Stage20 ((pubV.withSponge Duplex.empty).sealed.reideal (some (answerOf fs)))
        (eventsAt20 (pubV.withOracles fs hash).sealed proof messages k) k proof messages := by
  obtain ⟨-, scripted, none_⟩ := replay_of_accepted pubV proof fs hash accepted decoded
  have ran := (replay_none_iff _ _ _ _).mp none_
  have bits : Bits20 (pubV.withOracles fs hash).sealed := bits20_of_scripted layout scripted
  have upto := runFrom_take_ok _ proof messages _
    (stageIndex20 (pubV.withOracles fs hash).sealed k) _ ran
  have layoutR : ((pubV.withSponge Duplex.empty).sealed.reideal
      (some (answerOf fs))).layout = layout20 zMax rice := layout
  have stage : Stage20 (((pubV.withSponge Duplex.empty).sealed.reideal
      (some (answerOf fs))).withDigest (digestOf hash))
      (eventsAt20 (pubV.withOracles fs hash).sealed proof messages k) k proof messages :=
    ⟨inside, ⟨rest, decoded⟩, upto, rfl⟩
  exact ⟨bits, (stage20_withDigest ((pubV.withSponge Duplex.empty).sealed.reideal
    (some (answerOf fs))) (digestOf hash) layoutR _ k proof messages).mp stage⟩

/-- For an accepted proof of `layout20` and last blocks `Last20`: the digest in the state before
the work of each stage `k ≥ 379` has the bytes of the table `hash` at the digest input of the
last block of `t`, and the digest of each stage `k ≥ 472` has those at the digest input of the
last block of `e`. -/
theorem digestAt20_hash (layout : pubV.layout = layout20 zMax rice)
    (accepted : verifyProof (pubV.withOracles fs hash) proof = .verdict .accept)
    (fits : 4128 ≤ D)
    (last : Last20 (E := E) (pubV.withOracles fs hash).sealed proof messages tLast eLast) :
    (∀ k, 379 ≤ k →
      tDigestAt ((pubV.withSponge Duplex.empty).sealed.reideal (some (answerOf fs))) proof
          messages k =
        Bytes.array (hash (strOf (tInput Key20.terminal tLast)
          ((size_tInput20 tLast).trans_le fits)))) ∧
      ∀ k, 472 ≤ k →
        eDigestAt ((pubV.withSponge Duplex.empty).sealed.reideal (some (answerOf fs))) proof
            messages k =
          Bytes.array (hash (strOf (eInput eLast)
            ((size_eInput20 eLast).trans_le (Nat.le_trans (by decide) fits)))) := by
  have ran := ((verifyProof_accept_iff _ _).mp accepted).2
  have layoutA : (pubV.withOracles fs hash).sealed.layout = layout20 zMax rice := layout
  have layoutR : ((pubV.withSponge Duplex.empty).sealed.reideal
      (some (answerOf fs))).layout = layout20 zMax rice := layout
  have digestA : (pubV.withOracles fs hash).sealed.digest = digestOf hash := rfl
  have tA : ∀ k, tDigestAt (pubV.withOracles fs hash).sealed proof messages k =
      tDigestAt ((pubV.withSponge Duplex.empty).sealed.reideal (some (answerOf fs))) proof
        messages k := fun k =>
    tDigestAt_withDigest ((pubV.withSponge Duplex.empty).sealed.reideal (some (answerOf fs)))
      (digestOf hash) proof messages layoutR k
  have eA : ∀ k, eDigestAt (pubV.withOracles fs hash).sealed proof messages k =
      eDigestAt ((pubV.withSponge Duplex.empty).sealed.reideal (some (answerOf fs))) proof
        messages k := fun k =>
    eDigestAt_withDigest ((pubV.withSponge Duplex.empty).sealed.reideal (some (answerOf fs)))
      (digestOf hash) proof messages layoutR k
  obtain ⟨t, e⟩ := digestAt_of_last layoutA ran last
  refine ⟨fun k late => ?_, fun k late => ?_⟩
  · have step := t k late
    rw [digestA, digestOf_short hash ((size_tInput20 tLast).trans_le fits), tA k] at step
    exact step.symm
  · have step := e k late
    rw [digestA, digestOf_short hash
      ((size_eInput20 eLast).trans_le (Nat.le_trans (by decide) fits)), eA k] at step
    exact step.symm

/-- For an accepted proof of `layout20` and last blocks `Last20`: the answer of the digest
oracle for the last block of `t` is in the digest positions of the key of each stage
`379 ≤ k < 473`, and the answer for the last block of `e` is in those of the key of the stage
472. -/
theorem positions20_hit (layout : pubV.layout = layout20 zMax rice)
    (accepted : verifyProof (pubV.withOracles fs hash) proof = .verdict .accept) {rest : ℕ}
    (decoded : decode (pubV.withOracles fs hash).sealed.ops proof = .ok (messages, rest))
    (fits : 4128 ≤ D)
    (last : Last20 (E := E) (pubV.withOracles fs hash).sealed proof messages tLast eLast)
    {k : ℕ} (inside : k < 473) {base : Sponge.Base L B C}
    (events : base.events = eventsAt20 (pubV.withOracles fs hash).sealed proof messages k) :
    (379 ≤ k →
      hash (strOf (tInput Key20.terminal tLast) ((size_tInput20 tLast).trans_le fits)) ∈
        positions20 (D := D) (pubV.withSponge Duplex.empty).sealed fs (.inl base)) ∧
      (472 ≤ k →
        hash (strOf (eInput eLast)
            ((size_eInput20 eLast).trans_le (Nat.le_trans (by decide) fits))) ∈
          positions20 (D := D) (pubV.withSponge Duplex.empty).sealed fs (.inl base)) := by
  obtain ⟨bits, stage⟩ := stage20_of_accepted layout accepted decoded inside
  rw [← events] at stage
  obtain ⟨t, e⟩ := digestAt20_hash layout accepted fits last
  rw [positions20_stage bits stage]
  unfold stageDigests20
  refine ⟨fun late => ?_, fun late => ?_⟩
  · rw [ite_eq_left late, t k late, bytes32_array]
    exact List.mem_append_left _ (List.mem_singleton.mpr rfl)
  · rw [ite_eq_left late, e k late, bytes32_array]
    exact List.mem_append_right _ (List.mem_singleton.mpr rfl)

/-- A record with one input for a digest, as byte arrays. -/
theorem recordOf_single {record : List (Sponge.Str D × Bytes 32)} {message : Sponge.Str D}
    {digest : Bytes 32} (listed : (message, digest) ∈ record)
    (single : ∀ other, (other, digest) ∈ record → other = message) :
    (message.array, Bytes.array digest) ∈ recordOf record ∧
      ∀ input, (input, Bytes.array digest) ∈ recordOf record → input = message.array := by
  refine ⟨List.mem_map.mpr ⟨(message, digest), listed, rfl⟩, fun input member => ?_⟩
  obtain ⟨⟨other, value⟩, inside, mapped⟩ := List.mem_map.mp member
  obtain ⟨first, second⟩ := Prod.mk.inj mapped
  have equal : value = digest := Bytes.array_injective second
  rw [equal] at inside
  exact first.symm.trans (congrArg Sponge.Str.array (single other inside))

/-- The hypotheses `tListed` and `eListed` of `resolveWith20_eq` for the key of a stage of an
accepted proof, with the record of the digest queries before a query of this key: with no bad
event for `positions20`, the record has the digest input of the last block of `t` for
`379 ≤ k` and that of the last block of `e` for `472 ≤ k`, each with the digest of the proof,
and no other input with this digest. -/
theorem record20_of_fresh {Output : Type} (layout : pubV.layout = layout20 zMax rice)
    (accepted : verifyProof (pubV.withOracles fs hash) proof = .verdict .accept) {rest : ℕ}
    (decoded : decode (pubV.withOracles fs hash).sealed.ops proof = .ok (messages, rest))
    (fits : 4128 ≤ D)
    (last : Last20 (E := E) (pubV.withOracles fs hash).sealed proof messages tLast eLast)
    {k : ℕ} (inside : k < 473) {base : Sponge.Base L B C}
    (events : base.events = eventsAt20 (pubV.withOracles fs hash).sealed proof messages k)
    {run : Digest.Run (Sponge.Base L B C) (Sponge.Y S C) (Sponge.Str D) (Bytes 32) Output}
    (unique : ¬ Digest.Run.Collided fs hash run)
    (fresh : ¬ Digest.Run.Guessed fs (positions20 (pubV.withSponge Duplex.empty).sealed fs)
      hash run)
    (tAsked : Sum.inr (strOf (tInput Key20.terminal tLast)
      ((size_tInput20 tLast).trans_le fits)) ∈ run.trace fs hash)
    (eAsked : Sum.inr (strOf (eInput eLast)
      ((size_eInput20 eLast).trans_le (Nat.le_trans (by decide) fits))) ∈ run.trace fs hash)
    {pre post : List (Sponge.Base L B C ⊕ Sponge.Str D)}
    (parts : run.trace fs hash = pre ++ Sum.inl base :: post) :
    (379 ≤ k →
      (tInput Key20.terminal tLast,
          tDigestAt ((pubV.withSponge Duplex.empty).sealed.reideal (some (answerOf fs))) proof
            messages k) ∈ recordOf (Simulate.recorded hash pre) ∧
        ∀ input,
          (input, tDigestAt ((pubV.withSponge Duplex.empty).sealed.reideal
            (some (answerOf fs))) proof messages k) ∈ recordOf (Simulate.recorded hash pre) →
          input = tInput Key20.terminal tLast) ∧
      (472 ≤ k →
        (eInput eLast,
            eDigestAt ((pubV.withSponge Duplex.empty).sealed.reideal (some (answerOf fs))) proof
              messages k) ∈ recordOf (Simulate.recorded hash pre) ∧
          ∀ input,
            (input, eDigestAt ((pubV.withSponge Duplex.empty).sealed.reideal
              (some (answerOf fs))) proof messages k) ∈ recordOf (Simulate.recorded hash pre) →
            input = eInput eLast) := by
  obtain ⟨tHit, eHit⟩ := positions20_hit layout accepted decoded fits last inside events
  obtain ⟨t, e⟩ := digestAt20_hash layout accepted fits last
  refine ⟨fun late => ?_, fun late => ?_⟩
  · obtain ⟨listed, single⟩ :=
      Simulate.recorded_of_fresh _ unique fresh tAsked parts (tHit late)
    have pair := recordOf_single listed single
    rw [array_strOf] at pair
    rw [t k late]
    exact pair
  · obtain ⟨listed, single⟩ :=
      Simulate.recorded_of_fresh _ unique fresh eAsked parts (eHit late)
    have pair := recordOf_single listed single
    rw [array_strOf] at pair
    rw [e k late]
    exact pair

end Hit

end NightstreamFPrime.Spec.Compression.Akita.Replay
