import NightstreamFPrime.Spec.Compression.Akita.Replay.VerifierRun

/-! The queries of the byte verifier `verifierRun` for `key20.aks` (Akita revision `d50f700`;
ePrint 2026/1983, revision of 2026-11-23, Section 8.2 `[KECCAK]` and Lemma 10.47;
`Replay/VerifierRun.lean`; `docs/akita-work/w21/design.md`, sections 2 and 5).

- `askCount20` and `digestCount20`: the skeleton of `layout20` has 473 works, each draw directly
  after a work or a draw, and one terminal. `work_bits20`: no work of a script of `layout20` has
  zero bits.
- `verifierRun_bound20`: for `layout20`, `verifierRun` has at most 473 queries of the sponge and
  2 queries of the digest oracle.
- `size_digestInput`, `size_eInput`, `size_tInput`: the bytes that the terminal hashes have a
  closed size. For `Key20.terminal` they have 544 bytes (the last block of `e`) and 4128 bytes
  (the last block of `t`): `size_eInput20`, `size_tInput20`. `size_eBytes` and `size_tBytes` are
  these sizes for the blocks of the code (`lastBlocks`).
- `lastBlocks_of_ok`: a terminal that accepts has last blocks. `lastBlocks_unique`: last blocks
  that pass the checks (128) and (129) of the terminal are the last blocks of the code, so their
  bytes are `eBytes` and `tBytes` of the blocks of the code.
- `verifierRun_asked20`: for an accepted proof of `layout20`, the events before the work of each
  stage `k < 473` are the events of a base, and this base is a query of `verifierRun`.
- `verifierRun_bases20`: these 473 bases are the queries of `verifierRun` to the sponge, each
  one time.
- `verifierRun_digests20`: for an accepted proof of `layout20` and last blocks with `Last20`,
  `eInput eLast` and `tInput Key20.terminal tLast` are queries of `verifierRun`, when the digest
  oracle takes 4128 bytes. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

open NightstreamFPrime.Spec.Compression.Lib
open NightstreamFPrime.Spec.Compression.Akita
open Sponge (split answerOf spongeOf)
open Replayer

section Counts

theorem isWork_erase (op : Op) : isWork op.erase = isWork op := by cases op <;> rfl

theorem Op.isTerminal_erase (op : Op) : op.erase.isTerminal = op.isTerminal := by
  cases op <;> rfl

theorem Verifier.askCount_erase : ∀ (l : List (Op × ℕ)) (covered : Bool),
    Verifier.askCount (l.map eraseAt) covered = Verifier.askCount l covered
  | [], _ => rfl
  | x :: l, covered => by
      show (if isWork x.1.erase then Verifier.askCount (l.map eraseAt) true + 1
        else if isDraw x.1.erase then
          Verifier.askCount (l.map eraseAt) true + (if covered then 0 else 1)
        else Verifier.askCount (l.map eraseAt) false) =
        (if isWork x.1 then Verifier.askCount l true + 1
        else if isDraw x.1 then Verifier.askCount l true + (if covered then 0 else 1)
        else Verifier.askCount l false)
      rw [isWork_erase, isDraw_erase, Verifier.askCount_erase l true,
        Verifier.askCount_erase l false]

theorem Verifier.digestCount_erase : ∀ l : List (Op × ℕ),
    Verifier.digestCount (l.map eraseAt) = Verifier.digestCount l
  | [] => rfl
  | x :: l => by
      show Verifier.digestCount (l.map eraseAt) + (if x.1.erase.isTerminal then 2 else 0) =
        Verifier.digestCount l + (if x.1.isTerminal then 2 else 0)
      rw [Op.isTerminal_erase, Verifier.digestCount_erase l]

/-- The skeleton of `layout20` has 473 works, and each draw is directly after a work or a
draw. -/
theorem askCount20 : Verifier.askCount sk20 false = 473 := by decide +kernel

/-- The skeleton of `layout20` has one terminal. -/
theorem digestCount20 : Verifier.digestCount sk20 = 2 := by decide +kernel

/-- No work of a script of `layout20` has zero bits. -/
theorem work_bits20 {zMax rice : ℕ} {plan : Array Run} {ops : Array Op}
    (scripted : script (layout20 zMax rice) plan = .ok ops) :
    ∀ x ∈ ops.zipIdx.toList, ∀ site level, x.1 ≠ .work site level 0 := by
  intro x member site level work
  rw [Array.toList_zipIdx] at member
  have listed : x.1 ∈ ops.toList := List.mem_of_getElem? (List.mem_zipIdx_iff_getElem?.mp member)
  have zero : 0 ∈ ops.toList.flatMap Op.bits :=
    List.mem_flatMap.mpr ⟨x.1, listed, by rw [work]; exact List.mem_singleton.mpr rfl⟩
  rw [script_bits scripted, planBits20] at zero
  exact absurd (bits20_pos 0 zero) (by decide)

variable {L B C S D : ℕ}

/-- For `layout20`, the verifier has at most 473 queries of the sponge and 2 queries of the
digest oracle. -/
theorem verifierRun_bound20 (pub : Public) (proof : ByteArray) {zMax rice : ℕ}
    (layout : pub.layout = layout20 zMax rice) :
    (verifierRun (L := L) (B := B) (C := C) (S := S) (D := D) pub proof).Bound 473 2 := by
  refine verifierRun_bound pub proof fun ops scripted => ?_
  rw [layout] at scripted
  have erased : ops.toList.map Op.erase = skeleton (layout20 0 0) := by
    rw [script_erase scripted, skeleton20]
  refine ⟨work_bits20 scripted, ?_, ?_⟩
  · rw [← Verifier.askCount_erase, zipIdx_erase erased]
    exact Nat.le_of_eq askCount20
  · rw [← Verifier.digestCount_erase, zipIdx_erase erased]
    exact Nat.le_of_eq digestCount20

end Counts

section Sizes

theorem size_foldl_bytes {α : Type} (g : α → ByteArray) (n : ℕ) :
    ∀ (l : List α) (init : ByteArray), (∀ a ∈ l, (g a).size = n) →
      (l.foldl (fun out a => out ++ g a) init).size = init.size + l.length * n
  | [], _, _ => by simp
  | a :: l, init, sized => by
      rw [List.foldl_cons, size_foldl_bytes g n l _ fun x member =>
        sized x (List.mem_cons_of_mem a member), ByteArray.size_append,
        sized a List.mem_cons_self, List.length_cons]
      ring

theorem size_foldl_array {α β : Type} (g : α → Array β) (n : ℕ) :
    ∀ (l : List α) (init : Array β), (∀ a ∈ l, (g a).size = n) →
      (l.foldl (fun out a => out ++ g a) init).size = init.size + l.length * n
  | [], _, _ => by simp
  | a :: l, init, sized => by
      rw [List.foldl_cons, size_foldl_array g n l _ fun x member =>
        sized x (List.mem_cons_of_mem a member), Array.size_append,
        sized a List.mem_cons_self, List.length_cons]
      ring

theorem Setup.size_littleEndian (count value : ℕ) :
    (Setup.littleEndian count value).size = count := by
  simp [Setup.littleEndian, ByteArray.size]

theorem size_digestHeader :
    ("akita-pcs/field-digest/v8".toUTF8 ++ ByteArray.mk (Array.replicate 7 0)).size = 32 := by
  decide

/-- The bytes that `fieldDigest` hashes: 32 bytes of the tag and 8 bytes for each word. -/
theorem size_digestInput (values : Array Fq) :
    (digestInput values).size = 32 + 8 * values.size := by
  unfold digestInput
  rw [← Array.foldl_toList, size_foldl_bytes _ 8 _ _ fun a _ => Setup.size_littleEndian 8 _,
    size_digestHeader, Array.length_toList, Nat.mul_comm]

theorem size_eInput {d : ℕ} [NeZero d] (y : Rq d) : (eInput y).size = 32 + 8 * d := by
  rw [eInput, size_digestInput, Poly.size_ofFn]

theorem size_tInput (S : Shape) (rows : ℕ → Rq S.innerDegree) :
    (tInput S rows).size = 32 + 8 * (S.innerRank * S.innerDegree) := by
  rw [tInput, size_digestInput, size_foldl_array _ S.innerDegree _ _ fun r _ =>
    Poly.size_ofFn _ _, List.length_range]
  simp

/-- The bytes of the last block of `e` of `key20.aks` that the terminal hashes: 544 bytes. -/
theorem size_eInput20 (y : Rq Key20.terminal.innerDegree) : (eInput y).size = 544 :=
  size_eInput y

/-- The bytes of the last block of `t` of `key20.aks` that the terminal hashes: 4128 bytes. -/
theorem size_tInput20 (rows : ℕ → Rq Key20.terminal.innerDegree) :
    (tInput Key20.terminal rows).size = 4128 :=
  size_tInput Key20.terminal rows

theorem size_eBytes (S : Shape) (eLast : Array Fq) :
    (eBytes S eLast).size = 32 + 8 * S.innerDegree := by
  rw [eBytes, size_digestInput, Poly.canon, Poly.size_ofFn]

theorem size_tBytes (S : Shape) (tLast : List (Array Fq)) :
    (tBytes S tLast).size = 32 + 8 * (tLast.length * S.innerDegree) := by
  rw [tBytes, size_digestInput, size_foldl_array _ S.innerDegree _ _ fun x _ => by
    rw [Poly.canon, Poly.size_ofFn]]
  simp

theorem length_mapM {α β : Type} (f : α → Option β) : ∀ (l : List α) {out : List β},
    l.mapM f = some out → out.length = l.length
  | [], out, found => by
      cases found
      rfl
  | a :: l, out, found => by
      rw [List.mapM_cons] at found
      cases first : f a with
      | none =>
          rw [first] at found
          cases found
      | some b =>
          rw [first] at found
          cases rest : l.mapM f with
          | none =>
              rw [rest] at found
              cases found
          | some tail =>
              rw [rest] at found
              cases found
              exact congrArg Nat.succ (length_mapM f l rest)

/-- The last block of `t` of the code has one ring element for each row of `A`. -/
theorem lastBlocks_length (S : Shape) {cap : ℕ} {P : Golomb.Params} {setup : Array Fq}
    {msgs : TerminalMessages} {ch : TerminalChallenges} {eLast : Array Fq}
    {tLast : List (Array Fq)}
    (found : lastBlocks S cap P setup msgs ch = some (eLast, tLast)) :
    tLast.length = S.innerRank := by
  unfold lastBlocks at found
  split at found
  · cases found
  · split at found
    · split at found
      · split at found
        · cases found
        · split at found
          · cases found
          · split at found
            · cases found
            · rename_i rows
              cases found
              exact (length_mapM _ _ rows).trans List.length_range
      · cases found
    · cases found

end Sizes

section Blocks

variable (S : Shape) {E : Type} [CommRing E] [Algebra Zq E] [Degree2 E]

/-- A terminal that accepts has a payload that decodes and last blocks. -/
theorem lastBlocks_of_ok {digest : ByteArray → ByteArray} {cap : ℕ} {P : Golomb.Params}
    {setup : Array Fq} {msgs : TerminalMessages} {ch : TerminalChallenges} {out : TerminalValues}
    (ok : terminalRun S digest cap P setup msgs ch = .ok out) :
    ∃ blocks z, Golomb.decode P msgs.payload = some z ∧
      lastBlocks S cap P setup msgs ch = some blocks := by
  unfold terminalRun at ok
  cases decoded : Golomb.decode P msgs.payload with
  | none =>
      rw [decoded] at ok
      cases ok
  | some z =>
      rw [decoded] at ok
      dsimp only at ok
      unfold terminalChecks at ok
      by_cases norm : centeredNormSq z ≤ cap
      · by_cases valid : ch.fold.length = S.blocks ∧
            ch.fold.all (sparseValid S.innerDegree) = true
        · rw [ite_eq_left norm, ite_eq_left valid] at ok
          cases inverse : Poly.inverse (Nat.log 2 S.innerDegree)
              (challenge S ch (S.blocks - 1)) with
          | none =>
              rw [inverse] at ok
              cases ok
          | some inv =>
              rw [inverse] at ok
              dsimp only at ok
              cases recovered : recover S (challenge S ch (S.blocks - 1)) inv (folded S msgs ch)
                  (reduced S ch z) with
              | none =>
                  rw [recovered] at ok
                  cases ok
              | some eLast =>
                  rw [recovered] at ok
                  dsimp only at ok
                  cases rows : tLastOf S msgs ch inv
                      ((List.range S.innerRank).map (aRow S setup z)) with
                  | none =>
                      rw [rows] at ok
                      cases ok
                  | some tLast =>
                      exact ⟨(eLast, tLast), z, rfl, by
                        simp only [lastBlocks, decoded, norm, valid, and_self, ↓reduceIte,
                          inverse, recovered, rows]⟩
        · rw [ite_eq_left norm, ite_eq_right valid] at ok
          cases ok
      · rw [ite_eq_right norm] at ok
        cases ok

/-- Last blocks that pass the checks (128) and (129) of the terminal are the last blocks of the
code: the bytes that the terminal hashes for them are `eBytes` and `tBytes` of the blocks of the
code. -/
theorem lastBlocks_unique (quarter : 4 * (S.innerDegree / 4) = S.innerDegree)
    (blocks : 1 ≤ S.blocks) (power : 2 ^ Nat.log 2 S.innerDegree = S.innerDegree) {cap : ℕ}
    {P : Golomb.Params} {setup : Array Fq} {msgs : TerminalMessages} {ch : TerminalChallenges}
    {eBlock : Array Fq} {tBlock : List (Array Fq)} {z : Array ℤ}
    (found : lastBlocks S cap P setup msgs ch = some (eBlock, tBlock))
    (decoded : Golomb.decode P msgs.payload = some z) {eLast : Rq S.innerDegree}
    {tLast : ℕ → Rq S.innerDegree}
    (evaluation : ∑ i, challengesOf S ch i * partialsOf S msgs eLast i =
      traceContract (subfield (E := E) quarter).toRingHom
        (fun x : Fin S.blockLength => eqAt ((positionPoint S ch.point).map Ext.into) x)
        (gadget S.sourceBase (zSource S z)))
    (inner : (matrixOf S setup).mulVec (zSource S z) =
      ∑ i, challengesOf S ch i • stateOf S msgs tLast i) :
    eInput eLast = eBytes S eBlock ∧ tInput S tLast = tBytes S tBlock := by
  unfold lastBlocks at found
  rw [decoded] at found
  dsimp only at found
  split at found
  swap
  · cases found
  split at found
  swap
  · cases found
  split at found
  · cases found
  rename_i inv invFound
  split at found
  · cases found
  rename_i block recovered
  split at found
  · cases found
  rename_i rowBlocks rows
  cases found
  have inverse := inverse_mul' power invFound
  obtain rfl := recover_eq S recovered
  have rowsSome : tLastOf S msgs ch inv ((List.range S.innerRank).map (aRow S setup z)) =
      some ((List.range S.innerRank).map fun r => Poly.mul S.innerDegree inv
        (Poly.sub S.innerDegree (aRow S setup z r) (tPartial S msgs ch r))) := by
    refine mapM_some _ _ _ fun r member => ?_
    rw [List.getD_eq_getElem?_getD, List.getElem?_map,
      List.getElem?_range (List.mem_range.mp member), Option.map_some, Option.getD_some]
    exact recover_some S inverse
  rw [rowsSome] at rows
  obtain rfl := Option.some.inj rows
  have eSame : eLast = Poly.toRq S.innerDegree (Poly.mul S.innerDegree inv
      (Poly.sub S.innerDegree (reduced S ch z) (folded S msgs ch))) :=
    (solution_iff S inverse eLast).mp
      (((sum_partials (S := S) blocks msgs ch eLast).symm.trans evaluation).trans
        (toRq_reduced (E := E) S quarter ch z).symm)
  have tSame : ∀ r < S.innerRank, tLast r = Poly.toRq S.innerDegree (Poly.mul S.innerDegree inv
      (Poly.sub S.innerDegree (aRow S setup z r) (tPartial S msgs ch r))) := fun r small => by
    have row := congrFun inner ⟨r, small⟩
    rw [sum_state (S := S) blocks] at row
    exact (solution_iff S inverse (tLast r)).mp
      (row.symm.trans (toRq_aRow S setup z ⟨r, small⟩).symm)
  refine ⟨?_, ?_⟩
  · show digestInput (Poly.ofFn S.innerDegree fun k => Fq.ofNat (coeff eLast k).val) =
      digestInput (Poly.canon S.innerDegree _)
    rw [eSame, Poly.canon_toRq]
  · unfold tInput tBytes
    rw [List.foldl_map]
    congr 1
    refine List.foldl_ext _ _ _ fun out r member => ?_
    rw [tSame r (List.mem_range.mp member), Poly.canon_toRq]

end Blocks

section Queries

variable {L B C S D : ℕ} (pub : Public) (proof : ByteArray)
  (fs : Sponge.Base L B C → Sponge.Y S C) (hash : Sponge.Str D → Bytes 32) {zMax rice : ℕ}
  {messages : Array Message} {rest : ℕ}

/-- An accepted proof: the statement has the shape of the schedule row, the layout takes the
plan, and the replay of the decoded messages has no stop. -/
theorem replay_of_accepted
    (accepted : verifyProof (pub.withOracles fs hash) proof = .verdict .accept)
    (decoded : decode (pub.withOracles fs hash).sealed.ops proof = .ok (messages, rest)) :
    pub.shaped = true ∧
      script pub.layout pub.plan = .ok (pub.withOracles fs hash).sealed.ops ∧
      (replay (pub.withOracles fs hash).sealed (pub.withOracles fs hash).sealed.ops proof
        messages).1 = none := by
  obtain ⟨shaped, ran⟩ := (verifyProof_accept_iff _ _).mp accepted
  have scripted : script (pub.withOracles fs hash).sealed.layout
      (pub.withOracles fs hash).sealed.plan = .ok (pub.withOracles fs hash).sealed.ops :=
    verifyProof_scripted ran
  refine ⟨shaped, scripted, ?_⟩
  rw [verifyProof_script _ proof scripted] at ran
  obtain ⟨messages', rest', decoded', -, stops⟩ := (verifyBytes_accept _ _ _).mp ran
  rw [decoded] at decoded'
  cases decoded'
  rcases stop : (replay (pub.withOracles fs hash).sealed (pub.withOracles fs hash).sealed.ops
    proof messages).1 with _ | (_ | _ | _)
  · rfl
  · exact absurd stop (stops _ _).1
  · exact absurd stop (stops _ _).2.1
  · exact absurd stop (stops _ "").2.2

theorem terminalWith_ok {i level : ℕ} {r : Option (Except String TerminalValues)}
    {s : Replayer} (ok : (((terminalWith i level r).run).run s).1 = .ok ()) :
    ∃ out, r = some (.ok out) := by
  rcases r with _ | _ | out
  · cases ok
  · cases ok
  · exact ⟨out, rfl⟩

theorem inr_strOf_congr {Fs : Type} {a b : ByteArray} (same : a = b) (short : a.size ≤ D) :
    (Sum.inr (strOf a short) : Fs ⊕ Sponge.Str D) = Sum.inr (strOf b (same ▸ short)) := by
  subst same
  rfl

/-- For an accepted proof of `layout20`, the events of the sponge before the squeeze of the work
of each stage `k < 473` are the events of a base, and this base is a query of the verifier. -/
theorem verifierRun_asked20 (layout : pub.layout = layout20 zMax rice)
    (accepted : verifyProof (pub.withOracles fs hash) proof = .verdict .accept)
    (decoded : decode (pub.withOracles fs hash).sealed.ops proof = .ok (messages, rest)) {k : ℕ}
    (inside : k < 473) :
    ∃ base : Sponge.Base L B C,
      base.events = eventsAt20 (pub.withOracles fs hash).sealed proof messages k ∧
        Sum.inl base ∈ (verifierRun pub proof).trace fs hash := by
  obtain ⟨shaped, scripted, none_⟩ := replay_of_accepted pub proof fs hash accepted decoded
  have ran := (replay_none_iff _ _ _ _).mp none_
  have bits : Bits20 (pub.withOracles fs hash).sealed := bits20_of_scripted layout scripted
  obtain ⟨site, level, at_⟩ := bits k inside
  have positive := bits20_ne_zero inside
  have upto := runFrom_take_ok _ proof messages _
    (stageIndex20 (pub.withOracles fs hash).sealed k + 1) _ ran
  have ideal : (pub.withOracles fs hash).sealed.sponge.ideal = some (answerOf fs) := rfl
  obtain ⟨m, found, events, -, -⟩ := work_squeeze20 at_ positive upto ideal
  have passed := work20_answer bits inside upto ideal
  obtain ⟨base, sub, split_⟩ :=
    Sponge.split_of_worked fs (Nat.one_le_iff_ne_zero.mpr positive) passed
  have before := runFrom_take_ok _ proof messages _
    (stageIndex20 (pub.withOracles fs hash).sealed k) _ upto
  rw [List.take_take, Nat.min_eq_left (Nat.le_succ _)] at before
  refine ⟨base, ?_, verifierRun_asked pub proof fs hash shaped scripted decoded at_ before ?_⟩
  · obtain ⟨parts, -⟩ := Sponge.split_eq_some.mp split_
    unfold Sponge.Sub.events at parts
    cases counts : sub.counts with
    | nil =>
        rw [counts] at parts
        exact parts.symm
    | cons c tail =>
        rw [counts, events, List.map_cons, List.cons_append] at parts
        exact absurd (List.cons.inj parts).1 (fun different => by cases different)
  · simp only [asks, positive, ↓reduceIte, found, Option.map_some]
    rw [← events]
    simp only [Verifier.baseOf, Option.bind_some, split_, Option.map_some]

/-- For an accepted proof of `layout20`, the queries of the verifier to the sponge are the 473
bases of the events before the works of the stages, each one time. -/
theorem verifierRun_bases20 (layout : pub.layout = layout20 zMax rice)
    (accepted : verifyProof (pub.withOracles fs hash) proof = .verdict .accept)
    (decoded : decode (pub.withOracles fs hash).sealed.ops proof = .ok (messages, rest)) :
    ∃ bases : Fin 473 → Sponge.Base L B C,
      (∀ k : Fin 473, (bases k).events =
        eventsAt20 (pub.withOracles fs hash).sealed proof messages k) ∧
      (((verifierRun pub proof).trace fs hash).filterMap Sum.getLeft?).Perm
        (List.ofFn bases) := by
  have each : ∀ k : Fin 473, ∃ base : Sponge.Base L B C,
      base.events = eventsAt20 (pub.withOracles fs hash).sealed proof messages k ∧
        Sum.inl base ∈ (verifierRun pub proof).trace fs hash := fun k =>
    verifierRun_asked20 pub proof fs hash layout accepted decoded k.isLt
  choose bases events asked using each
  refine ⟨bases, events, ?_⟩
  obtain ⟨-, scripted, none_⟩ := replay_of_accepted pub proof fs hash accepted decoded
  have ran := (replay_none_iff _ _ _ _).mp none_
  have bits : Bits20 (pub.withOracles fs hash).sealed := bits20_of_scripted layout scripted
  have different : ∀ k k' : Fin 473, (k' : ℕ) < k → bases k ≠ bases k' := by
    intro k k' lt same
    refine stage20_not_lt bits lt k.isLt decoded decoded
      (runFrom_take_ok _ proof messages _ _ _ ran) (runFrom_take_ok _ proof messages _ _ _ ran) ?_
    rw [← events k, ← events k', same]
  have injective : Function.Injective bases := by
    intro k k' same
    rcases Nat.lt_trichotomy (k : ℕ) k' with lt | eq | gt
    · exact absurd same.symm (different k' k lt)
    · exact Fin.ext eq
    · exact absurd same (different k k' gt)
  have perm := ((List.nodup_ofFn.mpr injective).subperm fun base member => by
    obtain ⟨k, rfl⟩ := List.mem_ofFn.mp member
    exact (List.mem_filterMap (f := Sum.getLeft?)).mpr
      ⟨Sum.inl (bases k), asked k, rfl⟩).perm_of_length_le (by
      rw [List.length_ofFn]
      exact (verifierRun_bound20 pub proof layout).length_left fs hash)
  exact perm.symm

/-- For an accepted proof and the terminal of its script: the setup prefix of the terminal, the
response of its payload and the last blocks of the code, whose bytes are queries of the
verifier. -/
theorem verifierRun_blocks
    (accepted : verifyProof (pub.withOracles fs hash) proof = .verdict .accept)
    (decoded : decode (pub.withOracles fs hash).sealed.ops proof = .ok (messages, rest))
    {c level : ℕ}
    (at_ : (pub.withOracles fs hash).sealed.ops[c]? = some (.terminal level)) :
    ∃ (setup eBlock : Array Fq) (tBlock : List (Array Fq)) (z : Array ℤ),
      (pub.withOracles fs hash).sealed.setupFor pub.layout.terminal.setupCount = some setup ∧
      Golomb.decode pub.layout.terminal.params (terminalMessagesIn (stateBefore
        (pub.withOracles fs hash).sealed proof messages (pub.withOracles fs hash).sealed.ops c)
        level).payload = some z ∧
      lastBlocks pub.layout.terminal.shape pub.layout.terminal.limit pub.layout.terminal.params
        setup
        (terminalMessagesIn (stateBefore (pub.withOracles fs hash).sealed proof messages
          (pub.withOracles fs hash).sealed.ops c) level)
        (terminalChallengesIn (stateBefore (pub.withOracles fs hash).sealed proof messages
          (pub.withOracles fs hash).sealed.ops c) level) = some (eBlock, tBlock) ∧
      (∀ short, Sum.inr (strOf (eBytes pub.layout.terminal.shape eBlock) short) ∈
        (verifierRun pub proof).trace fs hash) ∧
      ∀ short, Sum.inr (strOf (tBytes pub.layout.terminal.shape tBlock) short) ∈
        (verifierRun pub proof).trace fs hash := by
  obtain ⟨shaped, scripted, none_⟩ := replay_of_accepted pub proof fs hash accepted decoded
  obtain ⟨before, stepped⟩ :=
    replay_step_ok (pub.withOracles fs hash).sealed proof messages none_ at_
  rw [step_terminal] at stepped
  obtain ⟨out, value⟩ := terminalWith_ok stepped
  have value' : ((pub.withOracles fs hash).sealed.setupFor pub.layout.terminal.setupCount).map
      (fun setup => terminalRun pub.layout.terminal.shape (digestOf hash)
        pub.layout.terminal.limit pub.layout.terminal.params setup
        (terminalMessagesIn (stateBefore (pub.withOracles fs hash).sealed proof messages
          (pub.withOracles fs hash).sealed.ops c) level)
        (terminalChallengesIn (stateBefore (pub.withOracles fs hash).sealed proof messages
          (pub.withOracles fs hash).sealed.ops c) level)) = some (.ok out) := value
  cases expanded : (pub.withOracles fs hash).sealed.setupFor
      pub.layout.terminal.setupCount with
  | none =>
      rw [expanded] at value'
      cases value'
  | some setup =>
      rw [expanded, Option.map_some] at value'
      obtain ⟨⟨eBlock, tBlock⟩, z, decodedZ, found⟩ :=
        lastBlocks_of_ok _ (Option.some.inj value')
      have listed : blocksIn pub.sealed (stateBefore (pub.withOracles fs hash).sealed proof
          messages (pub.withOracles fs hash).sealed.ops c) level = some (eBlock, tBlock) :=
        (congrArg (fun o : Option (Array Fq) => o.bind fun setup =>
          lastBlocks pub.layout.terminal.shape pub.layout.terminal.limit
            pub.layout.terminal.params setup
            (terminalMessagesIn (stateBefore (pub.withOracles fs hash).sealed proof messages
              (pub.withOracles fs hash).sealed.ops c) level)
            (terminalChallengesIn (stateBefore (pub.withOracles fs hash).sealed proof messages
              (pub.withOracles fs hash).sealed.ops c) level)) expanded).trans found
      obtain ⟨eAsked, tAsked⟩ := verifierRun_digests pub proof fs hash shaped scripted decoded
        at_ before listed
      exact ⟨setup, eBlock, tBlock, z, rfl, decodedZ, found, eAsked, tAsked⟩

variable {E : Type} [Field E] [Algebra Zq E] [Degree2 E]

/-- For an accepted proof of `layout20` and last blocks with `Last20`, the bytes that the
terminal hashes for the two last blocks are queries of the verifier, when the digest oracle takes
4128 bytes. -/
theorem verifierRun_digests20 (layout : pub.layout = layout20 zMax rice)
    (accepted : verifyProof (pub.withOracles fs hash) proof = .verdict .accept)
    (decoded : decode (pub.withOracles fs hash).sealed.ops proof = .ok (messages, rest))
    (fits : 4128 ≤ D) {tLast : ℕ → Rq Key20.terminal.innerDegree}
    {eLast : Rq Key20.terminal.innerDegree}
    (last : Last20 (E := E) (pub.withOracles fs hash).sealed proof messages tLast eLast) :
    Sum.inr (strOf (eInput eLast) ((size_eInput20 eLast).trans_le
        (Nat.le_trans (by decide) fits))) ∈ (verifierRun pub proof).trace fs hash ∧
      Sum.inr (strOf (tInput Key20.terminal tLast) ((size_tInput20 tLast).trans_le fits)) ∈
        (verifierRun pub proof).trace fs hash := by
  obtain ⟨-, scripted, -⟩ := replay_of_accepted pub proof fs hash accepted decoded
  rw [layout] at scripted
  obtain ⟨-, terminalAt, -⟩ := script20 scripted
  obtain ⟨setup, eBlock, tBlock, z, expanded, decodedZ, found, eAsked, tAsked⟩ :=
    verifierRun_blocks pub proof fs hash accepted decoded terminalAt
  have power : 2 ^ Nat.log 2 Key20.terminal.innerDegree = Key20.terminal.innerDegree := by
    have six : Key20.terminal.innerDegree = 2 ^ 6 := by decide
    rw [six, Nat.log_pow (by norm_num)]
  have accepts := last.2.2.2
  unfold terminalStatement20 terminalStatementIn at accepts
  have full : full20 (pub.withOracles fs hash).sealed = setup :=
    congrArg (fun o : Option (Array Fq) => o.getD #[]) expanded
  have response : response20 (pub.withOracles fs hash).sealed proof messages = z :=
    congrArg (fun o : Option (Array ℤ) => o.getD #[]) decodedZ
  rw [full, response] at accepts
  rw [layout] at found decodedZ eAsked tAsked
  obtain ⟨eSame, tSame⟩ := lastBlocks_unique Key20.terminal (E := E) (by decide) (by decide)
    power found decodedZ accepts.evaluation accepts.inner
  exact ⟨by rw [inr_strOf_congr eSame]; exact eAsked _,
    by rw [inr_strOf_congr tSame]; exact tAsked _⟩

end Queries

end NightstreamFPrime.Spec.Compression.Akita.Replay
