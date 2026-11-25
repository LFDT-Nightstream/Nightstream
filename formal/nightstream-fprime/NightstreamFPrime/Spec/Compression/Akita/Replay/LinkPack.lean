import NightstreamFPrime.Spec.Compression.Akita.Replay.LinkKey20

/-! The link from a level of the replay to the next packing level with one group (Akita revision
`d50f700`, `fold/suffix.rs`; ePrint 2026/1983, revision of 2026-11-23, Figure 4, steps 3 and 5,
Section 4.5 with (40) to (42); `Tier.Linked` of `Extract/Recursion.lean`).

The code absorbs the point `point.0` and the values `openings` of the packing level `ℓ + 1` from
the recorded `fused.point` and `witness_evaluation` of the level `ℓ` (`Build.prefix_` without
Stage 3 at the level `ℓ`).

- `verifyProof_recorded`: when `runProof` accepts, a key with one writer has on a later state
  the values that its writer appends; `verifyProof_absorbed` is it for the absorb of one recorded
  key.
- `verifyProof_fusedPoint`: the recorded `fused.point` of a fold of the shape `gameShape` has
  `μ = F.vars` coordinates.
- `point_segments`: the segments `(r_pack, r_tail, r_pos, r_blk)` of `pointOf` are the point.
- `verifyProof_packLink`: the statement `packStatementFor` of the replay at the level `ℓ + 1` is
  the folding statement of the claim of `Tier.Linked` for the result `(r_2, v_W)` of the level
  `ℓ`: the point `Point.ofSegments` of the segments of `r_2`, the value `v_W`, the payload of the
  words `next_witness` of the level `ℓ`.
- `verifyProof_key20_packLink`: the link from the root to the level 1 of `key20.aks`. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

open NightstreamFPrime.Spec.Compression.Lib NightstreamFPrime.Spec.Compression.Game
open NightstreamFPrime.Spec.Compression (Game)
open NightstreamFPrime.Spec.Compression.Akita

open Replayer

theorem claimPointIn_succ (s : Replayer) (level : ℕ) :
    claimPointIn s (level + 1) 0 = (recordedIn s ⟨level + 1, "point.0", none⟩).toList := by
  unfold claimPointIn
  rw [ite_eq_right (Nat.succ_ne_zero level)]
  rfl

/-- The segments `(r_pack, r_tail, r_pos, r_blk)` of lengths `a`, `b`, `c` and the rest are the
point. -/
theorem point_segments (a b c : ℕ) (r : List Ext) :
    r.take a ++ ((r.drop a).take b ++ ((r.drop (a + b)).take c ++ r.drop (a + b + c))) = r := by
  have last : r.drop (a + b + c) = (r.drop (a + b)).drop c := by rw [List.drop_drop]
  have middle : r.drop (a + b) = (r.drop a).drop b := by rw [List.drop_drop]
  rw [last, List.take_append_drop, middle, List.take_append_drop, List.take_append_drop]

theorem point_lengths (a b c : ℕ) (r : List Ext) (long : a + b + c ≤ r.length) :
    (r.take a).length = a ∧ ((r.drop a).take b).length = b ∧
      ((r.drop (a + b)).take c).length = c ∧
      (r.drop (a + b + c)).length = r.length - (a + b + c) := by
  refine ⟨?_, ?_, ?_, ?_⟩
  · rw [List.length_take]
    omega
  · rw [List.length_take, List.length_drop]
    omega
  · rw [List.length_take, List.length_drop]
    omega
  · rw [List.length_drop]

section Proof

variable (pub : Public) (proof : ByteArray) {ops : Array Op}

/-- An operation `op` at `a` that appends `values s` to the recorded values of `key`, with no
other writer of `key` before `c'`: when `runProof` accepts, `key` has on the state before `c'`
the values `values` of the state before `a`. -/
theorem verifyProof_recorded (scripted : script pub.layout pub.plan = .ok ops)
    (ends : endsFit ops.zipIdx.toList = true) {full : Array Fq}
    (terminalSetup : pub.setupFor pub.layout.terminal.setupCount = some full)
    (accepted : runProof pub proof = .verdict .accept) {messages : Array Message} {rest : ℕ}
    (decoded : decode ops proof = .ok (messages, rest)) {a c' : ℕ} {key : Key} {op : Op}
    (at_ : ops[a]? = some op) (values : Replayer → Array Ext)
    (records : ∀ s, recordedIn (((step pub proof messages a op).run).run s).2 key =
      recordedIn s key ++ values s)
    (fresh : frees (ops.zipIdx.toList.take a) [key] [] = true) (after : a + 1 ≤ c')
    (kept : frees ((ops.zipIdx.toList.take c').drop (a + 1)) [key] [] = true) :
    recordedIn (stateBefore pub proof messages ops c') key =
      values (stateBefore pub proof messages ops a) := by
  obtain ⟨-, none_⟩ := replay_of_decoded pub proof scripted ends terminalSetup accepted decoded
  obtain ⟨before, stepped⟩ := replay_step_ok pub proof messages none_ at_
  have next : stateBefore pub proof messages ops (a + 1) =
      (((step pub proof messages a op).run).run (stateBefore pub proof messages ops a)).2 := by
    unfold stateBefore
    rw [take_succ_zipIdx ops at_, runFrom_append_snd pub proof messages _ _ _ before,
      runFrom_cons_snd pub proof messages _ _ _ stepped, runFrom_nil]
  have empty : recordedIn (stateBefore pub proof messages ops a) key = #[] := by
    unfold stateBefore
    rw [recorded_free pub proof messages _ _ key (frees_writes fresh List.mem_cons_self),
      channelStart_fresh]
  obtain ⟨frames, -⟩ := stateBefore_frees pub proof messages after kept
  rw [(frames key List.mem_cons_self).1, next, records, empty, Array.empty_append]

/-- An absorb of one recorded key `k` into `key` at `a`, with no other writer of `key` before
`c'`: when `runProof` accepts, `key` has on the state before `c'` the recorded values of `k`
on the state before `a`. -/
theorem verifyProof_absorbed (scripted : script pub.layout pub.plan = .ok ops)
    (ends : endsFit ops.zipIdx.toList = true) {full : Array Fq}
    (terminalSetup : pub.setupFor pub.layout.terminal.setupCount = some full)
    (accepted : runProof pub proof = .verdict .accept) {messages : Array Message} {rest : ℕ}
    (decoded : decode ops proof = .ok (messages, rest)) {a c' : ℕ} {key k : Key}
    (at_ : ops[a]? = some (.absorb key (.recorded [k])))
    (fresh : frees (ops.zipIdx.toList.take a) [key] [] = true) (after : a + 1 ≤ c')
    (kept : frees ((ops.zipIdx.toList.take c').drop (a + 1)) [key] [] = true) :
    recordedIn (stateBefore pub proof messages ops c') key =
      recordedIn (stateBefore pub proof messages ops a) k := by
  rw [verifyProof_recorded pub proof scripted ends terminalSetup accepted decoded at_
    (fun s => [k].foldl (fun out k => out ++ recordedIn s k) #[])
    (fun s => (step_absorb_recorded pub proof messages a key [k] s).2) fresh after kept]
  simp

/-- The recorded `fused.point` of a fold of the shape `gameShape` at its `fused.end` has `F.vars`
coordinates, when `runProof` accepts. -/
theorem verifyProof_fusedPoint (scripted : script pub.layout pub.plan = .ok ops)
    (ends : endsFit ops.zipIdx.toList = true) {full : Array Fq}
    (terminalSetup : pub.setupFor pub.layout.terminal.setupCount = some full)
    (accepted : runProof pub proof = .verdict .accept) {messages : Array Message} {rest : ℕ}
    (decoded : decode ops proof = .ok (messages, rest)) {c : ℕ} {key : Key} {level : ℕ}
    {F : Fold} (at_ : ops[c]? = some (.check key (.fused level F)))
    (game : gameShape (ops.zipIdx.toList.take c) level F = true) :
    (recordedIn (stateBefore pub proof messages ops c) ⟨level, "fused.point", none⟩).toList.length =
      F.vars := by
  obtain ⟨-, none_⟩ := replay_of_decoded pub proof scripted ends terminalSetup accepted decoded
  obtain ⟨before, _⟩ := replay_step_ok pub proof messages none_ at_
  simp only [gameShape, Bool.and_eq_true] at game
  obtain ⟨⟨⟨⟨_, _⟩, fused⟩, _⟩, _⟩ := game
  obtain ⟨e, shaped⟩ := roundsAt_sound fused
  obtain ⟨count, _, points⟩ := rounds_facts pub proof messages _ (channelStart_fresh pub)
    (channelStart_empty pub) _ _ _ _ e shaped
    (aligned_take messages _ c _ (decode_aligned decoded)) before
  calc (recordedIn (stateBefore pub proof messages ops c) ⟨level, "fused.point", none⟩).toList.length
      = ((roundsIn (runFrom pub proof messages (ops.zipIdx.toList.take c) (channelStart pub)).2
          ⟨level, "fused", none⟩).map (·.2)).length := congrArg List.length points.symm
    _ = F.vars := by rw [List.length_map]; exact count

variable {E : Type} [Field E] [Fintype E] [DecidableEq E] [Algebra Zq E] [Degree2 E]

omit [Fintype E] [DecidableEq E] in
/-- The link `Tier.Linked` on the replay (Akita Figure 4, steps 3 and 5): when `runProof`
accepts, the fold `G` of the level `level` ends at `c`, and the packing level `level + 1` with one
group absorbs `point.0` at `p` and `openings` at `o` from its `fused.point` and
`witness_evaluation`, on the state before `c'` the statement `packStatementFor` of the level
`level + 1` is the folding statement of the claim with the point `Point.ofSegments` of the
segments `r_2 = r_pack ++ (r_tail ++ (r_pos ++ r_blk))` of the recorded `r_2` of the level
`level`, the value `v_W` of the level `level` and the payload of the words `next_witness` of the
level `level`. The segments have the lengths of `Tier.Linked` (`stride`, `degree`, `length`,
`long`, `blocks`: numbers of the layout). -/
theorem verifyProof_packLink (scripted : script pub.layout pub.plan = .ok ops)
    (ends : endsFit ops.zipIdx.toList = true) {full : Array Fq}
    (terminalSetup : pub.setupFor pub.layout.terminal.setupCount = some full)
    (accepted : runProof pub proof = .verdict .accept) {messages : Array Message} {rest : ℕ}
    (decoded : decode ops proof = .ok (messages, rest)) {level c p o c' : ℕ} {key : Key}
    {G : Fold} (at_ : ops[c]? = some (.check key (.fused level G)))
    (game : gameShape (ops.zipIdx.toList.take c) level G = true)
    (pointAt : ops[p]? = some (.absorb ⟨level + 1, "point.0", none⟩
      (.recorded [⟨level, "fused.point", none⟩])))
    (openingsAt : ops[o]? = some (.absorb ⟨level + 1, "openings", none⟩
      (.recorded [⟨level, "witness_evaluation", none⟩])))
    (pointBefore : c ≤ p) (openingsBefore : c ≤ o) (pointAfter : p + 1 ≤ c')
    (openingsAfter : o + 1 ≤ c')
    (pointHeld : frees ((ops.zipIdx.toList.take p).drop c) [⟨level, "fused.point", none⟩] [] =
      true)
    (openingsHeld : frees ((ops.zipIdx.toList.take o).drop c)
      [⟨level, "witness_evaluation", none⟩] [] = true)
    (pointFresh : frees (ops.zipIdx.toList.take p) [⟨level + 1, "point.0", none⟩] [] = true)
    (openingsFresh : frees (ops.zipIdx.toList.take o) [⟨level + 1, "openings", none⟩] [] = true)
    (pointKept : frees ((ops.zipIdx.toList.take c').drop (p + 1))
      [⟨level + 1, "point.0", none⟩] [] = true)
    (openingsKept : frees ((ops.zipIdx.toList.take c').drop (o + 1))
      [⟨level + 1, "openings", none⟩] [] = true)
    {F : Fold} (plain : F.trace = none) (single : F.groups.length = 1) {S : Shape}
    (O : OpeningShape S) (stride : O.stride = 2 ^ Nat.log 2 O.stride)
    (degree : O.carrierDegree = 2 ^ Nat.log 2 O.carrierDegree)
    (length : S.blockLength = 2 ^ Nat.log 2 S.blockLength)
    (long : Nat.log 2 O.stride + Nat.log 2 O.carrierDegree + Nat.log 2 S.blockLength ≤ G.vars)
    (blocks : S.blocks ≤
      2 ^ (G.vars - (Nat.log 2 O.stride + Nat.log 2 O.carrierDegree + Nat.log 2 S.blockLength)))
    (setup : Array Fq) :
    let s := stateBefore pub proof messages ops c
    let s' := stateBefore pub proof messages ops c'
    let r2 := (recordedIn s ⟨level, "fused.point", none⟩).toList
    let a := Nat.log 2 O.stride
    let b := Nat.log 2 O.carrierDegree
    let m := Nat.log 2 S.blockLength
    let pack := (r2.take a).map (Ext.into (E := E))
    let tail := ((r2.drop a).take b).map (Ext.into (E := E))
    let position := ((r2.drop (a + b)).take m).map (Ext.into (E := E))
    let block := (r2.drop (a + b + m)).map (Ext.into (E := E))
    pack ++ (tail ++ (position ++ block)) = r2.map Ext.into ∧ S.blocks ≤ 2 ^ block.length ∧
      S.blockLength = 2 ^ position.length ∧ O.stride = 2 ^ pack.length ∧
      O.carrierDegree = 2 ^ tail.length ∧
      packStatementFor (E := E) O setup (claimPointIn s' (level + 1) 0) (dataIn pub s' (level + 1))
          .witness (rowsIn s' (level + 1) F) (claimsIn s' (level + 1) F) =
        (⟨payloadOf S.chain _ _ ((s'.fields.get? ⟨level, "next_witness", none⟩).getD #[]),
          payloadOf O.chain _ _ ((s'.fields.get? ⟨level + 1, "opening_payload", none⟩).getD #[]),
          Point.ofSegments O block position pack tail,
          (firstIn s ⟨level, "witness_evaluation", none⟩).into⟩ : Extract.Claim O E).statement
          (Key.ofSetup (flatOf setup) S) (OpeningKey.ofSetup (flatOf setup) O) := by
  have vars := verifyProof_fusedPoint pub proof scripted ends terminalSetup accepted decoded at_ game
  have point := verifyProof_absorbed pub proof scripted ends terminalSetup accepted decoded pointAt
    pointFresh pointAfter pointKept
  have openings := verifyProof_absorbed pub proof scripted ends terminalSetup accepted decoded
    openingsAt openingsFresh openingsAfter openingsKept
  obtain ⟨pointFrame, -⟩ := stateBefore_frees pub proof messages pointBefore pointHeld
  obtain ⟨openingsFrame, -⟩ := stateBefore_frees pub proof messages openingsBefore openingsHeld
  rw [(pointFrame _ List.mem_cons_self).1] at point
  rw [(openingsFrame _ List.mem_cons_self).1] at openings
  obtain ⟨witness, payload⟩ := dataIn_runs pub (stateBefore pub proof messages ops c') level S O
  dsimp only
  obtain ⟨packLength, tailLength, positionLength, blockLength⟩ := point_lengths
    (Nat.log 2 O.stride) (Nat.log 2 O.carrierDegree) (Nat.log 2 S.blockLength)
    (recordedIn (stateBefore pub proof messages ops c) ⟨level, "fused.point", none⟩).toList
    (vars ▸ long)
  have rows : rowsIn (stateBefore pub proof messages ops c') (level + 1) F = #[Ext.one] := by
    simp [rowsIn, single]
  have claims : claimsIn (stateBefore pub proof messages ops c') (level + 1) F =
      recordedIn (stateBefore pub proof messages ops c') ⟨level + 1, "openings", none⟩ := by
    simp [claimsIn, plain]
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_⟩
  · simp only [← List.map_append]
    rw [point_segments]
  · rw [List.length_map, blockLength, vars]
    exact blocks
  · rw [List.length_map, positionLength]
    exact length
  · rw [List.length_map, packLength]
    exact stride
  · rw [List.length_map, tailLength]
    exact degree
  · unfold packStatementFor
    rw [claimPointIn_succ, point, rows, claims, into_dot_one, openings, witness, payload]
    rfl

omit [Fintype E] [DecidableEq E] in
/-- `Tier.Linked` from the root (the level 0) to the level 1 of `key20.aks` on the replay: the
statement `packStatementFor` of the level 1 is the folding statement of the claim with the
segments `(r_pack, r_tail, r_pos, r_blk)` of lengths 4, 6, 7, 5 of the recorded `r_2` of the root
(`Key20.points`, `Key20.blocks_live`), the value `v_W` of the root and the payload of the words
`next_witness` of the root. -/
theorem verifyProof_key20_packLink {zMax rice : ℕ}
    (scripted : script pub.layout pub.plan = .ok ops)
    (ends : endsFit ops.zipIdx.toList = true) {full : Array Fq}
    (terminalSetup : pub.setupFor pub.layout.terminal.setupCount = some full)
    (accepted : runProof pub proof = .verdict .accept) {messages : Array Message} {rest : ℕ}
    (decoded : decode ops proof = .ok (messages, rest)) {c p o c' : ℕ} {key : Key}
    {G : Fold} (at_ : ops[c]? = some (.check key (.fused 0 G)))
    (root : (layout20 zMax rice).folds[0]? = some G)
    (game : gameShape (ops.zipIdx.toList.take c) 0 G = true)
    (pointAt : ops[p]? = some (.absorb ⟨1, "point.0", none⟩
      (.recorded [⟨0, "fused.point", none⟩])))
    (openingsAt : ops[o]? = some (.absorb ⟨1, "openings", none⟩
      (.recorded [⟨0, "witness_evaluation", none⟩])))
    (pointBefore : c ≤ p) (openingsBefore : c ≤ o) (pointAfter : p + 1 ≤ c')
    (openingsAfter : o + 1 ≤ c')
    (pointHeld : frees ((ops.zipIdx.toList.take p).drop c) [⟨0, "fused.point", none⟩] [] = true)
    (openingsHeld : frees ((ops.zipIdx.toList.take o).drop c)
      [⟨0, "witness_evaluation", none⟩] [] = true)
    (pointFresh : frees (ops.zipIdx.toList.take p) [⟨1, "point.0", none⟩] [] = true)
    (openingsFresh : frees (ops.zipIdx.toList.take o) [⟨1, "openings", none⟩] [] = true)
    (pointKept : frees ((ops.zipIdx.toList.take c').drop (p + 1)) [⟨1, "point.0", none⟩] [] =
      true)
    (openingsKept : frees ((ops.zipIdx.toList.take c').drop (o + 1)) [⟨1, "openings", none⟩] [] =
      true)
    {F : Fold} (fold : (layout20 zMax rice).folds[1]? = some F) (setup : Array Fq) :
    let s := stateBefore pub proof messages ops c
    let s' := stateBefore pub proof messages ops c'
    let r2 := (recordedIn s ⟨0, "fused.point", none⟩).toList
    let pack := (r2.take 4).map (Ext.into (E := E))
    let tail := ((r2.drop 4).take 6).map (Ext.into (E := E))
    let position := ((r2.drop (4 + 6)).take 7).map (Ext.into (E := E))
    let block := (r2.drop (4 + 6 + 7)).map (Ext.into (E := E))
    pack ++ (tail ++ (position ++ block)) = r2.map Ext.into ∧
      Key20.level₁.shape.blocks ≤ 2 ^ block.length ∧
      Key20.level₁.shape.blockLength = 2 ^ position.length ∧
      Key20.level₁.opening.stride = 2 ^ pack.length ∧
      Key20.level₁.opening.carrierDegree = 2 ^ tail.length ∧
      packStatementFor (E := E) Key20.level₁.opening setup (claimPointIn s' 1 0) (dataIn pub s' 1)
          .witness (rowsIn s' 1 F) (claimsIn s' 1 F) =
        (⟨payloadOf Key20.level₁.shape.chain _ _
            ((s'.fields.get? ⟨0, "next_witness", none⟩).getD #[]),
          payloadOf Key20.level₁.opening.chain _ _
            ((s'.fields.get? ⟨1, "opening_payload", none⟩).getD #[]),
          Point.ofSegments Key20.level₁.opening block position pack tail,
          (firstIn s ⟨0, "witness_evaluation", none⟩).into⟩ :
            Extract.Claim Key20.level₁.opening E).statement
          (Key.ofSetup (flatOf setup) Key20.level₁.shape)
          (OpeningKey.ofSetup (flatOf setup) Key20.level₁.opening) := by
  rw [show (layout20 zMax rice).folds = (layout20 0 0).folds from rfl] at root fold
  have vars : G.vars = Key20.rootVars := fold_field root Fold.vars rfl
  have numbers : Nat.log 2 Key20.level₁.opening.stride = 4 ∧
      Nat.log 2 Key20.level₁.opening.carrierDegree = 6 ∧
      Nat.log 2 Key20.level₁.shape.blockLength = 7 := by decide
  have linked := verifyProof_packLink (E := E) pub proof scripted ends terminalSetup accepted
    decoded at_ game pointAt openingsAt pointBefore openingsBefore pointAfter openingsAfter
    pointHeld openingsHeld pointFresh openingsFresh pointKept openingsKept
    (fold_field fold Fold.trace rfl) (fold_field fold (fun F => F.groups.length) rfl)
    Key20.level₁.opening (by decide) (by decide) (by decide) (by rw [vars]; decide)
    (by rw [vars]; decide) setup
  rw [numbers.1, numbers.2.1, numbers.2.2] at linked
  exact linked

end Proof

end NightstreamFPrime.Spec.Compression.Akita.Replay
