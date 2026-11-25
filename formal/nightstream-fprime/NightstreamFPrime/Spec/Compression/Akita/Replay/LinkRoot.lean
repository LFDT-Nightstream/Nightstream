import NightstreamFPrime.Spec.Compression.Akita.Replay.LinkPack

/-! The link from the public statement to the root fold of `key20.aks` on the replay (Akita
revision `d50f700`, `fold/root.rs`; ePrint 2026/1983, revision of 2026-11-23, Section 5.4,
"Combining the scalar claims", (72) with `φ = 1`, Theorem 10.46, "Batching root";
`RootLinked` of `Extract/BatchLevel.lean`).

The code absorbs the point `root.point.g` of each public group `g` and the values `openings` of
the public groups in the order of the groups, and draws the batching weights `ϑ` as the challenges
`row_coefficients`: three independent draws, not the powers of one challenge (difference (b) of
`decisions/akita-compression.md`). The runs of the relation have the groups in the order
`groupOrder` (the last group first); the run of the group `g` is the run at
`(groupOrder 3).symm g`, and its words are the commitment of the public group `g`.

- `step_absorb_point`, `step_absorb_values`: the absorb of `Value.point g` records the point of
  the public group `g`, and the absorb of `Value.values` records the values of the public groups.
- `batchRun_source`: the run of the group `g` reads the commitment of the public group `g`.
- `into_dot_three`: `Σ_i a_i·b_i` for three entries.
- `verifyProof_key20_rootLink`: the statement `batchStatementFor` of the replay at the root is the
  folding statement of the claim of `RootLinked`: the payload of each group from the commitment
  of its public group, the weights `ϑ` of `row_coefficients`, the target `Σ_g ϑ_g·v_g` of the
  values `v` of the public groups, and the point of each group as `Point.ofSegments` of the
  segments `(r_pack, r_tail, r_pos, r_blk)` of the point of its public group.
- `key20_rootLink_before`: `verifyProof_key20_rootLink` when the run before the `fused.end` of the
  root has no stop. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

open NightstreamFPrime.Spec.Compression.Lib NightstreamFPrime.Spec.Compression.Game
open NightstreamFPrime.Spec.Compression (Game)
open NightstreamFPrime.Spec.Compression.Akita

open Replayer

section Steps

variable (pub : Public) (proof : ByteArray) (messages : Array Message)

/-- The absorb of the point of the public group `g` records this point. -/
theorem step_absorb_point (i : ℕ) (key : Key) (g : ℕ) (s : Replayer) :
    recordedIn (((step pub proof messages i (.absorb key (.point g))).run).run s).2 key =
      recordedIn s key ++ (pub.groups[g]?.map (·.point)).getD #[] := by
  simp only [step, value]
  simp only [run_bind, run_pure, run_absorb, decodeExts_extBytes_array, run_record, run_modify]
  exact recordedIn_of_env rfl

/-- The absorb of the values of the public groups records them, in the order of the groups. -/
theorem step_absorb_values (i : ℕ) (key : Key) (s : Replayer) :
    recordedIn (((step pub proof messages i (.absorb key .values)).run).run s).2 key =
      recordedIn s key ++ pub.groups.flatMap (·.value) := by
  simp only [step, value]
  simp only [run_bind, run_pure, run_absorb, decodeExts_extBytes_array, run_record, run_modify]
  exact recordedIn_of_env rfl

end Steps

/-- The run of the relation of the group `g` of a fold with opening groups is the run at
`(groupOrder n).symm g`, and it reads the commitment of the public group `g`. -/
theorem batchRun_source {n : ℕ} (B : Batch (Fin n)) (g : Fin n) :
    (groupRun (fun k => B.shape (groupOrder n k)) B.openRank B.chain
      (fun k => .root (groupOrder n k).val) ((groupOrder n).symm g)).source = .root g.val := by
  unfold groupRun
  rw [pathRun_source]
  simp only [Equiv.apply_symm_apply]

variable {E : Type} [Field E] [Fintype E] [DecidableEq E] [Algebra Zq E] [Degree2 E]

omit [Fintype E] [DecidableEq E] in
/-- `Σ_i a_i·b_i` for three entries. -/
theorem into_dot_three {a b : Array Ext} (three : a.size = 3) (three' : b.size = 3) :
    ((dot a b).into : E) = ∑ g : Fin 3, (a.getD g Ext.zero).into * (b.getD g Ext.zero).into := by
  rcases a with ⟨_ | ⟨a0, _ | ⟨a1, _ | ⟨a2, _ | ⟨a3, l⟩⟩⟩⟩⟩ <;> simp at three
  rcases b with ⟨_ | ⟨b0, _ | ⟨b1, _ | ⟨b2, _ | ⟨b3, l⟩⟩⟩⟩⟩ <;> simp at three'
  unfold dot
  rw [← Array.foldl_toList, Array.toList_zip]
  simp [Fin.sum_univ_three, Ext.into_add, Ext.into_mul, Ext.into_zero]

section Proof

variable (pub : Public) (proof : ByteArray) {ops : Array Op}

omit [Fintype E] [DecidableEq E] in
/-- `RootLinked` for the root of `key20.aks` on the replay (Akita Section 5.4, (72) with `φ = 1`),
when the run before the `fused.end` of the root at `c` has no stop: the statement
`batchStatementFor` of the root on the state before `c` is the folding statement of the claim with
the payload of each group `g` from the commitment of the public group `g`, the opening payload of
the root, the weights `ϑ_g` of `row_coefficients`, the target `Σ_g ϑ_g·v_g` and the point of each
group as `Point.ofSegments` of the segments of the point of the public group `g`. -/
theorem key20_rootLink_before {zMax rice : ℕ} {messages : Array Message}
    {c o : ℕ} {F : Fold}
    (before : (runFrom pub proof messages (ops.zipIdx.toList.take c) (channelStart pub)).1 =
      .ok ())
    (root : (layout20 zMax rice).folds[0]? = some F)
    (weights : drawnCount ⟨0, "row_coefficients", none⟩ (ops.zipIdx.toList.take c) = some 3)
    (p : Fin 3 → ℕ)
    (pointAt : ∀ g : Fin 3,
      ops[p g]? = some (.absorb ⟨0, s!"root.point.{g.val}", none⟩ (.point g.val)))
    (pointFresh : ∀ g : Fin 3,
      frees (ops.zipIdx.toList.take (p g)) [⟨0, s!"root.point.{g.val}", none⟩] [] = true)
    (pointAfter : ∀ g, p g + 1 ≤ c)
    (pointKept : ∀ g : Fin 3, frees ((ops.zipIdx.toList.take c).drop (p g + 1))
      [⟨0, s!"root.point.{g.val}", none⟩] [] = true)
    (openingsAt : ops[o]? = some (.absorb ⟨0, "openings", none⟩ .values))
    (openingsFresh : frees (ops.zipIdx.toList.take o) [⟨0, "openings", none⟩] [] = true)
    (openingsAfter : o + 1 ≤ c)
    (openingsKept : frees ((ops.zipIdx.toList.take c).drop (o + 1)) [⟨0, "openings", none⟩] [] =
      true)
    (points : ∀ g : Fin 3, ((pub.groups[g.val]?.map (·.point)).getD #[]).size = ![20, 16, 16] g)
    (values : (pub.groups.flatMap (·.value)).size = 3) (setup : Array Fq) :
    let s := stateBefore pub proof messages ops c
    let ϑ := fun g : Fin 3 =>
      (((recordedIn s ⟨0, "row_coefficients", none⟩).getD g Ext.zero).into : E)
    let v := fun g : Fin 3 => (((pub.groups.flatMap (·.value)).getD g Ext.zero).into : E)
    let r := fun g : Fin 3 => ((pub.groups[g.val]?.map (·.point)).getD #[]).toList
    let a := fun g => Nat.log 2 (Key20.root.opening g).stride
    let b := fun g => Nat.log 2 (Key20.root.opening g).carrierDegree
    let m := fun g => Nat.log 2 (Key20.root.shape g).blockLength
    let pack := fun g => ((r g).take (a g)).map (Ext.into (E := E))
    let tail := fun g => (((r g).drop (a g)).take (b g)).map (Ext.into (E := E))
    let position := fun g => (((r g).drop (a g + b g)).take (m g)).map (Ext.into (E := E))
    let block := fun g => ((r g).drop (a g + b g + m g)).map (Ext.into (E := E))
    let run := fun g => groupRun (fun k => Key20.root.shape (groupOrder 3 k)) Key20.root.openRank
      Key20.root.chain (fun k => .root (groupOrder 3 k).val) ((groupOrder 3).symm g)
    (∀ g, (Key20.root.shape g).blocks ≤ 2 ^ (block g).length ∧
      (Key20.root.shape g).blockLength = 2 ^ (position g).length ∧
      (Key20.root.opening g).stride = 2 ^ (pack g).length ∧
      Key20.root.carrierDegree g = 2 ^ (tail g).length ∧
      pack g ++ (tail g ++ (position g ++ block g)) = (r g).map Ext.into) ∧
      batchStatementFor (E := E) Key20.root setup (claimPointIn s 0) (dataIn pub s 0)
          (fun k => .root (groupOrder 3 k).val) (rowsIn s 0 F) (claimsIn s 0 F) =
        (⟨fun g => payloadOf (Key20.root.shape g).chain _ _
            (commitmentRows ((pub.groups[g.val]?.map (·.commitment)).getD .empty)
              ((run g).degree * (run g).count)),
          payloadOf Key20.root.chain _ _ ((s.fields.get? ⟨0, "opening_payload", none⟩).getD #[]),
          fun g => Point.ofSegments (Key20.root.opening g) (block g) (position g) (pack g)
            (tail g),
          ϑ, ∑ g, ϑ g * v g⟩ : Key20.root.Claim E).statement
          (batchStatementFor (E := E) Key20.root setup (claimPointIn s 0) (dataIn pub s 0)
            (fun k => .root (groupOrder 3 k).val) (rowsIn s 0 F) (claimsIn s 0 F)).key
          (batchStatementFor (E := E) Key20.root setup (claimPointIn s 0) (dataIn pub s 0)
            (fun k => .root (groupOrder 3 k).val) (rowsIn s 0 F) (claimsIn s 0 F)).openKey := by
  rw [show (layout20 zMax rice).folds = (layout20 0 0).folds from rfl] at root
  have many : F.groups.length = 3 := fold_field root (fun F => F.groups.length) rfl
  have plain : F.trace = none := fold_field root Fold.trace rfl
  have coefficients : (recordedIn (stateBefore pub proof messages ops c)
      ⟨0, "row_coefficients", none⟩).size = 3 := by
    unfold stateBefore
    rw [size_drawnCount pub proof messages _ _ _ weights before, channelStart_fresh]
    rfl
  have openings := recorded_before pub proof openingsAt
    (before_le pub proof messages openingsAfter before) (fun _ => pub.groups.flatMap (·.value))
    (fun s => step_absorb_values pub proof messages o _ s) openingsFresh openingsAfter openingsKept
  have point := fun g : Fin 3 => recorded_before pub proof (pointAt g)
    (before_le pub proof messages (pointAfter g) before)
    (fun _ => (pub.groups[g.val]?.map (·.point)).getD #[])
    (fun s => step_absorb_point pub proof messages (p g) _ g.val s) (pointFresh g) (pointAfter g)
    (pointKept g)
  have rows : rowsIn (stateBefore pub proof messages ops c) 0 F =
      recordedIn (stateBefore pub proof messages ops c) ⟨0, "row_coefficients", none⟩ := by
    simp [rowsIn, many]
  have claims : claimsIn (stateBefore pub proof messages ops c) 0 F =
      pub.groups.flatMap (·.value) := by
    simp only [claimsIn, plain, Option.isSome_none, Bool.false_eq_true, ↓reduceIte]
    exact openings
  have claimed : ∀ g : Fin 3, claimPointIn (stateBefore pub proof messages ops c) 0 g =
      ((pub.groups[g.val]?.map (·.point)).getD #[]).toList := fun g => by
    unfold claimPointIn
    rw [ite_eq_left rfl, point g]
  refine ⟨fun g => ?_, ?_⟩
  · have sized : ((pub.groups[g.val]?.map (·.point)).getD #[]).toList.length = ![20, 16, 16] g := by
      rw [Array.length_toList]
      exact points g
    have long := point_lengths (Nat.log 2 (Key20.root.opening g).stride)
      (Nat.log 2 (Key20.root.opening g).carrierDegree)
      (Nat.log 2 (Key20.root.shape g).blockLength)
      ((pub.groups[g.val]?.map (·.point)).getD #[]).toList
    simp only [List.length_map]
    fin_cases g <;>
    · obtain ⟨packLength, tailLength, positionLength, blockLength⟩ := long (by rw [sized]; decide)
      refine ⟨?_, ?_, ?_, ?_, ?_⟩
      · rw [blockLength, sized]
        decide
      · rw [positionLength]
        decide
      · rw [packLength]
        decide
      · rw [tailLength]
        decide
      · rw [← List.map_append, ← List.map_append, ← List.map_append, point_segments]
  · dsimp only
    have runs : ∀ g : Fin 3, dataIn pub (stateBefore pub proof messages ops c) 0
        (groupRun (fun k => Key20.root.shape (groupOrder 3 k)) Key20.root.openRank Key20.root.chain
          (fun k => .root (groupOrder 3 k).val) ((groupOrder 3).symm g)) =
        commitmentRows ((pub.groups[g.val]?.map (·.commitment)).getD .empty)
          ((groupRun (fun k => Key20.root.shape (groupOrder 3 k)) Key20.root.openRank
            Key20.root.chain (fun k => .root (groupOrder 3 k).val) ((groupOrder 3).symm g)).degree *
          (groupRun (fun k => Key20.root.shape (groupOrder 3 k)) Key20.root.openRank
            Key20.root.chain (fun k => .root (groupOrder 3 k).val)
              ((groupOrder 3).symm g)).count) := fun g => by
      unfold dataIn
      rw [batchRun_source]
    have payload : dataIn pub (stateBefore pub proof messages ops c) 0
        (openRun (fun k => Key20.root.shape (groupOrder 3 k)) Key20.root.openRank
          Key20.root.openDegree Key20.root.chain) =
        ((stateBefore pub proof messages ops c).fields.get? ⟨0, "opening_payload", none⟩).getD #[] :=
      dataIn_payload pub _ 0 (pathRun_source _ _ _ _ _ _)
    have value : ((dot (recordedIn (stateBefore pub proof messages ops c)
        ⟨0, "row_coefficients", none⟩) (pub.groups.flatMap (·.value))).into : E) =
        ∑ g : Fin 3, ((recordedIn (stateBefore pub proof messages ops c)
          ⟨0, "row_coefficients", none⟩).getD g Ext.zero).into *
          ((pub.groups.flatMap (·.value)).getD g Ext.zero).into :=
      into_dot_three coefficients values
    unfold batchStatementFor
    simp only [rows, claims, claimed, runs, payload, value]
    rfl

omit [Fintype E] [DecidableEq E] in
/-- `RootLinked` for the root of `key20.aks` on the replay (Akita Section 5.4, (72) with `φ = 1`):
when `runProof` accepts, the root absorbs the point of each public group `g` at `p g` and the
values of the public groups at `o`, and draws three weights `row_coefficients` before its
`fused.end` at `c`, the statement `batchStatementFor` of the root is the folding statement of the
claim with the payload of each group `g` from the commitment of the public group `g`, the opening
payload of the root, the weights `ϑ_g`, the target `Σ_g ϑ_g·v_g` of the values `v` of the public
groups in their order, and the point of each group `g` as `Point.ofSegments` of the segments
`(r_pack, r_tail, r_pos, r_blk)` of the point of the public group `g`, of the lengths of
`RootLinked`. `points` and `values`: the point of each public group has `μ_g = 20, 16, 16`
coordinates, and the public groups have three values. -/
theorem verifyProof_key20_rootLink {zMax rice : ℕ}
    (scripted : script pub.layout pub.plan = .ok ops)
    (ends : endsFit ops.zipIdx.toList = true) {full : Array Fq}
    (terminalSetup : pub.setupFor pub.layout.terminal.setupCount = some full)
    (accepted : runProof pub proof = .verdict .accept) {messages : Array Message} {rest : ℕ}
    (decoded : decode ops proof = .ok (messages, rest)) {c o : ℕ} {key : Key} {F : Fold}
    (at_ : ops[c]? = some (.check key (.fused 0 F)))
    (root : (layout20 zMax rice).folds[0]? = some F)
    (weights : drawnCount ⟨0, "row_coefficients", none⟩ (ops.zipIdx.toList.take c) = some 3)
    (p : Fin 3 → ℕ)
    (pointAt : ∀ g : Fin 3,
      ops[p g]? = some (.absorb ⟨0, s!"root.point.{g.val}", none⟩ (.point g.val)))
    (pointFresh : ∀ g : Fin 3,
      frees (ops.zipIdx.toList.take (p g)) [⟨0, s!"root.point.{g.val}", none⟩] [] = true)
    (pointAfter : ∀ g, p g + 1 ≤ c)
    (pointKept : ∀ g : Fin 3, frees ((ops.zipIdx.toList.take c).drop (p g + 1))
      [⟨0, s!"root.point.{g.val}", none⟩] [] = true)
    (openingsAt : ops[o]? = some (.absorb ⟨0, "openings", none⟩ .values))
    (openingsFresh : frees (ops.zipIdx.toList.take o) [⟨0, "openings", none⟩] [] = true)
    (openingsAfter : o + 1 ≤ c)
    (openingsKept : frees ((ops.zipIdx.toList.take c).drop (o + 1)) [⟨0, "openings", none⟩] [] =
      true)
    (points : ∀ g : Fin 3, ((pub.groups[g.val]?.map (·.point)).getD #[]).size = ![20, 16, 16] g)
    (values : (pub.groups.flatMap (·.value)).size = 3) (setup : Array Fq) :
    let s := stateBefore pub proof messages ops c
    let ϑ := fun g : Fin 3 =>
      (((recordedIn s ⟨0, "row_coefficients", none⟩).getD g Ext.zero).into : E)
    let v := fun g : Fin 3 => (((pub.groups.flatMap (·.value)).getD g Ext.zero).into : E)
    let r := fun g : Fin 3 => ((pub.groups[g.val]?.map (·.point)).getD #[]).toList
    let a := fun g => Nat.log 2 (Key20.root.opening g).stride
    let b := fun g => Nat.log 2 (Key20.root.opening g).carrierDegree
    let m := fun g => Nat.log 2 (Key20.root.shape g).blockLength
    let pack := fun g => ((r g).take (a g)).map (Ext.into (E := E))
    let tail := fun g => (((r g).drop (a g)).take (b g)).map (Ext.into (E := E))
    let position := fun g => (((r g).drop (a g + b g)).take (m g)).map (Ext.into (E := E))
    let block := fun g => ((r g).drop (a g + b g + m g)).map (Ext.into (E := E))
    let run := fun g => groupRun (fun k => Key20.root.shape (groupOrder 3 k)) Key20.root.openRank
      Key20.root.chain (fun k => .root (groupOrder 3 k).val) ((groupOrder 3).symm g)
    (∀ g, (Key20.root.shape g).blocks ≤ 2 ^ (block g).length ∧
      (Key20.root.shape g).blockLength = 2 ^ (position g).length ∧
      (Key20.root.opening g).stride = 2 ^ (pack g).length ∧
      Key20.root.carrierDegree g = 2 ^ (tail g).length ∧
      pack g ++ (tail g ++ (position g ++ block g)) = (r g).map Ext.into) ∧
      batchStatementFor (E := E) Key20.root setup (claimPointIn s 0) (dataIn pub s 0)
          (fun k => .root (groupOrder 3 k).val) (rowsIn s 0 F) (claimsIn s 0 F) =
        (⟨fun g => payloadOf (Key20.root.shape g).chain _ _
            (commitmentRows ((pub.groups[g.val]?.map (·.commitment)).getD .empty)
              ((run g).degree * (run g).count)),
          payloadOf Key20.root.chain _ _ ((s.fields.get? ⟨0, "opening_payload", none⟩).getD #[]),
          fun g => Point.ofSegments (Key20.root.opening g) (block g) (position g) (pack g)
            (tail g),
          ϑ, ∑ g, ϑ g * v g⟩ : Key20.root.Claim E).statement
          (batchStatementFor (E := E) Key20.root setup (claimPointIn s 0) (dataIn pub s 0)
            (fun k => .root (groupOrder 3 k).val) (rowsIn s 0 F) (claimsIn s 0 F)).key
          (batchStatementFor (E := E) Key20.root setup (claimPointIn s 0) (dataIn pub s 0)
            (fun k => .root (groupOrder 3 k).val) (rowsIn s 0 F) (claimsIn s 0 F)).openKey :=
  key20_rootLink_before pub proof
    (replay_step_ok pub proof messages
      (replay_of_decoded pub proof scripted ends terminalSetup accepted decoded).2 at_).1
    root weights p pointAt pointFresh pointAfter pointKept openingsAt openingsFresh openingsAfter
    openingsKept points values setup

end Proof

end NightstreamFPrime.Spec.Compression.Akita.Replay
