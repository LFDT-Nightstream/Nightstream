import tests.AxiomAudit
import tests.AxiomsAkitaStage3
import NightstreamFPrime.Spec.Compression.Akita.Replay.FinalGame
import NightstreamFPrime.Spec.Compression.Akita.Replay.FinalKey20
import NightstreamFPrime.Spec.Compression.Akita.Replay.FinalKey30
import NightstreamFPrime.Spec.Compression.Akita.Replay.Schedules

/-! Axiom audits for the drawn fold challenges on the state of a run (`Replay/FinalDrawn.lean`)
and the games of the packing levels with the weights as theorems (`Replay/FinalGame.lean`,
`Replay/FinalKey20.lean`, `Replay/FinalKey30.lean`), and
instances of the hypotheses of these theorems at the folds of `layout20` and `layout30` (both
rows). The guards: at the root of `layout30` the plan of `stage3.end` is `F.setup` (`Build.fold`)
and is `SetupPlan.ofBatch (root r) (groupOrder 3) F.low F.logBasis` (`planned`),
`F.stage3 = some setupVars` (`offloaded`), `F.low = (root r).low` (`low`, `blocked`) and
`F.logBasis = 3` (`gadget` with `key30_root_numbers`); the other packing folds have no Stage 3
(`direct`). -/

#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Op.drawsGroup
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.squeezeOf
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.drawnShape
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.squeezeOf_of_shape
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.KeepsDrawn
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.keepsDrawn_bind
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.keepsDrawn_pure
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.keepsDrawn_throw
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.keepsDrawn_get
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.keepsDrawn_modify
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.keepsDrawn_mapM
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.keepsDrawn_foldlM
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.keepsDrawn_recorded
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.keepsDrawn_first
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.keepsDrawn_fieldsOf
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.keepsDrawn_pointBefore
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.keepsDrawn_absorb
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.keepsDrawn_record
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.keepsDrawn_squeeze
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.keepsDrawn_message
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.keepsDrawn_stage2Of
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.keepsDrawn_value
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.keepsDrawn_weightsOf
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.keepsDrawn_holds
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.keepsDrawn_checkWith
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.keepsDrawn_terminalWith
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.keepsDrawn_decodeExts
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.step_drawn
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.bind_draw_drawn
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.step_squeeze_drawn
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.find_drawn_grows
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.drawnIn_of_run
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.drawnIn_before
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.packGroupsIn_one
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.packGroup_fits
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.verifyProof_drawnIn
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.verifyProof_setupFor
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.verifyProof_pack_game
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.verifyProof_batch_game
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.verifyProof_offload_game
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.fold_field
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.key20Root_fits
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.verifyProof_key20_root
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.verifyProof_key20_level1
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.key30Root_low
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.key30Root_numbers
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.key30Root_pow
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.key30Root_fits
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.key30RootGroup
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.key30RootShape
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Key30RootFold
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.key30Root_fields
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.key30Root_fold
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.verifyProof_key30_root

namespace AkitaFinalGuards

open NightstreamFPrime.Spec.Compression.Akita NightstreamFPrime.Spec.Compression.Akita.Replay
open NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer

/-- `drawShape`: a script whose first squeeze of the group has the family of the layout. -/
example (G : FoldGroup) (i : ℕ) : drawnShape [(.squeeze 0 0 G, i)] 0 [G] = true := by
  simp [drawnShape, squeezeOf, Op.drawsGroup]

/-- `G`, `grouped`, `ringed`, `degreed` of `verifyProof_batch_game` and
`verifyProof_offload_game` at the root of `layout20` and at the root and level 1 of `layout30`. -/
example (zMax rice : ℕ) : ∃ G : Fin 3 → FoldGroup,
    (layout20 zMax rice).folds.head?.map (·.groups) = some ((List.finRange 3).map G) ∧
      (∀ g, (G g).ring = (Key20.root.shape g).innerDegree) ∧
      ∀ g, (G g).degree = Key20.root.carrierDegree g :=
  ⟨fun g => ⟨(Key20.root.shape g).blocks, (Key20.root.shape g).innerDegree, Key20.root.carrierDegree g,
    (Key20.family 0 g).1, (Key20.family 0 g).2, false⟩, rfl, fun _ => rfl, fun _ => rfl⟩

example (r : Key30.Row) (zMax rice : ℕ) : ∃ G : Fin 3 → FoldGroup,
    (layout30 r zMax rice).folds.head?.map (·.groups) = some ((List.finRange 3).map G) ∧
      (∀ g, (G g).ring = ((Key30.root r).shape g).innerDegree) ∧
      ∀ g, (G g).degree = (Key30.root r).carrierDegree g :=
  ⟨fun g => ⟨((Key30.root r).shape g).blocks, ((Key30.root r).shape g).innerDegree, (Key30.root r).carrierDegree g,
    (Key30.family 0 g).1, (Key30.family 0 g).2, false⟩, rfl, fun _ => rfl, fun _ => rfl⟩

example (r : Key30.Row) (zMax rice : ℕ) : ∃ G : Fin 2 → FoldGroup,
    (layout30 r zMax rice).folds[1]?.map (·.groups) = some ((List.finRange 2).map G) ∧
      (∀ g, (G g).ring = ((Key30.level₁ r).shape g).innerDegree) ∧
      ∀ g, (G g).degree = (Key30.level₁ r).carrierDegree g :=
  ⟨fun g => ⟨((Key30.level₁ r).shape g).blocks, ((Key30.level₁ r).shape g).innerDegree, (Key30.level₁ r).carrierDegree g,
    (Key30.family 1 g).1, (Key30.family 1 g).2, false⟩, rfl, fun _ => rfl, fun _ => rfl⟩

/-- `G`, `grouped`, `ringed`, `degreed` of `verifyProof_pack_game` at level 1 of `layout20` and at
levels 2 and 3 of `layout30`. -/
example (zMax rice : ℕ) : ∃ G : FoldGroup,
    (layout20 zMax rice).folds[1]?.map (·.groups) = some [G] ∧
      G.ring = Key20.level₁.shape.innerDegree ∧ G.degree = Key20.level₁.opening.carrierDegree :=
  ⟨_, rfl, rfl, rfl⟩

example (r : Key30.Row) (zMax rice : ℕ) : ∃ G : FoldGroup,
    (layout30 r zMax rice).folds[2]?.map (·.groups) = some [G] ∧
      G.ring = (Key30.level₂ r).shape.innerDegree ∧
      G.degree = (Key30.level₂ r).opening.carrierDegree :=
  ⟨_, rfl, rfl, rfl⟩

example (r : Key30.Row) (zMax rice : ℕ) : ∃ G : FoldGroup,
    (layout30 r zMax rice).folds[3]?.map (·.groups) = some [G] ∧
      G.ring = (Key30.level₃ r).shape.innerDegree ∧ G.degree = (Key30.level₃ r).opening.carrierDegree :=
  ⟨_, rfl, rfl, rfl⟩

/-- `shaped`, `slots`, `opening` and `digited` of `verifyProof_pack_game` at level 1 of `layout20`
and at levels 2 and 3 of `layout30`. -/
example (zMax rice : ℕ) : (layout20 zMax rice).folds[1]?.map (·.fused) =
    some (.packing ⟨[⟨0, Key20.level₁.shape, Key20.level₁.opening, Key20.level₁.fold⟩],
      ⟨Key20.level₁.opening.openDegree, Key20.level₁.opening.openRank, Key20.level₁.opening.chain⟩,
      Key20.quotientDigits⟩) := rfl

example (zMax rice : ℕ) : ∀ r ∈ Key30.rows, (layout30 r zMax rice).folds[2]?.map (·.fused) =
    some (.packing ⟨[⟨0, (Key30.level₂ r).shape, (Key30.level₂ r).opening, (Key30.level₂ r).fold⟩],
      ⟨(Key30.level₂ r).opening.openDegree, (Key30.level₂ r).opening.openRank,
        (Key30.level₂ r).opening.chain⟩, Key30.quotientDigits⟩) := by
  intro r member
  simp only [Key30.rows, List.mem_cons, List.mem_nil_iff, or_false] at member
  rcases member with rfl | rfl <;> rfl

example (zMax rice : ℕ) : ∀ r ∈ Key30.rows, (layout30 r zMax rice).folds[3]?.map (·.fused) =
    some (.packing ⟨[⟨0, (Key30.level₃ r).shape, (Key30.level₃ r).opening, (Key30.level₃ r).fold⟩],
      ⟨(Key30.level₃ r).opening.openDegree, (Key30.level₃ r).opening.openRank, (Key30.level₃ r).opening.chain⟩,
      Key30.quotientDigits⟩) := by
  intro r member
  simp only [Key30.rows, List.mem_cons, List.mem_nil_iff, or_false] at member
  rcases member with rfl | rfl <;> rfl

/-- `numbers` of `verifyProof_batch_game` and `verifyProof_offload_game`: each group of the root of
`layout20`, of the root of `layout30` and of level 1 of `layout30` (both rows). The form of
`verifyProof_pack_game` is in `tests/AxiomsAkitaPacking.lean`. -/
example : ∀ k, (batchGroup Key20.root (fun _ => []) (fun _ => []) (fun _ => []) (fun _ => Ext.one)
    k).Fits Key20.rootVars := by
  intro k
  fin_cases k
  · exact ⟨⟨6, rfl⟩, ⟨10, rfl⟩, ⟨7, rfl⟩, ⟨7, rfl⟩, ⟨6, rfl⟩, ⟨5, rfl⟩, ⟨6, rfl⟩, ⟨5, rfl⟩,
      by decide, by decide, by decide, by decide, by decide, by decide, by decide, by decide,
      by decide, by decide, by decide⟩
  · exact ⟨⟨7, rfl⟩, ⟨10, rfl⟩, ⟨7, rfl⟩, ⟨7, rfl⟩, ⟨6, rfl⟩, ⟨5, rfl⟩, ⟨6, rfl⟩, ⟨5, rfl⟩,
      by decide, by decide, by decide, by decide, by decide, by decide, by decide, by decide,
      by decide, by decide, by decide⟩
  · exact ⟨⟨6, rfl⟩, ⟨9, rfl⟩, ⟨7, rfl⟩, ⟨7, rfl⟩, ⟨6, rfl⟩, ⟨5, rfl⟩, ⟨6, rfl⟩, ⟨5, rfl⟩,
      by decide, by decide, by decide, by decide, by decide, by decide, by decide, by decide,
      by decide, by decide, by decide⟩

example : ∀ r ∈ Key30.rows, ∀ k, (batchGroup (Key30.root r) (fun _ => []) (fun _ => [])
    (fun _ => []) (fun _ => Ext.one) k).Fits Key30.rootVars := by
  intro r member k
  simp only [Key30.rows, List.mem_cons, List.mem_nil_iff, or_false] at member
  rcases member with rfl | rfl <;> fin_cases k
  all_goals first
    | exact ⟨⟨7, rfl⟩, ⟨9, rfl⟩, ⟨8, rfl⟩, ⟨8, rfl⟩, ⟨6, rfl⟩, ⟨5, rfl⟩, ⟨6, rfl⟩, ⟨5, rfl⟩,
        by decide, by decide, by decide, by decide, by decide, by decide, by decide, by decide,
        by decide, by decide, by decide⟩
    | exact ⟨⟨7, rfl⟩, ⟨11, rfl⟩, ⟨8, rfl⟩, ⟨8, rfl⟩, ⟨6, rfl⟩, ⟨5, rfl⟩, ⟨6, rfl⟩, ⟨5, rfl⟩,
        by decide, by decide, by decide, by decide, by decide, by decide, by decide, by decide,
        by decide, by decide, by decide⟩
    | exact ⟨⟨7, rfl⟩, ⟨11, rfl⟩, ⟨7, rfl⟩, ⟨8, rfl⟩, ⟨6, rfl⟩, ⟨5, rfl⟩, ⟨6, rfl⟩, ⟨5, rfl⟩,
        by decide, by decide, by decide, by decide, by decide, by decide, by decide, by decide,
        by decide, by decide, by decide⟩

example : ∀ r ∈ Key30.rows, ∀ k, (batchGroup (Key30.level₁ r) (fun _ => []) (fun _ => [])
    (fun _ => []) (fun _ => Ext.one) k).Fits Key30.vars₁ := by
  intro r member k
  simp only [Key30.rows, List.mem_cons, List.mem_nil_iff, or_false] at member
  rcases member with rfl | rfl <;> fin_cases k
  all_goals exact ⟨⟨7, rfl⟩, ⟨9, rfl⟩, ⟨8, rfl⟩, ⟨8, rfl⟩, ⟨6, rfl⟩, ⟨5, rfl⟩, ⟨6, rfl⟩, ⟨5, rfl⟩,
    by decide, by decide, by decide, by decide, by decide, by decide, by decide, by decide,
    by decide, by decide, by decide⟩

/-- `challenge` with `challenged` of `verifyProof_pack_game` (the form of the batch theorems is in
`tests/AxiomsAkitaPacking.lean`). -/
example (s : Replayer) (level : ℕ) (S : Shape) (O : OpeningShape S) :
    ∃ challenge : Fin S.blocks → Rq O.carrierDegree, ∀ i, challenge i =
      Poly.toRq _ (((drawnIn s level 0).map fun c => Poly.ofSparse O.carrierDegree
        (c.map fun x => (x.1 / (S.innerDegree / O.carrierDegree), x.2))).getD i
          (Poly.zero O.carrierDegree)) :=
  ⟨_, fun _ => rfl⟩

/-- `blocked` of `verifyProof_offload_game` at the root of `layout30`: `F.low` is `(root r).low`
(the guard below) and the root is blocked for it. -/
example : ∀ r ∈ Key30.rows, (Key30.root r).Blocked (Key30.root r).low := fun r member => by
  rw [key30_root_low r member]
  exact Key30.root_blocked r member

#guard Key30.rows.all fun r => ((layout30 r 0 0).folds[0]?.map fun F =>
  F.setup == some (SetupPlan.ofBatch (Key30.root r) (groupOrder 3) F.low F.logBasis) &&
    F.stage3 == some Key30.setupVars && F.low == (Key30.root r).low && F.logBasis == 3) ==
  some true

#guard ((layout20 0 0).folds.take 2).all fun F => F.stage3.isNone
#guard Key30.rows.all fun r => (((layout30 r 0 0).folds.drop 1).take 3).all fun F => F.stage3.isNone

end AkitaFinalGuards
