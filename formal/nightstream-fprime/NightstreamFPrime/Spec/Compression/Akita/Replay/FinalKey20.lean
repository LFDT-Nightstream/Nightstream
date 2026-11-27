import NightstreamFPrime.Spec.Compression.Akita.Replay.FinalGame
import NightstreamFPrime.Spec.Compression.Akita.Replay.Schedules

/-! The games of the packing levels of `key20.aks` when `runProof` accepts, with no hypothesis
of the layout (Akita revision `d50f700`; the folds of `layout20`, `Key20.root`, `Key20.level₁`).

- `fold_field`: a field of the fold `F` at the place `i` of a list of folds is the value that the
  list computes there.
- `key20Root_fits`: `PackGroup.Fits` of each group of the root.
- `verifyProof_key20_root` (the root, a fold with opening groups) and `verifyProof_key20_level1`
  (level 1, a packing level with one group): `verifyProof_batch_game` and `verifyProof_pack_game`
  for the fold `F` of `layout20 zMax rice` at the place 0 and 1 (`fold`). The hypotheses are the
  shapes of the script at its `fused.end`, `endsFit`, the setup prefixes, and the choices of the
  basis, the challenge of the specification and `w`; the bounds `fits`, `room` and `rowsFit` of
  the reduction exist.
- `key20_root_before`, `key20_level1_before`: the same games after a run without a stop before
  the `fused.end`, to the result of `foldRunIn` on the state before it. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

open NightstreamFPrime.Spec.Compression.Lib NightstreamFPrime.Spec.Compression.Game
open NightstreamFPrime.Spec.Compression (Game)
open NightstreamFPrime.Spec.Compression.Akita

open Replayer

theorem fold_field {l : List Fold} {i : ℕ} {F : Fold} (fold : l[i]? = some F) {α : Type}
    (f : Fold → α) {x : α} (computed : l[i]?.map f = some x) : f F = x := by
  rw [fold] at computed
  exact Option.some.inj computed

/-- `PackGroup.Fits` of each group of the root of `key20.aks`. -/
theorem key20Root_fits : ∀ k, (batchGroup Key20.root (fun _ => []) (fun _ => []) (fun _ => [])
    (fun _ => Ext.one) k).Fits Key20.rootVars := by
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

variable {E : Type} [Field E] [Fintype E] [DecidableEq E] [Algebra Zq E] [Degree2 E]

/-- The root of `key20.aks` after a run without a stop before the `fused.end` at `c` of the fold
`F` of the root of `layout20 zMax rice`: `foldCheck` of `Batch.liftedReduction` of the root runs on
the path of the code to the result of `foldRunIn` on the state before `c`. -/
theorem key20_root_before {zMax rice : ℕ} (pub : Public) (proof : ByteArray) {ops : Array Op}
    {messages : Array Message} (aligned : ReadsAligned messages ops.zipIdx.toList 0) {c : ℕ}
    {level : ℕ} {F : Fold} (fold : (layout20 zMax rice).folds[0]? = some F)
    (game : gameShape (ops.zipIdx.toList.take c) level F = true)
    (drawShape : drawnShape (ops.zipIdx.toList.take c) level F.groups = true)
    (before : (runFrom pub proof messages (ops.zipIdx.toList.take c) (channelStart pub)).1 =
      .ok ()) {s : Replayer}
    (state : stateBefore pub proof messages ops c = s) {setup : Array Fq}
    (expanded : pub.setupFor F.setupCount = some setup)
    (β : ∀ g, Module.Basis (Fin (Key20.root.planes g)) Zq E)
    (basis : ∀ g t, β g t = (extBasis.getD t Ext.zero).into) (challenge : Key20.root.Challenge)
    (challenged : ∀ k i, challenge (groupOrder 3 k) i = Poly.toRq _
      ((drawnCarrier Key20.root s level k).getD i
        (Poly.zero (Key20.root.carrierDegree (groupOrder 3 k)))))
    (w : Table) :
    ∃ (fits : Key20.root.cells ≤ 2 ^ F.vars)
      (room : Key20.root.quotientRoom Key20.quotientDigits ≤ 2 ^ F.vars)
      (rowsFit : Key20.root.rowCount + 1 ≤ 2 ^ F.batch),
      (foldCheck (Batch.liftedReduction β (batchStatementFor Key20.root setup (claimPointIn s level)
          (dataIn pub s level) (fun k => .root (groupOrder 3 k).val) (rowsIn s level F)
          (claimsIn s level F)) challenge F.vars fits 8 Key20.quotientDigits room F.batch rowsFit)
          8 (Key20.root.gaps Key20.quotientDigits).support Key20.root.compressed w).Runs
        (foldPathIn E pub s level F setup F.vars F.batch)
        ((foldRunIn pub s level F).map fun o => (o.1.map Ext.into, o.2.into)) := by
  rw [show (layout20 zMax rice).folds = (layout20 0 0).folds from rfl] at fold
  have vars : F.vars = Key20.rootVars := fold_field fold Fold.vars rfl
  have batch : F.batch = Key20.rootBatch := fold_field fold Fold.batch rfl
  refine ⟨vars ▸ Key20.root_sized.1, vars ▸ Key20.root_room, batch ▸ Key20.root_sized.2.1, ?_⟩
  exact batch_game_before pub proof aligned game drawShape before state
    (P := ⟨(List.finRange 3).map fun k => ⟨(groupOrder 3 k).val, Key20.root.shape (groupOrder 3 k),
      Key20.root.opening (groupOrder 3 k), Key20.root.fold (groupOrder 3 k)⟩,
      ⟨Key20.root.openDegree, Key20.root.openRank, Key20.root.chain⟩, Key20.quotientDigits⟩)
    (fold_field fold Fold.fused rfl) (fold_field fold Fold.stage3 rfl) expanded
    (fun k => .root (groupOrder 3 k).val) (fold_field fold (fun F => (F.rhs, F.openingRow)) rfl) 8
    (fold_field fold (fun F => 2 ^ F.logBasis) (by decide)) (fold_field fold Fold.low rfl) β basis
    challenge challenged _ _ _ w (fold_field fold Fold.euclidean rfl)
    (fold_field fold Fold.compressed (by decide))
    (fold_field fold (fun F => (productStages F.logBasis).map (·.1)) (by decide))
    (fold_field fold (fun F => leafDegree F.logBasis) (by decide)) rfl rfl rfl
    (fun k => by rw [vars]; exact key20Root_fits k)
    (fun g => ⟨(Key20.root.shape g).blocks, (Key20.root.shape g).innerDegree,
      Key20.root.carrierDegree g, (Key20.family 0 g).1, (Key20.family 0 g).2, false⟩)
    (fold_field fold Fold.groups rfl) (fun _ => rfl) (fun _ => rfl) 0

/-- The root of `key20.aks`: when `runProof` accepts and the check `fused.end` at `c` has the
fold `F` of the root of `layout20 zMax rice`, `foldCheck` of `Batch.liftedReduction` of the root
runs on the path of the code to acceptance with the recorded `(fused.point, witness_evaluation)`. -/
theorem verifyProof_key20_root {zMax rice : ℕ} (pub : Public) (proof : ByteArray) {ops : Array Op}
    (scripted : script pub.layout pub.plan = .ok ops) (ends : endsFit ops.zipIdx.toList = true)
    {full : Array Fq} (terminalSetup : pub.setupFor pub.layout.terminal.setupCount = some full)
    (accepted : runProof pub proof = .verdict .accept) {messages : Array Message} {rest : ℕ}
    (decoded : decode ops proof = .ok (messages, rest)) {c : ℕ} {key : Key} {level : ℕ}
    {F : Fold} (at_ : ops[c]? = some (.check key (.fused level F)))
    (fold : (layout20 zMax rice).folds[0]? = some F) {i a e : ℕ}
    (shape : foldShape (ops.zipIdx.toList.take c) level F i a e = true)
    (game : gameShape (ops.zipIdx.toList.take c) level F = true)
    (drawShape : drawnShape (ops.zipIdx.toList.take c) level F.groups = true) {s : Replayer}
    (state : stateBefore pub proof messages ops c = s) {setup : Array Fq}
    (expanded : pub.setupFor F.setupCount = some setup)
    (β : ∀ g, Module.Basis (Fin (Key20.root.planes g)) Zq E)
    (basis : ∀ g t, β g t = (extBasis.getD t Ext.zero).into) (challenge : Key20.root.Challenge)
    (challenged : ∀ k i, challenge (groupOrder 3 k) i = Poly.toRq _
      ((drawnCarrier Key20.root s level k).getD i
        (Poly.zero (Key20.root.carrierDegree (groupOrder 3 k)))))
    (w : Table) :
    ∃ (fits : Key20.root.cells ≤ 2 ^ F.vars)
      (room : Key20.root.quotientRoom Key20.quotientDigits ≤ 2 ^ F.vars)
      (rowsFit : Key20.root.rowCount + 1 ≤ 2 ^ F.batch),
      (foldCheck (Batch.liftedReduction β (batchStatementFor Key20.root setup (claimPointIn s level)
          (dataIn pub s level) (fun k => .root (groupOrder 3 k).val) (rowsIn s level F)
          (claimsIn s level F)) challenge F.vars fits 8 Key20.quotientDigits room F.batch rowsFit)
          8 (Key20.root.gaps Key20.quotientDigits).support Key20.root.compressed w).Runs
        (foldPathIn E pub s level F setup F.vars F.batch)
        (some ((recordedIn s ⟨level, "fused.point", none⟩).toList.map Ext.into,
          (firstIn s ⟨level, "witness_evaluation", none⟩).into)) := by
  have none_ := (replay_of_decoded pub proof scripted ends terminalSetup accepted decoded).2
  subst state
  obtain ⟨fits, room, rowsFit, runs⟩ := key20_root_before (E := E) pub proof
    (decode_aligned decoded) fold game drawShape (replay_step_ok pub proof messages none_ at_).1
    rfl expanded β basis challenge challenged w
  exact ⟨fits, room, rowsFit, runs_level pub proof messages none_ at_ shape runs⟩

/-- Level 1 of `key20.aks` after a run without a stop before the `fused.end` at `c` of the fold
`F` of level 1 of `layout20 zMax rice`: `foldCheck` of `liftedReduction` of the level runs on the
path of the code to the result of `foldRunIn` on the state before `c`. -/
theorem key20_level1_before {zMax rice : ℕ} (pub : Public) (proof : ByteArray) {ops : Array Op}
    {messages : Array Message} (aligned : ReadsAligned messages ops.zipIdx.toList 0) {c : ℕ}
    {level : ℕ} {F : Fold} (fold : (layout20 zMax rice).folds[1]? = some F)
    (game : gameShape (ops.zipIdx.toList.take c) level F = true)
    (drawShape : drawnShape (ops.zipIdx.toList.take c) level F.groups = true)
    (before : (runFrom pub proof messages (ops.zipIdx.toList.take c) (channelStart pub)).1 =
      .ok ()) {s : Replayer}
    (state : stateBefore pub proof messages ops c = s) {setup : Array Fq}
    (expanded : pub.setupFor F.setupCount = some setup)
    (β : Module.Basis (Fin Key20.level₁.opening.planes) Zq E)
    (basis : ∀ t, β t = (extBasis.getD t Ext.zero).into)
    (challenge : Fin Key20.level₁.shape.blocks → Rq Key20.level₁.opening.carrierDegree)
    (challenged : ∀ i, challenge i = Poly.toRq _ (((drawnIn s level 0).map fun c =>
      Poly.ofSparse Key20.level₁.opening.carrierDegree (c.map fun x =>
        (x.1 / (Key20.level₁.shape.innerDegree / Key20.level₁.opening.carrierDegree), x.2))).getD i
          (Poly.zero Key20.level₁.opening.carrierDegree)))
    (w : Table) :
    ∃ (fits : Key20.level₁.opening.cells Key20.level₁.fold ≤ 2 ^ F.vars)
      (room : Key20.level₁.opening.quotientRoom Key20.level₁.fold Key20.quotientDigits ≤ 2 ^ F.vars)
      (rowsFit : Key20.level₁.opening.rowCount + 1 ≤ 2 ^ F.batch),
      (foldCheck (liftedReduction β (packStatementFor Key20.level₁.opening setup
          (claimPointIn s level 0) (dataIn pub s level) .witness (rowsIn s level F)
          (claimsIn s level F)) challenge F.vars fits Key20.quotientDigits room F.batch rowsFit)
          Key20.level₁.shape.certifiedBase
          (Key20.level₁.opening.gaps Key20.level₁.fold Key20.quotientDigits).support
          (Key20.level₁.opening.compressed Key20.level₁.fold) w).Runs
        (foldPathIn E pub s level F setup F.vars F.batch)
        ((foldRunIn pub s level F).map fun o => (o.1.map Ext.into, o.2.into)) := by
  rw [show (layout20 zMax rice).folds = (layout20 0 0).folds from rfl] at fold
  have vars : F.vars = Key20.level₁.vars := fold_field fold Fold.vars rfl
  have batch : F.batch = Key20.level₁.batch := fold_field fold Fold.batch rfl
  refine ⟨vars ▸ Key20.level₁_sized.1, vars ▸ Key20.level₁_sized.2.1,
    batch ▸ Key20.level₁_sized.2.2.1, ?_⟩
  exact pack_game_before pub proof aligned game drawShape before state
    (P := ⟨[⟨0, Key20.level₁.shape, Key20.level₁.opening, Key20.level₁.fold⟩],
      ⟨Key20.level₁.opening.openDegree, Key20.level₁.opening.openRank, Key20.level₁.opening.chain⟩,
      Key20.quotientDigits⟩)
    (fold_field fold Fold.fused rfl) (fold_field fold Fold.stage3 rfl) expanded
    (fold_field fold (fun F => (F.rhs, F.openingRow)) rfl)
    (fold_field fold (fun F => 2 ^ F.logBasis) (by decide)) (fold_field fold Fold.low rfl) β basis
    challenge challenged _ _ _ w (fold_field fold Fold.euclidean rfl)
    (fold_field fold Fold.compressed (by decide))
    (fold_field fold (fun F => (productStages F.logBasis).map (·.1)) (by decide))
    (fold_field fold (fun F => leafDegree F.logBasis) (by decide)) rfl rfl rfl
    (by rw [vars]
        exact ⟨⟨6, rfl⟩, ⟨10, rfl⟩, ⟨7, rfl⟩, ⟨7, rfl⟩, ⟨6, rfl⟩, ⟨5, rfl⟩, ⟨6, rfl⟩, ⟨5, rfl⟩,
          by decide, by decide, by decide, by decide, by decide, by decide, by decide, by decide,
          by decide, by decide, by decide⟩)
    ⟨Key20.level₁.shape.blocks, Key20.level₁.shape.innerDegree, Key20.level₁.opening.carrierDegree,
      (Key20.family 1 0).1, (Key20.family 1 0).2, false⟩
    (fold_field fold Fold.groups rfl) rfl rfl

/-- Level 1 of `key20.aks`: when `runProof` accepts and the check `fused.end` at `c` has the
fold `F` of level 1 of `layout20 zMax rice`, `foldCheck` of `liftedReduction` of the level runs on
the path of the code to acceptance with the recorded `(fused.point, witness_evaluation)`. -/
theorem verifyProof_key20_level1 {zMax rice : ℕ} (pub : Public) (proof : ByteArray)
    {ops : Array Op} (scripted : script pub.layout pub.plan = .ok ops)
    (ends : endsFit ops.zipIdx.toList = true) {full : Array Fq}
    (terminalSetup : pub.setupFor pub.layout.terminal.setupCount = some full)
    (accepted : runProof pub proof = .verdict .accept) {messages : Array Message} {rest : ℕ}
    (decoded : decode ops proof = .ok (messages, rest)) {c : ℕ} {key : Key} {level : ℕ}
    {F : Fold} (at_ : ops[c]? = some (.check key (.fused level F)))
    (fold : (layout20 zMax rice).folds[1]? = some F) {i a e : ℕ}
    (shape : foldShape (ops.zipIdx.toList.take c) level F i a e = true)
    (game : gameShape (ops.zipIdx.toList.take c) level F = true)
    (drawShape : drawnShape (ops.zipIdx.toList.take c) level F.groups = true) {s : Replayer}
    (state : stateBefore pub proof messages ops c = s) {setup : Array Fq}
    (expanded : pub.setupFor F.setupCount = some setup)
    (β : Module.Basis (Fin Key20.level₁.opening.planes) Zq E)
    (basis : ∀ t, β t = (extBasis.getD t Ext.zero).into)
    (challenge : Fin Key20.level₁.shape.blocks → Rq Key20.level₁.opening.carrierDegree)
    (challenged : ∀ i, challenge i = Poly.toRq _ (((drawnIn s level 0).map fun c =>
      Poly.ofSparse Key20.level₁.opening.carrierDegree (c.map fun x =>
        (x.1 / (Key20.level₁.shape.innerDegree / Key20.level₁.opening.carrierDegree), x.2))).getD i
          (Poly.zero Key20.level₁.opening.carrierDegree)))
    (w : Table) :
    ∃ (fits : Key20.level₁.opening.cells Key20.level₁.fold ≤ 2 ^ F.vars)
      (room : Key20.level₁.opening.quotientRoom Key20.level₁.fold Key20.quotientDigits ≤ 2 ^ F.vars)
      (rowsFit : Key20.level₁.opening.rowCount + 1 ≤ 2 ^ F.batch),
      (foldCheck (liftedReduction β (packStatementFor Key20.level₁.opening setup
          (claimPointIn s level 0) (dataIn pub s level) .witness (rowsIn s level F)
          (claimsIn s level F)) challenge F.vars fits Key20.quotientDigits room F.batch rowsFit)
          Key20.level₁.shape.certifiedBase
          (Key20.level₁.opening.gaps Key20.level₁.fold Key20.quotientDigits).support
          (Key20.level₁.opening.compressed Key20.level₁.fold) w).Runs
        (foldPathIn E pub s level F setup F.vars F.batch)
        (some ((recordedIn s ⟨level, "fused.point", none⟩).toList.map Ext.into,
          (firstIn s ⟨level, "witness_evaluation", none⟩).into)) := by
  have none_ := (replay_of_decoded pub proof scripted ends terminalSetup accepted decoded).2
  subst state
  obtain ⟨fits, room, rowsFit, runs⟩ := key20_level1_before (E := E) pub proof
    (decode_aligned decoded) fold game drawShape (replay_step_ok pub proof messages none_ at_).1
    rfl expanded β basis challenge challenged w
  exact ⟨fits, room, rowsFit, runs_level pub proof messages none_ at_ shape runs⟩

end NightstreamFPrime.Spec.Compression.Akita.Replay
