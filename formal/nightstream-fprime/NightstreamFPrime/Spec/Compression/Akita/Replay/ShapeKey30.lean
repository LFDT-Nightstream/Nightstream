import NightstreamFPrime.Spec.Compression.Akita.Replay.ShapeLayout30
import NightstreamFPrime.Spec.Compression.Akita.Replay.FinalKey30

/-! The game of the offloaded root of a row of `key30.aks` with the layout `layout30` as the only
schedule hypothesis, and the `mismatch` stop of the replay of `layout30` (Akita revision
`d50f700`; `Replay/ShapeLayout30.lean`, `Replay/FinalKey30.lean`).

- `verifyProof_key30_root_of`: `verifyProof_key30_root` on the state before the `fused.end` of the
  level 0 of `pub.ops` at `fusedIndex pub.layout 0`, with `fold30 r 0`.
- `replay_matches30` and `verifyProof_accept_iff30`: with the setup prefix of the terminal, the
  replay of a script of `layout30` never stops with `mismatch`, so `verifyProof` accepts exactly
  when the proof decodes with no byte left and the replay has no stop. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

open NightstreamFPrime.Spec.Compression.Lib NightstreamFPrime.Spec.Compression.Game
open NightstreamFPrime.Spec.Compression (Game)
open NightstreamFPrime.Spec.Compression.Akita

open Replayer

/-- With the setup prefix of the terminal, the replay of a script of `layout30` on decoded
messages does not stop with `mismatch`. -/
theorem replay_matches30 {r : Key30.Row} (member : r ∈ Key30.rows) {zMax rice : ℕ} {pub : Public}
    {proof : ByteArray} {ops : Array Op} (layout : pub.layout = layout30 r zMax rice)
    (scripted : script pub.layout pub.plan = .ok ops) {messages : Array Message} {rest : ℕ}
    (decoded : decode ops proof = .ok (messages, rest)) {full : Array Fq}
    (terminalSetup : pub.setupFor pub.layout.terminal.setupCount = some full) (k : ℕ) :
    (replay pub ops proof messages).1 ≠ some (.mismatch k) :=
  replay_matches pub proof messages ops decoded (script30 member (layout ▸ scripted)).1
    terminalSetup k

/-- With the setup prefix of the terminal, `verifyProof` accepts a proof of `layout30` exactly
when it decodes with no byte left and the replay of `pub.ops` has no stop. -/
theorem verifyProof_accept_iff30 {r : Key30.Row} (member : r ∈ Key30.rows) {zMax rice : ℕ}
    (pub : Public) (proof : ByteArray) (layout : pub.layout = layout30 r zMax rice)
    (scripted : script pub.layout pub.plan = .ok pub.ops) {full : Array Fq}
    (terminalSetup : pub.setupFor pub.layout.terminal.setupCount = some full) :
    verifyProof pub proof = .verdict .accept ↔
      ∃ messages, decode pub.ops proof = .ok (messages, 0) ∧
        (replay pub pub.ops proof messages).1 = none := by
  have ends := (script30 member (layout ▸ scripted)).1
  constructor
  · exact verifyProof_replay pub proof scripted ends terminalSetup
  · rintro ⟨messages, decoded, none_⟩
    rw [verifyProof_script pub proof scripted, verifyBytes_verdict _ _ proof messages decoded]
    congr 1
    exact (verdict_accept_iff pub proof decoded ends terminalSetup).mpr none_

variable {E : Type} [Field E] [Fintype E] [DecidableEq E] [Algebra Zq E] [Degree2 E]

/-- The offloaded root of a row of `key30.aks`: when `verifyProof` accepts a proof of `layout30`,
Stage 3 gives `some o` and `offloadCheck` of `Batch.liftedReduction` of the root over the setup
prefix `flat`, with the split `Batch.codeScan` of the claims, runs on the path of the code on the
state before the `fused.end` of the level 0 to acceptance with the recorded
`(fused.point, witness_evaluation)` and `o`. -/
theorem verifyProof_key30_root_of (r : Key30.Row) (member : r ∈ Key30.rows) {zMax rice : ℕ}
    (pub : Public) (proof : ByteArray) (layout : pub.layout = layout30 r zMax rice)
    (accepted : verifyProof pub proof = .verdict .accept) {full : Array Fq}
    (terminalSetup : pub.setupFor pub.layout.terminal.setupCount = some full)
    {messages : Array Message} {rest : ℕ} (decoded : decode pub.ops proof = .ok (messages, rest))
    {s : Replayer} (state : stateBefore pub proof messages pub.ops (fusedIndex pub.layout 0) = s)
    {setup : Array Fq} (expanded : pub.setupFor (fold30 r 0).setupCount = some setup)
    (β : ∀ g, Module.Basis (Fin ((Key30.root r).planes g)) Zq E)
    (basis : ∀ g t, β g t = (extBasis.getD t Ext.zero).into) (challenge : (Key30.root r).Challenge)
    (challenged : ∀ k i, challenge (groupOrder 3 k) i = Poly.toRq _
      ((drawnCarrier (Key30.root r) s 0 k).getD i
        (Poly.zero ((Key30.root r).carrierDegree (groupOrder 3 k)))))
    (w flat : Table) :
    ∃ o, stage3In s 0 (SetupPlan.ofBatch (Key30.root r) (groupOrder 3) (fold30 r 0).low 3) =
        some o ∧
      (offloadCheck (Batch.liftedReduction β ((batchStatementFor (Key30.root r) setup
          (claimPointIn s 0) (dataIn pub s 0) (fun k => .root (groupOrder 3 k).val)
          (rowsIn s 0 (fold30 r 0)) (claimsIn s 0 (fold30 r 0))).over
            fun p : Fin (2 ^ Key30.setupVars) => flat p)
          challenge (fold30 r 0).vars (key30Root_fold member (layout30_fold r zMax rice 0)).fits 8
          Key30.quotientDigits₀ (key30Root_fold member (layout30_fold r zMax rice 0)).room
          (fold30 r 0).batch (key30Root_fold member (layout30_fold r zMax rice 0)).rowsFit) 8
          ((Key30.root r).gaps Key30.quotientDigits₀).support (Key30.root r).compressed w
          Key30.setupVars flat
          (Batch.codeScan β (batchStatementFor (Key30.root r) setup (claimPointIn s 0)
            (dataIn pub s 0) (fun k => .root (groupOrder 3 k).val) (rowsIn s 0 (fold30 r 0))
            (claimsIn s 0 (fold30 r 0))) challenge (fold30 r 0).vars
            (key30Root_fold member (layout30_fold r zMax rice 0)).fits 8
            Key30.quotientDigits₀ (key30Root_fold member (layout30_fold r zMax rice 0)).room
            (fold30 r 0).batch (key30Root_fold member (layout30_fold r zMax rice 0)).rowsFit
            Key30.setupVars flat (fold30 r 0).low
            (key30Root_fold member (layout30_fold r zMax rice 0)).blocked
            (key30Root_fold member (layout30_fold r zMax rice 0)).lowShort)).Runs
        (offloadPathIn E pub s 0 (fold30 r 0) setup (fold30 r 0).vars (fold30 r 0).batch)
        (some (((recordedIn s ⟨0, "fused.point", none⟩).toList.map Ext.into,
            (firstIn s ⟨0, "witness_evaluation", none⟩).into),
          (o.1.map Ext.into, o.2.into))) := by
  obtain ⟨scripted, ends, ⟨⟨i, a, e, b, t, shape⟩, setupShape⟩, levels⟩ :=
    verifyProof_script30 member layout accepted
  obtain ⟨at_, -, game, drawn⟩ := levels 0
  exact verifyProof_key30_root r member pub proof scripted ends terminalSetup accepted decoded at_
    (layout30_fold r zMax rice 0) shape game setupShape drawn state expanded β basis challenge
    challenged w flat

end NightstreamFPrime.Spec.Compression.Akita.Replay
