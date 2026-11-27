import NightstreamFPrime.Spec.Compression.Akita.Replay.ShapeLayout20
import NightstreamFPrime.Spec.Compression.Akita.Replay.FinalKey20

/-! The games of the levels of `key20.aks` with the layout `layout20` as the only schedule
hypothesis (Akita revision `d50f700`; `Replay/ShapeLayout20.lean`, `Replay/FinalKey20.lean`,
`Replay/RunProof.lean`).

- `verifyProof_key20_root_of`, `verifyProof_key20_level1_of`, `verifyProof_key20_trace_of` (level
  2) and `verifyProof_key20_traceSq_of` (levels 3 to 5): the games of the levels on the state before
  `fused.end` of `pub.ops` at `fusedIndex pub.layout ℓ`. The hypotheses are the layout, the
  acceptance, the decoded messages, the setup prefixes (facts of the seed), the state, and the
  choices of the basis, the challenge of the specification and `w`. The layout facts of the
  levels (`shaped`, `rows`, `based`, `low`, `quarter`, `two`, `fits`, `rowsFit`, the route,
  `compressed`, `arities`, `leaf`, `positive`, `small`, `digits`) are proved for `fold20 ℓ`.
- `replay_matches20` and `verifyProof_accept_iff20`: with the setup prefix of the terminal, the
  replay of a script of `layout20` never stops with `mismatch`, so `runProof` accepts exactly
  when the proof decodes with no byte left and the replay has no stop. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

open NightstreamFPrime.Spec.Compression.Lib NightstreamFPrime.Spec.Compression.Game
open NightstreamFPrime.Spec.Compression (Game)
open NightstreamFPrime.Spec.Compression.Akita

open Replayer

/-- With the setup prefix of the terminal, the replay of a script of `layout20` on decoded
messages does not stop with `mismatch`. -/
theorem replay_matches20 {zMax rice : ℕ} {pub : Public} {proof : ByteArray} {ops : Array Op}
    (layout : pub.layout = layout20 zMax rice) (scripted : script pub.layout pub.plan = .ok ops)
    {messages : Array Message} {rest : ℕ} (decoded : decode ops proof = .ok (messages, rest))
    {full : Array Fq} (terminalSetup : pub.setupFor pub.layout.terminal.setupCount = some full)
    (k : ℕ) : (replay pub ops proof messages).1 ≠ some (.mismatch k) :=
  replay_matches pub proof messages ops decoded (script20 (layout ▸ scripted)).1 terminalSetup k

/-- With the setup prefix of the terminal, `runProof` accepts a proof of `layout20` exactly
when it decodes with no byte left and the replay of `pub.ops` has no stop. -/
theorem verifyProof_accept_iff20 {zMax rice : ℕ} (pub : Public) (proof : ByteArray)
    (layout : pub.layout = layout20 zMax rice) (scripted : script pub.layout pub.plan = .ok pub.ops)
    {full : Array Fq} (terminalSetup : pub.setupFor pub.layout.terminal.setupCount = some full) :
    runProof pub proof = .verdict .accept ↔
      ∃ messages, decode pub.ops proof = .ok (messages, 0) ∧
        (replay pub pub.ops proof messages).1 = none := by
  have ends := (script20 (layout ▸ scripted)).1
  constructor
  · exact verifyProof_replay pub proof scripted ends terminalSetup
  · rintro ⟨messages, decoded, none_⟩
    rw [verifyProof_script pub proof scripted, verifyBytes_verdict _ _ proof messages decoded]
    congr 1
    exact (verdict_accept_iff pub proof decoded ends terminalSetup).mpr none_

variable {E : Type} [Field E] [Fintype E] [DecidableEq E] [Algebra Zq E] [Degree2 E]

/-- The root of `key20.aks`: when `runProof` accepts a proof of `layout20`, `foldCheck` of
`Batch.liftedReduction` of the root runs on the path of the code on the state before the
`fused.end` of the level 0 to acceptance with the recorded `(fused.point, witness_evaluation)`. -/
theorem verifyProof_key20_root_of {zMax rice : ℕ} (pub : Public) (proof : ByteArray)
    (layout : pub.layout = layout20 zMax rice)
    (accepted : runProof pub proof = .verdict .accept) {full : Array Fq}
    (terminalSetup : pub.setupFor pub.layout.terminal.setupCount = some full)
    {messages : Array Message} {rest : ℕ} (decoded : decode pub.ops proof = .ok (messages, rest))
    {s : Replayer} (state : stateBefore pub proof messages pub.ops (fusedIndex pub.layout 0) = s)
    {setup : Array Fq} (expanded : pub.setupFor (fold20 0).setupCount = some setup)
    (β : ∀ g, Module.Basis (Fin (Key20.root.planes g)) Zq E)
    (basis : ∀ g t, β g t = (extBasis.getD t Ext.zero).into) (challenge : Key20.root.Challenge)
    (challenged : ∀ k i, challenge (groupOrder 3 k) i = Poly.toRq _
      ((drawnCarrier Key20.root s 0 k).getD i
        (Poly.zero (Key20.root.carrierDegree (groupOrder 3 k)))))
    (w : Table) :
    ∃ (fits : Key20.root.cells ≤ 2 ^ (fold20 0).vars)
      (room : Key20.root.quotientRoom Key20.quotientDigits ≤ 2 ^ (fold20 0).vars)
      (rowsFit : Key20.root.rowCount + 1 ≤ 2 ^ (fold20 0).batch),
      (foldCheck (Batch.liftedReduction β (batchStatementFor Key20.root setup (claimPointIn s 0)
          (dataIn pub s 0) (fun k => .root (groupOrder 3 k).val) (rowsIn s 0 (fold20 0))
          (claimsIn s 0 (fold20 0))) challenge (fold20 0).vars fits 8 Key20.quotientDigits room
          (fold20 0).batch rowsFit)
          8 (Key20.root.gaps Key20.quotientDigits).support Key20.root.compressed w).Runs
        (foldPathIn E pub s 0 (fold20 0) setup (fold20 0).vars (fold20 0).batch)
        (some ((recordedIn s ⟨0, "fused.point", none⟩).toList.map Ext.into,
          (firstIn s ⟨0, "witness_evaluation", none⟩).into)) := by
  obtain ⟨scripted, ends, levels⟩ := verifyProof_script20 layout accepted
  obtain ⟨at_, ⟨r, a, e, shape⟩, game, drawn⟩ := levels 0
  exact verifyProof_key20_root pub proof scripted ends terminalSetup accepted decoded at_
    (layout20_fold zMax rice 0) shape game drawn state expanded β basis challenge challenged w

/-- Level 1 of `key20.aks`: when `runProof` accepts a proof of `layout20`, `foldCheck` of
`liftedReduction` of the level runs on the path of the code on the state before the `fused.end` of
the level 1 to acceptance with the recorded `(fused.point, witness_evaluation)`. -/
theorem verifyProof_key20_level1_of {zMax rice : ℕ} (pub : Public) (proof : ByteArray)
    (layout : pub.layout = layout20 zMax rice)
    (accepted : runProof pub proof = .verdict .accept) {full : Array Fq}
    (terminalSetup : pub.setupFor pub.layout.terminal.setupCount = some full)
    {messages : Array Message} {rest : ℕ} (decoded : decode pub.ops proof = .ok (messages, rest))
    {s : Replayer} (state : stateBefore pub proof messages pub.ops (fusedIndex pub.layout 1) = s)
    {setup : Array Fq} (expanded : pub.setupFor (fold20 1).setupCount = some setup)
    (β : Module.Basis (Fin Key20.level₁.opening.planes) Zq E)
    (basis : ∀ t, β t = (extBasis.getD t Ext.zero).into)
    (challenge : Fin Key20.level₁.shape.blocks → Rq Key20.level₁.opening.carrierDegree)
    (challenged : ∀ i, challenge i = Poly.toRq _ (((drawnIn s 1 0).map fun c =>
      Poly.ofSparse Key20.level₁.opening.carrierDegree (c.map fun x =>
        (x.1 / (Key20.level₁.shape.innerDegree / Key20.level₁.opening.carrierDegree), x.2))).getD i
          (Poly.zero Key20.level₁.opening.carrierDegree)))
    (w : Table) :
    ∃ (fits : Key20.level₁.opening.cells Key20.level₁.fold ≤ 2 ^ (fold20 1).vars)
      (room : Key20.level₁.opening.quotientRoom Key20.level₁.fold Key20.quotientDigits ≤
        2 ^ (fold20 1).vars)
      (rowsFit : Key20.level₁.opening.rowCount + 1 ≤ 2 ^ (fold20 1).batch),
      (foldCheck (liftedReduction β (packStatementFor Key20.level₁.opening setup
          (claimPointIn s 1 0) (dataIn pub s 1) .witness (rowsIn s 1 (fold20 1))
          (claimsIn s 1 (fold20 1))) challenge (fold20 1).vars fits Key20.quotientDigits room
          (fold20 1).batch rowsFit)
          Key20.level₁.shape.certifiedBase
          (Key20.level₁.opening.gaps Key20.level₁.fold Key20.quotientDigits).support
          (Key20.level₁.opening.compressed Key20.level₁.fold) w).Runs
        (foldPathIn E pub s 1 (fold20 1) setup (fold20 1).vars (fold20 1).batch)
        (some ((recordedIn s ⟨1, "fused.point", none⟩).toList.map Ext.into,
          (firstIn s ⟨1, "witness_evaluation", none⟩).into)) := by
  obtain ⟨scripted, ends, levels⟩ := verifyProof_script20 layout accepted
  obtain ⟨at_, ⟨r, a, e, shape⟩, game, drawn⟩ := levels 1
  exact verifyProof_key20_level1 pub proof scripted ends terminalSetup accepted decoded at_
    (layout20_fold zMax rice 1) shape game drawn state expanded β basis challenge challenged w

/-- A natural number below the modulus and not zero is not zero in `Zq`. -/
theorem shape20_cast_ne_zero {n : ℕ} (positive : 0 < n) (small : n < fieldModulus) :
    ((n : ℕ) : Zq) ≠ 0 :=
  fun zero => absurd (Nat.le_of_dvd positive ((ZMod.natCast_eq_zero_iff _ _).mp zero))
    (not_le.mpr small)

/-- Level 2 of `key20.aks`, a trace level on the coefficient route: when `runProof` accepts a
proof of `layout20`, `foldCheck` of `traceReduction` runs on the path of the code on the state
before the `fused.end` of the level 2 to acceptance with the recorded
`(fused.point, witness_evaluation)`. -/
theorem verifyProof_key20_trace_of {zMax rice : ℕ} (pub : Public) (proof : ByteArray)
    (layout : pub.layout = layout20 zMax rice)
    (accepted : runProof pub proof = .verdict .accept) {full : Array Fq}
    (terminalSetup : pub.setupFor pub.layout.terminal.setupCount = some full)
    {messages : Array Message} {rest : ℕ} (decoded : decode pub.ops proof = .ok (messages, rest))
    {s : Replayer} (state : stateBefore pub proof messages pub.ops (fusedIndex pub.layout 2) = s)
    {setup : Array Fq} (expanded : pub.setupFor (fold20 2).setupCount = some setup)
    (challenge : Fin Key20.level₂.shape.blocks → Rq Key20.level₂.shape.innerDegree)
    (challenged : ∀ i, challenge i = Poly.toRq Key20.level₂.shape.innerDegree
      (((drawnIn s 2 0).map (Poly.ofSparse Key20.level₂.shape.innerDegree)).getD i
        (Poly.zero Key20.level₂.shape.innerDegree)))
    (w : Table) :
    ∃ (quarter : 4 * (Key20.level₂.shape.innerDegree / 4) = Key20.level₂.shape.innerDegree)
      (fits : Key20.level₂.opening.cells Key20.level₂.fold ≤ 2 ^ (fold20 2).vars)
      (rowsFit : Key20.level₂.opening.traceRowCount + 1 ≤ 2 ^ (fold20 2).batch),
      (foldCheck (traceReduction (traceStatementFor (E := E) Key20.level₂.opening quarter setup
          (traceValuesIn s 2 (fold20 2)) (dataIn pub s 2) .witness (rowsIn s 2 (fold20 2))
          (claimsIn s 2 (fold20 2))) challenge (fold20 2).vars fits (fold20 2).batch rowsFit)
          Key20.level₂.shape.certifiedBase (Key20.level₂.opening.binarySupport Key20.level₂.fold)
          (Key20.level₂.opening.compressed Key20.level₂.fold) w).Runs
        (foldPathIn E pub s 2 (fold20 2) setup (fold20 2).vars (fold20 2).batch)
        (some ((recordedIn s ⟨2, "fused.point", none⟩).toList.map Ext.into,
          (firstIn s ⟨2, "witness_evaluation", none⟩).into)) := by
  obtain ⟨scripted, ends, levels⟩ := verifyProof_script20 layout accepted
  obtain ⟨at_, ⟨r, a, e, shape⟩, game, _⟩ := levels 2
  have sized := Key20.traces_sized Key20.level₂ (by simp [Key20.traces])
  refine ⟨by decide, sized.1, sized.2.1, ?_⟩
  exact verifyProof_trace_game pub proof scripted ends terminalSetup accepted decoded at_ shape game
    state (T := ⟨Key20.level₂.shape, Key20.level₂.opening, Key20.level₂.fold⟩) rfl expanded rfl (by decide) rfl (by decide) (shape20_cast_ne_zero (by decide) (by decide))
    challenge challenged _ _ w rfl (by decide) (by decide) (by decide)

/-- The levels 3 to 5 of `key20.aks`, trace levels on the Euclidean route: when `runProof`
accepts a proof of `layout20`, `foldCheckSq` of `traceReduction` with the norm stage runs on the
path of the code on the state before the `fused.end` of the level to acceptance with the recorded
`(fused.point, witness_evaluation)`. -/
theorem verifyProof_key20_traceSq_of {zMax rice : ℕ} (ℓ : Fin 6) (L : Key20.Level)
    (listed : (ℓ, L) ∈ [(3, Key20.level₃), (4, Key20.level₄), (5, Key20.level₅)])
    (pub : Public) (proof : ByteArray) (layout : pub.layout = layout20 zMax rice)
    (accepted : runProof pub proof = .verdict .accept) {full : Array Fq}
    (terminalSetup : pub.setupFor pub.layout.terminal.setupCount = some full)
    {messages : Array Message} {rest : ℕ} (decoded : decode pub.ops proof = .ok (messages, rest))
    {s : Replayer} (state : stateBefore pub proof messages pub.ops (fusedIndex pub.layout ℓ) = s)
    {setup : Array Fq} (expanded : pub.setupFor (fold20 ℓ).setupCount = some setup)
    (challenge : Fin L.shape.blocks → Rq L.shape.innerDegree)
    (challenged : ∀ i, challenge i = Poly.toRq L.shape.innerDegree
      (((drawnIn s ℓ 0).map (Poly.ofSparse L.shape.innerDegree)).getD i
        (Poly.zero L.shape.innerDegree)))
    (w : Table) :
    ∃ (quarter : 4 * (L.shape.innerDegree / 4) = L.shape.innerDegree)
      (fits : L.opening.cells L.fold ≤ 2 ^ (fold20 ℓ).vars)
      (rowsFit : L.opening.traceRowCount + 1 ≤ 2 ^ (fold20 ℓ).batch),
      (foldCheckSq (traceReduction (traceStatementFor (E := E) L.opening quarter setup
          (traceValuesIn s ℓ (fold20 ℓ)) (dataIn pub s ℓ) .witness (rowsIn s ℓ (fold20 ℓ))
          (claimsIn s ℓ (fold20 ℓ))) challenge (fold20 ℓ).vars fits (fold20 ℓ).batch rowsFit)
          L.shape.certifiedBase (L.opening.binarySupport L.fold) (L.opening.compressed L.fold)
          (fold20 ℓ).limit (L.shape.normKernel L.fold) w).Runs
        (foldPathIn E pub s ℓ (fold20 ℓ) setup (fold20 ℓ).vars (fold20 ℓ).batch)
        (some ((recordedIn s ⟨ℓ, "fused.point", none⟩).toList.map Ext.into,
          (firstIn s ⟨ℓ, "witness_evaluation", none⟩).into)) := by
  obtain ⟨scripted, ends, levels⟩ := verifyProof_script20 layout accepted
  obtain ⟨at_, ⟨r, a, e, shape⟩, game, _⟩ := levels ℓ
  simp only [List.mem_cons, Prod.mk.injEq, List.not_mem_nil, or_false] at listed
  have sized := Key20.traces_sized L (by
    rcases listed with ⟨-, rfl⟩ | ⟨-, rfl⟩ | ⟨-, rfl⟩ <;> simp [Key20.traces])
  rcases listed with ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩
  all_goals
    refine ⟨by decide, sized.1, sized.2.1, ?_⟩
    exact verifyProof_traceSq_game pub proof scripted ends terminalSetup accepted decoded at_ shape
      game state (T := ⟨_, _, _⟩) rfl expanded rfl rfl (by decide) (by decide)
      (shape20_cast_ne_zero (by decide) (by decide)) challenge challenged _ _ w (by decide)
      (by decide) (by decide) rfl (by decide) (by decide)

end NightstreamFPrime.Spec.Compression.Akita.Replay
