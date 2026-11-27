import NightstreamFPrime.Spec.Compression.Akita.Replay.JoinStage
import NightstreamFPrime.Spec.Compression.Akita.Replay.Resolve

/-! The digests of the last blocks in the state before the work of a stage are the digests of the
terminal, for `key20.aks` (Akita revision `d50f700`, `fold/terminal.rs`; ePrint 2026/1983,
revision of 2026-11-23, Lemma 10.47; `docs/akita-work/w21/design.md`, section 5;
`Replay/Resolve.lean`, `tDigestAt` and `eDigestAt`; `Replay/JoinTerminal.lean`, `Last20`).

- `stateBefore_bytes`, `held_bytes`: a key that no operation from `c` to `c'` writes has the same
  bytes before `c` and before `c'`.
- `held20_tDigest`, `held20_eDigest`: no operation of the skeleton of `layout20` after the work
  of the stage 379 writes the digest of the last block of `t`, and none after the work of the
  stage 472 writes the digest of the last block of `e`.
- `tDigestAt_terminal`, `eDigestAt_terminal`: `tDigestAt … k` for `379 ≤ k` and `eDigestAt … k`
  for `472 ≤ k` are the digests of the terminal.
- `digestAt_of_last`: for last blocks `Last20` of an accepted proof, the hash `pub.digest` of
  their digest inputs gives `tDigestAt … k` and `eDigestAt … k`. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

open NightstreamFPrime.Spec.Compression.Akita

open Replayer

theorem held20_tDigest :
    held20 1218 1521 [⟨5, "terminal_t_last_block_digest", none⟩] = true := by decide +kernel

theorem held20_eDigest :
    held20 1518 1521 [⟨6, "terminal.e_last_block_digest", none⟩] = true := by decide +kernel

section Frames

variable (pub : Public) (proof : ByteArray) (messages : Array Message)

/-- A key that no operation from `c` to `c'` writes has the same bytes before `c` and before
`c'`. -/
theorem stateBefore_bytes {ops : Array Op} {c c' : ℕ} (le : c ≤ c') {K R : List Key}
    (free : frees ((ops.zipIdx.toList.take c').drop c) K R = true) :
    ∀ k ∈ K, (stateBefore pub proof messages ops c').derived.bytes.get? k =
      (stateBefore pub proof messages ops c).derived.bytes.get? k := by
  have split : ops.zipIdx.toList.take c' =
      ops.zipIdx.toList.take c ++ (ops.zipIdx.toList.take c').drop c := by
    conv_lhs => rw [← List.take_append_drop c (ops.zipIdx.toList.take c')]
    rw [List.take_take, Nat.min_eq_left le]
  unfold stateBefore
  rw [split, runFrom_append]
  rcases runFrom pub proof messages (ops.zipIdx.toList.take c) (channelStart pub) with
    ⟨_ | _, s₁⟩
  · exact fun _ _ => rfl
  · exact fun k inside => ((runFrom_grows pub proof messages _ s₁).same k fun written => by
      obtain ⟨y, member, found⟩ := List.mem_flatMap.mp written
      exact frees_writes free inside y member found).2.2.2

/-- On a script of `layout20`, a key that no operation of the skeleton from `c` to `c'` writes
has the same bytes before `c'` and before each operation `m` from `c` to `c'`. -/
theorem held_bytes {ops : Array Op} (erased : ops.toList.map Op.erase = skeleton (layout20 0 0))
    {c c' : ℕ} {K : List Key} (shaped : held20 c c' K = true) {m : ℕ} (lower : c ≤ m)
    (upper : m ≤ c') :
    ∀ k ∈ K, (stateBefore pub proof messages ops c').derived.bytes.get? k =
      (stateBefore pub proof messages ops m).derived.bytes.get? k := by
  unfold held20 at shaped
  simp only [Bool.and_eq_true, decide_eq_true_eq] at shaped
  have base := joined_frees erased _ _ _ _ shaped.2
  exact stateBefore_bytes pub proof messages upper (K := K) (R := [])
    (decide_eq_true fun y member =>
      (of_decide_eq_true base) y ((List.drop_sublist_drop_left _ lower).subset member))

/-- On a script of `layout20`, the bytes of a key that no operation of the skeleton after the
work of the stage `j` writes are, before the work of each stage `k ≥ j`, the bytes before the
terminal. -/
theorem bytesIn_terminal (erased : pub.ops.toList.map Op.erase = skeleton (layout20 0 0))
    {c : ℕ} {key : Key} (shaped : held20 c 1521 [key] = true) {j k : ℕ}
    (start : workIndices20.getD j 0 = c) (late : j ≤ k) :
    bytesIn (stateBefore pub proof messages pub.ops (stageIndex20 pub k)) key =
      bytesIn (terminalState20 pub proof messages) key := by
  have last : pub.ops.size - 1 = 1521 := by rw [size_skeleton erased, skeleton20_length]
  by_cases inside : k < 473
  · obtain ⟨-, -, -, -, -, -, -, -, -, -, -, -, w472⟩ := works20
    have lower := stageIndex20_mono pub late inside
    rw [stageIndex20_lt pub (Nat.lt_of_le_of_lt late inside), start] at lower
    have upper := stageIndex20_mono pub (Nat.le_of_lt_succ inside) (by decide : 472 < 473)
    rw [stageIndex20_lt pub (by decide : 472 < 473), w472] at upper
    have same := held_bytes pub proof messages erased shaped lower
      (Nat.le_trans upper (by decide)) key (List.mem_singleton.mpr rfl)
    show ((stateBefore pub proof messages pub.ops (stageIndex20 pub k)).derived.bytes.get? key).getD
        .empty =
      ((stateBefore pub proof messages pub.ops (pub.ops.size - 1)).derived.bytes.get? key).getD
        .empty
    rw [last, same]
  · have index : stageIndex20 pub k = pub.ops.size - 1 := by
      unfold stageIndex20
      rw [ite_eq_right inside]
    exact congrArg (fun c => bytesIn (stateBefore pub proof messages pub.ops c) key) index

/-- On a script of `layout20`, the digest of the last block of `t` in the state before the work
of a stage `k ≥ 379` is the digest of the terminal. -/
theorem tDigestAt_terminal (erased : pub.ops.toList.map Op.erase = skeleton (layout20 0 0))
    {k : ℕ} (late : 379 ≤ k) :
    tDigestAt pub proof messages k =
      (terminalMessagesIn (terminalState20 pub proof messages) 6).tDigest := by
  obtain ⟨-, -, -, -, -, -, -, -, -, -, -, w379, -⟩ := works20
  exact bytesIn_terminal pub proof messages erased held20_tDigest w379 late

/-- On a script of `layout20`, the digest of the last block of `e` in the state before the work
of a stage `k ≥ 472` is the digest of the terminal. -/
theorem eDigestAt_terminal (erased : pub.ops.toList.map Op.erase = skeleton (layout20 0 0))
    {k : ℕ} (late : 472 ≤ k) :
    eDigestAt pub proof messages k =
      (terminalMessagesIn (terminalState20 pub proof messages) 6).eDigest := by
  obtain ⟨-, -, -, -, -, -, -, -, -, -, -, -, w472⟩ := works20
  exact bytesIn_terminal pub proof messages erased held20_eDigest w472 late

end Frames

variable {E : Type} [Field E] [Fintype E] [DecidableEq E] [Algebra Zq E] [Degree2 E]

omit [Fintype E] [DecidableEq E] in
/-- For an accepted proof of `layout20` and last blocks `Last20`: the hash `pub.digest` of the
digest input of the last block of `t` is the digest in the state before the work of each stage
`k ≥ 379`, and the hash of the digest input of the last block of `e` is the digest in the state
before the work of each stage `k ≥ 472`. -/
theorem digestAt_of_last {pub : Public} {proof : ByteArray} {messages : Array Message}
    {zMax rice : ℕ} (layout : pub.layout = layout20 zMax rice)
    (accepted : runProof pub proof = .verdict .accept)
    {tLast : ℕ → Rq Key20.terminal.innerDegree} {eLast : Rq Key20.terminal.innerDegree}
    (last : Last20 (E := E) pub proof messages tLast eLast) :
    (∀ k, 379 ≤ k → pub.digest (tInput Key20.terminal tLast) = tDigestAt pub proof messages k) ∧
      ∀ k, 472 ≤ k → pub.digest (eInput eLast) = eDigestAt pub proof messages k := by
  have erased : pub.ops.toList.map Op.erase = skeleton (layout20 0 0) := by
    have scripted := verifyProof_scripted accepted
    rw [layout] at scripted
    rw [script_erase scripted, skeleton20]
  exact ⟨fun k late => (rowsDigestOf_input Key20.terminal pub.digest tLast).symm.trans
      (last.2.1.trans (tDigestAt_terminal pub proof messages erased late).symm),
    fun k late => (ringDigest_input pub.digest eLast).symm.trans
      (last.1.trans (eDigestAt_terminal pub proof messages erased late).symm)⟩

end NightstreamFPrime.Spec.Compression.Akita.Replay
