import NightstreamFPrime.Spec.Compression.Akita.Replay.CausalCells

/-! The page of the stage `k` of the model from the state of the replay before the work of the
stage `k`, for `key20.aks` (Akita revision `d50f700`; ePrint 2026/1983, revision of 2026-11-23,
Section 10.3; `docs/akita-work/w20/design.md`, decisions D2 and D3).

- `pageAt20 … k` has the data `noteAt20 … k` (with the trace of the stage), the cells
  `cellsAt20 … k` and the nonce 0. `pageAt20_eq`: for an accepted proof it is `page20 … k`.
- `pageAt20_causal`: the page `k < 473` is a function of the events of the sponge before the
  squeeze of the work of the stage `k` and of the last blocks.
- `pageAt20_events`: on a script of `layout20`, the pages `0 … k` give these events. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

open NightstreamFPrime.Spec.Compression.Akita

open Replayer

variable {E : Type} [Field E] [Algebra Zq E] [Degree2 E]
  (pub : Public) (proof proof' : ByteArray) (messages messages' : Array Message)
  (tLast tLast' : ℕ → Rq Key20.terminal.innerDegree)
  (eLast eLast' : Rq Key20.terminal.innerDegree)

/-- The page `k` of a proof from the state of the replay before the work of the stage `k`: the
data `noteAt20 … k`, the cells `cellsAt20 … k` and the nonce 0. -/
noncomputable def pageAt20 (k : ℕ) : Key20.Wire E :=
  ⟨noteAt20 pub proof messages tLast eLast k, cellsAt20 E pub proof messages k, 0⟩

variable {pub proof proof' messages messages'}

/-- For an accepted proof of `layout20` with its decoded messages, `pageAt20 … k` is the page
`page20 … k` of theorem (R). -/
theorem pageAt20_eq [Fintype E] [DecidableEq E] {zMax rice : ℕ}
    (layout : pub.layout = layout20 zMax rice)
    (accepted : runProof pub proof = .verdict .accept) {full : Array Fq}
    (terminalSetup : pub.setupFor pub.layout.terminal.setupCount = some full) {rest : ℕ}
    (decoded : decode pub.ops proof = .ok (messages, rest)) (k : ℕ) :
    pageAt20 (E := E) pub proof messages tLast eLast k =
      page20 pub proof messages tLast eLast k := by
  unfold pageAt20 page20
  rw [cellsAt20_eq E layout accepted terminalSetup decoded k]

/-- The page `k < 473` is a function of the public input, the events of the sponge before the
squeeze of the work of the stage `k` and the last blocks. -/
theorem pageAt20_causal {k : ℕ} (inside : k < 473) {rest rest' : ℕ}
    (decoded : decode pub.ops proof = .ok (messages, rest))
    (decoded' : decode pub.ops proof' = .ok (messages', rest'))
    (ran : (runFrom pub proof messages (pub.ops.zipIdx.toList.take (stageIndex20 pub k))
      (channelStart pub)).1 = .ok ())
    (ran' : (runFrom pub proof' messages' (pub.ops.zipIdx.toList.take (stageIndex20 pub k))
      (channelStart pub)).1 = .ok ())
    (events : eventsAt20 pub proof messages k = eventsAt20 pub proof' messages' k) :
    pageAt20 (E := E) pub proof messages tLast eLast k =
      pageAt20 pub proof' messages' tLast eLast k := by
  unfold pageAt20
  rw [noteAt20_causal tLast eLast inside decoded decoded' ran ran' events,
    cellsAt20_causal E decoded decoded' ran ran' events]

/-- The trace of the page `k < 473`. -/
theorem pageAt20_trace {k : ℕ} (inside : k < 473) :
    (pageAt20 (E := E) pub proof messages tLast eLast k).data.trace =
      traceAt20 pub proof messages k := by
  show (if k < 473 then traceAt20 pub proof messages k else Extract.Trace.empty 5 16384) = _
  rw [ite_eq_left inside]

/-- On a script of `layout20`, the pages `0 … k` give the events of the sponge before the squeeze
of the work of the stage `k`: two proofs with their decoded messages, whose runs have no stop
before this work and have the same pages `j ≤ k`, for each last blocks, have the same
`eventsAt20 … k`. -/
theorem pageAt20_events (erased : pub.ops.toList.map Op.erase = skeleton (layout20 0 0)) {k : ℕ}
    (inside : k < 473) {rest rest' : ℕ}
    (decoded : decode pub.ops proof = .ok (messages, rest))
    (decoded' : decode pub.ops proof' = .ok (messages', rest'))
    (ran : (runFrom pub proof messages (pub.ops.zipIdx.toList.take (stageIndex20 pub k))
      (channelStart pub)).1 = .ok ())
    (ran' : (runFrom pub proof' messages' (pub.ops.zipIdx.toList.take (stageIndex20 pub k))
      (channelStart pub)).1 = .ok ())
    (pages : ∀ j ≤ k, pageAt20 (E := E) pub proof messages tLast eLast j =
      pageAt20 pub proof' messages' tLast' eLast' j) :
    eventsAt20 pub proof messages k = eventsAt20 pub proof' messages' k :=
  (stage_alike inside decoded decoded' ran ran' fun j le =>
    traceAt20_inj pub proof proof' messages messages' erased decoded decoded' (Nat.lt_of_le_of_lt le inside)
      ((pageAt20_trace (E := E) tLast eLast (Nat.lt_of_le_of_lt le inside)).symm.trans
        ((congrArg (fun w : Key20.Wire E => w.data.trace) (pages j le)).trans
          (pageAt20_trace (E := E) tLast' eLast' (Nat.lt_of_le_of_lt le inside))))).2

end NightstreamFPrime.Spec.Compression.Akita.Replay
