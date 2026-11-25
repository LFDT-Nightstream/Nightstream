import tests.AkitaVectors
import NightstreamFPrime.Spec.Compression.Akita.Replay.Terminal

/-! The gate of the terminal checks (`Replay/Terminal.lean`). For each positive vector the replay of `Replay/Transcript.lean` (the one
replay of `tests/AkitaAllVectors.lean`) runs from the public inputs and the proof bytes, and its
terminal operation runs `terminalRun` on the recorded messages and values (`terminalMessagesIn`,
`terminalChallengesIn`: the fold challenges of the terminal group are the drawn challenges of the
replay) and records its values (`Derived.terminal`). `tests/AkitaVectors.lean` compares its values
and its three checks with the events. This file checks what has no event, with nothing from the
events: the rows of `A` and the digest of the last block of `t`.

- The hypotheses of `step_terminal_accepts` hold on the replay: the drawn challenges of the
  terminal group are `foldChallenges` of its message, the payload decodes, the group has the
  degree `2^6` of the shape, a coefficient `±1` and stride 1.
- The digest of the last block of `t`, from the rows of `A·z` with `A` the setup prefix of the
  seed, is the digest of the proof; it fails for `z + 1` at one coefficient and for `t + 1` at one
  element. `θ + 1` fails the trace check. -/

namespace AkitaTerminalVectors

open NightstreamFPrime.Spec.Compression.Akita
open NightstreamFPrime.Spec.Compression.Akita.Replay
open AkitaVectors (Replayed)

def failWith {α : Type} (message : String) : IO α := throw <| IO.userError message

/-- The digest of the last block of `t` for a decoded `z` and the sent `t`, from the rows of
`A·z`. -/
def tDigestOf (S : Shape) (msgs : TerminalMessages) (ch : TerminalChallenges)
    (rows : List (Array Fq)) : Option ByteArray := do
  let inv ← Poly.inverse (Nat.log 2 S.innerDegree) (challenge S ch (S.blocks - 1))
  return rowsDigest S (← tLastOf S msgs ch inv rows)

/-- The gate on the replay of one positive file. -/
def gate (r : Replayed) : IO Unit := do
  let name := r.v.name
  let d := r.d
  let s := r.state
  let level := r.layout.folds.length
  let msgs := Replayer.terminalMessagesIn s level
  let ch := Replayer.terminalChallengesIn s level
  let T := r.layout.terminal
  let S := T.shape
  let some setup := r.pub.setupFor T.setupCount | failWith s!"{name}: the setup prefix"
  let some out := d.terminal | failWith s!"{name}: the replay has no terminal values"
  -- the rows of `A·z` for `z + 1` at one coefficient, in a task
  let z' := out.z.set! 0 (out.z[0]! + 1)
  let rows' := Task.spawn fun _ => (List.range S.innerRank).map (aRow S setup z')
  let some (_, _, bytes) := d.foldMessages.find? fun (l, g, _) => l == level && g == 0
    | failWith s!"{name}: no message of the terminal group"
  unless foldChallenges T.group bytes == some (Replayer.drawnIn s level 0) ∧
      (Golomb.decode T.params msgs.payload).isSome ∧ T.group.degree = S.innerDegree ∧
      0 < T.group.pm1 ∧ T.group.ring = T.group.degree ∧ 2 ^ (5 + 1) = S.innerDegree ∧
      1 ≤ S.blocks do
    failWith s!"{name}: a hypothesis of step_terminal_accepts does not hold"
  unless tDigestOf S msgs ch out.rows == some msgs.tDigest do failWith s!"{name}: the digest of t"
  if tDigestOf S msgs ch rows'.get == some msgs.tDigest then
    failWith s!"{name}: a changed coefficient of z keeps the digest of t"
  let t' := msgs.t.set! 0 (msgs.t[0]!.add Fq.one)
  if tDigestOf S { msgs with t := t' } ch out.rows == some msgs.tDigest then
    failWith s!"{name}: a changed element of t keeps the digest of t"
  if ((ch.scale + Ext.one) * out.opening).equals ch.target then
    failWith s!"{name}: a changed θ passes the trace check"
  IO.println s!"terminal {name}: the digest of t from the rows of A of the replay holds; it fails \
    for z + 1 at one coefficient and for t + 1 at one element; θ + 1 fails the trace check"

/-- The hypotheses `power`, `quarter`, `two` and `blocks` of `terminalChecks_ok_iff` hold at the
terminal of `key20.aks` (`Key30.terminal = Key20.terminal`). -/
example : 2 ^ Nat.log 2 Key20.terminal.innerDegree = Key20.terminal.innerDegree ∧
    4 * (Key20.terminal.innerDegree / 4) = Key20.terminal.innerDegree ∧
    ((2 * (Key20.terminal.innerDegree / 4) : ℕ) : NightstreamFPrime.Spec.Compression.Akita.Zq) ≠ 0 ∧
    1 ≤ Key20.terminal.blocks := by
  have degree : Key20.terminal.innerDegree = 2 ^ 6 := rfl
  rw [degree, Nat.log_pow (by norm_num)]
  exact ⟨rfl, rfl, by decide, by decide⟩

end AkitaTerminalVectors
