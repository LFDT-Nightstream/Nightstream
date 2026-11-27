import tests.AkitaVectors
import NightstreamFPrime.Spec.Compression.Akita.Replay.Sound20

/-! The run check of `verifyProof_sound20` (`Replay/Sound20.lean`) on the real opening
`tests/vectors/akita/key20-opening.json` (Akita revision `d50f700`). This is an executed test, not
a theorem.

- The public input has the layout `layout20 zMax rice` with the two numbers of the schedule file,
  and the statement shape `root20` (`AkitaVectors.rootOf`). The test checks `Public.shaped`, the
  groups of `root20`, and the setup facts `terminalSetup` and `expanded` of the theorem:
  `Setup.expand` gives a prefix of the seed for the count of the terminal and of each fold
  (`sealed_setupFor`, `Setup.expandTasks_eq`).
- The replay of the proof has no stop. From the events of its sponge the test gives the bounds
  that the key types `Sponge.Base L B C` and `Sponge.Sub S C` of the experiment must have for
  the honest run: the largest count of events of a base (`L`), the largest absorb (`B`), the
  largest squeeze (`C`), and the largest count of squeezes between an absorb and a squeeze
  (`S`). The digest inputs have 544 and 4128 bytes (`size_eInput20`, `size_tInput20`).

Lake does not track the files that the run reads. -/

namespace AkitaSound20Run

open NightstreamFPrime.Spec.Compression.Akita
open NightstreamFPrime.Spec.Compression.Akita.Replay
open Replayer
open AkitaVectors

/-- The counts of the events of a sponge, read from the oldest event. -/
structure Stats where
  events : ℕ := 0
  run : ℕ := 0
  base : ℕ := 0
  before : ℕ := 0
  absorb : ℕ := 0
  squeeze : ℕ := 0
  leading : ℕ := 0
  bases : ℕ := 0
  squeezes : ℕ := 0

def Stats.step (s : Stats) : Duplex.Event → Stats
  | .absorb input =>
      { s with events := s.events + 1, run := 0, absorb := max s.absorb input.size }
  | .squeeze count =>
      { s with events := s.events + 1, run := s.run + 1, base := max s.base (s.events - s.run)
               before := max s.before s.events, squeeze := max s.squeeze count
               leading := max s.leading s.run, bases := s.bases + (if s.run = 0 then 1 else 0)
               squeezes := s.squeezes + 1 }

/-- The counts for the events of a sponge, the newest first. -/
def stats (events : List Duplex.Event) : Stats := events.reverse.foldl Stats.step {}

#eval show IO Unit from do
  let (v, o) ← load "key20-opening.json"
  unless v.schedule == "crates/ns-compress/artifacts/key20.aks" do fail "the schedule"
  let file ← terminalNumbers v
  let pub ← publicOf v o (layout20 file.zMax file.rice)
  unless pub.shaped do fail "Public.shaped"
  unless pub.root.size == root20.size ∧ (pub.root.zip root20).all (fun (a, b) =>
      a.vars == b.vars && a.header.data == b.header.data && a.rows == b.rows) do
    fail "the statement shape is not root20"
  let counts := pub.layout.terminal.setupCount ::
    (List.finRange 6).map fun ℓ => (fold20 ℓ).setupCount
  for count in counts do
    unless 0 < count ∧ (Setup.expandTasks pub.seed count).isSome do
      fail s!"no setup prefix of {count} elements"
  IO.println s!"setup facts: Setup.expand gives the prefixes of {counts} elements"
  let ops ← IO.ofExcept (script pub.layout pub.plan)
  let some (messages, _) := (decode ops v.proof).toOption | fail "decode"
  let some setup := Setup.expandTasks pub.seed (setupCount pub.layout) | fail "the setup prefix"
  let full := { pub with setup }
  let (result, state) := runFrom full v.proof messages ops.zipIdx.toList (channelStart full)
  unless result matches .ok () do fail "the replay stops"
  let s := stats state.sponge.events
  IO.println s!"honest run of key20-opening.json: {s.events} events of the sponge, \
    {s.squeezes} squeezes, {s.bases} bases"
  IO.println s!"bounds for the honest run: L ≥ {s.base} events of a base ({s.before} events \
    before the last squeeze), B ≥ {s.absorb} bytes of an absorb, C ≥ {s.squeeze} bytes of a \
    squeeze, S ≥ {s.leading} squeezes after an absorb; D ≥ 4128 bytes of a digest input"
  unless s.bases == 473 do fail s!"{s.bases} bases, not 473"
  unless s.squeeze ≤ drawBytes20 ∧ s.leading ≤ 3 do
    fail "the honest run does not fit C = 5608 and S = 3"

end AkitaSound20Run
