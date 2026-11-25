import tests.AkitaVectors
import NightstreamFPrime.Spec.Compression.Akita.Replay.SamplerSigns
import NightstreamFPrime.Spec.Compression.Akita.Replay.SamplerStage

/-! The gate of the fold challenge sampler: the replay of `Replay/Transcript.lean` from the public inputs and the proof bytes
(the one replay of `tests/AkitaAllVectors.lean`) squeezes the message of the fold challenges of
each group (`Derived.foldMessages`) and draws its challenges with `foldChallenges`
(`Derived.drawn`). Each equals the `fold_challenges` event of
its level and group, pair by pair in the ring of `A`. The events are read for the comparison
only.

- The groups: 9 in each `key20.aks` vector and 11 in each `key30.aks` vector, in the order of the
  squeezes.
- The response nonce enters the message through the sponge only (`read_fold_response`: the nonce
  word is absorbed before the work of `FoldChallengeWork` and the squeeze). The retry vectors
  have the nonce 1 at one level (from the decoded proof). With the nonce 0 absorbed at that level
  and the work not checked, the message of the group differs and its challenges differ from the
  event.
- Mutations: one changed bit of a message (the sign bit of the first accepted candidate), `pm1`
  one less and `pm2` one more, one challenge more, the filter toggled, and the stride 1 at a
  packing group: each gives no challenge or other challenges.
- The stage `fold_challenges` on made messages (`stageGate`): the draw of the replay
  (`Replayer.draw`, `draw_run`).
- Instances of the hypotheses of the theorems: each group has `0 < rankSpace`, `0 < degree`,
  `128 ≤ K` and `stride · degree = ring`; the two `example`s are a `Sparse.Valid` and a
  `Positions`. The group of level 1 of `key20.aks` is `level₁Group` of `level₁_search_code`. -/

namespace AkitaSamplerVectors

open Lean (Json)
open NightstreamFPrime.Spec.Compression.Akita.Replay
open AkitaVectors (Replayed nat messageOps eventSparse withoutEnds)

def failWith {α : Type} (message : String) : IO α := throw <| IO.userError message

example : Sparse.Valid 64 [(0, 1), (5, -2)] := ⟨by decide, by decide, by decide⟩

example : Positions 4 1 [0] [2] := ⟨rfl, by decide, by decide, by decide, by decide⟩

/-- The groups of a layout in the order of the squeezes: level, group and family. -/
def groupsOf (layout : Layout) : List (ℕ × ℕ × FoldGroup) :=
  (layout.folds.zipIdx.flatMap fun (F, level) => F.groups.zipIdx.map fun (G, g) => (level, g, G)) ++
    [(layout.folds.length, 0, layout.terminal.group)]

/-- The index of the first candidate of a message that gives a challenge. -/
def firstIndex (G : FoldGroup) (bytes : ByteArray) : ℕ :=
  let B := chooseAt (binomials G.degree (max G.pm1 G.pm2))
  let F := Filter.of G.degree
  ((List.range G.candidates).find? fun j =>
    ((candidate G B (readBits bytes (j * G.candidateBits) G.candidateBits)).filter
      fun c => !G.filtered || F.accepts c).isSome).getD 0

def flipBit (bytes : ByteArray) (bit : ℕ) : ByteArray :=
  bytes.set! (bit / 8) (bytes.get! (bit / 8) ^^^ (1 <<< (bit % 8)).toUInt8)

/-- The operation `i` of the replay, with the nonce `nonce` absorbed in place of the response
nonce of `level` and with no work check of `FoldChallengeWork` at `level`. -/
def withNonce (pub : Public) (proof : ByteArray) (messages : Array Message) (level nonce : ℕ)
    (i : ℕ) : Op → ReplayM Unit
  | .foldNonce l width =>
      if l = level then do
        discard (Replayer.message i messages)
        Replayer.absorb (le64 nonce)
      else Replayer.step pub proof messages i (.foldNonce l width)
  | .work (.foldChallengeWork l) lv bits =>
      if l = level ∧ bits ≠ 0 then do
        let m ← Replayer.message i messages
        let .nonce n _ := m.data | throw (.mismatch i)
        Replayer.absorb (le64 n)
        discard (Replayer.squeeze 8)
      else Replayer.step pub proof messages i (.work (.foldChallengeWork l) lv bits)
  | op => Replayer.step pub proof messages i op

/-- The squeezed message of group 0 of `level` with the nonce `nonce` at `level`: the replay of
the script `ops` up to that squeeze. -/
def messageWith (pub : Public) (ops : Array Op) (proof : ByteArray) (messages : Array Message)
    (level nonce : ℕ) : Option ByteArray :=
  let last := (ops.findIdx? fun | .squeeze l 0 _ => l == level | _ => false).getD ops.size
  let run : ReplayM Unit := (ops.zipIdx.extract 0 (last + 1)).forM fun (op, i) =>
    withNonce pub proof messages level nonce i op
  let (_, s) := (run.run).run ⟨channel pub.session pub.descriptor, 0, {}, {}, {}, {}⟩
  (s.derived.foldMessages.find? fun (l, g, _) => l == level && g == 0).map (·.2.2)

/-- The stage `fold_challenges` (`Replayer.draw`, `draw_run`): each draw of the replay of a
positive file gives its challenges. The draw of a group of the script on a made message of 255
bytes of its length gives `noWork` at the squeeze of that group when the group is filtered (each
candidate is out of range: `rankBits = ⌈log_2 rankSpace⌉` and `rankSpace` is not a power of 2),
and its challenges when the group is unfiltered (`foldChallenges_isSome`). -/
def stageGate (name : String) (ops : Array Op) (d : Derived) : IO Unit := do
  unless d.drawn.size = d.foldMessages.size do failWith s!"{name}: a draw of the replay stops"
  let mut filtered := 0
  let mut unfiltered := 0
  for (op, i) in ops.zipIdx do
    let .squeeze level group G := op | continue
    let made := ByteArray.mk (Array.replicate G.messageBytes 255)
    let start : Replayer := ⟨Duplex.empty, 0, {}, {}, {}, {}⟩
    match ((Replayer.draw i level group G made).run).run start, G.filtered with
    | (.error (.noWork j site), _), true =>
        unless j = i ∧ site = (Site.foldChallengeGroup level group).text do
          failWith s!"{name}: the stop of group {group} of level {level} is at another site"
        filtered := filtered + 1
    | (.ok (), s), false =>
        unless s.derived.drawn.size = 1 do failWith s!"{name}: no challenges recorded"
        unfiltered := unfiltered + 1
    | _, _ => failWith s!"{name}: group {group} of level {level} with the made message"
  IO.println s!"stage {name}: each draw of the replay gives its challenges; made messages: \
    {filtered} filtered groups give noWork at their squeeze, {unfiltered} unfiltered groups give \
    their challenges"

/-- The gate on the replay of one positive file: returns the count of groups and of challenges.
The replays with another nonce run the script without the end checks (`withoutEnds`). -/
def gate (r : Replayed) : IO (ℕ × ℕ) := do
  let v := r.v
  let o := r.o
  let name := v.name
  let layout := r.layout
  let d := r.d
  let pub := r.pub
  let messages := r.messages
  let ops := withoutEnds r.ops
  let groups := groupsOf layout
  if v.schedule == "crates/ns-compress/artifacts/key20.aks" && groupAt layout 1 0 != some level₁Group then
    failWith s!"{name}: the group of level 1 is not `level₁Group` of `level₁_search_code`"
  let events := o.events.filter (·.op == "fold_challenges")
  unless groups.length = d.foldMessages.size ∧ groups.length = d.drawn.size ∧
      groups.length = events.size do
    failWith s!"{name}: {groups.length} groups, {d.foldMessages.size} messages, \
      {events.size} events"
  let mut challenges := 0
  let mut mutations := 0
  let mut sampling := 0
  for ((((level, g, G), (level', g', bytes)), (level'', g'', found)), e) in
      ((groups.zip d.foldMessages.toList).zip d.drawn.toList).zip events.toList do
    let label := s!"{name}: group {g} of level {level}"
    unless level = level' ∧ g = g' ∧ level = level'' ∧ g = g'' ∧ level = e.level ∧
        (nat e.json ["group"]).toOption == some g do
      failWith s!"{label}: the order of the groups differs"
    unless 0 < G.rankSpace ∧ 0 < G.degree ∧ 128 ≤ G.candidateBits ∧
        G.ring / G.degree * G.degree = G.ring do
      failWith s!"{label}: a hypothesis of the theorems of the sampler does not hold"
    let expected ← IO.ofExcept (eventSparse e)
    let start ← IO.monoNanosNow
    unless foldChallenges G bytes == some found do failWith s!"{label}: the draw is not the sampler"
    sampling := sampling + ((← IO.monoNanosNow) - start)
    unless found == expected do failWith s!"{label}: the challenges differ from the event"
    challenges := challenges + found.length
    let flipped := flipBit bytes (firstIndex G bytes * G.candidateBits)
    let others : List (String × Option (List Sparse)) :=
      [("a changed bit", foldChallenges G flipped),
        ("pm1 - 1", foldChallenges { G with pm1 := G.pm1 - 1, pm2 := G.pm2 + 1 } bytes),
        ("one challenge more", foldChallenges { G with coordinates := G.coordinates + 1 } bytes),
        ("the filter toggled", foldChallenges { G with filtered := !G.filtered } bytes)] ++
      (if G.ring = G.degree then [] else [("stride 1", foldChallenges { G with ring := G.degree } bytes)])
    for (mutation, result) in others do
      if result == some expected then failWith s!"{label}: {mutation} gives the challenges"
      mutations := mutations + 1
  let nonces := ((messageOps ops).toList.zip messages.toList).filterMap fun
    | (.foldNonce level _, m) => match m.data with
      | .nonce n _ => if n = 0 then none else some (level, n)
      | _ => none
    | _ => none
  for (level, n) in nonces do
    let some (_, _, bytes) := d.foldMessages.find? fun (l, g, _) => l == level && g == 0
      | failWith s!"{name}: no message at level {level}"
    unless messageWith pub ops v.proof messages level n == some bytes do
      failWith s!"{name}: the replay with the nonce {n} at level {level} differs"
    let some zero := messageWith pub ops v.proof messages level 0
      | failWith s!"{name}: no message with the nonce 0 at level {level}"
    let some (_, _, G) := groups.find? fun (l, g, _) => l == level && g == 0
      | failWith s!"{name}: no group at level {level}"
    let some e := events.find? (·.level == level) | failWith s!"{name}: no event at {level}"
    if zero == bytes ∨ foldChallenges G zero == some (← IO.ofExcept (eventSparse e)) then
      failWith s!"{name}: the challenges of level {level} do not depend on the nonce"
  stageGate name ops d
  IO.println s!"sampler {name}: {groups.length} groups, {challenges} challenges equal to the \
    events ({sampling / 1000000} ms in foldChallenges); {mutations} mutations each differ; nonces \
    not 0 (level, nonce) {nonces}: with the nonce 0 the message and the challenges differ"
  return (groups.length, challenges)

end AkitaSamplerVectors
