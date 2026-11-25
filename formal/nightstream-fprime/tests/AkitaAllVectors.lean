import tests.AkitaSumcheckVectors
import tests.AkitaSamplerVectors
import tests.AkitaTerminalVectors
import tests.AkitaFusedVectors
import tests.AkitaStage3Vectors
import tests.AkitaPackingVectors

/-! The run of the vector tests: the gates of `tests/AkitaVectors.lean` and of the five other
vector tests on one replay of each positive proof (`replay` of the script, with each end check
and the terminal). The setup prefix of the seed is expanded once, for the largest count of the
four layouts (`Setup.expandTasks`, one task for each page; the four vectors have the same seed),
and the four replays run in parallel tasks. A gate replays again only a changed proof, a changed
layout or a changed script.

Lake does not track the files that the run reads, so `scripts/validate.sh` runs this file
again. -/

namespace AkitaAllVectors

open NightstreamFPrime.Spec.Compression.Akita
open NightstreamFPrime.Spec.Compression.Akita.Replay
open AkitaVectors

#eval show IO Unit from do
  let start ← IO.monoMsNow
  let mut inputs : Array (Vector × Oracles × Layout × Array Op × Public × Array Message) := #[]
  for name in positives do
    let (v, o) ← load name
    gate1 v o
    let levels ← gate2 v o
    IO.println s!"gate 2 {v.name}: norm.square = virtual² and norm.end on {levels} levels"
    let ops ← gate3 v o
    writesGate v ops
    if name == "key20-opening.json" then
      discard <| gate3Negative "key20-opening-negative.json" ops v.proof
    if name == "key30-opening-26-26.json" then
      discard <| gate3Negative "key30-opening-26-26-negative.json" ops v.proof
    let layout ← layoutOf v
    let pub ← publicOf v o layout
    shapedGate v pub
    if name == "key20-opening.json" then
      statementNegative "key20-statement-negative.json" pub v.proof
    digests v
    planGate v layout
    let some (messages, _) := (decode ops v.proof).toOption | fail s!"{name}: decode"
    inputs := inputs.push (v, o, layout, ops, pub, messages)
  let some seed := inputs[0]?.map (·.2.2.2.2.1.seed) | fail "no vector"
  unless inputs.all (·.2.2.2.2.1.seed.data == seed.data) do fail "the vectors have other seeds"
  let count := inputs.foldl (fun m x => max m (setupCount x.2.2.1)) 0
  let loaded ← IO.monoMsNow
  let some setup := Setup.expandTasks seed count | fail "the setup prefix"
  let expanded ← IO.monoMsNow
  IO.println s!"the gates before the replays {loaded - start} ms; the setup prefix of \
    {setup.size} elements {expanded - loaded} ms"
  let runs ← inputs.mapM fun (v, o, layout, ops, pub, messages) => do
    let pub := { pub with setup }
    let replayed := Task.spawn fun _ => replay pub ops v.proof messages
    return (v, o, layout, ops, pub, messages,
      ← IO.mapTask (fun out => return (out, ← IO.monoMsNow)) replayed)
  let vk ← IO.FS.readBinFile "../../crates/ns-compress/artifacts/key30.vk"
  AkitaStage3Vectors.pageGate 0
  AkitaStage3Vectors.pageGate 511
  let mut groups := []
  for (v, o, layout, ops, pub, messages, task) in runs do
    let ((stop, d), done) ← IO.ofExcept task.get
    let r : Replayed := ⟨v, o, layout, ops, pub, messages, stop, d⟩
    IO.println s!"{v.name}: the replay is done {done - expanded} ms after the setup prefix"
    let begin ← IO.monoMsNow
    AkitaPackingVectors.diagnoseStop r
    gate4 r
    orderGate v o pub
    if v.name == "key20-opening.json" then
      gate4Negative "key20-opening-negative.json" ops pub v.proof
    if v.name == "key30-opening-26-26.json" then
      gate4Negative "key30-opening-26-26-negative.json" ops pub v.proof
    AkitaSumcheckVectors.gate r
    let (count, _) ← AkitaSamplerVectors.gate r
    groups := groups ++ [count]
    AkitaTerminalVectors.gate r
    IO.println s!"fused {v.name}:"
    AkitaFusedVectors.gate r
    IO.println s!"packing {v.name}:"
    AkitaPackingVectors.gate r
    if v.schedule == "crates/ns-compress/artifacts/key30.aks" then AkitaStage3Vectors.gate r vk
    IO.println s!"{v.name}: the gates on the replay {(← IO.monoMsNow) - begin} ms"
  unless groups = [9, 9, 11, 11] do fail s!"group counts {groups}"
  IO.println s!"time {(← IO.monoMsNow) - start} ms"

end AkitaAllVectors
