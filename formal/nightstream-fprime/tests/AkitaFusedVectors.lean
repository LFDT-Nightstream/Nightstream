import tests.AkitaVectors
import NightstreamFPrime.Spec.Compression.Akita.Replay.Sampler
import NightstreamFPrime.Spec.Compression.Akita.Replay.Fused

/-! The gate of the end check of the fused sum-check (`fused.end`) at the trace levels
(`Replay/Fused.lean`,
`Replay/FusedJoin.lean`). The one replay of each positive vector (`tests/AkitaAllVectors.lean`)
runs from the public inputs and the proof bytes; it evaluates `fused.end` at each level with
evaluation trace with the points, the challenges, `v_W`, the end claim of the rounds and the
fold challenges that it derives (`Derived.drawn`, the sampler on the squeezed bytes), and the
setup prefix of `Public.seed`, the statement input `sample_akita_setup_seed` of the code
(`Setup.sampleSeed`; the vector files do not hold the seed). This file reads the result of the
replay (`Derived.stages`). Nothing comes from the events: the vectors have no event for
`fused.end` (it is in `unchecked`). At each trace level of `key20.aks` (2 to 5) and of `key30.aks`
(4 to 6) `fused.end` holds with the values of the replay, and it fails for `v_W + 1` (the weight
does not read `v_W`, so the check with `v_W + 1` is the record of the replay with `v_W + 1`); at
level 5 of `key20-opening.json` it fails also for a changed setup element and a changed fold
challenge. -/

namespace AkitaFusedVectors

open NightstreamFPrime.Spec.Compression.Akita
open NightstreamFPrime.Spec.Compression.Akita.Replay
open AkitaVectors (Replayed)

def failWith {α : Type} (message : String) : IO α := throw <| IO.userError message

/-- The gate on the replay of one positive file: `fused.end` of the replay holds at each trace
level, and it fails for a changed `v_W`, a changed setup element and a changed fold challenge. -/
def gate (r : Replayed) : IO Unit := do
  let name := r.v.name
  let s := r.state
  let levels := r.layout.folds.zipIdx.filterMap fun (F, level) => match F.fused with
    | .trace T => some (level, F, T)
    | .packing _ => none
  let last := (levels.getLast?.map (·.1)).getD 0
  for (level, F, T) in levels do
    let begin ← IO.monoMsNow
    let some (_, .fused (some f)) := r.d.stages.find? (·.1 == ⟨level, "fused.end", none⟩)
      | failWith s!"{name}: the replay has no passing fused.end at level {level}"
    let V := Replayer.traceValuesIn s level F
    unless f.output == V.output && f.value == V.value && f.batch == V.batch do
      failWith s!"{name}: fused.end of the replay has other values at level {level}"
    if { f with value := f.value + Ext.one }.holds then
      failWith s!"{name}: a changed witness evaluation passes fused.end at level {level}"
    let mut changes := "v_W + 1"
    if name == "key20-opening.json" && level == last then
      let some setup := r.pub.setupFor (traceSetupCount T.opening) | failWith s!"{name}: setup"
      let cs := (Replayer.drawnIn s level 0).map (Poly.ofSparse T.shape.innerDegree)
      let setup' := setup.set! 1 ((setup[1]!).add Fq.one)
      let deg := T.shape.innerDegree
      let cs' := cs.set 0 (Poly.add deg (cs.getD 0 #[]) (unitRing deg))
      let mutated : List (String × Task Bool) := [
        ("a changed setup element",
          Task.spawn fun _ => (traceFused T.opening T.fold setup' cs V).holds),
        ("a changed fold challenge",
          Task.spawn fun _ => (traceFused T.opening T.fold setup cs' V).holds)]
      for (what, holds) in mutated do
        if holds.get then failWith s!"{name}: {what} passes fused.end at level {level}"
      changes := "v_W + 1, setup[1] + 1 and c_0 + 1"
    IO.println s!"  level {level}: fused.end of the replay holds; {changes}: each fails it \
      ({(← IO.monoMsNow) - begin} ms)"

end AkitaFusedVectors
