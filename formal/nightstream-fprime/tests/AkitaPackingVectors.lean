import tests.AkitaVectors
import NightstreamFPrime.Spec.Compression.Akita.Replay.Sampler
import NightstreamFPrime.Spec.Compression.Akita.Replay.WeightPacking

/-! The gate of the end check of the fused sum-check (`fused.end`) at the packing levels
(`Replay/WeightPacking.lean`,
`Replay/FusedJoin.lean`): level 1 and the root of `key20.aks`, the levels 3, 2, 1 and the root of
`key30.aks`. The one replay of each positive vector (`tests/AkitaAllVectors.lean`) evaluates
`fused.end` at each packing level (`packingIn`: `foldFused` on the points, the challenges, `v_W`,
the end claim of the rounds, the claim `σ` of Stage 3 and the fold challenges of the replay, and
the setup prefix of `Public.seed`); this file reads its result (`Derived.stages`). Nothing comes
from the events: the vectors have no event for `fused.end`. At each packing level `fused.end`
holds, and it fails for `v_W + 1` (the weight does not read `v_W`, so the check with `v_W + 1` is
the record of the replay with `v_W + 1`). At level 1 of `key20-opening.json` it fails also for a
changed setup element, a changed fold challenge and the order plane, digit, coefficient of the
cells of the quotient of the carrier row; at the root of `key20-opening.json` for the main
segments, the columns of `D` and the spans of `F` of the groups in the order `Fin n`
(`Batch.mainStart`, `Batch.openSetupCode` and `Batch.firstStart` have the order `groupOrder` of
the code). In the proof bytes, one flipped
bit of the message `witness_evaluation` of a packing level makes the replay reject at `fused.end`
of that level, or stop at a work between them (the rounds of Stage 3 at the root of
`key30.aks`). -/

namespace AkitaPackingVectors

open NightstreamFPrime.Spec.Compression.Akita
open NightstreamFPrime.Spec.Compression.Akita.Replay
open AkitaVectors (Replayed messageOps resumeOf outcomeOf)

def failWith {α : Type} (message : String) : IO α := throw <| IO.userError message

/-- A diagnosis of a failed end check: the parts whose negation or omission makes it hold. -/
def diagnose (f : FusedEnd) (parts : List (String × Ext)) : IO Unit := do
  let virtTerms := (f.batch * f.virt + f.binary * f.support) * (f.value * (f.value + Ext.one))
  let target := f.output - virtTerms
  let total := sumList parts (·.2)
  let flips := parts.filter fun (_, p) => (f.value * (total - p - p)).equals target
  let drops := parts.filter fun (_, p) => (f.value * (total - p)).equals target
  IO.println s!"  diagnosis: flip one {flips.map (·.1)}, drop one {drops.map (·.1)}"

/-- The inputs of the end check of a packing level on the replay. -/
def inputsAt (r : Replayed) (level : ℕ) (F : Fold) (P : PackShape) :
    IO (FoldInputs × List PackGroup × PackingValues × Option Ext) := do
  let s := r.state
  let some setup := r.pub.setupFor (P.setupCount F.stage3.isSome)
    | failWith s!"{r.v.name}: the setup prefix of level {level}"
  return (Replayer.packInputsIn P setup s level F, Replayer.packGroupsIn P s level F,
    Replayer.packingValuesIn s level F, Replayer.setupClaimIn s level F)

/-- When the replay rejects at `fused.end` of a packing level: the diagnosis of its parts. -/
def diagnoseStop (r : Replayed) : IO Unit := do
  let some (.reject i "fused.end") := r.stop | return
  let some (.check _ (.fused level F)) := r.ops[i]? | return
  let .packing P := F.fused | return
  let (FI, gs, V, σ) ← inputsAt r level F P
  let parts := foldParts FI gs (foldTables FI gs V) σ
  diagnose (packingEnd FI gs V (sumList parts (·.2))) parts

/-- The groups with the coefficient 0 of the fold challenge `c_0` of the first group changed by
one, in the subring and in the ring of `A`. -/
def changeChallenge (gs : List PackGroup) : List PackGroup :=
  match gs with
  | [] => []
  | P :: rest =>
      let bump (a : Array Fq) : Array Fq := a.modify 0 (·.add Fq.one)
      { P with challenges := P.challenges.modifyHead bump, ambient := P.ambient.modifyHead bump } ::
        rest

/-- The changes of the inputs of the end check at a level of `key20-opening.json`: at level 1 a
setup element, a fold challenge and the quotient order plane, digit, coefficient; at the root the
cells of the groups in the order `Fin n`: the rank of the group at position `k` of the code order
is its index. -/
def changes (level : ℕ) (F : FoldInputs) (gs : List PackGroup) :
    List (String × FoldInputs × List PackGroup) :=
  if level == 1 then
    [("setup[1] + 1", { F with setup := F.setup.modify 1 (·.add Fq.one) }, gs),
      ("c_0 + 1", F, changeChallenge gs),
      ("the quotient order plane, digit", { F with order := .planeFirst }, gs)]
  else if level == 0 then
    let ranks := (List.finRange 3).map fun k => (groupOrder 3 k).val
    [("the main segments in the order Fin n", { F with cells := { main := some ranks } }, gs),
      ("the columns of D in the order Fin n", { F with cells := { columns := some ranks } }, gs),
      ("the spans of F in the order Fin n", { F with cells := { spans := some ranks } }, gs)]
  else []

/-- The proof with the low bit of the message `witness_evaluation` of each packing level flipped:
the replay of the changed proof (from the states of the positive proof) rejects at `fused.end` of
the level when no work comes between the message and the check, and stops before it otherwise. -/
def bytesGate (r : Replayed) : IO Unit := do
  let resume := resumeOf r.pub r.ops r.v.proof r.messages
  let sent := (messageOps r.ops).toList.zip r.messages.toList
  let index (f : Op → Bool) : ℕ := (r.ops.findIdx? f).getD 0
  let levels := r.layout.folds.zipIdx.filterMap fun (F, level) => match F.fused with
    | .packing _ => some level
    | .trace _ => none
  let tasks ← levels.mapM fun level => do
    let key : Key := ⟨level, "witness_evaluation", none⟩
    let some (_, m) := sent.find? fun (op, _) => match op with
        | .exts k _ => k == key
        | _ => false
      | failWith s!"{r.v.name}: no witness_evaluation at level {level}"
    let changed := r.v.proof.set! m.offset (r.v.proof.get! m.offset ^^^ 1)
    let start := index fun | .exts k _ => k == key | _ => false
    let check := index fun | .check k _ => k == ⟨level, "fused.end", none⟩ | _ => false
    let worked := (List.range (check - start)).any fun j => match r.ops[start + j]? with
      | some (.work _ _ bits) => bits != 0
      | _ => false
    return (level, worked, Task.spawn fun _ => outcomeOf r.ops r.pub resume changed m.offset)
  let mut seen := []
  for (level, worked, task) in tasks do
    let (klass, at_) := task.get
    let stopped := klass == "noWork" || klass == "reject"
    unless (if worked then stopped else klass == "reject" && at_ == some (level, "fused.end")) do
      failWith s!"{r.v.name}: a changed witness_evaluation of level {level}: {klass} at {at_}"
    seen := seen ++ [(level, klass, at_.getD (0, ""))]
  IO.println s!"  a changed witness_evaluation in the bytes (level, Lean class, site): {seen}"

/-- The gate on the replay of one positive file: `fused.end` of the replay holds at each packing
level and fails for a changed `v_W`; on `key20-opening.json` the changes of `changes` fail it. -/
def gate (r : Replayed) : IO Unit := do
  let name := r.v.name
  let levels := r.layout.folds.zipIdx.filterMap fun (F, level) => match F.fused with
    | .packing P => some (level, F, P)
    | .trace _ => none
  for (level, F, P) in levels do
    let begin ← IO.monoMsNow
    let some (_, .fused (some f)) := r.d.stages.find? (·.1 == ⟨level, "fused.end", none⟩)
      | failWith s!"{name}: the replay has no passing fused.end at level {level}"
    if { f with value := f.value + Ext.one }.holds then
      failWith s!"{name}: a changed witness evaluation passes fused.end at level {level}"
    let mut text := "v_W + 1"
    if name == "key20-opening.json" then
      let (FI, gs, V, σ) ← inputsAt r level F P
      let mutated := (changes level FI gs).map fun (what, FI', gs') =>
        (what, Task.spawn fun _ => (foldFused FI' gs' V σ).holds)
      for (what, holds) in mutated do
        if holds.get then failWith s!"{name}: {what} passes fused.end at level {level}"
      text := ", ".intercalate (text :: mutated.map (·.1))
    IO.println s!"  level {level} ({P.slots.length} groups): fused.end of the replay holds; {text}: \
      each fails it ({(← IO.monoMsNow) - begin} ms)"
  bytesGate r

end AkitaPackingVectors
