import tests.AkitaVectors
import NightstreamFPrime.Spec.Compression.Akita.Replay.SumCheck

/-! The gate of work item 5 (`docs/akita-work/match-design.md` section 7): `sumcheckRun` on the
round messages that `decode` reads from the proof and on the challenges that the replay derives
from the bytes gives each `<name>.point`, `<name>.output`, `<name>.claim_chain` and
`<name>.equality_factor` event of the four positive vectors, and each equality factor is not
zero.

- The input claim of a sum-check is the claim that the replay of `tests/AkitaVectors.lean`
  computes and absorbs.
- The point before a stage of the range tree is the point of the stage before, and for the first
  stage `τ_0` with its last `ℓ` coordinates first (`from_column_then_ring_challenges`), where `ℓ`
  is `log_2` of the least carrier degree of the fold groups of the level in the layout of the Lean
  numbers (`relation_coefficient_variable_count`). The equality factors confirm `ℓ`.
- The form of a sum-check is equality-factored for the stages of the range tree and plain for the
  others. The `rounds` event of each sum-check has its count of rounds, its message length and
  its form.
- Each run with the input claim plus one gives another end claim. -/

namespace AkitaSumcheckVectors

open Lean (Json)
open NightstreamFPrime.Spec.Compression.Akita.Replay
open AkitaVectors (Vector Oracles load layoutOf publicOf joined messageOps field nat array eventExts
  positives)

def failWith {α : Type} (message : String) : IO α := throw <| IO.userError message

/-- `log_2` of the least carrier degree of the fold groups of each level of a layout. -/
def lowCounts (layout : Layout) : Array ℕ :=
  (layout.folds.map fun F => Nat.log 2 ((F.groups.map (·.degree)).foldl min
    ((F.groups.head?.map (·.degree)).getD 0))).toArray

/-- The round messages of one sum-check of a level, in the order of the rounds. -/
structure Rounds where
  level : ℕ
  name : String
  messages : Array (Array Ext)

/-- The sum-checks of a decoded proof in the order of the script. -/
def roundsOf (ops : Array Op) (messages : Array Message) : Array Rounds := Id.run do
  let mut out : Array Rounds := #[]
  for (op, m) in (messageOps ops).toList.zip messages.toList do
    let .exts key _ := op | continue
    let some _ := key.index | continue
    let .exts values := m.data | continue
    let name := (key.site.splitOn ".round").headD ""
    match out.back? with
    | some s =>
        if s.level == key.level && s.name == name then
          out := out.pop.push { s with messages := s.messages.push values }
        else out := out.push ⟨key.level, name, #[values]⟩
    | none => out := out.push ⟨key.level, name, #[values]⟩
  return out

/-- `τ_0` with its last `low` coordinates first. -/
def rotate (low : ℕ) (tau0 : Array Ext) : List Ext :=
  tau0.toList.drop (tau0.size - low) ++ tau0.toList.take (tau0.size - low)

/-- The point before a stage of the range tree: the point of the stage before, or the rotated
`τ_0` for the first stage. -/
def pointBefore (derived : Std.HashMap Key (Array Ext)) (low level : ℕ) (name : String) :
    Option (List Ext) :=
  let stage (k : ℕ) := derived.get? ⟨level, s!"range.{k}.point", none⟩
  let first := (derived.get? ⟨level, "tau0", none⟩).map (rotate low)
  if name == "range.leaf" then
    match (List.range 4).reverse.findSome? stage with
    | some point => some point.toList
    | none => first
  else match ((name.splitOn ".")[1]?).bind String.toNat? with
    | some 0 => first
    | some (k + 1) => (stage k).map (·.toList)
    | none => none

def word (bytes : ByteArray) : Ext := ⟨⟨Wire.word bytes 0⟩, ⟨Wire.word bytes 8⟩⟩

/-- The first event of an `op` and a `site` at a level. -/
def event? (o : Oracles) (op : String) (level : ℕ) (site : String) : Option AkitaVectors.Event :=
  o.events.find? fun e => e.op == op && e.level == level && e.site == site

def oracleExts (o : Oracles) (op : String) (level : ℕ) (site : String) : IO (Array Ext) := do
  let some e := event? o op level site | failWith s!"no event {op} {site} at level {level}"
  IO.ofExcept (eventExts e)

structure Counts where
  plain : ℕ := 0
  equality : ℕ := 0
  rounds : ℕ := 0
  chains : ℕ := 0
  factors : ℕ := 0
deriving Repr

def gate (name : String) : IO Counts := do
  let (v, o) ← load name
  let layout ← layoutOf v
  let ops ← IO.ofExcept (script layout v.plan)
  let some (messages, _) := (decode ops v.proof).toOption | failWith s!"{name}: decode"
  let pub ← publicOf v o layout
  let (stop, d) := replay pub ops v.proof messages
  if let some s := stop then failWith s!"{name}: the replay stops: {repr s}"
  let derived := joined d
  let lows := lowCounts layout
  let sumchecks := roundsOf ops messages
  let roundsEvents := o.events.filter (·.op == "rounds")
  unless roundsEvents.size = sumchecks.size do
    failWith s!"{name}: {sumchecks.size} sum-checks, the file has {roundsEvents.size}"
  let mut counts : Counts := {}
  for s in sumchecks do
    let label := s!"{name}: {s.name} of level {s.level}"
    let some (_, bytes) := d.absorbs.find? (·.1 == ⟨s.level, s.name ++ ".claim", none⟩)
      | failWith s!"{label}: no absorbed claim"
    let some point := derived.get? ⟨s.level, s.name ++ ".point", none⟩
      | failWith s!"{label}: no derived point"
    unless point.size = s.messages.size do failWith s!"{label}: point and rounds"
    let equality := s.name.startsWith "range."
    let form ← if equality then
        match pointBefore derived (lows[s.level]?.getD 0) s.level s.name with
        | some before => pure (Form.equality before)
        | none => failWith s!"{label}: no point before"
      else pure Form.plain
    let rounds := s.messages.toList.map (·.toList) |>.zip point.toList
    let some out := sumcheckRun form (word bytes) rounds | failWith s!"{label}: shape"
    let some shape := event? o "rounds" s.level s.name | failWith s!"{label}: no rounds event"
    let degree := (s.messages[0]?.map (·.size)).getD 0
    unless (nat shape.json ["rounds"]).toOption == some s.messages.size ∧
        (nat shape.json ["degree"]).toOption == some degree ∧
        ((field shape.json ["equality_factor"]).toOption.bind (·.getBool?.toOption)) ==
          some equality ∧ s.messages.all (·.size == degree) do
      failWith s!"{label}: the rounds event differs"
    unless out.point.toArray == (← oracleExts o "challenge" s.level (s.name ++ ".point")) do
      failWith s!"{label}: point"
    unless #[out.claim] == (← oracleExts o "value" s.level (s.name ++ ".output")) do
      failWith s!"{label}: output"
    let some _ := event? o "check" s.level (s.name ++ ".claim_chain") | failWith s!"{label}: chain"
    let some changed := sumcheckRun form (word bytes + Ext.one) rounds | failWith s!"{label}: shape"
    if changed.claim == out.claim then failWith s!"{label}: the end does not depend on the claim"
    if equality then
      unless #[out.factor] == (← oracleExts o "value" s.level (s.name ++ ".equality_factor")) do
        failWith s!"{label}: equality factor"
      if out.factor == Ext.zero then failWith s!"{label}: the equality factor is zero"
      counts := { counts with equality := counts.equality + 1 }
      counts := { counts with factors := counts.factors + 1 }
    else counts := { counts with plain := counts.plain + 1 }
    counts := { counts with rounds := counts.rounds + s.messages.size }
    counts := { counts with chains := counts.chains + 1 }
  IO.println s!"gate 5 {name}: {counts.plain + counts.equality} sum-checks ({counts.plain} plain, \
    {counts.equality} equality-factored), {counts.rounds} rounds; point, output and claim_chain \
    equal for {counts.chains}; {counts.factors} equality factors equal and not zero; ℓ per level \
    {lows}"
  return counts

#eval show IO Unit from do
  let start ← IO.monoMsNow
  for name in positives do
    discard <| gate name
  IO.println s!"time {(← IO.monoMsNow) - start} ms"

end AkitaSumcheckVectors
