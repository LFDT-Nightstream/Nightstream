import tests.AkitaVectors

/-! The gate of the sum-checks of the replay:
the replay of `Replay/Transcript.lean` runs `sumcheckRun` after the rounds of each sum-check, on
the claim that it absorbs, the round messages that `decode` reads from the proof and the
challenges that it derives from the bytes. Each `<name>.point`, `<name>.output`,
`<name>.claim_chain` and `<name>.equality_factor` event of the four positive vectors equals the
end of the replay (`Derived.ends` of the one replay of `tests/AkitaAllVectors.lean`), and each
equality factor is not zero.

- The point before the first stage of the range tree is `τ_0` with its last `ℓ` coordinates first
  (`from_column_then_ring_challenges`), `ℓ = Fold.low` of the layout of the Lean numbers. A replay
  with `ℓ + 1` gives another equality factor and rejects at the end check of the first range
  stage, so the events confirm `ℓ`.
- The `rounds` event of each sum-check has its count of rounds, its message length and its
  form. -/

namespace AkitaSumcheckVectors

open Lean (Json)
open NightstreamFPrime.Spec.Compression.Akita.Replay
open AkitaVectors (Vector Oracles Replayed field nat eventExts find? withoutEnds)

def failWith {α : Type} (message : String) : IO α := throw <| IO.userError message

def oracleExts (o : Oracles) (op : String) (level : ℕ) (site : String) : IO (Array Ext) := do
  let some e := find? o op level site | failWith s!"no event {op} {site} at level {level}"
  IO.ofExcept (eventExts e)

/-- The stop and the ends of the sum-checks of the replay of a positive vector with the script
`ops`. -/
def ends (v : Vector) (pub : Public) (ops : Array Op) (messages : Array Message) :
    Option Stop × Array (Key × Before × Ended) :=
  let (stop, d) := replay pub ops v.proof messages
  (stop, d.ends)

def degreeOf (ops : Array Op) (key : Key) : Option ℕ :=
  ops.findSome? fun
    | .exts k degree => if k == ⟨key.level, key.site ++ ".round", some 0⟩ then some degree else none
    | _ => none

/-- The gate on the replay of one positive file. -/
def gate (r : Replayed) : IO Unit := do
  let v := r.v
  let o := r.o
  let name := v.name
  let layout := r.layout
  let ops := r.ops
  let found := r.d.ends
  let roundsEvents := o.events.filter (·.op == "rounds")
  unless roundsEvents.size = found.size do
    failWith s!"{name}: {found.size} sum-checks, the file has {roundsEvents.size}"
  let mut rounds := 0
  let mut factors := 0
  for (key, before, ended) in found do
    let label := s!"{name}: {key.site} of level {key.level}"
    let equality := match before with | .plain => false | _ => true
    let some shape := find? o "rounds" key.level key.site | failWith s!"{label}: no rounds event"
    unless (nat shape.json ["rounds"]).toOption == some ended.point.length ∧
        (nat shape.json ["degree"]).toOption == degreeOf ops key ∧
        ((field shape.json ["equality_factor"]).toOption.bind (·.getBool?.toOption)) ==
          some equality do
      failWith s!"{label}: the rounds event differs"
    unless ended.point.toArray == (← oracleExts o "challenge" key.level (key.site ++ ".point")) do
      failWith s!"{label}: point"
    unless #[ended.claim] == (← oracleExts o "value" key.level (key.site ++ ".output")) do
      failWith s!"{label}: output"
    let some _ := find? o "check" key.level (key.site ++ ".claim_chain")
      | failWith s!"{label}: chain"
    if equality then
      unless #[ended.factor] ==
          (← oracleExts o "value" key.level (key.site ++ ".equality_factor")) do
        failWith s!"{label}: equality factor"
      if ended.factor == Ext.zero then failWith s!"{label}: the equality factor is zero"
      factors := factors + 1
    rounds := rounds + ended.point.length
  IO.println s!"gate 5 {name}: {found.size} sum-checks ({found.size - factors} plain, {factors} \
    equality-factored), {rounds} rounds; point, output and claim_chain equal for each; {factors} \
    equality factors equal and not zero; ℓ per level {layout.folds.map (·.low)}"
  let moved := (withoutEnds ops).map fun
    | .ended key count (.turned point low) => .ended key count (.turned point (low + 1))
    | .check key (.range level base arities (.turned point low)) =>
        .check key (.range level base arities (.turned point (low + 1)))
    | .check key (.rangeNorm level leaf limit arities (.turned point low)) =>
        .check key (.rangeNorm level leaf limit arities (.turned point (low + 1)))
    | op => op
  let (stopped, other) := ends v r.pub moved r.messages
  let first (list : Array (Key × Before × Ended)) := (list.findSome? fun (_, before, ended) =>
    match before with | .turned .. => some ended.factor | _ => none)
  if first found == first other then failWith s!"{name}: an equality factor does not depend on ℓ"
  let some (.reject op site) := stopped
    | failWith s!"{name}: the replay with ℓ + 1 does not stop at an equation"
  let some (.check key _) := moved[op]? | failWith s!"{name}: the replay stops at no check"
  unless key.level = 0 do failWith s!"{name}: with ℓ + 1 the replay stops at level {key.level}"
  IO.println s!"gate 5 {name}: with ℓ + 1 the equality factor of the first range stage differs \
    and the replay rejects at {site} of the root"

end AkitaSumcheckVectors
