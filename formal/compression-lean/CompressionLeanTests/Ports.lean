import CompressionLean
import CompressionLeanTests.Inventory

/-!
The ports gate and the status gate. `lake build` runs both, and rebuilds them
when `ports.json` or `STATUS.md` changes.

- `Ports/` (`Inventory.portRoot`) holds definitions only: a theorem written
  there fails. Lean's own declarations (constructors, projections,
  recursors, equation lemmas, derived instances) are not written there.
- A port's namespace is its path without `Ports/`: a declaration of
  `Ports/X.lean` or `Ports/X/…` is in `CompressionLean.X`.
- Every definition, structure and inductive written in `Ports/` has an entry
  in `ports.json`, or in `notPorts` with the reason. Instances and notation
  are not checked: an instance holds an operation of a type that has its own
  entry.
- Every Lean name in `ports.json` is declared, and in `Ports/` or in
  `portsOutside` with the reason.
- `STATUS.md` is what `lake exe status` prints. The page fails when a
  Prop-valued binder of a claim has no entry in `Inventory.premises`.
-/

open Lean Elab Command CompressionLeanTests.Inventory

/-- The `ports.json` entries outside `Ports/`, each with the reason. -/
def portsOutside : List (Name × String) :=
  let lower := "a generic definition that the layer-1 game reads too; `Layer1/` is below `Ports/`"
  [(``CompressionLean.Lib.eqAt, lower), (``CompressionLean.Lib.eqEval, lower),
    (``CompressionLean.Lib.roundValues, lower), (``CompressionLean.Game.sumcheck, lower)]

/-- The definitions of `Ports/` that have no `ports.json` entry, by reason. -/
def notPorts : List (String × List Name) := [
  ("the program as data and its handles (`Var`, `ExtVar`, the instruction list and the state that writes it); Rust calls `Backend` directly and keeps no list",
    [``CompressionLean.Circuit.Var, ``CompressionLean.Circuit.ExtVar, ``CompressionLean.Circuit.Builder,
      ``CompressionLean.Circuit.Build, ``CompressionLean.Circuit.Build.emit, ``CompressionLean.Circuit.Build.handles,
      ``CompressionLean.Circuit.Build.program]),
  ("a monad, a state or a log through which Lean passes what Rust keeps in a `&mut` argument, or a read or write of that state; Rust has no such item",
    [``CompressionLean.Circuit.Lower, ``CompressionLean.Circuit.Transcript, ``CompressionLean.Circuit.readValue,
      ``CompressionLean.Circuit.readValues, ``CompressionLean.Circuit.readCoordinates, ``CompressionLean.Circuit.push,
      ``CompressionLean.Circuit.steps, ``CompressionLean.Replay.ReplayM, ``CompressionLean.Replay.Transcript,
      ``CompressionLean.Replay.Draw, ``CompressionLean.Replay.record, ``CompressionLean.Replay.ensure]),
  ("a generic combinator (a sum, a product, a cast, `<[_]>::chunks`, `std::array::from_fn`, a sorted set); the entries of the definitions that use it name their Rust items",
    [``CompressionLean.Circuit.Build.coordinatewise, ``CompressionLean.Circuit.fromFn,
      ``CompressionLean.Circuit.chunks, ``CompressionLean.Circuit.R1csRow.inField, ``CompressionLean.Replay.sumRange,
      ``CompressionLean.Replay.prodRange, ``CompressionLean.Replay.prodRangeGl, ``CompressionLean.Replay.sumKx,
      ``CompressionLean.Replay.flatPairs, ``CompressionLean.Setup.sortedSet, ``CompressionLean.Terminal.Program.chunks,
      ``CompressionLean.Terminal.Program.pairs]),
  ("a default for a read past the end of a list, or an accessor; Rust indexes directly",
    [``CompressionLean.Circuit.Block.Linear.zero, ``CompressionLean.Setup.Push.pad,
      ``CompressionLean.Replay.GkrClaims.leafValue]),
  ("a record of the values that one stage gives the next; Rust keeps them in local variables",
    [``CompressionLean.Circuit.Block.Build, ``CompressionLean.Circuit.EarlyOut, ``CompressionLean.Circuit.HonestInput,
      ``CompressionLean.Circuit.HonestOut, ``CompressionLean.Circuit.LinearView,
      ``CompressionLean.Circuit.OpeningsInput, ``CompressionLean.Circuit.Mle.DivisionStates,
      ``CompressionLean.Replay.GkrClaims, ``CompressionLean.Replay.EarlyOut, ``CompressionLean.Replay.LinearOut,
      ``CompressionLean.Replay.HonestOut, ``CompressionLean.Replay.LateOut,
      ``CompressionLean.Terminal.Package.Numbering]),
  ("Plonky3's field arithmetic or constants, or a map between words and fields for the proofs; `ports.json` names items of this repository's crates",
    [``CompressionLean.Replay.Word.ofNat, ``CompressionLean.Replay.Word.toNat, ``CompressionLean.Replay.Word.pow,
      ``CompressionLean.Replay.Word.inv, ``CompressionLean.Replay.Word.toF, ``CompressionLean.Replay.Word.cast,
      ``CompressionLean.Replay.Ext.pow, ``CompressionLean.Replay.generator, ``CompressionLean.Replay.twoAdicGenerator]),
  ("the shrink CCS as a list of entries, the model's form; Rust's matrices are the block template and the glue rows",
    [``CompressionLean.Shrink.Ccs, ``CompressionLean.Shrink.Entry]),
  ("the game model's transcript, bad sets, true values and batch polynomial; Rust has none",
    [``CompressionLean.Shrink.Transcript, ``CompressionLean.Shrink.tauBad, ``CompressionLean.Shrink.trueValues,
      ``CompressionLean.Shrink.rhoBad, ``CompressionLean.Shrink.batchPoly])]

/-- The types of Lean's syntax declarations (notation, macros), which a port
file may hold. -/
def syntaxTypes : List Name := [``ParserDescr, ``TrailingParserDescr, ``Macro, ``PrettyPrinter.Unexpander]

/-- `n` or a prefix of it is an instance, so a derived instance's parts too. -/
def underInstance (env : Environment) : Name → Bool
  | .anonymous => false
  | n@(.str p _) | n@(.num p _) => Meta.isInstanceCore env n || underInstance env p

elab "#check_ports" : command => do
  let env ← getEnv
  let ports ← readPorts
  for p in ports do
    unless env.contains p.lean do
      throwError "ports.json names {p.lean}, which is not declared"
    unless portRoot.isPrefixOf (moduleOf env p.lean) || portsOutside.any (·.1 == p.lean) do
      throwError "ports.json names {p.lean}, which is outside {portRoot} and not in portsOutside"
  for (n, _) in portsOutside do
    unless ports.any (·.lean == n) do throwError "portsOutside names {n}, which has no ports.json entry"
  for (_, names) in notPorts do
    for n in names do
      unless portRoot.isPrefixOf (moduleOf env n) do throwError "notPorts names {n}, which is outside {portRoot}"
      if ports.any (·.lean == n) then throwError "notPorts names {n}, which ports.json lists"
  let listed := (ports.toList.map (·.lean) ++ notPorts.flatMap (·.2)).foldl NameSet.insert {}
  let modules := (env.header.moduleNames.zip env.header.moduleData).filter (portRoot.isPrefixOf ·.1)
  let mut theorems := #[]
  let mut misplaced := #[]
  let mut unlisted := #[]
  let mut count := 0
  for (m, data) in modules do
    let some top := m.components[2]? | continue
    let space := `CompressionLean ++ top
    for n in data.constNames do
      let some info := env.find? n | continue
      unless (declRangeExt.find? (level := .server) env n).isSome do continue
      if (env.getProjectionFnInfo? n).isSome || info matches .ctorInfo _ || info matches .recInfo _ then continue
      if syntaxTypes.any info.type.isConstOf then continue
      let user := privateToUserName n
      unless space.isPrefixOf user do misplaced := misplaced.push (user, m)
      if info matches .thmInfo _ then
        theorems := theorems.push user
        continue
      if underInstance env n then continue
      count := count + 1
      unless listed.contains user do unlisted := unlisted.push user
  unless theorems.isEmpty do
    throwError "{portRoot} holds definitions only; move these theorems to a proof folder:\n{theorems}"
  unless misplaced.isEmpty do
    throwError "these declarations are not in the namespace of their port's path:\n{misplaced}"
  unless unlisted.isEmpty do
    throwError "these definitions of {portRoot} have no entry; add each to ports.json or to notPorts:\n{unlisted}"
  logInfo m!"{ports.size} ports; {count} definitions under {portRoot}, each listed"

elab "#check_status" : command => do
  let page ← liftCoreM (statusPage (← readPorts))
  unless page == (← IO.FS.readFile "STATUS.md") do
    throwError "STATUS.md is out of date: run `lake exe status > STATUS.md`"

#check_ports
#check_status
