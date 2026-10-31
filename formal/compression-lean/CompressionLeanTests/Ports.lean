import CompressionLean
import CompressionLeanTests.Inventory

open Lean Elab Command CompressionLeanTests.Inventory

def portsOutside : List (Name × String) :=
  let lower := "a generic definition that the layer-1 game reads too; `Layer1/` is below `Ports/`"
  let reader := "a reader of a file that a Rust test writes; only the executables read these files"
  [(``CompressionLean.Lib.eqAt, lower), (``CompressionLean.Lib.eqEval, lower),
    (``CompressionLean.Lib.roundValues, lower), (``CompressionLean.Game.sumcheck, lower),
    (``CompressionLeanExe.Runs.readRuns, reader), (``CompressionLeanExe.Runs.readColumns, reader)]

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
  ("a literal, the equality of canonical words, a default, a Plonky3 constant, or a map between words and fields for the proofs; `ports.json` names items of this repository's crates",
    [``CompressionLean.Replay.Word.toNat, ``CompressionLean.Replay.Word.toF, ``CompressionLean.Replay.Word.cast,
      ``CompressionLean.Replay.generator, ``CompressionLean.Replay.twoAdicGenerator,
      ``CompressionLean.Replay.Word.instOfNat, ``CompressionLean.Replay.Word.instBEq,
      ``CompressionLean.Replay.Word.instInhabited, ``CompressionLean.Replay.Ext.instOfNat,
      ``CompressionLean.Replay.Ext.instBEq]),
  ("a decision procedure for a proposition of `Ports/`, which the executables evaluate; the proposition's entries name its Rust items",
    [``CompressionLean.Setup.instDecidableAccepts]),
  ("the shrink CCS as a list of entries, the model's form; Rust's matrices are the block template and the glue rows",
    [``CompressionLean.Shrink.Ccs, ``CompressionLean.Shrink.Entry]),
  ("the game model's transcript, bad sets, true values and batch polynomial; Rust has none",
    [``CompressionLean.Shrink.Transcript, ``CompressionLean.Shrink.tauBad, ``CompressionLean.Shrink.trueValues,
      ``CompressionLean.Shrink.rhoBad, ``CompressionLean.Shrink.batchPoly])]

def syntaxTypes : List Name := [``ParserDescr, ``TrailingParserDescr, ``Macro, ``PrettyPrinter.Unexpander]

def underInstance (env : Environment) : Name → Bool
  | .anonymous => false
  | n@(.str p _) | n@(.num p _) => Meta.isInstanceCore env n || underInstance env p

/-- An instance that `deriving` makes has no text of its own; its type has the entry. -/
def derived (env : Environment) (lines : List String) (n : Name) : Bool :=
  underInstance env n && match declRangeExt.find? (level := .server) env n with
    | some r => (((lines.getD (r.range.pos.line - 1) "").splitOn " ").filter (· != "")).head? == some "deriving"
    | none => false

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
    let lines := (← IO.FS.readFile (m.toString.replace "." "/" ++ ".lean")).splitOn "\n"
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
      if derived env lines n then continue
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

/-- The Rust citations that keep their line numbers: lines of crate versions
that the workspace pins exactly, so they do not move. -/
def pinnedLineCitations : List String := ["report.rs:100-111", "prescribed_pcs.rs:91-110"]

def citesRustLine (line : String) : Bool :=
  let rest := pinnedLineCitations.foldl (fun s pinned => s.replace pinned "") line
  (rest.splitOn ".rs:").drop 1 |>.any fun after => after.toList.head?.any Char.isDigit

elab "#check_citations" : command => do
  let mut bad := #[]
  for root in ["CompressionLean", "CompressionLeanExe", "CompressionLeanTests"] do
    for file in ← System.FilePath.walkDir root do
      unless file.extension == some "lean" do continue
      for (line, k) in ((← IO.FS.readFile file).splitOn "\n").zipIdx do
        if citesRustLine line then bad := bad.push s!"{file}:{k + 1}"
  unless bad.isEmpty do
    throwError "these lines cite Rust by line number, which drifts; cite the item (`file.rs::Item`):\n{bad}"
  logInfo m!"no Rust line citations outside the pinned crates"

elab "#check_audited" : command => do
  let path := "CompressionLeanTests/Axioms.lean"
  let (imports, _, _) ← Elab.parseImports (← IO.FS.readFile path) path
  let mut missing := #[]
  for file in ← System.FilePath.walkDir "CompressionLeanTests" do
    let some stem := file.fileStem | continue
    let module := `CompressionLeanTests ++ stem.toName
    unless file.extension != some "lean" || stem == "Axioms" || imports.any (·.module == module) do
      missing := missing.push module
  unless missing.isEmpty do
    throwError "Axioms.lean does not import these gates, so the axiom audit does not read them:\n{missing}"
  logInfo m!"Axioms.lean imports every gate"

#check_ports
#check_status
#check_citations
#check_audited
