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
  ("the builder's instruction set and its calls (`Backend`, `record.rs`); the digest pins compare the lowered programs with Rust's",
    [``CompressionLean.Circuit.Build, ``CompressionLean.Circuit.Build.add,
      ``CompressionLean.Circuit.Build.assertEqual, ``CompressionLean.Circuit.Build.assertExtEqual,
      ``CompressionLean.Circuit.Build.assertExtZero, ``CompressionLean.Circuit.Build.assertZero,
      ``CompressionLean.Circuit.Build.bits, ``CompressionLean.Circuit.Build.const,
      ``CompressionLean.Circuit.Build.coordinatewise, ``CompressionLean.Circuit.Build.emit,
      ``CompressionLean.Circuit.Build.extAdd, ``CompressionLean.Circuit.Build.extConstant,
      ``CompressionLean.Circuit.Build.extInv, ``CompressionLean.Circuit.Build.extMul,
      ``CompressionLean.Circuit.Build.extScale, ``CompressionLean.Circuit.Build.extSub,
      ``CompressionLean.Circuit.Build.handles, ``CompressionLean.Circuit.Build.hint,
      ``CompressionLean.Circuit.Build.linear, ``CompressionLean.Circuit.Build.mul,
      ``CompressionLean.Circuit.Build.permute, ``CompressionLean.Circuit.Build.priv,
      ``CompressionLean.Circuit.Build.program, ``CompressionLean.Circuit.Build.pub,
      ``CompressionLean.Circuit.Build.r1cs, ``CompressionLean.Circuit.Build.scale,
      ``CompressionLean.Circuit.Build.select, ``CompressionLean.Circuit.Build.sub, ``CompressionLean.Circuit.Builder,
      ``CompressionLean.Circuit.Instr, ``CompressionLean.Circuit.Instr.outputs, ``CompressionLean.Circuit.Var]),
  ("the Poseidon2 block template (`circuit/poseidon2.rs`); `permutation_template_matches_lean` pins its digest against Rust's",
    [``CompressionLean.Circuit.Block.Build, ``CompressionLean.Circuit.Block.Linear,
      ``CompressionLean.Circuit.Block.Linear.addTerm, ``CompressionLean.Circuit.Block.Linear.cell,
      ``CompressionLean.Circuit.Block.Linear.combine, ``CompressionLean.Circuit.Block.Linear.eval,
      ``CompressionLean.Circuit.Block.Linear.mergePart, ``CompressionLean.Circuit.Block.Linear.zero,
      ``CompressionLean.Circuit.Block.build, ``CompressionLean.Circuit.Block.entries,
      ``CompressionLean.Circuit.Block.externalLinear, ``CompressionLean.Circuit.Block.fullRound,
      ``CompressionLean.Circuit.Block.m4, ``CompressionLean.Circuit.Block.mixLinear,
      ``CompressionLean.Circuit.Block.output, ``CompressionLean.Circuit.Block.partialLanes,
      ``CompressionLean.Circuit.Block.partialRound, ``CompressionLean.Circuit.Block.poseidonTemplate,
      ``CompressionLean.Circuit.Block.push, ``CompressionLean.Circuit.Block.rowCell,
      ``CompressionLean.Circuit.Block.sboxes, ``CompressionLean.Circuit.Block.start,
      ``CompressionLean.Circuit.Block.step]),
  ("a type, or a size of one, that ported functions read or return, as Lean names it; their entries name the Rust items",
    [``CompressionLean.Circuit.EarlyOut, ``CompressionLean.Circuit.ExtVar, ``CompressionLean.Circuit.GkrLayerView,
      ``CompressionLean.Circuit.HonestInput, ``CompressionLean.Circuit.HonestOut,
      ``CompressionLean.Circuit.Layer1Input, ``CompressionLean.Circuit.LinearView, ``CompressionLean.Circuit.Lower,
      ``CompressionLean.Circuit.Mle.DivisionStates, ``CompressionLean.Circuit.OpeningRows,
      ``CompressionLean.Circuit.OpeningsInput, ``CompressionLean.Circuit.QueryRows,
      ``CompressionLean.Circuit.Transcript, ``CompressionLean.Circuit.WhirBatchKey,
      ``CompressionLean.Circuit.WhirKey, ``CompressionLean.Circuit.WhirPlacement,
      ``CompressionLean.Circuit.WhirRoundKey, ``CompressionLean.Circuit.WhirSelector, ``CompressionLean.Replay.D,
      ``CompressionLean.Replay.Draw, ``CompressionLean.Replay.EarlyOut, ``CompressionLean.Replay.Export,
      ``CompressionLean.Replay.GkrLayer, ``CompressionLean.Replay.GkrProof, ``CompressionLean.Replay.Opening,
      ``CompressionLean.Replay.Relation.blockVars, ``CompressionLean.Replay.Relation.cubeVars,
      ``CompressionLean.Replay.Relation.matrices, ``CompressionLean.Replay.Relation.matrixVars,
      ``CompressionLean.Replay.ReplayM, ``CompressionLean.Replay.SumcheckShape, ``CompressionLean.Replay.Transcript,
      ``CompressionLean.Replay.Variant, ``CompressionLean.Replay.WhirBatch, ``CompressionLean.Replay.WhirRound,
      ``CompressionLean.Replay.WhirRoundShape, ``CompressionLean.Replay.WhirShape, ``CompressionLean.Shrink.Call,
      ``CompressionLean.Shrink.Ccs, ``CompressionLean.Shrink.Entry, ``CompressionLean.Shrink.Transcript,
      ``CompressionLean.Terminal.Package.Numbering]),
  ("a helper of the ported functions of its file (a step, a loop or a check they share); their entries name the Rust items",
    [``CompressionLean.Circuit.R1csRow.inField, ``CompressionLean.Circuit.cells, ``CompressionLean.Circuit.chunks,
      ``CompressionLean.Circuit.constants, ``CompressionLean.Circuit.extInverse,
      ``CompressionLean.Circuit.extProduct, ``CompressionLean.Circuit.fromFn, ``CompressionLean.Circuit.negate,
      ``CompressionLean.Circuit.privateDigest, ``CompressionLean.Circuit.privateExts,
      ``CompressionLean.Circuit.privatePairs, ``CompressionLean.Circuit.push, ``CompressionLean.Circuit.rate,
      ``CompressionLean.Circuit.readCoordinates, ``CompressionLean.Circuit.readValue,
      ``CompressionLean.Circuit.readValues, ``CompressionLean.Circuit.steps,
      ``CompressionLean.Replay.GkrClaims.leafValue, ``CompressionLean.Replay.batchExpected,
      ``CompressionLean.Replay.chiFactor, ``CompressionLean.Replay.contribution, ``CompressionLean.Replay.ensure,
      ``CompressionLean.Replay.flatPairs, ``CompressionLean.Replay.layerShapeOk, ``CompressionLean.Replay.nextStart,
      ``CompressionLean.Replay.observeAll, ``CompressionLean.Replay.observeExt,
      ``CompressionLean.Replay.openingsAgree, ``CompressionLean.Replay.p1Values, ``CompressionLean.Replay.pathOk,
      ``CompressionLean.Replay.permute, ``CompressionLean.Replay.prefixSum, ``CompressionLean.Replay.prodRange,
      ``CompressionLean.Replay.prodRangeGl, ``CompressionLean.Replay.rate, ``CompressionLean.Replay.record,
      ``CompressionLean.Replay.reduce, ``CompressionLean.Replay.reducedOf, ``CompressionLean.Replay.roundValues,
      ``CompressionLean.Replay.sampleExt, ``CompressionLean.Replay.sampleWord, ``CompressionLean.Replay.sampleWords,
      ``CompressionLean.Replay.sumKx, ``CompressionLean.Replay.sumRange, ``CompressionLean.Replay.tableSumStep,
      ``CompressionLean.Replay.width, ``CompressionLean.Setup.Push.pad, ``CompressionLean.Setup.sortedSet,
      ``CompressionLean.Shrink.Layout.blockEntries, ``CompressionLean.Shrink.Layout.glueEntries,
      ``CompressionLean.Shrink.Layout.rowEntries, ``CompressionLean.Shrink.Run.rows,
      ``CompressionLean.Shrink.batchPoly, ``CompressionLean.Shrink.constraintTable,
      ``CompressionLean.Shrink.innerTables, ``CompressionLean.Shrink.outerComb, ``CompressionLean.Shrink.outerTables,
      ``CompressionLean.Shrink.rowValue, ``CompressionLean.Terminal.Package.Producer.rows,
      ``CompressionLean.Terminal.Package.callerSegments, ``CompressionLean.Terminal.Package.mark,
      ``CompressionLean.Terminal.Package.named, ``CompressionLean.Terminal.Package.number,
      ``CompressionLean.Terminal.Package.numberColumn, ``CompressionLean.Terminal.Package.reindex,
      ``CompressionLean.Terminal.Package.row, ``CompressionLean.Terminal.Program.chunks,
      ``CompressionLean.Terminal.Program.pairs]),
  ("the replay's form of a model or F′ definition, which its docstring names; it ports no Rust item",
    [``CompressionLean.Replay.GkrClaims, ``CompressionLean.Replay.HonestOut, ``CompressionLean.Replay.LateOut,
      ``CompressionLean.Replay.LinearOut, ``CompressionLean.Replay.activeTrees, ``CompressionLean.Replay.advance,
      ``CompressionLean.Replay.bar, ``CompressionLean.Replay.batchClaim, ``CompressionLean.Replay.chiList,
      ``CompressionLean.Replay.divFactor, ``CompressionLean.Replay.divStep, ``CompressionLean.Replay.eqAtList,
      ``CompressionLean.Replay.eqEvalList, ``CompressionLean.Replay.evalKStates, ``CompressionLean.Replay.headFrom,
      ``CompressionLean.Replay.layerCount, ``CompressionLean.Replay.leafWeight,
      ``CompressionLean.Replay.lessThanScan, ``CompressionLean.Replay.monomial,
      ``CompressionLean.Replay.omegaComputed, ``CompressionLean.Replay.partFactors,
      ``CompressionLean.Replay.samplePair, ``CompressionLean.Replay.stepSum]),
  ("the replay's word and extension arithmetic and the maps between words and fields; `Bridge/Field.lean` proves what they compute",
    [``CompressionLean.Replay.Ext, ``CompressionLean.Replay.Ext.basis, ``CompressionLean.Replay.Ext.inv,
      ``CompressionLean.Replay.Ext.isZero, ``CompressionLean.Replay.Ext.pow, ``CompressionLean.Replay.Ext.scale,
      ``CompressionLean.Replay.Word, ``CompressionLean.Replay.Word.cast, ``CompressionLean.Replay.Word.inv,
      ``CompressionLean.Replay.Word.ofF, ``CompressionLean.Replay.Word.ofNat, ``CompressionLean.Replay.Word.pow,
      ``CompressionLean.Replay.Word.toF, ``CompressionLean.Replay.Word.toNat]),
  ("a Plonky3 constant; `ports.json` names items of this repository's crates",
    [``CompressionLean.Replay.generator, ``CompressionLean.Replay.twoAdicGenerator]),
  ("the game model's bad sets and true values, which the shrink game reads; Rust has none",
    [``CompressionLean.Shrink.rhoBad, ``CompressionLean.Shrink.tauBad, ``CompressionLean.Shrink.trueValues])]

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
