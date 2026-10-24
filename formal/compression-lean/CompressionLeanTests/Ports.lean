import CompressionLean
import CompressionLeanTests.Inventory

/-!
The ports gate and the status gate. `lake build` runs both, and rebuilds them
when `ports.json` or `STATUS.md` changes.

- Every Lean name in `ports.json` is declared. Every declaration of a port
  module (`portModules`) whose docstring cites a Rust file (`.rs`) has an entry
  in `ports.json` or in `notPorts`.
- `STATUS.md` is what `lake exe status` prints.
-/

open Lean Elab Command CompressionLeanTests.Inventory

/-- Declarations of the port modules whose docstrings cite Rust but that
transcribe no Rust item, each with the reason. -/
def notPorts : List (Name × String) := [
  (``CompressionLean.Circuit.HonestInput.blockPoint, "a field; `honestStage` reads it as `verify` does"),
  (``CompressionLean.Circuit.HonestInput.commitmentRows, "a field; `honestStage` reads it as `verify` does"),
  (``CompressionLean.Circuit.HonestInput.lanePoint, "a field; `honestStage` reads it as `verify` does"),
  (``CompressionLean.Circuit.OpeningsInput, "the names of `verify`'s values; `openingsStage` reads them"),
  (``CompressionLean.Circuit.OpeningsInput.queryPoints, "a field; `openingsStage` reads it as `verify` does"),
  (``CompressionLean.Circuit.Layer1Input, "the format of `layer1.json`, which a Rust test writes"),
  (``CompressionLean.Shrink.Call, "the calls of the trait `Sink`; `Call.rows`, `Run.counts` and `glueTotal` port what the sinks do with them"),
  (``CompressionLean.Shrink.Wire.Valid, "a fact about the lowering's wires, which `lower_valid` proves"),
  (``CompressionLean.Shrink.Layout.blockEntries, "the block entries of the CCS; `blockPart_eq` gives their sum as `block_part`"),
  (``CompressionLean.Shrink.Layout.rowEntries, "the glue entries of the CCS; `glueTotal_eq` gives their sum as `Evaluate`'s total"),
  (``CompressionLean.Shrink.matrixValue, "the CCS's matrix value; `verifierMatrices_eq` gives the verifier's value as it"),
  (``CompressionLean.Shrink.mle_z, "a theorem: the extension of `z`")]

/-- `doc` names a Rust file: `.rs` not followed by a letter or a digit. -/
def citesRust (doc : String) : Bool :=
  (doc.splitOn ".rs").drop 1 |>.any fun rest => !(rest.toList.head?.any Char.isAlphanum)

elab "#check_ports" : command => do
  let env ← getEnv
  let ports ← readPorts
  for p in ports do
    unless env.contains p.lean do
      throwError "ports.json names {p.lean}, which is not declared"
  let listed := (ports.toList.map (·.lean) ++ notPorts.map (·.1)).foldl NameSet.insert {}
  let mut missing := #[]
  for root in portModules do
    let modules := (env.header.moduleNames.zip env.header.moduleData).filter (root.isPrefixOf ·.1)
    if modules.isEmpty then throwError "no port module {root}"
    for (_, data) in modules do
      for n in data.constNames do
        if let some doc ← findDocString? env n then
          if citesRust doc && !listed.contains n then missing := missing.push n
  unless missing.isEmpty do
    throwError "these declarations cite Rust; add each to ports.json or to notPorts:\n{missing}"
  logInfo m!"{ports.size} ports; every Rust citation of the port modules is listed"

elab "#check_status" : command => do
  let page ← liftCoreM (statusPage (← readPorts))
  unless page == (← IO.FS.readFile "STATUS.md") do
    throwError "STATUS.md is out of date: run `lake exe status > STATUS.md`"

#check_ports
#check_status
