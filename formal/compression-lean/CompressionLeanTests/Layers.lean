import CompressionLean

/-!
The folder gate. `lake build` runs it.

The folders of `CompressionLean/` are layers in `layers` order. A module
imports modules of its own folder or of a lower one. A module of a folder
that `layers` does not name fails the build.
-/

open Lean Elab Command

/-- The folders, lowest first. A top-level file (`Fields.lean`, `Claims.lean`)
is a folder of its own. -/
def layers : List String :=
  ["Lib", "Fields", "Game", "Terminal", "Layer1", "Ports", "Shrink", "Lowering", "Bridge",
    "Final", "Checks", "Sim", "Claims"]

/-- The folder of a library module: the component after `CompressionLean`. -/
def folderOf (m : Name) : Option String :=
  match m.components with
  | `CompressionLean :: f :: _ => some f.toString
  | _ => none

elab "#check_layers" : command => do
  let env ← getEnv
  let rank (m : Name) : CommandElabM Nat := do
    let some f := folderOf m | throwError "{m} is not a library module"
    let some r := layers.findIdx? (· == f) | throwError "{m} is in the folder {f}, which `layers` does not name"
    return r
  let mut count : Nat := 0
  for (m, data) in env.header.moduleNames.zip env.header.moduleData do
    if (folderOf m).isNone then continue
    for i in data.imports do
      if (folderOf i.module).isNone then continue
      if (← rank i.module) > (← rank m) then
        throwError "{m} imports {i.module}, a module of a higher folder"
      count := count + 1
  logInfo m!"{count} library imports stay in their folder or go down: {layers}"

#check_layers
