import CompressionLean

open Lean Elab Command

def layers : List String :=
  ["Lib", "Fields", "Production", "Game", "Terminal", "Layer1", "Ports", "Shrink", "Lowering", "Spec", "Bridge",
    "Final", "Checks", "Sim", "Claims"]

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
