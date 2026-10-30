import CompressionLean
import CompressionLeanTests.Inventory

open Lean Elab Command CompressionLeanTests.Inventory

def vocabularyFolders : List String := ["Fields", "Production", "Game", "Terminal", "Layer1", "Ports", "Spec"]

/-- F′, Mathlib and core. -/
def outsideRoots : List Name := [`NightstreamFPrime, `Mathlib, `Init, `Std, `Lean]

elab "#check_vocabulary" : command => do
  let env ← getEnv
  let mut named : NameSet := {}
  let mut bad : Array MessageData := #[]
  for (claim, _) in claims do
    let some info := env.find? claim | throwError "claim {claim} is not declared"
    for c in info.type.getUsedConstants do
      let m := moduleOf env c
      match m.components with
      | `CompressionLean :: folder :: _ =>
        if vocabularyFolders.contains folder.toString then
          named := named.insert c
        else
          bad := bad.push m!"{claim} names {c}, of {m}"
      | root :: _ => unless outsideRoots.contains root do bad := bad.push m!"{claim} names {c}, of {m}"
      | [] => bad := bad.push m!"{claim} names {c}, of no module"
  unless bad.isEmpty do
    throwError m!"a claim names only definitions of {vocabularyFolders}, F′, Mathlib or core:\n{MessageData.joinSep bad.toList "\n"}"
  logInfo m!"the claims name {named.size} library constants, each of {vocabularyFolders}"

#check_vocabulary
