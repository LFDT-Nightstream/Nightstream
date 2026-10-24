import CompressionLean
import CompressionLeanTests.Inventory

/-!
The vocabulary gate. `lake build` runs it.

A claim is stated in the vocabulary of `Spec/` and of the model and port
folders: every constant in the type of a claim of `Inventory.claims` is
declared in a module of a folder of `vocabularyFolders`, or outside
`CompressionLean` (F′, Mathlib, core). A constant of a proof folder (`Lib`,
`Shrink`, `Lowering`, `Bridge`, `Final`, `Checks`, `Sim`, `Claims`) fails the
build, so an auditor reads the claims without the proofs.
-/

open Lean Elab Command CompressionLeanTests.Inventory

/-- The folders whose definitions a claim's type may name. -/
def vocabularyFolders : List String := ["Fields", "Production", "Game", "Terminal", "Layer1", "Ports", "Spec"]

elab "#check_vocabulary" : command => do
  let env ← getEnv
  let mut named : NameSet := {}
  let mut bad : Array MessageData := #[]
  for (claim, _) in claims do
    let some info := env.find? claim | throwError "claim {claim} is not declared"
    for c in info.type.getUsedConstants do
      let m := moduleOf env c
      let `CompressionLean :: folder :: _ := m.components | continue
      if vocabularyFolders.contains folder.toString then
        named := named.insert c
      else
        bad := bad.push m!"{claim} names {c}, of {m}"
  unless bad.isEmpty do
    throwError m!"a claim names only definitions of {vocabularyFolders}, F′, Mathlib or core:\n{MessageData.joinSep bad.toList "\n"}"
  logInfo m!"the claims name {named.size} library constants, each of {vocabularyFolders}"

#check_vocabulary
