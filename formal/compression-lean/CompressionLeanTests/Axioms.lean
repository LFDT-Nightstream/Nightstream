import CompressionLean
import CompressionLeanTests.BlockDigest
import CompressionLeanTests.Inventory
import CompressionLeanTests.Layers
import CompressionLeanTests.Ports
import CompressionLeanTests.Reading
import CompressionLeanTests.SpecCensus
import CompressionLeanTests.Stack
import CompressionLeanTests.Vocabulary
open Lean Elab Command CompressionLeanTests.Inventory

def allowedAxioms : Array Name := #[``propext, ``Classical.choice, ``Quot.sound]

elab "#audit_library_axioms " root:ident : command => do
  let env ← getEnv
  let mut count := 0
  for (name, info) in env.constants.toList do
    unless info matches .thmInfo _ do continue
    let some idx := env.getModuleIdxFor? name | continue
    unless root.getId.isPrefixOf env.header.moduleNames[idx.toNat]! do continue
    count := count + 1
    let bad := (← liftCoreM (collectAxioms name)).filter fun a => !allowedAxioms.contains a
    unless bad.isEmpty do
      throwError m!"{name} depends on disallowed axioms: {bad}"
  logInfo m!"{count} theorems in {root.getId} use only {allowedAxioms}"
elab "#no_package_axioms" : command => do
  let env ← getEnv
  let roots := `CompressionLean :: upperRoots
  let mut declared := env.constants.toList.filterMap fun (name, info) =>
    let m := if (env.getModuleIdxFor? name).isSome then moduleOf env name else env.mainModule
    if info matches .axiomInfo _ && roots.contains m.getRoot then some name else none
  -- This file imports the library and every gate, so only executables remain. Lake builds
  -- them first (`needs`), except `Status`, which imports a gate.
  for m in ← modulesUnder "CompressionLeanExe" do
    if (env.getModuleIdx? m).isSome || m == `CompressionLeanExe.Status then continue
    let olean : System.FilePath := ".lake/build/lib/lean" / (m.toString.replace "." "/" ++ ".olean")
    unless ← olean.pathExists do throwError "{m} is not built"
    let (data, _) ← readModuleData olean
    declared := declared ++ (data.constants.filter (· matches .axiomInfo _)).toList.map (·.name)
  unless declared.isEmpty do throwError m!"these declarations of {roots} are axioms: {declared}"
  logInfo m!"no module of {roots} declares an axiom"
elab "#pin_claims" : command => do
  let env ← getEnv
  for (claim, pinned) in claims do
    unless env.contains claim do throwError "claim {claim} is not declared"
    unless moduleOf env claim == claimsModule do throwError "claim {claim} is not in {claimsModule}"
    let actual := (← liftCoreM (collectAxioms claim)).toList
    unless actual.mergeSort Name.lt == pinned.mergeSort Name.lt do
      throwError m!"{claim} uses {actual}, pinned {pinned}"
  let some idx := env.getModuleIdx? claimsModule | throwError "no module {claimsModule}"
  for n in env.header.moduleData[idx.toNat]!.constNames do
    if let some (.thmInfo _) := env.find? n then
      unless n.isInternal || claims.any (·.1 == n) do throwError "{n} is a theorem of {claimsModule} but has no pin"
#audit_library_axioms CompressionLean

#audit_library_axioms CompressionLeanTests

#no_package_axioms

#pin_claims