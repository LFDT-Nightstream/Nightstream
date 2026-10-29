import CompressionLean
import CompressionLeanTests.Inventory
import CompressionLeanTests.Reading
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

#pin_claims