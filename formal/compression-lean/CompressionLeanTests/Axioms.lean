import CompressionLean
import CompressionLeanTests.Inventory
import CompressionLeanTests.Reading

/-!
The axiom gate. `lake build` runs it.

- Every theorem in a `CompressionLean` module, a later one too, uses only
  `propext`, `Classical.choice` and `Quot.sound` (`#audit_library_axioms`).
  A `sorry` or a `native_decide` anywhere in the library fails the build.
  So does one in the gates' theorems (`CompressionLeanTests`, which today
  holds only the reading gate's `reads_of_witness`).
- The claims are the theorems of `CompressionLean.Claims`. Each claim of
  `Inventory.claims` is declared there and uses exactly the axioms pinned
  beside it, and each theorem there has a pin (`#pin_claims`). A renamed,
  deleted or new claim fails the build until `Inventory.claims` matches.
-/

open Lean Elab Command CompressionLeanTests.Inventory

/-- The axioms that every theorem of the library may use. -/
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
