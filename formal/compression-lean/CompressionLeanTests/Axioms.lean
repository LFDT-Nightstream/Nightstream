import CompressionLean
import CompressionLeanTests.Inventory

/-!
The axiom gate. `lake build` runs it.

- Every theorem in a `CompressionLean` module, a later one too, uses only
  `propext`, `Classical.choice` and `Quot.sound` (`#audit_library_axioms`).
  A `sorry` or a `native_decide` anywhere in the library fails the build.
- Each claim of `Inventory.claims` exists and uses exactly the axioms pinned
  beside it (`#pin_claims`). A renamed or deleted claim fails the build.
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
  for (claim, pinned) in claims do
    unless (← getEnv).contains claim do throwError "claim {claim} is not declared"
    let actual := (← liftCoreM (collectAxioms claim)).toList
    unless actual.mergeSort Name.lt == pinned.mergeSort Name.lt do
      throwError m!"{claim} uses {actual}, pinned {pinned}"

#audit_library_axioms CompressionLean

#pin_claims
