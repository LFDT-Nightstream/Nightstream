import CompressionLean

/-!
The axiom gate. `lake build` runs it.

- Every theorem in a `CompressionLean` module, a later one too, uses only
  `propext`, `Classical.choice` and `Quot.sound` (`#audit_library_axioms`).
  A `sorry` or a `native_decide` anywhere in the library fails the build.
- Each claim exists and uses exactly the axioms pinned beside it
  (`#pin_axioms`). A renamed or deleted claim fails the build.
-/

open Lean Elab Command

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

elab "#pin_axioms " decl:ident " [" pinned:ident,* "]" : command => do
  let name ← liftCoreM <| realizeGlobalConstNoOverloadWithInfo decl
  let expected ← pinned.getElems.mapM fun a => liftCoreM <| realizeGlobalConstNoOverloadWithInfo a
  let actual ← liftCoreM (collectAxioms name)
  unless (actual.qsort Name.lt) == (expected.qsort Name.lt) do
    throwError m!"{name} uses {actual}, pinned {expected}"

#audit_library_axioms CompressionLean

/-! ## Claims -/

#pin_axioms CompressionLean.Terminal.circuitPackage_implies_terminal [propext, Classical.choice, Quot.sound]
#pin_axioms CompressionLean.Layer1.layer1_sound [propext, Classical.choice, Quot.sound]
#pin_axioms CompressionLean.Layer1.layer1_restore_sound [propext, Classical.choice, Quot.sound]
#pin_axioms CompressionLean.Layer1.layer1_restore_budget [propext, Classical.choice, Quot.sound]
#pin_axioms CompressionLean.Shrink.shrink_sound [propext, Classical.choice, Quot.sound]
#pin_axioms CompressionLean.Circuit.lower_valid [propext, Classical.choice, Quot.sound]
#pin_axioms CompressionLean.Circuit.lower_sound [propext, Classical.choice, Quot.sound]
#pin_axioms CompressionLean.Bridge.replay_accepts_sound [propext, Classical.choice, Quot.sound]
#pin_axioms CompressionLean.Final.honestSetup_of_build [propext, Classical.choice, Quot.sound]
#pin_axioms CompressionLean.Final.parent_holds [propext, Classical.choice, Quot.sound]
#pin_axioms CompressionLean.Final.final_sound [propext, Classical.choice, Quot.sound]
#pin_axioms CompressionLean.Final.final_restore_sound [propext, Classical.choice, Quot.sound]
#pin_axioms CompressionLean.Final.final_budget [propext, Classical.choice, Quot.sound]
#pin_axioms CompressionLean.Final.final_restore_budget [propext, Classical.choice, Quot.sound]
#pin_axioms CompressionLean.Final.final_sound_parent [propext, Classical.choice, Quot.sound]
#pin_axioms CompressionLean.Final.final_restore_budget_parent [propext, Classical.choice, Quot.sound]
