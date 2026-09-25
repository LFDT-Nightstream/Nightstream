import Lean

/-! Shared fail-closed axiom audit command. -/

open Lean Elab Command in
/-- Fail unless `decl` depends only on the permitted kernel axioms. -/
elab "#audit_axioms " decl:ident : command => do
  let name ← liftCoreM <| realizeGlobalConstNoOverloadWithInfo decl
  let axioms ← liftCoreM <| Lean.collectAxioms name
  let allowed : List Name := [``propext, ``Classical.choice, ``Quot.sound]
  let bad := axioms.toList.filter (fun a => !allowed.contains a)
  if bad.isEmpty then
    logInfo m!"{name}: {axioms.toList}"
  else
    throwError m!"{name} depends on disallowed axioms: {bad}"
