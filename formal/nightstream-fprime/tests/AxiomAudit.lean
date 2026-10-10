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

open Lean Elab Command in
/-- Fail unless every full name that the document at `path` cites in
backticks exists as a declaration or a module, and every cited declaration
passes the axiom audit. Cited names omit the common `NightstreamFPrime`
namespace and start with `Export.`, `Layout.`, `Lifecycle.` or `Spec.`. -/
elab "#endpoint_census " path:str : command => do
  let text ← IO.FS.readFile path.getString
  let roots := ["Export.", "Layout.", "Lifecycle.", "Spec."]
  let allowed : List Name := [``propext, ``Classical.choice, ``Quot.sound]
  let pieces := text.splitOn "`"
  let mut cited : List String := []
  for index in [0:pieces.length] do
    let piece := pieces[index]!
    if index % 2 == 1 && roots.any (piece.startsWith ·) &&
        piece.all (fun c => Lean.isIdRest c || c == '.') && !cited.contains piece then
      cited := cited ++ [piece]
  let env ← getEnv
  let mut declarations : Nat := 0
  let mut modules : Nat := 0
  for piece in cited do
    let name := (`NightstreamFPrime).append piece.toName
    if env.contains name then
      let axioms ← liftCoreM <| Lean.collectAxioms name
      let bad := axioms.toList.filter (fun a => !allowed.contains a)
      unless bad.isEmpty do
        throwError m!"{name} depends on disallowed axioms: {bad}"
      declarations := declarations + 1
    else if (env.getModuleIdx? name).isSome then
      modules := modules + 1
    else
      throwError m!"{path.getString} cites `{piece}`, which is not a declaration or module"
  logInfo m!"{path.getString}: {declarations} cited declarations audited; {modules} cited modules exist"
