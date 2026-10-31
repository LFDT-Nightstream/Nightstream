import Lean
import CompressionLeanRust

namespace CompressionLeanRustTests

open Lean Elab Command Meta

def standardAxioms : List Name := [``propext, ``Classical.choice, ``Quot.sound]

def rustItem (f : Name) : Option String :=
  match f.components.map toString with
  | "neo_verified" :: rest@(_ :: _ :: _) =>
    some s!"crates/neo-verified/src/{"/".intercalate rest.dropLast}.rs::{rest.getLast!}"
  | _ => none

/-- The constant `c` on one tuple of its explicit arguments, `fun x => c x.1 x.2.1 … x.2.2`, or `c`
itself when it takes one argument. -/
def tupled (c : Name) : MetaM Expr := do
  let fn ← mkConstWithLevelParams c
  forallTelescope (← inferType fn) fun args _ => do
    for arg in args do
      unless (← arg.fvarId!.getBinderInfo).isExplicit do throwError "{c} takes an implicit argument"
    if args.size ≤ 1 then return fn
    let types ← args.mapM inferType
    let tuple ← types.pop.foldrM (fun t acc => mkAppM ``Prod #[t, acc]) types.back!
    withLocalDeclD `x tuple fun x => do
      let mut rest := x
      let mut parts := #[]
      for _ in [0:args.size - 1] do
        parts := parts.push (← mkAppM ``Prod.fst #[rest])
        rest ← mkAppM ``Prod.snd #[rest]
      mkLambdaFVars #[x] (mkAppN fn (parts.push rest))

def statement (thm shape : Name) : MetaM (List Expr) := do
  let some info := (← getEnv).find? thm | throwError "ports.json names {thm}, which is not declared"
  unless info.type.isAppOf shape do throwError "{thm} does not state {shape}"
  let axioms := (← collectAxioms thm).toList
  unless axioms.all standardAxioms.contains do throwError "{thm} uses the axioms {axioms}"
  return (info.type.getAppArgs.toList.drop 3)

def checkEntry (entry : Json) : MetaM Bool := do
  let some proof := (entry.getObjVal? "proof").toOption | return false
  let port ← IO.ofExcept (entry.getObjValAs? Name "lean")
  let rust ← IO.ofExcept (entry.getObjValAs? String "rust")
  let sound ← IO.ofExcept (proof.getObjValAs? Name "sound")
  let complete ← IO.ofExcept (proof.getObjValAs? Name "complete")
  let [f, view, copy] ← statement sound ``CompressionLeanRust.Sound
    | throwError "{sound} has the wrong number of arguments"
  let [_, f', view', copy'] ← statement complete ``CompressionLeanRust.Complete
    | throwError "{complete} has the wrong number of arguments"
  unless f == f' && view == view' && copy == copy' do
    throwError "{sound} and {complete} state different functions, views or copies"
  let some fn ← lambdaTelescope f fun _ body => pure body.getAppFn.constName?
    | throwError "{sound} states a function that is not a constant"
  unless ← withReducible (isDefEq f (← tupled fn)) do
    throwError "{sound} states {f}, which is not {fn} on one tuple of its arguments"
  unless rustItem fn == some rust do
    throwError "{sound} is about {fn}, which is not Aeneas's translation of {rust}"
  unless view.getAppFn.constName?.any CompressionLeanRust.views.contains do
    throwError "{sound} reads the output through {view}, which is not in CompressionLeanRust.views"
  unless (copy.find? (·.isConstOf port)).isSome do
    throwError "{sound} states a copy that does not use {port}"
  return true

elab "#check_rust_proofs" : command => liftTermElabM do
  for view in CompressionLeanRust.views do
    unless (← getEnv).contains (view.appendAfter "_injective") do
      throwError "{view} is in CompressionLeanRust.views and has no {view}_injective theorem"
  let ports ← IO.ofExcept ((Json.parse (← IO.FS.readFile "ports.json")).bind (·.getArr?))
  let checked ← ports.filterM fun entry => checkEntry entry
  if checked.isEmpty then throwError "ports.json names no proof"

#check_rust_proofs

end CompressionLeanRustTests
