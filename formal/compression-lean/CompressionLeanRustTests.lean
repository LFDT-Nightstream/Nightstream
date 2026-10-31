import Lean
import CompressionLeanRust

namespace CompressionLeanRustTests

open Lean Elab Command

def standardAxioms : List Name := [``propext, ``Classical.choice, ``Quot.sound]

def rustItem (f : Name) : Option String :=
  match f.components.map toString with
  | "neo_verified" :: rest@(_ :: _ :: _) =>
    some s!"crates/neo-verified/src/{"/".intercalate rest.dropLast}.rs::{rest.getLast!}"
  | _ => none

def statement (thm shape : Name) : CoreM (List Expr) := do
  let some info := (← getEnv).find? thm | throwError "ports.json names {thm}, which is not declared"
  unless info.type.isAppOf shape do throwError "{thm} does not state {shape}"
  let axioms := (← collectAxioms thm).toList
  unless axioms.all standardAxioms.contains do throwError "{thm} uses the axioms {axioms}"
  return (info.type.getAppArgs.toList.drop 3)

def checkEntry (entry : Json) : CoreM Bool := do
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
  let some fn := f.constName? | throwError "{sound} states a function that is not a constant"
  unless rustItem fn == some rust do
    throwError "{sound} is about {fn}, which is not Aeneas's translation of {rust}"
  unless view.getAppFn.constName?.any CompressionLeanRust.views.contains do
    throwError "{sound} reads the output through {view}, which is not in CompressionLeanRust.views"
  unless (copy.find? (·.isConstOf port)).isSome do
    throwError "{sound} states a copy that does not use {port}"
  return true

elab "#check_rust_proofs" : command => liftCoreM do
  for view in CompressionLeanRust.views do
    unless (← getEnv).contains (view.appendAfter "_injective") do
      throwError "{view} is in CompressionLeanRust.views and has no {view}_injective theorem"
  let ports ← IO.ofExcept ((Json.parse (← IO.FS.readFile "ports.json")).bind (·.getArr?))
  let checked ← ports.filterM checkEntry
  if checked.isEmpty then throwError "ports.json names no proof"

#check_rust_proofs

end CompressionLeanRustTests
