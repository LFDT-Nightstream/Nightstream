import tests.AxiomAudit

/-! `#premise_census "file" "Package" "Root"` reads each theorem of the namespace `Package.Root`.
It fails when the namespace holds a declaration that is not a theorem, when the axioms of a
theorem are not among the allowed three, or when a proof is not one application of a constant
outside the namespace. It lists every hypothesis of each theorem and compares the list with `file`.
`scripts/check_premises.py` holds the classification of each line (`SECURITY_MODEL.md`, section
Premises), so a new hypothesis fails the `static` phase until it has a class. -/

register_option premiseCensus.write : Bool := {
  defValue := false
  descr := "write the premise list in place of a comparison with it"
}

namespace PremiseCensus

open Lean Meta Elab Command

/-- The constant at the head of a hypothesis, under its leading quantifiers. -/
partial def headOf (type : Expr) : String :=
  match type.consumeMData with
  | .forallE _ _ body _ => headOf body
  | other => match other.getAppFn with
    | .const name _ => name.toString
    | _ => "_"

/-- The hypotheses that a binder of type `type` carries: the binder when its type is a
proposition, and each proof field when its type is a record of this package. -/
partial def premisesOf (label : String) (type : Expr) (depth : Nat) :
    MetaM (Array (String × String)) := do
  let type ← whnfR type
  let env ← getEnv
  match type.getAppFn with
  | .const record levels =>
    if depth > 0 && isStructure env record && DocumentCensus.inPackage env record then
      let ctor := getStructureCtor env record
      let fields := getStructureFields env record
      let applied ← instantiateForall
        (ctor.type.instantiateLevelParams ctor.levelParams levels)
        (type.getAppArgs.extract 0 ctor.numParams)
      return ← forallTelescope applied fun values _ => do
        let mut found := #[]
        for value in values, field in fields do
          found := found ++ (← premisesOf s!"{label}.{field}" (← inferType value) (depth - 1))
        return found
  | _ => pure ()
  if ← isProp type then return #[(label, headOf type)] else return #[]

def premiseLines (space : Name) (decl : Name) : MetaM (Array String) := do
  let info ← getConstInfo decl
  let shown := (decl.replacePrefix space .anonymous).toString
  forallTelescope info.type fun binders _ => do
    let mut lines := #[]
    for binder in binders do
      let name := (← binder.fvarId!.getUserName).eraseMacroScopes.toString
      for (label, head) in ← premisesOf name (← inferType binder) 4 do
        let head := if head.startsWith s!"{space}." then (head.drop (s!"{space}.".length)).toString else head
        lines := lines.push s!"{shown}\t{label}\t{head}"
    if lines.isEmpty then
      lines := lines.push s!"{shown}\t-\t-"
    return lines

/-- The constant that the proof of a claim applies to its own binders, under those binders. -/
def proofHead : Expr → Option Name
  | .lam _ _ body _ => proofHead body
  | .mdata _ body => proofHead body
  | body => match body.getAppFn with
    | .const name _ => if body.getAppArgs.all (·.isBVar) then some name else none
    | _ => none

elab "#premise_census " path:str space:str root:str : command => do
  let package := space.getString.toName
  let claims := package ++ root.getString.toName
  let env ← getEnv
  let allowed : List Name := [``propext, ``Classical.choice, ``Quot.sound]
  let mut theorems : Array Name := #[]
  for (name, info) in env.constants.toList do
    unless claims.isPrefixOf name && !name.isInternalDetail do continue
    match info with
    | .thmInfo value =>
      match proofHead value.value with
      | some head =>
        if claims.isPrefixOf head then
          throwError "the proof of {name} applies {head}, which is in {claims}"
        unless env.find? head matches some (.thmInfo _) do
          throwError "the proof of {name} applies {head}, which is not a theorem"
      | none => throwError "the proof of {name} is not one theorem applied to the binders of {name}"
      let bad := (← liftCoreM <| collectAxioms name).toList.filter (fun a => !allowed.contains a)
      unless bad.isEmpty do throwError "{name} depends on disallowed axioms: {bad}"
      theorems := theorems.push name
    | _ => throwError "{name} is in {claims}, but it is not a theorem"
  theorems := theorems.qsort fun left right => Name.cmp left right == .lt
  let mut lines := #[]
  for name in theorems do
    lines := lines ++ (← liftTermElabM <| premiseLines package name)
  let text := String.intercalate "\n" lines.toList ++ "\n"
  let file : System.FilePath := path.getString
  if premiseCensus.write.get (← getOptions) then
    IO.FS.writeFile file text
    logInfo m!"wrote {lines.size} premise lines to {file}"
  else if (← file.pathExists) && (← IO.FS.readFile file) == text then
    logInfo m!"{file}: {lines.size} premise lines of {theorems.size} theorems are current"
  else
    throwError "{file} is not the premise list of its theorems: run \
      `scripts/validate.sh premises`, then classify each new line in SECURITY_MODEL.md"

end PremiseCensus
