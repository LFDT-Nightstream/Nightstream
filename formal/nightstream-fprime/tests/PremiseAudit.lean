import tests.AxiomAudit

/-! `#premise_census "file" theorem*` lists every hypothesis of each named theorem and compares
the list with `file`. `scripts/check_premises.py` holds the classification of each line
(`SECURITY_MODEL.md`, section Premises), so a new hypothesis fails the `static` phase until it has
a class. -/

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

elab "#premise_census " path:str space:str theorems:ident* : command => do
  let mut lines := #[]
  for decl in theorems do
    let name ← liftCoreM <| realizeGlobalConstNoOverloadWithInfo decl
    unless (← getEnv).find? name matches some (.thmInfo _) do
      throwError "{name} is not a theorem"
    lines := lines ++ (← liftTermElabM <| premiseLines space.getString.toName name)
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
