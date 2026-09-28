import Lean
import tests.AxiomAudit

/-! Owns declaration metadata and exact closure checks for the evidence CLI.
Metadata is inspection data. It is not a proof of semantic correspondence.
-/

namespace LeanGraph

open Lean Elab Command Meta

/-- Preserve sharing in Lean terms. Expanding a proof DAG into a JSON tree
can duplicate large subterms. Canonical nodes use exact JSON equality. -/
structure ExpressionGraph where
  expressions : ExprMap Nat := {}
  canonical : Std.HashMap Json Nat := {}
  nodes : Array Json := #[]

partial def encodeExpression (value : Expr) : StateM ExpressionGraph Nat := do
  if let some index := (← get).expressions[value]? then return index
  if let .mdata _ inner := value then
    let index ← encodeExpression inner
    modify fun state => { state with expressions := state.expressions.insert value index }
    return index
  let node (tag : String) (values : Array Json) := Json.arr (#[toJson tag] ++ values)
  let reference (term : Expr) : StateM ExpressionGraph Json := do
    return toJson (← encodeExpression term)
  let encoded : Json ← match value with
    | .bvar index => pure <| node "bound" #[toJson index]
    | .fvar name => pure <| node "free" #[toJson name.name.toString]
    | .mvar name => pure <| node "meta" #[toJson name.name.toString]
    | .sort level => pure <| node "sort" #[toJson (reprStr level)]
    | .const name levels => pure <| node "constant" #[toJson name.toString, toJson (levels.map reprStr)]
    | .app fn arg => do pure <| node "application" #[← reference fn, ← reference arg]
    | .lam _ type body binder => do
        pure <| node "lambda" #[toJson (reprStr binder), ← reference type, ← reference body]
    | .forallE _ type body binder => do
        pure <| node "forall" #[toJson (reprStr binder), ← reference type, ← reference body]
    | .letE _ type value body nondep => do
        pure <| node "let" #[← reference type, ← reference value, ← reference body, toJson nondep]
    | .lit value => pure <| node "literal" #[toJson (reprStr value)]
    | .mdata _ inner => reference inner
    | .proj name index inner => do
        pure <| node "projection" #[toJson name.toString, toJson index, ← reference inner]
  let state : ExpressionGraph ← get
  let index := state.canonical[encoded]?.getD state.nodes.size
  let nodes := if index == state.nodes.size then state.nodes.push encoded else state.nodes
  set ({ expressions := state.expressions.insert value index,
         canonical := state.canonical.insert encoded index, nodes := nodes } : ExpressionGraph)
  return index

/-- Remove source metadata and bound variable names without unfolding terms. -/
def expression (value : Expr) : Json :=
  let (root, graph) := (encodeExpression value).run {}
  Json.mkObj [("root", toJson root), ("nodes", toJson graph.nodes)]

def meaningExpressions (info : ConstantInfo) : Array Expr :=
  match info with
  | .defnInfo value => #[value.type, value.value]
  | .opaqueInfo value => #[value.type, value.value]
  | .recInfo value => #[value.type] ++ value.rules.toArray.map (·.rhs)
  | _ => #[info.type]

def extraDependencies (info : ConstantInfo) : Array Name :=
  match info with
  | .inductInfo value => (value.all ++ value.ctors).toArray
  | .ctorInfo value => #[value.induct]
  | .recInfo value => value.rules.toArray.map (·.ctor)
  | _ => #[]

def declarationKind (info : ConstantInfo) : String :=
  match info with
  | .defnInfo _ => "definition"
  | .thmInfo _ => "theorem"
  | .opaqueInfo _ => "opaque"
  | .inductInfo _ => "inductive"
  | .ctorInfo _ => "constructor"
  | .recInfo _ => "recursor"
  | .axiomInfo _ => "kernel-assumption"
  | .quotInfo _ => "quotient"

def declarationShape (info : ConstantInfo) : Json :=
  match info with
  | .inductInfo value => toJson
      (value.numParams, value.numIndices, value.numNested, value.isRec,
        value.isReflexive, value.all.map Name.toString, value.ctors.map Name.toString)
  | .ctorInfo value => toJson (value.induct.toString, value.cidx, value.numParams, value.numFields)
  | .recInfo value => toJson (value.numParams, value.numIndices,
      value.numMotives, value.numMinors, value.k,
      value.rules.map fun rule => (rule.ctor.toString, rule.nfields))
  | _ => Json.null

/-- Report the actual resolved module file. External provenance is checked by
the reader against its source manifest and pinned toolchain, never a namespace. -/
def origin (env : Environment) (name : Name) : CoreM (Name × String × Bool) := do
  let moduleName := match env.getModuleIdxFor? name with
    | some index => env.header.moduleNames[index.toNat]!
    | none => env.mainModule
  let relative := moduleName.toString.replace "." "/" ++ ".lean"
  let cwd ← IO.FS.realPath (← IO.currentDir)
  let source := cwd / relative
  if env.getModuleIdxFor? name |>.isNone then
    return (moduleName, source.toString, true)
  let compiled ← IO.FS.realPath (← findOLean moduleName)
  let localBuild := (cwd / ".lake/build/lib/lean").toString ++ "/"
  if compiled.toString.startsWith localBuild && (← source.pathExists) then
    return (moduleName, source.toString, true)
  return (moduleName, compiled.toString, false)

def metadata (root : Name) : MetaM Json := do
  let env ← getEnv
  let mut pending := #[root]
  let mut seen : NameSet := {}
  let mut nodes : Array Json := #[]
  let mut complete := true
  while !pending.isEmpty do
    let name := pending.back!
    pending := pending.pop
    if seen.contains name then continue
    seen := seen.insert name
    let some info := env.find? name | do
      complete := false
      nodes := nodes.push (Json.mkObj [("name", toJson name.toString), ("missing", toJson true)])
      continue
    let (moduleName, path, localSource) ← origin env name
    let expressions := meaningExpressions info
    let dependencies := ((expressions.flatMap Expr.getUsedConstants) ++ extraDependencies info).qsort Name.lt
    let proof := match info with
      | .thmInfo value => value.value.getUsedConstants.qsort Name.lt
      | _ => #[]
    if expressions.any (fun value => value.hasFVar || value.hasMVar) then
      complete := false
    let display (value : Expr) : MetaM String :=
      withOptions (fun options => options.setBool `pp.fullNames true |>.setBool `pp.universes true) do
        return (← ppExpr value).pretty
    let statement ← display info.type
    let proposition ← match info with
      | .defnInfo value =>
          if value.type == mkSort .zero then display value.value else pure ""
      | _ => pure ""
    nodes := nodes.push (Json.mkObj [
      ("name", toJson name.toString), ("module", toJson moduleName.toString),
      ("origin", toJson path), ("local", toJson localSource),
      ("kind", toJson (declarationKind info)),
      ("levels", toJson (info.levelParams.map Name.toString)),
      ("statement", toJson statement), ("proposition", toJson proposition),
      ("type_expression", if localSource then expression info.type else Json.null),
      ("meaning", if localSource then toJson (expressions.map expression) else Json.null),
      ("shape", if localSource then declarationShape info else Json.null),
      ("meaning_dependencies", toJson (dependencies.map Name.toString)),
      ("proof_dependencies", toJson (proof.map Name.toString)),
      ("proof", match info with
        | .thmInfo value => if localSource then expression value.value else Json.null
        | _ => Json.null)])
    if localSource then pending := pending ++ dependencies ++ proof
  return Json.mkObj [("schema", toJson (1 : Nat)), ("root", toJson root.toString),
    ("runtime", toJson (← getBuildDir).toString),
    ("complete", toJson complete), ("nodes", toJson nodes)]

elab "#evidence_export " target:ident : command => do
  let name ← liftCoreM <| realizeGlobalConstNoOverloadWithInfo target
  let result ← liftTermElabM <| metadata name
  liftIO <| IO.println ("LEAN_GRAPH_METADATA " ++ result.compress)

elab "#evidence_closed " target:ident " by " witness:ident : command => do
  let targetName ← liftCoreM <| realizeGlobalConstNoOverloadWithInfo target
  let witnessName ← liftCoreM <| realizeGlobalConstNoOverloadWithInfo witness
  let env ← getEnv
  let some targetInfo := env.find? targetName | throwError "missing target"
  let some witnessInfo := env.find? witnessName | throwError "missing closure witness"
  unless targetInfo.levelParams.isEmpty && witnessInfo.levelParams.isEmpty do
    throwError "register a complete target with explicit universe parameters"
  unless witnessInfo matches .thmInfo _ do
    throwError "closure witness must be a checked theorem"
  let matchesTarget ← liftTermElabM <| isDefEq witnessInfo.type (mkConst targetName)
  unless matchesTarget do throwError "closure theorem does not prove the exact registered target"
  let used ← liftCoreM <| Lean.collectAxioms witnessName
  let allowed : List Name := [``propext, ``Classical.choice, ``Quot.sound]
  unless used.all (allowed.contains ·) do throwError "closure uses disallowed kernel assumptions"
  liftIO <| IO.println ("LEAN_GRAPH_CLOSED " ++ (Json.mkObj [
    ("target", toJson targetName.toString), ("witness", toJson witnessName.toString)]).compress)

end LeanGraph
