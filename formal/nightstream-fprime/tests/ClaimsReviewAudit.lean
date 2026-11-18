import Lean

/-! `#claims_review "file"` lists each theorem of `NightstreamFPrime.Claims` with its source range,
and each Nightstream name that its statement uses, one level deep, and writes them to `file` as
JSON when `claimsReview.write` is set (`scripts/validate.sh review`). `scripts/claims_review.py`
writes the review files from that JSON. -/

register_option claimsReview.write : Bool := {
  defValue := false
  descr := "write the claims review data to the file"
}

open Lean Elab Command

namespace ClaimsReview

/-- The name that a glossary shows for a constant: a structure for its projections and
constructors, nothing for a generated or foreign name. -/
def glossaryName (env : Environment) (name : Name) : Option Name :=
  let owner := match env.find? name with
    | some (.ctorInfo info) => info.induct
    | _ => match env.getProjectionFnInfo? name with
      | some info => match env.find? info.ctorName with
        | some (.ctorInfo ctor) => ctor.induct
        | _ => name
      | none => name
  let text := owner.toString
  let generated := ["match_", "proof_", "_sunfold", "_unsafe_rec", ".eq_", "._", ".inst"].any
    fun part => (text.splitOn part).length > 1
  let last := match owner.components.getLast? with
    | some component => component.toString
    | none => ""
  if !(`NightstreamFPrime).isPrefixOf owner || owner.isInternalDetail || generated ||
      last.startsWith "inst" then none
  else some owner

def kindOf (env : Environment) (name : Name) : String :=
  if isStructure env name then "structure" else
  match env.find? name with
  | some (.defnInfo _) => "def"
  | some (.thmInfo _) => "theorem"
  | some (.inductInfo _) => "inductive"
  | some (.opaqueInfo _) => "opaque"
  | some (.axiomInfo _) => "axiom"
  | _ => "other"

def location (name : Name) : CoreM Json := do
  let env ← getEnv
  let modName := match env.getModuleIdxFor? name with
    | some idx => (env.allImportedModuleNames[idx.toNat]?).getD .anonymous
    | none => .anonymous
  match ← findDeclarationRanges? name with
  | some ranges => return Json.mkObj [("module", toJson modName.toString),
      ("start", toJson ranges.range.pos.line), ("stop", toJson ranges.range.endPos.line)]
  | none => return Json.mkObj [("module", toJson modName.toString)]

elab "#claims_review " path:str : command => do
  let env ← getEnv
  let claims := env.constants.toList.filterMap fun (name, info) =>
    if (`NightstreamFPrime.Claims).isPrefixOf name && !name.isInternalDetail &&
        info matches .thmInfo _ then some (name, info.type) else none
  let mut claimRows : Array Json := #[]
  let mut used : Std.HashSet Name := {}
  for (name, type) in claims do
    let names := (type.getUsedConstants.filterMap (glossaryName env)).toList.eraseDups
    used := names.foldl (·.insert ·) used
    let place ← liftCoreM (location name)
    claimRows := claimRows.push (Json.mkObj [("name", toJson name.toString),
      ("location", place), ("uses", toJson (names.map Name.toString))])
  let mut glossary : Array Json := #[]
  for name in used.toList do
    let place ← liftCoreM (location name)
    glossary := glossary.push (Json.mkObj [("name", toJson name.toString),
      ("kind", toJson (kindOf env name)), ("location", place)])
  let out := Json.mkObj [("claims", Json.arr claimRows), ("glossary", Json.arr glossary)]
  if claimsReview.write.get (← getOptions) then
    IO.FS.writeFile path.getString out.pretty
    logInfo m!"{claimRows.size} claims, {glossary.size} glossary names: {path.getString}"
  else
    logInfo m!"{claimRows.size} claims, {glossary.size} glossary names"

end ClaimsReview
