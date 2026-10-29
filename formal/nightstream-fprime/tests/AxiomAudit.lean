import Lean

open Lean Elab Command in
elab "#audit_axioms " decl:ident : command => do
  let name ← liftCoreM <| realizeGlobalConstNoOverloadWithInfo decl
  let axioms ← liftCoreM <| Lean.collectAxioms name
  let allowed : List Name := [``propext, ``Classical.choice, ``Quot.sound]
  let bad := axioms.toList.filter (fun a => !allowed.contains a)
  if bad.isEmpty then
    logInfo m!"{name}: {axioms.toList}"
  else
    throwError m!"{name} depends on disallowed axioms: {bad}"

namespace DocumentCensus

open Lean Elab Command

-- Declarations outside these packages come from the pinned toolchain and Mathlib.
def packageRoots : List Name := [`NightstreamFPrime, `CompressionLean, `tests, `CompressionLeanTests]

def inPackage (env : Environment) (name : Name) : Bool :=
  match env.getModuleIdxFor? name with
  | some index => packageRoots.contains (env.header.moduleNames[index.toNat]!).getRoot
  | none => true

-- A proof is not part of a theorem's meaning, so only a definition contributes its value.
def meaningOf (info : ConstantInfo) : UInt64 × Array Name :=
  let type := info.type
  match info with
  | .defnInfo value =>
    (mixHash type.hash value.value.hash, type.getUsedConstants ++ value.value.getUsedConstants)
  | .opaqueInfo value =>
    (mixHash type.hash value.value.hash, type.getUsedConstants ++ value.value.getUsedConstants)
  | .inductInfo value => (type.hash, type.getUsedConstants ++ value.ctors.toArray)
  | .ctorInfo value => (type.hash, type.getUsedConstants.push value.induct)
  | _ => (type.hash, type.getUsedConstants)

def fingerprint (env : Environment) (root : Name) : UInt64 := Id.run do
  let mut seen : NameSet := {}
  let mut pending : Array Name := #[root]
  let mut meanings : Array (Name × UInt64) := #[]
  while !pending.isEmpty do
    let name := pending.back!
    pending := pending.pop
    if seen.contains name then continue
    seen := seen.insert name
    let some info := env.find? name | continue
    unless inPackage env name do continue
    let (meaning, uses) := meaningOf info
    meanings := meanings.push (name, meaning)
    pending := pending ++ uses
  let ordered := meanings.qsort fun left right => Name.cmp left.1 right.1 == .lt
  return ordered.foldl (fun acc (name, meaning) => mixHash (mixHash acc name.hash) meaning) 0

def hex (value : UInt64) : String :=
  let digits := Nat.toDigits 16 value.toNat
  (List.replicate (16 - digits.length) '0' ++ digits).foldl String.push ""

def census (path : String) (space : Name) (roots : List String) (pins : Bool) :
    CommandElabM Unit := do
  let text ← IO.FS.readFile path
  let allowed : List Name := [``propext, ``Classical.choice, ``Quot.sound]
  let pieces := text.splitOn "`"
  let env ← getEnv
  let mut errors : Array String := #[]
  let mut audited : NameSet := {}
  let mut declarations : Nat := 0
  let mut modules : NameSet := {}
  let mut moduleCount : Nat := 0
  let mut prints : NameMap String := {}
  let mut pinCount : Nat := 0
  for index in [0:pieces.length] do
    let piece := pieces[index]!
    unless index % 2 == 1 && roots.any (piece.startsWith ·) &&
        piece.all (fun c => isIdRest c || c == '.') do continue
    let name := space.append piece.toName
    let some info := env.find? name
      | if (env.getModuleIdx? name).isSome then
          unless modules.contains name do
            modules := modules.insert name
            moduleCount := moduleCount + 1
        else
          errors := errors.push s!"{path} cites `{piece}`, which is not a declaration or module"
        continue
    unless audited.contains name do
      audited := audited.insert name
      declarations := declarations + 1
      let axioms ← liftCoreM <| collectAxioms name
      let bad := axioms.toList.filter (fun a => !allowed.contains a)
      unless bad.isEmpty do
        errors := errors.push s!"{name} depends on disallowed axioms: {bad}"
    unless pins do continue
    let pin := if pieces.getD (index + 1) "" == " (" && (pieces.getD (index + 3) "").startsWith ")"
      then some (pieces.getD (index + 2) "") else none
    match info, pin with
    | .thmInfo _, _ =>
      let current := (prints.find? name).getD (hex (fingerprint env name))
      prints := prints.insert name current
      match pin with
      | some value =>
        if value == current then
          pinCount := pinCount + 1
        else
          errors := errors.push s!"`{piece}` is pinned at `{value}`, but its statement now has \
            fingerprint `{current}`: reread the section that cites it, then update the pin"
      | none => errors := errors.push s!"`{piece}` is a theorem without a pin: write `{piece}` (`{current}`)"
    | _, some _ => errors := errors.push s!"`{piece}` has a pin, but it is not a theorem"
    | _, none => pure ()
  unless errors.isEmpty do
    throwError (String.intercalate "\n" errors.toList)
  logInfo m!"{path}: {declarations} cited declarations audited; {pinCount} current pins; \
    {moduleCount} cited modules exist"

end DocumentCensus

open Lean Elab Command in
elab "#endpoint_census " path:str : command =>
  DocumentCensus.census path.getString `NightstreamFPrime ["Export.", "Layout.", "Lifecycle.", "Spec."] false

-- `#spec_census "path" "Namespace" "Root." ...`: every cited theorem also needs its current pin.
open Lean Elab Command in
elab "#spec_census " path:str space:str roots:str* : command =>
  DocumentCensus.census path.getString
    (if space.getString.isEmpty then .anonymous else space.getString.toName)
    (roots.toList.map (·.getString)) true
