/-
Provenance: the full-name check and the compiled-body override check of `#audit_axioms` come
from dcSpark/ProtogaLattice `formal/tests/Meta/AxiomAudit.lean` at commit 96b57a1c801b, which
adapted this file and ported the checks from Ironwood's `Zcash/Meta/AxiomCheck.lean`. Unlike that
copy, the override walk here does not enter proofs.
-/
import Lean

namespace AxiomAudit

open Lean Elab Command

/-- Module roots of the Lean toolchain and Mathlib's pinned dependency stack. Compiled-body
overrides under these roots are trusted; every other root, ours and CompPoly included, is not. -/
def ambientRoots : List Name :=
  [`Init, `Std, `Lean, `Lake, `Batteries, `Mathlib, `Aesop, `Qq, `Plausible, `ProofWidgets,
    `ImportGraph, `LeanSearchClient, `Cli]

def isAmbient (env : Environment) (c : Name) : Bool :=
  match env.getModuleIdxFor? c with
  | some idx => ambientRoots.contains env.header.moduleNames[idx.toNat]!.getRoot
  | none => false

/-- The attributes on `c` that make compiled code run a body the kernel never checked. -/
def overrideAttrs (env : Environment) (c : Name) : List Name :=
  (if (Compiler.getImplementedBy? env c).isSome then [Compiler.implementedByAttr.attr.name] else [])
    ++ (if isExtern env c then [externAttr.attr.name] else [])

/-- Imported constants whose closure an earlier audit in this module walked without finding an
override. Attributes can only be attached in a constant's own module, so the verdict holds for the
rest of this module; constants of this module are walked every time. -/
initialize cleanImports : EnvExtension NameSet ← registerEnvExtension (pure {})

/-- The non-ambient constants that carry an override in the closure of `root` through statements
and definition bodies, rendered as `c (@[attr])`, and the imported constants the walk visited. The
walk does not enter ambient constants: no ambient module imports ours, so everything an ambient
constant mentions is ambient too. -/
def overridesInClosure (env : Environment) (root : Name) : List String × Array Name := Id.run do
  let clean := cleanImports.getState env
  let mut seen : NameSet := {}
  let mut imported : Array Name := #[]
  let mut todo := #[root]
  let mut found : Array (Name × List Name) := #[]
  while !todo.isEmpty do
    let c := todo.back!
    todo := todo.pop
    if seen.contains c || clean.contains c || isAmbient env c then continue
    seen := seen.insert c
    if (env.getModuleIdxFor? c).isSome then imported := imported.push c
    let attrs := overrideAttrs env c
    unless attrs.isEmpty do found := found.push (c, attrs)
    let some info := env.find? c | continue
    todo := todo ++ info.type.getUsedConstants
    -- The kernel checks proofs by reduction, never by compiled code, so an override reached
    -- only through a proof changes nothing that runs. Walking proofs made the audit 4x slower.
    unless info matches .thmInfo _ do
      if let some v := info.value? (allowOpaque := true) then todo := todo ++ v.getUsedConstants
    if let .inductInfo v := info then todo := todo ++ v.ctors.toArray
  let rendered := (found.qsort (fun a b => Name.lt a.1 b.1)).toList.map fun (c, attrs) =>
    s!"{c} ({", ".intercalate (attrs.map fun a => s!"@[{a}]")})"
  return (rendered, imported)

end AxiomAudit

open Lean Elab Command in
/-- `#audit_axioms foo` fails unless all three hold:
* `foo` is written as the full name of the declaration it resolves to, so no enclosing namespace
  or `open` can point the line at a different declaration;
* `foo` depends on no axiom beyond `propext`, `Classical.choice` and `Quot.sound`;
* no constant outside `AxiomAudit.ambientRoots` that `foo`'s statement reaches through statements
  and definition bodies has its compiled body overridden (`implemented_by`, `extern`).

`tests/AuditNegative.lean` pins its rejections. -/
elab "#audit_axioms " decl:ident : command => do
  let name ← liftCoreM <| realizeGlobalConstNoOverloadWithInfo decl
  let written := decl.getId
  let env ← getEnv
  if written != name then
    if env.contains written then
      throwError m!"{written} resolves here to {name}, a different declaration; audit it \
        outside the namespace or `open` that captures it"
    throwError m!"{written} is not fully qualified; it resolves to {name}. \
      Write `#audit_axioms {name}`."
  let axioms ← liftCoreM <| Lean.collectAxioms name
  let allowed : List Name := [``propext, ``Classical.choice, ``Quot.sound]
  let bad := axioms.toList.filter (fun a => !allowed.contains a)
  unless bad.isEmpty do
    throwError m!"{name} depends on disallowed axioms: {bad}"
  let (overrides, imported) := AxiomAudit.overridesInClosure env name
  unless overrides.isEmpty do
    throwError m!"{name} reaches compiled-body overrides outside \
      `AxiomAudit.ambientRoots`: {", ".intercalate overrides}"
  modifyEnv fun env =>
    AxiomAudit.cleanImports.modifyState env fun s => imported.foldl NameSet.insert s
  logInfo m!"{name}: {axioms.toList}"

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
