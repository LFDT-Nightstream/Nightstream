import Lean.Elab.Print

/-!
Lean 4.30 records each compiler-generated `native_decide` certificate as an
internal trusted declaration in addition to `Lean.trustCompiler`. The audited report retains
the complete fail-closed collection step, but presents those generated names as
their trusted-computation category so expectations stay reviewable. Every
non-native dependency remains distinct and therefore changes the guarded output.
-/

namespace NightstreamTests.Axioms

open Lean Elab Command

private def isNativeDecideCertificate (name : Name) : Bool :=
  match name.toString.splitOn "._native.native_decide.ax_" with
  | before :: after :: [] =>
      !before.isEmpty && !after.isEmpty &&
        after.toList.all (fun char => char.isDigit || char == '_' || char == '✝')
  | _ => false

#guard isNativeDecideCertificate `Example._native.native_decide.ax_1_1
#guard !isNativeDecideCertificate `Example._native.native_decide.ax_unreviewed

private def normalizeAxioms (axioms : Array Name) : Array Name :=
  axioms.foldl (init := #[]) fun normalized axiomName =>
    let normalizedAxiom :=
      if isNativeDecideCertificate axiomName then
        ``Lean.trustCompiler
      else
        axiomName
    if normalized.contains normalizedAxiom then
      normalized
    else
      normalized.push normalizedAxiom

syntax (name := printAuditedAxioms) "#audit_axioms " ident : command

@[command_elab printAuditedAxioms]
def elabPrintAuditedAxioms : CommandElab
  | `(#audit_axioms $id:ident) => withRef id do
      let constants ← liftCoreM <| realizeGlobalConstWithInfos id
      for constant in constants do
        let axioms ← collectAxioms constant
        let normalized := (normalizeAxioms axioms).qsort Name.lt
        if normalized.isEmpty then
          logInfo m!"'{constant}' does not depend on any axioms"
        else
          logInfo m!"'{constant}' depends on axioms: \
            {normalized.map MessageData.ofConstName |>.toList}"
  | _ => throwUnsupportedSyntax

end NightstreamTests.Axioms
