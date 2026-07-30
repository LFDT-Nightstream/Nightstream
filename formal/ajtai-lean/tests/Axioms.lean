import Ajtai.BorrowChunk
import Ajtai.SecurityBoundary
import Lean.Elab.Print

namespace AjtaiTests.Axioms

open Lean Elab Command

private def isNativeDecideCertificate (name : Name) : Bool :=
  match name.toString.splitOn "._native.native_decide.ax_" with
  | before :: after :: [] =>
      !before.isEmpty && !after.isEmpty &&
        after.toList.all (fun char => char.isDigit || char == '_' || char == '✝')
  | _ => false

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

#print axioms Nightstream.Implementation.R1CS.CenteredTernaryBorrowChunk.eval_stepPolynomial_eq_scalar
#print axioms Nightstream.Implementation.R1CS.CenteredTernaryBorrowChunk.eval_composePolynomial_eq_scalar
#print axioms Nightstream.Implementation.R1CS.CenteredTernaryBorrowChunk.chunkEquation_holds_iff_scalar
#print axioms Nightstream.Implementation.R1CS.CenteredTernaryBorrowChunk.chunkTwo_iff_scalarWitness
#print axioms Nightstream.Implementation.R1CS.CenteredTernaryBorrowChunk.scalarTwoValues_complement
#print axioms Nightstream.Implementation.R1CS.CenteredTernaryBorrowChunk.normalizedChunkBound_lt_five
#print axioms Nightstream.Implementation.R1CS.CenteredTernaryBorrowChunk.chunkSchedule_encoded_lt_modulus
#print axioms Nightstream.Implementation.R1CS.CenteredTernaryBorrowChunk.maximumChunkDegree_eq_five

#print axioms Ajtai.Parameters.goldilocksModulus_agrees
#print axioms Ajtai.Parameters.goldilocks_balancedTernary_window
#print axioms Ajtai.Parameters.everyCanonicalGoldilocks_has_41_trit_opening
#print axioms Ajtai.Parameters.forty_trits_insufficient
#print axioms Ajtai.Parameters.forty_one_trits_sufficient
#print axioms Ajtai.Parameters.fewer_than_41_trits_insufficient
#print axioms Ajtai.Parameters.digitCount_is_least_sufficient
#print axioms Ajtai.Parameters.phi81_ringDegree_eq

/-- info: 'Ajtai.LogInterval.ln_nat_bounds' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#audit_axioms Ajtai.LogInterval.ln_nat_bounds

/-- info: 'Ajtai.EstimatorModel.computedMaxRingColumns_is_largest' depends on axioms: [propext,
 Classical.choice,
 trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Ajtai.EstimatorModel.computedMaxRingColumns_is_largest

/-- info: 'Ajtai.EstimatorModel.computedMaxSourceFields_eq' depends on axioms: [propext,
 Classical.choice,
 trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Ajtai.EstimatorModel.computedMaxSourceFields_eq

/-- info: 'Ajtai.SecurityBoundary.collision_implies_msis_break' depends on axioms: [propext,
 Classical.choice,
 trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Ajtai.SecurityBoundary.collision_implies_msis_break

/-- info: 'Ajtai.SecurityBoundary.binding_of_msis_boundary' depends on axioms: [propext,
 Classical.choice,
 trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Ajtai.SecurityBoundary.binding_of_msis_boundary

end AjtaiTests.Axioms
