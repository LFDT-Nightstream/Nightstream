import NightstreamFPrime.Export.Stage1.PerApplicationCompactPreservation

/-!
Owns the unconditional preservation theorem for the validated Stage 1 prefix
inside every Lean-authored per-application package.

All row categories are covered. This module does not select a production
application or claim final package conformance.
-/

namespace NightstreamFPrime.Export.Stage1.PerApplicationCanonicalPreservation

open NightstreamFPrime.Circuit
open NightstreamFPrime.Export.Package
open NightstreamFPrime.Export.Stage1.PerApplicationPackage
open NightstreamFPrime.Export.Stage1.PerApplicationPreservation
open NightstreamFPrime.Lifecycle

theorem canonicalCompactRows
    (program : Lifecycle.Stage1.Application.Program) :
    ∀ invocation ∈ basePackage.compactRowInvocations,
      ∀ template,
        basePackage.compactRowTemplates[invocation.templateIndex]? =
            some template →
          ∀ row ∈ template.rows,
            instantiateCompactRow
                (shiftCompactRowInvocation program invocation) row =
              CompactRows.renameRow (shiftColumn program)
                (instantiateCompactRow invocation row) := by
  intro invocation invocationMember template templateEquation row rowMember
  change invocation ∈ Data.compactRowInvocations () at invocationMember
  rw [Data.compactRowInvocations_eq, List.mem_append] at invocationMember
  rcases invocationMember with first54Member | combinationMember
  · exact PerApplicationCompactPreservation.first54Rows program invocation
      first54Member template templateEquation row rowMember
  · exact PerApplicationCompactPreservation.combinationRows program invocation
      combinationMember template templateEquation row rowMember

theorem canonicalShiftCompatible
    (program : Lifecycle.Stage1.Application.Program) :
    ShiftCompatible program where
  hashRows := canonicalHashRows program
  permutationRows := canonicalPermutationRows program
  compactRows := canonicalCompactRows program

/-- Every row of the final per-application package implies every row of the
already validated Stage 1 prefix under the exact column pullback. -/
theorem packageRows_imply_validatedPrefix
    (program : Lifecycle.Stage1.Application.Program) (env : Env)
    (holds : (package program).RowsHold env) :
    basePackage.RowsHold (baseEnv program env) :=
  packageRows_imply_basePackage program env (canonicalShiftCompatible program)
    holds

end NightstreamFPrime.Export.Stage1.PerApplicationCanonicalPreservation
