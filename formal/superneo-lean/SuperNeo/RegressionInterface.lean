import SuperNeo.Regression

/-!
Contract interface for `SuperNeo.Regression`.

Spec: `./formal/superneo-lean/specs/Regression.spec.md`
Paper: `./formal/superneo-lean/SuperNeo.pdf.md`
  - Appendix B.2 (Goldilocks parameters), lines 709–727: golden vectors validate Goldilocks parameters.
-/

namespace SuperNeo

namespace RegressionInterface

/-- Canonical implementation module name for this interface. -/
def implementationModule : String := "SuperNeo.Regression"

/-- Canonical paper source used for this module-level interface/spec pair. -/
def paperSource : String := "./formal/superneo-lean/SuperNeo.pdf.md"

/-- Paper sections used to ground this regression entrypoint. -/
def paperAnchors : List String := ["Appendix B.2 Goldilocks parameters"]

/-- Modules imported by the regression entrypoint. -/
def exportedModuleNames : List String := ["SuperNeo.Golden.GoldilocksGolden"]

/-- [Role: Definitional] Entry-point contract: compiling `SuperNeo.Regression` checks the golden-vector import surface. -/
def entrypointContract : Prop := True

theorem entrypointContract_true : entrypointContract := by
  trivial

end RegressionInterface

end SuperNeo
