import SuperNeo.ProofSystem.Security

/-!
Interface for `SuperNeo.ProofSystem.Security`.

Spec: `specs/ProofSystem/Security.spec.md`
Paper: `./formal/superneo-lean/SuperNeo.pdf.md`
Anchors: Definition 9 (Weak Interactive Reductions), lines 404–416; Definition 10 (Strong Interactive Reductions), lines 418–436.
Security model structures underlying probability and error accounting in Section 6.
-/

namespace SuperNeo

namespace ProofSystem.SecurityInterface

/-- Canonical implementation module name for this interface. -/
def implementationModule : String := "SuperNeo.ProofSystem.Security"

/-- Canonical paper source used for this module-level interface/spec pair. -/
def paperSource : String := "./formal/superneo-lean/SuperNeo.pdf.md"

/-- Paper sections used to ground this module boundary. -/
def paperAnchors : List String := ["§6 interactive reductions (security framing)", "§7 theorem-facing assumptions"]

/-- Public symbol inventory extracted from the implementation module. -/
def exportedSymbolNames : List String := ["ProbModel", "ErrorModel", "zeroErrorModel"]

/-- Assumption/boundary-oriented symbols extracted by naming convention. -/
def boundarySymbolNames : List String := []

/-- Interface inventory marker for the typed surface exposed here. -/
def interfaceScaffoldReady : Prop := True

theorem interfaceScaffoldReady_true : interfaceScaffoldReady := by
  trivial

end ProofSystem.SecurityInterface

end SuperNeo
