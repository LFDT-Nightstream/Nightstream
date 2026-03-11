import SuperNeo.ProofSystem.Types

/-!
Interface for `SuperNeo.ProofSystem.Types`.

Spec: `specs/ProofSystem/Types.spec.md`
Paper: `./formal/superneo-lean/SuperNeo.pdf.md`
Anchors: Definition 9 (Weak Interactive Reductions), lines 404–416; Definition 10 (Strong Interactive Reductions), lines 418–436.
Infrastructure for relations over (pp, s, u, w) in Definitions 9–10.
-/

namespace SuperNeo

namespace ProofSystem.TypesInterface

/-- Canonical implementation module name for this interface. -/
def implementationModule : String := "SuperNeo.ProofSystem.Types"

/-- Canonical paper source used for this module-level interface/spec pair. -/
def paperSource : String := "./formal/superneo-lean/SuperNeo.pdf.md"

/-- Paper sections used to ground this module boundary. -/
def paperAnchors : List String := ["§6 interactive reductions (security framing)", "§7 theorem-facing assumptions"]

/-- Public symbol inventory extracted from the implementation module. -/
def exportedSymbolNames : List String := ["Context", "Claim", "Witness"]

/-- Assumption/boundary-oriented symbols extracted by naming convention. -/
def boundarySymbolNames : List String := []

/-- Interface inventory marker for the typed surface exposed here. -/
def interfaceScaffoldReady : Prop := True

theorem interfaceScaffoldReady_true : interfaceScaffoldReady := by
  trivial

end ProofSystem.TypesInterface

end SuperNeo
