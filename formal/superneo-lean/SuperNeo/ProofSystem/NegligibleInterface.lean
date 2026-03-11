import SuperNeo.ProofSystem.Negligible

/-!
Interface for `SuperNeo.ProofSystem.Negligible`.

Spec: `specs/ProofSystem/Negligible.spec.md`
Paper: `./formal/superneo-lean/SuperNeo.pdf.md`
Anchors: Definition 9 (Weak Interactive Reductions), lines 404–416 (negligible error bound ε); Definition 10, lines 418–436.
Negligible function concept used throughout Section 6 security reductions.
-/

namespace SuperNeo

namespace ProofSystem.NegligibleInterface

/-- Canonical implementation module name for this interface. -/
def implementationModule : String := "SuperNeo.ProofSystem.Negligible"

/-- Canonical paper source used for this module-level interface/spec pair. -/
def paperSource : String := "./formal/superneo-lean/SuperNeo.pdf.md"

/-- Paper sections used to ground this module boundary. -/
def paperAnchors : List String := ["§6 interactive reductions (security framing)", "§7 theorem-facing assumptions"]

/-- Public symbol inventory extracted from the implementation module. -/
def exportedSymbolNames : List String := ["ErrorFn", "IsNegligible", "isNegligible_zero", "isNegligible_of_zero"]

/-- Assumption/boundary-oriented symbols extracted by naming convention. -/
def boundarySymbolNames : List String := ["IsNegligible", "isNegligible_zero", "isNegligible_of_zero"]

/-- Interface inventory marker for the typed surface exposed here. -/
def interfaceScaffoldReady : Prop := True

theorem interfaceScaffoldReady_true : interfaceScaffoldReady := by
  trivial

end ProofSystem.NegligibleInterface

end SuperNeo
