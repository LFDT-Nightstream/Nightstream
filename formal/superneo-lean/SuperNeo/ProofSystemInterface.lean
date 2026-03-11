import SuperNeo.ProofSystem

/-!
Contract interface for `SuperNeo.ProofSystem`.

Spec: `specs/ProofSystem.spec.md`

Paper anchors:
- Section 6 (Security Model), lines 404-447.
- Section 7 (Folding Protocol), lines 449-596.
-/

namespace SuperNeo

namespace ProofSystemInterface

/-- Canonical implementation module name for this interface. -/
def implementationModule : String := "SuperNeo.ProofSystem"

/-- Canonical paper source used for this module-level interface/spec pair. -/
def paperSource : String := "./formal/superneo-lean/SuperNeo.pdf.md"

/-- Paper sections used to ground this barrel interface. -/
def paperAnchors : List String := ["§6 Security model", "§7 Folding protocol"]

/-- Modules re-exported by the proof-system barrel. -/
def exportedModuleNames : List String :=
  [ "SuperNeo.ProofSystem.Types"
  , "SuperNeo.ProofSystem.Security"
  , "SuperNeo.ProofSystem.Lattice"
  , "SuperNeo.ProofSystem.ConstraintSystem"
  , "SuperNeo.ProofSystem.SumCheck"
  , "SuperNeo.ProofSystem.Folding"
  , "SuperNeo.ProofSystem.Protocol"
  ]

/-- [Role: Definitional] Barrel contract: importing `SuperNeo.ProofSystem` exposes the curated proof-system stack. -/
def barrelContract : Prop := True

theorem barrelContract_true : barrelContract := by
  trivial

end ProofSystemInterface

end SuperNeo
