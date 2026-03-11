import SuperNeo.SecurityModel

/-!
Contract interface for `SuperNeo.SecurityModel`.

Paper anchors:
- `./formal/superneo-lean/SuperNeo.pdf.md`, Section 6
  (Strong and weak interactive reductions), lines 402-446
- Definition 9 (Weak Interactive Reductions), Definition 10 (Strong Interactive Reductions)
- Theorem 6 (Strong-Weak Composition)
- Appendix C: Definition 16 (MSIS), Definition 18 (Ajtai commitment),
  Theorem 2 (Ajtai properties), Theorem 8 (Low-norm invertibility),
  Definition 17 (Strong sampling sets), Theorem 9 (Expansion factors)
-/

namespace SuperNeo

namespace SecurityModelInterface

/-- Canonical implementation module name for this interface. -/
def implementationModule : String := "SuperNeo.SecurityModel"

/-- Canonical paper source used for this module-level interface/spec pair. -/
def paperSource : String := "./formal/superneo-lean/SuperNeo.pdf.md"

/-- Paper sections used to ground this barrel interface. -/
def paperAnchors : List String :=
  ["§6 Security model", "Appendix B.2 parameters", "Appendix C security items"]

/-- Modules re-exported by the Section 6 security barrel. -/
def exportedModuleNames : List String :=
  [ "SuperNeo.InteractiveReductions"
  , "SuperNeo.ProofSystem.Types"
  , "SuperNeo.ProofSystem.Security"
  , "SuperNeo.ProofSystem.Negligible"
  , "SuperNeo.ProofSystem.Lattice"
  , "SuperNeo.ProofSystem.LatticePaper"
  , "SuperNeo.ProofSystem.LatticeReductions"
  , "SuperNeo.InvertibilityAxioms"
  , "SuperNeo.InvertibilityGoldilocks"
  , "SuperNeo.SamplingSet"
  ]

/-- [Role: Definitional] Barrel contract: importing `SuperNeo.SecurityModel` exposes the full Section 6 security layer. -/
def barrelContract : Prop := True

theorem barrelContract_true : barrelContract := by
  trivial

end SecurityModelInterface

end SuperNeo
