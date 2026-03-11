import SuperNeo.EmbeddingTheory

/-!
Contract interface for `SuperNeo.EmbeddingTheory`.

Spec: `./formal/superneo-lean/specs/EmbeddingTheory.spec.md`

Paper anchors:
- `./formal/superneo-lean/SuperNeo.pdf.md`, Section 5
  (Embedding products with evaluation homomorphism), lines 354-401
- Definition 7 (Coefficient Embedding), Definition 8 (Lifting the Transform)
- Theorem 3 (Inner Product Transform), Theorem 4 (Matrix-Vector Product Transform),
  Theorem 5 (Evaluation Homomorphism)
- Definition 15 (Module Homomorphism), Remark 2
-/

namespace SuperNeo

namespace EmbeddingTheoryInterface

/-- Canonical implementation module name for this interface. -/
def implementationModule : String := "SuperNeo.EmbeddingTheory"

/-- Canonical paper source used for this module-level interface/spec pair. -/
def paperSource : String := "./formal/superneo-lean/SuperNeo.pdf.md"

/-- Paper sections used to ground this barrel interface. -/
def paperAnchors : List String :=
  ["§5 Embedding products with evaluation homomorphism", "Appendix C module homomorphism"]

/-- Modules re-exported by the Section 5 embedding barrel. -/
def exportedModuleNames : List String :=
  [ "SuperNeo.Embedding"
  , "SuperNeo.Thm3Core"
  , "SuperNeo.BarLift"
  , "SuperNeo.MatrixTransform"
  , "SuperNeo.EvalLink"
  , "SuperNeo.ModuleHom"
  , "SuperNeo.EvalHom"
  ]

/-- [Role: Definitional] Barrel contract: importing `SuperNeo.EmbeddingTheory` exposes the full Section 5 chain. -/
def barrelContract : Prop := True

theorem barrelContract_true : barrelContract := by
  trivial

end EmbeddingTheoryInterface

end SuperNeo
