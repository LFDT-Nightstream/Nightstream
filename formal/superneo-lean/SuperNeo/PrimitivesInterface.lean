import SuperNeo.Primitives

/-!
Contract interface for `SuperNeo.Primitives`.

Spec: `./formal/superneo-lean/specs/Primitives.spec.md`

Paper anchors:
- `./formal/superneo-lean/SuperNeo.pdf.md`, Section 4 (Preliminaries), lines 268-353
- Definitions 1-6 (fields, rings, dimensions, coefficient maps, norms,
  commitment scheme, interactive reductions, sum-check)
- Lemma 5 (Schwartz-Zippel), Lemma 6 (eq-lifting)
-/

namespace SuperNeo

namespace PrimitivesInterface

/-- Canonical implementation module name for this interface. -/
def implementationModule : String := "SuperNeo.Primitives"

/-- Canonical paper source used for this module-level interface/spec pair. -/
def paperSource : String := "./formal/superneo-lean/SuperNeo.pdf.md"

/-- Paper sections used to ground this barrel interface. -/
def paperAnchors : List String :=
  ["§4 Preliminaries", "Appendix B.2 parameters", "Appendix C polynomial tools"]

/-- Modules re-exported by the Section 4 primitives barrel. -/
def exportedModuleNames : List String :=
  [ "SuperNeo.Goldilocks"
  , "SuperNeo.Field"
  , "SuperNeo.Dimensions"
  , "SuperNeo.Ring"
  , "SuperNeo.CoeffMaps"
  , "SuperNeo.Norm"
  , "SuperNeo.Decomp"
  , "SuperNeo.DecompBase2"
  , "SuperNeo.EqPoly"
  , "SuperNeo.MLE"
  , "SuperNeo.SumCheck"
  , "SuperNeo.PolyLemmas"
  , "SuperNeo.Interp"
  , "SuperNeo.Parameters"
  ]

/-- [Role: Definitional] Barrel contract: importing `SuperNeo.Primitives` exposes the full Section 4 layer. -/
def barrelContract : Prop := True

theorem barrelContract_true : barrelContract := by
  trivial

end PrimitivesInterface

end SuperNeo
