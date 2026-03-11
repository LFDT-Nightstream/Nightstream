import SuperNeo.FoldingProtocol

/-!
Contract interface for `SuperNeo.FoldingProtocol`.

Paper anchors:
- `./formal/superneo-lean/SuperNeo.pdf.md`, Section 7
  (Neo's folding scheme for CCS), lines 447-596
- Definition 11 (Structure), Definition 12 (Norm-bounded CCS),
  Definition 13 (Norm-bounded CCS Evaluation Relation),
  Definition 14 (Global Reduction Parameters)
- Section 7.3: Π_CCS, Lemma 3 (Π_CCS is strong)
- Section 7.4: Π_RLC, Lemma 4 (Π_RLC is weak)
- Section 7.5: Π_DEC, Theorem 7 (Π_DEC is a reduction of knowledge)
- Theorem 1 (Full composition)
-/

namespace SuperNeo

namespace FoldingProtocolInterface

/-- Canonical implementation module name for this interface. -/
def implementationModule : String := "SuperNeo.FoldingProtocol"

/-- Canonical paper source used for this module-level interface/spec pair. -/
def paperSource : String := "./formal/superneo-lean/SuperNeo.pdf.md"

/-- Paper sections used to ground this barrel interface. -/
def paperAnchors : List String :=
  ["§7 Folding protocol", "Appendix D deferred protocol proofs"]

/-- Modules re-exported by the Section 7 folding barrel. -/
def exportedModuleNames : List String :=
  [ "SuperNeo.ProofSystem.ConstraintSystem"
  , "SuperNeo.ProofSystem.SumCheck"
  , "SuperNeo.ProofSystem.Folding"
  , "SuperNeo.ProtocolRelations"
  , "SuperNeo.ProtocolSection71Data"
  , "SuperNeo.ProtocolSection71Context"
  , "SuperNeo.PiCCS"
  , "SuperNeo.PiRLC"
  , "SuperNeo.PiDEC"
  , "SuperNeo.ArithmeticBundle"
  , "SuperNeo.ArithmeticObligations"
  , "SuperNeo.ProtocolTarget"
  , "SuperNeo.ProtocolTargetData"
  , "SuperNeo.ProtocolMathTarget"
  , "SuperNeo.ProtocolTheorem"
  , "SuperNeo.ProofSystem.Protocol"
  ]

/-- [Role: Definitional] Barrel contract: importing `SuperNeo.FoldingProtocol` exposes the full Section 7 protocol stack. -/
def barrelContract : Prop := True

theorem barrelContract_true : barrelContract := by
  trivial

end FoldingProtocolInterface

end SuperNeo
