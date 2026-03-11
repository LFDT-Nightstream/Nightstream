import SuperNeo.ProofSystem.Folding

/-!
Interface for `SuperNeo.ProofSystem.Folding`.

Spec: `specs/ProofSystem/Folding.spec.md`
Paper: `./formal/superneo-lean/SuperNeo.pdf.md`
- Section 7.2–7.5 (Folding reductions), lines 468–593
- Lemma 3 (Π_CCS), Lemma 4 (Π_RLC), Theorem 7 (Π_DEC)

Barrel re-export of PiCCS, PiRLC, PiDEC. This interface file is the typed boundary companion.
-/

namespace SuperNeo

namespace ProofSystem.FoldingInterface

/-- Canonical implementation module name for this interface. -/
def implementationModule : String := "SuperNeo.ProofSystem.Folding"

/-- Canonical paper source used for this module-level interface/spec pair. -/
def paperSource : String := "./formal/superneo-lean/SuperNeo.pdf.md"

/-- Paper sections used to ground this module boundary. -/
def paperAnchors : List String := ["§6 interactive reductions", "§7.2-§7.5 folding reductions"]

/-- Public symbol inventory extracted from the implementation module. -/
def exportedSymbolNames : List String :=
  [ "PiCCSAssumptions"
  , "StrongStatement"
  , "soundness_relations_of_section71_ce"
  , "soundness_relations_of_section71Provider"
  , "soundness_relations_of_section71Specialization"
  , "soundness_relations_of_section71Setup"
  , "soundness_relations_of_section71TheoremInstance"
  , "soundness_relations_of_section71Context"
  , "soundness_relations_of_section71Data"
  , "soundness_relations_of_protocolTargetData"
  , "soundness_relations_of_ccsRelation"
  , "soundness_relations"
  , "PiRLCAssumptions"
  , "WeakStatement"
  , "weak_relaxed_of_section71_ce"
  , "weak_relaxed_of_section71Provider"
  , "weak_relaxed_of_section71Specialization"
  , "weak_relaxed_of_section71Setup"
  , "weak_relaxed_of_section71TheoremInstance"
  , "weak_relaxed_of_section71Context"
  , "weak_relaxed_of_section71Data"
  , "weak_relaxed_of_protocolTargetData"
  , "weak_relaxed_of_ccsRelation"
  , "weak_relaxed"
  , "PiDECAssumptions"
  , "FinalStatement"
  , "final_of_section71_ce"
  , "final_of_section71Provider"
  , "final_of_section71Specialization"
  , "final_of_section71Setup"
  , "final_of_section71TheoremInstance"
  , "final_of_section71Context"
  , "final_of_section71Data"
  , "final_of_protocolTargetData"
  , "final_of_ccsRelation"
  , "final_of_assumption"
  ]

/-- Assumption/boundary-oriented symbols extracted by naming convention. -/
def boundarySymbolNames : List String :=
  [ "PiCCSAssumptions"
  , "soundness_relations"
  , "PiRLCAssumptions"
  , "weak_relaxed"
  , "PiDECAssumptions"
  , "final_of_assumption"
  ]

end ProofSystem.FoldingInterface

end SuperNeo
