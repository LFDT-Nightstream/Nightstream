import SuperNeo.ProofSystem.ConstraintSystem

/-!
Interface for `SuperNeo.ProofSystem.ConstraintSystem`.

Spec: `specs/ProofSystem/ConstraintSystem.spec.md`
Paper: `./formal/superneo-lean/SuperNeo.pdf.md`
- Definition 11 (Structure), lines 449-455
- Definition 12 (Norm-bounded CCS), lines 457-459
- Definition 13 (Norm-bounded CCS Evaluation Relation), lines 461-465
- Definition 14 (Global Reduction Parameters), lines 467-475

Barrel re-export of the Section 7.1 proof-system relation layer.
-/

namespace SuperNeo

namespace ProofSystem.ConstraintSystemInterface

/-- Canonical implementation module name for this interface. -/
def implementationModule : String := "SuperNeo.ProofSystem.ConstraintSystem"

/-- Canonical paper source used for this module-level interface/spec pair. -/
def paperSource : String := "./formal/superneo-lean/SuperNeo.pdf.md"

/-- Paper sections used to ground this module boundary. -/
def paperAnchors : List String :=
  ["§7.1 Structure / CCS / CE", "§7.2 Global Reduction Parameters"]

/-- Public symbol inventory extracted from the implementation module. -/
def exportedSymbolNames : List String :=
  [ "CommitmentMap"
  , "InputProjector"
  , "CCSStructure"
  , "GlobalParams"
  , "Section71Objects"
  , "Section71Instance"
  , "CCS"
  , "CE"
  , "CERelaxed"
  ]

/-- Assumption/boundary-oriented symbols extracted by naming convention. -/
def boundarySymbolNames : List String := []

/-- [Role: Definitional] Barrel re-export of commitment maps. -/
abbrev CommitmentMap := SuperNeo.ProofSystem.ConstraintSystem.CommitmentMap

/-- [Role: Definitional] Barrel re-export of input projectors. -/
abbrev InputProjector := SuperNeo.ProofSystem.ConstraintSystem.InputProjector

/-- [Role: Definitional] Barrel re-export of Definition-11 structures. -/
abbrev CCSStructure := SuperNeo.ProofSystem.ConstraintSystem.CCSStructure

/-- [Role: Definitional] Barrel re-export of Definition-14 global parameters. -/
abbrev GlobalParams := SuperNeo.ProofSystem.ConstraintSystem.GlobalParams

/-- [Role: Boundary] Barrel re-export of coherent Section 7.1 tuple packages. -/
abbrev Section71Objects := SuperNeo.ProofSystem.ConstraintSystem.Section71Objects

/-- [Role: Boundary] Barrel re-export of concrete Section 7.1 theorem instances. -/
abbrev Section71Instance := SuperNeo.ProofSystem.ConstraintSystem.Section71Instance

/-- [Role: Definitional] Barrel re-export of Definition-12 CCS relations. -/
abbrev CCS := SuperNeo.ProofSystem.ConstraintSystem.CCS

/-- [Role: Definitional] Barrel re-export of Definition-13 CE relations. -/
abbrev CE := SuperNeo.ProofSystem.ConstraintSystem.CE

/-- [Role: Definitional] Barrel re-export of relaxed CE carriers. -/
abbrev CERelaxed := SuperNeo.ProofSystem.ConstraintSystem.CERelaxed

end ProofSystem.ConstraintSystemInterface

end SuperNeo
