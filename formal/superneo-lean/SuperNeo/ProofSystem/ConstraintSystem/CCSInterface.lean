import SuperNeo.ProofSystem.ConstraintSystem.CCS

/-!
Interface for `SuperNeo.ProofSystem.ConstraintSystem.CCS`.

Spec: `specs/ProofSystem/ConstraintSystem/CCS.spec.md`
Paper: `./formal/superneo-lean/SuperNeo.pdf.md`
- Definition 11 (Structure), lines 449-455
- Definition 12 (Norm-bounded CCS), lines 457-459
- Definition 13 (Norm-bounded CCS Evaluation Relation), lines 461-465
- Definition 14 (Global Reduction Parameters), lines 467-475

This interface file is the typed boundary companion for the implementation.
-/

namespace SuperNeo

namespace ProofSystem.ConstraintSystem.CCSInterface

universe u

/-- Canonical implementation module name for this interface. -/
def implementationModule : String := "SuperNeo.ProofSystem.ConstraintSystem.CCS"

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
  , "CCS.Statement"
  , "CCS.Witness"
  , "CCS.fullVector"
  , "CCS.Holds"
  , "CE"
  , "CE.Statement"
  , "CE.Witness"
  , "CE.Holds"
  , "CERelaxed"
  , "CERelaxed.Statement"
  , "CERelaxed.Witness"
  , "CERelaxed.Holds"
  , "CERelaxed.ofCE"
  , "CERelaxed.holds_of_ce"
  , "GlobalParams.ccs"
  , "GlobalParams.ce"
  , "GlobalParams.ceRelaxed"
  ]

/-- Assumption/boundary-oriented symbols extracted by naming convention. -/
def boundarySymbolNames : List String := []

/-- [Role: Definitional] Curated re-export of commitment maps. -/
abbrev CommitmentMap := SuperNeo.ProofSystem.ConstraintSystem.CommitmentMap

/-- [Role: Definitional] Curated re-export of input projectors. -/
abbrev InputProjector := SuperNeo.ProofSystem.ConstraintSystem.InputProjector

/-- [Role: Definitional] Curated re-export of Definition-11 structures. -/
abbrev CCSStructure := SuperNeo.ProofSystem.ConstraintSystem.CCSStructure

/-- [Role: Definitional] Curated re-export of Definition-14 global parameters. -/
abbrev GlobalParams := SuperNeo.ProofSystem.ConstraintSystem.GlobalParams

/-- [Role: Boundary] One coherent proof-system Section 7.1 tuple package. -/
abbrev Section71Objects := SuperNeo.ProofSystem.ConstraintSystem.Section71Objects

/-- [Role: Boundary] One concrete proof-system Section 7.1 theorem instance. -/
abbrev Section71Instance := SuperNeo.ProofSystem.ConstraintSystem.Section71Instance

/-- [Role: Definitional] Curated re-export of Definition-12 CCS relations. -/
abbrev CCS := SuperNeo.ProofSystem.ConstraintSystem.CCS

/-- [Role: Definitional] Curated re-export of CCS statements. -/
abbrev CCSStatement := SuperNeo.ProofSystem.ConstraintSystem.CCS.Statement

/-- [Role: Definitional] Curated re-export of CCS witnesses. -/
abbrev CCSWitness := SuperNeo.ProofSystem.ConstraintSystem.CCS.Witness

/-- [Role: Theorem-Target] Curated re-export of the CCS membership predicate. -/
def CCSHolds
  {Commitment : Type u}
  (ccs : CCS Commitment)
  (stmt : CCSStatement Commitment)
  (wit : CCSWitness) : Prop :=
  SuperNeo.ProofSystem.ConstraintSystem.CCS.Holds ccs stmt wit

/-- [Role: Theorem-Target] Curated re-export of `z := [x, w]`. -/
def CCS_fullVector
  {Commitment : Type u}
  (stmt : CCSStatement Commitment)
  (wit : CCSWitness) : Coeffs :=
  SuperNeo.ProofSystem.ConstraintSystem.CCS.fullVector stmt wit

/-- [Role: Definitional] Curated re-export of Definition-13 CE relations. -/
abbrev CE := SuperNeo.ProofSystem.ConstraintSystem.CE

/-- [Role: Definitional] Curated re-export of CE statements. -/
abbrev CEStatement := SuperNeo.ProofSystem.ConstraintSystem.CE.Statement

/-- [Role: Definitional] Curated re-export of CE witnesses. -/
abbrev CEWitness := SuperNeo.ProofSystem.ConstraintSystem.CE.Witness

/-- [Role: Theorem-Target] Curated re-export of the CE membership predicate. -/
def CEHolds
  {Commitment : Type u}
  (ce : CE Commitment)
  (stmt : CEStatement Commitment)
  (wit : CEWitness) : Prop :=
  SuperNeo.ProofSystem.ConstraintSystem.CE.Holds ce stmt wit

/-- [Role: Definitional] Curated re-export of relaxed CE carriers. -/
abbrev CERelaxed := SuperNeo.ProofSystem.ConstraintSystem.CERelaxed

/-- [Role: Definitional] Curated re-export of relaxed CE statements. -/
abbrev CERelaxedStatement := SuperNeo.ProofSystem.ConstraintSystem.CERelaxed.Statement

/-- [Role: Definitional] Curated re-export of relaxed CE witnesses. -/
abbrev CERelaxedWitness := SuperNeo.ProofSystem.ConstraintSystem.CERelaxed.Witness

/-- [Role: Theorem-Target] Curated re-export of the relaxed CE membership predicate. -/
def CERelaxedHolds
  {Commitment : Type u}
  (ce : CERelaxed Commitment)
  (stmt : CERelaxedStatement Commitment)
  (wit : CERelaxedWitness) : Prop :=
  SuperNeo.ProofSystem.ConstraintSystem.CERelaxed.Holds ce stmt wit

/-- [Role: Theorem-Target] Curated re-export of the CE → relaxed CE carrier promotion. -/
def CERelaxed_ofCE
  {Commitment : Type u}
  (ce : CE Commitment) : CERelaxed Commitment :=
  SuperNeo.ProofSystem.ConstraintSystem.CERelaxed.ofCE ce

/-- [Role: Theorem-Target] Curated theorem surface `CERelaxed.holds_of_ce`. -/
abbrev CERelaxed_holds_of_ce
  {Commitment : Type u}
  {ce : CE Commitment}
  {stmt : CEStatement Commitment}
  {wit : CEWitness} :
  CEHolds ce stmt wit →
    CERelaxedHolds (CERelaxed_ofCE ce) stmt ⟨wit.assignment, #[]⟩ :=
  SuperNeo.ProofSystem.ConstraintSystem.CERelaxed.holds_of_ce

/-- [Role: Theorem-Target] Curated re-export of the Definition-14 CCS constructor. -/
def GlobalParams_ccs
  {Commitment : Type u}
  (params : GlobalParams Commitment)
  (b : Nat) : CCS Commitment :=
  SuperNeo.ProofSystem.ConstraintSystem.GlobalParams.ccs params b

/-- [Role: Theorem-Target] Curated re-export of the Definition-14 CE constructor. -/
def GlobalParams_ce
  {Commitment : Type u}
  (params : GlobalParams Commitment)
  (b : Nat) : CE Commitment :=
  SuperNeo.ProofSystem.ConstraintSystem.GlobalParams.ce params b

/-- [Role: Theorem-Target] Curated re-export of the Definition-14 relaxed-CE constructor. -/
def GlobalParams_ceRelaxed
  {Commitment : Type u}
  (params : GlobalParams Commitment)
  (b B : Nat) : CERelaxed Commitment :=
  SuperNeo.ProofSystem.ConstraintSystem.GlobalParams.ceRelaxed params b B

/-- [Role: Theorem-Target] Recover that one Section 7.1 object package shares a commitment across CCS/CE. -/
theorem Section71Objects_sharedCommitment_eq
  {Commitment : Type u}
  (h : Section71Objects Commitment) :
  h.ccsStatement.commitment = h.ceStatement.commitment :=
  SuperNeo.ProofSystem.ConstraintSystem.Section71Objects.sharedCommitment_eq h

/-- [Role: Theorem-Target] Recover that one Section 7.1 object package shares a public input across CCS/CE. -/
theorem Section71Objects_sharedPublicInput_eq
  {Commitment : Type u}
  (h : Section71Objects Commitment) :
  h.ccsStatement.publicInput = h.ceStatement.publicInput :=
  SuperNeo.ProofSystem.ConstraintSystem.Section71Objects.sharedPublicInput_eq h

/-- [Role: Theorem-Target] Recover that the CE witness assignment is the CCS full vector `[x, w]`. -/
theorem Section71Objects_ceAssignment_eq_fullVector
  {Commitment : Type u}
  (h : Section71Objects Commitment) :
  h.ceWitness.assignment = CCS_fullVector h.ccsStatement h.ccsWitness :=
  SuperNeo.ProofSystem.ConstraintSystem.Section71Objects.ceAssignment_eq_fullVector h

end ProofSystem.ConstraintSystem.CCSInterface

end SuperNeo
