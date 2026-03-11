import SuperNeo.ProofSystem.Folding.PiCCS

/-!
Interface for `SuperNeo.ProofSystem.Folding.PiCCS`.

Spec: `specs/ProofSystem/Folding/PiCCS.spec.md`
Paper: `./formal/superneo-lean/SuperNeo.pdf.md`
- Section 7.3 (Π_CCS), lines 481–548
- Lemma 3 (Π_CCS is strong), line 545

This interface file is the typed boundary companion for the implementation.
-/

namespace SuperNeo

namespace ProofSystem.Folding.PiCCSInterface

/-- Canonical implementation module name for this interface. -/
def implementationModule : String := "SuperNeo.ProofSystem.Folding.PiCCS"

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
  ]

/-- Assumption/boundary-oriented symbols extracted by naming convention. -/
def boundarySymbolNames : List String := ["PiCCSAssumptions", "soundness_relations"]

/-- Proof-system wrapper assumptions for `Π_CCS`. -/
abbrev PiCCSAssumptions := SuperNeo.ProofSystem.Folding.PiCCSAssumptions

/-- Proof-system wrapper strong statement for `Π_CCS`. -/
abbrev StrongStatement := SuperNeo.ProofSystem.Folding.StrongStatement

/--
Strong `Π_CCS` from an explicit Section 7.1 proof-system CE realization.
-/
theorem soundness_relations_of_section71_ce
  {ctx : SuperNeo.ProtocolTargetContext}
  (hReal : SuperNeo.ProtocolSection71Realization ctx) :
  SuperNeo.ProofSystem.ConstraintSystem.CE.Holds
      hReal.ce hReal.ceStatement hReal.ceWitness →
    StrongStatement ctx :=
  SuperNeo.ProofSystem.Folding.soundness_relations_of_section71_ce hReal

/-- Strong `Π_CCS` from one concrete Section 7.1 theorem instance. -/
theorem soundness_relations_of_section71Provider
  {ctx : SuperNeo.ProtocolTargetContext} :
  SuperNeo.ProtocolSection71Provider ctx →
  StrongStatement ctx :=
  SuperNeo.ProofSystem.Folding.soundness_relations_of_section71Provider

/-- Strong `Π_CCS` from one generic Section 7.1 theorem instance plus compact specialization. -/
theorem soundness_relations_of_section71Specialization
  {ctx : SuperNeo.ProtocolTargetContext}
  {Commitment : Type}
  (hInst : SuperNeo.ProofSystem.ConstraintSystem.Section71Instance Commitment) :
  SuperNeo.ProtocolSection71Specialization ctx hInst →
  StrongStatement ctx :=
  SuperNeo.ProofSystem.Folding.soundness_relations_of_section71Specialization hInst

/-- Strong `Π_CCS` from one theorem-native Section 7.1 setup. -/
theorem soundness_relations_of_section71Setup
  {ctx : SuperNeo.ProtocolTargetContext} :
  SuperNeo.ProtocolSection71Setup ctx →
  StrongStatement ctx :=
  SuperNeo.ProofSystem.Folding.soundness_relations_of_section71Setup

/-- Strong `Π_CCS` from one paper-faithful Section 7.1 theorem instance. -/
theorem soundness_relations_of_section71TheoremInstance
  {ctx : SuperNeo.ProtocolTargetContext} :
  SuperNeo.ProtocolSection71TheoremInstance ctx →
  StrongStatement ctx :=
  SuperNeo.ProofSystem.Folding.soundness_relations_of_section71TheoremInstance

/-- Strong `Π_CCS` from one theorem-native Section 7.1 context object. -/
theorem soundness_relations_of_section71Context :
  (hCtx : SuperNeo.ProtocolSection71Context) →
  StrongStatement hCtx.target :=
  SuperNeo.ProofSystem.Folding.soundness_relations_of_section71Context

/-- Strong `Π_CCS` from one protocol-side Section 7.1 Definition-14 data package. -/
theorem soundness_relations_of_section71Data
  {ctx : SuperNeo.ProtocolTargetContext} :
  SuperNeo.ProtocolSection71Data ctx →
  StrongStatement ctx :=
  SuperNeo.ProofSystem.Folding.soundness_relations_of_section71Data

/-- Strong `Π_CCS` from one protocol-side Section 7.5 target-data owner and a SumCheck witness. -/
theorem soundness_relations_of_protocolTargetData
  {ctx : SuperNeo.ProtocolTargetContext} :
  SuperNeo.ProtocolTargetData ctx →
  SuperNeo.SumCheckTransitionWitness ctx →
  StrongStatement ctx :=
  SuperNeo.ProofSystem.Folding.soundness_relations_of_protocolTargetData

/-- Strong `Π_CCS` from compact CCS relation and a SumCheck witness. -/
theorem soundness_relations_of_ccsRelation
  {ctx : SuperNeo.ProtocolTargetContext} :
  SuperNeo.ccsRelation ctx →
  SuperNeo.SumCheckTransitionWitness ctx →
  StrongStatement ctx :=
  SuperNeo.ProofSystem.Folding.soundness_relations_of_ccsRelation

/-- Strong `Π_CCS` from the compatibility assumption bundle. -/
theorem soundness_relations
  {ctx : SuperNeo.ProtocolTargetContext} :
  PiCCSAssumptions ctx →
  SuperNeo.SumCheckTransitionWitness ctx →
  StrongStatement ctx :=
  SuperNeo.ProofSystem.Folding.soundness_relations

end ProofSystem.Folding.PiCCSInterface

end SuperNeo
