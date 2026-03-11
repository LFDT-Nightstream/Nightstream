import SuperNeo.ProofSystem.Folding.PiDEC

/-!
Interface for `SuperNeo.ProofSystem.Folding.PiDEC`.

Spec: `specs/ProofSystem/Folding/PiDEC.spec.md`
Paper: `./formal/superneo-lean/SuperNeo.pdf.md`
- Section 7.5 (Π_DEC), lines 585–593
- Theorem 7 (Π_DEC is a reduction of knowledge), line 593

This interface file is the typed boundary companion for the implementation.
-/

namespace SuperNeo

namespace ProofSystem.Folding.PiDECInterface

/-- Canonical implementation module name for this interface. -/
def implementationModule : String := "SuperNeo.ProofSystem.Folding.PiDEC"

/-- Canonical paper source used for this module-level interface/spec pair. -/
def paperSource : String := "./formal/superneo-lean/SuperNeo.pdf.md"

/-- Paper sections used to ground this module boundary. -/
def paperAnchors : List String := ["§6 interactive reductions", "§7.2-§7.5 folding reductions"]

/-- Public symbol inventory extracted from the implementation module. -/
def exportedSymbolNames : List String :=
  [ "PiDECAssumptions"
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
def boundarySymbolNames : List String := ["PiDECAssumptions", "final_of_assumption"]

/-- Proof-system wrapper assumptions for `Π_DEC`. -/
abbrev PiDECAssumptions := SuperNeo.ProofSystem.Folding.PiDECAssumptions

/-- Proof-system wrapper final statement for `Π_DEC`. -/
abbrev FinalStatement := SuperNeo.ProofSystem.Folding.FinalStatement

/-- Final `Π_DEC` theorem from an explicit Section 7.1 proof-system CE realization. -/
theorem final_of_section71_ce
  {ctx : SuperNeo.ProtocolTargetContext}
  (hReal : SuperNeo.ProtocolSection71Realization ctx) :
  SuperNeo.ProofSystem.ConstraintSystem.CE.Holds
      hReal.ce hReal.ceStatement hReal.ceWitness →
    FinalStatement ctx :=
  SuperNeo.ProofSystem.Folding.final_of_section71_ce hReal

/-- Final `Π_DEC` theorem from one concrete Section 7.1 theorem instance. -/
theorem final_of_section71Provider
  {ctx : SuperNeo.ProtocolTargetContext} :
  SuperNeo.ProtocolSection71Provider ctx →
  FinalStatement ctx :=
  SuperNeo.ProofSystem.Folding.final_of_section71Provider

/-- Final `Π_DEC` theorem from one generic Section 7.1 theorem instance plus compact specialization. -/
theorem final_of_section71Specialization
  {ctx : SuperNeo.ProtocolTargetContext}
  {Commitment : Type}
  (hInst : SuperNeo.ProofSystem.ConstraintSystem.Section71Instance Commitment) :
  SuperNeo.ProtocolSection71Specialization ctx hInst →
  FinalStatement ctx :=
  SuperNeo.ProofSystem.Folding.final_of_section71Specialization hInst

/-- Final `Π_DEC` theorem from one theorem-native Section 7.1 setup. -/
theorem final_of_section71Setup
  {ctx : SuperNeo.ProtocolTargetContext} :
  SuperNeo.ProtocolSection71Setup ctx →
  FinalStatement ctx :=
  SuperNeo.ProofSystem.Folding.final_of_section71Setup

/-- Final `Π_DEC` theorem from one paper-faithful Section 7.1 theorem instance. -/
theorem final_of_section71TheoremInstance
  {ctx : SuperNeo.ProtocolTargetContext} :
  SuperNeo.ProtocolSection71TheoremInstance ctx →
  FinalStatement ctx :=
  SuperNeo.ProofSystem.Folding.final_of_section71TheoremInstance

/-- Final `Π_DEC` theorem from one theorem-native Section 7.1 context object. -/
theorem final_of_section71Context :
  (hCtx : SuperNeo.ProtocolSection71Context) →
  FinalStatement hCtx.target :=
  SuperNeo.ProofSystem.Folding.final_of_section71Context

/-- Final `Π_DEC` theorem from one protocol-side Section 7.1 Definition-14 data package. -/
theorem final_of_section71Data
  {ctx : SuperNeo.ProtocolTargetContext} :
  SuperNeo.ProtocolSection71Data ctx →
  FinalStatement ctx :=
  SuperNeo.ProofSystem.Folding.final_of_section71Data

/-- Final `Π_DEC` theorem from one protocol-side Section 7.5 target-data owner and a SumCheck witness. -/
theorem final_of_protocolTargetData
  {ctx : SuperNeo.ProtocolTargetContext} :
  SuperNeo.ProtocolTargetData ctx →
  SuperNeo.SumCheckTransitionWitness ctx →
  FinalStatement ctx :=
  SuperNeo.ProofSystem.Folding.final_of_protocolTargetData

/-- Final `Π_DEC` theorem from compact CCS relation and a SumCheck witness. -/
theorem final_of_ccsRelation
  {ctx : SuperNeo.ProtocolTargetContext} :
  SuperNeo.ccsRelation ctx →
  SuperNeo.SumCheckTransitionWitness ctx →
  FinalStatement ctx :=
  SuperNeo.ProofSystem.Folding.final_of_ccsRelation

/-- Final `Π_DEC` theorem from the compatibility assumption bundle. -/
theorem final_of_assumption
  {ctx : SuperNeo.ProtocolTargetContext} :
  PiDECAssumptions ctx →
  SuperNeo.SumCheckTransitionWitness ctx →
  FinalStatement ctx :=
  SuperNeo.ProofSystem.Folding.final_of_assumption

end ProofSystem.Folding.PiDECInterface

end SuperNeo
