import SuperNeo.ProofSystem.Folding.PiRLC

/-!
Interface for `SuperNeo.ProofSystem.Folding.PiRLC`.

Spec: `specs/ProofSystem/Folding/PiRLC.spec.md`
Paper: `./formal/superneo-lean/SuperNeo.pdf.md`
- Section 7.4 (Π_RLC), lines 549–583
- Lemma 4 (Π_RLC is weak), line 581

This interface file is the typed boundary companion for the implementation.
-/

namespace SuperNeo

namespace ProofSystem.Folding.PiRLCInterface

/-- Canonical implementation module name for this interface. -/
def implementationModule : String := "SuperNeo.ProofSystem.Folding.PiRLC"

/-- Canonical paper source used for this module-level interface/spec pair. -/
def paperSource : String := "./formal/superneo-lean/SuperNeo.pdf.md"

/-- Paper sections used to ground this module boundary. -/
def paperAnchors : List String := ["§6 interactive reductions", "§7.2-§7.5 folding reductions"]

/-- Public symbol inventory extracted from the implementation module. -/
def exportedSymbolNames : List String :=
  [ "PiRLCAssumptions"
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
  ]

/-- Assumption/boundary-oriented symbols extracted by naming convention. -/
def boundarySymbolNames : List String := ["PiRLCAssumptions"]

/-- Proof-system wrapper assumptions for `Π_RLC`. -/
abbrev PiRLCAssumptions := SuperNeo.ProofSystem.Folding.PiRLCAssumptions

/-- Proof-system wrapper weak statement for `Π_RLC`. -/
abbrev WeakStatement := SuperNeo.ProofSystem.Folding.WeakStatement

/-- Weak `Π_RLC` from an explicit Section 7.1 proof-system CE realization. -/
theorem weak_relaxed_of_section71_ce
  {ctx : SuperNeo.ProtocolTargetContext}
  (hReal : SuperNeo.ProtocolSection71Realization ctx) :
  SuperNeo.ProofSystem.ConstraintSystem.CE.Holds
      hReal.ce hReal.ceStatement hReal.ceWitness →
    WeakStatement ctx :=
  SuperNeo.ProofSystem.Folding.weak_relaxed_of_section71_ce hReal

/-- Weak `Π_RLC` from one concrete Section 7.1 theorem instance. -/
theorem weak_relaxed_of_section71Provider
  {ctx : SuperNeo.ProtocolTargetContext} :
  SuperNeo.ProtocolSection71Provider ctx →
  WeakStatement ctx :=
  SuperNeo.ProofSystem.Folding.weak_relaxed_of_section71Provider

/-- Weak `Π_RLC` from one generic Section 7.1 theorem instance plus compact specialization. -/
theorem weak_relaxed_of_section71Specialization
  {ctx : SuperNeo.ProtocolTargetContext}
  {Commitment : Type}
  (hInst : SuperNeo.ProofSystem.ConstraintSystem.Section71Instance Commitment) :
  SuperNeo.ProtocolSection71Specialization ctx hInst →
  WeakStatement ctx :=
  SuperNeo.ProofSystem.Folding.weak_relaxed_of_section71Specialization hInst

/-- Weak `Π_RLC` from one theorem-native Section 7.1 setup. -/
theorem weak_relaxed_of_section71Setup
  {ctx : SuperNeo.ProtocolTargetContext} :
  SuperNeo.ProtocolSection71Setup ctx →
  WeakStatement ctx :=
  SuperNeo.ProofSystem.Folding.weak_relaxed_of_section71Setup

/-- Weak `Π_RLC` from one paper-faithful Section 7.1 theorem instance. -/
theorem weak_relaxed_of_section71TheoremInstance
  {ctx : SuperNeo.ProtocolTargetContext} :
  SuperNeo.ProtocolSection71TheoremInstance ctx →
  WeakStatement ctx :=
  SuperNeo.ProofSystem.Folding.weak_relaxed_of_section71TheoremInstance

/-- Weak `Π_RLC` from one theorem-native Section 7.1 context object. -/
theorem weak_relaxed_of_section71Context :
  (hCtx : SuperNeo.ProtocolSection71Context) →
  WeakStatement hCtx.target :=
  SuperNeo.ProofSystem.Folding.weak_relaxed_of_section71Context

/-- Weak `Π_RLC` from one protocol-side Section 7.1 Definition-14 data package. -/
theorem weak_relaxed_of_section71Data
  {ctx : SuperNeo.ProtocolTargetContext} :
  SuperNeo.ProtocolSection71Data ctx →
  WeakStatement ctx :=
  SuperNeo.ProofSystem.Folding.weak_relaxed_of_section71Data

/-- Weak `Π_RLC` from one protocol-side Section 7.5 target-data owner and a SumCheck witness. -/
theorem weak_relaxed_of_protocolTargetData
  {ctx : SuperNeo.ProtocolTargetContext} :
  SuperNeo.ProtocolTargetData ctx →
  SuperNeo.SumCheckTransitionWitness ctx →
  WeakStatement ctx :=
  SuperNeo.ProofSystem.Folding.weak_relaxed_of_protocolTargetData

/-- Weak `Π_RLC` from compact CCS relation and a SumCheck witness. -/
theorem weak_relaxed_of_ccsRelation
  {ctx : SuperNeo.ProtocolTargetContext} :
  SuperNeo.ccsRelation ctx →
  SuperNeo.SumCheckTransitionWitness ctx →
  WeakStatement ctx :=
  SuperNeo.ProofSystem.Folding.weak_relaxed_of_ccsRelation

/-- Weak `Π_RLC` from the compatibility assumption bundle. -/
theorem weak_relaxed
  {ctx : SuperNeo.ProtocolTargetContext} :
  PiRLCAssumptions ctx →
  SuperNeo.SumCheckTransitionWitness ctx →
  WeakStatement ctx :=
  SuperNeo.ProofSystem.Folding.weak_relaxed

end ProofSystem.Folding.PiRLCInterface

end SuperNeo
