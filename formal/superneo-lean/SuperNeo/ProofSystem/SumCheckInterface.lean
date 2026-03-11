import SuperNeo.ProofSystem.SumCheck

/-!
Interface for `SuperNeo.ProofSystem.SumCheck`.

Spec: `specs/ProofSystem/SumCheck.spec.md`
Paper: `./formal/superneo-lean/SuperNeo.pdf.md`
- Definition 6 (The sum-check protocol), lines 352–355
- Section 7.3 (Π_CCS), lines 481–548; Section 7.4 (Π_RLC), lines 549–583

Barrel re-export of General plus the prefix-dependent aligned positive-round endpoint.
This interface file is the typed boundary companion.
-/

namespace SuperNeo

namespace ProofSystem.SumCheckInterface

/-- Canonical implementation module name for this interface. -/
def implementationModule : String := "SuperNeo.ProofSystem.SumCheck"

/-- Canonical paper source used for this module-level interface/spec pair. -/
def paperSource : String := "./formal/superneo-lean/SuperNeo.pdf.md"

/-- Paper sections used to ground this module boundary. -/
def paperAnchors : List String := ["§2.1 Sum-check reduction role", "§7.3 Interactive reduction for CCS"]

/-- Public symbol inventory extracted from the implementation module. -/
def exportedSymbolNames : List String :=
  [ "lundSoundnessAssumptionFullFieldAlignedPosRounds_prefix"
  , "lundSoundnessAssumptionFullFieldAlignedPosRoundsPosDegree_prefix" ]

/-- Assumption/boundary-oriented symbols extracted by naming convention. -/
def boundarySymbolNames : List String := []

/-- Interface inventory marker for the typed surface exposed here. -/
def interfaceScaffoldReady : Prop := True

theorem interfaceScaffoldReady_true : interfaceScaffoldReady := by
  trivial

end ProofSystem.SumCheckInterface

end SuperNeo
