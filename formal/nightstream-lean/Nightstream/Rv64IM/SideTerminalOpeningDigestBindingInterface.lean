import Nightstream.Rv64IM.SideTerminalOpeningDigestBinding

/-!
Interface for the RV64IM side-terminal opening-digest theorem boundary.

Spec:
`./formal/nightstream-lean/specs/rv64im/Rv64IMSideTerminalOpeningDigestBinding.spec.md`
-/

namespace Nightstream.Rv64IM

namespace SideTerminalOpeningDigestBindingInterface

def implementationModule : String :=
  "Nightstream.Rv64IM.SideTerminalOpeningDigestBinding"

def exportedModuleNames : List String :=
  [ "Nightstream.Rv64IM.Generated.SideTerminalOpeningDigestBindingCorpus"
  , "Nightstream.Rv64IM.SideTerminalOpeningDigestBinding"
  ]

abbrev currentSideTerminalCheck :=
  Nightstream.Rv64IM.currentSideTerminalCheck
abbrev canonicalOpeningDigestBound :=
  Nightstream.Rv64IM.canonicalOpeningDigestBound
abbrev fixedSideTerminalCheck :=
  Nightstream.Rv64IM.fixedSideTerminalCheck
abbrev rustRefinesFixedSideTerminalCheck :=
  Nightstream.Rv64IM.rustRefinesFixedSideTerminalCheck
abbrev sideTerminalOpeningDigestCounterexample :=
  Nightstream.Rv64IM.sideTerminalOpeningDigestCounterexample
abbrev sideTerminalOpeningDigestBindingReport :=
  Nightstream.Rv64IM.sideTerminalOpeningDigestBindingReport
abbrev rv64imSideTerminalOpeningDigestBindingReports :=
  Nightstream.Rv64IM.rv64imSideTerminalOpeningDigestBindingReports
abbrev sideTerminalOpeningDigestBindingCounterexamples :=
  Nightstream.Rv64IM.sideTerminalOpeningDigestBindingCounterexamples
abbrev uniqueSideTerminalOpeningDigestBindingBlockers :=
  Nightstream.Rv64IM.uniqueSideTerminalOpeningDigestBindingBlockers
abbrev validGeneratedRv64imSideTerminalOpeningDigestBindingCases :=
  Nightstream.Rv64IM.validGeneratedRv64imSideTerminalOpeningDigestBindingCases

def entrypointContract : Prop := True

theorem entrypointContract_true : entrypointContract := by
  trivial

end SideTerminalOpeningDigestBindingInterface

end Nightstream.Rv64IM
