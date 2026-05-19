import Nightstream.Rv64IM.Generated.SideTerminalOpeningDigestBindingTypes

/-!
Lean-owned corpus for the RV64IM side-terminal opening-digest boundary.
The honest case reflects the accepted-artifact path with the canonical opening
digest. The tampered case preserves the historical counterexample against the
legacy native boundary shape: the local witness checks still pass after
rebinding the statement digest chain to a forged opening-artifact digest, even
though the corrected Rust boundary now rejects the case.
-/

namespace Nightstream.Rv64IM.Generated

namespace SideTerminalOpeningDigestBindingCases

def honestCase : SideTerminalOpeningDigestBindingCase :=
  { name := "control_flow_jal_skip_ecall_honest"
  , statementOpeningDigest := bytes [1, 7, 9, 11, 13, 15, 17, 19]
  , canonicalOpeningDigest := bytes [1, 7, 9, 11, 13, 15, 17, 19]
  , localDigestChainConsistent := true
  , claimWitnessAccepted := true
  , openingWitnessAccepted := true
  , rustObservedAccepted := true
  }

def tamperedCase : SideTerminalOpeningDigestBindingCase :=
  { name := "control_flow_jal_skip_ecall_wrong_opening_digest"
  , statementOpeningDigest := bytes [2, 7, 9, 11, 13, 15, 17, 19]
  , canonicalOpeningDigest := bytes [1, 7, 9, 11, 13, 15, 17, 19]
  , localDigestChainConsistent := true
  , claimWitnessAccepted := true
  , openingWitnessAccepted := true
  , rustObservedAccepted := false
  }

def cases : List SideTerminalOpeningDigestBindingCase :=
  [honestCase, tamperedCase]

def caseByName? (name : String) : Option SideTerminalOpeningDigestBindingCase :=
  cases.find? fun c => c.name = name

end SideTerminalOpeningDigestBindingCases

end Nightstream.Rv64IM.Generated
