import Nightstream.Rv64IM.Generated.ParityTypes

/-!
Generated-case surface for the RV64IM side-terminal opening-digest boundary.
This owner keeps the exported case shape intentionally small: just enough to
model the legacy native side-terminal checker shape, the missing semantic
digest binding, and the corrected theorem boundary that Rust is now expected to
refine.
-/

namespace Nightstream.Rv64IM.Generated

structure SideTerminalOpeningDigestBindingCase where
  name : String
  statementOpeningDigest : List Byte
  canonicalOpeningDigest : List Byte
  localDigestChainConsistent : Bool
  claimWitnessAccepted : Bool
  openingWitnessAccepted : Bool
  rustObservedAccepted : Bool
deriving DecidableEq, Repr

end Nightstream.Rv64IM.Generated
