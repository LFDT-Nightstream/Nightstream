import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.FPrimeFullHistoryTerminalLogicalLinkSound

/-!
Focused elaboration boundary for the full-history terminal logical link.
-/

namespace NightstreamTests.FPrimeFullHistoryTerminalLogicalLink

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalLogicalLinkSound
open Nightstream.Implementation.Rust.CanonicalConformance.NativeStep

example
    (goldilocksPrime : EuclidPrime goldilocksP)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (encodingSatisfies :
      Satisfies FPrimeFullHistoryOutputEncoding.rows assignment)
    (terminalLinkSatisfies :
      Satisfies FPrimeFullHistoryTerminalLink.rows assignment) :
    CanonicalPublicInputLink.check
      (outputDigest assignment canonical)
      (terminalLogicalPublic assignment) = true :=
  logicalCheck_of_rows goldilocksPrime canonical one
    encodingSatisfies terminalLinkSatisfies

end NightstreamTests.FPrimeFullHistoryTerminalLogicalLink
