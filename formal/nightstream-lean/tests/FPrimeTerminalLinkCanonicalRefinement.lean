import Nightstream.Implementation.R1CS.Correspondence.FPrime.FPrimeTerminalLinkCanonicalRefinement

/-!
Focused elaboration boundary for the exact terminal-link artifact refinement.
-/

namespace NightstreamTests.FPrimeTerminalLinkCanonicalRefinement

open Nightstream.Implementation.Encoding.FPrime
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.FPrimeTerminalLink
open Nightstream.Implementation.R1CS.FPrimeTerminalLinkCanonicalRefinement
open Nightstream.Implementation.Rust.CanonicalConformance.NativeStep

example (digest : Digest) {z : Nat → Nat}
    (canonical : ∀ column, z column < goldilocksP)
    (one : z 0 = 1)
    (producerAligned : ProducerAligned digest z) :
    Satisfies rows z ↔
      ∃ logical,
        CanonicalPublicInputLink.check digest logical = true /\
          claimOfAssignment z =
            CanonicalPlainCarrierLink.completeClaim logical :=
  satisfies_iff_logicalPaperLink digest canonical one producerAligned

example
    (goldilocksPrime : EuclidPrime goldilocksP)
    {producer terminal : Nat → Nat}
    (producerCanonical :
      ∀ column, producer column < goldilocksP)
    (producerOne : producer 0 = 1)
    (encodingSatisfies :
      Satisfies FPrimeEncoding.rows producer)
    (terminalCanonical :
      ∀ column, terminal column < goldilocksP)
    (terminalOne : terminal 0 = 1)
    (columnsAligned :
      ProducerColumnsAligned producer terminal) :
    Satisfies rows terminal ↔
      ∃ logical,
        CanonicalPublicInputLink.check
          (FPrimeEncodingCanonicalBits.digestOfAssignment
            producer producerCanonical)
          logical = true /\
        claimOfAssignment terminal =
          CanonicalPlainCarrierLink.completeClaim logical :=
  satisfies_iff_logicalPaperLink_of_encodingRows
    goldilocksPrime producerCanonical producerOne encodingSatisfies
    terminalCanonical terminalOne columnsAligned

end NightstreamTests.FPrimeTerminalLinkCanonicalRefinement
