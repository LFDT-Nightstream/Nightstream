import Nightstream.Implementation.R1CS.Correspondence.FPrime.FPrimeTerminalLinkBatch

/-!
Focused elaboration boundary for arbitrary-batch terminal-link ownership and
paper refinement.
-/

namespace NightstreamTests.FPrimeTerminalLinkBatch

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.FPrimeTerminalLinkBatch
open Nightstream.Implementation.Rust.CanonicalConformance.NativeStep

#check rows_one_eq_artifact
#check physicalIndex_injective
#check physicalIndex_surjective
#check publicColumn_injective
#check publicColumn_surjective_interval
#check satisfies_iff_holds
#check satisfies_iff_checks
#check satisfies_iff_logicalPaperLinks

example :
    rowCount 2 = 540 ∧
      publicColumnCount 2 = 540 ∧
      committedColumnCount 2 = 0 ∧
      auxiliaryColumnCount 2 = 256 := by
  decide

example
    (digest : LinkDigest)
    {batchSize : Nat} {z : Nat → Nat}
    (canonical : ∀ column, z column < goldilocksP)
    (one : z 0 = 1)
    (producerAligned : ProducerAligned digest z) :
    Satisfies (rows batchSize) z ↔
      ∀ claim : Fin batchSize,
        ∃ logical,
          CanonicalPublicInputLink.check digest logical = true ∧
          claimOfAssignment z claim =
            CanonicalPlainCarrierLink.completeClaim logical :=
  satisfies_iff_logicalPaperLinks
    digest canonical one producerAligned

end NightstreamTests.FPrimeTerminalLinkBatch
