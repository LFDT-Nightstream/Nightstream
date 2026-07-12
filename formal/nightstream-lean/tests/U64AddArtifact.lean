import Nightstream.Implementation.R1CS.Correspondence.U64.U64AddSound

/-!
Rust/Lean conformance for the exact no-wrap u64-add artifact used by the F'
step-count path. The overflow witness passes every row before the final
no-carry equation on both sides.
-/

set_option maxRecDepth 16384

namespace NightstreamTests.U64AddArtifact

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.U64Add

example : rows.length = rowCount := by decide

example : Satisfies rows (assignmentOf honestWitness) := by decide

example : ¬ Satisfies rows (assignmentOf overflowWitness) := by decide

example : ∀ r ∈ rows.take 318,
    RowHolds (assignmentOf overflowWitness) r := by decide

example : ∀ r ∈ rows.drop 318,
    ¬ RowHolds (assignmentOf overflowWitness) r := by decide

end NightstreamTests.U64AddArtifact
