import Nightstream.Implementation.R1CS.Correspondence.U64.U64IncrementSound

/-!
Rust/Lean conformance for the exact no-wrap u64 increment artifact. Both
witnesses are emitted by the Rust twin test; Lean independently recomputes
their row verdicts and pins the overflow failure to the final equation.
-/

set_option maxRecDepth 16384

namespace NightstreamTests.U64IncrementArtifact

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.U64Increment

example : rows.length = rowCount := by decide

example : Satisfies rows (assignmentOf honestWitness) := by decide

example : ¬ Satisfies rows (assignmentOf overflowWitness) := by decide

example : ∀ r ∈ rows.take 254,
    RowHolds (assignmentOf overflowWitness) r := by decide

example : ∀ r ∈ rows.drop 254,
    ¬ RowHolds (assignmentOf overflowWitness) r := by decide

end NightstreamTests.U64IncrementArtifact
