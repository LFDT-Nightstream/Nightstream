import Nightstream.Implementation.R1CS.Correspondence.FPrime.FPrimeCounterSound

/-!
Rust/Lean conformance for the exact production-used recursive F' counter
block. Rust emits all four assignments below; Lean independently recomputes
their verdicts and pins each forgery to the same first failing row.
-/

set_option maxRecDepth 32768
set_option maxHeartbeats 4000000

namespace NightstreamTests.FPrimeCounterArtifact

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.FPrimeCounter

example : schemaVersion = 1 := by decide

example : rows.length = rowCount := by decide

example : Satisfies rows (assignmentOf honestWitness) := by decide

example : ¬ Satisfies rows (assignmentOf wrongSourceWitness) := by decide

example : ∀ r ∈ rows.take 132,
    RowHolds (assignmentOf wrongSourceWitness) r := by decide

example : ¬ RowHolds (assignmentOf wrongSourceWitness)
    ((rows.drop 132).head (by decide)) := by decide

example : ¬ Satisfies rows (assignmentOf wrongStepWitness) := by decide

example : ∀ r ∈ rows.take 139,
    RowHolds (assignmentOf wrongStepWitness) r := by decide

example : ¬ RowHolds (assignmentOf wrongStepWitness)
    ((rows.drop 139).head (by decide)) := by decide

example : ¬ Satisfies rows (assignmentOf wrongRowsWitness) := by decide

example : ∀ r ∈ rows.take 400,
    RowHolds (assignmentOf wrongRowsWitness) r := by decide

example : ¬ RowHolds (assignmentOf wrongRowsWitness)
    ((rows.drop 400).head (by decide)) := by decide

end NightstreamTests.FPrimeCounterArtifact
