import Nightstream.Implementation.R1CS.Correspondence.CanonicalU64.CanonicalU64Sound

/-!
Rust/Lean conformance for the canonical-u64 artifact: the exact witness
vectors exported by `gadgets_lean_artifact` (Rust) must get the same verdicts
here, including the precise failing row for the forged vector.
-/

-- The `decide` evaluations below walk 69 rows × up to 65 sparse terms during
-- elaboration; the default recursion limit is too small for that.
set_option maxRecDepth 8192

namespace NightstreamTests.R1csArtifact

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.CanonicalU64

example : rows.length = rowCount := by decide

/-- Honest witness: accepted, as in the Rust twin test. -/
example : Satisfies rows (assignmentOf honestWitness) := by decide

/-- Forged `5 + p` re-encoding: rejected overall. -/
example : ¬ Satisfies rows (assignmentOf forgedWitness) := by decide

/-- The forgery passes every row except the last — it is caught by the
canonicity gate (row 68) and nothing earlier, matching Rust's
`first_unsatisfied_row() == Some(68)`. -/
example : ∀ r ∈ rows.take 68, RowHolds (assignmentOf forgedWitness) r := by decide

example : ∀ r ∈ rows.drop 68, ¬ RowHolds (assignmentOf forgedWitness) r := by decide

end NightstreamTests.R1csArtifact
