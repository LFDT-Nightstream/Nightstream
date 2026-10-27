import CompressionLean.Checks.Fact
import CompressionLean.Checks.KeyFacts
import CompressionLean.Checks.PoseidonRows
import CompressionLean.Checks.Runs
import CompressionLean.Checks.SetupFacts
import CompressionLean.Checks.Templates

/-!
`Checks/` owns the proofs that a pass of a check that an executable runs on
Rust's files gives the premise of a claim: a check without a failure
(`Fact`), the facts of the layer-1 key (`KeyFacts`), of the setup that Rust
builds (`SetupFacts`), the template names of the terminal package
(`Templates`), the runs that Rust pushes (`Runs`), and the rows of F′'s
Poseidon2 blocks that the run check reads (`PoseidonRows`).

It does not own the facts and the checks (`Spec/Checks.lean`) or the claims
(`Claims.lean`).
-/
