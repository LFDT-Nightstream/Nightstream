import CompressionLean.Checks.Fact
import CompressionLean.Checks.KeyFacts
import CompressionLean.Checks.PoseidonRows
import CompressionLean.Checks.SetupFacts
import CompressionLean.Checks.Templates

/-!
`Checks/` owns the decided facts that an executable checks on Rust's files,
and the proofs that a pass gives the premise of a claim: the shape of a check
(`Fact`), the facts of the layer-1 key (`KeyFacts`), of the setup that Rust
builds (`SetupFacts`), the template names of the terminal package
(`Templates`), and the rows of F′'s Poseidon2 blocks (`PoseidonRows`).

It does not own the claims (`Claims.lean`).
-/
