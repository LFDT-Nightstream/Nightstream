import CompressionLean.Final.Bridge
import CompressionLean.Final.Budget
import CompressionLean.Final.Claim
import CompressionLean.Final.Parent
import CompressionLean.Final.ParentClaim
import CompressionLean.Final.RustSetup
import CompressionLean.Final.Soundness

/-!
`Final/` owns the chaining of the layers: the terminal claim and the
encoding premise (`Claim`), the layer-1 budget at Rust's split (`Budget`),
the end-to-end bound (`Soundness`), the bound for F′'s parent claim
(`Parent`, `ParentClaim`), the maps from F′'s algebra to the model's
(`Bridge`), and what Rust's setup build gives (`RustSetup`).

It does not own the definitions that the claims name (`Spec/`), a check that
an executable runs (`Checks/`) or the proof of the encoding premise
(`Sim/Encodes.lean`).
-/
