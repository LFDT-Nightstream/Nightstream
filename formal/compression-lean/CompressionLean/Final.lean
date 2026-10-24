import CompressionLean.Final.Bridge
import CompressionLean.Final.Budget
import CompressionLean.Final.Claim
import CompressionLean.Final.Parent
import CompressionLean.Final.ParentClaim
import CompressionLean.Final.RustSetup
import CompressionLean.Final.Soundness

/-!
`Final/` owns the chaining of the layers: the claim of a final proof and the
encoding premise (`Claim`), the layer-1 budget at Rust's split (`Budget`),
the end-to-end bound (`Soundness`), F′'s parent
claim and the bound for it (`Parent`, `ParentClaim`), the maps from F′'s
algebra to the model's (`Bridge`), and Rust's setup build (`RustSetup`).

It does not own a check that an executable runs (`Checks/`) or the proof of
the encoding premise (`Sim/Encodes.lean`).
-/
