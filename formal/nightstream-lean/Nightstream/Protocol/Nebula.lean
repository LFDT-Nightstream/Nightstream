import Nightstream.Protocol.Nebula.Fingerprint
import Nightstream.Protocol.Nebula.Memory
import Nightstream.Protocol.Nebula.PaperFingerprint
import Nightstream.Protocol.Nebula.PaperFinalization

/-!
Public facade for the Lean-owned Nebula memory semantics.

Assurance tier: model-level.

This facade owns no additional definitions. `PaperFingerprint` is the literal
Corollary 8 model. `Fingerprint` is the separate packed production variant.
`PaperFinalization` records the Layer-1 and Layer-2 verifier obligations.
Physical CCS rows and F-prime integration remain outside this facade until
their refinement theorems exist.
-/
