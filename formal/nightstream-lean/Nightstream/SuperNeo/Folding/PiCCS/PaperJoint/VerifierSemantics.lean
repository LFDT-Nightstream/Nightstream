import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.SumCheckTruthPath
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.SumCheckInitial
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ProtocolVerifier.Types
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.FiatShamir
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ProtocolVerifier
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.UnifiedProtocolVerifier
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Sampling
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.OutputPoint

/-!
Verifier semantics for the paper-joint `Pi_CCS` model.

Owns: only typed finite SumCheck messages, verifier-derived abstract
challenges, terminal evaluation checks, output-point binding, and the
semantic acceptance decomposition.

Does not own: concrete Poseidon2 encoding or security, Split-NC equivalence,
production output projection, Rust/R1CS refinement, or constraint removal.

Emits constraints: no.

Authority boundary: certificates carry messages, never challenges or an
expected truth callback. The verifier derives its challenges and checks the
actual protocol polynomial at the resulting point.

| Verifier phase | Mathematical obligation |
|---|---|
| initial claim | accepted initial SumCheck claim is tied to the signed joint identity |
| rounds and sampling | finite messages determine verifier challenges and named bad events |
| terminal | message-derived evaluation equals the actual joint polynomial |
| output | the exported point is exactly the verifier's challenge vector |
-/
