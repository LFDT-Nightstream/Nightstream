import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPriorStateReplayTransitionExecutionCertificate
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPriorStateReplayDigestExecutionCertificate

/-!
Facade for exact prior-state replay execution certificates.

Owns only the handwritten import boundary. Transition slices and digest traces
remain opaque in their focused leaf modules. It owns no row satisfaction,
semantic refinement, lifecycle selection, or target authority.

Emits constraints: no.
-/

set_option autoImplicit false
