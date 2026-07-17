import Nightstream.Protocol.FPrime.Paper
import Nightstream.Protocol.FPrime.Paper.Completeness
import Nightstream.Protocol.FPrime.ConcretePhi81
import Nightstream.Protocol.Terminal.CE

/-!
Curated public surface for independent protocol semantics.

The callback-oriented `FPrime.Step`, compact production `FPrime.XOut`, and
generic `FPrime.Paper.CertificateVerifier` are implementation or legacy
surfaces and are intentionally absent. They do not authorize constraint
removal.
-/
