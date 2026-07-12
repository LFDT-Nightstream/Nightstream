import Nightstream.Implementation.R1CS.Ownership.FPrimeRecursive.FPrimeRecursiveManifest
import Nightstream.Assurance.FPrimeConcreteNifs

/-!
Ownership router for recursive F' circuit assurance.

`FPrimeRecursiveManifest` proves the exact generated row partition.
`FPrimeConcreteNifs` supplies the independent executable verifier, the
satisfaction-to-semantics compiler theorems, and the exact-or-projection-root
result used by the supported full-history shell.

This module intentionally defines no caller-filled semantic certificate.  In
particular, NIFS success is recomputed by `FPrimeConcreteNifs.recursiveCheck`;
it is never accepted as a proposition field.
-/
