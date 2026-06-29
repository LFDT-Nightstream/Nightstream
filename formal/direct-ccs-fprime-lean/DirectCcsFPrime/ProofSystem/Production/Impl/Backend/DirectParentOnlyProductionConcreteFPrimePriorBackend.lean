import DirectCcsFPrime.ProofSystem.Production.Impl.Backend.DirectParentOnlyProductionConcreteFPrimePriorBackendRawPublicIO

/-!
Compatibility facade for the concrete prior F' runtime backend.

The implementation is split by responsibility:

* `DirectParentOnlyProductionConcreteFPrimePriorBackendBase` owns the generic
  runtime backend surface and authority consequences.
* `DirectParentOnlyProductionConcreteFPrimePriorBackendExactPublicIO` owns the
  structured terminal/boundary public-IO adapter.
* `DirectParentOnlyProductionConcreteFPrimePriorBackendRawPublicIO` owns the
  raw public-vector adapter.

Downstream modules may keep importing this file when they need the complete
backend surface.
-/

