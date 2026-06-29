import DirectCcsFPrime.ProofSystem.Production.Impl.RawProduction.DirectParentOnlyProductionConcreteFPrimePriorRawProductionExactAccepted

/-!
Compatibility facade for the production prior F' verifier surface.

The implementation is split by responsibility:

* `DirectParentOnlyProductionConcreteFPrimePriorRawProductionRaw` owns raw
  public-vector replay, raw opening, and raw certified-verifier consequences.
* `DirectParentOnlyProductionConcreteFPrimePriorRawProductionExactAccepted`
  owns structured terminal/boundary public-IO acceptance and exact certified
  verifier consequences.

Downstream modules may keep importing this file when they need the complete
production verifier surface.
-/

