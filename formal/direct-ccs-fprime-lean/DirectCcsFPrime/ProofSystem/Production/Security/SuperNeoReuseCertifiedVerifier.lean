import DirectCcsFPrime.ProofSystem.Production.Security.SuperNeoReuseCertifiedVerifierReplay

/-!
Facade for the Section 7.1-backed certified prior verifier package.

The implementation is split by responsibility:

* `DirectParentOnlyProductionSuperNeoReuseCertifiedVerifierCore` owns the
  verifier object and F' authority-opening boundary.
* `DirectParentOnlyProductionSuperNeoReuseCertifiedVerifierTerminal` owns the
  one-terminal audit package.
* `DirectParentOnlyProductionSuperNeoReuseCertifiedVerifierReplay` owns
  same-proof replay and explicit no-swap projections.
-/
