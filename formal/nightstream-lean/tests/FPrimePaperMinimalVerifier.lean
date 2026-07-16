import Nightstream.Protocol.FPrime.Paper.Necessity.OutputHash
import Nightstream.Protocol.FPrime.Paper.Soundness

/-!
Public theorem-surface regression for the paper-derived minimal verifier.

| Theorem | Assurance tier | Guarantee |
|---|---|---|
| `minimalRecursiveVerifier_iff_recursiveHolds` | semantic | certificate-backed recursive acceptance is exactly the independent recursive relation |
| `minimalFPrimeVerifier_iff_holds` | semantic | base/recursive verifier acceptance is exactly `Paper.Holds` |
| `minimalPaperFPrimeStep_iff_paperFPrimeStep` | semantic | both public relations expose the same digest language |
-/

open Nightstream.Protocol.FPrime.Paper

#check RecursiveCertificate
#check RecursiveAccepts
#check MinimalRecursiveVerifierAccepts
#check MinimalFPrimeVerifierAccepts
#check minimalRecursiveVerifier_sound
#check minimalRecursiveVerifier_complete
#check minimalRecursiveVerifier_iff_recursiveHolds
#check minimalFPrimeVerifier_sound
#check minimalFPrimeVerifier_complete
#check minimalFPrimeVerifier_iff_holds
#check MinimalPaperFPrimeStep
#check minimalPaperFPrimeStep_iff_paperFPrimeStep
#check RecursiveCertificate.verifier
#check RecursiveCertificate.selectedOutput
#check KnowledgeBoundary
#check RecursiveCertificate.inputsValid_or_badEvent
#check Necessity.OutputHash.replaceDigest
#check Necessity.OutputHash.replaceDigestCertificate
#check Necessity.OutputHash.forged_accepts_withoutOutputHash
#check Necessity.OutputHash.forged_not_outputHolds
#check Necessity.OutputHash.forged_rejected_by_fullVerifier
#check Necessity.OutputHash.outputHash_is_necessary
