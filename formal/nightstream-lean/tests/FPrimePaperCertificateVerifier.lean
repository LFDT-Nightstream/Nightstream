import Nightstream.Protocol.FPrime.Paper.Necessity.OutputHash
import Nightstream.Protocol.FPrime.Paper.Soundness

/-!
Public theorem-surface regression for the paper-derived certificate verifier.

| Theorem | Assurance tier | Guarantee |
|---|---|---|
| `certificateRecursiveVerifier_iff_recursiveHolds` | semantic | certificate-backed recursive acceptance is exactly the independent recursive relation |
| `certificateFPrimeVerifier_iff_holds` | semantic | base/recursive verifier acceptance is exactly `Paper.Holds` |
| `certificatePaperFPrimeStep_iff_paperFPrimeStep` | semantic | both public relations expose the same digest language |
-/

open Nightstream.Protocol.FPrime.Paper

#check RecursiveCertificate
#check RecursiveAccepts
#check CertificateRecursiveVerifierAccepts
#check CertificateFPrimeVerifierAccepts
#check certificateRecursiveVerifier_sound
#check certificateRecursiveVerifier_complete
#check certificateRecursiveVerifier_iff_recursiveHolds
#check certificateFPrimeVerifier_sound
#check certificateFPrimeVerifier_complete
#check certificateFPrimeVerifier_iff_holds
#check CertificatePaperFPrimeStep
#check certificatePaperFPrimeStep_iff_paperFPrimeStep
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
