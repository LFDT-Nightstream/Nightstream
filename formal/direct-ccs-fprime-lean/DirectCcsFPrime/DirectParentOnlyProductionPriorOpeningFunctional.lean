import DirectCcsFPrime.DirectParentOnlyProductionPriorOpening

namespace DirectCcsFPrime

namespace DirectParentOnlyProductionPriorOpening

theorem proofFunctional_of_authority_opener
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : DirectParentOnlyProductionSoundness.Context Digest Boundary n params)
    (opener : PriorAuthorityOpener (PriorProof := PriorProof) ctx) :
    CompressedFPrimeAuthority.ProofFunctional
      (VerifyWithAuthorityOpener ctx opener) := by
  intro stepsA stepsB proof imageA imageB hA hB
  exact
    verifyWithAuthorityOpener_functional_for_same_proof
      ctx
      opener
      hA
      hB

theorem soundVerifier_of_authority_opener_proofFunctional
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : DirectParentOnlyProductionSoundness.Context Digest Boundary n params)
    (opener : PriorAuthorityOpener (PriorProof := PriorProof) ctx) :
    CompressedFPrimeAuthority.SoundVerifier.ProofFunctional
      (soundVerifier_of_authority_opener ctx opener) := by
  intro stepsA stepsB proof imageA imageB hA hB
  exact
    verifyWithAuthorityOpener_functional_for_same_proof
      ctx
      opener
      hA
      hB

theorem proofFunctional_of_priorVerifierAuthorityOpening
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : DirectParentOnlyProductionSoundness.Context Digest Boundary n params)
    {VerifyPrior :
      Nat →
        PriorProof →
        DirectParentOnlyProductionSoundness.PublicImage Digest Boundary →
          Prop}
    (opening : PriorVerifierAuthorityOpening ctx VerifyPrior) :
    CompressedFPrimeAuthority.ProofFunctional VerifyPrior := by
  intro stepsA stepsB proof imageA imageB hA hB
  exact
    priorVerifierAuthorityOpening_functional_for_same_proof
      ctx
      opening
      hA
      hB

theorem soundVerifier_of_priorVerifierAuthorityOpening_proofFunctional
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : DirectParentOnlyProductionSoundness.Context Digest Boundary n params)
    {VerifyPrior :
      Nat →
        PriorProof →
        DirectParentOnlyProductionSoundness.PublicImage Digest Boundary →
          Prop}
    (opening : PriorVerifierAuthorityOpening ctx VerifyPrior) :
    CompressedFPrimeAuthority.SoundVerifier.ProofFunctional
      (soundVerifier_of_priorVerifierAuthorityOpening ctx opening) := by
  intro stepsA stepsB proof imageA imageB hA hB
  exact
    priorVerifierAuthorityOpening_functional_for_same_proof
      ctx
      opening
      hA
      hB

end DirectParentOnlyProductionPriorOpening

end DirectCcsFPrime
