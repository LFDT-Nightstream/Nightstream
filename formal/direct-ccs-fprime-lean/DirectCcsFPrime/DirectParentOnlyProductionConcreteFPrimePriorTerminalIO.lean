import DirectCcsFPrime.DirectParentOnlyProductionConcreteFPrimePriorTerminalIOOpening

/-!
Terminal committed public-IO binding for the concrete prior F' verifier.

This module is the production-facing name for the opening-level terminal IO
surface. Verifier acceptance is replay plus terminal public-IO prefix/suffix
checking; folded F' authority is derived through the fixed opener and backend
soundness obligations owned by the opening surface.
-/

namespace DirectCcsFPrime

namespace DirectParentOnlyProductionConcreteFPrimePriorTerminalIO

/-- Section 7.1-backed production context. -/
abbrev ProductionContext :=
  DirectParentOnlyProductionConcreteFPrimePriorTerminalIOOpening.ProductionContext

/-- Public image for the parent-only production path. -/
abbrev PublicImage :=
  DirectParentOnlyProductionConcreteFPrimePriorTerminalIOOpening.PublicImage

/-- Proof-carrying folded prior authority for the induced production context. -/
abbrev ProofCarryingPriorProof :=
  @DirectParentOnlyProductionConcreteFPrimePriorTerminalIOOpening.ProofCarryingPriorProof

/--
Terminal committed verifier surface with explicit public-IO opening soundness.
-/
abbrev ConcreteTerminalCommittedIOSurface :=
  @DirectParentOnlyProductionConcreteFPrimePriorTerminalIOOpening.TerminalIOOpeningSurface

/-- Acceptance predicate induced by terminal committed public-IO binding. -/
abbrev VerifyPriorOfTerminalIO :=
  @DirectParentOnlyProductionConcreteFPrimePriorTerminalIOOpening.VerifyPrior

/-- Fixed authority opener induced by terminal committed public-IO binding. -/
abbrev authorityOpenerOfTerminalIO :=
  @DirectParentOnlyProductionConcreteFPrimePriorTerminalIOOpening.authorityOpener

/--
Fully exposed evidence returned by terminal public-IO verifier acceptance.
-/
abbrev AcceptedTerminalIOEvidence :=
  @DirectParentOnlyProductionConcreteFPrimePriorTerminalIOOpening.AcceptedTerminalIOEvidence

/--
Terminal public-IO verifier acceptance exposes all authority-relevant checks.
-/
abbrev verifyPriorOfTerminalIO_evidence :=
  @DirectParentOnlyProductionConcreteFPrimePriorTerminalIOOpening.verifyPrior_evidence

/--
Terminal committed public-IO binding proves the accepted-opens obligation.
-/
abbrev acceptedOpensOfTerminalIO :=
  @DirectParentOnlyProductionConcreteFPrimePriorTerminalIOOpening.acceptedOpens

/-- A terminal public-IO prior acceptance always opens some authority. -/
abbrev verifyPriorOfTerminalIO_openAuthority_ne_none :=
  @DirectParentOnlyProductionConcreteFPrimePriorTerminalIOOpening.verifyPrior_openAuthority_ne_none

/--
If the fixed terminal-IO opener returns a concrete authority for an accepted
proof, that exact authority accepts the same `(steps, image)` pair.
-/
abbrev verifyPriorOfTerminalIO_openedAuthority_accepts_of_open :=
  @DirectParentOnlyProductionConcreteFPrimePriorTerminalIOOpening.verifyPrior_openedAuthority_accepts

/-- Every accepted terminal-IO prior proof reaches its claimed prior image. -/
abbrev verifyPriorOfTerminalIO_reaches_prior :=
  @DirectParentOnlyProductionConcreteFPrimePriorTerminalIOOpening.verifyPrior_reaches_prior

/-- A terminal-IO verifier cannot accept an unreachable prior public image. -/
abbrev verifyPriorOfTerminalIO_cannot_accept_unreachable_prior :=
  @DirectParentOnlyProductionConcreteFPrimePriorTerminalIOOpening.verifyPrior_cannot_accept_unreachable_prior

/--
Accepted terminal public-IO verifier proofs expose prior public-image
invariants.
-/
abbrev verifyPriorOfTerminalIO_publicImageInvariants :=
  @DirectParentOnlyProductionConcreteFPrimePriorTerminalIOOpening.verifyPrior_publicImageInvariants

/--
The terminal public-IO concrete prior verifier is same-proof functional.
-/
abbrev proofFunctionalOfTerminalIO :=
  @DirectParentOnlyProductionConcreteFPrimePriorTerminalIOOpening.proofFunctional

/--
Certified prior verifier induced by terminal committed public-IO binding.
-/
abbrev certifiedPriorVerifierOfTerminalIO :=
  @DirectParentOnlyProductionConcreteFPrimePriorTerminalIOOpening.certifiedVerifier

/-- The certified verifier induced by terminal public IO uses that predicate. -/
abbrev certifiedPriorVerifierOfTerminalIO_verify :=
  @DirectParentOnlyProductionConcreteFPrimePriorTerminalIOOpening.certifiedVerifier_verify

/--
Terminal public-IO prior-plus-latest end-to-end theorem.
-/
abbrev certifiedSingleTerminalEndToEnd_ofConcreteTerminalIOLatestStep :=
  @DirectParentOnlyProductionConcreteFPrimePriorTerminalIOOpening.certifiedEndToEndOfLatestStep

/--
Terminal public-IO projection to the non-aggregate private DEC and stage facts.
-/
theorem nonAggregatePrivateDecStageFacts_ofConcreteTerminalIOLatestStep
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ConcreteTerminalCommittedIOSurface (PriorProof := PriorProof) ctx)
    {priorSteps : Nat}
    {priorProof : PriorProof}
    {priorImage nextImage : PublicImage Digest Boundary}
    {latestProof : Unit}
    (hPrior :
      VerifyPriorOfTerminalIO
        surface
        priorSteps
        priorProof
        priorImage)
    (hLatest :
      DirectParentOnlyProductionSoundness.VerifyLatestStep
        ctx.toProductionContext
        priorSteps
        priorProof
        priorImage
        nextImage
        latestProof) :
    DirectParentOnlyProductionSuperNeoReuseEndToEnd.CertifiedTerminalNonAggregatePrivateDecStageFacts
      ctx
      priorSteps
      priorImage
      nextImage
      nextImage :=
  DirectParentOnlyProductionSuperNeoReuseEndToEnd.nonAggregatePrivateDecStageFacts_ofCertifiedPriorVerifierLatestStep
    (certifiedPriorVerifierOfTerminalIO surface)
    hPrior
    hLatest

/--
Terminal public-IO projection to the Section 7.1 owner-target audit.
-/
theorem section71StageTargetAuditTrail_ofConcreteTerminalIOLatestStep
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ConcreteTerminalCommittedIOSurface (PriorProof := PriorProof) ctx)
    {priorSteps : Nat}
    {priorProof : PriorProof}
    {priorImage nextImage : PublicImage Digest Boundary}
    {latestProof : Unit}
    (hPrior :
      VerifyPriorOfTerminalIO
        surface
        priorSteps
        priorProof
        priorImage)
    (hLatest :
      DirectParentOnlyProductionSoundness.VerifyLatestStep
        ctx.toProductionContext
        priorSteps
        priorProof
        priorImage
        nextImage
        latestProof) :
    DirectParentOnlyProductionSuperNeoReuse.ProductionContext.TerminalSection71StageTargetAuditTrail
      ctx
      priorSteps
      priorImage
      nextImage
      nextImage :=
  DirectParentOnlyProductionSuperNeoReuseEndToEnd.section71StageTargetAuditTrail_ofCertifiedPriorVerifierLatestStep
    (certifiedPriorVerifierOfTerminalIO surface)
    hPrior
    hLatest

end DirectParentOnlyProductionConcreteFPrimePriorTerminalIO

end DirectCcsFPrime
