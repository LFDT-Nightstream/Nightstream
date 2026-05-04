import DirectCcsFPrime.DirectParentOnlyProductionConcreteFPrimePriorRawIOSoundness

/-!
Concrete split discipline for exact public-IO prior verification.

The raw public vector is not enough by itself: a verifier output also needs a
canonical terminal/boundary split. This module proves that the split follows
from one implementation-facing length check: the terminal slice has the
canonical terminal public-vector length.
-/

namespace DirectCcsFPrime

namespace DirectParentOnlyProductionConcreteFPrimePriorExactIOLayout

/-- Section 7.1-backed production context. -/
abbrev ProductionContext :=
  DirectParentOnlyProductionConcreteFPrimePriorRawIOSoundness.ProductionContext

/-- Public image for the parent-only production path. -/
abbrev PublicImage :=
  DirectParentOnlyProductionConcreteFPrimePriorRawIOSoundness.PublicImage

/-- Proof-carrying folded prior authority for the induced production context. -/
abbrev ProofCarryingPriorProof :=
  @DirectParentOnlyProductionConcreteFPrimePriorRawIOSoundness.ProofCarryingPriorProof

/-- Exact public-IO layout binding consumed by the split raw-IO surface. -/
abbrev ExactPublicIOLayoutBinding :=
  @DirectParentOnlyProductionConcreteFPrimePriorRawIOSoundness.ExactPublicIOLayoutBinding

/--
Implementation-facing terminal-length discipline for exact public IO.

For any verifier output whose raw vector equals the canonical terminal and
Construction-2 boundary concatenation, the exposed terminal slice has the
canonical terminal length. This is enough to make the raw vector equality force
the exact terminal and boundary slices.
-/
structure ExactPublicIOTerminalLengthBinding
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      DirectParentOnlyProductionConcreteFPrimePriorBackend.ConcreteRuntimeExactPublicIOSurface
        (PriorProof := PriorProof)
        ctx) where
  terminalLengthBindsExpected :
    ∀ steps proof image publicIO,
      surface.terminalVerifierPublicIO
        (surface.terminalCommittedProof proof) =
          some publicIO →
      publicIO.raw =
        surface.terminalPublicValues
          (surface.canonicalStatement steps image) ++
        surface.boundaryPublicValues
          (surface.statementBoundary
            (surface.canonicalStatement steps image)) →
      publicIO.terminal.length =
        (surface.terminalPublicValues
          (surface.canonicalStatement steps image)).length

/--
Canonical terminal-slice discipline for exact public IO.

The implementation may expose the split by taking the terminal segment from
the raw Spartan public vector at the canonical terminal length. This condition
is stronger than a bare length check and induces the terminal-length binding
used by the exact-layout proof.
-/
structure ExactPublicIOTerminalSliceBinding
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      DirectParentOnlyProductionConcreteFPrimePriorBackend.ConcreteRuntimeExactPublicIOSurface
        (PriorProof := PriorProof)
        ctx) where
  terminalSliceMatchesCanonicalPrefix :
    ∀ steps proof image publicIO,
      surface.terminalVerifierPublicIO
        (surface.terminalCommittedProof proof) =
          some publicIO →
      publicIO.raw =
        surface.terminalPublicValues
          (surface.canonicalStatement steps image) ++
        surface.boundaryPublicValues
          (surface.statementBoundary
            (surface.canonicalStatement steps image)) →
      publicIO.terminal =
        publicIO.raw.take
          (surface.terminalPublicValues
            (surface.canonicalStatement steps image)).length

/--
Canonical terminal-slice discipline induces terminal-length discipline.
-/
def exactPublicIOTerminalLengthBindingOfTerminalSlice
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      DirectParentOnlyProductionConcreteFPrimePriorBackend.ConcreteRuntimeExactPublicIOSurface
        (PriorProof := PriorProof)
        ctx)
    (sliceBinding : ExactPublicIOTerminalSliceBinding surface) :
    ExactPublicIOTerminalLengthBinding surface where
  terminalLengthBindsExpected := by
    intro steps proof image publicIO hPublicIO hRaw
    let expectedTerminal :=
      surface.terminalPublicValues
        (surface.canonicalStatement steps image)
    let expectedBoundary :=
      surface.boundaryPublicValues
        (surface.statementBoundary
          (surface.canonicalStatement steps image))
    have hTerminal :
        publicIO.terminal =
          publicIO.raw.take expectedTerminal.length := by
      simpa [expectedTerminal, expectedBoundary] using
        sliceBinding.terminalSliceMatchesCanonicalPrefix
          steps
          proof
          image
          publicIO
          hPublicIO
          hRaw
    have hRawLength :
        expectedTerminal.length ≤ publicIO.raw.length := by
      rw [hRaw]
      simp [expectedTerminal]
    calc
      publicIO.terminal.length =
          (publicIO.raw.take expectedTerminal.length).length := by
        rw [hTerminal]
      _ = expectedTerminal.length := by
        simp [List.length_take, hRawLength]

/--
Terminal-length discipline instantiates exact public-IO layout binding.
-/
def exactPublicIOLayoutBindingOfTerminalLength
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      DirectParentOnlyProductionConcreteFPrimePriorBackend.ConcreteRuntimeExactPublicIOSurface
        (PriorProof := PriorProof)
        ctx)
    (lengthBinding : ExactPublicIOTerminalLengthBinding surface) :
    ExactPublicIOLayoutBinding surface where
  rawOutputBindsTerminalBoundary := by
    intro steps proof image publicIO hPublicIO hRaw
    let expectedTerminal :=
      surface.terminalPublicValues
        (surface.canonicalStatement steps image)
    let expectedBoundary :=
      surface.boundaryPublicValues
        (surface.statementBoundary
          (surface.canonicalStatement steps image))
    have hAppend :
        publicIO.terminal ++ publicIO.boundary =
          expectedTerminal ++ expectedBoundary := by
      calc
        publicIO.terminal ++ publicIO.boundary = publicIO.raw :=
          publicIO.raw_eq.symm
        _ = expectedTerminal ++ expectedBoundary := by
          simpa [expectedTerminal, expectedBoundary] using hRaw
    have hLength :
        publicIO.terminal.length = expectedTerminal.length := by
      simpa [expectedTerminal] using
        lengthBinding.terminalLengthBindsExpected
          steps
          proof
          image
          publicIO
          hPublicIO
          hRaw
    have hTerminal : publicIO.terminal = expectedTerminal := by
      have hTake :=
        congrArg (fun xs => xs.take publicIO.terminal.length) hAppend
      simpa [expectedTerminal, expectedBoundary, hLength] using hTake
    have hBoundary : publicIO.boundary = expectedBoundary := by
      have hDrop :=
        congrArg (fun xs => xs.drop publicIO.terminal.length) hAppend
      simpa [expectedTerminal, expectedBoundary, hLength] using hDrop
    exact ⟨hTerminal, hBoundary⟩

/--
Canonical terminal-slice discipline instantiates exact public-IO layout binding.
-/
def exactPublicIOLayoutBindingOfTerminalSlice
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      DirectParentOnlyProductionConcreteFPrimePriorBackend.ConcreteRuntimeExactPublicIOSurface
        (PriorProof := PriorProof)
        ctx)
    (sliceBinding : ExactPublicIOTerminalSliceBinding surface) :
    ExactPublicIOLayoutBinding surface :=
  exactPublicIOLayoutBindingOfTerminalLength
    surface
    (exactPublicIOTerminalLengthBindingOfTerminalSlice
      surface
      sliceBinding)

/--
Runtime verifier predicate induced by exact public IO and terminal-length split
discipline.
-/
def RuntimeVerifyPriorOfExactPublicIOTerminalLengthSoundness
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      DirectParentOnlyProductionConcreteFPrimePriorBackend.ConcreteRuntimeExactPublicIOSurface
        (PriorProof := PriorProof)
        ctx)
    (lengthBinding : ExactPublicIOTerminalLengthBinding surface) :
    Nat →
      PriorProof →
      PublicImage Digest Boundary →
        Prop :=
  DirectParentOnlyProductionConcreteFPrimePriorRawIOSoundness.RuntimeVerifyPriorOfExactPublicIORawSoundness
    surface
    (exactPublicIOLayoutBindingOfTerminalLength surface lengthBinding)

/--
Certified prior verifier induced by exact public IO and terminal-length split
discipline.
-/
def certifiedPriorVerifierOfExactPublicIOTerminalLengthSoundness
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      DirectParentOnlyProductionConcreteFPrimePriorBackend.ConcreteRuntimeExactPublicIOSurface
        (PriorProof := PriorProof)
        ctx)
    (lengthBinding : ExactPublicIOTerminalLengthBinding surface) :
    DirectParentOnlyProductionSuperNeoReuseCertifiedVerifier.CertifiedPriorVerifier
      (PriorProof := PriorProof)
      ctx :=
  DirectParentOnlyProductionConcreteFPrimePriorRawIOSoundness.certifiedPriorVerifierOfExactPublicIORawSoundness
    surface
    (exactPublicIOLayoutBindingOfTerminalLength surface lengthBinding)

/-- The terminal-length certified verifier uses the terminal-length predicate. -/
theorem certifiedPriorVerifierOfExactPublicIOTerminalLengthSoundness_verify
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      DirectParentOnlyProductionConcreteFPrimePriorBackend.ConcreteRuntimeExactPublicIOSurface
        (PriorProof := PriorProof)
        ctx)
    (lengthBinding : ExactPublicIOTerminalLengthBinding surface) :
    (certifiedPriorVerifierOfExactPublicIOTerminalLengthSoundness
      surface
      lengthBinding).verify =
      RuntimeVerifyPriorOfExactPublicIOTerminalLengthSoundness
        surface
        lengthBinding :=
  rfl

/--
Terminal-length exact-public-IO acceptance yields a bound raw-vector statement.
-/
theorem runtimeVerifyPriorOfExactPublicIOTerminalLengthSoundness_boundStatement
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      DirectParentOnlyProductionConcreteFPrimePriorBackend.ConcreteRuntimeExactPublicIOSurface
        (PriorProof := PriorProof)
        ctx)
    (lengthBinding : ExactPublicIOTerminalLengthBinding surface)
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    (hVerify :
      RuntimeVerifyPriorOfExactPublicIOTerminalLengthSoundness
        surface
        lengthBinding
        steps
        proof
        image) :
    ∃ rawPublicIO : List surface.PublicField,
      DirectParentOnlyProductionConcreteFPrimePriorRawIOSoundness.RawPublicIOBoundStatementAccepted
        (DirectParentOnlyProductionConcreteFPrimePriorRawIOSoundness.rawPublicIOVerifierChecksOfExactPublicIO
          surface)
        steps
        proof
        image
        rawPublicIO :=
  DirectParentOnlyProductionConcreteFPrimePriorRawIOSoundness.runtimeVerifyPriorOfExactPublicIORawSoundness_boundStatement
    surface
    (exactPublicIOLayoutBindingOfTerminalLength surface lengthBinding)
    hVerify

/--
Terminal-length exact-public-IO acceptance opens folded F' authority for the
same public pair.
-/
theorem runtimeVerifyPriorOfExactPublicIOTerminalLengthSoundness_acceptedOpens
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      DirectParentOnlyProductionConcreteFPrimePriorBackend.ConcreteRuntimeExactPublicIOSurface
        (PriorProof := PriorProof)
        ctx)
    (lengthBinding : ExactPublicIOTerminalLengthBinding surface) :
    ∀ steps proof image,
      RuntimeVerifyPriorOfExactPublicIOTerminalLengthSoundness
        surface
        lengthBinding
        steps
        proof
        image →
        ∃ authority : ProofCarryingPriorProof ctx,
          surface.openAuthority proof = some authority ∧
            FoldedFPrimeAuthority.Accepts
              (Transition :=
                DirectParentOnlyProductionSoundness.Transition
                  ctx.toProductionContext)
              (initial := ctx.initial)
              steps
              authority
              image := by
  intro steps proof image hVerify
  exact
    DirectParentOnlyProductionConcreteFPrimePriorRawIOSoundness.runtimeVerifyPriorOfExactPublicIORawSoundness_acceptedOpens
      surface
      (exactPublicIOLayoutBindingOfTerminalLength surface lengthBinding)
      steps
      proof
      image
      hVerify

/-- Terminal-length exact-public-IO acceptance reaches its claimed prior image. -/
theorem runtimeVerifyPriorOfExactPublicIOTerminalLengthSoundness_reaches_prior
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      DirectParentOnlyProductionConcreteFPrimePriorBackend.ConcreteRuntimeExactPublicIOSurface
        (PriorProof := PriorProof)
        ctx)
    (lengthBinding : ExactPublicIOTerminalLengthBinding surface)
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    (hVerify :
      RuntimeVerifyPriorOfExactPublicIOTerminalLengthSoundness
        surface
        lengthBinding
        steps
        proof
        image) :
    FPrimeInduction.Reachable
      (DirectParentOnlyProductionSoundness.Transition
        ctx.toProductionContext)
      ctx.initial
      steps
      image :=
  DirectParentOnlyProductionConcreteFPrimePriorRawIOSoundness.runtimeVerifyPriorOfExactPublicIORawSoundness_reaches_prior
    surface
    (exactPublicIOLayoutBindingOfTerminalLength surface lengthBinding)
    hVerify

/--
Terminal-length exact-public-IO acceptance cannot authorize an unreachable
prior image.
-/
theorem runtimeVerifyPriorOfExactPublicIOTerminalLengthSoundness_cannot_accept_unreachable_prior
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      DirectParentOnlyProductionConcreteFPrimePriorBackend.ConcreteRuntimeExactPublicIOSurface
        (PriorProof := PriorProof)
        ctx)
    (lengthBinding : ExactPublicIOTerminalLengthBinding surface)
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    (hVerify :
      RuntimeVerifyPriorOfExactPublicIOTerminalLengthSoundness
        surface
        lengthBinding
        steps
        proof
        image)
    (hUnreachable :
      ¬ FPrimeInduction.Reachable
        (DirectParentOnlyProductionSoundness.Transition
          ctx.toProductionContext)
        ctx.initial
        steps
        image) :
    False :=
  hUnreachable
    (runtimeVerifyPriorOfExactPublicIOTerminalLengthSoundness_reaches_prior
      surface
      lengthBinding
      hVerify)

/--
Terminal-length exact-public-IO acceptance exposes prior public-image
invariants.
-/
theorem runtimeVerifyPriorOfExactPublicIOTerminalLengthSoundness_publicImageInvariants
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      DirectParentOnlyProductionConcreteFPrimePriorBackend.ConcreteRuntimeExactPublicIOSurface
        (PriorProof := PriorProof)
        ctx)
    (lengthBinding : ExactPublicIOTerminalLengthBinding surface)
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    (hVerify :
      RuntimeVerifyPriorOfExactPublicIOTerminalLengthSoundness
        surface
        lengthBinding
        steps
        proof
        image) :
    DirectParentOnlyProductionConcreteFPrimePrior.AcceptedPriorPublicImageInvariants
      ctx
      steps
      image :=
  DirectParentOnlyProductionConcreteFPrimePriorRawIOSoundness.runtimeVerifyPriorOfExactPublicIORawSoundness_publicImageInvariants
    surface
    (exactPublicIOLayoutBindingOfTerminalLength surface lengthBinding)
    hVerify

/-- The terminal-length exact-public-IO prior verifier is same-proof functional. -/
theorem proofFunctionalOfExactPublicIOTerminalLengthSoundness
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      DirectParentOnlyProductionConcreteFPrimePriorBackend.ConcreteRuntimeExactPublicIOSurface
        (PriorProof := PriorProof)
        ctx)
    (lengthBinding : ExactPublicIOTerminalLengthBinding surface) :
    CompressedFPrimeAuthority.ProofFunctional
      (RuntimeVerifyPriorOfExactPublicIOTerminalLengthSoundness
        surface
        lengthBinding) :=
  DirectParentOnlyProductionConcreteFPrimePriorRawIOSoundness.proofFunctionalOfExactPublicIORawSoundness
    surface
    (exactPublicIOLayoutBindingOfTerminalLength surface lengthBinding)

/--
Terminal-length exact-public-IO prior-plus-latest end-to-end theorem.
-/
theorem certifiedSingleTerminalEndToEnd_ofExactPublicIOTerminalLengthSoundnessLatestStep
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      DirectParentOnlyProductionConcreteFPrimePriorBackend.ConcreteRuntimeExactPublicIOSurface
        (PriorProof := PriorProof)
        ctx)
    (lengthBinding : ExactPublicIOTerminalLengthBinding surface)
    {priorSteps : Nat}
    {priorProof : PriorProof}
    {priorImage nextImage : PublicImage Digest Boundary}
    {latestProof : Unit}
    (hPrior :
      RuntimeVerifyPriorOfExactPublicIOTerminalLengthSoundness
        surface
        lengthBinding
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
    DirectParentOnlyProductionSuperNeoReuseEndToEnd.CertifiedTerminalEndToEnd
      ctx
      (certifiedPriorVerifierOfExactPublicIOTerminalLengthSoundness
        surface
        lengthBinding).opening
      priorSteps
      priorSteps
      priorProof
      priorImage
      priorImage
      nextImage
      nextImage :=
  DirectParentOnlyProductionConcreteFPrimePriorRawIOSoundness.certifiedSingleTerminalEndToEnd_ofExactPublicIORawSoundnessLatestStep
    surface
    (exactPublicIOLayoutBindingOfTerminalLength surface lengthBinding)
    hPrior
    hLatest

/--
Terminal-length exact-public-IO projection to the non-aggregate private DEC and
stage facts.
-/
theorem nonAggregatePrivateDecStageFacts_ofExactPublicIOTerminalLengthSoundnessLatestStep
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      DirectParentOnlyProductionConcreteFPrimePriorBackend.ConcreteRuntimeExactPublicIOSurface
        (PriorProof := PriorProof)
        ctx)
    (lengthBinding : ExactPublicIOTerminalLengthBinding surface)
    {priorSteps : Nat}
    {priorProof : PriorProof}
    {priorImage nextImage : PublicImage Digest Boundary}
    {latestProof : Unit}
    (hPrior :
      RuntimeVerifyPriorOfExactPublicIOTerminalLengthSoundness
        surface
        lengthBinding
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
  DirectParentOnlyProductionConcreteFPrimePriorRawIOSoundness.nonAggregatePrivateDecStageFacts_ofExactPublicIORawSoundnessLatestStep
    surface
    (exactPublicIOLayoutBindingOfTerminalLength surface lengthBinding)
    hPrior
    hLatest

/--
Terminal-length exact-public-IO projection to the Section 7.1 owner-target
stage audit.
-/
theorem section71StageTargetAuditTrail_ofExactPublicIOTerminalLengthSoundnessLatestStep
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      DirectParentOnlyProductionConcreteFPrimePriorBackend.ConcreteRuntimeExactPublicIOSurface
        (PriorProof := PriorProof)
        ctx)
    (lengthBinding : ExactPublicIOTerminalLengthBinding surface)
    {priorSteps : Nat}
    {priorProof : PriorProof}
    {priorImage nextImage : PublicImage Digest Boundary}
    {latestProof : Unit}
    (hPrior :
      RuntimeVerifyPriorOfExactPublicIOTerminalLengthSoundness
        surface
        lengthBinding
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
  DirectParentOnlyProductionConcreteFPrimePriorRawIOSoundness.section71StageTargetAuditTrail_ofExactPublicIORawSoundnessLatestStep
    surface
    (exactPublicIOLayoutBindingOfTerminalLength surface lengthBinding)
    hPrior
    hLatest

end DirectParentOnlyProductionConcreteFPrimePriorExactIOLayout

end DirectCcsFPrime
