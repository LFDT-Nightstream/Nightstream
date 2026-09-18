import DirectCcsFPrime.ProofSystem.Production.Impl.PublicIO.Raw.Core

/-!
Focused soundness certificate for the raw public-IO prior F' verifier.

This module splits the raw verifier boundary into two auditable pieces:
verifier-visible raw public-IO acceptance and the trusted compressed-verifier
authority theorem for a bound statement. The trusted theorem receives an exact
raw vector equality for the same canonical `(steps, image)` statement; callers
do not provide a loose accepted-opens proof.
-/

namespace DirectCcsFPrime

namespace DirectParentOnlyProductionConcreteFPrimePriorRawIOSoundness

/-- Section 7.1-backed production context. -/
abbrev ProductionContext :=
  DirectParentOnlyProductionConcreteFPrimePriorRawIO.ProductionContext

/-- Public image for the parent-only production path. -/
abbrev PublicImage :=
  DirectParentOnlyProductionConcreteFPrimePriorRawIO.PublicImage

/-- Proof-carrying folded prior authority for the induced production context. -/
abbrev ProofCarryingPriorProof :=
  @DirectParentOnlyProductionConcreteFPrimePriorRawIO.ProofCarryingPriorProof

/--
Verifier-visible raw public-vector checks, without an authority theorem.
-/
structure ConcreteRawPublicIOVerifierChecks
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : ProductionContext Digest Boundary n params) where
  Statement : Type
  PublicBoundary : Type
  PublicField : Type
  TerminalCommittedProof : Type
  canonicalStatement :
    Nat →
      PublicImage Digest Boundary →
        Statement
  proofStatement :
    PriorProof →
      Statement
  statementBoundary :
    Statement →
      PublicBoundary
  proofBoundary :
    PriorProof →
      PublicBoundary
  terminalPublicValues :
    Statement →
      List PublicField
  boundaryPublicValues :
    PublicBoundary →
      List PublicField
  terminalCommittedProof :
    PriorProof →
      TerminalCommittedProof
  statementPublicValid :
    Statement →
      Prop
  terminalVerifierPublicIO :
    TerminalCommittedProof →
      Option (List PublicField)
  compactImageReplay :
    Nat →
      PriorProof →
      PublicImage Digest Boundary →
        Prop
  construction2BoundaryReplay :
    Nat →
      PriorProof →
      PublicImage Digest Boundary →
        Prop
  transcriptReplay :
    Nat →
      PriorProof →
      PublicImage Digest Boundary →
        Prop
  openAuthority :
    PriorProof →
      Option (ProofCarryingPriorProof ctx)

/--
Verifier-visible raw public-vector acceptance for a chosen raw vector.
-/
def RawPublicIOVerifierAccepted
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (checks :
      ConcreteRawPublicIOVerifierChecks (PriorProof := PriorProof) ctx)
    (steps : Nat)
    (proof : PriorProof)
    (image : PublicImage Digest Boundary)
    (rawPublicIO : List checks.PublicField) : Prop :=
  checks.compactImageReplay steps proof image ∧
    checks.construction2BoundaryReplay steps proof image ∧
    checks.transcriptReplay steps proof image ∧
    checks.statementPublicValid
      (checks.canonicalStatement steps image) ∧
    checks.proofBoundary proof =
      checks.statementBoundary
        (checks.canonicalStatement steps image) ∧
    checks.terminalVerifierPublicIO
      (checks.terminalCommittedProof proof) =
        some rawPublicIO ∧
    rawPublicIO =
      checks.terminalPublicValues
        (checks.canonicalStatement steps image) ++
      checks.boundaryPublicValues
        (checks.statementBoundary
          (checks.canonicalStatement steps image))

/--
Raw public-vector acceptance after replay binds the opaque proof statement to
the canonical statement for the same `(steps, image)` pair.
-/
def RawPublicIOBoundStatementAccepted
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (checks :
      ConcreteRawPublicIOVerifierChecks (PriorProof := PriorProof) ctx)
    (steps : Nat)
    (proof : PriorProof)
    (image : PublicImage Digest Boundary)
    (rawPublicIO : List checks.PublicField) : Prop :=
  RawPublicIOVerifierAccepted checks steps proof image rawPublicIO ∧
    checks.proofStatement proof =
      checks.canonicalStatement steps image

/--
Authority soundness certificate for the raw public-vector verifier.

The only trusted cryptographic field is `rawBoundStatementAuthoritySound`: once
replay has bound the proof to the canonical statement and the terminal verifier
has returned exactly `terminal_public_values ++ boundary_public_values`, the
proof opens to folded F' reachability authority for the same public pair.
-/
structure ConcreteRawPublicIOSoundnessSurface
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : ProductionContext Digest Boundary n params) where
  checks :
    ConcreteRawPublicIOVerifierChecks (PriorProof := PriorProof) ctx
  replayBindsProofStatement :
    ∀ steps proof image,
      checks.compactImageReplay steps proof image →
      checks.construction2BoundaryReplay steps proof image →
      checks.transcriptReplay steps proof image →
        checks.proofStatement proof =
          checks.canonicalStatement steps image
  rawBoundStatementAuthoritySound :
    ∀ steps proof image rawPublicIO,
      RawPublicIOBoundStatementAccepted
        checks
        steps
        proof
        image
        rawPublicIO →
        ∃ authority : ProofCarryingPriorProof ctx,
          checks.openAuthority proof = some authority ∧
            FoldedFPrimeAuthority.Accepts
              (Transition :=
                DirectParentOnlyProductionSoundness.Transition
                  ctx.toProductionContext)
              (initial := ctx.initial)
              steps
              authority
              image

/--
Verifier-visible checks extracted from the older raw-runtime surface.

This adapter is intentionally one-way: it lets existing raw-runtime
instantiations be audited through the split F' boundary without making the
split boundary depend on the monolithic runtime soundness field.
-/
def rawPublicIOVerifierChecksOfRuntimeRawPublicIO
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      DirectParentOnlyProductionConcreteFPrimePriorRawIO.ConcreteRuntimeRawPublicIOSurface
        (PriorProof := PriorProof)
        ctx) :
    ConcreteRawPublicIOVerifierChecks (PriorProof := PriorProof) ctx where
  Statement := surface.Statement
  PublicBoundary := surface.PublicBoundary
  PublicField := surface.PublicField
  TerminalCommittedProof := surface.TerminalCommittedProof
  canonicalStatement := surface.canonicalStatement
  proofStatement := surface.proofStatement
  statementBoundary := surface.statementBoundary
  proofBoundary := surface.proofBoundary
  terminalPublicValues := surface.terminalPublicValues
  boundaryPublicValues := surface.boundaryPublicValues
  terminalCommittedProof := surface.terminalCommittedProof
  statementPublicValid := surface.statementPublicValid
  terminalVerifierPublicIO := surface.terminalVerifierPublicIO
  compactImageReplay := surface.compactImageReplay
  construction2BoundaryReplay := surface.construction2BoundaryReplay
  transcriptReplay := surface.transcriptReplay
  openAuthority := surface.openAuthority

/--
Split F' soundness surface induced by an older raw-runtime surface.

The verifier predicate and opener are inherited unchanged. The proof exposes
the exact two obligations hidden inside the old runtime surface: replay binds
the opaque proof to the canonical statement, and raw public-vector authority
soundness applies only after that binding.
-/
def rawPublicIOSoundnessSurfaceOfRuntimeRawPublicIO
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      DirectParentOnlyProductionConcreteFPrimePriorRawIO.ConcreteRuntimeRawPublicIOSurface
        (PriorProof := PriorProof)
        ctx) :
    ConcreteRawPublicIOSoundnessSurface (PriorProof := PriorProof) ctx where
  checks := rawPublicIOVerifierChecksOfRuntimeRawPublicIO surface
  replayBindsProofStatement := by
    intro steps proof image hCompact hBoundaryReplay hTranscript
    exact
      surface.replayBindsProofStatement
        steps
        proof
        image
        hCompact
        hBoundaryReplay
        hTranscript
  rawBoundStatementAuthoritySound := by
    intro steps proof image rawPublicIO hBound
    rcases hBound with ⟨hAccepted, hStatement⟩
    rcases hAccepted with
      ⟨hCompact,
        hBoundaryReplay,
        hTranscript,
        hValid,
        hBoundary,
        hRawPublicIO,
        hRawEq⟩
    exact
      surface.rawRuntimeSound
        steps
        proof
        image
        rawPublicIO
        hCompact
        hBoundaryReplay
        hTranscript
        hStatement
        hValid
        hBoundary
        hRawPublicIO
        hRawEq

/--
Layout binding for the structured exact-public-IO verifier output.

The raw vector returned by the verifier must determine the terminal and
Construction-2 boundary split used by the exact backend soundness theorem.
This rules out accepting the right concatenation while hiding a different
internal terminal/boundary decomposition inside the verifier output.
-/
structure ExactPublicIOLayoutBinding
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      DirectParentOnlyProductionConcreteFPrimePriorBackend.ConcreteRuntimeExactPublicIOSurface
        (PriorProof := PriorProof)
        ctx) where
  rawOutputBindsTerminalBoundary :
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
          surface.terminalPublicValues
            (surface.canonicalStatement steps image) ∧
        publicIO.boundary =
          surface.boundaryPublicValues
            (surface.statementBoundary
              (surface.canonicalStatement steps image))

/--
Verifier-visible raw public-vector checks extracted from exact public IO.

The verifier output is projected to its raw vector, while the split layout is
kept out of the verifier predicate and used only by the authority theorem.
-/
def rawPublicIOVerifierChecksOfExactPublicIO
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      DirectParentOnlyProductionConcreteFPrimePriorBackend.ConcreteRuntimeExactPublicIOSurface
        (PriorProof := PriorProof)
        ctx) :
    ConcreteRawPublicIOVerifierChecks (PriorProof := PriorProof) ctx where
  Statement := surface.Statement
  PublicBoundary := surface.PublicBoundary
  PublicField := surface.PublicField
  TerminalCommittedProof := surface.TerminalCommittedProof
  canonicalStatement := surface.canonicalStatement
  proofStatement := surface.proofStatement
  statementBoundary := surface.statementBoundary
  proofBoundary := surface.proofBoundary
  terminalPublicValues := surface.terminalPublicValues
  boundaryPublicValues := surface.boundaryPublicValues
  terminalCommittedProof := surface.terminalCommittedProof
  statementPublicValid := surface.statementPublicValid
  terminalVerifierPublicIO := fun proof =>
    Option.map
      (fun publicIO => publicIO.raw)
      (surface.terminalVerifierPublicIO proof)
  compactImageReplay := surface.compactImageReplay
  construction2BoundaryReplay := surface.construction2BoundaryReplay
  transcriptReplay := surface.transcriptReplay
  openAuthority := surface.openAuthority

/--
Split F' soundness surface induced by exact public IO plus layout binding.
-/
def rawPublicIOSoundnessSurfaceOfExactPublicIO
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      DirectParentOnlyProductionConcreteFPrimePriorBackend.ConcreteRuntimeExactPublicIOSurface
        (PriorProof := PriorProof)
        ctx)
    (layout : ExactPublicIOLayoutBinding surface) :
    ConcreteRawPublicIOSoundnessSurface (PriorProof := PriorProof) ctx where
  checks := rawPublicIOVerifierChecksOfExactPublicIO surface
  replayBindsProofStatement := by
    intro steps proof image hCompact hBoundaryReplay hTranscript
    exact
      surface.replayBindsProofStatement
        steps
        proof
        image
        hCompact
        hBoundaryReplay
        hTranscript
  rawBoundStatementAuthoritySound := by
    intro steps proof image rawPublicIO hBound
    rcases hBound with ⟨hAccepted, hStatement⟩
    rcases hAccepted with
      ⟨hCompact,
        hBoundaryReplay,
        hTranscript,
        hValid,
        hBoundary,
        hRawPublicIO,
        hRawEq⟩
    have hCompact' :
        surface.compactImageReplay steps proof image := by
      simpa [rawPublicIOVerifierChecksOfExactPublicIO] using hCompact
    have hBoundaryReplay' :
        surface.construction2BoundaryReplay steps proof image := by
      simpa [rawPublicIOVerifierChecksOfExactPublicIO] using hBoundaryReplay
    have hTranscript' :
        surface.transcriptReplay steps proof image := by
      simpa [rawPublicIOVerifierChecksOfExactPublicIO] using hTranscript
    have hStatement' :
        surface.proofStatement proof =
          surface.canonicalStatement steps image := by
      simpa [rawPublicIOVerifierChecksOfExactPublicIO] using hStatement
    have hValid' :
        surface.statementPublicValid
          (surface.canonicalStatement steps image) := by
      simpa [rawPublicIOVerifierChecksOfExactPublicIO] using hValid
    have hBoundary' :
        surface.proofBoundary proof =
          surface.statementBoundary
            (surface.canonicalStatement steps image) := by
      simpa [rawPublicIOVerifierChecksOfExactPublicIO] using hBoundary
    have hRawEq' :
        rawPublicIO =
          surface.terminalPublicValues
            (surface.canonicalStatement steps image) ++
          surface.boundaryPublicValues
            (surface.statementBoundary
              (surface.canonicalStatement steps image)) := by
      simpa [rawPublicIOVerifierChecksOfExactPublicIO] using hRawEq
    cases hPublicIO :
        surface.terminalVerifierPublicIO
          (surface.terminalCommittedProof proof) with
    | none =>
        simp [rawPublicIOVerifierChecksOfExactPublicIO, hPublicIO] at hRawPublicIO
    | some publicIO =>
        have hRawPublicIO' : publicIO.raw = rawPublicIO := by
          simpa [rawPublicIOVerifierChecksOfExactPublicIO, hPublicIO]
            using hRawPublicIO
        have hRawExpected :
            publicIO.raw =
              surface.terminalPublicValues
                (surface.canonicalStatement steps image) ++
              surface.boundaryPublicValues
                (surface.statementBoundary
                  (surface.canonicalStatement steps image)) :=
          hRawPublicIO'.trans hRawEq'
        rcases
          layout.rawOutputBindsTerminalBoundary
            steps
            proof
            image
            publicIO
            hPublicIO
            hRawExpected with
          ⟨hTerminal, hBoundaryValues⟩
        exact
          surface.exactRuntimeSound
            steps
            proof
            image
            publicIO
            hCompact'
            hBoundaryReplay'
            hTranscript'
            hStatement'
            hValid'
            hBoundary'
            hPublicIO
            hTerminal
            hBoundaryValues

/--
Convert the split soundness certificate into the raw public-IO runtime surface.
-/
def runtimeRawPublicIOSurfaceOfSoundness
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ConcreteRawPublicIOSoundnessSurface (PriorProof := PriorProof) ctx) :
    DirectParentOnlyProductionConcreteFPrimePriorRawIO.ConcreteRuntimeRawPublicIOSurface
      (PriorProof := PriorProof)
      ctx where
  Statement := surface.checks.Statement
  PublicBoundary := surface.checks.PublicBoundary
  PublicField := surface.checks.PublicField
  TerminalCommittedProof := surface.checks.TerminalCommittedProof
  canonicalStatement := surface.checks.canonicalStatement
  proofStatement := surface.checks.proofStatement
  statementBoundary := surface.checks.statementBoundary
  proofBoundary := surface.checks.proofBoundary
  terminalPublicValues := surface.checks.terminalPublicValues
  boundaryPublicValues := surface.checks.boundaryPublicValues
  terminalCommittedProof := surface.checks.terminalCommittedProof
  statementPublicValid := surface.checks.statementPublicValid
  terminalVerifierPublicIO := surface.checks.terminalVerifierPublicIO
  compactImageReplay := surface.checks.compactImageReplay
  construction2BoundaryReplay := surface.checks.construction2BoundaryReplay
  transcriptReplay := surface.checks.transcriptReplay
  openAuthority := surface.checks.openAuthority
  replayBindsProofStatement := surface.replayBindsProofStatement
  rawRuntimeSound := by
    intro steps proof image rawPublicIO
      hCompact hBoundaryReplay hTranscript hStatement hValid hBoundary
      hRawPublicIO hRawEq
    exact
      surface.rawBoundStatementAuthoritySound
        steps
        proof
        image
        rawPublicIO
        ⟨⟨hCompact,
            hBoundaryReplay,
            hTranscript,
            hValid,
            hBoundary,
            hRawPublicIO,
            hRawEq⟩,
          hStatement⟩

/-- Runtime verifier predicate induced by the split soundness surface. -/
def RuntimeVerifyPriorOfRawPublicIOSoundness
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ConcreteRawPublicIOSoundnessSurface (PriorProof := PriorProof) ctx) :
    Nat →
      PriorProof →
      PublicImage Digest Boundary →
        Prop :=
  DirectParentOnlyProductionConcreteFPrimePriorRawIO.RuntimeVerifyPriorOfRawPublicIO
    (runtimeRawPublicIOSurfaceOfSoundness surface)

/-- Runtime verifier predicate induced by exact public IO through raw soundness. -/
def RuntimeVerifyPriorOfExactPublicIORawSoundness
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      DirectParentOnlyProductionConcreteFPrimePriorBackend.ConcreteRuntimeExactPublicIOSurface
        (PriorProof := PriorProof)
        ctx)
    (layout : ExactPublicIOLayoutBinding surface) :
    Nat →
      PriorProof →
      PublicImage Digest Boundary →
        Prop :=
  RuntimeVerifyPriorOfRawPublicIOSoundness
    (rawPublicIOSoundnessSurfaceOfExactPublicIO surface layout)

/--
The split-surface adapter preserves the older raw-runtime verifier predicate.
-/
theorem runtimeVerifyPriorOfRawPublicIOSoundness_ofRuntimeRawPublicIO
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      DirectParentOnlyProductionConcreteFPrimePriorRawIO.ConcreteRuntimeRawPublicIOSurface
        (PriorProof := PriorProof)
        ctx) :
    RuntimeVerifyPriorOfRawPublicIOSoundness
      (rawPublicIOSoundnessSurfaceOfRuntimeRawPublicIO surface) =
      DirectParentOnlyProductionConcreteFPrimePriorRawIO.RuntimeVerifyPriorOfRawPublicIO
        surface := by
  funext steps proof image
  rfl

/--
Accepted raw public-vector verification yields a bound statement witness.
-/
theorem runtimeVerifyPriorOfRawPublicIOSoundness_boundStatement
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ConcreteRawPublicIOSoundnessSurface (PriorProof := PriorProof) ctx)
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    (hVerify :
      RuntimeVerifyPriorOfRawPublicIOSoundness
        surface
        steps
        proof
        image) :
    ∃ rawPublicIO : List surface.checks.PublicField,
      RawPublicIOBoundStatementAccepted
        surface.checks
        steps
        proof
        image
        rawPublicIO := by
  rcases hVerify with
    ⟨hCompact,
      hBoundaryReplay,
      hTranscript,
      hValid,
      hBoundary,
      hRawPublicIO⟩
  rcases hRawPublicIO with
    ⟨rawPublicIO, hRawPublicIO, hRawEq⟩
  have hStatement :
      surface.checks.proofStatement proof =
        surface.checks.canonicalStatement steps image :=
    surface.replayBindsProofStatement
      steps
      proof
      image
      hCompact
      hBoundaryReplay
      hTranscript
  exact
      ⟨rawPublicIO,
        ⟨⟨hCompact,
          hBoundaryReplay,
          hTranscript,
          hValid,
          hBoundary,
          hRawPublicIO,
          hRawEq⟩,
        hStatement⟩⟩

/--
Older raw-runtime acceptance also yields a bound canonical statement when
viewed through the split F' surface.
-/
theorem runtimeVerifyPriorOfRawPublicIO_boundStatement
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      DirectParentOnlyProductionConcreteFPrimePriorRawIO.ConcreteRuntimeRawPublicIOSurface
        (PriorProof := PriorProof)
        ctx)
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    (hVerify :
      DirectParentOnlyProductionConcreteFPrimePriorRawIO.RuntimeVerifyPriorOfRawPublicIO
        surface
        steps
        proof
        image) :
    ∃ rawPublicIO : List surface.PublicField,
      RawPublicIOBoundStatementAccepted
        (rawPublicIOVerifierChecksOfRuntimeRawPublicIO surface)
        steps
        proof
        image
        rawPublicIO := by
  have hVerifySplit :
      RuntimeVerifyPriorOfRawPublicIOSoundness
        (rawPublicIOSoundnessSurfaceOfRuntimeRawPublicIO surface)
        steps
        proof
        image := by
    simpa [runtimeVerifyPriorOfRawPublicIOSoundness_ofRuntimeRawPublicIO surface]
      using hVerify
  simpa [rawPublicIOSoundnessSurfaceOfRuntimeRawPublicIO]
    using
      runtimeVerifyPriorOfRawPublicIOSoundness_boundStatement
        (rawPublicIOSoundnessSurfaceOfRuntimeRawPublicIO surface)
        hVerifySplit

/--
Accepted raw public-vector verification opens folded F' authority for the same
public pair.
-/
theorem runtimeVerifyPriorOfRawPublicIOSoundness_acceptedOpens
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ConcreteRawPublicIOSoundnessSurface (PriorProof := PriorProof) ctx) :
    ∀ steps proof image,
      RuntimeVerifyPriorOfRawPublicIOSoundness surface steps proof image →
        ∃ authority : ProofCarryingPriorProof ctx,
          surface.checks.openAuthority proof = some authority ∧
            FoldedFPrimeAuthority.Accepts
              (Transition :=
                DirectParentOnlyProductionSoundness.Transition
                  ctx.toProductionContext)
              (initial := ctx.initial)
              steps
              authority
              image := by
  intro steps proof image hVerify
  rcases
    runtimeVerifyPriorOfRawPublicIOSoundness_boundStatement
      surface
      hVerify with
    ⟨rawPublicIO, hBound⟩
  exact
    surface.rawBoundStatementAuthoritySound
      steps
      proof
      image
      rawPublicIO
      hBound

/-- Certified prior verifier induced by the split raw public-IO surface. -/
def certifiedPriorVerifierOfRawPublicIOSoundness
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ConcreteRawPublicIOSoundnessSurface (PriorProof := PriorProof) ctx) :
    DirectParentOnlyProductionSuperNeoReuseCertifiedVerifier.CertifiedPriorVerifier
      (PriorProof := PriorProof)
      ctx :=
  DirectParentOnlyProductionConcreteFPrimePriorRawIO.certifiedPriorVerifierOfRawPublicIO
    (runtimeRawPublicIOSurfaceOfSoundness surface)

/-- The certified verifier induced by the split surface uses the raw predicate. -/
theorem certifiedPriorVerifierOfRawPublicIOSoundness_verify
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ConcreteRawPublicIOSoundnessSurface (PriorProof := PriorProof) ctx) :
    (certifiedPriorVerifierOfRawPublicIOSoundness surface).verify =
      RuntimeVerifyPriorOfRawPublicIOSoundness surface :=
  rfl

/-- Certified prior verifier induced by exact public IO through split raw soundness. -/
def certifiedPriorVerifierOfExactPublicIORawSoundness
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      DirectParentOnlyProductionConcreteFPrimePriorBackend.ConcreteRuntimeExactPublicIOSurface
        (PriorProof := PriorProof)
        ctx)
    (layout : ExactPublicIOLayoutBinding surface) :
    DirectParentOnlyProductionSuperNeoReuseCertifiedVerifier.CertifiedPriorVerifier
      (PriorProof := PriorProof)
      ctx :=
  certifiedPriorVerifierOfRawPublicIOSoundness
    (rawPublicIOSoundnessSurfaceOfExactPublicIO surface layout)

/-- The exact-public-IO certified verifier uses the exact-through-raw predicate. -/
theorem certifiedPriorVerifierOfExactPublicIORawSoundness_verify
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      DirectParentOnlyProductionConcreteFPrimePriorBackend.ConcreteRuntimeExactPublicIOSurface
        (PriorProof := PriorProof)
        ctx)
    (layout : ExactPublicIOLayoutBinding surface) :
    (certifiedPriorVerifierOfExactPublicIORawSoundness surface layout).verify =
      RuntimeVerifyPriorOfExactPublicIORawSoundness surface layout :=
  rfl

/-- Split raw public-IO acceptance reaches its claimed prior image. -/
theorem runtimeVerifyPriorOfRawPublicIOSoundness_reaches_prior
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ConcreteRawPublicIOSoundnessSurface (PriorProof := PriorProof) ctx)
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    (hVerify :
      RuntimeVerifyPriorOfRawPublicIOSoundness
        surface
        steps
        proof
        image) :
    FPrimeInduction.Reachable
      (DirectParentOnlyProductionSoundness.Transition
        ctx.toProductionContext)
      ctx.initial
      steps
      image := by
  rcases
    runtimeVerifyPriorOfRawPublicIOSoundness_acceptedOpens
      surface
      steps
      proof
      image
      hVerify with
    ⟨authority, _hOpen, hAccepts⟩
  exact
    FoldedFPrimeAuthority.accepts_sound
      (Transition :=
        DirectParentOnlyProductionSoundness.Transition
          ctx.toProductionContext)
      (initial := ctx.initial)
      steps
      authority
      image
      hAccepts

/-- The split raw public-IO prior verifier is same-proof functional. -/
theorem proofFunctionalOfRawPublicIOSoundness
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ConcreteRawPublicIOSoundnessSurface (PriorProof := PriorProof) ctx) :
    CompressedFPrimeAuthority.ProofFunctional
      (RuntimeVerifyPriorOfRawPublicIOSoundness surface) :=
  DirectParentOnlyProductionConcreteFPrimePriorRawIO.proofFunctionalOfRawPublicIO
    (runtimeRawPublicIOSurfaceOfSoundness surface)

/-- Split raw public-IO acceptance cannot authorize an unreachable prior image. -/
theorem runtimeVerifyPriorOfRawPublicIOSoundness_cannot_accept_unreachable_prior
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ConcreteRawPublicIOSoundnessSurface (PriorProof := PriorProof) ctx)
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    (hVerify :
      RuntimeVerifyPriorOfRawPublicIOSoundness
        surface
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
    (runtimeVerifyPriorOfRawPublicIOSoundness_reaches_prior
      surface
      hVerify)

/-- Split raw public-IO acceptance exposes prior public-image invariants. -/
theorem runtimeVerifyPriorOfRawPublicIOSoundness_publicImageInvariants
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ConcreteRawPublicIOSoundnessSurface (PriorProof := PriorProof) ctx)
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    (hVerify :
      RuntimeVerifyPriorOfRawPublicIOSoundness
        surface
        steps
        proof
        image) :
    DirectParentOnlyProductionConcreteFPrimePrior.AcceptedPriorPublicImageInvariants
      ctx
      steps
      image :=
  DirectParentOnlyProductionConcreteFPrimePriorRawIO.runtimeVerifyPriorOfRawPublicIO_publicImageInvariants
    (runtimeRawPublicIOSurfaceOfSoundness surface)
    hVerify

/-- Exact-public-IO acceptance yields a bound raw-vector statement witness. -/
theorem runtimeVerifyPriorOfExactPublicIORawSoundness_boundStatement
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      DirectParentOnlyProductionConcreteFPrimePriorBackend.ConcreteRuntimeExactPublicIOSurface
        (PriorProof := PriorProof)
        ctx)
    (layout : ExactPublicIOLayoutBinding surface)
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    (hVerify :
      RuntimeVerifyPriorOfExactPublicIORawSoundness
        surface
        layout
        steps
        proof
        image) :
    ∃ rawPublicIO : List surface.PublicField,
      RawPublicIOBoundStatementAccepted
        (rawPublicIOVerifierChecksOfExactPublicIO surface)
        steps
        proof
        image
        rawPublicIO :=
  runtimeVerifyPriorOfRawPublicIOSoundness_boundStatement
    (rawPublicIOSoundnessSurfaceOfExactPublicIO surface layout)
    hVerify

/--
Exact-public-IO acceptance opens folded F' authority for the same public pair.
-/
theorem runtimeVerifyPriorOfExactPublicIORawSoundness_acceptedOpens
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      DirectParentOnlyProductionConcreteFPrimePriorBackend.ConcreteRuntimeExactPublicIOSurface
        (PriorProof := PriorProof)
        ctx)
    (layout : ExactPublicIOLayoutBinding surface) :
    ∀ steps proof image,
      RuntimeVerifyPriorOfExactPublicIORawSoundness
        surface
        layout
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
    runtimeVerifyPriorOfRawPublicIOSoundness_acceptedOpens
      (rawPublicIOSoundnessSurfaceOfExactPublicIO surface layout)
      steps
      proof
      image
      hVerify

/-- Exact-public-IO acceptance reaches its claimed prior image. -/
theorem runtimeVerifyPriorOfExactPublicIORawSoundness_reaches_prior
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      DirectParentOnlyProductionConcreteFPrimePriorBackend.ConcreteRuntimeExactPublicIOSurface
        (PriorProof := PriorProof)
        ctx)
    (layout : ExactPublicIOLayoutBinding surface)
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    (hVerify :
      RuntimeVerifyPriorOfExactPublicIORawSoundness
        surface
        layout
        steps
        proof
        image) :
    FPrimeInduction.Reachable
      (DirectParentOnlyProductionSoundness.Transition
        ctx.toProductionContext)
      ctx.initial
      steps
      image :=
  runtimeVerifyPriorOfRawPublicIOSoundness_reaches_prior
    (rawPublicIOSoundnessSurfaceOfExactPublicIO surface layout)
    hVerify

/--
Exact-public-IO acceptance cannot authorize an unreachable prior image.
-/
theorem runtimeVerifyPriorOfExactPublicIORawSoundness_cannot_accept_unreachable_prior
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      DirectParentOnlyProductionConcreteFPrimePriorBackend.ConcreteRuntimeExactPublicIOSurface
        (PriorProof := PriorProof)
        ctx)
    (layout : ExactPublicIOLayoutBinding surface)
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    (hVerify :
      RuntimeVerifyPriorOfExactPublicIORawSoundness
        surface
        layout
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
    (runtimeVerifyPriorOfExactPublicIORawSoundness_reaches_prior
      surface
      layout
      hVerify)

/-- Exact-public-IO acceptance exposes prior public-image invariants. -/
theorem runtimeVerifyPriorOfExactPublicIORawSoundness_publicImageInvariants
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      DirectParentOnlyProductionConcreteFPrimePriorBackend.ConcreteRuntimeExactPublicIOSurface
        (PriorProof := PriorProof)
        ctx)
    (layout : ExactPublicIOLayoutBinding surface)
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    (hVerify :
      RuntimeVerifyPriorOfExactPublicIORawSoundness
        surface
        layout
        steps
        proof
        image) :
    DirectParentOnlyProductionConcreteFPrimePrior.AcceptedPriorPublicImageInvariants
      ctx
      steps
      image :=
  runtimeVerifyPriorOfRawPublicIOSoundness_publicImageInvariants
    (rawPublicIOSoundnessSurfaceOfExactPublicIO surface layout)
    hVerify

/-- The exact-public-IO prior verifier is same-proof functional. -/
theorem proofFunctionalOfExactPublicIORawSoundness
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      DirectParentOnlyProductionConcreteFPrimePriorBackend.ConcreteRuntimeExactPublicIOSurface
        (PriorProof := PriorProof)
        ctx)
    (layout : ExactPublicIOLayoutBinding surface) :
    CompressedFPrimeAuthority.ProofFunctional
      (RuntimeVerifyPriorOfExactPublicIORawSoundness surface layout) :=
  proofFunctionalOfRawPublicIOSoundness
    (rawPublicIOSoundnessSurfaceOfExactPublicIO surface layout)

/--
Split raw public-IO prior-plus-latest end-to-end theorem.
-/
theorem certifiedSingleTerminalEndToEnd_ofRawPublicIOSoundnessLatestStep
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ConcreteRawPublicIOSoundnessSurface (PriorProof := PriorProof) ctx)
    {priorSteps : Nat}
    {priorProof : PriorProof}
    {priorImage nextImage : PublicImage Digest Boundary}
    {latestProof : Unit}
    (hPrior :
      RuntimeVerifyPriorOfRawPublicIOSoundness
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
    DirectParentOnlyProductionSuperNeoReuseEndToEnd.CertifiedTerminalEndToEnd
      ctx
      (certifiedPriorVerifierOfRawPublicIOSoundness surface).opening
      priorSteps
      priorSteps
      priorProof
      priorImage
      priorImage
      nextImage
      nextImage :=
  DirectParentOnlyProductionConcreteFPrimePriorRawIO.certifiedSingleTerminalEndToEnd_ofRawPublicIOLatestStep
    (runtimeRawPublicIOSurfaceOfSoundness surface)
    hPrior
    hLatest

/--
Split raw public-IO projection to the non-aggregate private DEC and stage facts.
-/
theorem nonAggregatePrivateDecStageFacts_ofRawPublicIOSoundnessLatestStep
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ConcreteRawPublicIOSoundnessSurface (PriorProof := PriorProof) ctx)
    {priorSteps : Nat}
    {priorProof : PriorProof}
    {priorImage nextImage : PublicImage Digest Boundary}
    {latestProof : Unit}
    (hPrior :
      RuntimeVerifyPriorOfRawPublicIOSoundness
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
  DirectParentOnlyProductionConcreteFPrimePriorRawIO.nonAggregatePrivateDecStageFacts_ofRawPublicIOLatestStep
    (runtimeRawPublicIOSurfaceOfSoundness surface)
    hPrior
    hLatest

/--
Split raw public-IO projection to the Section 7.1 owner-target stage audit.
-/
theorem section71StageTargetAuditTrail_ofRawPublicIOSoundnessLatestStep
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ConcreteRawPublicIOSoundnessSurface (PriorProof := PriorProof) ctx)
    {priorSteps : Nat}
    {priorProof : PriorProof}
    {priorImage nextImage : PublicImage Digest Boundary}
    {latestProof : Unit}
    (hPrior :
      RuntimeVerifyPriorOfRawPublicIOSoundness
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
  DirectParentOnlyProductionConcreteFPrimePriorRawIO.section71StageTargetAuditTrail_ofRawPublicIOLatestStep
    (runtimeRawPublicIOSurfaceOfSoundness surface)
    hPrior
    hLatest

/--
Exact-public-IO prior-plus-latest theorem through the split raw public-IO
soundness surface.
-/
theorem certifiedSingleTerminalEndToEnd_ofExactPublicIORawSoundnessLatestStep
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      DirectParentOnlyProductionConcreteFPrimePriorBackend.ConcreteRuntimeExactPublicIOSurface
        (PriorProof := PriorProof)
        ctx)
    (layout : ExactPublicIOLayoutBinding surface)
    {priorSteps : Nat}
    {priorProof : PriorProof}
    {priorImage nextImage : PublicImage Digest Boundary}
    {latestProof : Unit}
    (hPrior :
      RuntimeVerifyPriorOfExactPublicIORawSoundness
        surface
        layout
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
      (certifiedPriorVerifierOfRawPublicIOSoundness
        (rawPublicIOSoundnessSurfaceOfExactPublicIO surface layout)).opening
      priorSteps
      priorSteps
      priorProof
      priorImage
      priorImage
      nextImage
      nextImage :=
  certifiedSingleTerminalEndToEnd_ofRawPublicIOSoundnessLatestStep
    (rawPublicIOSoundnessSurfaceOfExactPublicIO surface layout)
    hPrior
    hLatest

/--
Exact-public-IO projection to non-aggregate private DEC and stage facts through
the split raw public-IO soundness surface.
-/
theorem nonAggregatePrivateDecStageFacts_ofExactPublicIORawSoundnessLatestStep
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      DirectParentOnlyProductionConcreteFPrimePriorBackend.ConcreteRuntimeExactPublicIOSurface
        (PriorProof := PriorProof)
        ctx)
    (layout : ExactPublicIOLayoutBinding surface)
    {priorSteps : Nat}
    {priorProof : PriorProof}
    {priorImage nextImage : PublicImage Digest Boundary}
    {latestProof : Unit}
    (hPrior :
      RuntimeVerifyPriorOfExactPublicIORawSoundness
        surface
        layout
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
  nonAggregatePrivateDecStageFacts_ofRawPublicIOSoundnessLatestStep
    (rawPublicIOSoundnessSurfaceOfExactPublicIO surface layout)
    hPrior
    hLatest

/--
Exact-public-IO projection to the Section 7.1 owner-target stage audit through
the split raw public-IO soundness surface.
-/
theorem section71StageTargetAuditTrail_ofExactPublicIORawSoundnessLatestStep
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      DirectParentOnlyProductionConcreteFPrimePriorBackend.ConcreteRuntimeExactPublicIOSurface
        (PriorProof := PriorProof)
        ctx)
    (layout : ExactPublicIOLayoutBinding surface)
    {priorSteps : Nat}
    {priorProof : PriorProof}
    {priorImage nextImage : PublicImage Digest Boundary}
    {latestProof : Unit}
    (hPrior :
      RuntimeVerifyPriorOfExactPublicIORawSoundness
        surface
        layout
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
  section71StageTargetAuditTrail_ofRawPublicIOSoundnessLatestStep
    (rawPublicIOSoundnessSurfaceOfExactPublicIO surface layout)
    hPrior
    hLatest

/--
Older raw-runtime prior-plus-latest theorem through the split raw public-IO
soundness surface.
-/
theorem certifiedSingleTerminalEndToEnd_ofRuntimeRawPublicIOSoundnessLatestStep
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      DirectParentOnlyProductionConcreteFPrimePriorRawIO.ConcreteRuntimeRawPublicIOSurface
        (PriorProof := PriorProof)
        ctx)
    {priorSteps : Nat}
    {priorProof : PriorProof}
    {priorImage nextImage : PublicImage Digest Boundary}
    {latestProof : Unit}
    (hPrior :
      DirectParentOnlyProductionConcreteFPrimePriorRawIO.RuntimeVerifyPriorOfRawPublicIO
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
    DirectParentOnlyProductionSuperNeoReuseEndToEnd.CertifiedTerminalEndToEnd
      ctx
      (certifiedPriorVerifierOfRawPublicIOSoundness
        (rawPublicIOSoundnessSurfaceOfRuntimeRawPublicIO surface)).opening
      priorSteps
      priorSteps
      priorProof
      priorImage
      priorImage
      nextImage
      nextImage := by
  have hPriorSplit :
      RuntimeVerifyPriorOfRawPublicIOSoundness
        (rawPublicIOSoundnessSurfaceOfRuntimeRawPublicIO surface)
        priorSteps
        priorProof
        priorImage := by
    simpa [runtimeVerifyPriorOfRawPublicIOSoundness_ofRuntimeRawPublicIO surface]
      using hPrior
  exact
    certifiedSingleTerminalEndToEnd_ofRawPublicIOSoundnessLatestStep
      (rawPublicIOSoundnessSurfaceOfRuntimeRawPublicIO surface)
      hPriorSplit
      hLatest

/--
Older raw-runtime projection to non-aggregate private DEC and stage facts
through the split raw public-IO soundness surface.
-/
theorem nonAggregatePrivateDecStageFacts_ofRuntimeRawPublicIOSoundnessLatestStep
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      DirectParentOnlyProductionConcreteFPrimePriorRawIO.ConcreteRuntimeRawPublicIOSurface
        (PriorProof := PriorProof)
        ctx)
    {priorSteps : Nat}
    {priorProof : PriorProof}
    {priorImage nextImage : PublicImage Digest Boundary}
    {latestProof : Unit}
    (hPrior :
      DirectParentOnlyProductionConcreteFPrimePriorRawIO.RuntimeVerifyPriorOfRawPublicIO
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
      nextImage := by
  have hPriorSplit :
      RuntimeVerifyPriorOfRawPublicIOSoundness
        (rawPublicIOSoundnessSurfaceOfRuntimeRawPublicIO surface)
        priorSteps
        priorProof
        priorImage := by
    simpa [runtimeVerifyPriorOfRawPublicIOSoundness_ofRuntimeRawPublicIO surface]
      using hPrior
  exact
    nonAggregatePrivateDecStageFacts_ofRawPublicIOSoundnessLatestStep
      (rawPublicIOSoundnessSurfaceOfRuntimeRawPublicIO surface)
      hPriorSplit
      hLatest

/--
Older raw-runtime projection to the Section 7.1 owner-target stage audit
through the split raw public-IO soundness surface.
-/
theorem section71StageTargetAuditTrail_ofRuntimeRawPublicIOSoundnessLatestStep
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      DirectParentOnlyProductionConcreteFPrimePriorRawIO.ConcreteRuntimeRawPublicIOSurface
        (PriorProof := PriorProof)
        ctx)
    {priorSteps : Nat}
    {priorProof : PriorProof}
    {priorImage nextImage : PublicImage Digest Boundary}
    {latestProof : Unit}
    (hPrior :
      DirectParentOnlyProductionConcreteFPrimePriorRawIO.RuntimeVerifyPriorOfRawPublicIO
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
      nextImage := by
  have hPriorSplit :
      RuntimeVerifyPriorOfRawPublicIOSoundness
        (rawPublicIOSoundnessSurfaceOfRuntimeRawPublicIO surface)
        priorSteps
        priorProof
        priorImage := by
    simpa [runtimeVerifyPriorOfRawPublicIOSoundness_ofRuntimeRawPublicIO surface]
      using hPrior
  exact
    section71StageTargetAuditTrail_ofRawPublicIOSoundnessLatestStep
      (rawPublicIOSoundnessSurfaceOfRuntimeRawPublicIO surface)
      hPriorSplit
      hLatest

end DirectParentOnlyProductionConcreteFPrimePriorRawIOSoundness

end DirectCcsFPrime
