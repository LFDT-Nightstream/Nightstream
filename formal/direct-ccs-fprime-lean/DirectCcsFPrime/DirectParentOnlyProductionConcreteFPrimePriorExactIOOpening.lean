import DirectCcsFPrime.DirectParentOnlyProductionConcreteFPrimePriorRawIOOpening

/-!
Opening-level certificate for the exact public-IO prior F' verifier.

This module keeps the production exact-public-IO shape while avoiding the older
monolithic backend soundness field. The trusted cryptographic boundary is split
into exact statement opening and opened-authority binding. Lean proves the raw
public-vector bridge, then packages the resulting certified prior verifier.
-/

namespace DirectCcsFPrime

namespace DirectParentOnlyProductionConcreteFPrimePriorExactIOOpening

/-- Section 7.1-backed production context. -/
abbrev ProductionContext :=
  DirectParentOnlyProductionConcreteFPrimePriorRawIOOpening.ProductionContext

/-- Public image for the parent-only production path. -/
abbrev PublicImage :=
  DirectParentOnlyProductionConcreteFPrimePriorRawIOOpening.PublicImage

/-- Proof-carrying folded prior authority for the induced production context. -/
abbrev ProofCarryingPriorProof :=
  @DirectParentOnlyProductionConcreteFPrimePriorRawIOOpening.ProofCarryingPriorProof

/-- Structured terminal/boundary public IO returned by the production verifier. -/
abbrev ExactTerminalBoundaryPublicIO :=
  DirectParentOnlyProductionConcreteFPrimePriorBackend.ExactTerminalBoundaryPublicIO

/-- Verifier-visible raw public-vector checks. -/
abbrev ConcreteRawPublicIOVerifierChecks :=
  @DirectParentOnlyProductionConcreteFPrimePriorRawIOOpening.ConcreteRawPublicIOVerifierChecks

/-- Bound raw public-vector acceptance for the canonical statement. -/
abbrev RawPublicIOBoundStatementAccepted :=
  @DirectParentOnlyProductionConcreteFPrimePriorRawIOOpening.RawPublicIOBoundStatementAccepted

/-- Opening-level raw public-IO surface. -/
abbrev ConcreteRawPublicIOOpeningSurface :=
  @DirectParentOnlyProductionConcreteFPrimePriorRawIOOpening.ConcreteRawPublicIOOpeningSurface

/-- Split raw public-IO soundness surface. -/
abbrev ConcreteRawPublicIOSoundnessSurface :=
  @DirectParentOnlyProductionConcreteFPrimePriorRawIOOpening.ConcreteRawPublicIOSoundnessSurface

/--
Verifier-visible exact public-IO checks, without authority soundness.

This is the production verifier surface: replay checks, statement/public
boundary checks, the terminal committed proof public IO, and the fixed authority
opener. No field here says that accepted proofs are sound.
-/
structure ConcreteExactPublicIOVerifierChecks
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
      Option (ExactTerminalBoundaryPublicIO PublicField)
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
Project exact public IO to raw public-vector checks.
-/
def rawPublicIOVerifierChecksOfExactPublicIOOpening
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (checks :
      ConcreteExactPublicIOVerifierChecks (PriorProof := PriorProof) ctx) :
    ConcreteRawPublicIOVerifierChecks (PriorProof := PriorProof) ctx where
  Statement := checks.Statement
  PublicBoundary := checks.PublicBoundary
  PublicField := checks.PublicField
  TerminalCommittedProof := checks.TerminalCommittedProof
  canonicalStatement := checks.canonicalStatement
  proofStatement := checks.proofStatement
  statementBoundary := checks.statementBoundary
  proofBoundary := checks.proofBoundary
  terminalPublicValues := checks.terminalPublicValues
  boundaryPublicValues := checks.boundaryPublicValues
  terminalCommittedProof := checks.terminalCommittedProof
  statementPublicValid := checks.statementPublicValid
  terminalVerifierPublicIO := fun proof =>
    Option.map
      (fun publicIO => publicIO.raw)
      (checks.terminalVerifierPublicIO proof)
  compactImageReplay := checks.compactImageReplay
  construction2BoundaryReplay := checks.construction2BoundaryReplay
  transcriptReplay := checks.transcriptReplay
  openAuthority := checks.openAuthority

/--
Verifier-visible exact public-IO acceptance for a chosen public-IO object.
-/
def ExactPublicIOVerifierAccepted
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (checks :
      ConcreteExactPublicIOVerifierChecks (PriorProof := PriorProof) ctx)
    (steps : Nat)
    (proof : PriorProof)
    (image : PublicImage Digest Boundary)
    (publicIO : ExactTerminalBoundaryPublicIO checks.PublicField) : Prop :=
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
        some publicIO ∧
    publicIO.terminal =
      checks.terminalPublicValues
        (checks.canonicalStatement steps image) ∧
    publicIO.boundary =
      checks.boundaryPublicValues
        (checks.statementBoundary
          (checks.canonicalStatement steps image))

/--
Exact public-IO acceptance after replay binds the opaque proof statement to the
canonical statement for the same `(steps, image)` pair.
-/
def ExactPublicIOBoundStatementAccepted
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (checks :
      ConcreteExactPublicIOVerifierChecks (PriorProof := PriorProof) ctx)
    (steps : Nat)
    (proof : PriorProof)
    (image : PublicImage Digest Boundary)
    (publicIO : ExactTerminalBoundaryPublicIO checks.PublicField) : Prop :=
  ExactPublicIOVerifierAccepted checks steps proof image publicIO ∧
    checks.proofStatement proof =
      checks.canonicalStatement steps image

/--
Implementation-facing terminal-length discipline for exact public IO.

Once the verifier raw vector equals the canonical terminal/boundary
concatenation, the exposed terminal slice has the canonical terminal length.
Together with `raw = terminal ++ boundary`, this forces exact terminal and
boundary values, not an aggregate property.
-/
structure ExactPublicIOTerminalLengthBinding
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (checks :
      ConcreteExactPublicIOVerifierChecks (PriorProof := PriorProof) ctx) where
  terminalLengthBindsExpected :
    ∀ steps proof image publicIO,
      checks.terminalVerifierPublicIO
        (checks.terminalCommittedProof proof) =
          some publicIO →
      publicIO.raw =
        checks.terminalPublicValues
          (checks.canonicalStatement steps image) ++
        checks.boundaryPublicValues
          (checks.statementBoundary
            (checks.canonicalStatement steps image)) →
      publicIO.terminal.length =
        (checks.terminalPublicValues
          (checks.canonicalStatement steps image)).length

/--
Canonical terminal-slice discipline for implementations that expose the
terminal segment as the canonical-length prefix of the raw public vector.
-/
structure ExactPublicIOTerminalSliceBinding
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (checks :
      ConcreteExactPublicIOVerifierChecks (PriorProof := PriorProof) ctx) where
  terminalSliceMatchesCanonicalPrefix :
    ∀ steps proof image publicIO,
      checks.terminalVerifierPublicIO
        (checks.terminalCommittedProof proof) =
          some publicIO →
      publicIO.raw =
        checks.terminalPublicValues
          (checks.canonicalStatement steps image) ++
        checks.boundaryPublicValues
          (checks.statementBoundary
            (checks.canonicalStatement steps image)) →
      publicIO.terminal =
        publicIO.raw.take
          (checks.terminalPublicValues
            (checks.canonicalStatement steps image)).length

/-- Canonical terminal-slice discipline induces terminal-length discipline. -/
def exactPublicIOTerminalLengthBindingOfTerminalSlice
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (checks :
      ConcreteExactPublicIOVerifierChecks (PriorProof := PriorProof) ctx)
    (sliceBinding : ExactPublicIOTerminalSliceBinding checks) :
    ExactPublicIOTerminalLengthBinding checks where
  terminalLengthBindsExpected := by
    intro steps proof image publicIO hPublicIO hRaw
    let expectedTerminal :=
      checks.terminalPublicValues
        (checks.canonicalStatement steps image)
    let expectedBoundary :=
      checks.boundaryPublicValues
        (checks.statementBoundary
          (checks.canonicalStatement steps image))
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
Raw bound-statement acceptance determines exact terminal and boundary public IO.
-/
theorem exactPublicIOBoundStatementAccepted_ofRawBoundStatement
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (checks :
      ConcreteExactPublicIOVerifierChecks (PriorProof := PriorProof) ctx)
    (lengthBinding : ExactPublicIOTerminalLengthBinding checks)
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    {rawPublicIO :
      List (rawPublicIOVerifierChecksOfExactPublicIOOpening checks).PublicField}
    (hBound :
      RawPublicIOBoundStatementAccepted
        (rawPublicIOVerifierChecksOfExactPublicIOOpening checks)
        steps
        proof
        image
        rawPublicIO) :
    ∃ publicIO : ExactTerminalBoundaryPublicIO checks.PublicField,
      ExactPublicIOBoundStatementAccepted
        checks
        steps
        proof
        image
        publicIO := by
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
      checks.compactImageReplay steps proof image := by
    simpa [rawPublicIOVerifierChecksOfExactPublicIOOpening] using hCompact
  have hBoundaryReplay' :
      checks.construction2BoundaryReplay steps proof image := by
    simpa [rawPublicIOVerifierChecksOfExactPublicIOOpening] using
      hBoundaryReplay
  have hTranscript' :
      checks.transcriptReplay steps proof image := by
    simpa [rawPublicIOVerifierChecksOfExactPublicIOOpening] using hTranscript
  have hStatement' :
      checks.proofStatement proof =
        checks.canonicalStatement steps image := by
    simpa [rawPublicIOVerifierChecksOfExactPublicIOOpening] using hStatement
  have hValid' :
      checks.statementPublicValid
        (checks.canonicalStatement steps image) := by
    simpa [rawPublicIOVerifierChecksOfExactPublicIOOpening] using hValid
  have hBoundary' :
      checks.proofBoundary proof =
        checks.statementBoundary
          (checks.canonicalStatement steps image) := by
    simpa [rawPublicIOVerifierChecksOfExactPublicIOOpening] using hBoundary
  have hRawEq' :
      rawPublicIO =
        checks.terminalPublicValues
          (checks.canonicalStatement steps image) ++
        checks.boundaryPublicValues
          (checks.statementBoundary
            (checks.canonicalStatement steps image)) := by
    simpa [rawPublicIOVerifierChecksOfExactPublicIOOpening] using hRawEq
  cases hPublicIO :
      checks.terminalVerifierPublicIO
        (checks.terminalCommittedProof proof) with
  | none =>
      simp [rawPublicIOVerifierChecksOfExactPublicIOOpening, hPublicIO] at hRawPublicIO
  | some publicIO =>
      have hRawPublicIO' : publicIO.raw = rawPublicIO := by
        simpa [rawPublicIOVerifierChecksOfExactPublicIOOpening, hPublicIO]
          using hRawPublicIO
      have hRawExpected :
          publicIO.raw =
            checks.terminalPublicValues
              (checks.canonicalStatement steps image) ++
            checks.boundaryPublicValues
              (checks.statementBoundary
                (checks.canonicalStatement steps image)) :=
        hRawPublicIO'.trans hRawEq'
      let expectedTerminal :=
        checks.terminalPublicValues
          (checks.canonicalStatement steps image)
      let expectedBoundary :=
        checks.boundaryPublicValues
          (checks.statementBoundary
            (checks.canonicalStatement steps image))
      have hAppend :
          publicIO.terminal ++ publicIO.boundary =
            expectedTerminal ++ expectedBoundary := by
        calc
          publicIO.terminal ++ publicIO.boundary = publicIO.raw :=
            publicIO.raw_eq.symm
          _ = expectedTerminal ++ expectedBoundary := by
            simpa [expectedTerminal, expectedBoundary] using hRawExpected
      have hLength :
          publicIO.terminal.length = expectedTerminal.length := by
        simpa [expectedTerminal, expectedBoundary] using
          lengthBinding.terminalLengthBindsExpected
            steps
            proof
            image
            publicIO
            hPublicIO
            hRawExpected
      have hTerminal : publicIO.terminal = expectedTerminal := by
        have hTake :=
          congrArg (fun xs => xs.take publicIO.terminal.length) hAppend
        simpa [expectedTerminal, expectedBoundary, hLength] using hTake
      have hBoundaryValues : publicIO.boundary = expectedBoundary := by
        have hDrop :=
          congrArg (fun xs => xs.drop publicIO.terminal.length) hAppend
        simpa [expectedTerminal, expectedBoundary, hLength] using hDrop
      exact
        ⟨publicIO,
          ⟨⟨hCompact',
            hBoundaryReplay',
            hTranscript',
            hValid',
            hBoundary',
            hPublicIO,
            hTerminal,
            hBoundaryValues⟩,
          hStatement'⟩⟩

/--
Opening-level certificate for exact public-IO authority.

The trusted cryptographic obligations are stated over exact bound statements:
accepted exact public IO must open through the fixed opener, and the opened
authority must bind the same `(steps, image)` pair.
-/
structure ConcreteExactPublicIOOpeningSurface
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : ProductionContext Digest Boundary n params) where
  checks :
    ConcreteExactPublicIOVerifierChecks (PriorProof := PriorProof) ctx
  terminalLengthBinding :
    ExactPublicIOTerminalLengthBinding checks
  replayBindsProofStatement :
    ∀ steps proof image,
      checks.compactImageReplay steps proof image →
      checks.construction2BoundaryReplay steps proof image →
      checks.transcriptReplay steps proof image →
        checks.proofStatement proof =
          checks.canonicalStatement steps image
  exactBoundStatementOpens :
    ∀ steps proof image publicIO,
      ExactPublicIOBoundStatementAccepted
        checks
        steps
        proof
        image
        publicIO →
        ∃ authority : ProofCarryingPriorProof ctx,
          checks.openAuthority proof = some authority
  openedAuthorityBindsExactStatement :
    ∀ steps proof image publicIO authority,
      ExactPublicIOBoundStatementAccepted
        checks
        steps
        proof
        image
        publicIO →
      checks.openAuthority proof = some authority →
        authority.steps = steps ∧ authority.image = image

/--
Opening-level exact public-IO evidence derives folded F' authority.
-/
theorem exactBoundStatementAuthoritySound_ofOpeningSurface
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ConcreteExactPublicIOOpeningSurface (PriorProof := PriorProof) ctx)
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    {publicIO : ExactTerminalBoundaryPublicIO surface.checks.PublicField}
    (hBound :
      ExactPublicIOBoundStatementAccepted
        surface.checks
        steps
        proof
        image
        publicIO) :
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
  rcases
    surface.exactBoundStatementOpens
      steps
      proof
      image
      publicIO
      hBound with
    ⟨authority, hOpen⟩
  rcases
    surface.openedAuthorityBindsExactStatement
      steps
      proof
      image
      publicIO
      authority
      hBound
      hOpen with
    ⟨hSteps, hImage⟩
  exact ⟨authority, hOpen, ⟨hSteps, hImage⟩⟩

/--
Exact opening-level evidence instantiates the raw public-IO opening surface.
-/
def rawPublicIOOpeningSurfaceOfExactPublicIOOpening
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ConcreteExactPublicIOOpeningSurface (PriorProof := PriorProof) ctx) :
    ConcreteRawPublicIOOpeningSurface (PriorProof := PriorProof) ctx where
  checks := rawPublicIOVerifierChecksOfExactPublicIOOpening surface.checks
  replayBindsProofStatement := by
    intro steps proof image hCompact hBoundaryReplay hTranscript
    exact
      surface.replayBindsProofStatement
        steps
        proof
        image
        (by
          simpa [rawPublicIOVerifierChecksOfExactPublicIOOpening] using
            hCompact)
        (by
          simpa [rawPublicIOVerifierChecksOfExactPublicIOOpening] using
            hBoundaryReplay)
        (by
          simpa [rawPublicIOVerifierChecksOfExactPublicIOOpening] using
            hTranscript)
  rawBoundStatementOpens := by
    intro steps proof image rawPublicIO hBound
    rcases
      exactPublicIOBoundStatementAccepted_ofRawBoundStatement
        surface.checks
        surface.terminalLengthBinding
        hBound with
      ⟨publicIO, hExactBound⟩
    rcases
      surface.exactBoundStatementOpens
        steps
        proof
        image
        publicIO
        hExactBound with
      ⟨authority, hOpen⟩
    exact
      ⟨authority,
        by
          simpa [rawPublicIOVerifierChecksOfExactPublicIOOpening] using
            hOpen⟩
  openedAuthorityBindsBoundStatement := by
    intro steps proof image rawPublicIO authority hBound hOpen
    rcases
      exactPublicIOBoundStatementAccepted_ofRawBoundStatement
        surface.checks
        surface.terminalLengthBinding
        hBound with
      ⟨publicIO, hExactBound⟩
    exact
      surface.openedAuthorityBindsExactStatement
        steps
        proof
        image
        publicIO
        authority
        hExactBound
        (by
          simpa [rawPublicIOVerifierChecksOfExactPublicIOOpening] using
            hOpen)

/--
Exact opening-level evidence instantiates the split raw public-IO soundness
surface.
-/
def rawPublicIOSoundnessSurfaceOfExactPublicIOOpening
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ConcreteExactPublicIOOpeningSurface (PriorProof := PriorProof) ctx) :
    ConcreteRawPublicIOSoundnessSurface (PriorProof := PriorProof) ctx :=
  DirectParentOnlyProductionConcreteFPrimePriorRawIOOpening.rawPublicIOSoundnessSurfaceOfOpening
    (rawPublicIOOpeningSurfaceOfExactPublicIOOpening surface)

/-- Runtime verifier predicate induced by opening-level exact public IO. -/
def RuntimeVerifyPriorOfExactPublicIOOpening
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ConcreteExactPublicIOOpeningSurface (PriorProof := PriorProof) ctx) :
    Nat →
      PriorProof →
      PublicImage Digest Boundary →
        Prop :=
  DirectParentOnlyProductionConcreteFPrimePriorRawIOOpening.RuntimeVerifyPriorOfRawPublicIOOpening
    (rawPublicIOOpeningSurfaceOfExactPublicIOOpening surface)

/-- Certified prior verifier induced by opening-level exact public IO. -/
def certifiedPriorVerifierOfExactPublicIOOpening
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ConcreteExactPublicIOOpeningSurface (PriorProof := PriorProof) ctx) :
    DirectParentOnlyProductionSuperNeoReuseCertifiedVerifier.CertifiedPriorVerifier
      (PriorProof := PriorProof)
      ctx :=
  DirectParentOnlyProductionConcreteFPrimePriorRawIOOpening.certifiedPriorVerifierOfRawPublicIOOpening
    (rawPublicIOOpeningSurfaceOfExactPublicIOOpening surface)

/-- The exact-opening certified verifier uses the exact-opening predicate. -/
theorem certifiedPriorVerifierOfExactPublicIOOpening_verify
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ConcreteExactPublicIOOpeningSurface (PriorProof := PriorProof) ctx) :
    (certifiedPriorVerifierOfExactPublicIOOpening surface).verify =
      RuntimeVerifyPriorOfExactPublicIOOpening surface :=
  rfl

/--
Exact-opening acceptance yields a bound raw-vector statement witness.
-/
theorem runtimeVerifyPriorOfExactPublicIOOpening_boundStatement
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ConcreteExactPublicIOOpeningSurface (PriorProof := PriorProof) ctx)
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    (hVerify :
      RuntimeVerifyPriorOfExactPublicIOOpening
        surface
        steps
        proof
        image) :
    ∃ rawPublicIO :
      List
        (rawPublicIOVerifierChecksOfExactPublicIOOpening
          surface.checks).PublicField,
      RawPublicIOBoundStatementAccepted
        (rawPublicIOVerifierChecksOfExactPublicIOOpening
          surface.checks)
        steps
        proof
        image
        rawPublicIO :=
  DirectParentOnlyProductionConcreteFPrimePriorRawIOSoundness.runtimeVerifyPriorOfRawPublicIOSoundness_boundStatement
    (rawPublicIOSoundnessSurfaceOfExactPublicIOOpening surface)
    hVerify

/--
Exact-opening acceptance opens folded F' authority for the same public pair.
-/
theorem runtimeVerifyPriorOfExactPublicIOOpening_acceptedOpens
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ConcreteExactPublicIOOpeningSurface (PriorProof := PriorProof) ctx) :
    ∀ steps proof image,
      RuntimeVerifyPriorOfExactPublicIOOpening
        surface
        steps
        proof
        image →
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
    DirectParentOnlyProductionConcreteFPrimePriorRawIOOpening.runtimeVerifyPriorOfRawPublicIOOpening_acceptedOpens
      (rawPublicIOOpeningSurfaceOfExactPublicIOOpening surface)
      steps
      proof
      image
      hVerify with
    ⟨authority, hOpen, hAccepts⟩
  exact
    ⟨authority,
      by
        simpa [rawPublicIOVerifierChecksOfExactPublicIOOpening] using hOpen,
      hAccepts⟩

/--
If the fixed opener returns a concrete authority for an accepted proof, that
authority accepts the same `(steps, image)` pair.
-/
theorem runtimeVerifyPriorOfExactPublicIOOpening_openedAuthority_accepts_of_open
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ConcreteExactPublicIOOpeningSurface (PriorProof := PriorProof) ctx)
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    {authority : ProofCarryingPriorProof ctx}
    (hVerify :
      RuntimeVerifyPriorOfExactPublicIOOpening
        surface
        steps
        proof
        image)
    (hOpen : surface.checks.openAuthority proof = some authority) :
    FoldedFPrimeAuthority.Accepts
      (Transition :=
        DirectParentOnlyProductionSoundness.Transition
          ctx.toProductionContext)
      (initial := ctx.initial)
      steps
      authority
      image := by
  rcases
    runtimeVerifyPriorOfExactPublicIOOpening_acceptedOpens
      surface
      steps
      proof
      image
      hVerify with
    ⟨openedAuthority, hOpened, hAccepts⟩
  have hAuthority : openedAuthority = authority := by
    have hSome : some openedAuthority = some authority :=
      hOpened.symm.trans hOpen
    cases hSome
    rfl
  cases hAuthority
  exact hAccepts

/-- Exact-opening acceptance reaches its claimed prior image. -/
theorem runtimeVerifyPriorOfExactPublicIOOpening_reaches_prior
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ConcreteExactPublicIOOpeningSurface (PriorProof := PriorProof) ctx)
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    (hVerify :
      RuntimeVerifyPriorOfExactPublicIOOpening surface steps proof image) :
    FPrimeInduction.Reachable
      (DirectParentOnlyProductionSoundness.Transition
        ctx.toProductionContext)
      ctx.initial
      steps
      image :=
  DirectParentOnlyProductionConcreteFPrimePriorRawIOOpening.runtimeVerifyPriorOfRawPublicIOOpening_reaches_prior
    (rawPublicIOOpeningSurfaceOfExactPublicIOOpening surface)
    hVerify

/-- The exact-opening prior verifier is same-proof functional. -/
theorem proofFunctionalOfExactPublicIOOpening
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ConcreteExactPublicIOOpeningSurface (PriorProof := PriorProof) ctx) :
    CompressedFPrimeAuthority.ProofFunctional
      (RuntimeVerifyPriorOfExactPublicIOOpening surface) :=
  DirectParentOnlyProductionConcreteFPrimePriorRawIOOpening.proofFunctionalOfRawPublicIOOpening
    (rawPublicIOOpeningSurfaceOfExactPublicIOOpening surface)

/-- Exact-opening acceptance exposes prior public-image invariants. -/
theorem runtimeVerifyPriorOfExactPublicIOOpening_publicImageInvariants
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ConcreteExactPublicIOOpeningSurface (PriorProof := PriorProof) ctx)
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    (hVerify :
      RuntimeVerifyPriorOfExactPublicIOOpening surface steps proof image) :
    DirectParentOnlyProductionConcreteFPrimePrior.AcceptedPriorPublicImageInvariants
      ctx
      steps
      image :=
  DirectParentOnlyProductionConcreteFPrimePriorRawIOOpening.runtimeVerifyPriorOfRawPublicIOOpening_publicImageInvariants
    (rawPublicIOOpeningSurfaceOfExactPublicIOOpening surface)
    hVerify

/--
Exact-opening acceptance cannot authorize an unreachable prior image.
-/
theorem runtimeVerifyPriorOfExactPublicIOOpening_cannot_accept_unreachable_prior
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ConcreteExactPublicIOOpeningSurface (PriorProof := PriorProof) ctx)
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    (hVerify :
      RuntimeVerifyPriorOfExactPublicIOOpening surface steps proof image)
    (hUnreachable :
      ¬ FPrimeInduction.Reachable
        (DirectParentOnlyProductionSoundness.Transition
          ctx.toProductionContext)
        ctx.initial
        steps
        image) :
    False :=
  hUnreachable
    (runtimeVerifyPriorOfExactPublicIOOpening_reaches_prior
      surface
      hVerify)

/--
Exact-opening prior-plus-latest end-to-end theorem.
-/
theorem certifiedSingleTerminalEndToEnd_ofExactPublicIOOpeningLatestStep
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ConcreteExactPublicIOOpeningSurface (PriorProof := PriorProof) ctx)
    {priorSteps : Nat}
    {priorProof : PriorProof}
    {priorImage nextImage : PublicImage Digest Boundary}
    {latestProof : Unit}
    (hPrior :
      RuntimeVerifyPriorOfExactPublicIOOpening
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
      (certifiedPriorVerifierOfExactPublicIOOpening surface).opening
      priorSteps
      priorSteps
      priorProof
      priorImage
      priorImage
      nextImage
      nextImage :=
  DirectParentOnlyProductionConcreteFPrimePriorRawIOSoundness.certifiedSingleTerminalEndToEnd_ofRawPublicIOSoundnessLatestStep
    (rawPublicIOSoundnessSurfaceOfExactPublicIOOpening surface)
    hPrior
    hLatest

/--
Exact-opening projection to the non-aggregate private DEC and stage facts.
-/
theorem nonAggregatePrivateDecStageFacts_ofExactPublicIOOpeningLatestStep
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ConcreteExactPublicIOOpeningSurface (PriorProof := PriorProof) ctx)
    {priorSteps : Nat}
    {priorProof : PriorProof}
    {priorImage nextImage : PublicImage Digest Boundary}
    {latestProof : Unit}
    (hPrior :
      RuntimeVerifyPriorOfExactPublicIOOpening
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
  DirectParentOnlyProductionConcreteFPrimePriorRawIOSoundness.nonAggregatePrivateDecStageFacts_ofRawPublicIOSoundnessLatestStep
    (rawPublicIOSoundnessSurfaceOfExactPublicIOOpening surface)
    hPrior
    hLatest

/--
Exact-opening projection to the Section 7.1 owner-target stage audit.
-/
theorem section71StageTargetAuditTrail_ofExactPublicIOOpeningLatestStep
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ConcreteExactPublicIOOpeningSurface (PriorProof := PriorProof) ctx)
    {priorSteps : Nat}
    {priorProof : PriorProof}
    {priorImage nextImage : PublicImage Digest Boundary}
    {latestProof : Unit}
    (hPrior :
      RuntimeVerifyPriorOfExactPublicIOOpening
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
  DirectParentOnlyProductionConcreteFPrimePriorRawIOSoundness.section71StageTargetAuditTrail_ofRawPublicIOSoundnessLatestStep
    (rawPublicIOSoundnessSurfaceOfExactPublicIOOpening surface)
    hPrior
    hLatest

end DirectParentOnlyProductionConcreteFPrimePriorExactIOOpening

end DirectCcsFPrime
