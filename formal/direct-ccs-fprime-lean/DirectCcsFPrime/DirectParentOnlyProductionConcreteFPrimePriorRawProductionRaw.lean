import DirectCcsFPrime.DirectParentOnlyProductionConcreteFPrimePriorRawIOOpening
import DirectCcsFPrime.DirectParentOnlyProductionConcreteFPrimePriorExactIOOpening

/-!
Production-shaped raw public-IO bridge for the concrete prior F' verifier.

This module names the verifier-visible RV64IM compressed-main proof surface:
the published statement, the caller-derived expected IVC public image, the
SNARK-carried public image, Poseidon2 proof-binding digests, the terminal F'
committed proof public vector, final carried CE checks, and the fixed authority
opener. The only trusted backend facts are phrased over this production view:
accepted verifier checks must open folded F' authority, and that opened
authority must bind the same `(steps, image)` pair.
-/

namespace DirectCcsFPrime

namespace DirectParentOnlyProductionConcreteFPrimePriorRawProduction

/-- Section 7.1-backed production context. -/
abbrev ProductionContext :=
  DirectParentOnlyProductionConcreteFPrimePriorRawIOOpening.ProductionContext

/-- Public image for the parent-only production path. -/
abbrev PublicImage :=
  DirectParentOnlyProductionConcreteFPrimePriorRawIOOpening.PublicImage

/-- Proof-carrying folded prior authority for the induced production context. -/
abbrev ProofCarryingPriorProof :=
  @DirectParentOnlyProductionConcreteFPrimePriorRawIOOpening.ProofCarryingPriorProof

/-- Generic raw public-IO verifier checks used by the split opening bridge. -/
abbrev ConcreteRawPublicIOVerifierChecks :=
  @DirectParentOnlyProductionConcreteFPrimePriorRawIOOpening.ConcreteRawPublicIOVerifierChecks

/-- Bound raw public-vector acceptance for the canonical statement. -/
abbrev RawPublicIOBoundStatementAccepted :=
  @DirectParentOnlyProductionConcreteFPrimePriorRawIOOpening.RawPublicIOBoundStatementAccepted

/-- Opening-level raw public-IO bridge consumed by existing certified verifiers. -/
abbrev ConcreteRawPublicIOOpeningSurface :=
  @DirectParentOnlyProductionConcreteFPrimePriorRawIOOpening.ConcreteRawPublicIOOpeningSurface

/-- Structured terminal/boundary public IO returned by the production verifier. -/
abbrev ExactTerminalBoundaryPublicIO :=
  DirectParentOnlyProductionConcreteFPrimePriorExactIOOpening.ExactTerminalBoundaryPublicIO

/--
Verifier-visible production view of the compressed prior F' proof.

The names mirror the Rust verifier path:

* `expectedIvcPublicImage` models deriving the expected image from the
  published statement.
* `proofIvcPublicImage` models the public image carried by the compressed SNARK.
* `terminalVerifierPublicIO` models the terminal F' committed-step verifier
  returning Spartan public IO.
* `finalClaims*` fields model final carried CE binding and verification.
* `compressedProofDigest` and `bindingDigest` model the Poseidon2 transcript
  replays for the published statement, public image, and proof bytes.

No field here is an authority theorem.
-/
structure ProductionRawPriorVerifierChecks
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : ProductionContext Digest Boundary n params) where
  PublishedStatement : Type
  IvcPublicImage : Type
  Construction2Boundary : Type
  PublicField : Type
  TerminalCommittedProof : Type
  FinalMainClaims : Type
  FinalCeProof : Type
  ProofDigest : Type
  publishedStatement :
    PriorProof →
      PublishedStatement
  proofIvcPublicImage :
    PriorProof →
      IvcPublicImage
  expectedIvcPublicImage :
    PublishedStatement →
      Option IvcPublicImage
  canonicalIvcPublicImage :
    Nat →
      PublicImage Digest Boundary →
        IvcPublicImage
  publicImageOfIvc :
    IvcPublicImage →
      PublicImage Digest Boundary
  stepCountOfIvc :
    IvcPublicImage →
      Nat
  construction2Boundary :
    IvcPublicImage →
      Construction2Boundary
  terminalPublicValues :
    IvcPublicImage →
      List PublicField
  boundaryPublicValues :
    Construction2Boundary →
      List PublicField
  terminalCommittedProof :
    PriorProof →
      TerminalCommittedProof
  terminalVerifierPublicIO :
    TerminalCommittedProof →
      Option (List PublicField)
  finalMainClaims :
    PriorProof →
      FinalMainClaims
  finalCeProof :
    PriorProof →
      FinalCeProof
  publishedStatementValid :
    PublishedStatement →
      Prop
  statementPublicValid :
    IvcPublicImage →
      Prop
  finalClaimsCanonical :
    FinalMainClaims →
      Prop
  finalClaimsBindPublicImage :
    IvcPublicImage →
      FinalMainClaims →
        Prop
  finalCeVerifierAccepts :
    FinalMainClaims →
      FinalCeProof →
        Prop
  compressedProofDigest :
    PriorProof →
      ProofDigest
  recomputedCompressedProofDigest :
    PublishedStatement →
      IvcPublicImage →
      TerminalCommittedProof →
      FinalMainClaims →
      FinalCeProof →
        ProofDigest
  bindingDigest :
    PriorProof →
      ProofDigest
  recomputedBindingDigest :
    PublishedStatement →
      IvcPublicImage →
      TerminalCommittedProof →
      FinalMainClaims →
      FinalCeProof →
        ProofDigest
  openAuthority :
    PriorProof →
      Option (ProofCarryingPriorProof ctx)

/--
Compressed-public-image replay for the production proof.

This is the Lean shape of `self.public_image == expected_public_image` and the
caller-supplied `(steps, image)` pair agreeing with that public image.
-/
def ProductionCompactImageReplay
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (checks :
      ProductionRawPriorVerifierChecks (PriorProof := PriorProof) ctx)
    (steps : Nat)
    (proof : PriorProof)
    (image : PublicImage Digest Boundary) : Prop :=
  checks.proofIvcPublicImage proof =
      checks.canonicalIvcPublicImage steps image ∧
    checks.publicImageOfIvc (checks.proofIvcPublicImage proof) = image ∧
    checks.stepCountOfIvc (checks.proofIvcPublicImage proof) = steps

/--
Construction-2 and final carried-CE checks for the production proof.

This models the Rust checks that validate the published statement, derive the
expected IVC public image, bind final carried CE claims to that image, and
verify the final carried CE proof.
-/
def ProductionConstruction2BoundaryReplay
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (checks :
      ProductionRawPriorVerifierChecks (PriorProof := PriorProof) ctx)
    (steps : Nat)
    (proof : PriorProof)
    (image : PublicImage Digest Boundary) : Prop :=
  checks.publishedStatementValid (checks.publishedStatement proof) ∧
    checks.expectedIvcPublicImage (checks.publishedStatement proof) =
      some (checks.canonicalIvcPublicImage steps image) ∧
    checks.publicImageOfIvc (checks.canonicalIvcPublicImage steps image) =
      image ∧
    checks.stepCountOfIvc (checks.canonicalIvcPublicImage steps image) =
      steps ∧
    checks.finalClaimsCanonical (checks.finalMainClaims proof) ∧
    checks.finalClaimsBindPublicImage
      (checks.canonicalIvcPublicImage steps image)
      (checks.finalMainClaims proof) ∧
    checks.finalCeVerifierAccepts
      (checks.finalMainClaims proof)
      (checks.finalCeProof proof)

/--
Poseidon2 transcript replay for the production compressed-main proof.

The digest equations model the implementation re-hashing the published
statement, compressed public image, terminal proof, final claims, and final CE
proof into the proof digest and binding digest. Poseidon2 soundness itself
remains an external cryptographic assumption.
-/
def ProductionPoseidon2TranscriptReplay
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (checks :
      ProductionRawPriorVerifierChecks (PriorProof := PriorProof) ctx)
    (_steps : Nat)
    (proof : PriorProof)
    (_image : PublicImage Digest Boundary) : Prop :=
  checks.compressedProofDigest proof =
      checks.recomputedCompressedProofDigest
        (checks.publishedStatement proof)
        (checks.proofIvcPublicImage proof)
        (checks.terminalCommittedProof proof)
        (checks.finalMainClaims proof)
        (checks.finalCeProof proof) ∧
    checks.bindingDigest proof =
      checks.recomputedBindingDigest
        (checks.publishedStatement proof)
        (checks.proofIvcPublicImage proof)
        (checks.terminalCommittedProof proof)
        (checks.finalMainClaims proof)
        (checks.finalCeProof proof)

/--
Project the production verifier view to the existing raw public-IO checker.
-/
def rawPublicIOVerifierChecksOfProduction
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (checks :
      ProductionRawPriorVerifierChecks (PriorProof := PriorProof) ctx) :
    ConcreteRawPublicIOVerifierChecks (PriorProof := PriorProof) ctx where
  Statement := checks.IvcPublicImage
  PublicBoundary := checks.Construction2Boundary
  PublicField := checks.PublicField
  TerminalCommittedProof := checks.TerminalCommittedProof
  canonicalStatement := checks.canonicalIvcPublicImage
  proofStatement := checks.proofIvcPublicImage
  statementBoundary := checks.construction2Boundary
  proofBoundary := fun proof =>
    checks.construction2Boundary (checks.proofIvcPublicImage proof)
  terminalPublicValues := checks.terminalPublicValues
  boundaryPublicValues := checks.boundaryPublicValues
  terminalCommittedProof := checks.terminalCommittedProof
  statementPublicValid := checks.statementPublicValid
  terminalVerifierPublicIO := checks.terminalVerifierPublicIO
  compactImageReplay := ProductionCompactImageReplay checks
  construction2BoundaryReplay :=
    ProductionConstruction2BoundaryReplay checks
  transcriptReplay := ProductionPoseidon2TranscriptReplay checks
  openAuthority := checks.openAuthority

/-- Verifier-visible production acceptance for one raw public vector. -/
def ProductionRawVerifierAccepted
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (checks :
      ProductionRawPriorVerifierChecks (PriorProof := PriorProof) ctx)
    (steps : Nat)
    (proof : PriorProof)
    (image : PublicImage Digest Boundary)
    (rawPublicIO :
      List (rawPublicIOVerifierChecksOfProduction checks).PublicField) :
    Prop :=
  DirectParentOnlyProductionConcreteFPrimePriorRawIOSoundness.RawPublicIOVerifierAccepted
    (rawPublicIOVerifierChecksOfProduction checks)
    steps
    proof
    image
    rawPublicIO

/--
Production acceptance after replay binds the proof-carried IVC image to the
canonical IVC image for the same `(steps, image)` pair.
-/
def ProductionRawBoundStatementAccepted
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (checks :
      ProductionRawPriorVerifierChecks (PriorProof := PriorProof) ctx)
    (steps : Nat)
    (proof : PriorProof)
    (image : PublicImage Digest Boundary)
    (rawPublicIO :
      List (rawPublicIOVerifierChecksOfProduction checks).PublicField) :
    Prop :=
  ProductionRawVerifierAccepted
      checks
      steps
      proof
      image
      rawPublicIO ∧
    checks.proofIvcPublicImage proof =
      checks.canonicalIvcPublicImage steps image

/-- Raw bound-statement acceptance is already production bound acceptance. -/
theorem productionRawBoundStatementAccepted_ofRawBoundStatement
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (checks :
      ProductionRawPriorVerifierChecks (PriorProof := PriorProof) ctx)
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    {rawPublicIO :
      List (rawPublicIOVerifierChecksOfProduction checks).PublicField}
    (hBound :
      RawPublicIOBoundStatementAccepted
        (rawPublicIOVerifierChecksOfProduction checks)
        steps
        proof
        image
        rawPublicIO) :
    ProductionRawBoundStatementAccepted
      checks
      steps
      proof
      image
      rawPublicIO := by
  exact hBound

/--
Production-level opening certificate.

The trusted backend obligations are now stated over the production proof view,
not over an arbitrary verifier predicate:

* accepted production checks open through the fixed authority opener;
* the opened authority binds the same production `(steps, image)` statement.
-/
structure ProductionRawPriorOpeningSurface
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : ProductionContext Digest Boundary n params) where
  checks :
    ProductionRawPriorVerifierChecks (PriorProof := PriorProof) ctx
  productionBackendOpens :
    ∀ steps proof image rawPublicIO,
      ProductionRawBoundStatementAccepted
        checks
        steps
        proof
        image
        rawPublicIO →
        ∃ authority : ProofCarryingPriorProof ctx,
          checks.openAuthority proof = some authority
  openedAuthorityBindsProductionStatement :
    ∀ steps proof image rawPublicIO authority,
      ProductionRawBoundStatementAccepted
        checks
        steps
        proof
        image
        rawPublicIO →
      checks.openAuthority proof = some authority →
        authority.steps = steps ∧ authority.image = image

/-- Production replay binds the opaque proof statement to the canonical one. -/
theorem productionReplayBindsProofStatement
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (checks :
      ProductionRawPriorVerifierChecks (PriorProof := PriorProof) ctx) :
    ∀ steps proof image,
      (rawPublicIOVerifierChecksOfProduction checks).compactImageReplay
        steps
        proof
        image →
      (rawPublicIOVerifierChecksOfProduction checks).construction2BoundaryReplay
        steps
        proof
        image →
      (rawPublicIOVerifierChecksOfProduction checks).transcriptReplay
        steps
        proof
        image →
        (rawPublicIOVerifierChecksOfProduction checks).proofStatement proof =
          (rawPublicIOVerifierChecksOfProduction checks).canonicalStatement
            steps
            image := by
  intro steps proof image hCompact _hBoundaryReplay _hTranscript
  exact hCompact.1

/--
Convert the production opening certificate to the generic raw opening surface.
-/
def rawPublicIOOpeningSurfaceOfProduction
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ProductionRawPriorOpeningSurface (PriorProof := PriorProof) ctx) :
    ConcreteRawPublicIOOpeningSurface (PriorProof := PriorProof) ctx where
  checks := rawPublicIOVerifierChecksOfProduction surface.checks
  replayBindsProofStatement :=
    productionReplayBindsProofStatement surface.checks
  rawBoundStatementOpens := by
    intro steps proof image rawPublicIO hBound
    exact
      surface.productionBackendOpens
        steps
        proof
        image
        rawPublicIO
        (productionRawBoundStatementAccepted_ofRawBoundStatement
          surface.checks
          hBound)
  openedAuthorityBindsBoundStatement := by
    intro steps proof image rawPublicIO authority hBound hOpen
    exact
      surface.openedAuthorityBindsProductionStatement
        steps
        proof
        image
        rawPublicIO
        authority
        (productionRawBoundStatementAccepted_ofRawBoundStatement
          surface.checks
          hBound)
        hOpen

/-- Runtime verifier predicate induced by the production raw verifier view. -/
def RuntimeVerifyPriorOfProductionRaw
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ProductionRawPriorOpeningSurface (PriorProof := PriorProof) ctx) :
    Nat →
      PriorProof →
      PublicImage Digest Boundary →
        Prop :=
  DirectParentOnlyProductionConcreteFPrimePriorRawIOOpening.RuntimeVerifyPriorOfRawPublicIOOpening
    (rawPublicIOOpeningSurfaceOfProduction surface)

/--
Audit-facing evidence for an accepted production raw prior proof.

The evidence keeps both sides visible: the verifier-visible production checks
with their exact raw public IO, and the opened folded F' authority for the same
public pair.
-/
def AcceptedProductionRawEvidence
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ProductionRawPriorOpeningSurface (PriorProof := PriorProof) ctx)
    (steps : Nat)
    (proof : PriorProof)
    (image : PublicImage Digest Boundary) : Prop :=
  (∃ rawPublicIO :
      List (rawPublicIOVerifierChecksOfProduction surface.checks).PublicField,
    ProductionRawBoundStatementAccepted
      surface.checks
      steps
      proof
      image
      rawPublicIO) ∧
    ∃ authority : ProofCarryingPriorProof ctx,
      surface.checks.openAuthority proof = some authority ∧
        FoldedFPrimeAuthority.Accepts
          (Transition :=
            DirectParentOnlyProductionSoundness.Transition
              ctx.toProductionContext)
          (initial := ctx.initial)
          steps
          authority
          image

/--
Accepted production raw verification yields a bound statement witness.
-/
theorem runtimeVerifyPriorOfProductionRaw_boundStatement
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ProductionRawPriorOpeningSurface (PriorProof := PriorProof) ctx)
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    (hVerify :
      RuntimeVerifyPriorOfProductionRaw surface steps proof image) :
    ∃ rawPublicIO :
      List (rawPublicIOVerifierChecksOfProduction surface.checks).PublicField,
      ProductionRawBoundStatementAccepted
        surface.checks
        steps
        proof
        image
        rawPublicIO := by
  have hBound :
      ∃ rawPublicIO :
        List (rawPublicIOVerifierChecksOfProduction surface.checks).PublicField,
        RawPublicIOBoundStatementAccepted
          (rawPublicIOVerifierChecksOfProduction surface.checks)
          steps
          proof
          image
          rawPublicIO :=
    DirectParentOnlyProductionConcreteFPrimePriorRawIOSoundness.runtimeVerifyPriorOfRawPublicIOSoundness_boundStatement
      (DirectParentOnlyProductionConcreteFPrimePriorRawIOOpening.rawPublicIOSoundnessSurfaceOfOpening
        (rawPublicIOOpeningSurfaceOfProduction surface))
      hVerify
  rcases hBound with ⟨rawPublicIO, hRawBound⟩
  exact
    ⟨rawPublicIO,
      productionRawBoundStatementAccepted_ofRawBoundStatement
        surface.checks
        hRawBound⟩

/--
Accepted production raw verification exposes verifier checks and authority.
-/
theorem runtimeVerifyPriorOfProductionRaw_evidence
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ProductionRawPriorOpeningSurface (PriorProof := PriorProof) ctx)
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    (hVerify :
      RuntimeVerifyPriorOfProductionRaw surface steps proof image) :
    AcceptedProductionRawEvidence surface steps proof image := by
  exact
    ⟨runtimeVerifyPriorOfProductionRaw_boundStatement
        surface
        hVerify,
      DirectParentOnlyProductionConcreteFPrimePriorRawIOOpening.runtimeVerifyPriorOfRawPublicIOOpening_acceptedOpens
        (rawPublicIOOpeningSurfaceOfProduction surface)
        steps
        proof
        image
        hVerify⟩

/-- Certified prior verifier induced by production raw verifier checks. -/
def certifiedPriorVerifierOfProductionRaw
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ProductionRawPriorOpeningSurface (PriorProof := PriorProof) ctx) :
    DirectParentOnlyProductionSuperNeoReuseCertifiedVerifier.CertifiedPriorVerifier
      (PriorProof := PriorProof)
      ctx :=
  DirectParentOnlyProductionConcreteFPrimePriorRawIOOpening.certifiedPriorVerifierOfRawPublicIOOpening
    (rawPublicIOOpeningSurfaceOfProduction surface)

/-- The production certified verifier uses the production raw predicate. -/
theorem certifiedPriorVerifierOfProductionRaw_verify
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ProductionRawPriorOpeningSurface (PriorProof := PriorProof) ctx) :
    (certifiedPriorVerifierOfProductionRaw surface).verify =
      RuntimeVerifyPriorOfProductionRaw surface :=
  rfl

/-- Production raw acceptance opens folded F' authority for the same pair. -/
theorem runtimeVerifyPriorOfProductionRaw_acceptedOpens
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ProductionRawPriorOpeningSurface (PriorProof := PriorProof) ctx) :
    ∀ steps proof image,
      RuntimeVerifyPriorOfProductionRaw surface steps proof image →
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
  exact
    DirectParentOnlyProductionConcreteFPrimePriorRawIOOpening.runtimeVerifyPriorOfRawPublicIOOpening_acceptedOpens
      (rawPublicIOOpeningSurfaceOfProduction surface)
      steps
      proof
      image
      hVerify

/-- The production raw verifier reaches every prior image it accepts. -/
theorem runtimeVerifyPriorOfProductionRaw_reaches_prior
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ProductionRawPriorOpeningSurface (PriorProof := PriorProof) ctx)
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    (hVerify :
      RuntimeVerifyPriorOfProductionRaw surface steps proof image) :
    FPrimeInduction.Reachable
      (DirectParentOnlyProductionSoundness.Transition
        ctx.toProductionContext)
      ctx.initial
      steps
      image :=
  DirectParentOnlyProductionConcreteFPrimePriorRawIOOpening.runtimeVerifyPriorOfRawPublicIOOpening_reaches_prior
    (rawPublicIOOpeningSurfaceOfProduction surface)
    hVerify

/-- The production raw verifier cannot accept an unreachable prior image. -/
theorem runtimeVerifyPriorOfProductionRaw_cannot_accept_unreachable_prior
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ProductionRawPriorOpeningSurface (PriorProof := PriorProof) ctx)
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    (hVerify :
      RuntimeVerifyPriorOfProductionRaw surface steps proof image)
    (hUnreachable :
      ¬ FPrimeInduction.Reachable
        (DirectParentOnlyProductionSoundness.Transition
          ctx.toProductionContext)
        ctx.initial
        steps
        image) :
    False :=
  hUnreachable
    (runtimeVerifyPriorOfProductionRaw_reaches_prior
      surface
      hVerify)

/-- Production raw acceptance exposes prior public-image invariants. -/
theorem runtimeVerifyPriorOfProductionRaw_publicImageInvariants
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ProductionRawPriorOpeningSurface (PriorProof := PriorProof) ctx)
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    (hVerify :
      RuntimeVerifyPriorOfProductionRaw surface steps proof image) :
    DirectParentOnlyProductionConcreteFPrimePrior.AcceptedPriorPublicImageInvariants
      ctx
      steps
      image :=
  DirectParentOnlyProductionConcreteFPrimePriorRawIOOpening.runtimeVerifyPriorOfRawPublicIOOpening_publicImageInvariants
    (rawPublicIOOpeningSurfaceOfProduction surface)
    hVerify

/-- The production raw verifier is same-proof functional. -/
theorem proofFunctionalOfProductionRaw
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ProductionRawPriorOpeningSurface (PriorProof := PriorProof) ctx) :
    CompressedFPrimeAuthority.ProofFunctional
      (RuntimeVerifyPriorOfProductionRaw surface) :=
  DirectParentOnlyProductionConcreteFPrimePriorRawIOOpening.proofFunctionalOfRawPublicIOOpening
    (rawPublicIOOpeningSurfaceOfProduction surface)

/-- Strict `SoundVerifier` induced by production raw verifier checks. -/
def soundVerifierOfProductionRaw
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ProductionRawPriorOpeningSurface (PriorProof := PriorProof) ctx) :
    DirectParentOnlyProductionSuperNeoReuse.ProductionContext.SoundPriorVerifier
      (PriorProof := PriorProof)
      ctx :=
  DirectParentOnlyProductionSuperNeoReuseCertifiedVerifier.CertifiedPriorVerifier.soundVerifier
    (certifiedPriorVerifierOfProductionRaw surface)

/-- The production raw `SoundVerifier` accepts exactly production raw checks. -/
theorem soundVerifierOfProductionRaw_accepts_iff
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ProductionRawPriorOpeningSurface (PriorProof := PriorProof) ctx)
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary} :
    CompressedFPrimeAuthority.SoundVerifier.Accepts
        (soundVerifierOfProductionRaw surface)
        steps
        proof
        image <->
      RuntimeVerifyPriorOfProductionRaw
        surface
        steps
        proof
        image := by
  simpa [soundVerifierOfProductionRaw,
    certifiedPriorVerifierOfProductionRaw_verify]
    using
      DirectParentOnlyProductionSuperNeoReuseCertifiedVerifier.CertifiedPriorVerifier.soundVerifier_accepts_iff
        (certifiedPriorVerifierOfProductionRaw surface)

/-- Production raw strict acceptance opens to folded F' authority. -/
theorem soundVerifierOfProductionRaw_opensToFoldedAuthority
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    {surface :
      ProductionRawPriorOpeningSurface (PriorProof := PriorProof) ctx}
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    (hVerify :
      CompressedFPrimeAuthority.SoundVerifier.Accepts
        (soundVerifierOfProductionRaw surface)
        steps
        proof
        image) :
    ∃ authority : ProofCarryingPriorProof ctx,
      FoldedFPrimeAuthority.Accepts
        (Transition :=
          DirectParentOnlyProductionSoundness.Transition
            ctx.toProductionContext)
        (initial := ctx.initial)
        steps
        authority
        image :=
  (soundVerifierOfProductionRaw surface).opensToFoldedAuthority
    steps
    proof
    image
    hVerify

/-- One production raw proof cannot verify for two public pairs. -/
theorem soundVerifierOfProductionRaw_sameProof
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    {surface :
      ProductionRawPriorOpeningSurface (PriorProof := PriorProof) ctx}
    {stepsA stepsB : Nat}
    {proof : PriorProof}
    {imageA imageB : PublicImage Digest Boundary}
    (hA :
      CompressedFPrimeAuthority.SoundVerifier.Accepts
        (soundVerifierOfProductionRaw surface)
        stepsA
        proof
        imageA)
    (hB :
      CompressedFPrimeAuthority.SoundVerifier.Accepts
        (soundVerifierOfProductionRaw surface)
        stepsB
        proof
        imageB) :
    stepsA = stepsB ∧ imageA = imageB :=
  proofFunctionalOfProductionRaw surface hA hB

/--
Production raw prior-plus-latest end-to-end theorem.

The caller supplies production verifier acceptance and latest-step acceptance.
Lean constructs the certified verifier internally and returns the existing
terminal end-to-end package, including parent-only CE binding, no-swap, stage
audit, and public-image invariants.
-/
theorem certifiedSingleTerminalEndToEnd_ofProductionRawLatestStep
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ProductionRawPriorOpeningSurface (PriorProof := PriorProof) ctx)
    {priorSteps : Nat}
    {priorProof : PriorProof}
    {priorImage nextImage : PublicImage Digest Boundary}
    {latestProof : Unit}
    (hPrior :
      RuntimeVerifyPriorOfProductionRaw
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
      (certifiedPriorVerifierOfProductionRaw surface).opening
      priorSteps
      priorSteps
      priorProof
      priorImage
      priorImage
      nextImage
      nextImage :=
  DirectParentOnlyProductionSuperNeoReuseEndToEnd.certifiedSingleTerminalEndToEnd_ofCertifiedPriorVerifierLatestStep
    (certifiedPriorVerifierOfProductionRaw surface)
    hPrior
    hLatest

/-- Production raw projection to non-aggregate private DEC and stage facts. -/
theorem nonAggregatePrivateDecStageFacts_ofProductionRawLatestStep
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ProductionRawPriorOpeningSurface (PriorProof := PriorProof) ctx)
    {priorSteps : Nat}
    {priorProof : PriorProof}
    {priorImage nextImage : PublicImage Digest Boundary}
    {latestProof : Unit}
    (hPrior :
      RuntimeVerifyPriorOfProductionRaw
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
    (certifiedPriorVerifierOfProductionRaw surface)
    hPrior
    hLatest

/-- Production raw projection to the Section 7.1 owner-target stage audit. -/
theorem section71StageTargetAuditTrail_ofProductionRawLatestStep
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ProductionRawPriorOpeningSurface (PriorProof := PriorProof) ctx)
    {priorSteps : Nat}
    {priorProof : PriorProof}
    {priorImage nextImage : PublicImage Digest Boundary}
    {latestProof : Unit}
    (hPrior :
      RuntimeVerifyPriorOfProductionRaw
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
    (certifiedPriorVerifierOfProductionRaw surface)
    hPrior
    hLatest


end DirectParentOnlyProductionConcreteFPrimePriorRawProduction

end DirectCcsFPrime
