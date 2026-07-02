import DirectCcsFPrime.ProofSystem.Production.Impl.RawProduction.Raw

/-!
Structured exact public-IO production verifier surface.

This module owns the exact terminal/boundary public-IO verifier view and the
opening consequences that turn accepted production exact evidence into folded
F' authority for the same `(steps, image)` pair.
-/

namespace DirectCcsFPrime

namespace DirectParentOnlyProductionConcreteFPrimePriorRawProduction

/--
Production verifier view with structured terminal/boundary public IO.
-/
structure ProductionExactPriorVerifierChecks
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
  publishedStatement : PriorProof → PublishedStatement
  proofIvcPublicImage : PriorProof → IvcPublicImage
  expectedIvcPublicImage : PublishedStatement → Option IvcPublicImage
  canonicalIvcPublicImage :
    Nat →
      PublicImage Digest Boundary →
        IvcPublicImage
  publicImageOfIvc : IvcPublicImage → PublicImage Digest Boundary
  stepCountOfIvc : IvcPublicImage → Nat
  construction2Boundary : IvcPublicImage → Construction2Boundary
  terminalPublicValues : IvcPublicImage → List PublicField
  boundaryPublicValues : Construction2Boundary → List PublicField
  terminalCommittedProof : PriorProof → TerminalCommittedProof
  terminalVerifierPublicIO :
    TerminalCommittedProof →
      Option (ExactTerminalBoundaryPublicIO PublicField)
  finalMainClaims : PriorProof → FinalMainClaims
  finalCeProof : PriorProof → FinalCeProof
  publishedStatementValid : PublishedStatement → Prop
  statementPublicValid : IvcPublicImage → Prop
  finalClaimsCanonical : FinalMainClaims → Prop
  finalClaimsBindPublicImage :
    IvcPublicImage →
      FinalMainClaims →
        Prop
  finalCeVerifierAccepts :
    FinalMainClaims →
      FinalCeProof →
        Prop
  compressedProofDigest : PriorProof → ProofDigest
  recomputedCompressedProofDigest :
    PublishedStatement →
      IvcPublicImage →
      TerminalCommittedProof →
      FinalMainClaims →
      FinalCeProof →
        ProofDigest
  bindingDigest : PriorProof → ProofDigest
  recomputedBindingDigest :
    PublishedStatement →
      IvcPublicImage →
      TerminalCommittedProof →
      FinalMainClaims →
      FinalCeProof →
        ProofDigest
  openAuthority : PriorProof → Option (ProofCarryingPriorProof ctx)

/-- Forget structured terminal/boundary IO to the raw production verifier view. -/
def rawProductionVerifierChecksOfExact
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (checks :
      ProductionExactPriorVerifierChecks (PriorProof := PriorProof) ctx) :
    ProductionRawPriorVerifierChecks (PriorProof := PriorProof) ctx where
  PublishedStatement := checks.PublishedStatement
  IvcPublicImage := checks.IvcPublicImage
  Construction2Boundary := checks.Construction2Boundary
  PublicField := checks.PublicField
  TerminalCommittedProof := checks.TerminalCommittedProof
  FinalMainClaims := checks.FinalMainClaims
  FinalCeProof := checks.FinalCeProof
  ProofDigest := checks.ProofDigest
  publishedStatement := checks.publishedStatement
  proofIvcPublicImage := checks.proofIvcPublicImage
  expectedIvcPublicImage := checks.expectedIvcPublicImage
  canonicalIvcPublicImage := checks.canonicalIvcPublicImage
  publicImageOfIvc := checks.publicImageOfIvc
  stepCountOfIvc := checks.stepCountOfIvc
  construction2Boundary := checks.construction2Boundary
  terminalPublicValues := checks.terminalPublicValues
  boundaryPublicValues := checks.boundaryPublicValues
  terminalCommittedProof := checks.terminalCommittedProof
  terminalVerifierPublicIO := fun terminalProof =>
    Option.map
      (fun publicIO => publicIO.raw)
      (checks.terminalVerifierPublicIO terminalProof)
  finalMainClaims := checks.finalMainClaims
  finalCeProof := checks.finalCeProof
  publishedStatementValid := checks.publishedStatementValid
  statementPublicValid := checks.statementPublicValid
  finalClaimsCanonical := checks.finalClaimsCanonical
  finalClaimsBindPublicImage := checks.finalClaimsBindPublicImage
  finalCeVerifierAccepts := checks.finalCeVerifierAccepts
  compressedProofDigest := checks.compressedProofDigest
  recomputedCompressedProofDigest := checks.recomputedCompressedProofDigest
  bindingDigest := checks.bindingDigest
  recomputedBindingDigest := checks.recomputedBindingDigest
  openAuthority := checks.openAuthority

/--
Structured production exact-IO acceptance for one public-IO object.

The terminal committed proof must expose exactly the terminal F' public values
and exactly the Construction-2 boundary public values.
-/
def ProductionExactVerifierAccepted
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (checks :
      ProductionExactPriorVerifierChecks (PriorProof := PriorProof) ctx)
    (steps : Nat)
    (proof : PriorProof)
    (image : PublicImage Digest Boundary)
    (publicIO : ExactTerminalBoundaryPublicIO checks.PublicField) :
    Prop :=
  ProductionCompactImageReplay
      (rawProductionVerifierChecksOfExact checks)
      steps
      proof
      image ∧
    ProductionConstruction2BoundaryReplay
      (rawProductionVerifierChecksOfExact checks)
      steps
      proof
      image ∧
    ProductionPoseidon2TranscriptReplay
      (rawProductionVerifierChecksOfExact checks)
      steps
      proof
      image ∧
    checks.statementPublicValid
      (checks.canonicalIvcPublicImage steps image) ∧
    checks.construction2Boundary
        (checks.proofIvcPublicImage proof) =
      checks.construction2Boundary
        (checks.canonicalIvcPublicImage steps image) ∧
    checks.terminalVerifierPublicIO
      (checks.terminalCommittedProof proof) =
        some publicIO ∧
    publicIO.terminal =
      checks.terminalPublicValues
        (checks.canonicalIvcPublicImage steps image) ∧
    publicIO.boundary =
      checks.boundaryPublicValues
        (checks.construction2Boundary
          (checks.canonicalIvcPublicImage steps image))

/-- Structured production exact-IO acceptance after statement binding. -/
def ProductionExactBoundStatementAccepted
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (checks :
      ProductionExactPriorVerifierChecks (PriorProof := PriorProof) ctx)
    (steps : Nat)
    (proof : PriorProof)
    (image : PublicImage Digest Boundary)
    (publicIO : ExactTerminalBoundaryPublicIO checks.PublicField) :
    Prop :=
  ProductionExactVerifierAccepted checks steps proof image publicIO ∧
    checks.proofIvcPublicImage proof =
      checks.canonicalIvcPublicImage steps image

/-- Structured production exact-IO acceptance is already statement-bound. -/
theorem productionExactBoundStatementAccepted_ofAccepted
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    {checks :
      ProductionExactPriorVerifierChecks (PriorProof := PriorProof) ctx}
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    {publicIO : ExactTerminalBoundaryPublicIO checks.PublicField}
    (hAccepted :
      ProductionExactVerifierAccepted
        checks
        steps
        proof
        image
        publicIO) :
    ProductionExactBoundStatementAccepted
      checks
      steps
      proof
      image
      publicIO :=
  ⟨hAccepted, hAccepted.1.1⟩

/-- Accepted exact verification includes compact public-image replay. -/
theorem productionExactVerifierAccepted_compactImageReplay
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    {checks :
      ProductionExactPriorVerifierChecks (PriorProof := PriorProof) ctx}
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    {publicIO : ExactTerminalBoundaryPublicIO checks.PublicField}
    (hAccepted :
      ProductionExactVerifierAccepted checks steps proof image publicIO) :
    ProductionCompactImageReplay
      (rawProductionVerifierChecksOfExact checks)
      steps
      proof
      image :=
  hAccepted.1

/-- Accepted exact verification includes Construction-2/final-CE replay. -/
theorem productionExactVerifierAccepted_construction2BoundaryReplay
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    {checks :
      ProductionExactPriorVerifierChecks (PriorProof := PriorProof) ctx}
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    {publicIO : ExactTerminalBoundaryPublicIO checks.PublicField}
    (hAccepted :
      ProductionExactVerifierAccepted checks steps proof image publicIO) :
    ProductionConstruction2BoundaryReplay
      (rawProductionVerifierChecksOfExact checks)
      steps
      proof
      image :=
  hAccepted.2.1

/-- Accepted exact verification includes Poseidon2 transcript replay. -/
theorem productionExactVerifierAccepted_poseidon2TranscriptReplay
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    {checks :
      ProductionExactPriorVerifierChecks (PriorProof := PriorProof) ctx}
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    {publicIO : ExactTerminalBoundaryPublicIO checks.PublicField}
    (hAccepted :
      ProductionExactVerifierAccepted checks steps proof image publicIO) :
    ProductionPoseidon2TranscriptReplay
      (rawProductionVerifierChecksOfExact checks)
      steps
      proof
      image :=
  hAccepted.2.2.1

/-- Accepted exact verification binds the proof's public image to the canonical `(steps, image)` image. -/
theorem productionExactVerifierAccepted_proofIvcPublicImage_eq_canonical
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    {checks :
      ProductionExactPriorVerifierChecks (PriorProof := PriorProof) ctx}
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    {publicIO : ExactTerminalBoundaryPublicIO checks.PublicField}
    (hAccepted :
      ProductionExactVerifierAccepted checks steps proof image publicIO) :
    checks.proofIvcPublicImage proof =
      checks.canonicalIvcPublicImage steps image :=
  hAccepted.1.1

/-- Accepted exact verification projects the proof public image to the claimed public image. -/
theorem productionExactVerifierAccepted_proofPublicImage_eq
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    {checks :
      ProductionExactPriorVerifierChecks (PriorProof := PriorProof) ctx}
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    {publicIO : ExactTerminalBoundaryPublicIO checks.PublicField}
    (hAccepted :
      ProductionExactVerifierAccepted checks steps proof image publicIO) :
    checks.publicImageOfIvc (checks.proofIvcPublicImage proof) = image :=
  hAccepted.1.2.1

/-- Accepted exact verification projects the proof public image to the claimed step count. -/
theorem productionExactVerifierAccepted_proofStepCount_eq
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    {checks :
      ProductionExactPriorVerifierChecks (PriorProof := PriorProof) ctx}
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    {publicIO : ExactTerminalBoundaryPublicIO checks.PublicField}
    (hAccepted :
      ProductionExactVerifierAccepted checks steps proof image publicIO) :
    checks.stepCountOfIvc (checks.proofIvcPublicImage proof) = steps :=
  hAccepted.1.2.2

/-- Accepted exact verification validates the published statement. -/
theorem productionExactVerifierAccepted_publishedStatementValid
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    {checks :
      ProductionExactPriorVerifierChecks (PriorProof := PriorProof) ctx}
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    {publicIO : ExactTerminalBoundaryPublicIO checks.PublicField}
    (hAccepted :
      ProductionExactVerifierAccepted checks steps proof image publicIO) :
    checks.publishedStatementValid (checks.publishedStatement proof) :=
  hAccepted.2.1.1

/-- Accepted exact verification replays the statement's expected public image. -/
theorem productionExactVerifierAccepted_expectedIvcPublicImage_eq_canonical
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    {checks :
      ProductionExactPriorVerifierChecks (PriorProof := PriorProof) ctx}
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    {publicIO : ExactTerminalBoundaryPublicIO checks.PublicField}
    (hAccepted :
      ProductionExactVerifierAccepted checks steps proof image publicIO) :
    checks.expectedIvcPublicImage (checks.publishedStatement proof) =
      some (checks.canonicalIvcPublicImage steps image) :=
  hAccepted.2.1.2.1

/-- Accepted exact verification checks final main claims are canonical. -/
theorem productionExactVerifierAccepted_finalClaimsCanonical
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    {checks :
      ProductionExactPriorVerifierChecks (PriorProof := PriorProof) ctx}
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    {publicIO : ExactTerminalBoundaryPublicIO checks.PublicField}
    (hAccepted :
      ProductionExactVerifierAccepted checks steps proof image publicIO) :
    checks.finalClaimsCanonical (checks.finalMainClaims proof) :=
  hAccepted.2.1.2.2.2.2.1

/-- Accepted exact verification binds final main claims to the canonical public image. -/
theorem productionExactVerifierAccepted_finalClaimsBindPublicImage
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    {checks :
      ProductionExactPriorVerifierChecks (PriorProof := PriorProof) ctx}
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    {publicIO : ExactTerminalBoundaryPublicIO checks.PublicField}
    (hAccepted :
      ProductionExactVerifierAccepted checks steps proof image publicIO) :
    checks.finalClaimsBindPublicImage
      (checks.canonicalIvcPublicImage steps image)
      (checks.finalMainClaims proof) :=
  hAccepted.2.1.2.2.2.2.2.1

/-- Accepted exact verification includes final carried-CE proof acceptance. -/
theorem productionExactVerifierAccepted_finalCeVerifierAccepts
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    {checks :
      ProductionExactPriorVerifierChecks (PriorProof := PriorProof) ctx}
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    {publicIO : ExactTerminalBoundaryPublicIO checks.PublicField}
    (hAccepted :
      ProductionExactVerifierAccepted checks steps proof image publicIO) :
    checks.finalCeVerifierAccepts
      (checks.finalMainClaims proof)
      (checks.finalCeProof proof) :=
  hAccepted.2.1.2.2.2.2.2.2

/-- Accepted exact verification replays the compressed proof digest. -/
theorem productionExactVerifierAccepted_compressedProofDigestReplay
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    {checks :
      ProductionExactPriorVerifierChecks (PriorProof := PriorProof) ctx}
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    {publicIO : ExactTerminalBoundaryPublicIO checks.PublicField}
    (hAccepted :
      ProductionExactVerifierAccepted checks steps proof image publicIO) :
    checks.compressedProofDigest proof =
      checks.recomputedCompressedProofDigest
        (checks.publishedStatement proof)
        (checks.proofIvcPublicImage proof)
        (checks.terminalCommittedProof proof)
        (checks.finalMainClaims proof)
        (checks.finalCeProof proof) :=
  hAccepted.2.2.1.1

/-- Accepted exact verification replays the binding digest. -/
theorem productionExactVerifierAccepted_bindingDigestReplay
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    {checks :
      ProductionExactPriorVerifierChecks (PriorProof := PriorProof) ctx}
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    {publicIO : ExactTerminalBoundaryPublicIO checks.PublicField}
    (hAccepted :
      ProductionExactVerifierAccepted checks steps proof image publicIO) :
    checks.bindingDigest proof =
      checks.recomputedBindingDigest
        (checks.publishedStatement proof)
        (checks.proofIvcPublicImage proof)
        (checks.terminalCommittedProof proof)
        (checks.finalMainClaims proof)
        (checks.finalCeProof proof) :=
  hAccepted.2.2.1.2

/-- Accepted exact verification validates the canonical IVC public statement. -/
theorem productionExactVerifierAccepted_statementPublicValid
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    {checks :
      ProductionExactPriorVerifierChecks (PriorProof := PriorProof) ctx}
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    {publicIO : ExactTerminalBoundaryPublicIO checks.PublicField}
    (hAccepted :
      ProductionExactVerifierAccepted checks steps proof image publicIO) :
    checks.statementPublicValid
      (checks.canonicalIvcPublicImage steps image) :=
  hAccepted.2.2.2.1

/-- Accepted exact verification binds the proof boundary to the canonical boundary. -/
theorem productionExactVerifierAccepted_construction2Boundary_eq
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    {checks :
      ProductionExactPriorVerifierChecks (PriorProof := PriorProof) ctx}
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    {publicIO : ExactTerminalBoundaryPublicIO checks.PublicField}
    (hAccepted :
      ProductionExactVerifierAccepted checks steps proof image publicIO) :
    checks.construction2Boundary
        (checks.proofIvcPublicImage proof) =
      checks.construction2Boundary
        (checks.canonicalIvcPublicImage steps image) :=
  hAccepted.2.2.2.2.1

/-- Accepted exact verification binds the committed proof to this terminal public IO. -/
theorem productionExactVerifierAccepted_terminalVerifierPublicIO_eq
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    {checks :
      ProductionExactPriorVerifierChecks (PriorProof := PriorProof) ctx}
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    {publicIO : ExactTerminalBoundaryPublicIO checks.PublicField}
    (hAccepted :
      ProductionExactVerifierAccepted checks steps proof image publicIO) :
    checks.terminalVerifierPublicIO
      (checks.terminalCommittedProof proof) =
        some publicIO :=
  hAccepted.2.2.2.2.2.1

/-- Accepted exact verification binds terminal public values exactly. -/
theorem productionExactVerifierAccepted_terminalPublicValues_eq
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    {checks :
      ProductionExactPriorVerifierChecks (PriorProof := PriorProof) ctx}
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    {publicIO : ExactTerminalBoundaryPublicIO checks.PublicField}
    (hAccepted :
      ProductionExactVerifierAccepted checks steps proof image publicIO) :
    publicIO.terminal =
      checks.terminalPublicValues
        (checks.canonicalIvcPublicImage steps image) :=
  hAccepted.2.2.2.2.2.2.1

/-- Accepted exact verification binds Construction-2 boundary public values exactly. -/
theorem productionExactVerifierAccepted_boundaryPublicValues_eq
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    {checks :
      ProductionExactPriorVerifierChecks (PriorProof := PriorProof) ctx}
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    {publicIO : ExactTerminalBoundaryPublicIO checks.PublicField}
    (hAccepted :
      ProductionExactVerifierAccepted checks steps proof image publicIO) :
    publicIO.boundary =
      checks.boundaryPublicValues
        (checks.construction2Boundary
          (checks.canonicalIvcPublicImage steps image)) :=
  hAccepted.2.2.2.2.2.2.2

/--
Flat audit package for accepted exact verification.

This packages the verifier checks that matter for the F' authority path so
later proofs do not need to destruct nested conjunctions by hand.
-/
structure ProductionExactVerifierAcceptedAudit
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (checks :
      ProductionExactPriorVerifierChecks (PriorProof := PriorProof) ctx)
    (steps : Nat)
    (proof : PriorProof)
    (image : PublicImage Digest Boundary)
    (publicIO : ExactTerminalBoundaryPublicIO checks.PublicField) :
    Prop where
  compactImageReplay :
    ProductionCompactImageReplay
      (rawProductionVerifierChecksOfExact checks)
      steps
      proof
      image
  construction2BoundaryReplay :
    ProductionConstruction2BoundaryReplay
      (rawProductionVerifierChecksOfExact checks)
      steps
      proof
      image
  poseidon2TranscriptReplay :
    ProductionPoseidon2TranscriptReplay
      (rawProductionVerifierChecksOfExact checks)
      steps
      proof
      image
  proofIvcPublicImage_eq_canonical :
    checks.proofIvcPublicImage proof =
      checks.canonicalIvcPublicImage steps image
  proofPublicImage_eq :
    checks.publicImageOfIvc (checks.proofIvcPublicImage proof) = image
  proofStepCount_eq :
    checks.stepCountOfIvc (checks.proofIvcPublicImage proof) = steps
  publishedStatementValid :
    checks.publishedStatementValid (checks.publishedStatement proof)
  expectedIvcPublicImage_eq_canonical :
    checks.expectedIvcPublicImage (checks.publishedStatement proof) =
      some (checks.canonicalIvcPublicImage steps image)
  finalClaimsCanonical :
    checks.finalClaimsCanonical (checks.finalMainClaims proof)
  finalClaimsBindPublicImage :
    checks.finalClaimsBindPublicImage
      (checks.canonicalIvcPublicImage steps image)
      (checks.finalMainClaims proof)
  finalCeVerifierAccepts :
    checks.finalCeVerifierAccepts
      (checks.finalMainClaims proof)
      (checks.finalCeProof proof)
  compressedProofDigestReplay :
    checks.compressedProofDigest proof =
      checks.recomputedCompressedProofDigest
        (checks.publishedStatement proof)
        (checks.proofIvcPublicImage proof)
        (checks.terminalCommittedProof proof)
        (checks.finalMainClaims proof)
        (checks.finalCeProof proof)
  bindingDigestReplay :
    checks.bindingDigest proof =
      checks.recomputedBindingDigest
        (checks.publishedStatement proof)
        (checks.proofIvcPublicImage proof)
        (checks.terminalCommittedProof proof)
        (checks.finalMainClaims proof)
        (checks.finalCeProof proof)
  statementPublicValid :
    checks.statementPublicValid
      (checks.canonicalIvcPublicImage steps image)
  construction2Boundary_eq :
    checks.construction2Boundary
        (checks.proofIvcPublicImage proof) =
      checks.construction2Boundary
        (checks.canonicalIvcPublicImage steps image)
  terminalVerifierPublicIO_eq :
    checks.terminalVerifierPublicIO
      (checks.terminalCommittedProof proof) =
        some publicIO
  terminalPublicValues_eq :
    publicIO.terminal =
      checks.terminalPublicValues
        (checks.canonicalIvcPublicImage steps image)
  boundaryPublicValues_eq :
    publicIO.boundary =
      checks.boundaryPublicValues
        (checks.construction2Boundary
          (checks.canonicalIvcPublicImage steps image))

/-- Accepted exact verification exposes a flat audit package. -/
theorem productionExactVerifierAccepted_audit
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    {checks :
      ProductionExactPriorVerifierChecks (PriorProof := PriorProof) ctx}
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    {publicIO : ExactTerminalBoundaryPublicIO checks.PublicField}
    (hAccepted :
      ProductionExactVerifierAccepted checks steps proof image publicIO) :
    ProductionExactVerifierAcceptedAudit
      checks
      steps
      proof
      image
      publicIO where
  compactImageReplay :=
    productionExactVerifierAccepted_compactImageReplay hAccepted
  construction2BoundaryReplay :=
    productionExactVerifierAccepted_construction2BoundaryReplay hAccepted
  poseidon2TranscriptReplay :=
    productionExactVerifierAccepted_poseidon2TranscriptReplay hAccepted
  proofIvcPublicImage_eq_canonical :=
    productionExactVerifierAccepted_proofIvcPublicImage_eq_canonical hAccepted
  proofPublicImage_eq :=
    productionExactVerifierAccepted_proofPublicImage_eq hAccepted
  proofStepCount_eq :=
    productionExactVerifierAccepted_proofStepCount_eq hAccepted
  publishedStatementValid :=
    productionExactVerifierAccepted_publishedStatementValid hAccepted
  expectedIvcPublicImage_eq_canonical :=
    productionExactVerifierAccepted_expectedIvcPublicImage_eq_canonical hAccepted
  finalClaimsCanonical :=
    productionExactVerifierAccepted_finalClaimsCanonical hAccepted
  finalClaimsBindPublicImage :=
    productionExactVerifierAccepted_finalClaimsBindPublicImage hAccepted
  finalCeVerifierAccepts :=
    productionExactVerifierAccepted_finalCeVerifierAccepts hAccepted
  compressedProofDigestReplay :=
    productionExactVerifierAccepted_compressedProofDigestReplay hAccepted
  bindingDigestReplay :=
    productionExactVerifierAccepted_bindingDigestReplay hAccepted
  statementPublicValid :=
    productionExactVerifierAccepted_statementPublicValid hAccepted
  construction2Boundary_eq :=
    productionExactVerifierAccepted_construction2Boundary_eq hAccepted
  terminalVerifierPublicIO_eq :=
    productionExactVerifierAccepted_terminalVerifierPublicIO_eq hAccepted
  terminalPublicValues_eq :=
    productionExactVerifierAccepted_terminalPublicValues_eq hAccepted
  boundaryPublicValues_eq :=
    productionExactVerifierAccepted_boundaryPublicValues_eq hAccepted

/-- The flat audit package is equivalent to exact verifier acceptance. -/
theorem productionExactVerifierAccepted_of_audit
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    {checks :
      ProductionExactPriorVerifierChecks (PriorProof := PriorProof) ctx}
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    {publicIO : ExactTerminalBoundaryPublicIO checks.PublicField}
    (hAudit :
      ProductionExactVerifierAcceptedAudit
        checks
        steps
        proof
        image
        publicIO) :
    ProductionExactVerifierAccepted checks steps proof image publicIO :=
  ⟨hAudit.compactImageReplay,
    hAudit.construction2BoundaryReplay,
    hAudit.poseidon2TranscriptReplay,
    hAudit.statementPublicValid,
    hAudit.construction2Boundary_eq,
    hAudit.terminalVerifierPublicIO_eq,
    hAudit.terminalPublicValues_eq,
    hAudit.boundaryPublicValues_eq⟩

/--
One exact verifier proof cannot bind two different production public pairs.

This is a verifier-surface statement-binding fact: it follows from replaying
the proof-carried IVC public image, before using folded F' reachability.
-/
theorem productionExactVerifierAccepted_sameProofStatement
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    {checks :
      ProductionExactPriorVerifierChecks (PriorProof := PriorProof) ctx}
    {stepsA stepsB : Nat}
    {proof : PriorProof}
    {imageA imageB : PublicImage Digest Boundary}
    {publicIOA publicIOB :
      ExactTerminalBoundaryPublicIO checks.PublicField}
    (hA :
      ProductionExactVerifierAccepted
        checks
        stepsA
        proof
        imageA
        publicIOA)
    (hB :
      ProductionExactVerifierAccepted
        checks
        stepsB
        proof
        imageB
        publicIOB) :
    stepsA = stepsB ∧ imageA = imageB := by
  have hStepA :=
    productionExactVerifierAccepted_proofStepCount_eq hA
  have hStepB :=
    productionExactVerifierAccepted_proofStepCount_eq hB
  have hImageA :=
    productionExactVerifierAccepted_proofPublicImage_eq hA
  have hImageB :=
    productionExactVerifierAccepted_proofPublicImage_eq hB
  exact ⟨hStepA.symm.trans hStepB, hImageA.symm.trans hImageB⟩

/--
One exact verifier proof cannot expose two different terminal public-IO objects.
-/
theorem productionExactVerifierAccepted_sameProofPublicIO
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    {checks :
      ProductionExactPriorVerifierChecks (PriorProof := PriorProof) ctx}
    {stepsA stepsB : Nat}
    {proof : PriorProof}
    {imageA imageB : PublicImage Digest Boundary}
    {publicIOA publicIOB :
      ExactTerminalBoundaryPublicIO checks.PublicField}
    (hA :
      ProductionExactVerifierAccepted
        checks
        stepsA
        proof
        imageA
        publicIOA)
    (hB :
      ProductionExactVerifierAccepted
        checks
        stepsB
        proof
        imageB
        publicIOB) :
    publicIOA = publicIOB := by
  have hPublicIOA :=
    productionExactVerifierAccepted_terminalVerifierPublicIO_eq hA
  have hPublicIOB :=
    productionExactVerifierAccepted_terminalVerifierPublicIO_eq hB
  have hSome : some publicIOA = some publicIOB := by
    rw [← hPublicIOA, hPublicIOB]
  cases hSome
  rfl

/--
The flat exact verifier audit is statement-functional for a fixed proof.
-/
theorem productionExactVerifierAcceptedAudit_sameProofStatement
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    {checks :
      ProductionExactPriorVerifierChecks (PriorProof := PriorProof) ctx}
    {stepsA stepsB : Nat}
    {proof : PriorProof}
    {imageA imageB : PublicImage Digest Boundary}
    {publicIOA publicIOB :
      ExactTerminalBoundaryPublicIO checks.PublicField}
    (hA :
      ProductionExactVerifierAcceptedAudit
        checks
        stepsA
        proof
        imageA
        publicIOA)
    (hB :
      ProductionExactVerifierAcceptedAudit
        checks
        stepsB
        proof
        imageB
        publicIOB) :
    stepsA = stepsB ∧ imageA = imageB :=
  productionExactVerifierAccepted_sameProofStatement
    (productionExactVerifierAccepted_of_audit hA)
    (productionExactVerifierAccepted_of_audit hB)

/--
The flat exact verifier audit is terminal-public-IO-functional for a fixed proof.
-/
theorem productionExactVerifierAcceptedAudit_sameProofPublicIO
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    {checks :
      ProductionExactPriorVerifierChecks (PriorProof := PriorProof) ctx}
    {stepsA stepsB : Nat}
    {proof : PriorProof}
    {imageA imageB : PublicImage Digest Boundary}
    {publicIOA publicIOB :
      ExactTerminalBoundaryPublicIO checks.PublicField}
    (hA :
      ProductionExactVerifierAcceptedAudit
        checks
        stepsA
        proof
        imageA
        publicIOA)
    (hB :
      ProductionExactVerifierAcceptedAudit
        checks
        stepsB
        proof
        imageB
        publicIOB) :
    publicIOA = publicIOB :=
  productionExactVerifierAccepted_sameProofPublicIO
    (productionExactVerifierAccepted_of_audit hA)
    (productionExactVerifierAccepted_of_audit hB)

/--
Production-level opening certificate for structured exact IO.
-/
structure ProductionExactPriorOpeningSurface
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : ProductionContext Digest Boundary n params) where
  checks :
    ProductionExactPriorVerifierChecks (PriorProof := PriorProof) ctx
  productionExactBackendOpens :
    ∀ steps proof image publicIO,
      ProductionExactBoundStatementAccepted
        checks
        steps
        proof
        image
        publicIO →
        ∃ authority : ProofCarryingPriorProof ctx,
          checks.openAuthority proof = some authority
  openedAuthorityBindsProductionExactStatement :
    ∀ steps proof image publicIO authority,
      ProductionExactBoundStatementAccepted
        checks
        steps
        proof
        image
        publicIO →
      checks.openAuthority proof = some authority →
        authority.steps = steps ∧ authority.image = image

/-- Runtime verifier predicate induced by the structured production exact view. -/
def RuntimeVerifyPriorOfProductionExact
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ProductionExactPriorOpeningSurface (PriorProof := PriorProof) ctx) :
    Nat →
      PriorProof →
      PublicImage Digest Boundary →
        Prop :=
  fun steps proof image =>
    ∃ publicIO : ExactTerminalBoundaryPublicIO surface.checks.PublicField,
      ProductionExactVerifierAccepted
        surface.checks
        steps
        proof
        image
        publicIO

/-- Audit-facing evidence for accepted structured production exact verification. -/
def AcceptedProductionExactEvidence
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ProductionExactPriorOpeningSurface (PriorProof := PriorProof) ctx)
    (steps : Nat)
    (proof : PriorProof)
    (image : PublicImage Digest Boundary) : Prop :=
  (∃ publicIO : ExactTerminalBoundaryPublicIO surface.checks.PublicField,
    ProductionExactBoundStatementAccepted
      surface.checks
      steps
      proof
      image
      publicIO) ∧
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

/-- Structured production exact verification yields a bound statement witness. -/
theorem runtimeVerifyPriorOfProductionExact_boundStatement
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ProductionExactPriorOpeningSurface (PriorProof := PriorProof) ctx)
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    (hVerify :
      RuntimeVerifyPriorOfProductionExact surface steps proof image) :
    ∃ publicIO : ExactTerminalBoundaryPublicIO surface.checks.PublicField,
      ProductionExactBoundStatementAccepted
        surface.checks
        steps
        proof
        image
        publicIO := by
  rcases hVerify with ⟨publicIO, hAccepted⟩
  exact
    ⟨publicIO,
      productionExactBoundStatementAccepted_ofAccepted hAccepted⟩

/-- Structured production exact acceptance opens folded F' authority. -/
theorem runtimeVerifyPriorOfProductionExact_acceptedOpens
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ProductionExactPriorOpeningSurface (PriorProof := PriorProof) ctx) :
    ∀ steps proof image,
      RuntimeVerifyPriorOfProductionExact surface steps proof image →
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
    runtimeVerifyPriorOfProductionExact_boundStatement
      surface
      hVerify with
    ⟨publicIO, hBound⟩
  rcases
    surface.productionExactBackendOpens
      steps
      proof
      image
      publicIO
      hBound with
    ⟨authority, hOpen⟩
  rcases
    surface.openedAuthorityBindsProductionExactStatement
      steps
      proof
      image
      publicIO
      authority
      hBound
      hOpen with
    ⟨hSteps, hImage⟩
  exact ⟨authority, hOpen, ⟨hSteps, hImage⟩⟩

/-- Structured production exact verification exposes checks and authority. -/
theorem runtimeVerifyPriorOfProductionExact_evidence
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ProductionExactPriorOpeningSurface (PriorProof := PriorProof) ctx)
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    (hVerify :
      RuntimeVerifyPriorOfProductionExact surface steps proof image) :
    AcceptedProductionExactEvidence surface steps proof image :=
  ⟨runtimeVerifyPriorOfProductionExact_boundStatement surface hVerify,
    runtimeVerifyPriorOfProductionExact_acceptedOpens
      surface
      steps
      proof
      image
      hVerify⟩

/-- Certified prior verifier induced by structured production exact checks. -/
def certifiedPriorVerifierOfProductionExact
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ProductionExactPriorOpeningSurface (PriorProof := PriorProof) ctx) :
    DirectParentOnlyProductionSuperNeoReuseCertifiedVerifier.CertifiedPriorVerifier
      (PriorProof := PriorProof)
      ctx :=
  DirectParentOnlyProductionSuperNeoReuseCertifiedVerifier.CertifiedPriorVerifier.ofAcceptedOpens
    ctx
    (RuntimeVerifyPriorOfProductionExact surface)
    ({ openAuthority := surface.checks.openAuthority } :
      DirectParentOnlyProductionSuperNeoReuseReplayEndpoint.PriorAuthorityOpener
        (PriorProof := PriorProof)
        ctx)
    (runtimeVerifyPriorOfProductionExact_acceptedOpens surface)

/-- The structured production exact certified verifier uses the exact predicate. -/
theorem certifiedPriorVerifierOfProductionExact_verify
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ProductionExactPriorOpeningSurface (PriorProof := PriorProof) ctx) :
    (certifiedPriorVerifierOfProductionExact surface).verify =
      RuntimeVerifyPriorOfProductionExact surface :=
  rfl

/-- Strict `SoundVerifier` induced by structured production exact checks. -/
def soundVerifierOfProductionExact
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ProductionExactPriorOpeningSurface (PriorProof := PriorProof) ctx) :
    DirectParentOnlyProductionSuperNeoReuse.ProductionContext.SoundPriorVerifier
      (PriorProof := PriorProof)
      ctx :=
  DirectParentOnlyProductionSuperNeoReuseCertifiedVerifier.CertifiedPriorVerifier.soundVerifier
    (certifiedPriorVerifierOfProductionExact surface)

/-- The structured production exact `SoundVerifier` accepts exactly exact checks. -/
theorem soundVerifierOfProductionExact_accepts_iff
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ProductionExactPriorOpeningSurface (PriorProof := PriorProof) ctx)
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary} :
    CompressedFPrimeAuthority.SoundVerifier.Accepts
        (soundVerifierOfProductionExact surface)
        steps
        proof
        image <->
      RuntimeVerifyPriorOfProductionExact
        surface
        steps
        proof
        image := by
  simpa [soundVerifierOfProductionExact,
    certifiedPriorVerifierOfProductionExact_verify]
    using
      DirectParentOnlyProductionSuperNeoReuseCertifiedVerifier.CertifiedPriorVerifier.soundVerifier_accepts_iff
        (certifiedPriorVerifierOfProductionExact surface)

/-- Structured production exact acceptance reaches the claimed prior image. -/
theorem runtimeVerifyPriorOfProductionExact_reaches_prior
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ProductionExactPriorOpeningSurface (PriorProof := PriorProof) ctx)
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    (hVerify :
      RuntimeVerifyPriorOfProductionExact surface steps proof image) :
    FPrimeInduction.Reachable
      (DirectParentOnlyProductionSoundness.Transition
        ctx.toProductionContext)
      ctx.initial
      steps
      image :=
  DirectParentOnlyProductionSuperNeoReuseCertifiedVerifier.CertifiedPriorVerifier.reaches_prior
    (certifiedPriorVerifierOfProductionExact surface)
    hVerify

/-- Structured production exact verifier is same-proof functional. -/
theorem proofFunctionalOfProductionExact
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ProductionExactPriorOpeningSurface (PriorProof := PriorProof) ctx) :
    CompressedFPrimeAuthority.ProofFunctional
      (RuntimeVerifyPriorOfProductionExact surface) :=
  DirectParentOnlyProductionSuperNeoReuseCertifiedVerifier.CertifiedPriorVerifier.proofFunctional
    (certifiedPriorVerifierOfProductionExact surface)

/-- One structured production exact proof cannot verify for two public pairs. -/
theorem soundVerifierOfProductionExact_sameProof
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    {surface :
      ProductionExactPriorOpeningSurface (PriorProof := PriorProof) ctx}
    {stepsA stepsB : Nat}
    {proof : PriorProof}
    {imageA imageB : PublicImage Digest Boundary}
    (hA :
      CompressedFPrimeAuthority.SoundVerifier.Accepts
        (soundVerifierOfProductionExact surface)
        stepsA
        proof
        imageA)
    (hB :
      CompressedFPrimeAuthority.SoundVerifier.Accepts
        (soundVerifierOfProductionExact surface)
        stepsB
        proof
        imageB) :
    stepsA = stepsB ∧ imageA = imageB :=
  proofFunctionalOfProductionExact surface hA hB

/-- Structured production exact prior-plus-latest end-to-end theorem. -/
theorem certifiedSingleTerminalEndToEnd_ofProductionExactLatestStep
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ProductionExactPriorOpeningSurface (PriorProof := PriorProof) ctx)
    {priorSteps : Nat}
    {priorProof : PriorProof}
    {priorImage nextImage : PublicImage Digest Boundary}
    {latestProof : Unit}
    (hPrior :
      RuntimeVerifyPriorOfProductionExact
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
      (certifiedPriorVerifierOfProductionExact surface).opening
      priorSteps
      priorSteps
      priorProof
      priorImage
      priorImage
      nextImage
      nextImage :=
  DirectParentOnlyProductionSuperNeoReuseEndToEnd.certifiedSingleTerminalEndToEnd_ofCertifiedPriorVerifierLatestStep
    (certifiedPriorVerifierOfProductionExact surface)
    hPrior
    hLatest


end DirectParentOnlyProductionConcreteFPrimePriorRawProduction

end DirectCcsFPrime
