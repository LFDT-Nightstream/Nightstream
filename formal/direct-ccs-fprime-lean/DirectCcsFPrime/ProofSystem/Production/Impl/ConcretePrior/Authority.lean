import DirectCcsFPrime.ProofSystem.Production.Impl.ConcretePrior.Core

/-!
Concrete prior F' verifier authority consequences.

This module keeps the authority-facing elimination lemmas separate from the
base verifier-body model. The proofs expose the folded F' reachability supplied
by an accepted concrete prior verifier and rule out treating replayed digests
as authority.
-/

namespace DirectCcsFPrime

namespace DirectParentOnlyProductionConcreteFPrimePrior

/--
If the fixed opener returns a concrete authority for an accepted proof, that
exact authority accepts the same `(steps, image)` pair.
-/
theorem verifyPrior_openedAuthority_accepts_of_open
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (body : ConcreteVerifierBody (PriorProof := PriorProof) ctx)
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    {authority : ProofCarryingPriorProof ctx}
    (hVerify : VerifyPrior body steps proof image)
    (hOpen : body.openAuthority proof = some authority) :
    FoldedFPrimeAuthority.Accepts
      (Transition :=
        DirectParentOnlyProductionSoundness.Transition
          ctx.toProductionContext)
      (initial := ctx.initial)
      steps
      authority
      image := by
  rcases acceptedOpens body steps proof image hVerify with
    ⟨openedAuthority, hOpened, hAccepts⟩
  have hAuthority : openedAuthority = authority := by
    have hSome : some openedAuthority = some authority :=
      hOpened.symm.trans hOpen
    cases hSome
    rfl
  cases hAuthority
  exact hAccepts

/--
Checks-based concrete prior acceptance makes any concrete opened authority
accept the same `(steps, image)` pair.
-/
theorem verifyPriorOfChecks_openedAuthority_accepts_of_open
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (checks : ConcreteVerifierBodyChecks (PriorProof := PriorProof) ctx)
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    {authority : ProofCarryingPriorProof ctx}
    (hVerify : VerifyPriorOfChecks checks steps proof image)
    (hOpen : checks.openAuthority proof = some authority) :
    FoldedFPrimeAuthority.Accepts
      (Transition :=
        DirectParentOnlyProductionSoundness.Transition
          ctx.toProductionContext)
      (initial := ctx.initial)
      steps
      authority
      image :=
  verifyPrior_openedAuthority_accepts_of_open
    (concreteVerifierBodyOfChecks checks)
    hVerify
    hOpen

/--
Canonical-statement binding acceptance makes any concrete opened authority
accept the same `(steps, image)` pair.
-/
theorem verifyPriorOfStatementBinding_openedAuthority_accepts_of_open
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (binding :
      ConcreteVerifierStatementBinding (PriorProof := PriorProof) ctx)
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    {authority : ProofCarryingPriorProof ctx}
    (hVerify :
      VerifyPriorOfStatementBinding binding steps proof image)
    (hOpen : binding.openAuthority proof = some authority) :
    FoldedFPrimeAuthority.Accepts
      (Transition :=
        DirectParentOnlyProductionSoundness.Transition
          ctx.toProductionContext)
      (initial := ctx.initial)
      steps
      authority
      image :=
  verifyPriorOfChecks_openedAuthority_accepts_of_open
    (concreteVerifierBodyChecksOfStatementBinding binding)
    hVerify
    hOpen

/--
Statement-surface acceptance makes any concrete opened authority accept the
same `(steps, image)` pair.
-/
theorem verifyPriorOfStatementSurface_openedAuthority_accepts_of_open
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ConcreteVerifierStatementSurface (PriorProof := PriorProof) ctx)
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    {authority : ProofCarryingPriorProof ctx}
    (hVerify :
      VerifyPriorOfStatementSurface surface steps proof image)
    (hOpen : surface.openAuthority proof = some authority) :
    FoldedFPrimeAuthority.Accepts
      (Transition :=
        DirectParentOnlyProductionSoundness.Transition
          ctx.toProductionContext)
      (initial := ctx.initial)
      steps
      authority
      image :=
  verifyPriorOfStatementBinding_openedAuthority_accepts_of_open
    (concreteVerifierStatementBindingOfSurface surface)
    hVerify
    hOpen

/-- Every accepted concrete prior proof reaches its claimed prior image. -/
theorem verifyPrior_reaches_prior
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (body : ConcreteVerifierBody (PriorProof := PriorProof) ctx)
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    (hVerify : VerifyPrior body steps proof image) :
    FPrimeInduction.Reachable
      (DirectParentOnlyProductionSoundness.Transition
        ctx.toProductionContext)
      ctx.initial
      steps
      image := by
  rcases acceptedOpens body steps proof image hVerify with
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

/-- Every accepted checks-based concrete prior proof reaches its prior image. -/
theorem verifyPriorOfChecks_reaches_prior
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (checks : ConcreteVerifierBodyChecks (PriorProof := PriorProof) ctx)
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    (hVerify : VerifyPriorOfChecks checks steps proof image) :
    FPrimeInduction.Reachable
      (DirectParentOnlyProductionSoundness.Transition
        ctx.toProductionContext)
      ctx.initial
      steps
      image :=
  verifyPrior_reaches_prior
    (concreteVerifierBodyOfChecks checks)
    hVerify

/-- Canonical-statement binding acceptance reaches its prior image. -/
theorem verifyPriorOfStatementBinding_reaches_prior
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (binding :
      ConcreteVerifierStatementBinding (PriorProof := PriorProof) ctx)
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    (hVerify :
      VerifyPriorOfStatementBinding binding steps proof image) :
    FPrimeInduction.Reachable
      (DirectParentOnlyProductionSoundness.Transition
        ctx.toProductionContext)
      ctx.initial
      steps
      image :=
  verifyPriorOfChecks_reaches_prior
    (concreteVerifierBodyChecksOfStatementBinding binding)
    hVerify

/-- Statement-surface acceptance reaches its prior image. -/
theorem verifyPriorOfStatementSurface_reaches_prior
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ConcreteVerifierStatementSurface (PriorProof := PriorProof) ctx)
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    (hVerify :
      VerifyPriorOfStatementSurface surface steps proof image) :
    FPrimeInduction.Reachable
      (DirectParentOnlyProductionSoundness.Transition
        ctx.toProductionContext)
      ctx.initial
      steps
      image :=
  verifyPriorOfStatementBinding_reaches_prior
    (concreteVerifierStatementBindingOfSurface surface)
    hVerify

/-- A concrete prior verifier cannot accept an unreachable prior image. -/
theorem verifyPrior_cannot_accept_unreachable_prior
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (body : ConcreteVerifierBody (PriorProof := PriorProof) ctx)
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    (hVerify : VerifyPrior body steps proof image)
    (hUnreachable :
      ¬ FPrimeInduction.Reachable
        (DirectParentOnlyProductionSoundness.Transition
          ctx.toProductionContext)
        ctx.initial
        steps
        image) :
    False :=
  hUnreachable (verifyPrior_reaches_prior body hVerify)

/-- A checks-based concrete prior verifier cannot accept an unreachable image. -/
theorem verifyPriorOfChecks_cannot_accept_unreachable_prior
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (checks : ConcreteVerifierBodyChecks (PriorProof := PriorProof) ctx)
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    (hVerify : VerifyPriorOfChecks checks steps proof image)
    (hUnreachable :
      ¬ FPrimeInduction.Reachable
        (DirectParentOnlyProductionSoundness.Transition
          ctx.toProductionContext)
        ctx.initial
        steps
        image) :
    False :=
  hUnreachable
    (verifyPriorOfChecks_reaches_prior checks hVerify)

/--
Canonical-statement binding verifier acceptance cannot authorize an
unreachable image.
-/
theorem verifyPriorOfStatementBinding_cannot_accept_unreachable_prior
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (binding :
      ConcreteVerifierStatementBinding (PriorProof := PriorProof) ctx)
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    (hVerify :
      VerifyPriorOfStatementBinding binding steps proof image)
    (hUnreachable :
      ¬ FPrimeInduction.Reachable
        (DirectParentOnlyProductionSoundness.Transition
          ctx.toProductionContext)
        ctx.initial
        steps
        image) :
    False :=
  hUnreachable
    (verifyPriorOfStatementBinding_reaches_prior binding hVerify)

/--
Statement-surface verifier acceptance cannot authorize an unreachable image.
-/
theorem verifyPriorOfStatementSurface_cannot_accept_unreachable_prior
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ConcreteVerifierStatementSurface (PriorProof := PriorProof) ctx)
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    (hVerify :
      VerifyPriorOfStatementSurface surface steps proof image)
    (hUnreachable :
      ¬ FPrimeInduction.Reachable
        (DirectParentOnlyProductionSoundness.Transition
          ctx.toProductionContext)
        ctx.initial
        steps
        image) :
    False :=
  hUnreachable
    (verifyPriorOfStatementSurface_reaches_prior surface hVerify)

end DirectParentOnlyProductionConcreteFPrimePrior

end DirectCcsFPrime
