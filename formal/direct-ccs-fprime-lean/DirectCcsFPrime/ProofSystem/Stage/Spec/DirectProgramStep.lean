import DirectCcsFPrime.ProofSystem.Terminal.Impl.DirectConcreteInstantiation

/-!
Deterministic direct-program step boundary for terminal F' soundness.

This module removes the abstract latest-step boundary relation from the
strongest direct CCS terminal theorem. The direct computation boundary is a
deterministic function of `(step, priorBoundary)`.

It still does not define the concrete direct CCS/R1CS transition function. That
function is the application relation supplied by a concrete frontend.
-/

namespace DirectCcsFPrime

namespace DirectProgramStep

/-- Function-computed direct boundary relation. -/
def ComputedBoundaryStep
    {Boundary : Type}
    (computeBoundary : Nat → Boundary → Boundary)
    (i : Nat)
    (prior next : Boundary) : Prop :=
  next = computeBoundary i prior

/-- A function-computed direct boundary relation is functional. -/
theorem computedBoundaryStep_functional
    {Boundary : Type}
    {computeBoundary : Nat → Boundary → Boundary} :
    ∀ i prior nextA nextB,
      ComputedBoundaryStep computeBoundary i prior nextA →
      ComputedBoundaryStep computeBoundary i prior nextB →
        nextA = nextB := by
  intro _i _prior nextA nextB hA hB
  exact hA.trans hB.symm

/--
An accepted transition under a function-computed boundary relation exposes the
exact boundary value computed by the fixed direct application step.
-/
theorem latest_currentBoundary_eq_compute
    {Digest Boundary AccHandle : Type}
    {computeBoundary : Nat → Boundary → Boundary}
    {AccumulatorStep : Nat → AccHandle → AccHandle → Prop}
    {i : Nat}
    {prior next :
      Construction2DirectFPrime.PublicImage Digest Boundary AccHandle}
    (h :
      Construction2DirectFPrime.Transition
        (ComputedBoundaryStep computeBoundary)
        AccumulatorStep
        i
        prior
        next) :
    next.currentBoundary =
      computeBoundary i prior.currentBoundary := by
  rcases h with
    ⟨_hPrior, _hNext, _hVk, _hInitial, _hPriorPc,
      _hNextPc, hBoundary, _hAcc⟩
  exact hBoundary

/--
For a deterministic boundary update, two latest transitions from the same prior
image have the same public boundary output.
-/
theorem latest_currentBoundary_functional
    {Digest Boundary AccHandle : Type}
    {computeBoundary : Nat → Boundary → Boundary}
    {AccumulatorStep : Nat → AccHandle → AccHandle → Prop}
    {i : Nat}
    {prior nextA nextB :
      Construction2DirectFPrime.PublicImage Digest Boundary AccHandle}
    (hA :
      Construction2DirectFPrime.Transition
        (ComputedBoundaryStep computeBoundary)
        AccumulatorStep
        i
        prior
        nextA)
    (hB :
      Construction2DirectFPrime.Transition
        (ComputedBoundaryStep computeBoundary)
        AccumulatorStep
        i
        prior
        nextB) :
    nextA.currentBoundary = nextB.currentBoundary := by
  rcases hA with
    ⟨_hPriorA, _hNextA, _hVkA, _hInitialA, _hPriorPcA,
      _hNextPcA, hBoundaryA, _hAccA⟩
  rcases hB with
    ⟨_hPriorB, _hNextB, _hVkB, _hInitialB, _hPriorPcB,
      _hNextPcB, hBoundaryB, _hAccB⟩
  exact hBoundaryA.trans hBoundaryB.symm

/--
For deterministic boundary updates, if the reduced accumulator fields are also
functional, then the entire latest public image is functional.
-/
theorem latest_publicImage_functional_of_accumulator_fields
    {Digest Boundary Source : Type}
    {n : Nat}
    {computeBoundary : Nat → Boundary → Boundary}
    {AccumulatorStep :
      Nat →
        ReducedAccumulatorStep.AccumulatorHandle Source n →
        ReducedAccumulatorStep.AccumulatorHandle Source n →
          Prop}
    {i : Nat}
    {prior nextA nextB :
      Construction2DirectFPrime.PublicImage
        Digest
        Boundary
        (ReducedAccumulatorStep.AccumulatorHandle Source n)}
    (hAcc :
      nextA.accumulator.parentSource =
          nextB.accumulator.parentSource ∧
        nextA.accumulator.nextPiCCSInputs =
          nextB.accumulator.nextPiCCSInputs)
    (hA :
      Construction2DirectFPrime.Transition
        (ComputedBoundaryStep computeBoundary)
        AccumulatorStep
        i
        prior
        nextA)
    (hB :
      Construction2DirectFPrime.Transition
        (ComputedBoundaryStep computeBoundary)
        AccumulatorStep
        i
        prior
        nextB) :
    nextA = nextB := by
  rcases hA with
    ⟨_hPriorA, hNextA, hVkA, hInitialA, _hPriorPcA,
      hNextPcA, hBoundaryA, _hAccA⟩
  rcases hB with
    ⟨_hPriorB, hNextB, hVkB, hInitialB, _hPriorPcB,
      hNextPcB, hBoundaryB, _hAccB⟩
  cases nextA with
  | mk vkA stepA initialA boundaryA accA pcA =>
      cases nextB with
      | mk vkB stepB initialB boundaryB accB pcB =>
          cases accA with
          | mk sourceA inputsA =>
              cases accB with
              | mk sourceB inputsB =>
                  simp only
                    [Construction2DirectFPrime.PublicImage.mk.injEq,
                      ReducedAccumulatorStep.AccumulatorHandle.mk.injEq]
                  exact
                    ⟨hVkA.symm.trans hVkB,
                      hNextA.trans hNextB.symm,
                      hInitialA.symm.trans hInitialB,
                      hBoundaryA.trans hBoundaryB.symm,
                      ⟨hAcc.1, hAcc.2⟩,
                      hNextPcA.trans hNextPcB.symm⟩

/--
Concrete direct-program terminal theorem.

Compared with `DirectConcreteInstantiation`, this theorem also fixes the direct
application boundary to a deterministic function. Therefore an accepted latest
step cannot be swapped for another accepted latest step from the same prior
image that changes either the public computation boundary or the reduced
accumulator fields.
-/
theorem terminal_soundness_of_concrete_program_and_msis
    {Digest Boundary PiCCSOut : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (data :
      DirectConcreteInstantiation.ConcreteCEData n params)
    {hashEncoded : List Nat → Digest}
    {computeBoundary : Nat → Boundary → Boundary}
    {computePiCCS :
      Nat →
        DirectTerminalSoundness.AccHandle Digest n →
        PiCCSOut}
    {computePiRLC :
      Nat → PiCCSOut → DigestParentBinding.Source Digest}
    {commitmentOfParent :
      ParentEncoding.SomeParentCEB →
        SuperNeo.ProofSystem.Commitment}
    {initial priorImage nextImage altNext :
      Construction2DirectFPrime.PublicImage
        Digest
        Boundary
        (DirectTerminalSoundness.AccHandle Digest n)}
    {priorSteps : Nat}
    {priorAuthority :
      DirectTerminalSoundness.Authority
        (params := params)
        (ComputedBoundaryStep computeBoundary)
        (ParentSourceStep.ComputedPiCCS
          (n := n)
          computePiCCS)
        (ParentSourceStep.ComputedPiRLC computePiRLC)
        hashEncoded
        data.ce
        (ParentOpeningAuthorization.StatementEncodesByCommitment
          commitmentOfParent)
        initial}
    {proof : Unit}
    (hDigest :
      ParentEncoding.EncodedParentCEBDigestBinding hashEncoded)
    (hRed :
      SuperNeo.ProofSystem.MSISToAjtaiReductions params)
    (hMsis :
      SuperNeo.ProofSystem.MSISHardnessAssumption params)
    (hAccepted :
      FPrimeInduction.TerminalCompressionAccepted
        (FoldedFPrimeAuthority.Accepts
          (Transition :=
            DirectTerminalSoundness.Transition
              (params := params)
              (ComputedBoundaryStep computeBoundary)
              (ParentSourceStep.ComputedPiCCS
                (n := n)
                computePiCCS)
              (ParentSourceStep.ComputedPiRLC computePiRLC)
              hashEncoded
              data.ce
              (ParentOpeningAuthorization.StatementEncodesByCommitment
                commitmentOfParent))
          (initial := initial))
        (Construction2DirectFPrime.VerifyLatestStep
          (Authority :=
            DirectTerminalSoundness.Authority
              (params := params)
              (ComputedBoundaryStep computeBoundary)
              (ParentSourceStep.ComputedPiCCS
                (n := n)
                computePiCCS)
              (ParentSourceStep.ComputedPiRLC computePiRLC)
              hashEncoded
              data.ce
              (ParentOpeningAuthorization.StatementEncodesByCommitment
                commitmentOfParent)
              initial)
          (ComputedBoundaryStep computeBoundary)
          (DirectTerminalSoundness.AccumulatorStep
            (params := params)
            (ParentSourceStep.ComputedPiCCS
              (n := n)
              computePiCCS)
            (ParentSourceStep.ComputedPiRLC computePiRLC)
            hashEncoded
            data.ce
            (ParentOpeningAuthorization.StatementEncodesByCommitment
              commitmentOfParent)))
        priorSteps
        priorAuthority
        priorImage
        nextImage
        proof)
    (hAlt :
      Construction2DirectFPrime.Transition
        (ComputedBoundaryStep computeBoundary)
        (DirectTerminalSoundness.AccumulatorStep
          (params := params)
          (ParentSourceStep.ComputedPiCCS
            (n := n)
            computePiCCS)
          (ParentSourceStep.ComputedPiRLC computePiRLC)
          hashEncoded
          data.ce
          (ParentOpeningAuthorization.StatementEncodesByCommitment
            commitmentOfParent))
        priorSteps
        priorImage
        altNext) :
    FPrimeInduction.Reachable
        (DirectTerminalSoundness.Transition
          (params := params)
          (ComputedBoundaryStep computeBoundary)
          (ParentSourceStep.ComputedPiCCS
            (n := n)
            computePiCCS)
          (ParentSourceStep.ComputedPiRLC computePiRLC)
          hashEncoded
          data.ce
          (ParentOpeningAuthorization.StatementEncodesByCommitment
            commitmentOfParent))
        initial
        (priorSteps + 1)
        nextImage ∧
      nextImage = altNext := by
  have hBase :=
    DirectConcreteInstantiation.terminal_soundness_of_concrete_ce_and_msis
      data
      hDigest
      hRed
      hMsis
      hAccepted
      hAlt
  constructor
  · exact hBase.1
  · exact
      latest_publicImage_functional_of_accumulator_fields
        hBase.2
        hAccepted.latestAccepted
        hAlt

end DirectProgramStep

end DirectCcsFPrime
