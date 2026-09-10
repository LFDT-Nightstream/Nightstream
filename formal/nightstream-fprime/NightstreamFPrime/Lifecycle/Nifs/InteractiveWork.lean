import NightstreamFPrime.Lifecycle.Nifs.WeakExtraction
import NightstreamFPrime.Spec.Folding.Nifs.PaperCompositionWork

/-!
Expected work of the selected interactive NIFS extractor. Each returned
PiCCS receipt selects its own continuation and charged suffix law. The
decoder is the selected key's actual returned-list decoder. Probability
coupling tables are not part of this executed work.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Lifecycle.Nifs.InteractiveWork

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open StrongReduction ConcreteCarrier
open _root_.NightstreamFPrime.Spec.Folding.Nifs
open _root_.NightstreamFPrime.Lifecycle.PaperAlgebra
open _root_.NightstreamFPrime.Lifecycle.ProductionKey
open PiRLC.PaperForkExtraction PiRLC.CoordinateForkLaw
open PiRLC.PaperForkExtractionWork

variable {Context State Tape : Type*} {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth}
  (relation : LogicalRelation logicalWidth publicFits)
  (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
  (running : Context → Lifecycle.Running (logicalWidth := logicalWidth) (publicFits := publicFits))
  (fresh : Context → Lifecycle.Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
  (continuation : ∀ context (coins : PublicCoins K productionShape)
    (output : FullOutputCoordinates.FullOutput K productionShape), State →
      WeakExtraction.Continuation Tape relation ajtai (running context) (fresh context) coins output)

variable [DecidableEq RingF]
  [Fintype (Challenge (ProductionKey.key relation ajtai).piRlcAlgebra)]
  [Nonempty (Challenge (ProductionKey.key relation ajtai).piRlcAlgebra)]

/-- The response and clock law comes from the continuation selected by this
actual public output and captured private state. -/
noncomputable def law (context : Context) (receipt : Probe K productionShape × State) :=
  (continuation context receipt.1.coins receipt.1.response.fullOutput receipt.2).oracleLaw

/-- The same continuation supplies the costed parent-opening checker. -/
def parentChecker (context : Context) (receipt : Probe K productionShape × State) :=
  (continuation context receipt.1.coins receipt.1.response.fullOutput receipt.2).parentChecker

variable
  (call : Context → CubePoint K productionShape.cubeVariables → K →
    CubePoint K productionShape.cubeVariables → Result (Option (Probe K productionShape × State)))
  (extraction : Primitives RingF
    (PaperAlgebra.Assignment (logicalWidth := logicalWidth) (publicFits := publicFits)))
  (sourceProgram : Context → CheckedWitnessExtraction.Program productionShape
    (FullShape logicalWidth publicFits))

/-- Prefix work, one checked suffix-call mean, and the actual decode/C check
mean form the global EPT premise. -/
noncomputable def baseClock : Context → CubePoint K productionShape.cubeVariables → K →
    CubePoint K productionShape.cubeVariables → ℝ :=
  PaperCompositionWork.baseClock (ProductionKey.key relation ajtai).piRlcAlgebra call
    (law relation ajtai running fresh continuation)
    (parentChecker relation ajtai running fresh continuation) extraction sourceProgram
    (PaperWeakOutput.decode (ProductionKey.key relation ajtai))

/-- The total includes actual retries, terminal extraction, decode, checker,
and source projection for the same selected continuation. -/
noncomputable def totalClock : Context → CubePoint K productionShape.cubeVariables → K →
    CubePoint K productionShape.cubeVariables → ℝ :=
  PaperCompositionWork.totalClock (ProductionKey.key relation ajtai).piRlcAlgebra call
    (law relation ajtai running fresh continuation)
    (parentChecker relation ajtai running fresh continuation) extraction sourceProgram
    (PaperWeakOutput.decode (ProductionKey.key relation ajtai))

variable (originalFirstPhase : Context → InteractivePrefix.Prover State productionShape 9)
  (publicCheck : Context → Probe K productionShape → Bool)

/-- Public C rejection stops the program before the suffix. -/
def firstPhase (context : Context) : InteractivePrefix.Prover State productionShape 9 :=
  InteractivePrefix.checked (originalFirstPhase context) (publicCheck context)

variable (callCorrect : ∀ context alpha gamma point,
  (call context alpha gamma point).value =
    InteractivePrefix.run (firstPhase originalFirstPhase publicCheck context) alpha gamma point)

include callCorrect in
/-- Erasing the actual prefix clock gives the same checked prefix receipt
that selects the suffix in the probability experiment. -/
theorem totalClock_on_checked_prefix (context : Context)
    (alpha : CubePoint K productionShape.cubeVariables) (gamma : K)
    (point : CubePoint K productionShape.cubeVariables) :
    totalClock relation ajtai running fresh continuation call extraction sourceProgram
        context alpha gamma point =
      ((call context alpha gamma point).work : ℝ) +
      (match InteractivePrefix.run (firstPhase originalFirstPhase publicCheck context)
          alpha gamma point with
      | none => 0
      | some receipt =>
          PiRLC.CoordinateExtraction.expectedTotalWork (ProductionKey.key relation ajtai).piRlcAlgebra
            (law relation ajtai running fresh continuation context receipt)
            (parentChecker relation ajtai running fresh continuation context receipt) extraction +
          PaperCompositionWork.finishMean (ProductionKey.key relation ajtai).piRlcAlgebra
            (law relation ajtai running fresh continuation context receipt)
            (parentChecker relation ajtai running fresh continuation context receipt) extraction
            (sourceProgram context) (PaperWeakOutput.decode (ProductionKey.key relation ajtai)) receipt.1) + 1 := by
  simp only [totalClock, PaperCompositionWork.totalClock, callCorrect]
  cases InteractivePrefix.run (firstPhase originalFirstPhase publicCheck context)
    alpha gamma point <;> rfl

omit [DecidableEq RingF]
  [Fintype (Challenge (ProductionKey.key relation ajtai).piRlcAlgebra)]
  [Nonempty (Challenge (ProductionKey.key relation ajtai).piRlcAlgebra)] in
include callCorrect in
/-- Every actual continuation receipt passed the exact selected C verifier. -/
theorem call_returns_accepted
    (publicCheck_spec : ∀ context probe, publicCheck context probe = true ↔
      probe.FixedWidthAccepted extensionOps K.embed
        ((ProductionKey.key relation ajtai).statement (running context) (fresh context)) 9)
    (context : Context) (alpha : CubePoint K productionShape.cubeVariables) (gamma : K)
    (point : CubePoint K productionShape.cubeVariables) (receipt : Probe K productionShape × State)
    (returned : (call context alpha gamma point).value = some receipt) :
    receipt.1.FixedWidthAccepted extensionOps K.embed
      ((ProductionKey.key relation ajtai).statement (running context) (fresh context)) 9 := by
  have actual : InteractivePrefix.run
      (InteractivePrefix.checked (originalFirstPhase context) (publicCheck context))
      alpha gamma point = some receipt :=
    (callCorrect context alpha gamma point).symm.trans returned
  exact (publicCheck_spec context receipt.1).mp
    (InteractivePrefix.checked_returns (originalFirstPhase context) (publicCheck context)
      alpha gamma point receipt actual).2

variable
  (laws : ExtractionAlgebra (ProductionKey.key relation ajtai).piRlcSemantics
    (ProductionKey.key relation ajtai).params (ProductionKey.key relation ajtai).piRlcAlgebra)
  (strongSet : StrongSetUnits laws.ring (ProductionKey.key relation ajtai).piRlcAlgebra.challengeValid)
  (extractionCorrect : PiRLC.PaperForkExtractionWork.Correct laws.ring laws.assignmentModule extraction)
  (bounds : PrimitiveBounds)
  (extractionBounded : PiRLC.PaperForkExtractionWork.Bounded laws.ring extraction bounds)
  (accessBound : Nat)
  (accessBounded : ∀ context, CostedWitnessProjection.Bounded (sourceProgram context).access accessBound)

include strongSet extractionCorrect extractionBounded accessBounded in
/-- The concrete sequential program is EPT from its actual global base
moment and the proved extraction/access bounds. No individual context or
private call has a uniform time cap. -/
theorem expected_work_polynomial_bound (contexts : PMF Context)
    (baseSummable : Summable fun context => (contexts context).toReal *
      StrongProbability.verifierMean
        (baseClock relation ajtai running fresh continuation call extraction sourceProgram context))
    (securityParameter : Nat) (basePolynomial primitivePolynomial accessPolynomial : Polynomial ℝ)
    (basePPT : StrongProbability.clockMean contexts
      (baseClock relation ajtai running fresh continuation call extraction sourceProgram) ≤
        basePolynomial.eval (securityParameter : ℝ))
    (primitivePPT : (bounds.coordinateWork : ℝ) ≤
      primitivePolynomial.eval (securityParameter : ℝ))
    (accessPPT : (accessBound : ℝ) ≤ accessPolynomial.eval (securityParameter : ℝ)) :
    Summable (fun context => (contexts context).toReal * StrongProbability.verifierMean
      (totalClock relation ajtai running fresh continuation call extraction sourceProgram context)) ∧
    StrongProbability.clockMean contexts
      (totalClock relation ajtai running fresh continuation call extraction sourceProgram) ≤
      (Polynomial.C ((PaperProfile.arity.total : ℝ) + 1) * basePolynomial +
        Polynomial.C (PaperProfile.arity.total : ℝ) * (primitivePolynomial + Polynomial.C 3) +
        Polynomial.C (productionShape.freshCount : ℝ) *
          (Polynomial.C (WitnessProjection.privateWidth (FullShape logicalWidth publicFits) : ℝ) *
            (accessPolynomial + Polynomial.C 6) + Polynomial.C 9) +
        Polynomial.C (productionShape.runningCount : ℝ) *
          (Polynomial.C ((FullShape logicalWidth publicFits).carrierWidth : ℝ) *
            (accessPolynomial + Polynomial.C 6) + Polynomial.C 9) +
        Polynomial.C 13).eval (securityParameter : ℝ) := by
  exact PaperCompositionWork.expected_work_polynomial_bound
    (ProductionKey.key relation ajtai).piRlcAlgebra call
    (law relation ajtai running fresh continuation)
    (parentChecker relation ajtai running fresh continuation) extraction sourceProgram
    (PaperWeakOutput.decode (ProductionKey.key relation ajtai)) laws strongSet extractionCorrect
    bounds extractionBounded accessBound accessBounded contexts baseSummable securityParameter
    basePolynomial primitivePolynomial accessPolynomial basePPT primitivePPT accessPPT

end NightstreamFPrime.Lifecycle.Nifs.InteractiveWork
