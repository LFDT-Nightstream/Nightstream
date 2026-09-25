import NightstreamFPrime.Lifecycle.XOut
import NightstreamFPrime.Lifecycle.ProductionKey
import NightstreamFPrime.Spec.Folding.Nifs.PaperStrongInterface
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.StrongExtraction

/-!
Definition 17 for the selected NIFS PiCCS prefix. Every statement, opening
map, field law, and degree comes from ProductionKey. The probability-only
entry supports a mathematical suffix coupling. The costed entry charges an
actual one-call implementation; it does not charge sampling a coupling table.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Lifecycle.Nifs.StrongExtraction

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open StrongReduction ConcreteCarrier WitnessProjection
open NightstreamFPrime.Lifecycle
open _root_.NightstreamFPrime.Lifecycle.PaperAlgebra
open _root_.NightstreamFPrime.Lifecycle.ProductionKey

variable {Context Tape : Type*} {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth}
  (relation : Context → LogicalRelation logicalWidth publicFits)
  (ajtai : Context → AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
  (running : Context → Lifecycle.Running (logicalWidth := logicalWidth) (publicFits := publicFits))
  (fresh : Context → Lifecycle.Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))

/-- The strong statement is the literal statement selected by the NIFS key. -/
noncomputable def statement (context : Context) :
    StrongReduction.Statement K PaperAlgebra.Commitment
      (PaperAlgebra.PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits))
      productionShape (Phi81CarrierLayout.carrierWidth logicalWidth)
      (Phi81ColumnLayout.blockCount (Phi81CarrierLayout.carrierWidth logicalWidth)) baseOps :=
  (ProductionKey.key (relation context) (ajtai context)).statement
    (running context) (fresh context)

variable (contexts : PMF Context) (tapes : Context → PMF Tape)
  (prover : Context → Tape → CausalExecution.Prover productionShape
    (Phi81CarrierLayout.carrierWidth logicalWidth) 9)

/-- Probability-only form for the actual composed causal prover. A coupling
law is permitted here, with no assertion that its whole table is executed. -/
theorem source_success_ge :
    StrongProbability.globalSuccessProbability contexts tapes prover
        (fun context => PaperAlgebra.openingMaps (ajtai context)) productionGlobalParams
        (statement relation ajtai running fresh) -
      Real.sqrt (StrongProbability.globalDisagreementProbability contexts tapes prover
        (fun context => PaperAlgebra.openingMaps (ajtai context)) productionGlobalParams
        (statement relation ajtai running fresh) + IndependentExecution.testError productionShape 9) ≤
      StrongProbability.globalSourceProbability contexts tapes prover
        (fun context => PaperAlgebra.openingMaps (ajtai context)) productionGlobalParams
        (statement relation ajtai running fresh) := by
  exact StrongProbability.source_success_ge contexts tapes prover
    (fun context => PaperAlgebra.openingMaps (ajtai context)) productionGlobalParams
    (statement relation ajtai running fresh) rfl
    (fun context => (ProductionKey.key (relation context) (ajtai context)).constantLaw)
    (fun context => (ProductionKey.key (relation context) (ajtai context)).statement_sumcheckDegreeBound_le
      (running context) (fresh context))

variable (call : Context → OneRunExtraction.Call Tape productionShape (FullShape logicalWidth publicFits))
  (program : Context → CheckedWitnessExtraction.Program productionShape (FullShape logicalWidth publicFits))
  (callCorrect : ∀ context, OneRunExtraction.CallCorrect (call context) (prover context))
  (correct : ∀ context, CheckedWitnessExtraction.Correct (width := 9) (program context)
    (PaperAlgebra.openingMaps (ajtai context)).commit productionGlobalParams
    (statement relation ajtai running fresh context))

include callCorrect correct in
/-- The actual returned tails and full running assignments have exactly the
selected NIFS source-success probability. No intermediate opening is assumed. -/
theorem successProbability_eq :
    _root_.NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.StrongExtraction.successProbability contexts tapes call program
      (fun context => (PaperAlgebra.openingMaps (ajtai context)).commit) productionGlobalParams
      (statement relation ajtai running fresh) =
    StrongProbability.globalSourceProbability contexts tapes prover
      (fun context => PaperAlgebra.openingMaps (ajtai context)) productionGlobalParams
      (statement relation ajtai running fresh) := by
  exact _root_.NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.StrongExtraction.successProbability_eq contexts tapes call program prover
    (fun context => (PaperAlgebra.openingMaps (ajtai context)).commit) productionGlobalParams
    (statement relation ajtai running fresh) callCorrect correct

include callCorrect correct in
/-- The selected strong extractor has the paper loss and expected polynomial
work under the actual global call/check and access implementation premises. -/
theorem probability_and_expected_work
    (accessBound : Nat)
    (bounded : ∀ context, CostedWitnessProjection.Bounded (program context).access accessBound)
    (baseSummable : Summable fun sample =>
      (StrongProbability.jointTapeLaw contexts tapes sample).toReal *
        StrongProbability.verifierMean (_root_.NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.StrongExtraction.baseClock call program sample))
    (securityParameter : Nat) (callPolynomial accessPolynomial : Polynomial ℝ)
    (callPPT : StrongProbability.clockMean (StrongProbability.jointTapeLaw contexts tapes)
      (_root_.NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.StrongExtraction.baseClock call program) ≤
        callPolynomial.eval (securityParameter : ℝ))
    (accessPPT : (accessBound : ℝ) ≤ accessPolynomial.eval (securityParameter : ℝ)) :
    (StrongProbability.globalSuccessProbability contexts tapes prover
        (fun context => PaperAlgebra.openingMaps (ajtai context)) productionGlobalParams
        (statement relation ajtai running fresh) -
      Real.sqrt (StrongProbability.globalDisagreementProbability contexts tapes prover
        (fun context => PaperAlgebra.openingMaps (ajtai context)) productionGlobalParams
        (statement relation ajtai running fresh) + IndependentExecution.testError productionShape 9) ≤
      _root_.NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.StrongExtraction.successProbability contexts tapes call program
        (fun context => (PaperAlgebra.openingMaps (ajtai context)).commit) productionGlobalParams
        (statement relation ajtai running fresh)) ∧
    (Summable (fun sample => (StrongProbability.jointTapeLaw contexts tapes sample).toReal *
      StrongProbability.verifierMean (_root_.NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.StrongExtraction.totalClock call program sample)) ∧
      StrongProbability.clockMean (StrongProbability.jointTapeLaw contexts tapes)
        (_root_.NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.StrongExtraction.totalClock call program) ≤
        (callPolynomial +
          Polynomial.C (productionShape.freshCount : ℝ) *
            (Polynomial.C (privateWidth (FullShape logicalWidth publicFits) : ℝ) *
              (accessPolynomial + Polynomial.C 6) + Polynomial.C 9) +
          Polynomial.C (productionShape.runningCount : ℝ) *
            (Polynomial.C ((FullShape logicalWidth publicFits).carrierWidth : ℝ) *
              (accessPolynomial + Polynomial.C 6) + Polynomial.C 9) +
          Polynomial.C 10).eval (securityParameter : ℝ)) := by
  exact _root_.NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.StrongExtraction.probability_and_expected_work contexts tapes call program prover
    (fun context => (PaperAlgebra.openingMaps (ajtai context)).commit) productionGlobalParams
    (statement relation ajtai running fresh) callCorrect correct rfl
    (fun context => (ProductionKey.key (relation context) (ajtai context)).constantLaw)
    (fun context => (ProductionKey.key (relation context) (ajtai context)).statement_sumcheckDegreeBound_le
      (running context) (fresh context))
    accessBound bounded baseSummable securityParameter callPolynomial accessPolynomial callPPT accessPPT

end NightstreamFPrime.Lifecycle.Nifs.StrongExtraction
