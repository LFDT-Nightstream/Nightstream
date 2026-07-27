import Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc.Production

/-!
Focused regressions for the selected production Split-NC ideal-interactive
mixing theorem.

The companion `PiCcsSplitNcProductionMixingBoundary` regression retains the
zero polynomial, degree-zero, degree-at-least-support-cardinality, malformed
empty/duplicate support, maximally correlated core, and adaptive-NC
counterexamples. This file checks that the positive constructor avoids those
obstructions without weakening them.
-/

set_option autoImplicit false

namespace tests.PiCcsSplitNcIdealInteractiveMixing

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
open Nightstream.SuperNeo.InteractiveReduction.FiniteUniform
open Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc
open Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc.ProductionMixingBoundary

open Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc.ProductionMixingBoundary.IdealInteractiveCarrier
open Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc.ProductionMixingBoundary.IdealInteractiveExecution
open Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc.ProductionMixingBoundary.IdealInteractiveFeSoundness
open Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc.ProductionMixingBoundary.IdealInteractiveNcMixing
open Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc.ProductionMixingBoundary.IdealInteractiveSoundness

private abbrev ops := ConcreteCarrier.extensionOps

def twoPointSupport : Support K where
  values := [K.zero, K.one]
  nodup := by decide
  nonempty := by decide

example : twoPointSupport.cardinality = 2 := by
  rfl

example {shape : SemanticShape} :
    (support (shape := shape) twoPointSupport).values.Nodup :=
  (support (shape := shape) twoPointSupport).nodup

example {shape : SemanticShape} :
    (support (shape := shape) twoPointSupport).values ≠ [] :=
  (support (shape := shape) twoPointSupport).nonempty

/-! The selected carrier owns one shared gamma and exact delayed order. -/

#check derivePreSumcheck_shared_gamma
#check derivePreSumcheck_delayed
#check input_supportAligned

/-! FE completes before the adaptive NC callback receives its word. Each
round callback still receives only its typed prior prefix. -/

#check IdealInteractiveExecution.Strategy
#check Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.BlockLane.CausalSoundness.NcStrategy.message
#check certificate_fe_coordinates
#check certificate_nc_coordinates

/-! Exact named root families and their finite bounds. -/

#check mixingRootEvent_eq_true_iff
#check mixingRoot_probability_le
#check ncMixingRootEvent_eq_true_iff
#check ncMixingRoot_probability_le

/-! The final monitor is extensionally the actual dependent production
failure family, and the headline theorem has no generic soundness-contract
argument. -/

#check algebraicFailureEvent_eq_namedFailureEvent
#check namedFailure_probability_le
#check namedFailure_probability_le_of_productionField

universe uState

example
    {shape : SemanticShape}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits : ringDegree * publicRingColumns <= shape.carrierWidth}
    (alphabet : Support K)
    (baseInput : ProductionRefinement.AuthoritativeInput
      (shape := shape) (State := State)
      (publicRingColumns := publicRingColumns)
      (verifierRows := verifierRows)
      (publicFits := publicFits))
    (strategy : IdealInteractiveExecution.Strategy baseInput)
    (suffix : Seed shape ->
      Suffix shape publicRingColumns verifierRows publicFits)
    (noZeroDivisors : FiniteRootCounting.NoZeroDivisors ops) :
    ((support (shape := shape) alphabet).uniform).probabilityBool
        (namedFailureEvent alphabet baseInput strategy suffix) <=
      totalBudget baseInput alphabet.cardinality :=
  namedFailure_probability_le alphabet baseInput strategy suffix
    noZeroDivisors

end tests.PiCcsSplitNcIdealInteractiveMixing
