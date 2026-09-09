import NightstreamFPrime.Lifecycle.Nifs.WeakExtraction
import NightstreamFPrime.Spec.Folding.Nifs.PaperWeakAbort
import NightstreamFPrime.Spec.Folding.Nifs.InteractivePrefix

/-!
Only positive-context receipts of the actual checked prefix need a suffix
algorithm with a finite call moment. A proof-only total extension supplies
an abort-only algorithm elsewhere. No support test is charged as an executed
step: every sampled positive receipt uses the original supported algorithm.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Lifecycle.Nifs.SupportedContinuation

attribute [local instance] Classical.propDecidable
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open StrongReduction ConcreteCarrier
open _root_.NightstreamFPrime.Spec.Folding.Nifs
open _root_.NightstreamFPrime.Lifecycle.PaperAlgebra
open PiRLC.CoordinateForkLaw

variable (Tape : Type*) {Context State : Type*} {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth}
  (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
  (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
  (running : Context → Lifecycle.Running (logicalWidth := logicalWidth) (publicFits := publicFits))
  (fresh : Context → Lifecycle.Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
  (contexts : PMF Context)
  (firstPhase : Context → InteractivePrefix.Prover State productionShape 9)

/-- This is the checked prefix used by both probability and work. The public
coins, full output and captured state must be those of one actual receipt. -/
def Supported (context : Context) (coins : PublicCoins K productionShape)
    (output : FullOutputCoordinates.FullOutput K productionShape) (state : State) : Prop :=
  0 < (contexts context).toReal ∧
    ∃ probe : Probe K productionShape,
      InteractivePrefix.run (firstPhase context) coins.alpha coins.gamma coins.roundPoint =
        some (probe, state) ∧ probe.response.fullOutput = output

/-- Every actual receipt in a positive context satisfies the support condition
at its literal public coins, full output and captured private state. -/
theorem return_supported (context : Context)
    (positive : 0 < (contexts context).toReal)
    (alpha : CubePoint K productionShape.cubeVariables) (gamma : K)
    (point : CubePoint K productionShape.cubeVariables)
    (receipt : Probe K productionShape × State)
    (returned : InteractivePrefix.run (firstPhase context) alpha gamma point = some receipt) :
    Supported contexts firstPhase context receipt.1.coins receipt.1.response.fullOutput receipt.2 := by
  refine ⟨positive, receipt.1, ?_, rfl⟩
  have coins := InteractivePrefix.run_coins (firstPhase context) alpha gamma point receipt returned
  rw [coins]
  exact returned

/-- No algorithm or moment proof is requested for an impossible receipt or a
zero-mass context. All semantic parameters remain the selected production key. -/
abbrev Provider :=
  ∀ context (coins : PublicCoins K productionShape)
    (output : FullOutputCoordinates.FullOutput K productionShape) (state : State),
    Supported contexts firstPhase context coins output state →
      WeakExtraction.Continuation Tape relation ajtai (running context) (fresh context) coins output

variable {Tape}
  (abortTape : Tape)
  (provider : Provider Tape relation ajtai running fresh contexts firstPhase)

/-- Totalize only the mathematical family required by the coupling theorem.
Outside support the actual call is the proved constant abort, not an assumed law. -/
noncomputable def extension (context : Context) (coins : PublicCoins K productionShape)
    (output : FullOutputCoordinates.FullOutput K productionShape) (state : State) :
    WeakExtraction.Continuation Tape relation ajtai (running context) (fresh context) coins output :=
  if support : Supported contexts firstPhase context coins output state then
    provider context coins output state support
  else
    PaperWeakAbort.algorithm (ProductionKey.key relation ajtai).piRlcAlgebra
      (WeakExtraction.batchForOutput relation ajtai (running context) (fresh context) coins output)
      (ProductionKey.key relation ajtai).piDecAlgebra
      (ProductionKey.key relation ajtai).piDecPublicInputSplit
      (ProductionKey.key relation ajtai).piDecEvaluationArity abortTape

/-- The extension is exactly the supplied algorithm wherever the experiment
can sample a continuation. The proof argument carries no runtime data. -/
theorem extension_eq_on_support (context : Context) (coins : PublicCoins K productionShape)
    (output : FullOutputCoordinates.FullOutput K productionShape) (state : State)
    (support : Supported contexts firstPhase context coins output state) :
    extension relation ajtai running fresh contexts firstPhase abortTape provider context coins output state =
      provider context coins output state support := by
  simp only [extension, dif_pos support]

/-- In particular, an observed positive receipt preserves the same private
tape law, actual call, checks and clocks, not only its success rate. -/
theorem extension_eq_on_return (context : Context)
    (positive : 0 < (contexts context).toReal)
    (alpha : CubePoint K productionShape.cubeVariables) (gamma : K)
    (point : CubePoint K productionShape.cubeVariables)
    (receipt : Probe K productionShape × State)
    (returned : InteractivePrefix.run (firstPhase context) alpha gamma point = some receipt) :
    extension relation ajtai running fresh contexts firstPhase abortTape provider context
        receipt.1.coins receipt.1.response.fullOutput receipt.2 =
      provider context receipt.1.coins receipt.1.response.fullOutput receipt.2
        (return_supported contexts firstPhase context positive alpha gamma point receipt returned) :=
  extension_eq_on_support relation ajtai running fresh contexts firstPhase abortTape provider context
    receipt.1.coins receipt.1.response.fullOutput receipt.2
    (return_supported contexts firstPhase context positive alpha gamma point receipt returned)

/-- Impossible inputs have no raw reply and perform no raw suffix work. -/
theorem offsupport_rawCall_none (context : Context) (coins : PublicCoins K productionShape)
    (output : FullOutputCoordinates.FullOutput K productionShape) (state : State)
    (unsupported : ¬ Supported contexts firstPhase context coins output state)
    (vector : Fin (ProductionKey.key relation ajtai).arity.total →
      Challenge (ProductionKey.key relation ajtai).piRlcAlgebra) (tape : Tape) :
    ((extension relation ajtai running fresh contexts firstPhase abortTape provider
      context coins output state).rawCall vector tape).value = none ∧
    ((extension relation ajtai running fresh contexts firstPhase abortTape provider
      context coins output state).rawCall vector tape).work = 0 := by
  simp only [extension, dif_neg unsupported]
  exact PaperWeakAbort.rawCall_eq _ _ _ _ _ abortTape vector tape

/-- No semantic checker or recomposition executes off support. The two
observed control steps are the abort dispatch and outer return. -/
theorem offsupport_run_eq (context : Context) (coins : PublicCoins K productionShape)
    (output : FullOutputCoordinates.FullOutput K productionShape) (state : State)
    (unsupported : ¬ Supported contexts firstPhase context coins output state)
    (vector : Fin (ProductionKey.key relation ajtai).arity.total →
      Challenge (ProductionKey.key relation ajtai).piRlcAlgebra) (tape : Tape) :
    let selected := extension relation ajtai running fresh contexts firstPhase abortTape provider
      context coins output state
    PaperWeakOracle.run (ProductionKey.key relation ajtai).piRlcAlgebra
      selected.suffixProgram selected.rawCall vector tape = ⟨none, 2⟩ := by
  dsimp only
  simp only [extension, dif_neg unsupported]
  exact PaperWeakAbort.run_eq _ _ _ _ _ abortTape vector tape

end NightstreamFPrime.Lifecycle.Nifs.SupportedContinuation
