import NightstreamFPrime.Export.BoundedDepth.FinalVerifier
import NightstreamFPrime.Export.BoundedDepth.StepExtend
import NightstreamFPrime.Export.Generic.PackageRows

/-! The dense rows of the final verifier's setup (`SPEC.md` Algorithm 10, step 10). Rust builds
the factored rows of each circuit from its step package (`Relation::new`). Lean's model of these
rows is `packageRows`: the circuit's relation over the source columns, with the spans of the
package's slots (`Export.Generic.PackageRows.rows`). When the setup's rows have the dense
matrices of this model, `Export.BoundedDepth.FinalVerifier.Premises` holds (`premises_of_rows`).
The model represents each circuit's relation (`Export.Generic.PackageRows.rows_represents`), and
its columns are the blocks of the package's logical width
(`Export.Generic.PackageRows.rows_support`). -/

set_option autoImplicit false

namespace NightstreamFPrime.Export.BoundedDepth.FinalRows

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.BoundedDepth
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Export.Generic
open NightstreamFPrime.Export.BoundedDepth.TerminalBridge
open NightstreamFPrime.Export.BoundedDepth.StepExtend
open scoped NightstreamFPrime.Spec.Compression.Production

variable {logicalWidth : Nat}
  {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
    RingCarrierLayout.carrierWidth logicalWidth}
  {relation : ProductionKey.LogicalRelation logicalWidth publicFits}
  {ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits)}
  {circuitKeys : Fin 2 → CircuitKey logicalWidth publicFits}
  {step : AppState → AppWitness → AppState}
  {cubeFits : RingCarrierLayout.carrierWidth logicalWidth ≤ 2 ^ cubeVariables}
  {mode : FinalVerifier.Mode}

/-- The slots of the light package (`StepPackage.slotRuns`). -/
abbrev lightSlots (exports : Exports relation ajtai circuitKeys step cubeFits)
    (complete : Complete exports) :=
  SlotLayout.runSlots exports.lightWidth (StepPackage.slotRuns CircuitRelation.publicColumns
    (StepCompleteness.lightPieces exports.program relation ajtai) exports.lightFresh
    complete.lightBatches)

/-- The slots of the refresh package (`StepPackage.slotRuns`). -/
abbrev refreshSlots (exports : Exports relation ajtai circuitKeys step cubeFits)
    (complete : Complete exports) :=
  SlotLayout.runSlots exports.refreshWidth (StepPackage.slotRuns CircuitRelation.publicColumns
    (StepCompleteness.refreshPieces exports.program relation ajtai) exports.refreshFresh
    complete.refreshBatches)

/-- Lean's model of the factored rows of circuit `circuit` (Rust `FactoredRows`): the circuit's
relation through `PlanPush.identity`, with the spans of its package's slots and the blocks of
the package's logical width. -/
def packageRows (exports : Exports relation ajtai circuitKeys step cubeFits)
    (complete : Complete exports) (circuit : Fin 2) : Compression.Verifier.Factored F :=
  if circuit.val = 0 then
    PackageRows.rows (lightSlots exports complete)
      (CircuitRelation.plan _ _ exports.lightFresh exports.lightWidth exports.lightFitsPackage
        (PlanPush.identity exports.lightWidth exports.lightInside))
      (ringDegree * ConcreteRingColumnLayout.blockCount
        (ProductionAssignment.logicalWidth (lightSlots exports complete)))
  else
    PackageRows.rows (refreshSlots exports complete)
      (CircuitRelation.plan _ _ exports.refreshFresh exports.refreshWidth
        exports.refreshFitsPackage (PlanPush.identity exports.refreshWidth exports.refreshInside))
      (ringDegree * ConcreteRingColumnLayout.blockCount
        (ProductionAssignment.logicalWidth (refreshSlots exports complete)))

/-- When the setup's rows have the dense matrices of `packageRows`, the premises of the final
verifier hold at the logical widths of the packages. -/
theorem premises_of_rows (setup : FinalVerifier.Setup mode logicalWidth publicFits)
    (exports : Exports relation setup.ajtai setup.circuitKeys step cubeFits)
    (complete : Complete exports)
    (rows : ∀ circuit : Fin 2,
      (setup.rows circuit).matrices = (packageRows exports complete circuit).matrices) :
    FinalVerifier.Premises setup complete.packageWidth := by
  refine ⟨fun circuit => ?_, fun circuit => ?_⟩
  · rw [rows circuit]
    fin_cases circuit
    · have key := congrArg ProductionKey.LogicalRelation.matrices exports.lightKey
      rw [Plan.logicalRelation_matrices, complete.lightLayout] at key
      simp only [Fin.zero_eta, Fin.isValue, packageRows, ite_true, key]
      exact PackageRows.rows_represents _ _ _ _ exports.lightFitsPackage exports.lightInside
        (lightSlots exports complete) complete.lightFitsKeyWidth
    · have key := congrArg ProductionKey.LogicalRelation.matrices exports.refreshKey
      rw [Plan.logicalRelation_matrices, complete.refreshLayout] at key
      simp only [Fin.mk_one, Fin.isValue, packageRows, one_ne_zero, ite_false, key]
      exact PackageRows.rows_represents _ _ _ _ exports.refreshFitsPackage exports.refreshInside
        (refreshSlots exports complete) complete.refreshFitsKeyWidth
  · have columns := congrArg Compression.Verifier.Matrices.columns (rows circuit)
    rw [columns]
    fin_cases circuit
    · simp only [Fin.zero_eta, Fin.isValue, packageRows, ite_true,
        complete.packageWidth_zero]
      exact PackageRows.rows_support _ _
    · simp only [Fin.mk_one, Fin.isValue, packageRows, one_ne_zero, ite_false,
        complete.packageWidth_one]
      exact PackageRows.rows_support _ _

/-- `FinalVerifier.soundness` when the setup's rows have the dense matrices of `packageRows`. -/
theorem soundness (setup : FinalVerifier.Setup mode logicalWidth publicFits)
    (step : AppState → AppWitness → AppState) (cap : ℕ) (control : AppState → AppWitness → Fin 2)
    (exports : Exports relation setup.ajtai setup.circuitKeys step cubeFits)
    (complete : Complete exports)
    (rows : ∀ circuit : Fin 2,
      (setup.rows circuit).matrices = (packageRows exports complete circuit).matrices)
    (s : Compression.Game.Strategy (Option (Option (FinalVerifier.Outcome logicalWidth publicFits))))
    (q : ℕ) :
    Compression.Game.prob s (setup.game.restore q [])
        (FinalVerifier.Wins setup complete.packageWidth step cap control) ≤
      (q : ℝ) * FinalVerifier.finalError :=
  FinalVerifier.soundness setup step cap control (premises_of_rows setup exports complete rows) s q

/-- `FinalVerifier.real_soundness` when the setup's rows have the dense matrices of
`packageRows`. -/
theorem real_soundness (setup : FinalVerifier.Setup mode logicalWidth publicFits)
    (step : AppState → AppWitness → AppState) (cap : ℕ) (control : AppState → AppWitness → Fin 2)
    (exports : Exports relation setup.ajtai setup.circuitKeys step cubeFits)
    (complete : Complete exports)
    (rows : ∀ circuit : Fin 2,
      (setup.rows circuit).matrices = (packageRows exports complete circuit).matrices)
    (opening : Compression.Akita.Use) (real : Compression.Akita.Prover opening.scheme opening.claims)
    (event : real.Tape → Prop)
    (s : Compression.Game.Strategy (Option (Option (FinalVerifier.Outcome logicalWidth publicFits))))
    (q : ℕ) (akita : opening.Knowledge real) (binding : opening.Binding)
    (accepts : ∀ tape, event tape → opening.Opens real tape)
    (composition : opening.Composition real event
      (Compression.Game.prob s (setup.game.restore q [])
        (FinalVerifier.Wins setup complete.packageWidth step cap control))) :
    (real.tape.toOuterMeasure {tape | event tape}).toReal ≤
      (q : ℝ) * FinalVerifier.finalError + opening.error real :=
  FinalVerifier.real_soundness setup step cap control (premises_of_rows setup exports complete rows)
    opening real event s q akita binding accepts composition

end NightstreamFPrime.Export.BoundedDepth.FinalRows
