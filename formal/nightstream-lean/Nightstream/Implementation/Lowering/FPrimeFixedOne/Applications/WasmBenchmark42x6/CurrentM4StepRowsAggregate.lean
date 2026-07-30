import Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.CurrentM4StepPostRows

/-!
Contract: assemble exact physical row stability for the complete benchmark
Step encoding.

Assurance tier: model-level.

Owns: equality of all 19,859,562 emitted Step rows when the selected
constraint polynomial stays fixed.

Does not own: column stability, fixed-point compilation, production
selection, Rust equality, or a security reduction.

Emits constraints: no new rows.
-/

set_option autoImplicit false
set_option maxRecDepth 500000
set_option maxHeartbeats 2400000

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.CurrentM4StepRowsAggregate

open Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalKey
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.LeanCompiler.CurrentFixedPoint
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.CurrentM4StepPostRows
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.CurrentM4StepRows
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.CurrentM4PhysicalStability

theorem rows_exact_split
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (setup : RelationSetup dimensions verifierRows) :
    (encoding setup).rows =
      prefixRows setup ++ applicationRows setup ++ preNifsRows setup ++
        nifsReceiptRows setup ++ postNifsRows setup := by
  change
    (certificate setup).canonicalStep.program.toEncoding.rows =
      prefixRows setup ++ applicationRows setup ++ preNifsRows setup ++
        nifsReceiptRows setup ++ postNifsRows setup
  unfold Encoding.rows
  rw [
    SourceAlignment.AlignedReceiptProgram.toEncoding_receipts,
    ApplicationStepCostSplit.CompleteApplicationCertification.stepReceipts_exact_split
      (certificate setup)]
  unfold prefixRows applicationRows preNifsRows postNifsRows
    selectorRows activationRows baseRows recursivePreNifsRows
    joinRows continuationRows nifsReceiptRows selectedParameters flattenRows
  simp only [
    ApplicationStepCostSplit.CompleteApplicationCertification.stepSuffixReceipts,
    CanonicalStepPlan.bodyReceipts, List.tail_cons, List.flatMap_append,
    List.flatMap_cons, List.flatMap_nil, List.append_nil,
    List.append_assoc]

theorem rows_eq_of_constraintPolynomial_eq
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (template : SetupTemplate dimensions verifierRows)
    (left right : Structure dimensions.shape)
    (same :
      left.constraintPolynomial = right.constraintPolynomial) :
    (encoding (template.withSystem left)).rows =
      (encoding (template.withSystem right)).rows := by
  rw [
    rows_exact_split (template.withSystem left),
    rows_exact_split (template.withSystem right),
    prefixRows_eq_of_constraintPolynomial_eq template left right same,
    applicationRows_eq_of_constraintPolynomial_eq template left right same,
    preNifsRows_eq_of_constraintPolynomial_eq template left right same,
    nifsReceiptRows_eq_of_constraintPolynomial_eq template left right same,
    postNifsRows_eq_of_constraintPolynomial_eq template left right same]

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.CurrentM4StepRowsAggregate
