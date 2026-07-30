import Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.Deployment
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.LeanCompiler.CurrentCompiler
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.LeanCompiler.CurrentDeployment
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.LeanCompiler.CurrentFixedPoint
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.CurrentM4RawRows

/-!
Contract: project the complete benchmark Step encoding to the exact physical
columns and rows consumed by the selective-CCS compiler.

Assurance tier: model-level.

Owns: exact column identity stability and the common encoding interface used
by the current M4 fixed-point proof.

Does not own: post-raw Step row assembly, the final fixed point, production
selection, Rust equality, or a security reduction.

Emits constraints: no new rows.
-/

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.CurrentM4PhysicalStability

set_option autoImplicit false
set_option maxRecDepth 500000
set_option maxHeartbeats 2400000

open Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.CurrentM4RawRows
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalKey
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.LeanCompiler
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.LeanCompiler.CurrentFixedPoint
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

noncomputable def encoding
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (setup : RelationSetup dimensions verifierRows) :=
  (ConcreteNifsCanonicalCertification.complete
    setup
    (defaultRunning dimensions verifierRows)
    (machine benchmarkHashPlan)
    (terminalRelations dimensions verifierRows)
    (terminalChecks dimensions verifierRows)
    (widths setup)
    (selectedFootprints setup)
    (deployment setup)).canonicalStep.program.toEncoding

noncomputable def certificate
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (setup : RelationSetup dimensions verifierRows) :=
  ConcreteNifsCanonicalCertification.complete
    setup
    (defaultRunning dimensions verifierRows)
    (machine benchmarkHashPlan)
    (terminalRelations dimensions verifierRows)
    (terminalChecks dimensions verifierRows)
    (widths setup)
    (selectedFootprints setup)
    (deployment setup)

def publicWidth
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (setup : RelationSetup dimensions verifierRows) :
    270 ≤ (encoding setup).columnIds.length :=
  CurrentDeployment.deployment_step_columns_ge_270
    setup
    (defaultRunning dimensions verifierRows)
    (machine benchmarkHashPlan)
    (terminalRelations dimensions verifierRows)
    (terminalChecks dimensions verifierRows)
    (widths setup)
    (selectedFootprints setup)
    (deployment setup)

theorem proofCodec_eq_of_constraintPolynomial_eq
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (template : SetupTemplate dimensions verifierRows)
    (left right : Structure dimensions.shape)
    (same :
      left.constraintPolynomial = right.constraintPolynomial) :
    proofCodec (template.withSystem left) =
      proofCodec (template.withSystem right) := by
  cases left with
  | mk leftMatrices leftPolynomial =>
      cases right with
      | mk rightMatrices rightPolynomial =>
          simp only at same
          subst rightPolynomial
          unfold proofCodec
          change
            ConcreteNifsCanonicalProofCodec.proofCodec
                (ConcreteNifsPlain270Profile.Shape dimensions)
                leftPolynomial 0 publicRingColumns verifierRows
                (ConcreteNifsPlain270Profile.publicFits dimensions) =
              ConcreteNifsCanonicalProofCodec.proofCodec
                (ConcreteNifsPlain270Profile.Shape dimensions)
                leftPolynomial 0 publicRingColumns verifierRows
                (ConcreteNifsPlain270Profile.publicFits dimensions)
          rfl

theorem columnIds_eq_of_constraintPolynomial_eq
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (template : SetupTemplate dimensions verifierRows)
    (left right : Structure dimensions.shape)
    (same :
      left.constraintPolynomial = right.constraintPolynomial) :
    (encoding (template.withSystem left)).columnIds =
      (encoding (template.withSystem right)).columnIds := by
  unfold encoding
  change
    (certificate
      (template.withSystem left)).canonicalStep.program.toEncoding.columnIds =
    (certificate
      (template.withSystem right)).canonicalStep.program.toEncoding.columnIds
  unfold Encoding.columnIds Encoding.columns
  rw [
    SourceAlignment.AlignedReceiptProgram.toEncoding_receipts,
    SourceAlignment.AlignedReceiptProgram.toEncoding_receipts,
    ApplicationStepCostSplit.CompleteApplicationCertification.stepReceipts_exact_split
      (certificate (template.withSystem left)),
    ApplicationStepCostSplit.CompleteApplicationCertification.stepReceipts_exact_split
      (certificate (template.withSystem right))]
  simp only [List.flatMap_append, List.flatMap_cons,
    ApplicationStepCostSplit.CompleteApplicationCertification.stepPrefixReceipts,
    ApplicationStepCostSplit.CompleteApplicationCertification.applicationStepReceipt,
    ApplicationStepCostSplit.CompleteApplicationCertification.stepSuffixReceipts,
    CanonicalStepPlan.bodyReceipts, List.tail_cons,
    PrimitivePlan.receipt_allocations_exact,
    CanonicalBranchPlan.trueActivationReceipt,
    CanonicalBranchPlan.falseActivationReceipt,
    CanonicalBranchPlan.onePortJoinReceipt,
    InstructionReceipt.ofTrueActivation,
    InstructionReceipt.ofFalseActivation,
    InstructionReceipt.ofMux_allocations,
    List.map_append]
  cases left with
  | mk leftMatrices leftPolynomial =>
      cases right with
      | mk rightMatrices rightPolynomial =>
          simp only at same
          subst rightPolynomial
          rfl

noncomputable def nifsReceiptRows
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (setup : RelationSetup dimensions verifierRows) :
    List OwnedRow :=
  let selected := certificate setup
  (CanonicalStepPlan.recursiveNifsPlan.{0}
    (ConcreteNifsPlain270Profile.selected dimensions
      (ConcreteNifsCanonicalOperationalProfile.selectedKeys setup)
      (defaultRunning dimensions verifierRows)
      (machine benchmarkHashPlan)
      (terminalRelations dimensions verifierRows)
      (terminalChecks dimensions verifierRows)
      (widths setup)
      (selectedFootprints setup))
    selected.baseProfile selected.allRecipes).receipt.rows

theorem nifsReceiptRows_eq_of_constraintPolynomial_eq
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (template : SetupTemplate dimensions verifierRows)
    (left right : Structure dimensions.shape)
    (same :
      left.constraintPolynomial = right.constraintPolynomial) :
    nifsReceiptRows (template.withSystem left) =
      nifsReceiptRows (template.withSystem right) := by
  unfold nifsReceiptRows
  exact activatedRows_eq_of_constraintPolynomial_eq
    template left right same

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.CurrentM4PhysicalStability
