import Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.CurrentM4PhysicalStability

/-!
Contract: prove physical row stability for the main complete Step receipt
segments outside the final post-action tail.

Assurance tier: model-level.

Owns: exact equality of the pre-action and action-adjacent Step row segments.

Does not own: the final aggregate row theorem, fixed-point compilation,
production selection, Rust equality, or a security reduction.

Emits constraints: no new rows.
-/

set_option autoImplicit false
set_option maxRecDepth 500000
set_option maxHeartbeats 2400000

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.CurrentM4StepRows

open Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalKey
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.LeanCompiler.CurrentFixedPoint
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.CurrentM4PhysicalStability

def flattenRows (receipts : List InstructionReceipt) :
    List OwnedRow :=
  receipts.flatMap (fun receipt => receipt.rows)

noncomputable def prefixRows
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (setup : RelationSetup dimensions verifierRows) :
    List OwnedRow :=
  flattenRows
    (ApplicationStepCostSplit.CompleteApplicationCertification.stepPrefixReceipts
      (certificate setup))

noncomputable def applicationRows
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (setup : RelationSetup dimensions verifierRows) :
    List OwnedRow :=
  (ApplicationStepCostSplit.CompleteApplicationCertification.applicationStepReceipt
    (certificate setup)).rows

noncomputable def selectedParameters
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (setup : RelationSetup dimensions verifierRows) :=
  ConcreteNifsPlain270Profile.selected dimensions
    (ConcreteNifsCanonicalOperationalProfile.selectedKeys setup)
    (defaultRunning dimensions verifierRows)
    (machine benchmarkHashPlan)
    (terminalRelations dimensions verifierRows)
    (terminalChecks dimensions verifierRows)
    (widths setup)
    (selectedFootprints setup)

noncomputable def selectorRows
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (setup : RelationSetup dimensions verifierRows) :
    List OwnedRow :=
  let selected := selectedParameters setup
  let complete := certificate setup
  (CanonicalStepPlan.selectorPlan.{0}
    selected complete.baseProfile complete.allRecipes).receipt.rows

noncomputable def activationRows
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (setup : RelationSetup dimensions verifierRows) :
    List OwnedRow :=
  let selected := selectedParameters setup
  let complete := certificate setup
  flattenRows [
    CanonicalBranchPlan.trueActivationReceipt
      SourceOwners.stepBranchPath oneColumn oneColumn
      (CanonicalContexts.Step.selector selected complete.baseProfile),
    CanonicalBranchPlan.falseActivationReceipt
      SourceOwners.stepBranchPath oneColumn oneColumn
      (CanonicalContexts.Step.selector selected complete.baseProfile)]

noncomputable def baseRows
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (setup : RelationSetup dimensions verifierRows) :
    List OwnedRow :=
  let selected := selectedParameters setup
  let complete := certificate setup
  flattenRows [
    (CanonicalStepPlan.baseEqualityPlan.{0}
      selected complete.baseProfile complete.allRecipes).receipt,
    (CanonicalStepPlan.baseAssertionPlan.{0}
      selected complete.baseProfile).receipt,
    (CanonicalStepPlan.baseLiteralPlan.{0}
      selected complete.baseProfile
      complete.defaultRunningAdmissible).receipt]

noncomputable def recursivePreNifsRows
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (setup : RelationSetup dimensions verifierRows) :
    List OwnedRow :=
  let selected := selectedParameters setup
  let complete := certificate setup
  flattenRows [
    (CanonicalStepPlan.recursiveHashPlan.{0}
      selected complete.baseProfile complete.allRecipes).receipt,
    (CanonicalStepPlan.recursiveFreshPublicPlan.{0}
      selected complete.baseProfile complete.allRecipes).receipt,
    (CanonicalStepPlan.recursiveEncodePlan.{0}
      selected complete.baseProfile complete.allRecipes).receipt,
    (CanonicalStepPlan.recursiveEncodedEqualityPlan.{0}
      selected complete.baseProfile complete.allRecipes).receipt,
    (CanonicalStepPlan.recursiveAssertionPlan.{0}
      selected complete.baseProfile).receipt]

noncomputable def joinRows
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (setup : RelationSetup dimensions verifierRows) :
    List OwnedRow :=
  let selected := selectedParameters setup
  let complete := certificate setup
  (CanonicalBranchPlan.onePortJoinReceipt
    SourceOwners.stepBranchPath
    (CanonicalContexts.Step.selector selected complete.baseProfile)
    (Ports.committedRunning selected)
    (CanonicalContexts.Step.baseRunning selected)
    (CanonicalContexts.Step.recursiveRunning selected)).rows

noncomputable def continuationRows
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (setup : RelationSetup dimensions verifierRows) :
    List OwnedRow :=
  let selected := selectedParameters setup
  let complete := certificate setup
  (CanonicalStepPlan.continuationHashPlan.{0}
    selected complete.baseProfile complete.allRecipes).receipt.rows

noncomputable def preNifsRows
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (setup : RelationSetup dimensions verifierRows) :
    List OwnedRow :=
  selectorRows setup ++ activationRows setup ++ baseRows setup ++
    recursivePreNifsRows setup

noncomputable def postNifsRows
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (setup : RelationSetup dimensions verifierRows) :
    List OwnedRow :=
  joinRows setup ++ continuationRows setup

theorem prefixRows_eq_of_constraintPolynomial_eq
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (template : SetupTemplate dimensions verifierRows)
    (left right : Structure dimensions.shape)
    (same :
      left.constraintPolynomial = right.constraintPolynomial) :
    prefixRows (template.withSystem left) =
      prefixRows (template.withSystem right) := by
  cases left with
  | mk leftMatrices polynomial =>
      cases right with
      | mk rightMatrices rightPolynomial =>
          simp only at same
          subst rightPolynomial
          rfl

theorem applicationRows_eq_of_constraintPolynomial_eq
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (template : SetupTemplate dimensions verifierRows)
    (left right : Structure dimensions.shape)
    (same :
      left.constraintPolynomial = right.constraintPolynomial) :
    applicationRows (template.withSystem left) =
      applicationRows (template.withSystem right) := by
  cases left with
  | mk leftMatrices polynomial =>
      cases right with
      | mk rightMatrices rightPolynomial =>
          simp only at same
          subst rightPolynomial
          rfl

theorem preNifsRows_eq_of_constraintPolynomial_eq
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (template : SetupTemplate dimensions verifierRows)
    (left right : Structure dimensions.shape)
    (same :
      left.constraintPolynomial = right.constraintPolynomial) :
    preNifsRows (template.withSystem left) =
      preNifsRows (template.withSystem right) := by
  cases left with
  | mk leftMatrices polynomial =>
      cases right with
      | mk rightMatrices rightPolynomial =>
          simp only at same
          subst rightPolynomial
          unfold preNifsRows
          congr 1

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.CurrentM4StepRows
