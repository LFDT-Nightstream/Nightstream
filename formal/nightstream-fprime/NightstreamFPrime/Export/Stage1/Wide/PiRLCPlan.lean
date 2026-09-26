import NightstreamFPrime.Export.Stage1.PiRLCProductRingSchedule
import NightstreamFPrime.Layout.PiRlcWideSampler.Challenges
import NightstreamFPrime.Layout.ProductionRelation.Phi81ProductFamilyPlan

/-! Candidate sampler and quotient-product rows. The product schedule keeps
family/source/block/cell order. Every left operand comes from the checked
wide-sampler bits. Caller forms supply the existing inputs and retained
output/quotient values. This module does not select a production package. -/

namespace NightstreamFPrime.Export.Stage1.Wide.PiRLCPlan

open NightstreamFPrime.Spec NightstreamFPrime.Layout
open ProductionRelation
open Spec.Folding.PiCCS.PaperJoint
open Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

abbrev RingIndex := Fin PiRLCProductRingSchedule.invocationCount
abbrev State := Phi81ProductPlan.State

structure Interface (columns : Nat) where
  sampler : PiRlcWideSampler.BatchPlan.Interface columns
  value : RingIndex → State columns
  quotient : RingIndex → State columns
  prior : RingIndex → State columns
  output : RingIndex → State columns

def products {columns : Nat} (interface : Interface columns) :
    Phi81ProductFamilyPlan.Interface columns PiRLCProductRingSchedule.invocationCount where
  oneColumn := interface.sampler.oneColumn
  left := fun ring => PiRlcWideSampler.Challenges.form interface.sampler
    (PiRLCProductRingSchedule.descriptor ring).source
  right := interface.value
  quotient := interface.quotient
  prior := interface.prior
  output := interface.output

theorem productsFit : PiRLCProductRingSchedule.invocationCount * 108 ≤
    2 ^ Lifecycle.cubeVariables := by decide

def productPlan {columns : Nat} (interface : Interface columns) : ProductionRelation.Plan columns :=
  Phi81ProductFamilyPlan.plan (products interface) productsFit

theorem productPlan_rows {columns : Nat} (interface : Interface columns) :
    (productPlan interface).rowCount = 104652 := rfl

def plan {columns : Nat} (compiled : PiRlcWideSampler.RangePlan.Compiled)
    (interface : Interface columns) : ProductionRelation.Plan columns :=
  ProductionRelation.Plan.append (PiRlcWideSampler.BatchPlan.plan compiled interface.sampler)
    (productPlan interface) (by rw [PiRlcWideSampler.BatchPlan.rowCount_eq, productPlan_rows]; decide)

theorem plan_rows {columns : Nat} (compiled : PiRlcWideSampler.RangePlan.Compiled)
    (interface : Interface columns) : (plan compiled interface).rowCount = 119153 := by
  change (PiRlcWideSampler.BatchPlan.plan compiled interface.sampler).rowCount +
    (productPlan interface).rowCount = _
  rw [PiRlcWideSampler.BatchPlan.rowCount_eq, productPlan_rows]

theorem rows_iff {columns : Nat} (compiled : PiRlcWideSampler.RangePlan.Compiled)
    (interface : Interface columns) (assignment : Assignment F columns) :
    (plan compiled interface).RowsZero assignment ↔
      (PiRlcWideSampler.BatchPlan.plan compiled interface.sampler).RowsZero assignment ∧
      (productPlan interface).RowsZero assignment :=
  ProductionRelation.Plan.append_rowsZero_iff _ _ _ _

def challenge {columns : Nat} (interface : Interface columns)
    (assignment : Assignment F columns) (ring : RingIndex) : RingF :=
  Spec.Folding.Nifs.NonInteractive.PiRlcWideSampler.Transcript.challengeAt
    (PiRlcWideSampler.StateSemantics.state assignment interface.sampler.initialState)
    (PiRLCProductRingSchedule.descriptor ring).source.val

theorem soundness {columns : Nat} (compiled : PiRlcWideSampler.RangePlan.Compiled)
    (interface : Interface columns) (assignment : Assignment F columns)
    (one : assignment interface.sampler.oneColumn = 1)
    (rows : (plan compiled interface).RowsZero assignment) (ring : RingIndex) :
    Phi81ProductPlan.evalState assignment (interface.output ring) =
      ringFAdd (Phi81ProductPlan.evalState assignment (interface.prior ring))
        (ringFMul (challenge interface assignment ring)
          (Phi81ProductPlan.evalState assignment (interface.value ring))) := by
  obtain ⟨sampleRows, productRows⟩ := (rows_iff compiled interface assignment).mp rows
  have multiplication := Phi81ProductFamilyPlan.planRowsZero_implies_ringProduct
    (products interface) productsFit assignment one productRows ring
  have left := PiRlcWideSampler.Challenges.exact_challenge compiled interface.sampler
    assignment one sampleRows (PiRLCProductRingSchedule.descriptor ring).source
  have leftState : Phi81ProductPlan.evalState assignment ((products interface).left ring) =
      challenge interface assignment ring := by
    funext lane
    exact congrFun left lane
  rw [leftState] at multiplication
  exact multiplication

/-- The canonical quotient of the actual checked challenge and right operand. -/
def honestQuotient {columns : Nat} (interface : Interface columns)
    (assignment : Assignment F columns) (ring : RingIndex) : RingF :=
  fun lane => Phi81Relation.QuotientProduct.quotientCoeff (challenge interface assignment ring)
    (Phi81ProductPlan.evalState assignment (interface.value ring)) lane

/-- Constructive quotient values complete the matrix relation whenever the
caller encodes the corresponding running outputs. -/
theorem completeness {columns : Nat} (compiled : PiRlcWideSampler.RangePlan.Compiled)
    (interface : Interface columns) (assignment : Assignment F columns)
    (one : assignment interface.sampler.oneColumn = 1)
    (sampleRows : (PiRlcWideSampler.BatchPlan.plan compiled interface.sampler).RowsZero assignment)
    (outputs : ∀ ring, Phi81ProductPlan.evalState assignment (interface.output ring) =
      ringFAdd (Phi81ProductPlan.evalState assignment (interface.prior ring))
        (ringFMul (challenge interface assignment ring)
          (Phi81ProductPlan.evalState assignment (interface.value ring))))
    (quotients : ∀ ring, Phi81ProductPlan.evalState assignment (interface.quotient ring) =
      honestQuotient interface assignment ring) :
    (plan compiled interface).RowsZero assignment := by
  apply (rows_iff compiled interface assignment).mpr
  refine ⟨sampleRows, ?_⟩
  apply (Phi81ProductFamilyPlan.planRowsZero_iff (products interface) productsFit assignment one).mpr
  intro ring point
  have left := PiRlcWideSampler.Challenges.exact_challenge compiled interface.sampler
    assignment one sampleRows (PiRLCProductRingSchedule.descriptor ring).source
  have leftState : Phi81ProductPlan.evalState assignment ((products interface).left ring) =
      challenge interface assignment ring := by
    funext lane
    exact congrFun left lane
  change Phi81Relation.QuotientProduct.evaluate
      (Phi81ProductPlan.evalState assignment ((products interface).left ring)) _ *
      Phi81Relation.QuotientProduct.evaluate (Phi81ProductPlan.evalState assignment (interface.value ring)) _ =
    Phi81Relation.QuotientProduct.evaluate (Phi81ProductPlan.evalState assignment (interface.output ring)) _ -
      Phi81Relation.QuotientProduct.evaluate (Phi81ProductPlan.evalState assignment (interface.prior ring)) _ +
      Phi81Relation.QuotientProduct.modulusValue _ *
        Phi81Relation.QuotientProduct.evaluate (Phi81ProductPlan.evalState assignment (interface.quotient ring)) _
  rw [leftState, outputs ring, quotients ring]
  have quotientEq : honestQuotient interface assignment ring =
      Phi81Relation.QuotientProduct.quotient (challenge interface assignment ring)
        (Phi81ProductPlan.evalState assignment (interface.value ring)) := by
    funext lane
    exact Phi81Relation.QuotientProduct.quotientCoeff_eq_quotient _ _ lane
  rw [quotientEq]
  exact Phi81Relation.QuotientProduct.complete_add _ _ _ point

end NightstreamFPrime.Export.Stage1.Wide.PiRLCPlan
