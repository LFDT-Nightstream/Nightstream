import NightstreamFPrime.Layout.MatrixProgram.Phi81Product

/-!
Load the existing Phi81 product interface with direct finite-index selection.
The returned interface and every rejection agree with the canonical loader.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiDECProductInterface

open NightstreamFPrime.Spec
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Layout.MatrixProgram

private def select {Alpha : Type} {count : Nat}
    (head : Alpha) (tail : Fin count → Alpha) : Fin (count + 1) → Alpha
  | ⟨0, _⟩ => head
  | ⟨index + 1, bound⟩ => tail ⟨index, Nat.lt_of_succ_lt_succ bound⟩

private theorem select_eq_cases {Alpha : Type} {count : Nat}
    (head : Alpha) (tail : Fin count → Alpha) :
    select head tail = Fin.cases head tail := by
  funext index
  refine Fin.cases ?_ (fun _ => ?_) index <;> rfl

/-- Load all elements in order, then select only the requested tail index.
Any missing element rejects the complete function. -/
def loadFin? {Alpha : Type} :
    (count : Nat) → (Fin count → Option Alpha) →
      Option (Fin count → Alpha)
  | 0, _ => some Fin.elim0
  | count + 1, load => do
      let head ← load 0
      let tail ← loadFin? count (fun index => load index.succ)
      pure (select head tail)

/-- Total equality includes every pattern of missing elements. -/
theorem loadFin?_value {Alpha : Type} (count : Nat)
    (load : Fin count → Option Alpha) :
    loadFin? count load = Phi81Product.loadFin? count load := by
  induction count with
  | zero => rfl
  | succ count inductionHypothesis =>
      simp only [loadFin?, Phi81Product.loadFin?,
        inductionHypothesis, select_eq_cases]

/-- The quotient operands and guards, using the proved direct loader. -/
def interface? (block : Phi81Product.Block) (logicalWidth : Nat)
    (descriptor : Phi81Product.Descriptor) :
    Option (Phi81ProductPlan.Interface logicalWidth) := do
  let oneColumn ← block.oneColumn? logicalWidth
  let left ← loadFin? ringDegree fun lane =>
    block.challenge.form? oneColumn descriptor.source.val lane.val
  let input ← loadFin? ringDegree fun lane =>
    block.input.form? logicalWidth (descriptor.invocationAtLane lane)
  let quotient ← loadFin? ringDegree fun lane =>
    block.group.form? logicalWidth (descriptor.invocationAtLane lane)
  let prior ← if descriptor.source.val = 0 then
      some (fun _ => SparseForm.empty)
    else
      loadFin? ringDegree fun lane =>
        block.output.form? logicalWidth
          (descriptor.invocationAtLane lane - descriptor.family.privateCount)
  let output ← loadFin? ringDegree fun lane =>
    block.output.form? logicalWidth (descriptor.invocationAtLane lane)
  pure { oneColumn, left, right := input, quotient, prior, output }

/-- Exact equality includes all missing-input rejections. -/
theorem interface?_value (block : Phi81Product.Block) (logicalWidth : Nat)
    (descriptor : Phi81Product.Descriptor) :
    interface? block logicalWidth descriptor =
      block.interface? logicalWidth descriptor := by
  by_cases first : descriptor.source.val = 0 <;>
    simp [interface?, Phi81Product.Block.interface?,
      Phi81Product.Block.challengeState?, Phi81Product.Block.inputState?,
      Phi81Product.Block.quotientState?, Phi81Product.Block.priorState?,
      Phi81Product.Block.outputState?, loadFin?_value, first]

end NightstreamFPrime.Export.Stage1.PiDECProductInterface
