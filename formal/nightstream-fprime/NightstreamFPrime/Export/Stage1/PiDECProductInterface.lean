import NightstreamFPrime.Layout.MatrixProgram.RingProduct

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

def loadFin? {Alpha : Type} :
    (count : Nat) → (Fin count → Option Alpha) →
      Option (Fin count → Alpha)
  | 0, _ => some Fin.elim0
  | count + 1, load => do
      let head ← load 0
      let tail ← loadFin? count (fun index => load index.succ)
      pure (select head tail)

theorem loadFin?_value {Alpha : Type} (count : Nat)
    (load : Fin count → Option Alpha) :
    loadFin? count load = RingProduct.loadFin? count load := by
  induction count with
  | zero => rfl
  | succ count inductionHypothesis =>
      simp only [loadFin?, RingProduct.loadFin?,
        inductionHypothesis, select_eq_cases]

def interface? (block : RingProduct.Block) (logicalWidth : Nat)
    (descriptor : RingProduct.Descriptor) :
    Option (RingProductPlan.Interface logicalWidth) := do
  let oneColumn ← block.oneColumn? logicalWidth
  let challenge ← loadFin? ringDegree fun lane =>
    block.challenge.form? logicalWidth
      (block.challengeSlotStart +
        descriptor.source.val * block.challengeSourceStride + lane.val)
  let input ← loadFin? ringDegree fun lane =>
    block.input.form? logicalWidth (descriptor.invocationAtLane lane)
  let quotient ← loadFin? ringDegree fun lane =>
    block.quotient.form? logicalWidth (descriptor.invocationAtLane lane)
  let prior ← if descriptor.source.val = 0 then
      some (fun _ => SparseForm.empty)
    else
      loadFin? ringDegree fun lane =>
        block.output.form? logicalWidth
          (descriptor.invocationAtLane lane - descriptor.family.privateCount)
  let output ← loadFin? ringDegree fun lane =>
    block.output.form? logicalWidth (descriptor.invocationAtLane lane)
  let left : RingProductPlan.State logicalWidth := fun lane =>
    SparseForm.add (challenge lane) (SparseForm.singleton oneColumn (-2))
  pure { left, right := input, quotient, prior, output }

theorem interface?_value (block : RingProduct.Block) (logicalWidth : Nat)
    (descriptor : RingProduct.Descriptor) :
    interface? block logicalWidth descriptor =
      block.interface? logicalWidth descriptor := by
  by_cases first : descriptor.source.val = 0 <;>
    simp [interface?, RingProduct.Block.interface?,
      RingProduct.Block.challengeState?, RingProduct.Block.inputState?,
      RingProduct.Block.quotientState?, RingProduct.Block.priorState?,
      RingProduct.Block.outputState?, loadFin?_value, first]

end NightstreamFPrime.Export.Stage1.PiDECProductInterface
