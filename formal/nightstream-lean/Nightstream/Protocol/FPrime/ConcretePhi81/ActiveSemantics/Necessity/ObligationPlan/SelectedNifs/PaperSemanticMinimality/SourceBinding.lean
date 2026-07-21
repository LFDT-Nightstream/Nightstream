import Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan.SelectedNifs.SemanticMinimality.PiCcs
import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.PaperProfile.ObligationPlan

/-!
Focused source-stage preservation fact for the corrected paper-plan removal
witness.

Assurance tier: model-level.

Owns: proof that changing only the fresh source's norm-stage tag leaves the
canonical `Pi_CCS -> Pi_RLC` parent unchanged, because `Pi_CCS.honestOutput`
copies the source structure, commitment, and public input but computes its own
fresh output stage.

Does not own: source-binding necessity, operational `Pi_DEC` acceptance,
Rust/R1CS refinement, costs, or row removal.

Emits constraints: no.
-/

namespace Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan.SelectedNifs.PaperSemanticMinimality.SourceBinding

open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.Nifs
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.HonestBaseline
open Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan.SelectedNifs.SemanticMinimality

def profile := FixedActive.paperProfileOf Context.context

def witness : FixedActive.PaperProfile.Witness Sources.shape :=
  FixedActive.paperWitnessOf baselineWitness

theorem outputs_eq :
    FixedActive.PaperProfile.outputs profile mismatchedSourceProduct
        Sources.data witness =
      FixedActive.PaperProfile.outputs profile Context.context.input
        Sources.data witness := by
  funext index
  refine Fin.addCases ?_ ?_ index
  · intro source
    have source_eq : source = Fin.last 0 := by
      apply Fin.ext
      have source_lt : source.val < 1 := by
        simpa only [FixedActive.arity_freshCount] using source.isLt
      change source.val = 0
      omega
    subst source
    simp [FixedActive.PaperProfile.outputs, PiCCS.honestOutputs,
      PiCCS.honestOutput, PiCCS.InputProduct.source,
      PiCCS.Source.constraintSystem, PiCCS.Source.commitment,
      PiCCS.Source.publicInput, Fin.addCases, mismatchedSourceProduct,
      wrongStageFreshStatement]
  · intro source
    simp [FixedActive.PaperProfile.outputs, PiCCS.honestOutputs,
      PiCCS.honestOutput, PiCCS.InputProduct.source,
      mismatchedSourceProduct]

theorem parent_eq :
    FixedActive.PaperProfile.parentOf profile mismatchedSourceProduct
        Sources.data witness =
      FixedActive.PaperProfile.parentOf profile Context.context.input
        Sources.data witness := by
  unfold FixedActive.PaperProfile.parentOf
  rw [outputs_eq]

end Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan.SelectedNifs.PaperSemanticMinimality.SourceBinding
