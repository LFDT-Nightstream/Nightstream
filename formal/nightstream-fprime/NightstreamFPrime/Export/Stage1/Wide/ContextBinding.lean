import NightstreamFPrime.Export.Stage1.Wide.PublicBinding
import NightstreamFPrime.Layout.Stage1.PiCCSSecurity

/-! Bind an arbitrary accepted candidate assignment to the context in the
verifier's checked state hash, or expose the existing state-hash collision.
The supplied expected preimage is authoritative verifier input. -/

namespace NightstreamFPrime.Export.Stage1.Wide.ContextBinding

open NightstreamFPrime.Circuit NightstreamFPrime.Spec NightstreamFPrime.Layout
open NightstreamFPrime.Lifecycle NightstreamFPrime.Lifecycle.PaperAlgebra
open Spec.Folding Spec.Folding.PiCCS.PaperJoint Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
open Spec.HyperNova.Construction2.Paper ProductionRelation Layout.Stage1
open FixedPointSoundness

variable (program : RetainedLayout.Program)
  (assignment : Assignment F (RetainedLayout.logicalWidth program))
  (compiled : PiRlcWideSampler.RangePlan.Compiled)
  (fits : PerApplicationFixedPoint.FitsTwoPow28 program)
  (ajtai : AjtaiKey (logicalWidth := RetainedLayout.logicalWidth program)
    (publicFits := FixedPoint.publicFits program))

noncomputable def decodedNext : HashPreimage (logicalWidth := RetainedLayout.logicalWidth program)
    (publicFits := FixedPoint.publicFits program) :=
  nextHashPreimage
    (Lifecycle.Stage1.Wide.Relation.setup (FixedPoint.relation program compiled fits) ajtai
      (contextKey program assignment))
    (input program assignment (FixedPoint.relation program compiled fits))
    (output program assignment (RetainedLayout.logicalWidth program) (FixedPoint.publicFits program))

/-- Public equality is the actual carrier projection. The checked digest
binds the expected context; no honest witness or context-equality premise
is used. A different encoding with the same hash is reported explicitly. -/
theorem step_or_collision (expected : KeyDigest) (expectedLength : expected.length = 4)
    (claimed : HashPreimage (logicalWidth := RetainedLayout.logicalWidth program)
      (publicFits := FixedPoint.publicFits program))
    (publicDigest : Digest) (digestLength : publicDigest.length = 4)
    (publicEqual : PublicBinding.publicInput program assignment = encHash publicDigest)
    (checkedDigest : publicDigest = stateHash {claimed with verifierKeys := fun _ => expected})
    (rows : (FixedPoint.structuralPlan program compiled fits).RowsZero assignment) :
    Lifecycle.Stage1.Wide.Relation.StepHoldsFor (FixedPoint.relation program compiled fits) ajtai
      expected program (input program assignment (FixedPoint.relation program compiled fits))
      (output program assignment (RetainedLayout.logicalWidth program) (FixedPoint.publicFits program)) ∨
    PiCCSSecurity.StateHashCollision (decodedNext program assignment compiled fits ajtai)
      {claimed with verifierKeys := fun _ => expected} := by
  have accepted := PublicBinding.step program assignment compiled fits ajtai
    publicDigest digestLength publicEqual rows
  by_cases same : contextKey program assignment = expected
  · exact Or.inl (same ▸ accepted.1)
  · apply Or.inr
    have outputHash : digest program assignment =
        stateHash (decodedNext program assignment compiled fits ajtai) := accepted.1.2.2.1
    have hashes := outputHash.symm.trans (accepted.2.trans checkedDigest)
    refine ⟨?_, hashes⟩
    intro encodedEqual
    have keyLength : ((decodedNext program assignment compiled fits ajtai).verifierKeys functionIndex).length =
        ({claimed with verifierKeys := fun _ => expected}.verifierKeys functionIndex).length := by
      change (StateDecoder.keyDigest (priorState program assignment)).length = expected.length
      simp only [StateDecoder.keyDigest, StateDecoder.slice, List.length_ofFn, expectedLength]
      rfl
    exact same (StateEncoding.serializePreimage_eq_implies_context_eq
      (decodedNext program assignment compiled fits ajtai)
      {claimed with verifierKeys := fun _ => expected} keyLength encodedEqual)

end NightstreamFPrime.Export.Stage1.Wide.ContextBinding
