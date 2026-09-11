import NightstreamFPrime.Lifecycle.PiRLC.v1_1.Formal

/-!
Owns Boolean accessors for the two sampler source families retained as bits
in the selected assignment: candidate rejection flags and First54 position
outputs. Each parent extracts only its immediate child's rows and applies
that child's proved interface. The erased sampler output relation alone is
not used to infer internal witness values.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Lifecycle.PiRLC.v1_1

open NightstreamFPrime.Circuit
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Gadgets.Sampling

private theorem immediateChildRows
    (name : String) (child : FormalCircuit) (offset : Nat)
    (env : Env) (operations : List Op)
    (rows : holdsFlat env operations)
    (member : Sequence.childOp name child offset ∈ operations) :
    holdsFlat env (Circuit.ops child.main offset) := by
  intro expression expressionMember
  apply rows expression
  apply List.mem_flatMap.mpr
  refine ⟨Sequence.childOp name child offset, member, ?_⟩
  simpa only [Sequence.childOp, Op.flatConstraints,
    FormalCircuit.asSubcircuit_constraints] using expressionMember

namespace Sampler

/-- The exact stored rejection and position expressions are Boolean. This
predicate introduces no source coordinates or additional witness data. -/
def RetainedBits (offset : Nat) (env : Env) : Prop :=
  (∀ candidate : Fin First54.candidateCount,
    (DigestWindow.reject
      (windowOffset offset (candidateRound candidate).val)
      (candidatePosition candidate)).eval env = 0 ∨
    (DigestWindow.reject
      (windowOffset offset (candidateRound candidate).val)
      (candidatePosition candidate)).eval env = 1) ∧
  ∀ round : Fin First54.candidateCount, ∀ slot : Fin First54Step.slotCount,
    (First54Step.output
      (First54.positionOffset (selectorOffset offset) round.val) slot).eval env = 0 ∨
    (First54Step.output
      (First54.positionOffset (selectorOffset offset) round.val) slot).eval env = 1

/-- Actual scalar-sampler rows determine every retained bit. Decoder specs
give the rejection flag; the selector's internal position specs give its
one-hot trace. No Boolean witness value is supplied by the caller. -/
theorem retainedBits_of_rows
    (interface : Interface) (coordinate offset : Nat) (env : Env)
    (assumptions : Assumptions interface offset env)
    (rows : holdsFlat env (Circuit.ops (main interface coordinate) offset)) :
    RetainedBits offset env := by
  have specification := soundness interface coordinate env offset assumptions
    (holdsFlat_implies_holds env _ rows)
  have selectorAssumptions := selectorAssumptions interface coordinate offset
    env specification.window
  have selectorRows : holdsFlat env
      (Circuit.ops (First54.main (selectorInterface interface coordinate offset))
        (selectorOffset offset)) := by
    apply immediateChildRows selectorName (selectorCircuit interface coordinate offset)
      (selectorOffset offset) env (opsAt interface coordinate offset) rows
    simp [opsAt, selectorOp]
  have selectorSpec := First54.soundness
    (selectorInterface interface coordinate offset) env (selectorOffset offset)
    selectorAssumptions (holdsFlat_implies_holds env _ selectorRows)
  constructor
  · intro candidate
    have decoder := (specification.window (candidateRound candidate)).lane
      (DigestWindow.laneOf (candidatePosition candidate))
    have rejectEq := (decoder.decoder (DigestWindow.partOf
      (candidatePosition candidate))).reject_eq
    have actual :
        (DigestWindow.reject
          (windowOffset offset (candidateRound candidate).val)
          (candidatePosition candidate)).eval env =
        if ((DigestWindow.candidate
          (windowOffset offset (candidateRound candidate).val)
          (candidatePosition candidate)).eval env).val =
            Candidate16Five.rejectionBucket then 1 else 0 := by
      simpa only [DigestWindow.reject, DigestWindow.candidate, DigestLane.reject,
        DigestLane.candidate] using rejectEq
    rw [actual]
    split <;> simp
  · intro round slot
    have trace := First54.positionTrace
      (selectorInterface interface coordinate offset) (selectorOffset offset) env
      selectorAssumptions selectorSpec (round.val + 1)
      (Nat.succ_le_of_lt round.isLt)
    have value := trace slot
    change (First54Step.output
        (First54.positionOffset (selectorOffset offset) round.val) slot).eval env =
      First54.oneHotPosition
        (First54.semanticAcceptedCount (selectorInterface interface coordinate offset)
          (selectorOffset offset) env (round.val + 1)) slot at value
    rw [value, First54.oneHotPosition]
    split <;> simp

end Sampler

namespace SamplerChain

/-- Actual chain rows give the two retained Boolean families for each
scalar source, at the existing source offsets. -/
theorem retainedBits_of_rows
    (interface : Interface) (offset : Nat) (env : Env)
    (assumptions : Assumptions interface offset env)
    (rows : holdsFlat env (Circuit.ops (main interface) offset)) :
    ∀ source : Fin sourceCount,
      Sampler.RetainedBits (sourceOffset offset source.val) env := by
  intro source
  apply Sampler.retainedBits_of_rows (childInterface interface offset source.val)
    source.val (sourceOffset offset source.val) env
    (childAssumptions interface offset source.val source.isLt env assumptions)
  apply immediateChildRows (childName source.val)
    (Sampler.circuit (childInterface interface offset source.val) source.val)
    (sourceOffset offset source.val) env (opsAt interface offset) rows
  change childOp interface offset source.val ∈ opsAt interface offset
  apply List.mem_map.mpr
  exact ⟨source.val, List.mem_range.mpr source.isLt, rfl⟩

end SamplerChain

namespace Formal

/-- Actual PiRLC phase rows imply Boolean rejection and position sources
for every scalar sampler. Selected assignment construction consumes this
interface without expanding sampler children or their internal allocation. -/
theorem retainedSamplerBits_of_rows
    {logicalWidth : Nat}
    {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface logicalWidth publicFits) (offset : Nat) (env : Env)
    (assumptions : Assumptions relation interface offset env)
    (rows : holdsFlat env (Circuit.ops (main relation interface) offset)) :
    ∀ source : Fin SamplerChain.sourceCount,
      Sampler.RetainedBits
        (SamplerChain.sourceOffset (samplerOffset offset) source.val) env := by
  apply SamplerChain.retainedBits_of_rows
    (samplerInterface (atOffset interface offset)) (samplerOffset offset) env
    assumptions.sampler
  apply immediateChildRows "pirlc.v1_1.sampler_chain"
    (samplerCircuit (atOffset interface offset)) (samplerOffset offset) env
    (opsAt relation interface offset) rows
  simp [opsAt, childOp]

end Formal

end NightstreamFPrime.Lifecycle.PiRLC.v1_1
