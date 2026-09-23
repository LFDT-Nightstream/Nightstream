import NightstreamFPrime.Export.Stage1.PiRLCCombinationReadSupport
import NightstreamFPrime.Export.Stage1.PiCCSOrdinaryDirectSupport
import NightstreamFPrime.Lifecycle.PiCCS.v1_1.WitnessSupport

/-! Actual PiCCS arithmetic recipe support before the discarded product scratch. -/

namespace NightstreamFPrime.Export.Stage1.PiRLCCombinationPiCCSReadSupport

open NightstreamFPrime.Circuit
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.PiCCS.v1_1
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open PiRLCCombinationWitnessReadSupport (Outside)
open PiRLCCombinationReadSupport

theorem source_before (column : Nat) (source : PiCCSOrdinarySourceSupport.Source column) :
    column < PiRLCStarts.commitmentFreshStart := by
  change column < 20572642
  rcases source with ((external | transcript | ordinary) | fresh)
  · rcases external with prior | publicInput | output | context | proof
    · have bound := prior.2
      change column < 0 + 49393 at bound
      omega
    · have bound := publicInput.2
      change column < 49393 + 270 at bound
      omega
    · have bound := output.2
      change column < 49663 + 49393 at bound
      omega
    · have bound := context.2
      change column < 14722512 + 4 at bound
      omega
    · have bound := proof.2
      change column < 14722516 + (14751804 - 14722516) at bound
      omega
  · rcases transcript with ⟨invocation, lane, rfl⟩
    have invocationBound := invocation.isLt
    have laneBound := lane.isLt
    simp only [PiCCSOrdinarySourceSupport.transcriptInvocationCount_eq] at invocationBound
    change lane.val < 8 at laneBound
    rw [PiCCSInputs.phaseOffset_eq]
    omega
  · have bound := ordinary.2
    rw [PiCCSOrdinarySourceSupport.ordinaryLogicalCount_eq] at bound
    change column < 15176860 + 52734 at bound
    omega
  · have bound := fresh.2
    change column < 19513117 at bound
    omega

theorem source_outside (column : Nat) (source : PiCCSOrdinarySourceSupport.Source column) :
    Outside (Spartan.sourceToSpartan column) := mapped_before column (source_before column source)

private theorem childBatches_supported (main : Circuit Unit) (offset : Nat)
    (covered : WitnessesFromConstraints (Circuit.ops main offset))
    (supported : ∀ expression ∈ flatConstraints (Circuit.ops main offset),
      expression.VarsSatisfy PiCCSOrdinarySourceSupport.Source) :
    ∀ batch ∈ WitnessProgram.childBatches main offset, batch.ReadsSatisfy Outside := by
  intro batch member
  rcases List.mem_map.mp member with ⟨source, sourceMember, rfl⟩
  apply remapBatch_supported
  have reads := covered PiCCSOrdinarySourceSupport.Source supported source sourceMember
  exact ⟨fun expression member => (reads.1 expression member).mono expression source_outside,
    fun hint member => (reads.2 hint member).mono hint.source source_outside⟩

theorem piCcsBatches_readsSatisfy (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth) :
    ∀ batches ∈ [WitnessProgram.initialClaimBatches logicalWidth publicFits,
      WitnessProgram.sumcheckBatches logicalWidth publicFits,
      WitnessProgram.evalKBatches logicalWidth publicFits,
      WitnessProgram.evalABatches logicalWidth publicFits,
      WitnessProgram.ccsBatches logicalWidth publicFits,
      WitnessProgram.normBatches logicalWidth publicFits,
      WitnessProgram.finalIdentityBatches logicalWidth publicFits],
    ∀ batch ∈ batches, batch.ReadsSatisfy Outside := by
  have support := PiCCSOrdinaryDirectSupport.emittedConstraints_varsSatisfy
    (logicalWidth := logicalWidth) (publicFits := publicFits)
  intro batches family
  simp only [List.mem_cons, List.not_mem_nil, or_false] at family
  rcases family with rfl | rfl | rfl | rfl | rfl | rfl | rfl
  all_goals
    apply childBatches_supported
  · exact InitialClaim.witnessesFromConstraints
      (Formal.initialClaimInterface (PiCCSArithmetic.sharedInterface logicalWidth publicFits))
      PiCCSArithmetic.initialClaimLogicalStart
  · intro expression member
    apply support expression
    change expression ∈ PiCCSArithmetic.initialClaimConstraints logicalWidth publicFits at member
    simp only [PiCCSCompleteness.emittedConstraints, PiCCSCompleteness.packetConstraints,
      List.mem_append]
    exact Or.inr (Or.inl member)
  · exact SumcheckChain.witnessesFromConstraints
      (Formal.sumcheckInterface (PiCCSArithmetic.sharedInterface logicalWidth publicFits))
      PiCCSArithmetic.sumcheckLogicalStart
  · intro expression member
    apply support expression
    change expression ∈ PiCCSArithmetic.sumcheckConstraints logicalWidth publicFits at member
    simp only [PiCCSCompleteness.emittedConstraints, PiCCSCompleteness.packetConstraints,
      List.mem_append]
    exact Or.inr (Or.inr (Or.inl member))
  · exact EvalKTerminal.witnessesFromConstraints
      (Formal.evalKInterface (PiCCSArithmetic.sharedInterface logicalWidth publicFits))
      PiCCSArithmetic.evalKLogicalStart
  · intro expression member
    apply support expression
    change expression ∈ PiCCSArithmetic.evalKConstraints logicalWidth publicFits at member
    simp only [PiCCSCompleteness.emittedConstraints, PiCCSCompleteness.packetConstraints,
      List.mem_append]
    exact Or.inr (Or.inr (Or.inr (Or.inl member)))
  · exact EvalATerminal.witnessesFromConstraints
      (Formal.evalAInterface (PiCCSArithmetic.sharedInterface logicalWidth publicFits))
      PiCCSArithmetic.evalALogicalStart
  · intro expression member
    apply support expression
    change expression ∈ PiCCSArithmetic.evalAConstraints logicalWidth publicFits at member
    simp only [PiCCSCompleteness.emittedConstraints, PiCCSCompleteness.packetConstraints,
      List.mem_append]
    exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inl member))))
  · exact Gadgets.Polynomial.Sparse.Owned.witnessesFromConstraints Formal.ccsRowPolynomial
      (CcsTerminal.sparseInterface
        (Formal.ccsRowInterface (PiCCSArithmetic.sharedInterface logicalWidth publicFits)))
      PiCCSArithmetic.ccsLogicalStart
  · intro expression member
    apply support expression
    change expression ∈ PiCCSArithmetic.ccsConstraints logicalWidth publicFits at member
    simp only [PiCCSCompleteness.emittedConstraints, PiCCSCompleteness.packetConstraints,
      List.mem_append]
    exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl member)))))
  · exact NormTerminal.witnessesFromConstraints
      (Formal.normRowInterface (PiCCSArithmetic.sharedInterface logicalWidth publicFits))
      PiCCSArithmetic.normLogicalStart
  · intro expression member
    apply support expression
    change expression ∈ PiCCSArithmetic.normConstraints logicalWidth publicFits at member
    simp only [PiCCSCompleteness.emittedConstraints, PiCCSCompleteness.packetConstraints,
      List.mem_append]
    exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl member))))))
  · exact FinalIdentity.witnessesFromConstraints
      (Formal.finalIdentityRowInterface (PiCCSArithmetic.sharedInterface logicalWidth publicFits))
      PiCCSArithmetic.finalIdentityLogicalStart
  · intro expression member
    apply support expression
    change expression ∈ PiCCSArithmetic.finalIdentityConstraints logicalWidth publicFits at member
    simp only [PiCCSCompleteness.emittedConstraints, PiCCSCompleteness.packetConstraints,
      List.mem_append]
    exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr member))))))

end NightstreamFPrime.Export.Stage1.PiRLCCombinationPiCCSReadSupport
