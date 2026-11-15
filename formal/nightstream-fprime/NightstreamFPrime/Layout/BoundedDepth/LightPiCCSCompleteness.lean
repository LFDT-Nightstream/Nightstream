import NightstreamFPrime.Layout.BoundedDepth.LightPiCCSBounds
import NightstreamFPrime.Lifecycle.PiCCS.v1_2.Completeness.Core
import NightstreamFPrime.Circuit.Sequence

/-! Completeness of the light Π_CCS layout (assembler A3a). When the external words carry a
running claim, a fresh instance and a light proof that the light Π_CCS check accepts, the eleven
leaves can be filled one after another (`completeFrom`). Nine leaves compute their outputs from
their inputs. The SumCheck chain and the final identity check a relation: the accepting check
gives the chain from `T` to `Q(r')` (`LightPiCCSPhase.chain_of_check`), and the chain's terminal
claim is the final identity's terminal value. Each check reads the outputs of earlier leaves,
whose values the rows of the prefix give by soundness. -/

namespace NightstreamFPrime.Layout.BoundedDepth.LightPiCCSCompleteness

open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.BoundedDepth

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Gadgets.SumCheck
open NightstreamFPrime.Gadgets.Multilinear
open NightstreamFPrime.Gadgets.Polynomial
open NightstreamFPrime.Lifecycle.PiCCS.v1_2
open NightstreamFPrime.Lifecycle.PiCCS.v1_2.Formal.CompletenessSupport
open NightstreamFPrime.Lifecycle.BoundedDepth.LightPiCCSLayout

variable {logicalWidth : Nat}
  {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
    RingCarrierLayout.carrierWidth logicalWidth}
  (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
  (interface : LightPiCCSLayout.Interface logicalWidth 8 publicFits) (offset : Nat)

/-- A fixed SumCheck chain has one terminal claim for given rounds and challenges. -/
private theorem chain_terminal_unique {degree : Nat} :
    ∀ (current : K) (rounds : List (SumCheck.Finite.FixedPolynomial K degree))
      (challenges : List K) (left right : K),
      SumCheck.Finite.FixedPhase.Chain extensionOps.toOps current rounds challenges left →
      SumCheck.Finite.FixedPhase.Chain extensionOps.toOps current rounds challenges right →
      left = right
  | _, [], [], _, _, leftChain, rightChain => leftChain.symm.trans rightChain
  | _, _ :: rounds, _ :: challenges, left, right, leftChain, rightChain =>
      chain_terminal_unique _ rounds challenges left right leftChain.2 rightChain.2
  | _, [], _ :: _, _, _, leftChain, _ => leftChain.elim
  | _, _ :: _, [], _, _, leftChain, _ => leftChain.elim

private theorem point_eq {count : Nat} {left right : CubePoint K count}
    (coordinates : left.coordinates = right.coordinates) : left = right := by
  cases left
  cases right
  simp_all

/-- The light statement absorption only absorbs, so it has no assertion rows. -/
private theorem statement_noAssertions :
    Duplex.Formal.Owned.allAssertions (statementDuplex interface offset) offset = [] := by
  apply List.eq_nil_of_length_eq_zero
  change (Duplex.Formal.compile offset Hash.zeroE
    (StatementAbsorption.domainActions lightFoldDomainChunk
      (statementInterface interface offset) offset)).assertions.length = 0
  rw [Duplex.Formal.compile_assertions_length]
  rfl

/-- A leaf of a prefix satisfies its spec on the prefix's values. -/
private theorem spec_of_prefix {initial : Env} {base : Nat}
    (completed : Sequence.Prefix initial base) (name : String) (child : FormalCircuit)
    (childOffset : Nat)
    (member : Formal.childOp name child childOffset ∈ completed.operations)
    (assumptions : child.assumptions childOffset completed.current) :
    child.spec childOffset completed.current := by
  have callHolds := holdsFlat_implies_holds completed.current completed.operations
    completed.rows (Formal.childOp name child childOffset) member
  change child.assumptions childOffset completed.current →
    child.spec childOffset completed.current at callHolds
  exact callHolds assumptions

private theorem mem_grow {initial : Env} {base : Nat} {before after : Sequence.Prefix initial base}
    {extra : Op} (operations : after.operations = before.operations ++ [extra]) {op : Op}
    (member : op ∈ before.operations) : op ∈ after.operations := by
  rw [operations]
  exact List.mem_append_left _ member

private theorem mem_last {initial : Env} {base : Nat} {before after : Sequence.Prefix initial base}
    {extra : Op} (operations : after.operations = before.operations ++ [extra]) :
    extra ∈ after.operations := by
  rw [operations]
  exact List.mem_append_right _ (List.mem_singleton_self _)

variable (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
  (running : Spec.Folding.Nifs.PaperNonInteractive.Running K PaperAlgebra.Commitment
    (PaperAlgebra.PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits))
    LightShape)
  (fresh : Spec.Folding.Nifs.PaperNonInteractive.Fresh PaperAlgebra.Commitment
    (PaperAlgebra.PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits))
    LightShape)
  (proof : Spec.Folding.Light.Proof LightShape 8)

/-- The external words keep what they carry when only words at or above the offset change. -/
theorem represents_of_agree_below (below : LightPiCCSBounds.InputsBelow interface offset)
    {left right : Env} (agrees : ∀ index, index < offset → left index = right index)
    (represents : Represents running fresh proof interface offset left) :
    Represents running fresh proof interface offset right := by
  have kEq : ∀ value : KExpr, value.VarsBelow offset → value.eval right = value.eval left :=
    fun value bound => (value.eval_eq_of_agree_below offset left right bound agrees).symm
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [← represents.fresh]
    unfold StatementAbsorption.evalFresh
    congr 1
    · funext source row coefficient
      exact (((interface.fresh offset).commitment source row coefficient).eval_eq_of_agree_below
        offset left right (below.freshCommitment source row coefficient) agrees).symm
    · funext source column
      exact (((interface.fresh offset).publicInput source column).eval_eq_of_agree_below
        offset left right (below.freshPublicInput source column) agrees).symm
  · refine Eq.trans ?_ represents.runningPoint
    exact congrArg (List.map · (canonicalFinIndices LightShape.cubeVariables))
      (funext fun coordinate => kEq _ (below.runningPoint coordinate))
  · intro coefficient
    rw [kEq _ (below.runningPad coefficient)]
    exact represents.runningPad coefficient
  · intro matrix coefficient
    rw [kEq _ (below.runningMatrix matrix coefficient)]
    exact represents.runningMatrix matrix coefficient
  · intro roundIndex
    rw [← represents.rounds roundIndex]
    unfold RoundTranscript.Message.semanticPolynomial
    exact FixedChain.Round.semanticPolynomial_eq_of_agree_below _ offset right left
      ⟨fun coefficient => below.round roundIndex coefficient, ⟨trivial, trivial⟩⟩
      (fun index bound => (agrees index bound).symm)
  · intro source coefficient
    rw [kEq _ (below.outputPad source coefficient)]
    exact represents.outputPad source coefficient
  · intro source matrix coefficient
    rw [kEq _ (below.outputMatrix source matrix coefficient)]
    exact represents.outputMatrix source matrix coefficient

/-- Completeness of the light Π_CCS layout after a prefix that ends at the layout's offset. The
external words carry the running claim, the fresh instance and the light proof, and the light
Π_CCS check of the relation `target` accepts them. -/
theorem completeFrom {initial : Env} {base : Nat} (before : Sequence.Prefix initial base)
    (startEq : base + localLength before.operations = offset)
    (below : LightPiCCSBounds.InputsBelow interface offset)
    (target : ProductionKey.LogicalRelation logicalWidth publicFits)
    (represents : Represents running fresh proof interface offset before.current)
    (accepted : Spec.Folding.Light.PiCCS.check (keys target ajtai).light.extensionOps
      (keys target ajtai).light.lift ((keys target ajtai).light.statement running fresh)
      (ProductionKey.degreeBound target) ((keys target ajtai).light.probe running fresh proof) =
        true) :
    ∃ after : Sequence.Prefix initial base,
      after.operations = before.operations ++ opsAt relation interface offset ∧
      base + localLength after.operations = finalOffset relation interface offset ∧
      Sequence.PreservesPrefix before after := by
  have assumptionsAt := LightPiCCSBounds.assumptions relation interface offset below
  have representsAt : ∀ completed : Sequence.Prefix initial base,
      Sequence.PreservesPrefix before completed →
      Represents running fresh proof interface offset completed.current :=
    fun completed preserves => represents_of_agree_below interface offset running fresh proof
      below (fun index bound => (preserves.values index (by omega)).symm) represents
  -- Transcript: statement absorption, challenge derivation, round transcript.
  obtain ⟨built1, agrees1, rows1⟩ := Duplex.Formal.Owned.build_of_no_assertions
    (statementDuplex interface offset) before.current offset
    (assumptionsAt before.current).statement (statement_noAssertions interface offset)
  obtain ⟨p1, ops1, end1, preserves1, -⟩ := appendBuiltAt before
    "light.piccs.statement_absorption" (statementCircuit interface offset) offset startEq
    (Duplex.Formal.Owned.flatConstraints_varsBelow _ offset before.current
      (assumptionsAt before.current).statement) built1 agrees1 rows1
  obtain ⟨built2, agrees2, rows2⟩ := ChallengeDerivation.build (challengeInterface interface offset)
    p1.current (challengeOffset interface offset) (assumptionsAt p1.current).challenge
  rw [← ChallengeDerivation.opsAt_localLength] at agrees2
  obtain ⟨p2, ops2, end2, preserves2, -⟩ := appendBuiltAt p1
    "light.piccs.challenge_derivation" (challengeCircuit interface offset)
    (challengeOffset interface offset) end1
    (ChallengeDerivation.flatConstraints_varsBelow _ _ p1.current
      (assumptionsAt p1.current).challenge) built2 agrees2 rows2
  obtain ⟨built3, agrees3, rows3⟩ := RoundTranscript.build (roundsInterface interface offset)
    p2.current (roundsOffset interface offset) (assumptionsAt p2.current).rounds
  rw [← RoundTranscript.opsAt_localLength] at agrees3
  obtain ⟨p3, ops3, end3, preserves3, -⟩ := appendBuiltAt p2
    "light.piccs.round_transcript" (roundsCircuit interface offset)
    (roundsOffset interface offset) end2
    (RoundTranscript.flatConstraints_varsBelow _ _ p2.current
      (assumptionsAt p2.current).rounds) built3 agrees3 rows3
  -- Evaluation: claimed sum, SumCheck chain, evaluation terminals.
  obtain ⟨built4, agrees4, rows4⟩ := InitialClaim.build (initialInterface interface offset)
    p3.current (initialOffset interface offset) (assumptionsAt p3.current).initial
  obtain ⟨p4, ops4, end4, preserves4, -⟩ := appendBuiltAt p3
    "light.piccs.initial_claim" (initialCircuit interface offset)
    (initialOffset interface offset) end3
    (InitialClaim.flatConstraints_varsBelow _ _ p3.current
      (assumptionsAt p3.current).initial) built4 agrees4 rows4
  have through4 := preserves1.trans (preserves2.trans (preserves3.trans preserves4))
  have wired4 := wired relation offset target ajtai running fresh proof interface p4.current
    (representsAt p4 through4)
  have statementEq4 := LightPiCCS.statementState_eq target ajtai running fresh
    (statementInterface interface offset) offset p4.current _ wired4.statementFresh
    (spec_of_prefix p4 _ (statementCircuit interface offset) offset
      (mem_grow ops4 (mem_grow ops3 (mem_grow ops2 (mem_last ops1))))
      (assumptionsAt p4.current).statement)
  have challenges4 := LightPiCCS.challenges_eq target ajtai running fresh proof.piCcsRounds
    (challengeInterface interface offset) (challengeOffset interface offset) p4.current
    (wired4.challengeStart.trans statementEq4)
    (spec_of_prefix p4 _ (challengeCircuit interface offset) _
      (mem_grow ops4 (mem_grow ops3 (mem_last ops2))) (assumptionsAt p4.current).challenge)
  have rounds4 := LightPiCCS.roundPoint_eq target ajtai running fresh proof.piCcsRounds
    (roundsInterface interface offset) (roundsOffset interface offset) p4.current
    (wired4.roundsStart.trans challenges4.2.2) wired4.roundsMessages
    (spec_of_prefix p4 _ (roundsCircuit interface offset) _
      (mem_grow ops4 (mem_last ops3)) (assumptionsAt p4.current).rounds)
  have initial4 := LightPiCCS.initial_eq target ajtai running fresh
    (initialInterface interface offset) (initialOffset interface offset) p4.current _
    (wired4.initialGamma.trans challenges4.2.1) wired4.initialPad wired4.initialMatrix
    (spec_of_prefix p4 _ (initialCircuit interface offset) _
      (mem_last ops4) (assumptionsAt p4.current).initial)
  have chain := LightPiCCSPhase.chain_of_check target ajtai running fresh proof accepted
  obtain ⟨built5, agrees5, rows5, -⟩ := SumcheckChain.chain_build
    (sumcheckInterface interface offset) (sumcheckOffset interface offset) p4.current
    (assumptionsAt p4.current).sumcheck _ proof.piCcsRounds _ _
    (wired4.sumcheckInitial.trans initial4) wired4.sumcheckRounds
    (wired4.sumcheckPoint.trans rounds4.1) chain
  obtain ⟨p5, ops5, end5, preserves5, -⟩ := appendBuiltAt p4
    "light.piccs.sumcheck_chain" (sumcheckCircuit interface offset)
    (sumcheckOffset interface offset) end4
    (fun expression member => by
      unfold sumcheckCircuit
      rw [SumcheckChain.localLength_eq]
      exact SumcheckChain.flatConstraints_varsBelow _ _
        (assumptionsAt (fun _ => 0)).sumcheck expression member)
    built5 agrees5 rows5
  obtain ⟨built6, agrees6, rows6⟩ := PointWeightedHorner.Owned.build
    (EvalKTerminal.coreInterface (evalKInterface interface offset)) p5.current
    (evalKOffset interface offset) (by decide) (assumptionsAt p5.current).evalK
  obtain ⟨p6, ops6, end6, preserves6, -⟩ := appendBuiltAt p5
    "light.piccs.eval_K_terminal" (evalKCircuit interface offset)
    (evalKOffset interface offset) end5
    (PointWeightedHorner.Owned.flatConstraints_varsBelow _ (by decide) _ p5.current
      (assumptionsAt p5.current).evalK) built6 agrees6 rows6
  obtain ⟨built7, agrees7, rows7⟩ := PointWeightedHorner.Owned.build
    (EvalATerminal.coreInterface (evalAInterface interface offset)) p6.current
    (evalAOffset interface offset) (by decide) (assumptionsAt p6.current).evalA
  obtain ⟨p7, ops7, end7, preserves7, -⟩ := appendBuiltAt p6
    "light.piccs.eval_A_terminal" (evalACircuit interface offset)
    (evalAOffset interface offset) end6
    (PointWeightedHorner.Owned.flatConstraints_varsBelow _ (by decide) _ p6.current
      (assumptionsAt p6.current).evalA) built7 agrees7 rows7
  -- Terminal: CCS and norm terms, final identity, output binding.
  obtain ⟨built8, agrees8, rows8⟩ := CcsTerminal.build relation (ccsInterface relation interface offset)
    p7.current (ccsOffset interface offset) (assumptionsAt p7.current).ccs
  obtain ⟨p8, ops8, end8, preserves8, -⟩ := appendBuiltAt p7
    "light.piccs.ccs_terminal" (ccsCircuit relation interface offset)
    (ccsOffset interface offset) end7
    (CcsTerminal.flatConstraints_varsBelow relation _ _ (assumptionsAt (fun _ => 0)).ccs)
    built8 agrees8 rows8
  obtain ⟨built9, agrees9, rows9⟩ := NormTerminal.build (normInterface relation interface offset)
    p8.current (normOffset relation interface offset) (assumptionsAt p8.current).norm
  obtain ⟨p9, ops9, end9, preserves9, -⟩ := appendBuiltAt p8
    "light.piccs.norm_terminal" (normCircuit relation interface offset)
    (normOffset relation interface offset) end8
    (Horner.Owned.flatConstraints_varsBelow
      (NormTerminal.ownedInterface (normInterface relation interface offset)) _ p8.current
      (assumptionsAt p8.current).norm) built9 agrees9 rows9
  have through9 := through4.trans (preserves5.trans (preserves6.trans (preserves7.trans
    (preserves8.trans preserves9))))
  have wired9 := wired relation offset target ajtai running fresh proof interface p9.current
    (representsAt p9 through9)
  have statementEq9 := LightPiCCS.statementState_eq target ajtai running fresh
    (statementInterface interface offset) offset p9.current _ wired9.statementFresh
    (spec_of_prefix p9 _ (statementCircuit interface offset) offset
      
      (mem_grow ops9 (mem_grow ops8 (mem_grow ops7 (mem_grow ops6 (mem_grow ops5 (mem_grow ops4 (mem_grow ops3 (mem_grow ops2 (mem_last ops1)))))))))
      (assumptionsAt p9.current).statement)
  have challenges9 := LightPiCCS.challenges_eq target ajtai running fresh proof.piCcsRounds
    (challengeInterface interface offset) (challengeOffset interface offset) p9.current
    (wired9.challengeStart.trans statementEq9)
    (spec_of_prefix p9 _ (challengeCircuit interface offset) _
      
      (mem_grow ops9 (mem_grow ops8 (mem_grow ops7 (mem_grow ops6 (mem_grow ops5 (mem_grow ops4 (mem_grow ops3 (mem_last ops2))))))))
      (assumptionsAt p9.current).challenge)
  have rounds9 := LightPiCCS.roundPoint_eq target ajtai running fresh proof.piCcsRounds
    (roundsInterface interface offset) (roundsOffset interface offset) p9.current
    (wired9.roundsStart.trans challenges9.2.2) wired9.roundsMessages
    (spec_of_prefix p9 _ (roundsCircuit interface offset) _
      
      (mem_grow ops9 (mem_grow ops8 (mem_grow ops7 (mem_grow ops6 (mem_grow ops5 (mem_grow ops4 (mem_last ops3)))))))
      (assumptionsAt p9.current).rounds)
  have initial9 := LightPiCCS.initial_eq target ajtai running fresh
    (initialInterface interface offset) (initialOffset interface offset) p9.current _
    (wired9.initialGamma.trans challenges9.2.1) wired9.initialPad wired9.initialMatrix
    (spec_of_prefix p9 _ (initialCircuit interface offset) _
      
      (mem_grow ops9 (mem_grow ops8 (mem_grow ops7 (mem_grow ops6 (mem_grow ops5 (mem_last ops4))))))
      (assumptionsAt p9.current).initial)
  have chain9 := SumcheckChain.spec_implies_chain (sumcheckInterface interface offset)
    (sumcheckOffset interface offset) p9.current _ proof.piCcsRounds _ _
    (wired9.sumcheckInitial.trans initial9) wired9.sumcheckRounds
    (wired9.sumcheckPoint.trans rounds9.1) rfl
    (spec_of_prefix p9 _ (sumcheckCircuit interface offset) _
      
      (mem_grow ops9 (mem_grow ops8 (mem_grow ops7 (mem_grow ops6 (mem_last ops5)))))
      (assumptionsAt p9.current).sumcheck)
  have terminal9 := chain_terminal_unique _ _ _ _ _ chain9 chain
  have pad9 := LightPiCCS.padAtMessage_eq target ajtai running fresh
    (evalKInterface interface offset) (evalKOffset interface offset) p9.current _ _
    (LightPiCCSPhase.lightMessage target ajtai running fresh proof)
    (point_eq (wired9.evalKRoundPoint.trans rounds9.1)) wired9.evalKPriorPoint
    (wired9.evalKGamma.trans challenges9.2.1) wired9.evalKOutput
    (spec_of_prefix p9 _ (evalKCircuit interface offset) _
      (mem_grow ops9 (mem_grow ops8 (mem_grow ops7 (mem_last ops6))))
      (assumptionsAt p9.current).evalK)
  have matrix9 := LightPiCCS.matrixAtMessage_eq target ajtai running fresh
    (evalAInterface interface offset) (evalAOffset interface offset) p9.current _ _
    (LightPiCCSPhase.lightMessage target ajtai running fresh proof)
    (point_eq (wired9.evalARoundPoint.trans rounds9.1)) wired9.evalAPriorPoint
    (wired9.evalAGamma.trans challenges9.2.1) wired9.evalAOutput
    (spec_of_prefix p9 _ (evalACircuit interface offset) _
      (mem_grow ops9 (mem_grow ops8 (mem_last ops7)))
      (assumptionsAt p9.current).evalA)
  have ccs9 := LightPiCCS.ccsAtMessage_eq target ajtai running fresh
    (ccsInterface relation interface offset) (ccsOffset interface offset) p9.current
    (LightPiCCS.lightCoins target ajtai running fresh proof.piCcsRounds).gamma
    (LightPiCCSPhase.lightMessage target ajtai running fresh proof) wired9.ccsFresh
    (spec_of_prefix p9 _ (ccsCircuit relation interface offset) _
      (mem_grow ops9 (mem_last ops8)) (assumptionsAt p9.current).ccs)
  have norm9 := LightPiCCS.normAtMessage_eq (normInterface relation interface offset)
    (normOffset relation interface offset) p9.current _
    (LightPiCCSPhase.lightMessage target ajtai running fresh proof)
    (wired9.normGamma.trans challenges9.2.1) wired9.normSource
    (spec_of_prefix p9 _ (normCircuit relation interface offset) _
      (mem_last ops9) (assumptionsAt p9.current).norm)
  have finalSpec := FinalIdentity.terminal_implies_spec
    (finalIdentityInterface relation interface offset)
    (finalIdentityOffset relation interface offset) p9.current rfl
    (LightPiCCS.lightInput target ajtai running fresh) _ _ _
    (LightPiCCSPhase.lightMessage target ajtai running fresh proof)
    (point_eq (wired9.finalRoundPoint.trans rounds9.1))
    (point_eq (wired9.finalAlpha.trans challenges9.1)) (wired9.finalGamma.trans challenges9.2.1)
    (wired9.finalPad.trans pad9) (wired9.finalMatrix.trans matrix9) (wired9.finalCcs.trans ccs9)
    (wired9.finalNorm.trans norm9) (wired9.sumcheckTerminal.symm.trans terminal9)
  obtain ⟨built10, agrees10, rows10⟩ := FinalIdentity.shapeCompleteness
    (finalIdentityInterface relation interface offset) (by decide) p9.current
    (finalIdentityOffset relation interface offset) (assumptionsAt p9.current).finalIdentity
    finalSpec
  obtain ⟨p10, ops10, end10, preserves10, -⟩ := appendBuiltAt p9
    "light.piccs.final_identity" (finalIdentityCircuit relation interface offset)
    (finalIdentityOffset relation interface offset) end9
    (FinalIdentity.shapeFlatConstraints_varsBelow _ (by decide) _ p9.current
      (assumptionsAt p9.current).finalIdentity) built10 agrees10 rows10
  obtain ⟨built11, agrees11, rows11⟩ := OutputBinding.build (outputInterface interface offset)
    p10.current (outputOffset relation interface offset) (assumptionsAt p10.current).output
  obtain ⟨p11, ops11, end11, preserves11, -⟩ := appendBuiltAt p10
    "light.piccs.output_binding" (outputCircuit interface offset)
    (outputOffset relation interface offset) end10
    (OutputBinding.flatConstraints_varsBelow _ _ p10.current
      (assumptionsAt p10.current).output) built11 agrees11 rows11
  refine ⟨p11, ?_, end11, through9.trans (preserves10.trans preserves11)⟩
  rw [ops11, ops10, ops9, ops8, ops7, ops6, ops5, ops4, ops3, ops2, ops1]
  simp only [opsAt, List.append_assoc, List.cons_append, List.nil_append]

end NightstreamFPrime.Layout.BoundedDepth.LightPiCCSCompleteness
