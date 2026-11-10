import NightstreamFPrime.Lifecycle.BoundedDepth.Keys
import NightstreamFPrime.Lifecycle.PiCCS.v1_2.RoundTranscript
import NightstreamFPrime.Lifecycle.PiCCS.v1_2.StatementAbsorption

/-! The refresh Π_CCS statement (CW-3.1). The refresh key is today's production key with the
refresh transcript (`Keys.refreshKey`): it absorbs the refresh fold domain, then the fresh
instance's public-input blocks and the 16 Π_DEC children's commitments and evaluation families
(`refreshWords`). The statement absorption is the duplex gadget on these two actions
(`statementActions`), and its trace ends in the refresh key's public-input state
(`statementState_eq`). -/

namespace NightstreamFPrime.Lifecycle.BoundedDepth.RefreshPiCCS

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Lifecycle.PiCCS.v1_2

variable {logicalWidth : Nat}
  {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}

/-- The 16 children's commitment and evaluation words, in the order of `childWords`. -/
def childWordsExpr (running : StatementAbsorption.RunningExpr logicalWidth publicFits) :
    List Expr :=
  (List.finRange productionShape.runningCount).flatMap fun child =>
    StatementAbsorption.serializeCommitmentExpr (running.commitment child) ++
      (StatementAbsorption.serializeEvalKExpr (running.evaluation child) ++
        StatementAbsorption.serializeEvalAExpr (running.evaluation child))

/-- The child words evaluate to the children's serialized commitments and evaluations. -/
theorem childWordsExpr_eval (running : StatementAbsorption.RunningExpr logicalWidth publicFits)
    (env : Env) (children : Children logicalWidth publicFits)
    (commitments : ∀ child row lane,
      (running.commitment child row lane).eval env = children.commitments child row lane)
    (evaluations : ∀ child,
      StatementAbsorption.evalEvaluation (running.evaluation child) env =
        children.evaluations child) :
    Hash.evalList env (childWordsExpr running) = childWords children := by
  unfold childWordsExpr childWords
  simp only [Hash.evalList, List.map_flatMap]
  apply congrArg (List.flatMap · (List.finRange productionShape.runningCount))
  funext child
  simp only [List.map_append]
  rw [← Hash.evalList, ← Hash.evalList, ← Hash.evalList,
    StatementAbsorption.serializeCommitmentExpr_eval, StatementAbsorption.serializeEvalKExpr_eval,
    StatementAbsorption.serializeEvalAExpr_eval, evaluations child]
  have commitmentEq : (fun row lane => (running.commitment child row lane).eval env) =
      children.commitments child :=
    funext fun row => funext fun lane => commitments child row lane
  rw [commitmentEq]
  rfl

/-- The refresh statement absorption: the refresh fold domain, then the fresh blocks and the
child words. -/
def statementActions (statement : StatementAbsorption.Interface logicalWidth publicFits)
    (offset : Nat) : List Duplex.Formal.Action :=
  [.absorb (StatementAbsorption.constantWords refreshFoldDomainChunk),
    .absorb ((StatementAbsorption.publicInputBlocks statement offset).flatten ++
      childWordsExpr (statement.running offset))]

/-- The refresh statement trace ends in the refresh key's public-input state, from which its
Π_CCS challenges start (SuperNeo v1.2 Section 7.3). -/
theorem statementState_eq (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (children : Children logicalWidth publicFits)
    (fresh : Spec.Folding.Nifs.PaperNonInteractive.Fresh PaperAlgebra.Commitment
      (PaperAlgebra.PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits))
      productionShape)
    (statement : StatementAbsorption.Interface logicalWidth publicFits) (offset : Nat)
    (env : Env) (final : Transcript.State)
    (freshWords : StatementAbsorption.evalFresh (statement.fresh offset) env = fresh)
    (commitments : ∀ child row lane,
      ((statement.running offset).commitment child row lane).eval env =
        children.commitments child row lane)
    (evaluations : ∀ child,
      StatementAbsorption.evalEvaluation ((statement.running offset).evaluation child) env =
        children.evaluations child)
    (trace : Duplex.Formal.TraceHolds (StatementAbsorption.evalState env Hash.zeroE)
      ((statementActions statement offset).map (Duplex.Formal.Action.eval env)) final) :
    final = (refreshKey relation ajtai).publicInputState children fresh := by
  rw [StatementAbsorption.wordsTrace_eq refreshFoldDomainChunk _ env final trace]
  have words : Hash.evalList env
      ((StatementAbsorption.publicInputBlocks statement offset).flatten ++
        childWordsExpr (statement.running offset)) = refreshWords children fresh := by
    simp only [Hash.evalList, List.map_append]
    rw [← Hash.evalList, ← Hash.evalList, StatementAbsorption.publicInputWords_eval,
      childWordsExpr_eval _ env children commitments evaluations, freshWords]
    rfl
  rw [words]
  rfl

section Transcript

variable (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
  (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
  (children : Children logicalWidth publicFits)
  (fresh : Spec.Folding.Nifs.PaperNonInteractive.Fresh PaperAlgebra.Commitment
    (PaperAlgebra.PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits))
    productionShape)
  (proof : Spec.Folding.Nifs.PaperNonInteractive.Proof K PaperAlgebra.Commitment productionShape
    (ProductionKey.degreeBound relation))

/-- The refresh Π_CCS transcript statement: the state after the refresh statement absorption
and the refresh statement's verifier input. -/
noncomputable def refreshContext : ChallengeDerivation.Context where
  priorState := (refreshKey relation ajtai).publicInputState children fresh
  input := ((refreshKey relation ajtai).statement children fresh).verifierInput
    (refreshKey relation ajtai).lift

/-- The refresh key's Π_CCS coins (SuperNeo v1.2 Section 7.3, Steps 1 and 3). -/
noncomputable abbrev refreshCoins :=
  ((refreshKey relation ajtai).piCcsExecution children fresh proof).coins

theorem refreshCoins_eq : refreshCoins relation ajtai children fresh proof =
    FiatShamir.derive ChallengeDerivation.oracle (refreshContext relation ajtai children fresh)
      { rounds := fun roundIndex => (proof.piCcsRounds roundIndex).toMessage } :=
  rfl

/-- The challenge-derivation rows from the refresh statement state give the refresh key's `α`
and `γ` and the state the SumCheck rounds start from. -/
theorem challenges_eq (interface : ChallengeDerivation.Interface) (offset : Nat) (env : Env)
    (initialEq : ChallengeDerivation.evalState env (interface.initialState offset) =
      (refreshKey relation ajtai).publicInputState children fresh)
    (specification : ChallengeDerivation.SpecHolds interface offset env) :
    ChallengeDerivation.evalAlpha interface offset env =
        (refreshCoins relation ajtai children fresh proof).alpha ∧
      ChallengeDerivation.evalGamma interface offset env =
        (refreshCoins relation ajtai children fresh proof).gamma ∧
      ChallengeDerivation.evalState env (ChallengeDerivation.finalState interface offset) =
        (FiatShamir.derivePreSumcheck ChallengeDerivation.oracle
          (refreshContext relation ajtai children fresh)).state := by
  have derived := ChallengeDerivation.spec_implies_derivePreSumcheck interface offset env
    (refreshContext relation ajtai children fresh) initialEq specification
  let certificate : FiatShamir.Certificate K productionShape :=
    { rounds := fun roundIndex => (proof.piCcsRounds roundIndex).toMessage }
  rw [refreshCoins_eq]
  exact ⟨derived.1.trans (Spec.Folding.PiCCS.Transcript.derive_alpha_eq_preSumcheck
      ChallengeDerivation.oracle (refreshContext relation ajtai children fresh) certificate).symm,
    derived.2.1.trans (Spec.Folding.PiCCS.Transcript.derive_gamma_eq_preSumcheck
      ChallengeDerivation.oracle (refreshContext relation ajtai children fresh) certificate).symm,
    derived.2.2⟩

/-- The round-transcript rows give the refresh key's SumCheck point and final state. -/
theorem roundPoint_eq (interface : RoundTranscript.Interface (ProductionKey.degreeBound relation))
    (offset : Nat) (env : Env)
    (initialEq : RoundTranscript.evalState env (interface.initialState offset) =
      (FiatShamir.derivePreSumcheck ChallengeDerivation.oracle
        (refreshContext relation ajtai children fresh)).state)
    (roundsEq : ∀ roundIndex,
      (interface.round offset roundIndex).semanticPolynomial env = proof.piCcsRounds roundIndex)
    (specification : RoundTranscript.SpecHolds interface offset env) :
    (RoundTranscript.evalRoundPoint interface offset env).coordinates =
        (refreshCoins relation ajtai children fresh proof).roundPoint.coordinates ∧
      RoundTranscript.evalState env (RoundTranscript.finalState interface offset) =
        (refreshCoins relation ajtai children fresh proof).finalState := by
  let certificate : FiatShamir.Certificate K productionShape :=
    { rounds := fun roundIndex => (proof.piCcsRounds roundIndex).toMessage }
  have semantic : RoundTranscript.semanticRounds interface offset env = certificate.rounds :=
    funext fun roundIndex =>
      congrArg SumCheck.Finite.FixedPolynomial.toMessage (roundsEq roundIndex)
  obtain ⟨pointEq, finalEq⟩ := specification
  rw [initialEq, semantic] at pointEq finalEq
  have canonical := Spec.Folding.PiCCS.Transcript.derive_rounds_holds
    ChallengeDerivation.oracle (refreshContext relation ajtai children fresh) certificate
  rw [refreshCoins_eq]
  exact ⟨pointEq.trans canonical.roundPoint_eq.symm, finalEq.trans canonical.finalState_eq.symm⟩

end Transcript

end NightstreamFPrime.Lifecycle.BoundedDepth.RefreshPiCCS
