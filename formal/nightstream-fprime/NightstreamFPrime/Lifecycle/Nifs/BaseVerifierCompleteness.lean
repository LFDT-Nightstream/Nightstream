import NightstreamFPrime.Lifecycle.Nifs.BaseCompleteness

/-!
Owns acceptance of the canonical base dummy by the complete production NIFS
verifier when its actual bounded sampler succeeds. Public-input bounds are
derived from the encoded prior hash and valid sampled challenges. Acceptance
does not assert a fresh opening or valid openings for the dummy D children.
-/

namespace NightstreamFPrime.Lifecycle.Nifs.BaseCompleteness

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism
open NightstreamFPrime.Lifecycle.PaperAlgebra
open ConcreteCarrier

private theorem combined_public_bound
    {shape : Phi81Relation.Shape} {count : Nat}
    (challenges : Fin count → RingF)
    (inputs : Fin count → Phi81Relation.PublicInput shape)
    (valid : ∀ source, Phi81Relation.PiRLCAlgebra.Challenge.challengeValid
      (challenges source))
    (bounded : ∀ source column, centeredMagnitude (inputs source column) < 2)
    (column : Fin shape.publicWidth) :
    centeredMagnitude
      (Phi81Relation.PiRLCAlgebra.PublicInput.combinePublicInputs challenges inputs column) ≤
      count * 216 := by
  induction count with
  | zero =>
      exact Nat.le_refl 0
  | succ count ih =>
      have headBound := Phi81Relation.PiRLCAlgebra.Norm.Product.ringFMul_le_expansion
        (challenges 0)
        (Phi81Relation.PiRLCAlgebra.PublicInput.publicBlock (inputs 0)
          (Phi81Relation.PiRLCAlgebra.PublicInput.publicBlockIndex shape column))
        (valid 0) (fun _ => bounded 0 _)
        (Phi81Relation.PiRLCAlgebra.PublicInput.publicLaneIndex column)
      have tailBound := ih (fun source => challenges source.succ)
        (fun source => inputs source.succ) (fun source => valid source.succ)
        (fun source => bounded source.succ)
      change centeredMagnitude
        (Phi81Relation.PiRLCAlgebra.PublicInput.publicAct (challenges 0) (inputs 0) column) ≤
          216 at headBound
      change centeredMagnitude
        (Phi81Relation.PiRLCAlgebra.PublicInput.publicAct (challenges 0) (inputs 0) column +
          Phi81Relation.PiRLCAlgebra.PublicInput.combinePublicInputs
            (fun source => challenges source.succ) (fun source => inputs source.succ) column) ≤
        (count + 1) * 216
      exact Nat.le_trans (Phi81Relation.PiRLCAlgebra.Norm.Centered.centeredMagnitude_add_le _ _)
        (by have added := Nat.add_le_add headBound tailBound; omega)

private theorem combine_commitments_zero {rows count : Nat}
    (challenges : Fin count → RingF) :
    Phi81Relation.PiRLCAlgebra.Commitment.combineCommitments challenges
      (fun _ => Phi81Relation.PiRLCAlgebra.Commitment.commitmentZero (verifierRows := rows)) =
      Phi81Relation.PiRLCAlgebra.Commitment.commitmentZero := by
  induction count with
  | zero => rfl
  | succ count ih =>
      rw [Phi81Relation.PiRLCAlgebra.Commitment.combineCommitments, ih]
      funext row lane
      change (ringFMul (challenges 0) ringFZero) lane + 0 = 0
      rw [CarrierAction.ringFMul_zero_right]
      rfl

private theorem recompose_commitments_zero {rows count : Nat}
    (weights : Fin count → F) :
    Phi81Relation.PiDECAlgebra.Commitment.combineCommitments weights
      (fun _ => Phi81Relation.PiRLCAlgebra.Commitment.commitmentZero (verifierRows := rows)) =
      Phi81Relation.PiRLCAlgebra.Commitment.commitmentZero := by
  induction count with
  | zero => rfl
  | succ count ih =>
      rw [Phi81Relation.PiDECAlgebra.Commitment.combineCommitments, ih]
      funext row lane
      change weights 0 * (0 : F) + 0 = 0
      exact Fin.add_zero _ |>.trans (Fin.mul_zero _)

private theorem combine_evaluation_zero {count : Nat}
    (challenges : Fin count → RingF) :
    PiRLCFinite.combineEvaluation challenges (fun _ => BaseLinear.evaluationZero) =
      BaseLinear.evaluationZero := by
  induction count with
  | zero => rfl
  | succ count ih =>
      rw [PiRLCFinite.combineEvaluation, ih]
      change ringKAdd (ringKMul _ ringKZero) ringKZero = ringKZero
      rw [RingKAction.ringKMul_right_zero]
      rfl

private theorem recompose_evaluation_zero {count : Nat}
    (weights : Fin count → F) :
    BaseLinear.combineEvaluations weights (fun _ => BaseLinear.evaluationZero) =
      BaseLinear.evaluationZero := by
  induction count with
  | zero => rfl
  | succ count ih =>
      rw [BaseLinear.combineEvaluations, ih]
      funext lane
      change K.add (K.mul (K.embed (weights 0)) K.zero) K.zero = K.zero
      have mulZero : K.mul (K.embed (weights 0)) K.zero = K.zero :=
        extensionLaws.mul_zero _
      rw [mulZero]
      rfl

private theorem combined_evaluations_zero {count : Nat}
    (positive : 0 < count) (challenges : Fin count → RingF) :
    PaperAlgebra.combineEvaluations challenges (fun _ => #[evaluationZero]) =
      #[evaluationZero] := by
  rw [combineEvaluations_singletons positive]
  have family : combineEvaluationFamily challenges (fun _ => evaluationZero) =
      evaluationZero := by
    simp only [combineEvaluationFamily, evaluationZero, combine_evaluation_zero]
  rw [family]

private theorem recomposed_evaluations_zero :
    PaperAlgebra.recomposeEvaluations (fun _ => #[evaluationZero]) = #[evaluationZero] := by
  change #[recomposeEvaluationFamily (fun _ => evaluationZero)] = #[evaluationZero]
  have family : recomposeEvaluationFamily (fun _ => evaluationZero) = evaluationZero := by
    simp only [recomposeEvaluationFamily, evaluationZero, recompose_evaluation_zero]
  exact congrArg (fun value => #[value]) family

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}
  (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
  (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
  (prior : HashPreimage (logicalWidth := logicalWidth) (publicFits := publicFits))

private theorem source_public_bounded
    (source : Fin (ProductionKey.key relation ajtai).arity.total)
    (column : Fin (ringDegree * publicRingColumns)) :
    centeredMagnitude
      (((ProductionKey.key relation ajtai).piCcsOutputs defaultRunning
        (baseFresh prior) zeroProof source).publicInput column) < 2 := by
  change centeredMagnitude
    (Fin.addCases (fun _ : Fin 1 => encHash (publicFits := publicFits) (stateHash prior))
      (fun _ : Fin 16 => fun _ => (0 : F)) source column) < 2
  refine Fin.addCases (m := 1) (n := 16) ?_ ?_ source
  · intro index
    have branch :
        Fin.addCases (motive := fun _ : Fin (1 + 16) => Fin (ringDegree * publicRingColumns) → F)
          (fun _ => encHash (publicFits := publicFits) (stateHash prior))
          (fun _ _ => (0 : F)) (Fin.castAdd 16 index) =
          encHash (publicFits := publicFits) (stateHash prior) := Fin.addCases_left index
    exact Eq.mpr (congrArg (fun value : F => centeredMagnitude value < 2)
      (congrFun branch column)) (encHash_norm (stateHash prior) column)
  · intro index
    have branch :
        Fin.addCases (motive := fun _ : Fin (1 + 16) => Fin (ringDegree * publicRingColumns) → F)
          (fun _ => encHash (publicFits := publicFits) (stateHash prior))
          (fun _ _ => (0 : F)) (Fin.natAdd 1 index) = fun _ => (0 : F) :=
      Fin.addCases_right index
    exact Eq.mpr (congrArg (fun value : F => centeredMagnitude value < 2)
      (congrFun branch column)) (Nat.zero_lt_succ 1)

private theorem source_commitments_zero :
    (fun source => ((ProductionKey.key relation ajtai).piCcsOutputs defaultRunning
      (baseFresh prior) zeroProof source).commitment) =
      fun _ => Phi81Relation.PiRLCAlgebra.Commitment.commitmentZero := by
  funext source
  change Fin.addCases (motive := fun _ : Fin (1 + 16) => PaperAlgebra.Commitment)
    (fun _ : Fin 1 => fun _ _ => (0 : F))
    (fun _ : Fin 16 => fun _ => ringFZero) source =
      Phi81Relation.PiRLCAlgebra.Commitment.commitmentZero
  refine Fin.addCases (m := 1) (n := 16) ?_ ?_ source
  · intro index
    rw [Fin.addCases_left (motive := fun _ : Fin (1 + 16) => PaperAlgebra.Commitment)]
    rfl
  · intro index
    rw [Fin.addCases_right (motive := fun _ : Fin (1 + 16) => PaperAlgebra.Commitment)]
    rfl

private theorem parent_evaluations_zero
    (challenges : Fin (ProductionKey.key relation ajtai).arity.total → RingF) :
    ((ProductionKey.key relation ajtai).parentForChallenges defaultRunning
      (baseFresh prior) zeroProof challenges).evaluations = #[evaluationZero] := by
  change PaperAlgebra.combineEvaluations challenges (fun _ => #[evaluationZero]) = _
  exact combined_evaluations_zero (show 0 < 17 from by decide) challenges

/-- The verifier's sampled base parent satisfies the public norm bound. This
uses the real encoded prior hash and derives the bound from the production
17-source inequality; it assumes no opening for the base dummy claim. -/
theorem zeroProof_parentBounded
    (challenges : Fin (ProductionKey.key relation ajtai).arity.total → RingF)
    (sampled : (ProductionKey.key relation ajtai).piRlcChallenges defaultRunning
      (baseFresh prior) zeroProof = some challenges) :
    (ProductionKey.key relation ajtai).piDecPublicInputSplit.parentBounded
      ((ProductionKey.key relation ajtai).parentForChallenges defaultRunning
        (baseFresh prior) zeroProof challenges).publicInput := by
  intro column
  have valid := (ProductionKey.key relation ajtai).piRlcResponseValid
    ((ProductionKey.key relation ajtai).piCcsExecution defaultRunning
      (baseFresh prior) zeroProof).outgoingState challenges sampled
  have bound := combined_public_bound challenges
    (fun source => ((ProductionKey.key relation ajtai).piCcsOutputs defaultRunning
      (baseFresh prior) zeroProof source).publicInput) valid
    (source_public_bounded relation ajtai prior) column
  change centeredMagnitude
    (Phi81Relation.PiRLCAlgebra.PublicInput.combinePublicInputs challenges
      (fun source => ((ProductionKey.key relation ajtai).piCcsOutputs defaultRunning
        (baseFresh prior) zeroProof source).publicInput) column) < 65536
  have numeric : 17 * 216 < 65536 := by simpa using production_rlc_bound_one_fresh
  exact Nat.lt_of_le_of_lt bound numeric

private theorem piDecCheck_of_sampled_parent
    (key : ProductionKey.KeyType relation)
    (running : Running (logicalWidth := logicalWidth) (publicFits := publicFits))
    (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
    (proof : Proof 9) (challenges : Fin key.arity.total → RingF)
    (sampled : key.piRlcChallenges running fresh proof = some challenges)
    (accepted : PiDEC.PaperVerifier.Accepted key.piDecAlgebra
      key.piDecPublicInputSplit key.piDecEvaluationArity
      (key.piDecAttemptForParent proof (key.parentForChallenges running fresh proof challenges))) :
    Nifs.PaperNonInteractive.piDecCheck key running fresh proof = true := by
  apply (Nifs.PaperNonInteractive.piDecCheck_eq_true_iff key running fresh proof).mpr
  refine ⟨_, ?_, accepted⟩
  simp only [Nifs.PaperNonInteractive.Key.piDecAttempt, Nifs.PaperNonInteractive.Key.parent,
    sampled, Option.map_some]

private theorem verify_some_of_checks
    (key : ProductionKey.KeyType relation)
    (running : Running (logicalWidth := logicalWidth) (publicFits := publicFits))
    (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
    (proof : Proof 9)
    (cCheck : Nifs.PaperNonInteractive.piCcsCheck key running fresh proof = true)
    (dCheck : Nifs.PaperNonInteractive.piDecCheck key running fresh proof = true) :
    ∃ output, Nifs.PaperNonInteractive.verify key running fresh proof = some output := by
  obtain ⟨attempt, attemptEq, accepted⟩ :=
    (Nifs.PaperNonInteractive.piDecCheck_eq_true_iff key running fresh proof).mp dCheck
  refine ⟨key.outputForAttempt proof attempt
    (key.piDecPublicInputSplit.split attempt.parent.publicInput), ?_⟩
  apply (Nifs.PaperNonInteractive.verify_eq_some_iff key running fresh proof _).mpr
  refine ⟨cCheck, dCheck, ?_⟩
  rw [Nifs.PaperNonInteractive.Key.output, attemptEq, Option.bind_some,
    PiDEC.PaperVerifier.PublicInputSplit.checked_eq_some _ _ accepted.parentBounded]
  rfl

/-- The zero base messages pass the actual PiDEC checks after successful
sampling. This is verifier acceptance, not validity of the dummy child openings. -/
theorem zeroProof_piDecCheck_of_sampler
    (challenges : Fin (ProductionKey.key relation ajtai).arity.total → RingF)
    (sampled : (ProductionKey.key relation ajtai).piRlcChallenges defaultRunning
      (baseFresh prior) zeroProof = some challenges) :
    Nifs.PaperNonInteractive.piDecCheck (ProductionKey.key relation ajtai)
      defaultRunning (baseFresh prior) zeroProof = true := by
  apply piDecCheck_of_sampled_parent relation (ProductionKey.key relation ajtai)
    defaultRunning (baseFresh prior) zeroProof challenges sampled
  refine ⟨zeroProof_parentBounded relation ajtai prior challenges sampled, rfl, ?_,
    fun _ => rfl, ?_, ?_⟩
  · change ((ProductionKey.key relation ajtai).parentForChallenges defaultRunning
      (baseFresh prior) zeroProof challenges).evaluations.size = 1
    rw [parent_evaluations_zero relation ajtai prior challenges]
    rfl
  · change Phi81Relation.PiRLCAlgebra.Commitment.combineCommitments challenges
        (fun source => ((ProductionKey.key relation ajtai).piCcsOutputs defaultRunning
          (baseFresh prior) zeroProof source).commitment) =
      Phi81Relation.PiDECAlgebra.Commitment.recomposeCommitment
        (fun _ => Phi81Relation.PiRLCAlgebra.Commitment.commitmentZero)
    rw [source_commitments_zero relation ajtai prior, combine_commitments_zero]
    exact (recompose_commitments_zero _).symm
  · change ((ProductionKey.key relation ajtai).parentForChallenges defaultRunning
      (baseFresh prior) zeroProof challenges).evaluations =
        PaperAlgebra.recomposeEvaluations (fun _ => #[evaluationZero])
    rw [parent_evaluations_zero relation ajtai prior challenges, recomposed_evaluations_zero]

/-- The canonical base dummy is accepted by the complete production NIFS
verifier when its actual bounded sampler succeeds. The result uses the
verifier-computed point and split public inputs; no child opening is asserted. -/
theorem zeroProof_verify_of_sampler
    (challenges : Fin (ProductionKey.key relation ajtai).arity.total → RingF)
    (sampled : (ProductionKey.key relation ajtai).piRlcChallenges defaultRunning
      (baseFresh prior) zeroProof = some challenges) :
    ∃ output, Nifs.PaperNonInteractive.verify (ProductionKey.key relation ajtai)
      defaultRunning (baseFresh prior) zeroProof = some output := by
  exact verify_some_of_checks relation (ProductionKey.key relation ajtai)
    defaultRunning (baseFresh prior) zeroProof
    (zeroProof_piCcsCheck relation ajtai prior)
    (zeroProof_piDecCheck_of_sampler relation ajtai prior challenges sampled)

end NightstreamFPrime.Lifecycle.Nifs.BaseCompleteness
