import NightstreamFPrime.Gadgets.Poseidon2.Duplex.WiringShift
import NightstreamFPrime.Lifecycle.PiCCS.v1_1.Formal
import NightstreamFPrime.Lifecycle.PiCCS.v1_1.FormalRows

/-!
Owns offset relocation for the symbolic PiCCS transcript outputs used by the
compact Stage 1 assembler.

The proofs compare only the fixed action shape and use the existing
recipe-free-wiring agreement theorem. They change no transcript action,
challenge order, circuit, row, or semantic predicate.
-/

namespace NightstreamFPrime.Layout.Stage1.PiCCSTranscriptRelocation

open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.SupportRange
open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Gadgets.Poseidon2.Duplex.Formal.WiringShift
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PiCCS.v1_1
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

theorem statementActions_sameShape
    {logicalWidth : Nat}
    {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (left right : StatementAbsorption.Interface logicalWidth publicFits)
    (leftOffset rightOffset : Nat) :
    SameShape (StatementAbsorption.actions left leftOffset)
      (StatementAbsorption.actions right rightOffset) := by
  simp only [StatementAbsorption.actions,
    StatementAbsorption.publicInputActions,
    StatementAbsorption.publicInputBlocks,
    List.map_append, List.map_cons, List.map_nil]
  refine .absorb _ _ rfl _ _ ?_
  refine .absorb _ _ ?_ _ _ ?_
  · apply inputChunks_length_eq_of_length_eq
    simp [StatementAbsorption.blockExpr,
      StatementAbsorption.priorDigestExpr]
  · rw [List.map_flatMap, List.map_flatMap]
    apply SameShape.flatMap
    intro index _
    simp only [List.map_cons, List.map_nil]
    refine .absorb _ _ ?_ _ _ ?_
    · apply inputChunks_length_eq_of_length_eq
      simp [StatementAbsorption.blockExpr,
        StatementAbsorption.serializeCommitmentExpr]
    · refine .absorb _ _ ?_ _ _ .nil
      apply inputChunks_length_eq_of_length_eq
      simp [StatementAbsorption.blockExpr,
        StatementAbsorption.serializePublicInputExpr]

private theorem zeroState_shift (delta : Nat) :
    state delta Hash.zeroE = Hash.zeroE := by
  funext lane
  rfl

theorem statementFinalState_localSupport
    {logicalWidth : Nat}
    {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : StatementAbsorption.Interface logicalWidth publicFits)
    (offset : Nat) :
    Duplex.Formal.StateSupported
      (StatementAbsorption.finalState interface offset)
      (Extend (fun _ => False) offset (offset + 224368)) := by
  have zeroSupport : Duplex.Formal.StateSupported Hash.zeroE
      (Extend (fun _ => False) offset offset) := by
    intro lane
    trivial
  have projected := Duplex.Formal.compileWiring_supported (fun _ => False)
    offset offset Hash.zeroE (StatementAbsorption.actions interface offset)
    (Nat.le_refl offset) zeroSupport
  have count := StatementAbsorption.recipeCount_eq interface offset
  unfold StatementAbsorption.recipeCount at count
  rw [count] at projected
  change Duplex.Formal.StateSupported
    (Duplex.Formal.compile offset Hash.zeroE
      (StatementAbsorption.actions interface offset)).output
    (Extend (fun _ => False) offset (offset + 224368))
  rw [← (Duplex.Formal.compileWiring_matches offset Hash.zeroE
    (StatementAbsorption.actions interface offset)).2]
  exact projected.2

/-- The statement transcript's symbolic final state moves by exactly the
parent offset delta. Payload values cannot change these wire positions. -/
theorem statementFinalState_shift
    {logicalWidth : Nat}
    {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (left right : StatementAbsorption.Interface logicalWidth publicFits)
    (leftOffset delta : Nat) :
    StatementAbsorption.finalState right (leftOffset + delta) =
      state delta (StatementAbsorption.finalState left leftOffset) := by
  let leftActions := StatementAbsorption.actions left leftOffset
  let rightActions := StatementAbsorption.actions right (leftOffset + delta)
  have shifted := compileWiring_shift_of_sameShape delta leftOffset Hash.zeroE
    (statementActions_sameShape left right leftOffset (leftOffset + delta))
  rw [zeroState_shift] at shifted
  calc
    StatementAbsorption.finalState right (leftOffset + delta) =
        (Duplex.Formal.compileWiring (leftOffset + delta) Hash.zeroE
          rightActions).output := by
      exact (Duplex.Formal.compileWiring_matches (leftOffset + delta)
        Hash.zeroE rightActions).2.symm
    _ = state delta
        (Duplex.Formal.compileWiring leftOffset Hash.zeroE leftActions).output :=
      shifted.2
    _ = state delta
        (StatementAbsorption.finalState left leftOffset) := by
      exact congrArg (state delta)
        (Duplex.Formal.compileWiring_matches leftOffset Hash.zeroE
          leftActions).2

/-- The fixed challenge schedule preserves the same uniform relocation from
its incoming statement state to its final state. -/
theorem challengeFinalState_shift
    (left right : ChallengeDerivation.Interface)
    (leftOffset delta : Nat)
    (initialShift : right.initialState (leftOffset + delta) =
      state delta (left.initialState leftOffset)) :
    ChallengeDerivation.finalState right (leftOffset + delta) =
      state delta (ChallengeDerivation.finalState left leftOffset) := by
  rw [ChallengeDerivation.finalState_eq_finalStateFast_pointwise,
    ChallengeDerivation.finalState_eq_finalStateFast_pointwise]
  unfold ChallengeDerivation.finalStateFast
  rw [ChallengeDerivation.layoutWiring_eq_compileWiring,
    ChallengeDerivation.layoutWiring_eq_compileWiring, initialShift]
  exact (compileWiring_shift delta leftOffset (left.initialState leftOffset)
    ChallengeDerivation.layoutActions).2

private theorem challengeLayoutRecipeCount
    (interface : ChallengeDerivation.Interface) (offset : Nat) :
    Duplex.Formal.recipeCount ChallengeDerivation.layoutActions = 51504 := by
  calc
    Duplex.Formal.recipeCount ChallengeDerivation.layoutActions =
        (ChallengeDerivation.layoutProgram interface offset).recipes.length :=
      (Duplex.Formal.compile_recipes_length offset
        (interface.initialState offset) ChallengeDerivation.layoutActions).symm
    _ = (ChallengeDerivation.program interface offset).recipes.length :=
      congrArg List.length
        (ChallengeDerivation.program_shape_eq_layout interface offset).1.symm
    _ = 51504 := ChallengeDerivation.program_recipes_length interface offset

theorem challengeFinalState_localSupport
    (interface : ChallengeDerivation.Interface) (base offset : Nat)
    (baseLeOffset : base ≤ offset)
    (initialSupport : Duplex.Formal.StateSupported
      (interface.initialState offset) (Extend (fun _ => False) base offset)) :
    Duplex.Formal.StateSupported
      (ChallengeDerivation.finalState interface offset)
      (Extend (fun _ => False) base (offset + 51504)) := by
  have projected := Duplex.Formal.compileWiring_supported (fun _ => False)
    base offset (interface.initialState offset)
    ChallengeDerivation.layoutActions baseLeOffset initialSupport
  rw [challengeLayoutRecipeCount interface offset] at projected
  rw [ChallengeDerivation.finalState_eq_finalStateFast_pointwise]
  unfold ChallengeDerivation.finalStateFast
  rw [ChallengeDerivation.layoutWiring_eq_compileWiring]
  exact projected.2

private theorem roundGroup_sameShape
    {degreeBound : Nat}
    (left right : RoundTranscript.Interface degreeBound)
    (leftOffset rightOffset : Nat)
    (roundIndex : Fin productionShape.cubeVariables) :
    SameShape
      (RoundTranscript.roundActionsWithExpected left leftOffset roundIndex
        Quadratic.KExpr.zero)
      (RoundTranscript.roundActionsWithExpected right rightOffset roundIndex
        Quadratic.KExpr.zero) := by
  unfold RoundTranscript.roundActionsWithExpected
  dsimp only
  refine .absorb _ _ ?_ _ _ ?_
  · apply inputChunks_length_eq_of_length_eq
    simp [RoundTranscript.blockExpr, RoundTranscript.serializeRoundExpr,
      RoundTranscript.serializeKExprs, RoundTranscript.serializeKExpr]
    apply congrArg List.sum
    apply List.ext_get
    · simp
    · intro index leftBound rightBound
      simp
  · refine .absorb _ _ rfl _ _ ?_
    exact .squeeze _ _ [] [] .nil

theorem roundActions_sameShape
    {degreeBound : Nat}
    (left right : RoundTranscript.Interface degreeBound)
    (leftOffset rightOffset : Nat) :
    SameShape (RoundTranscript.layoutActions left leftOffset)
      (RoundTranscript.layoutActions right rightOffset) := by
  unfold RoundTranscript.layoutActions
  apply SameShape.flatMap
  intro roundIndex _
  exact roundGroup_sameShape left right leftOffset rightOffset roundIndex

private theorem roundLayoutRecipeCount
    (interface : RoundTranscript.Interface 9) (offset : Nat) :
    Duplex.Formal.recipeCount (RoundTranscript.layoutActions interface offset) =
      149184 := by
  calc
    Duplex.Formal.recipeCount (RoundTranscript.layoutActions interface offset) =
        (RoundTranscript.layoutProgram interface offset).recipes.length :=
      (Duplex.Formal.compile_recipes_length offset
        (interface.initialState offset)
        (RoundTranscript.layoutActions interface offset)).symm
    _ = (RoundTranscript.program interface offset).recipes.length :=
      congrArg List.length
        (RoundTranscript.program_shape_eq_layout interface offset).1.symm
    _ = 149184 := by
      rw [RoundTranscript.program_recipes_length]
      norm_num [productionShape, Phi81MatrixSource.phi81Shape,
        cubeVariables, RoundTranscript.perRoundRecipeCount]

theorem roundOutputs_localSupport
    (interface : RoundTranscript.Interface 9) (base offset : Nat)
    (baseLeOffset : base ≤ offset)
    (initialSupport : Duplex.Formal.StateSupported
      (interface.initialState offset) (Extend (fun _ => False) base offset)) :
    (∀ coordinate,
      Duplex.Formal.KSupported
        (RoundTranscript.challenge interface offset coordinate)
        (Extend (fun _ => False) base (offset + 149184))) ∧
      Duplex.Formal.StateSupported
        (RoundTranscript.finalState interface offset)
        (Extend (fun _ => False) base (offset + 149184)) := by
  have projected := Duplex.Formal.compileWiring_supported (fun _ => False)
    base offset (interface.initialState offset)
    (RoundTranscript.layoutActions interface offset) baseLeOffset initialSupport
  rw [roundLayoutRecipeCount interface offset] at projected
  constructor
  · intro coordinate
    rw [RoundTranscript.challenge_eq_challengeFast_pointwise]
    unfold RoundTranscript.challengeFast
    apply Duplex.Formal.sampleGetD_supported _ _ _ _ ?_ (by
      rw [RoundTranscript.layoutWiring_samples_length]
      exact coordinate.isLt)
    intro sample member
    rw [RoundTranscript.layoutWiring_eq_compileWiring] at member
    exact projected.1 sample member
  · rw [RoundTranscript.finalState_eq_finalStateFast_pointwise]
    unfold RoundTranscript.finalStateFast
    rw [RoundTranscript.layoutWiring_eq_compileWiring]
    exact projected.2

/-- All verifier-derived round samples and the final transcript state move by
one uniform parent offset delta. -/
theorem roundWiring_shift
    {degreeBound : Nat}
    (left right : RoundTranscript.Interface degreeBound)
    (leftOffset delta : Nat)
    (initialShift : right.initialState (leftOffset + delta) =
      state delta (left.initialState leftOffset)) :
    (RoundTranscript.layoutWiring right (leftOffset + delta)).samples =
        (RoundTranscript.layoutWiring left leftOffset).samples.map
          (quadratic delta) ∧
      (RoundTranscript.layoutWiring right (leftOffset + delta)).output =
        state delta (RoundTranscript.layoutWiring left leftOffset).output := by
  rw [RoundTranscript.layoutWiring_eq_compileWiring,
    RoundTranscript.layoutWiring_eq_compileWiring, initialShift]
  exact compileWiring_shift_of_sameShape delta leftOffset
    (left.initialState leftOffset)
    (roundActions_sameShape left right leftOffset (leftOffset + delta))

private theorem quadratic_zero (delta : Nat) :
    quadratic delta Quadratic.KExpr.zero = Quadratic.KExpr.zero := by
  rfl

theorem roundChallenge_shift
    {degreeBound : Nat}
    (left right : RoundTranscript.Interface degreeBound)
    (leftOffset delta : Nat)
    (initialShift : right.initialState (leftOffset + delta) =
      state delta (left.initialState leftOffset))
    (roundIndex : Fin productionShape.cubeVariables) :
    RoundTranscript.challenge right (leftOffset + delta) roundIndex =
      quadratic delta
        (RoundTranscript.challenge left leftOffset roundIndex) := by
  rw [RoundTranscript.challenge_eq_challengeFast_pointwise,
    RoundTranscript.challenge_eq_challengeFast_pointwise]
  unfold RoundTranscript.challengeFast
  rw [(roundWiring_shift left right leftOffset delta initialShift).1]
  rw [← quadratic_zero delta]
  exact List.getD_map (n := roundIndex.val)
    (RoundTranscript.layoutWiring left leftOffset).samples
    Quadratic.KExpr.zero (quadratic delta)

theorem roundFinalState_shift
    {degreeBound : Nat}
    (left right : RoundTranscript.Interface degreeBound)
    (leftOffset delta : Nat)
    (initialShift : right.initialState (leftOffset + delta) =
      state delta (left.initialState leftOffset)) :
    RoundTranscript.finalState right (leftOffset + delta) =
      state delta (RoundTranscript.finalState left leftOffset) := by
  rw [RoundTranscript.finalState_eq_finalStateFast_pointwise,
    RoundTranscript.finalState_eq_finalStateFast_pointwise]
  exact (roundWiring_shift left right leftOffset delta initialShift).2

theorem outputActions_sameShape
    (left right : OutputBinding.Interface)
    (leftOffset rightOffset : Nat) :
    SameShape (OutputBinding.actions left leftOffset)
      (OutputBinding.actions right rightOffset) := by
  unfold OutputBinding.actions StatementAbsorption.absorbBlock
  refine .absorb _ _ ?_ [] [] .nil
  apply inputChunks_length_eq_of_length_eq
  simp [StatementAbsorption.blockExpr, OutputBinding.outputWords_length]

theorem outputFinalState_shift
    (left right : OutputBinding.Interface)
    (leftOffset delta : Nat)
    (initialShift : right.initialState (leftOffset + delta) =
      state delta (left.initialState leftOffset)) :
    OutputBinding.finalState right (leftOffset + delta) =
      state delta (OutputBinding.finalState left leftOffset) := by
  let leftActions := OutputBinding.actions left leftOffset
  let rightActions := OutputBinding.actions right (leftOffset + delta)
  have shifted := compileWiring_shift_of_sameShape delta leftOffset
    (left.initialState leftOffset)
    (outputActions_sameShape left right leftOffset (leftOffset + delta))
  calc
    OutputBinding.finalState right (leftOffset + delta) =
        (Duplex.Formal.compileWiring (leftOffset + delta)
          (right.initialState (leftOffset + delta)) rightActions).output := by
      rw [OutputBinding.finalState_eq_compile]
      exact (Duplex.Formal.compileWiring_matches (leftOffset + delta)
        (right.initialState (leftOffset + delta)) rightActions).2.symm
    _ = state delta
        (Duplex.Formal.compileWiring leftOffset
          (left.initialState leftOffset) leftActions).output := by
      rw [initialShift]
      exact shifted.2
    _ = state delta (OutputBinding.finalState left leftOffset) := by
      rw [OutputBinding.finalState_eq_compile]
      exact congrArg (state delta)
        (Duplex.Formal.compileWiring_matches leftOffset
          (left.initialState leftOffset) leftActions).2

theorem outputFinalState_localSupport
    (interface : OutputBinding.Interface) (base offset : Nat)
    (baseLeOffset : base ≤ offset)
    (initialSupport : Duplex.Formal.StateSupported
      (interface.initialState offset) (Extend (fun _ => False) base offset)) :
    Duplex.Formal.StateSupported (OutputBinding.finalState interface offset)
      (Extend (fun _ => False) base (offset + 4076512)) := by
  have projected := Duplex.Formal.compileWiring_supported (fun _ => False)
    base offset (interface.initialState offset)
    (OutputBinding.actions interface offset) baseLeOffset initialSupport
  have count := OutputBinding.recipeCount_eq interface offset
  unfold OutputBinding.recipeCount at count
  rw [count] at projected
  rw [OutputBinding.finalState_eq_compile]
  rw [← (Duplex.Formal.compileWiring_matches offset
    (interface.initialState offset) (OutputBinding.actions interface offset)).2]
  exact projected.2

/-- Output binding is one nonempty absorb action, so its final symbolic state
is the output of the last permutation in the fixed 6,886-block schedule. -/
theorem outputFinalState_direct
    (interface : OutputBinding.Interface) (offset : Nat) :
    OutputBinding.finalState interface offset =
      Permutation.scheduleOutput (offset + 4075920) := by
  rw [OutputBinding.finalState_eq_compile]
  rw [← (Duplex.Formal.compileWiring_matches offset
    (interface.initialState offset) (OutputBinding.actions interface offset)).2]
  unfold OutputBinding.actions StatementAbsorption.absorbBlock
  simp only [Duplex.Formal.compileWiring]
  generalize blocksEq : Hash.inputChunks
    (StatementAbsorption.blockExpr
      (OutputBinding.outputWords interface offset)) = blocks
  cases blocks with
  | nil =>
      have lengthEq := congrArg List.length blocksEq
      simp [Hash.inputChunks, StatementAbsorption.blockExpr,
        OutputBinding.outputWords_length, Poseidon2.rate] at lengthEq
  | cons block rest =>
      rw [Duplex.Formal.compileAbsorbWiring_output_cons]
      apply congrArg Permutation.scheduleOutput
      have lengthEq := congrArg List.length blocksEq
      simp [Hash.inputChunks, StatementAbsorption.blockExpr,
        OutputBinding.outputWords_length, Poseidon2.rate] at lengthEq
      omega

theorem formalStatementFinalState_shift
    {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (left right : Formal.Interface logicalWidth degreeBound publicFits)
    (leftOffset delta : Nat) :
    Formal.statementFinalState right (leftOffset + delta) =
      state delta (Formal.statementFinalState left leftOffset) := by
  exact statementFinalState_shift
    (Formal.statementAbsorptionInterface (Formal.atOffset left leftOffset))
    (Formal.statementAbsorptionInterface
      (Formal.atOffset right (leftOffset + delta)))
    leftOffset delta

/-- Through statement absorption, challenge derivation, and all indexed round
absorptions, every parent-facing PiCCS transcript output moves by the exact
parent offset delta. -/
theorem formalRoundOutputs_shift
    {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (left right : Formal.Interface logicalWidth degreeBound publicFits)
    (leftOffset delta : Nat) :
    (∀ coordinate,
      Formal.roundPoint (Formal.atOffset right (leftOffset + delta))
          (leftOffset + delta) coordinate =
        quadratic delta
          (Formal.roundPoint (Formal.atOffset left leftOffset)
            leftOffset coordinate)) ∧
      Formal.roundTranscriptFinalState
          (Formal.atOffset right (leftOffset + delta))
          (leftOffset + delta) =
        state delta
          (Formal.roundTranscriptFinalState (Formal.atOffset left leftOffset)
            leftOffset) := by
  let leftShared := Formal.atOffset left leftOffset
  let rightShared := Formal.atOffset right (leftOffset + delta)
  have statementShift :
      Formal.statementFinalState rightShared (leftOffset + delta) =
        state delta
          (Formal.statementFinalState leftShared leftOffset) :=
    formalStatementFinalState_shift leftShared rightShared leftOffset delta
  have challengeStartShift :
      Formal.challengeStart rightShared =
        Formal.challengeStart leftShared + delta := by
    simp [leftShared, rightShared, Formal.challengeStart, Formal.atOffset]
    omega
  have challengeInitialShift :
      (Formal.challengeInterface rightShared
          (leftOffset + delta)).initialState
          (Formal.challengeStart leftShared + delta) =
        state delta
          ((Formal.challengeInterface leftShared leftOffset).initialState
            (Formal.challengeStart leftShared)) := by
    exact statementShift
  have challengeShiftRaw := challengeFinalState_shift
    (Formal.challengeInterface leftShared leftOffset)
    (Formal.challengeInterface rightShared (leftOffset + delta))
    (Formal.challengeStart leftShared) delta challengeInitialShift
  have roundStartShift :
      Formal.roundTranscriptStart rightShared =
        Formal.roundTranscriptStart leftShared + delta := by
    simp [leftShared, rightShared, Formal.roundTranscriptStart, Formal.atOffset]
    omega
  have roundInitialShift :
      (Formal.roundTranscriptInterface rightShared).initialState
          (Formal.roundTranscriptStart leftShared + delta) =
        state delta
          ((Formal.roundTranscriptInterface leftShared).initialState
            (Formal.roundTranscriptStart leftShared)) := by
    dsimp only [Formal.roundTranscriptInterface, Formal.challengeFinalState]
    rw [challengeStartShift]
    exact challengeShiftRaw
  refine ⟨?_, ?_⟩
  · intro coordinate
    unfold Formal.roundPoint
    rw [roundStartShift]
    exact roundChallenge_shift (Formal.roundTranscriptInterface leftShared)
      (Formal.roundTranscriptInterface rightShared)
      (Formal.roundTranscriptStart leftShared) delta roundInitialShift coordinate
  · unfold Formal.roundTranscriptFinalState
    rw [roundStartShift]
    exact roundFinalState_shift (Formal.roundTranscriptInterface leftShared)
      (Formal.roundTranscriptInterface rightShared)
      (Formal.roundTranscriptStart leftShared) delta roundInitialShift

theorem outputBindingOffset_shift
    {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (left right : Formal.Interface logicalWidth degreeBound publicFits)
    (leftOffset delta : Nat) :
    Formal.outputBindingOffset relation right (leftOffset + delta) =
      Formal.outputBindingOffset relation left leftOffset + delta := by
  rw [Formal.outputBindingOffset_eq_outputBindingRowOffset,
    Formal.outputBindingOffset_eq_outputBindingRowOffset]
  unfold Formal.outputBindingRowOffset Formal.finalIdentityRowOffset
    Formal.normRowOffset Formal.ccsRowOffset Formal.evalARowOffset
    Formal.evalKRowOffset Formal.sumcheckRowOffset
    Formal.initialClaimRowOffset
  omega

/-- The complete post-PiCCS transcript state relocates lane by lane. The
proof uses only the direct last-permutation endpoint formula. -/
theorem formalOutputFinalState_shift
    {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (left right : Formal.Interface logicalWidth degreeBound publicFits)
    (leftOffset delta : Nat) :
    Formal.outputBindingFinalState relation right (leftOffset + delta) =
      state delta
        (Formal.outputBindingFinalState relation left leftOffset) := by
  funext lane
  unfold Formal.outputBindingFinalState
  rw [congrFun (outputFinalState_direct
      (Formal.outputBindingInterface
        (Formal.atOffset right (leftOffset + delta)))
      (Formal.outputBindingOffset relation right (leftOffset + delta))) lane]
  simp only [state]
  rw [congrFun (outputFinalState_direct
      (Formal.outputBindingInterface (Formal.atOffset left leftOffset))
      (Formal.outputBindingOffset relation left leftOffset)) lane]
  rw [outputBindingOffset_shift relation left right leftOffset delta]
  simp only [expression, Permutation.scheduleOutput,
    Permutation.freshState]
  congr 1
  omega

def formalRoundEnd
    {logicalWidth : Nat}
    {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Formal.Interface logicalWidth 9 publicFits)
    (offset : Nat) : Nat :=
  Formal.roundTranscriptStart (Formal.atOffset interface offset) + 149184

theorem formalRoundOutputs_exactLocalSupport
    {logicalWidth : Nat}
    {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Formal.Interface logicalWidth 9 publicFits)
    (offset : Nat) :
    (∀ coordinate,
      Duplex.Formal.KSupported
        (Formal.roundPoint (Formal.atOffset interface offset) offset coordinate)
        (Extend (fun _ => False) offset (formalRoundEnd interface offset))) ∧
      Duplex.Formal.StateSupported
        (Formal.roundTranscriptFinalState
          (Formal.atOffset interface offset) offset)
        (Extend (fun _ => False) offset (formalRoundEnd interface offset)) := by
  let shared := Formal.atOffset interface offset
  have statementSupport := statementFinalState_localSupport
    (Formal.statementAbsorptionInterface (Formal.atOffset shared offset)) offset
  have challengeStartEq : Formal.challengeStart shared = offset + 224368 := by
    simp [shared, Formal.challengeStart, Formal.atOffset]
  have challengeInitialSupport : Duplex.Formal.StateSupported
      ((Formal.challengeInterface shared offset).initialState
        (Formal.challengeStart shared))
      (Extend (fun _ => False) offset (Formal.challengeStart shared)) := by
    dsimp only [Formal.challengeInterface]
    rw [challengeStartEq]
    exact statementSupport
  have challengeSupport := challengeFinalState_localSupport
    (Formal.challengeInterface shared offset) offset
    (Formal.challengeStart shared) (by omega) challengeInitialSupport
  have roundStartEq : Formal.roundTranscriptStart shared =
      Formal.challengeStart shared + 51504 := by
    simp [shared, Formal.roundTranscriptStart, Formal.challengeStart,
      Formal.atOffset]
  have roundInitialSupport : Duplex.Formal.StateSupported
      ((Formal.roundTranscriptInterface shared).initialState
        (Formal.roundTranscriptStart shared))
      (Extend (fun _ => False) offset (Formal.roundTranscriptStart shared)) := by
    dsimp only [Formal.roundTranscriptInterface, Formal.challengeFinalState]
    rw [roundStartEq]
    exact challengeSupport
  simpa [formalRoundEnd, shared] using roundOutputs_localSupport
    (Formal.roundTranscriptInterface shared) offset
    (Formal.roundTranscriptStart shared) (by omega) roundInitialSupport

theorem formalRoundOutputs_localSupport
    {logicalWidth : Nat}
    {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Formal.Interface logicalWidth 9 publicFits)
    (offset : Nat) :
    (∀ coordinate,
      Duplex.Formal.KSupported
        (Formal.roundPoint (Formal.atOffset interface offset) offset coordinate)
        (Extend (fun _ => False) offset (offset + 4581414))) ∧
      Duplex.Formal.StateSupported
        (Formal.roundTranscriptFinalState
          (Formal.atOffset interface offset) offset)
        (Extend (fun _ => False) offset (offset + 4581414)) := by
  let shared := Formal.atOffset interface offset
  have roundSupport := formalRoundOutputs_exactLocalSupport interface offset
  have finishLe : formalRoundEnd interface offset ≤
      offset + 4581414 := by
    simp [formalRoundEnd, Formal.roundTranscriptStart, Formal.atOffset]
  constructor
  · intro coordinate
    have support := roundSupport.1 coordinate
    apply Duplex.Formal.KSupported.mono support
    intro index localSupport
    exact SupportRange.mono_finish localSupport finishLe
  · change Duplex.Formal.StateSupported
      (RoundTranscript.finalState (Formal.roundTranscriptInterface shared)
        (Formal.roundTranscriptStart shared))
      (Extend (fun _ => False) offset (offset + 4581414))
    intro lane
    have support := roundSupport.2 lane
    exact Expr.VarsSatisfy.mono _ support
      (fun index localSupport =>
        SupportRange.mono_finish localSupport finishLe)

private theorem cubePoint_ext
    {Field : Type} {variableCount : Nat}
    (left right : CubePoint Field variableCount)
    (coordinates : left.coordinates = right.coordinates) : left = right := by
  cases left
  cases right
  simp_all

theorem evalRoundPoint_eq_of_shift_agreement
    {logicalWidth : Nat}
    {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (left right : Formal.Interface logicalWidth 9 publicFits)
    (leftOffset delta : Nat) (leftEnv rightEnv : Env)
    (agrees : ∀ index,
      Extend (fun _ => False) leftOffset (leftOffset + 4581414) index →
        rightEnv (index + delta) = leftEnv index) :
    RoundTranscript.evalRoundPoint
        (Formal.roundTranscriptInterface
          (Formal.atOffset right (leftOffset + delta)))
        (Formal.roundTranscriptOffset right (leftOffset + delta)) rightEnv =
      RoundTranscript.evalRoundPoint
        (Formal.roundTranscriptInterface (Formal.atOffset left leftOffset))
        (Formal.roundTranscriptOffset left leftOffset) leftEnv := by
  apply cubePoint_ext
  rw [← Formal.roundTranscriptStart_atOffset right (leftOffset + delta),
    ← Formal.roundTranscriptStart_atOffset left leftOffset]
  change (canonicalFinIndices productionShape.cubeVariables).map
      (fun coordinate =>
        (Formal.roundPoint (Formal.atOffset right (leftOffset + delta))
          (leftOffset + delta) coordinate).eval rightEnv) =
    (canonicalFinIndices productionShape.cubeVariables).map
      (fun coordinate =>
        (Formal.roundPoint (Formal.atOffset left leftOffset)
          leftOffset coordinate).eval leftEnv)
  apply List.map_congr_left
  intro coordinate _
  rw [(formalRoundOutputs_shift left right leftOffset delta).1 coordinate]
  exact quadratic_eval_eq_of_shift_agreement delta
    (Formal.roundPoint (Formal.atOffset left leftOffset)
      leftOffset coordinate)
    (Extend (fun _ => False) leftOffset (leftOffset + 4581414))
    leftEnv rightEnv
    ((formalRoundOutputs_localSupport left leftOffset).1 coordinate) agrees

end NightstreamFPrime.Layout.Stage1.PiCCSTranscriptRelocation
