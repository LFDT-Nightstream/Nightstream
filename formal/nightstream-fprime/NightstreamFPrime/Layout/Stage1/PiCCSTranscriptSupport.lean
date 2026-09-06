import NightstreamFPrime.Gadgets.Polynomial.HornerSupport
import NightstreamFPrime.Gadgets.Poseidon2.Duplex.WiringSupport
import NightstreamFPrime.Layout.Stage1.PiCCSOrdinarySourceSupportData

/-!
Owns retained-source support for the verifier-derived PiCCS transcript values.

The proof uses the recipe-free Poseidon2 wiring projection. Existing shape
theorems transfer the exact semantic-program footprints to that projection.
It changes no transcript action, challenge order, circuit, or row.
-/

namespace NightstreamFPrime.Layout.Stage1.PiCCSOrdinarySourceSupport

open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Circuit.SupportRange
open NightstreamFPrime.Gadgets.Polynomial
open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.PiCCS.v1_1
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

private theorem outputPrefix_source (count : Nat)
    (countLe : count ≤ transcriptInvocationCount) :
    ∀ index,
      Duplex.Formal.OutputPrefix External PiCCSInputs.phaseOffset count index →
        Source index := by
  intro index support
  rcases support with external | ⟨invocation, lane, source⟩
  · exact external_source index external
  · apply transcript_output_source index
    exact ⟨⟨invocation.val,
      Nat.lt_of_lt_of_le invocation.isLt countLe⟩, lane, source⟩

private theorem zeroState_outputPrefix_supported :
    Duplex.Formal.StateSupported Hash.zeroE
      (Duplex.Formal.OutputPrefix External PiCCSInputs.phaseOffset 0) := by
  intro lane
  trivial

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

private theorem roundLayoutRecipeCount {degreeBound : Nat}
    (interface : RoundTranscript.Interface degreeBound) (offset : Nat) :
    Duplex.Formal.recipeCount (RoundTranscript.layoutActions interface offset) =
      productionShape.cubeVariables *
        RoundTranscript.perRoundRecipeCount degreeBound := by
  calc
    Duplex.Formal.recipeCount (RoundTranscript.layoutActions interface offset) =
        (RoundTranscript.layoutProgram interface offset).recipes.length :=
      (Duplex.Formal.compile_recipes_length offset
        (interface.initialState offset)
        (RoundTranscript.layoutActions interface offset)).symm
    _ = (RoundTranscript.program interface offset).recipes.length :=
      congrArg List.length
        (RoundTranscript.program_shape_eq_layout interface offset).1.symm
    _ = productionShape.cubeVariables *
        RoundTranscript.perRoundRecipeCount degreeBound :=
      RoundTranscript.program_recipes_length interface offset

private theorem statementPermutationCount
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : StatementAbsorption.Interface logicalWidth publicFits)
    (offset : Nat) :
    Duplex.Formal.permutationCount
      (StatementAbsorption.actions interface offset) = 379 := by
  have recipes := Duplex.Formal.recipeCount_eq_permutationCount_mul
    (StatementAbsorption.actions interface offset)
  change StatementAbsorption.recipeCount interface offset =
    Duplex.Formal.permutationCount
      (StatementAbsorption.actions interface offset) * 592 at recipes
  rw [StatementAbsorption.recipeCount_eq] at recipes
  omega

private theorem challengePermutationCount
    (interface : ChallengeDerivation.Interface) (offset : Nat) :
    Duplex.Formal.permutationCount ChallengeDerivation.layoutActions = 87 := by
  have recipes := Duplex.Formal.recipeCount_eq_permutationCount_mul
    ChallengeDerivation.layoutActions
  rw [challengeLayoutRecipeCount interface offset] at recipes
  omega

private theorem roundPermutationCount
    (interface : RoundTranscript.Interface 9) (offset : Nat) :
    Duplex.Formal.permutationCount
      (RoundTranscript.layoutActions interface offset) = 252 := by
  have recipes := Duplex.Formal.recipeCount_eq_permutationCount_mul
    (RoundTranscript.layoutActions interface offset)
  rw [roundLayoutRecipeCount interface offset] at recipes
  norm_num [productionShape, Phi81MatrixSource.phi81Shape,
    cubeVariables, RoundTranscript.perRoundRecipeCount] at recipes
  omega

private theorem statementFinalState_outputPrefix_supported
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    let shared := Formal.atOffset
      (PiCCSInputs.interface logicalWidth publicFits) PiCCSInputs.phaseOffset
    Duplex.Formal.StateSupported
      (Formal.statementFinalState shared PiCCSInputs.phaseOffset)
      (Duplex.Formal.OutputPrefix External PiCCSInputs.phaseOffset 379) := by
  dsimp only
  let shared := Formal.atOffset
    (PiCCSInputs.interface logicalWidth publicFits) PiCCSInputs.phaseOffset
  let interface := Formal.statementAbsorptionInterface shared
  let actions := StatementAbsorption.actions interface PiCCSInputs.phaseOffset
  have projected := Duplex.Formal.compileWiring_outputPrefix_supported External
    PiCCSInputs.phaseOffset 0 Hash.zeroE actions
    zeroState_outputPrefix_supported
  have count := statementPermutationCount interface PiCCSInputs.phaseOffset
  rw [count] at projected
  have outputSupport := projected.2
  simp only [Nat.zero_mul, Nat.add_zero] at outputSupport
  rw [(Duplex.Formal.compileWiring_matches PiCCSInputs.phaseOffset Hash.zeroE
    actions).2] at outputSupport
  exact outputSupport

private theorem challengeWiring_outputPrefix_supported
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    let shared := Formal.atOffset
      (PiCCSInputs.interface logicalWidth publicFits) PiCCSInputs.phaseOffset
    let interface := Formal.challengeInterface shared PiCCSInputs.phaseOffset
    (∀ sample ∈
      (ChallengeDerivation.layoutWiring interface
          (Formal.challengeStart shared)).samples,
      Duplex.Formal.KSupported sample
        (Duplex.Formal.OutputPrefix External PiCCSInputs.phaseOffset 466)) ∧
      Duplex.Formal.StateSupported
        (ChallengeDerivation.finalState interface
          (Formal.challengeStart shared))
        (Duplex.Formal.OutputPrefix External PiCCSInputs.phaseOffset 466) := by
  dsimp only
  let shared := Formal.atOffset
    (PiCCSInputs.interface logicalWidth publicFits) PiCCSInputs.phaseOffset
  let interface := Formal.challengeInterface shared PiCCSInputs.phaseOffset
  let start := Formal.challengeStart shared
  have incoming : Duplex.Formal.StateSupported (interface.initialState start)
      (Duplex.Formal.OutputPrefix External PiCCSInputs.phaseOffset 379) := by
    simpa [interface, Formal.challengeInterface_initialState] using
      statementFinalState_outputPrefix_supported logicalWidth publicFits
  have startEq : start = PiCCSInputs.phaseOffset + 379 * 592 := by
    dsimp [start, shared]
    rw [Formal.challengeStart_atOffset, Formal.challengeOffset_eq]
  have projected := Duplex.Formal.compileWiring_outputPrefix_supported External
    PiCCSInputs.phaseOffset 379 (interface.initialState start)
    ChallengeDerivation.layoutActions incoming
  rw [← startEq] at projected
  rw [← ChallengeDerivation.layoutWiring_eq_compileWiring] at projected
  have count := challengePermutationCount interface start
  rw [count] at projected
  change
    (∀ value ∈ (ChallengeDerivation.layoutWiring interface start).samples,
      Duplex.Formal.KSupported value
        (Duplex.Formal.OutputPrefix External PiCCSInputs.phaseOffset 466)) ∧
      Duplex.Formal.StateSupported
        (ChallengeDerivation.finalState interface start)
        (Duplex.Formal.OutputPrefix External PiCCSInputs.phaseOffset 466)
  refine ⟨projected.1, ?_⟩
  rw [ChallengeDerivation.finalState_eq_finalStateFast_pointwise]
  exact projected.2

private theorem roundWiring_outputPrefix_supported
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    let shared := Formal.atOffset
      (PiCCSInputs.interface logicalWidth publicFits) PiCCSInputs.phaseOffset
    let interface := Formal.roundTranscriptInterface shared
    ∀ sample ∈
      (RoundTranscript.layoutWiring interface
        (Formal.roundTranscriptStart shared)).samples,
      Duplex.Formal.KSupported sample
        (Duplex.Formal.OutputPrefix External PiCCSInputs.phaseOffset 718) := by
  dsimp only
  let shared := Formal.atOffset
    (PiCCSInputs.interface logicalWidth publicFits) PiCCSInputs.phaseOffset
  let challengeInterface :=
    Formal.challengeInterface shared PiCCSInputs.phaseOffset
  let interface := Formal.roundTranscriptInterface shared
  let start := Formal.roundTranscriptStart shared
  have challengeSupport :=
    challengeWiring_outputPrefix_supported logicalWidth publicFits
  have incoming : Duplex.Formal.StateSupported (interface.initialState start)
      (Duplex.Formal.OutputPrefix External PiCCSInputs.phaseOffset 466) := by
    simpa [interface, Formal.roundTranscriptInterface,
      Formal.challengeFinalState, challengeInterface, shared, Formal.atOffset]
      using challengeSupport.2
  have startEq : start = PiCCSInputs.phaseOffset + 466 * 592 := by
    dsimp [start, shared]
    rw [Formal.roundTranscriptStart_atOffset,
      Formal.roundTranscriptOffset_eq, Formal.challengeOffset_eq]
  have projected := Duplex.Formal.compileWiring_outputPrefix_supported External
    PiCCSInputs.phaseOffset 466 (interface.initialState start)
    (RoundTranscript.layoutActions interface start) incoming
  rw [← startEq] at projected
  rw [← RoundTranscript.layoutWiring_eq_compileWiring] at projected
  have count := roundPermutationCount interface start
  rw [count] at projected
  change ∀ value ∈ (RoundTranscript.layoutWiring interface start).samples,
    Duplex.Formal.KSupported value
      (Duplex.Formal.OutputPrefix External PiCCSInputs.phaseOffset 718)
  exact projected.1

theorem challengeWiring_supported
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    let shared := Formal.atOffset
      (PiCCSInputs.interface logicalWidth publicFits) PiCCSInputs.phaseOffset
    let interface := Formal.challengeInterface shared PiCCSInputs.phaseOffset
    ∀ sample ∈
      (ChallengeDerivation.layoutWiring interface
        (Formal.challengeStart shared)).samples,
      Duplex.Formal.KSupported sample Source := by
  dsimp only
  have precise := challengeWiring_outputPrefix_supported logicalWidth publicFits
  intro sample member
  exact (precise.1 sample member).mono
    (outputPrefix_source 466 (by
      rw [transcriptInvocationCount_eq]
      omega))

private theorem roundWiring_supported
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    let shared := Formal.atOffset
      (PiCCSInputs.interface logicalWidth publicFits) PiCCSInputs.phaseOffset
    let interface := Formal.roundTranscriptInterface shared
    ∀ sample ∈
      (RoundTranscript.layoutWiring interface
        (Formal.roundTranscriptStart shared)).samples,
      Duplex.Formal.KSupported sample Source := by
  dsimp only
  have precise := roundWiring_outputPrefix_supported logicalWidth publicFits
  intro sample member
  exact (precise sample member).mono
    (outputPrefix_source 718 (by
      rw [transcriptInvocationCount_eq]))

/-- Exact retained support for every verifier-derived value consumed by the
seven ordinary PiCCS children after statement binding. -/
structure TranscriptValuesSupported
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) : Prop where
  alpha : ∀ coordinate,
    Horner.KSupported
      (Formal.challengeAlpha
        (Formal.atOffset (PiCCSInputs.interface logicalWidth publicFits)
          PiCCSInputs.phaseOffset)
        PiCCSInputs.phaseOffset coordinate) Source
  gamma : Horner.KSupported
    (Formal.challengeGamma
      (Formal.atOffset (PiCCSInputs.interface logicalWidth publicFits)
        PiCCSInputs.phaseOffset)
      PiCCSInputs.phaseOffset) Source
  roundPoint : ∀ coordinate,
    Horner.KSupported
      (Formal.roundPoint
        (Formal.atOffset (PiCCSInputs.interface logicalWidth publicFits)
          PiCCSInputs.phaseOffset)
        PiCCSInputs.phaseOffset coordinate) Source

theorem transcriptValuesSupported
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    TranscriptValuesSupported logicalWidth publicFits := by
  let shared := Formal.atOffset
    (PiCCSInputs.interface logicalWidth publicFits) PiCCSInputs.phaseOffset
  let challengeInterface :=
    Formal.challengeInterface shared PiCCSInputs.phaseOffset
  let roundInterface := Formal.roundTranscriptInterface shared
  have challengeSupport := challengeWiring_supported logicalWidth publicFits
  have roundSupport := roundWiring_supported logicalWidth publicFits
  refine {
    alpha := ?_
    gamma := ?_
    roundPoint := ?_ }
  · intro coordinate
    rw [Formal.challengeAlpha,
      ChallengeDerivation.alpha_eq_alphaFast_pointwise]
    unfold ChallengeDerivation.alphaFast
    apply Duplex.Formal.sampleGetD_supported
    · intro sample member
      apply challengeSupport sample
      exact List.mem_of_mem_take member
    · simp [ChallengeDerivation.layoutWiring_samples_length,
        productionShape, Phi81MatrixSource.phi81Shape, cubeVariables]
  · rw [Formal.challengeGamma,
      ChallengeDerivation.gamma_eq_gammaFast_pointwise]
    unfold ChallengeDerivation.gammaFast
    apply Duplex.Formal.sampleGetD_supported
    · exact challengeSupport
    · simp [ChallengeDerivation.layoutWiring_samples_length,
        productionShape, Phi81MatrixSource.phi81Shape, cubeVariables]
  · intro coordinate
    rw [Formal.roundPoint,
      RoundTranscript.challenge_eq_challengeFast_pointwise]
    unfold RoundTranscript.challengeFast
    apply Duplex.Formal.sampleGetD_supported
    · exact roundSupport
    · rw [RoundTranscript.layoutWiring_samples_length]
      change coordinate.val < productionShape.cubeVariables
      exact coordinate.isLt

/-- At any parent-selected PiCCS allocation, the verifier-derived round point
and final round-transcript state use only caller sources or transcript
permutation outputs owned by that same phase interval. -/
theorem roundOutputs_outputPrefix_supported
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Formal.Interface logicalWidth 9 publicFits)
    (offset : Nat) :
    (∀ coordinate, Horner.KSupported
        (Formal.roundPoint (Formal.atOffset interface offset) offset coordinate)
        (Duplex.Formal.OutputPrefix External offset 718)) ∧
      Duplex.Formal.StateSupported
        (Formal.roundTranscriptFinalState
          (Formal.atOffset interface offset) offset)
        (Duplex.Formal.OutputPrefix External offset 718) := by
  let shared := Formal.atOffset interface offset
  let statementInterface := Formal.statementAbsorptionInterface shared
  let statementActions := StatementAbsorption.actions statementInterface offset
  have zeroSupport : Duplex.Formal.StateSupported Hash.zeroE
      (Duplex.Formal.OutputPrefix External offset 0) := by
    intro lane
    trivial
  have statementProjected :=
    Duplex.Formal.compileWiring_outputPrefix_supported External offset 0
      Hash.zeroE statementActions zeroSupport
  have statementCount := statementPermutationCount statementInterface offset
  rw [statementCount] at statementProjected
  simp only [Nat.zero_mul, Nat.add_zero] at statementProjected
  rw [(Duplex.Formal.compileWiring_matches offset Hash.zeroE
    statementActions).2] at statementProjected
  have statementSupport : Duplex.Formal.StateSupported
      (Formal.statementFinalState shared offset)
      (Duplex.Formal.OutputPrefix External offset 379) := by
    exact statementProjected.2

  let challengeInterface := Formal.challengeInterface shared offset
  let challengeStart := Formal.challengeStart shared
  have challengeIncoming : Duplex.Formal.StateSupported
      (challengeInterface.initialState challengeStart)
      (Duplex.Formal.OutputPrefix External offset 379) := by
    simpa [challengeInterface, Formal.challengeInterface_initialState] using
      statementSupport
  have challengeStartEq : challengeStart = offset + 379 * 592 := by
    dsimp [challengeStart, shared]
    rw [Formal.challengeStart_atOffset, Formal.challengeOffset_eq]
  have challengeProjected :=
    Duplex.Formal.compileWiring_outputPrefix_supported External offset 379
      (challengeInterface.initialState challengeStart)
      ChallengeDerivation.layoutActions challengeIncoming
  rw [← challengeStartEq] at challengeProjected
  rw [← ChallengeDerivation.layoutWiring_eq_compileWiring] at challengeProjected
  have challengeCount := challengePermutationCount challengeInterface
    challengeStart
  rw [challengeCount] at challengeProjected
  have challengeFinalSupport : Duplex.Formal.StateSupported
      (ChallengeDerivation.finalState challengeInterface challengeStart)
      (Duplex.Formal.OutputPrefix External offset 466) := by
    rw [ChallengeDerivation.finalState_eq_finalStateFast_pointwise]
    exact challengeProjected.2

  let roundInterface := Formal.roundTranscriptInterface shared
  let roundStart := Formal.roundTranscriptStart shared
  have roundIncoming : Duplex.Formal.StateSupported
      (roundInterface.initialState roundStart)
      (Duplex.Formal.OutputPrefix External offset 466) := by
    simpa [roundInterface, Formal.roundTranscriptInterface,
      Formal.challengeFinalState, challengeInterface, shared, Formal.atOffset]
      using challengeFinalSupport
  have roundStartEq : roundStart = offset + 466 * 592 := by
    dsimp [roundStart, shared]
    rw [Formal.roundTranscriptStart_atOffset,
      Formal.roundTranscriptOffset_eq, Formal.challengeOffset_eq]
  have roundProjected :=
    Duplex.Formal.compileWiring_outputPrefix_supported External offset 466
      (roundInterface.initialState roundStart)
      (RoundTranscript.layoutActions roundInterface roundStart) roundIncoming
  rw [← roundStartEq] at roundProjected
  rw [← RoundTranscript.layoutWiring_eq_compileWiring] at roundProjected
  have roundCount := roundPermutationCount roundInterface roundStart
  rw [roundCount] at roundProjected
  constructor
  · intro coordinate
    rw [Formal.roundPoint,
      RoundTranscript.challenge_eq_challengeFast_pointwise]
    unfold RoundTranscript.challengeFast
    apply Duplex.Formal.sampleGetD_supported
    · exact roundProjected.1
    · rw [RoundTranscript.layoutWiring_samples_length]
      change coordinate.val < productionShape.cubeVariables
      exact coordinate.isLt
  · change Duplex.Formal.StateSupported
      (RoundTranscript.finalState roundInterface roundStart)
      (Duplex.Formal.OutputPrefix External offset 718)
    rw [RoundTranscript.finalState_eq_finalStateFast_pointwise]
    exact roundProjected.2

/-- Offset-generic support for one verifier-derived round-point coordinate. -/
theorem roundPoint_outputPrefix_supported
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Formal.Interface logicalWidth 9 publicFits)
    (offset : Nat) (coordinate : Fin productionShape.cubeVariables) :
    Horner.KSupported
      (Formal.roundPoint (Formal.atOffset interface offset) offset coordinate)
      (Duplex.Formal.OutputPrefix External offset 718) :=
  (roundOutputs_outputPrefix_supported interface offset).1 coordinate

private theorem cubePoint_ext
    {Field : Type} {variableCount : Nat}
    (left right : CubePoint Field variableCount)
    (coordinates : left.coordinates = right.coordinates) : left = right := by
  cases left
  cases right
  simp_all

/-- Evaluation of the verifier-derived round point is stable when two
environments agree on its exact caller and transcript-output support. -/
theorem evalRoundPoint_eq_of_agree_outputPrefix
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Formal.Interface logicalWidth 9 publicFits)
    (offset : Nat) (left right : Env)
    (agrees : ∀ index,
      Duplex.Formal.OutputPrefix External offset 718 index →
        left index = right index) :
    RoundTranscript.evalRoundPoint
        (Formal.roundTranscriptInterface (Formal.atOffset interface offset))
        (Formal.roundTranscriptOffset interface offset) left =
      RoundTranscript.evalRoundPoint
        (Formal.roundTranscriptInterface (Formal.atOffset interface offset))
        (Formal.roundTranscriptOffset interface offset) right := by
  apply cubePoint_ext
  change (canonicalFinIndices productionShape.cubeVariables).map
      (fun coordinate =>
        (RoundTranscript.challenge
          (Formal.roundTranscriptInterface (Formal.atOffset interface offset))
          (Formal.roundTranscriptOffset interface offset) coordinate).eval left) =
    (canonicalFinIndices productionShape.cubeVariables).map
      (fun coordinate =>
        (RoundTranscript.challenge
          (Formal.roundTranscriptInterface (Formal.atOffset interface offset))
          (Formal.roundTranscriptOffset interface offset) coordinate).eval right)
  apply List.map_congr_left
  intro coordinate _member
  rw [← Formal.roundTranscriptStart_atOffset interface offset]
  change (Formal.roundPoint (Formal.atOffset interface offset) offset
      coordinate).eval left =
    (Formal.roundPoint (Formal.atOffset interface offset) offset
      coordinate).eval right
  have support := roundPoint_outputPrefix_supported interface offset coordinate
  exact congrArg₂ K.mk
    (Expr.eval_eq_of_agree_satisfy _ _ left right support.1 agrees)
    (Expr.eval_eq_of_agree_satisfy _ _ left right support.2 agrees)

/-- Offset-generic support for the state handed to output binding. -/
theorem roundFinalState_outputPrefix_supported
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Formal.Interface logicalWidth 9 publicFits)
    (offset : Nat) :
    Duplex.Formal.StateSupported
      (Formal.roundTranscriptFinalState
        (Formal.atOffset interface offset) offset)
      (Duplex.Formal.OutputPrefix External offset 718) :=
  (roundOutputs_outputPrefix_supported interface offset).2

end NightstreamFPrime.Layout.Stage1.PiCCSOrdinarySourceSupport
