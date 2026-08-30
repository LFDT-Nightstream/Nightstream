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

private theorem extend_source (finish : Nat)
    (finishLe : finish ≤ PiCCSStarts.outputBindingWitnessStart) :
    ∀ index, Extend Source PiCCSInputs.phaseOffset finish index → Source index := by
  intro index support
  rcases support with support | ⟨lower, upper⟩
  · exact support
  · exact local_source index lower (Nat.lt_of_lt_of_le upper finishLe)

private theorem zeroState_supported :
    Duplex.Formal.StateSupported Hash.zeroE Source := by
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

private theorem statementFinalState_supported
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    let shared := Formal.atOffset
      (PiCCSInputs.interface logicalWidth publicFits) PiCCSInputs.phaseOffset
    Duplex.Formal.StateSupported
      (Formal.statementFinalState shared PiCCSInputs.phaseOffset) Source := by
  dsimp only
  let shared := Formal.atOffset
    (PiCCSInputs.interface logicalWidth publicFits) PiCCSInputs.phaseOffset
  let interface := Formal.statementAbsorptionInterface shared
  let actions := StatementAbsorption.actions interface PiCCSInputs.phaseOffset
  have initial : Duplex.Formal.StateSupported Hash.zeroE
      (Extend Source PiCCSInputs.phaseOffset PiCCSInputs.phaseOffset) :=
    zeroState_supported.mono (fun _ support => SupportRange.base support)
  have projected := Duplex.Formal.compileWiring_supported Source
    PiCCSInputs.phaseOffset PiCCSInputs.phaseOffset Hash.zeroE actions
    (by omega) initial
  have outputSupport := projected.2
  rw [(Duplex.Formal.compileWiring_matches PiCCSInputs.phaseOffset Hash.zeroE
    actions).2] at outputSupport
  have finishLe :
      PiCCSInputs.phaseOffset + Duplex.Formal.recipeCount actions ≤
        PiCCSStarts.outputBindingWitnessStart := by
    change PiCCSInputs.phaseOffset +
        StatementAbsorption.recipeCount interface PiCCSInputs.phaseOffset ≤ _
    rw [StatementAbsorption.recipeCount_eq,
      PiCCSInputs.phaseOffset_eq, PiCCSStarts.outputBindingWitnessStart_eq]
    norm_num
  apply Duplex.Formal.StateSupported.mono outputSupport
  exact extend_source _ finishLe

private theorem challengeWiring_supported
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    let shared := Formal.atOffset
      (PiCCSInputs.interface logicalWidth publicFits) PiCCSInputs.phaseOffset
    let interface := Formal.challengeInterface shared PiCCSInputs.phaseOffset
    (∀ sample ∈
        (ChallengeDerivation.layoutWiring interface
          (Formal.challengeStart shared)).samples,
      Duplex.Formal.KSupported sample Source) ∧
      Duplex.Formal.StateSupported
        (ChallengeDerivation.finalState interface
          (Formal.challengeStart shared)) Source := by
  dsimp only
  let shared := Formal.atOffset
    (PiCCSInputs.interface logicalWidth publicFits) PiCCSInputs.phaseOffset
  let interface := Formal.challengeInterface shared PiCCSInputs.phaseOffset
  let start := Formal.challengeStart shared
  have incoming : Duplex.Formal.StateSupported
      (interface.initialState start) Source := by
    simpa [interface, Formal.challengeInterface_initialState] using
      statementFinalState_supported logicalWidth publicFits
  have initial : Duplex.Formal.StateSupported (interface.initialState start)
      (Extend Source PiCCSInputs.phaseOffset start) := by
    apply incoming.mono
    intro index support
    exact SupportRange.base support
  have projected := Duplex.Formal.compileWiring_supported Source
    PiCCSInputs.phaseOffset start (interface.initialState start)
    ChallengeDerivation.layoutActions (by
      change PiCCSInputs.phaseOffset ≤ PiCCSInputs.phaseOffset + 192400
      omega) initial
  rw [← ChallengeDerivation.layoutWiring_eq_compileWiring] at projected
  have finishLe :
      start + Duplex.Formal.recipeCount ChallengeDerivation.layoutActions ≤
        PiCCSStarts.outputBindingWitnessStart := by
    rw [challengeLayoutRecipeCount interface start]
    change PiCCSInputs.phaseOffset + 192400 + 51504 ≤
      PiCCSStarts.outputBindingWitnessStart
    rw [PiCCSInputs.phaseOffset_eq,
      PiCCSStarts.outputBindingWitnessStart_eq]
    norm_num
  change
    (∀ value ∈ (ChallengeDerivation.layoutWiring interface start).samples,
      Duplex.Formal.KSupported value Source) ∧
      Duplex.Formal.StateSupported
        (ChallengeDerivation.finalState interface start) Source
  constructor
  · intro value member
    exact (projected.1 value member).mono (extend_source _ finishLe)
  · rw [ChallengeDerivation.finalState_eq_finalStateFast_pointwise]
    exact projected.2.mono (extend_source _ finishLe)

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
  let shared := Formal.atOffset
    (PiCCSInputs.interface logicalWidth publicFits) PiCCSInputs.phaseOffset
  let challengeInterface :=
    Formal.challengeInterface shared PiCCSInputs.phaseOffset
  let interface := Formal.roundTranscriptInterface shared
  let start := Formal.roundTranscriptStart shared
  have challengeSupport := challengeWiring_supported logicalWidth publicFits
  have incoming : Duplex.Formal.StateSupported
      (interface.initialState start) Source := by
    simpa [interface, Formal.roundTranscriptInterface,
      Formal.challengeFinalState, challengeInterface, shared, Formal.atOffset]
      using challengeSupport.2
  have initial : Duplex.Formal.StateSupported (interface.initialState start)
      (Extend Source PiCCSInputs.phaseOffset start) := by
    apply incoming.mono
    intro index support
    exact SupportRange.base support
  have projected := Duplex.Formal.compileWiring_supported Source
    PiCCSInputs.phaseOffset start (interface.initialState start)
    (RoundTranscript.layoutActions interface start) (by
      change PiCCSInputs.phaseOffset ≤
        PiCCSInputs.phaseOffset + 192400 + 51504
      omega) initial
  rw [← RoundTranscript.layoutWiring_eq_compileWiring] at projected
  have finishLe :
      start + Duplex.Formal.recipeCount
          (RoundTranscript.layoutActions interface start) ≤
        PiCCSStarts.outputBindingWitnessStart := by
    rw [roundLayoutRecipeCount interface start]
    change PiCCSInputs.phaseOffset + 192400 + 51504 +
        productionShape.cubeVariables *
          RoundTranscript.perRoundRecipeCount 9 ≤
      PiCCSStarts.outputBindingWitnessStart
    rw [PiCCSInputs.phaseOffset_eq,
      PiCCSStarts.outputBindingWitnessStart_eq]
    norm_num [productionShape, Phi81MatrixSource.phi81Shape,
      cubeVariables, RoundTranscript.perRoundRecipeCount]
  change ∀ value ∈ (RoundTranscript.layoutWiring interface start).samples,
    Duplex.Formal.KSupported value Source
  intro value member
  exact (projected.1 value member).mono (extend_source _ finishLe)

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
      apply challengeSupport.1 sample
      exact List.mem_of_mem_take member
    · simp [ChallengeDerivation.layoutWiring_samples_length,
        productionShape, Phi81MatrixSource.phi81Shape, cubeVariables]
  · rw [Formal.challengeGamma,
      ChallengeDerivation.gamma_eq_gammaFast_pointwise]
    unfold ChallengeDerivation.gammaFast
    apply Duplex.Formal.sampleGetD_supported
    · exact challengeSupport.1
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

end NightstreamFPrime.Layout.Stage1.PiCCSOrdinarySourceSupport
