import Nightstream.Implementation.R1CS.Canonical.KSplitNcEndpointsHonest

/-!
Contract: derive constructive endpoint-row completeness from the unchanged
Split-NC endpoint relation.

This module eliminates the temporary local `Bindings` interface used by the
three-program witness composition.  Each binding is derived from
`KSplitNcOperational.EndpointAgrees`, transcript replay, and authoritative
public-input decoding.  No endpoint equation is accepted as a new caller
field.
-/

set_option autoImplicit false
set_option maxRecDepth 100000

namespace Nightstream.Implementation.R1CS.Canonical.KSplitNcEndpointsSemanticHonest

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.KCompositeRowSupport
open Nightstream.Implementation.R1CS.Canonical.KMul
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.OutputClaims
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.TranscriptAuthority.BlockLane

private theorem cubePoint_eq_of_coordinates_eq
    {Field : Type} {count : Nat}
    (left right : CubePoint Field count)
    (coordinates : left.coordinates = right.coordinates) :
    left = right := by
  cases left
  cases right
  simp_all

/-- Decoding one carried extension value is invariant when every column it
mentions is preserved below the supplied boundary. -/
theorem decoded_eq_of_preserved
    (oldAssignment newAssignment : Nat → Nat)
    (value : Carried) (boundary : Nat)
    (below : CarriedBelow value boundary)
    (preserved :
      ∀ column, column < boundary →
        newAssignment column = oldAssignment column) :
    KPointEquality.decoded newAssignment value =
      KPointEquality.decoded oldAssignment value := by
  apply KConcreteBridge.ofConcrete_injective
  rw [KPointEquality.ofConcrete_decoded,
    KPointEquality.ofConcrete_decoded]
  unfold KHorner.carriedValue
  simp only [KHorner.Pair.mk.injEq]
  constructor
  · apply KMulHonest.lcEval_congr
    intro column mentioned
    exact preserved column (below.1 column mentioned)
  · apply KMulHonest.lcEval_congr
    intro column mentioned
    exact preserved column (below.2 column mentioned)

private theorem decodedPointOf_eq_of_preserved
    {count : Nat}
    (oldAssignment newAssignment : Nat → Nat)
    (values : Fin count → Carried) (boundary : Nat)
    (below : ∀ index, CarriedBelow (values index) boundary)
    (preserved :
      ∀ column, column < boundary →
        newAssignment column = oldAssignment column) :
    KSplitNcEndpoints.decodedPointOf values newAssignment =
      KSplitNcEndpoints.decodedPointOf values oldAssignment := by
  apply cubePoint_eq_of_coordinates_eq
  unfold KSplitNcEndpoints.decodedPointOf
  apply congrArg List.ofFn
  funext index
  exact decoded_eq_of_preserved oldAssignment newAssignment
    (values index) boundary (below index) preserved

private theorem coordinates_of_decoded_eq
    (assignment : Nat → Nat) (left right : Carried)
    (equal :
      KPointEquality.decoded assignment left =
        KPointEquality.decoded assignment right) :
    lcEval assignment left.low = lcEval assignment right.low ∧
      lcEval assignment left.high = lcEval assignment right.high := by
  have pairEqual := congrArg KConcreteBridge.ofConcrete equal
  rw [KPointEquality.ofConcrete_decoded,
    KPointEquality.ofConcrete_decoded] at pairEqual
  simpa only [KHorner.carriedValue, KHorner.Pair.mk.injEq]
    using pairEqual

private theorem feCoins_eq_of_fields
    {shape : SemanticShape} {domain : FlatNcDomain}
    (left right : Polynomial.Fe.Coins shape domain)
    (alpha : left.alpha = right.alpha)
    (betaA : left.betaA = right.betaA)
    (betaR : left.betaR = right.betaR)
    (gamma : left.gamma = right.gamma) :
    left = right := by
  cases left
  cases right
  simp_all

private theorem ncCoins_eq_of_fields
    {domain : BlockNcDomain}
    (left right : Polynomial.Nc.BlockLane.Mixing.Coins domain)
    (betaBlock : left.betaBlock = right.betaBlock)
    (betaA : left.betaA = right.betaA)
    (gamma : left.gamma = right.gamma) :
    left = right := by
  cases left
  cases right
  simp_all

private def preservedAuthority
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    {message : OutputMessage shape}
    (input : KSplitNcEndpoints.Input polynomialInput domains)
    (oldAssignment newAssignment : Nat → Nat)
    (sources : KSplitNcEndpointsSupport.InputsBelow input)
    (preserved :
      ∀ column, column < input.frameBase →
        newAssignment column = oldAssignment column)
    (authority :
      KSplitNcEndpoints.DecodedAuthority
        input oldAssignment message) :
    KSplitNcEndpoints.DecodedAuthority
      input newAssignment message where
  priorPoint := fun coordinate => by
    rw [decoded_eq_of_preserved oldAssignment newAssignment
      (input.authority.priorPoint coordinate) input.frameBase
      (sources.feTerminalPriorPoint coordinate) preserved]
    exact authority.priorPoint coordinate
  claimedYRing := fun running matrix lane => by
    rw [decoded_eq_of_preserved oldAssignment newAssignment
      (input.authority.claimedYRing running matrix lane) input.frameBase
      (sources.feInitialClaims running matrix lane) preserved]
    exact authority.claimedYRing running matrix lane
  outputYRing := fun source matrix lane => by
    rw [decoded_eq_of_preserved oldAssignment newAssignment
      (input.authority.outputYRing source matrix lane) input.frameBase
      (sources.feTerminalMessage source matrix lane) preserved]
    exact authority.outputYRing source matrix lane
  outputYZcol := fun source lane => by
    rw [decoded_eq_of_preserved oldAssignment newAssignment
      (input.authority.outputYZcol source lane) input.frameBase
      (sources.ncMessage source lane) preserved]
    exact authority.outputYZcol source lane

private def calculatedFeInitialInput
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (input : KSplitNcEndpoints.Input polynomialInput domains) :
    KSplitNcFeInitial.Input shape domains.fe :=
  { KSplitNcEndpoints.feInitialInput input with
    initial :=
      KSplitNcFeInitial.evaluated
        (KSplitNcEndpoints.feInitialInput input) }

private theorem outer_le_feTerminalBase
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (input : KSplitNcEndpoints.Input polynomialInput domains) :
    input.frameBase ≤
      (KSplitNcEndpoints.feTerminalInput input).frameBase := by
  change input.frameBase ≤ KSplitNcEndpoints.feTerminalBase input
  unfold KSplitNcEndpoints.feTerminalBase
  omega

private theorem outer_le_ncBase
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (input : KSplitNcEndpoints.Input polynomialInput domains) :
    input.frameBase ≤ (KSplitNcEndpoints.ncInput input).frameBase := by
  change input.frameBase ≤ KSplitNcEndpoints.ncBase input
  unfold KSplitNcEndpoints.ncBase KSplitNcEndpoints.feTerminalBase
  omega

private def calculatedFeTerminalInput
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (input : KSplitNcEndpoints.Input polynomialInput domains) :
    KSplitNcFeTerminal.Input polynomialInput domains.fe :=
  { KSplitNcEndpoints.feTerminalInput input with
    terminal :=
      KSplitNcFeTerminal.terminalExpression
        (KSplitNcEndpoints.feTerminalInput input) }

private def feTerminalSources
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (input : KSplitNcEndpoints.Input polynomialInput domains)
    (sources : KSplitNcEndpointsSupport.InputsBelow input) :
    KSplitNcFeTerminalProductsHonest.SourceBounds
      (calculatedFeTerminalInput input) where
  gamma := carried_mono sources.feTerminalGamma
    (outer_le_feTerminalBase input)
  pointLane := fun coordinate =>
    carried_mono (sources.feTerminalPointLane coordinate)
      (outer_le_feTerminalBase input)
  pointRow := fun coordinate =>
    carried_mono (sources.feTerminalPointRow coordinate)
      (outer_le_feTerminalBase input)
  betaA := fun coordinate =>
    carried_mono (sources.feTerminalBetaA coordinate)
      (outer_le_feTerminalBase input)
  betaR := fun coordinate =>
    carried_mono (sources.feTerminalBetaR coordinate)
      (outer_le_feTerminalBase input)
  alpha := fun coordinate =>
    carried_mono (sources.feTerminalAlpha coordinate)
      (outer_le_feTerminalBase input)
  priorPoint := fun coordinate =>
    carried_mono (sources.feTerminalPriorPoint coordinate)
      (outer_le_feTerminalBase input)
  message := fun source matrix lane =>
    carried_mono (sources.feTerminalMessage source matrix lane)
      (outer_le_feTerminalBase input)

private def calculatedNcInput
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (input : KSplitNcEndpoints.Input polynomialInput domains) :
    KSplitNcNcEndpoint.Input shape domains.nc :=
  { KSplitNcEndpoints.ncInput input with
    initial := KLinear.zeroCarried
    terminal :=
      KSplitNcNcEndpoint.terminalExpression
        (KSplitNcEndpoints.ncInput input) }

private def ncSources
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (input : KSplitNcEndpoints.Input polynomialInput domains)
    (sources : KSplitNcEndpointsSupport.InputsBelow input) :
    KSplitNcNcEndpointHonest.SourceBounds
      (KSplitNcEndpoints.ncInput input) where
  arithmetic :=
    {
      gamma := carried_mono sources.ncGamma (outer_le_ncBase input)
      pointLane := fun coordinate =>
        carried_mono (sources.ncPointLane coordinate)
          (outer_le_ncBase input)
      message := fun source lane =>
        carried_mono (sources.ncMessage source lane)
          (outer_le_ncBase input)
    }
  betaBlock := fun coordinate =>
    carried_mono (sources.ncBetaBlock coordinate) (outer_le_ncBase input)
  betaA := fun coordinate =>
    carried_mono (sources.ncBetaA coordinate) (outer_le_ncBase input)
  pointBlock := fun coordinate =>
    carried_mono (sources.ncPointBlock coordinate) (outer_le_ncBase input)
  initial := carried_mono sources.ncInitialEndpoint (outer_le_ncBase input)
  terminal := carried_mono sources.ncTerminalEndpoint (outer_le_ncBase input)

private theorem feInitial_binding
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (profile : Polynomial.Fe.SupportedProfile shape domains.fe)
    (constants : Poseidon2Schedule.Constants)
    (assignment : Nat → Nat)
    (constantWire : assignment 0 = 1)
    (input : KSplitNcEndpoints.Input polynomialInput domains)
    (message : OutputMessage shape)
    (transcriptValid :
      SymbolicDuplexSemantics.Valid
        input.transcript.transcriptBase constants assignment
        (KSplitNcTranscript.outputBuilder input.transcript))
    (authority :
      KSplitNcEndpoints.DecodedAuthority input assignment message)
    (sources : KSplitNcEndpointsSupport.InputsBelow input)
    (endpoints :
      KSplitNcOperational.EndpointAgrees
        profile constants assignment input.transcript message)
    (positive : 0 < input.frameBase) :
    KSplitNcFeInitialHonest.Binding
      (KSplitNcEndpoints.feInitialInput input) assignment := by
  let localInput := KSplitNcEndpoints.feInitialInput input
  let calculatedInput := calculatedFeInitialInput input
  let localWitness :=
    KSplitNcFeInitialHonest.witness localInput assignment
  have localWitnessDef :
      KSplitNcFeInitialHonest.witness calculatedInput assignment =
        localWitness := rfl
  have localOne : localWitness 0 = 1 := by
    unfold localWitness
    rw [KSplitNcFeInitialHonest.witness_off_block
      localInput assignment 0 positive]
    exact constantWire
  have calculatedBinding :
      KSplitNcFeInitialHonest.Binding calculatedInput assignment := by
    constructor <;> rfl
  have calculatedSatisfied :
      Satisfies (KSplitNcFeInitial.rows calculatedInput) localWitness := by
    rw [← localWitnessDef]
    exact KSplitNcFeInitialHonest.rows_honest_of_binding
      calculatedInput assignment positive constantWire
      sources.feInitialGamma sources.feInitialAlpha
      sources.feInitialClaims calculatedBinding
  have localAuthority :=
    preservedAuthority input assignment localWitness sources
      (KSplitNcEndpointsHonest.afterFeInitial_off_source
        input assignment) authority
  have calculatedSound :=
    KSplitNcFeInitial.rows_sound profile polynomialInput
      calculatedInput localWitness localOne
      (fun running matrix lane =>
        (KSplitNcEndpoints.feInitial_decoder_eq localWitness
          (input.authority.claimedYRing running matrix lane)).trans
          (localAuthority.claimedYRing running matrix lane))
      calculatedSatisfied
  have pre :
      KSplitNcTranscriptSemantics.PreAgrees assignment
        (KSplitNcTranscript.replay input.transcript)
        (KSplitNcTranscriptPhases.semanticPre
          constants assignment input.transcript) := by
    simpa only [KSplitNcTranscriptPhases.semanticPre] using
      KSplitNcTranscriptSemantics.decoded_preSumcheck
        constants assignment constantWire input.transcript transcriptValid
  have alphaPreserved :
      KSplitNcFeInitial.decodedAlpha calculatedInput localWitness =
        KSplitNcEndpoints.decodedPointOf
          (KSplitNcEndpoints.coreAlpha input) assignment := by
    change
      KSplitNcFeInitial.decodedAlpha
          (KSplitNcEndpoints.feInitialInput input) localWitness =
        _
    rw [KSplitNcEndpoints.feInitialAlpha_eq_core]
    exact decodedPointOf_eq_of_preserved assignment localWitness
      (KSplitNcEndpoints.coreAlpha input) input.frameBase
      sources.feInitialAlpha
      (KSplitNcEndpointsHonest.afterFeInitial_off_source input assignment)
  have gammaPreserved :
      KSplitNcFeInitial.decoded localWitness calculatedInput.gamma =
        KPointEquality.decoded assignment
          (KSplitNcEndpoints.feInitialInput input).gamma := by
    rw [KSplitNcEndpoints.feInitial_decoder_eq]
    exact decoded_eq_of_preserved assignment localWitness
      (KSplitNcEndpoints.feInitialInput input).gamma input.frameBase
      sources.feInitialGamma
      (KSplitNcEndpointsHonest.afterFeInitial_off_source input assignment)
  have gammaSemantic :
      KPointEquality.decoded assignment
          (KSplitNcEndpoints.feInitialInput input).gamma =
        (KSplitNcTranscriptPhases.semanticPre
          constants assignment input.transcript).challenges.gamma := by
    simpa only [KSplitNcEndpoints.feInitialInput] using
      KSplitNcEndpoints.coreGamma_eq
        constants assignment input pre
  have computedSemantic :
      KSplitNcFeInitial.decoded localWitness
          (KSplitNcFeInitial.evaluated localInput) =
        KSplitNcTranscriptPhases.semanticFeInitial
          profile constants assignment input.transcript := by
    change
      KSplitNcFeInitial.decoded localWitness calculatedInput.initial =
        _
    rw [calculatedSound]
    unfold KSplitNcTranscriptPhases.semanticFeInitial
      KSplitNcFeInitial.decodedCoins Polynomial.Fe.initial
    rw [alphaPreserved, gammaPreserved,
      KSplitNcEndpoints.coreAlpha_eq constants assignment input pre,
      gammaSemantic]
    rfl
  have targetSemantic :
      KSplitNcFeInitial.decoded localWitness localInput.initial =
        KSplitNcTranscriptPhases.semanticFeInitial
          profile constants assignment input.transcript := by
    calc
      KSplitNcFeInitial.decoded localWitness localInput.initial =
          KPointEquality.decoded localWitness localInput.initial :=
        KSplitNcEndpoints.feInitial_decoder_eq _ _
      _ = KPointEquality.decoded assignment localInput.initial :=
        decoded_eq_of_preserved assignment localWitness
          localInput.initial input.frameBase sources.feInitialEndpoint
          (KSplitNcEndpointsHonest.afterFeInitial_off_source input assignment)
      _ =
          KSplitNcTranscriptSemantics.decodedColumns
            assignment input.transcript.fe.initial :=
        KSplitNcEndpoints.decoded_carried
          assignment input.transcript.fe.initial
      _ = _ := endpoints.feInitial
  have decodedEqual :
      KPointEquality.decoded localWitness
          (KSplitNcFeInitial.evaluated localInput) =
        KPointEquality.decoded localWitness localInput.initial := by
    rw [← KSplitNcEndpoints.feInitial_decoder_eq,
      ← KSplitNcEndpoints.feInitial_decoder_eq]
    exact computedSemantic.trans targetSemantic.symm
  exact
    {
      low := (coordinates_of_decoded_eq localWitness
        (KSplitNcFeInitial.evaluated localInput)
        localInput.initial decodedEqual).1
      high := (coordinates_of_decoded_eq localWitness
        (KSplitNcFeInitial.evaluated localInput)
        localInput.initial decodedEqual).2
    }

private theorem feTerminal_binding
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (profile : Polynomial.Fe.SupportedProfile shape domains.fe)
    (constants : Poseidon2Schedule.Constants)
    (assignment : Nat → Nat)
    (constantWire : assignment 0 = 1)
    (input : KSplitNcEndpoints.Input polynomialInput domains)
    (message : OutputMessage shape)
    (transcriptValid :
      SymbolicDuplexSemantics.Valid
        input.transcript.transcriptBase constants assignment
        (KSplitNcTranscript.outputBuilder input.transcript))
    (authority :
      KSplitNcEndpoints.DecodedAuthority input assignment message)
    (sources : KSplitNcEndpointsSupport.InputsBelow input)
    (endpoints :
      KSplitNcOperational.EndpointAgrees
        profile constants assignment input.transcript message)
    (positive : 0 < input.frameBase) :
    KSplitNcFeTerminalProductsHonest.TerminalBinding
      (KSplitNcEndpoints.feTerminalInput input)
      (KSplitNcEndpointsHonest.afterFeInitial input assignment) := by
  let stage := KSplitNcEndpointsHonest.afterFeInitial input assignment
  let localInput := KSplitNcEndpoints.feTerminalInput input
  let calculatedInput := calculatedFeTerminalInput input
  let localWitness :=
    KSplitNcFeTerminalProductsHonest.witness localInput stage
  have localWitnessDef :
      KSplitNcFeTerminalProductsHonest.witness calculatedInput stage =
        localWitness := rfl
  have stageOne : stage 0 = 1 := by
    unfold stage KSplitNcEndpointsHonest.afterFeInitial
    rw [KSplitNcFeInitialHonest.witness_off_block
      (KSplitNcEndpoints.feInitialInput input) assignment 0 positive]
    exact constantWire
  have terminalPositive : 0 < localInput.frameBase :=
    Nat.lt_of_lt_of_le positive (outer_le_feTerminalBase input)
  have localOne : localWitness 0 = 1 := by
    unfold localWitness
    exact KSplitNcFeTerminalProductsHonest.witness_constantWire
      localInput stage terminalPositive stageOne
  have calculatedBinding :
      KSplitNcFeTerminalProductsHonest.TerminalBinding
        calculatedInput stage := by
    constructor <;> rfl
  have calculatedSatisfied :
      Satisfies (KSplitNcFeTerminal.rows calculatedInput) localWitness := by
    rw [← localWitnessDef]
    exact KSplitNcFeTerminalProductsHonest.rows_honest_of_binding
      calculatedInput stage terminalPositive stageOne
      (feTerminalSources input sources) calculatedBinding
  have calculatedSound :=
    KSplitNcFeTerminal.rows_sound profile calculatedInput localWitness
      localOne calculatedSatisfied
  have pre :
      KSplitNcTranscriptSemantics.PreAgrees assignment
        (KSplitNcTranscript.replay input.transcript)
        (KSplitNcTranscriptPhases.semanticPre
          constants assignment input.transcript) := by
    simpa only [KSplitNcTranscriptPhases.semanticPre] using
      KSplitNcTranscriptSemantics.decoded_preSumcheck
        constants assignment constantWire input.transcript transcriptValid
  have feReplay :=
    KSplitNcTranscriptPhases.decoded_fe
      profile constants assignment constantWire input.transcript
      transcriptValid endpoints.feInitial
  have preserved :
      ∀ column, column < input.frameBase →
        localWitness column = assignment column :=
    KSplitNcEndpointsHonest.afterFeTerminal_off_source input assignment
  have localAuthority :=
    preservedAuthority input assignment localWitness sources preserved authority
  have publicInputSemantic :
      KSplitNcFeTerminal.decodedPublicInput calculatedInput localWitness =
        polynomialInput := by
    change
      KSplitNcFeTerminal.decodedPublicInput localInput localWitness =
        polynomialInput
    exact KSplitNcEndpoints.feTerminal_publicInput_eq
      input localWitness localAuthority
  have coinsPreserved :
      KSplitNcFeTerminal.decodedCoins calculatedInput localWitness =
        KSplitNcFeTerminal.decodedCoins localInput assignment := by
    apply feCoins_eq_of_fields
    · change
        KPointEquality.decodedRight
            (KSplitNcFeTerminal.carriedLaneEqualityInput localInput)
            localWitness =
          KPointEquality.decodedRight
            (KSplitNcFeTerminal.carriedLaneEqualityInput localInput)
            assignment
      rw [KSplitNcEndpoints.decodedRight_eq_decodedPointOf,
        KSplitNcEndpoints.decodedRight_eq_decodedPointOf]
      exact decodedPointOf_eq_of_preserved assignment localWitness
        (KSplitNcEndpoints.coreAlpha input) input.frameBase
        sources.feTerminalAlpha preserved
    · change
        KPointEquality.decodedRight
            (KSplitNcFeTerminal.freshLaneEqualityInput localInput)
            localWitness =
          KPointEquality.decodedRight
            (KSplitNcFeTerminal.freshLaneEqualityInput localInput)
            assignment
      rw [KSplitNcEndpoints.decodedRight_eq_decodedPointOf,
        KSplitNcEndpoints.decodedRight_eq_decodedPointOf]
      exact decodedPointOf_eq_of_preserved assignment localWitness
        (KSplitNcEndpoints.coreBetaA input) input.frameBase
        sources.feTerminalBetaA preserved
    · change
        KPointEquality.decodedRight
            (KSplitNcFeTerminal.freshRowEqualityInput localInput)
            localWitness =
          KPointEquality.decodedRight
            (KSplitNcFeTerminal.freshRowEqualityInput localInput)
            assignment
      rw [KSplitNcEndpoints.decodedRight_eq_decodedPointOf,
        KSplitNcEndpoints.decodedRight_eq_decodedPointOf]
      exact decodedPointOf_eq_of_preserved assignment localWitness
        (KSplitNcEndpoints.coreBetaR input) input.frameBase
        sources.feTerminalBetaR preserved
    · exact decoded_eq_of_preserved assignment localWitness
        localInput.gamma input.frameBase sources.feTerminalGamma preserved
  have coinsSemantic :
      KSplitNcFeTerminal.decodedCoins calculatedInput localWitness =
        (KSplitNcTranscriptPhases.semanticPre
          constants assignment input.transcript).challenges.feCoins :=
    coinsPreserved.trans
      (KSplitNcEndpoints.feTerminal_coins_eq
        constants assignment input pre)
  have pointPreserved :
      KSplitNcFeTerminal.decodedPoint calculatedInput localWitness =
        KSplitNcFeTerminal.decodedPoint localInput assignment := by
    apply Polynomial.Fe.Point.ext
    · change
        KPointEquality.decodedLeft
            (KSplitNcFeTerminal.freshRowEqualityInput localInput)
            localWitness =
          KPointEquality.decodedLeft
            (KSplitNcFeTerminal.freshRowEqualityInput localInput)
            assignment
      rw [KSplitNcEndpoints.decodedLeft_eq_decodedPointOf,
        KSplitNcEndpoints.decodedLeft_eq_decodedPointOf]
      exact decodedPointOf_eq_of_preserved assignment localWitness
        (KSplitNcEndpoints.feRowPoint input) input.frameBase
        sources.feTerminalPointRow preserved
    · change
        KPointEquality.decodedLeft
            (KSplitNcFeTerminal.freshLaneEqualityInput localInput)
            localWitness =
          KPointEquality.decodedLeft
            (KSplitNcFeTerminal.freshLaneEqualityInput localInput)
            assignment
      rw [KSplitNcEndpoints.decodedLeft_eq_decodedPointOf,
        KSplitNcEndpoints.decodedLeft_eq_decodedPointOf]
      exact decodedPointOf_eq_of_preserved assignment localWitness
        (KSplitNcEndpoints.feLanePoint input) input.frameBase
        sources.feTerminalPointLane preserved
  have pointSemantic :
      KSplitNcFeTerminal.decodedPoint calculatedInput localWitness =
        (KSplitNcTranscriptPhases.semanticFeExecution
          profile constants assignment input.transcript).challengePoint := by
    rw [pointPreserved,
      KSplitNcEndpoints.feTerminal_point_eq input assignment,
      feReplay.point]
  have messageYRing :
      (KSplitNcFeTerminal.decodedMessage
        calculatedInput localWitness).yRing = message.yRing := by
    funext source matrix lane
    change KPointEquality.decoded localWitness
      (input.authority.outputYRing source matrix lane) =
        message.yRing source matrix lane
    rw [decoded_eq_of_preserved assignment localWitness
      (input.authority.outputYRing source matrix lane) input.frameBase
      (sources.feTerminalMessage source matrix lane) preserved]
    exact authority.outputYRing source matrix lane
  have computedSemantic :
      KSplitNcFeTerminal.decoded localWitness
          (KSplitNcFeTerminal.terminalExpression localInput) =
        Polynomial.Fe.terminalFromMessage profile polynomialInput
          (KSplitNcTranscriptPhases.semanticPre
            constants assignment input.transcript).challenges.feCoins
          (KSplitNcTranscriptPhases.semanticFeExecution
            profile constants assignment input.transcript).challengePoint
          message := by
    change
      KSplitNcFeTerminal.decoded localWitness calculatedInput.terminal = _
    rw [calculatedSound, publicInputSemantic, coinsSemantic, pointSemantic]
    unfold Polynomial.Fe.terminalFromMessage
    rw [messageYRing]
  have targetSemantic :
      KSplitNcFeTerminal.decoded localWitness localInput.terminal =
        Polynomial.Fe.terminalFromMessage profile polynomialInput
          (KSplitNcTranscriptPhases.semanticPre
            constants assignment input.transcript).challenges.feCoins
          (KSplitNcTranscriptPhases.semanticFeExecution
            profile constants assignment input.transcript).challengePoint
          message := by
    calc
      KSplitNcFeTerminal.decoded localWitness localInput.terminal =
          KPointEquality.decoded assignment localInput.terminal :=
        decoded_eq_of_preserved assignment localWitness
          localInput.terminal input.frameBase sources.feTerminalEndpoint
          preserved
      _ =
          KSplitNcTranscriptSemantics.decodedColumns
            assignment input.transcript.fe.terminal :=
        KSplitNcEndpoints.decoded_carried
          assignment input.transcript.fe.terminal
      _ = _ := endpoints.feTerminal
  have decodedEqual :
      KPointEquality.decoded localWitness
          (KSplitNcFeTerminal.terminalExpression localInput) =
        KPointEquality.decoded localWitness localInput.terminal :=
    computedSemantic.trans targetSemantic.symm
  exact
    {
      low := (coordinates_of_decoded_eq localWitness
        (KSplitNcFeTerminal.terminalExpression localInput)
        localInput.terminal decodedEqual).1
      high := (coordinates_of_decoded_eq localWitness
        (KSplitNcFeTerminal.terminalExpression localInput)
        localInput.terminal decodedEqual).2
    }

private theorem nc_binding
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (profile : Polynomial.Fe.SupportedProfile shape domains.fe)
    (constants : Poseidon2Schedule.Constants)
    (assignment : Nat → Nat)
    (constantWire : assignment 0 = 1)
    (input : KSplitNcEndpoints.Input polynomialInput domains)
    (message : OutputMessage shape)
    (transcriptValid :
      SymbolicDuplexSemantics.Valid
        input.transcript.transcriptBase constants assignment
        (KSplitNcTranscript.outputBuilder input.transcript))
    (authority :
      KSplitNcEndpoints.DecodedAuthority input assignment message)
    (sources : KSplitNcEndpointsSupport.InputsBelow input)
    (endpoints :
      KSplitNcOperational.EndpointAgrees
        profile constants assignment input.transcript message)
    (positive : 0 < input.frameBase) :
    KSplitNcNcEndpointHonest.EndpointBinding
      (KSplitNcEndpoints.ncInput input)
      (KSplitNcEndpointsHonest.afterFeTerminal input assignment) := by
  let stage := KSplitNcEndpointsHonest.afterFeTerminal input assignment
  let localInput := KSplitNcEndpoints.ncInput input
  let calculatedInput := calculatedNcInput input
  let localWitness := KSplitNcNcEndpointHonest.witness localInput stage
  have localWitnessDef :
      KSplitNcNcEndpointHonest.witness calculatedInput stage =
        localWitness := rfl
  have afterInitialOne :
      KSplitNcEndpointsHonest.afterFeInitial input assignment 0 = 1 := by
    unfold KSplitNcEndpointsHonest.afterFeInitial
    rw [KSplitNcFeInitialHonest.witness_off_block
      (KSplitNcEndpoints.feInitialInput input) assignment 0 positive]
    exact constantWire
  have terminalPositive :
      0 < (KSplitNcEndpoints.feTerminalInput input).frameBase :=
    Nat.lt_of_lt_of_le positive (outer_le_feTerminalBase input)
  have stageOne : stage 0 = 1 := by
    unfold stage KSplitNcEndpointsHonest.afterFeTerminal
    exact KSplitNcFeTerminalProductsHonest.witness_constantWire
      (KSplitNcEndpoints.feTerminalInput input)
      (KSplitNcEndpointsHonest.afterFeInitial input assignment)
      terminalPositive afterInitialOne
  have ncPositive : 0 < localInput.frameBase :=
    Nat.lt_of_lt_of_le positive (outer_le_ncBase input)
  have localOne : localWitness 0 = 1 := by
    unfold localWitness
    exact KSplitNcNcEndpointHonest.witness_constantWire
      localInput stage ncPositive stageOne
  have calculatedSatisfied :
      Satisfies (KSplitNcNcEndpoint.rows calculatedInput) localWitness := by
    rw [KSplitNcNcEndpointHonest.rows_eq_initial_append_computed_append_terminal]
    have initialSatisfied :
        Satisfies (KSplitNcNcEndpoint.initialRows calculatedInput)
          localWitness := by
      apply KEquality.rows_complete localWitness
        KLinear.zeroCarried calculatedInput.initial localOne <;> rfl
    have computedSatisfied :
        Satisfies (KSplitNcNcEndpointHonest.computedRows calculatedInput)
          localWitness := by
      change
        Satisfies
          (KSplitNcNcEndpointHonest.computedRows localInput)
          (KSplitNcNcEndpointHonest.witness localInput stage)
      exact KSplitNcNcEndpointHonest.computedRows_honest
        localInput stage ncPositive (ncSources input sources)
    have terminalSatisfied :
        Satisfies
          (KEquality.rows
            (KSplitNcNcEndpoint.terminalExpression calculatedInput)
            calculatedInput.terminal) localWitness := by
      apply KEquality.rows_complete localWitness
        (KSplitNcNcEndpoint.terminalExpression calculatedInput)
        calculatedInput.terminal localOne <;> rfl
    intro row member
    rcases List.mem_append.1 member with inInitial | inRest
    · exact initialSatisfied row inInitial
    rcases List.mem_append.1 inRest with inComputed | inTerminal
    · exact computedSatisfied row inComputed
    · exact terminalSatisfied row inTerminal
  have calculatedSound :=
    KSplitNcNcEndpoint.rows_sound calculatedInput localWitness
      localOne calculatedSatisfied
  have pre :
      KSplitNcTranscriptSemantics.PreAgrees assignment
        (KSplitNcTranscript.replay input.transcript)
        (KSplitNcTranscriptPhases.semanticPre
          constants assignment input.transcript) := by
    simpa only [KSplitNcTranscriptPhases.semanticPre] using
      KSplitNcTranscriptSemantics.decoded_preSumcheck
        constants assignment constantWire input.transcript transcriptValid
  have ncReplay :=
    KSplitNcTranscriptPhases.decoded_nc
      profile constants assignment constantWire input.transcript
      transcriptValid endpoints.feInitial
  have preserved :
      ∀ column, column < input.frameBase →
        localWitness column = assignment column :=
    KSplitNcEndpointsHonest.witness_off_source input assignment
  have coinsPreserved :
      KSplitNcNcEndpoint.decodedCoins calculatedInput localWitness =
        KSplitNcNcEndpoint.decodedCoins localInput assignment := by
    apply ncCoins_eq_of_fields
    · change
        KPointEquality.decodedRight
            (KSplitNcNcEndpoint.blockEqualityInput localInput)
            localWitness =
          KPointEquality.decodedRight
            (KSplitNcNcEndpoint.blockEqualityInput localInput)
            assignment
      rw [KSplitNcEndpoints.decodedRight_eq_decodedPointOf,
        KSplitNcEndpoints.decodedRight_eq_decodedPointOf]
      exact decodedPointOf_eq_of_preserved assignment localWitness
        (KSplitNcEndpoints.coreBetaBlock input) input.frameBase
        sources.ncBetaBlock preserved
    · change
        KPointEquality.decodedRight
            (KSplitNcNcEndpoint.laneEqualityInput localInput)
            localWitness =
          KPointEquality.decodedRight
            (KSplitNcNcEndpoint.laneEqualityInput localInput)
            assignment
      rw [KSplitNcEndpoints.decodedRight_eq_decodedPointOf,
        KSplitNcEndpoints.decodedRight_eq_decodedPointOf]
      exact decodedPointOf_eq_of_preserved assignment localWitness
        (KSplitNcEndpoints.coreBetaA input) input.frameBase
        sources.ncBetaA preserved
    · exact decoded_eq_of_preserved assignment localWitness
        localInput.gamma input.frameBase sources.ncGamma preserved
  have coinsSemantic :
      KSplitNcNcEndpoint.decodedCoins calculatedInput localWitness =
        (KSplitNcTranscriptPhases.semanticPre
          constants assignment input.transcript).challenges.ncCoins :=
    coinsPreserved.trans
      (KSplitNcEndpoints.ncCoins_eq constants assignment input pre)
  have pointPreserved :
      KSplitNcNcEndpoint.decodedPoint calculatedInput localWitness =
        KSplitNcNcEndpoint.decodedPoint localInput assignment := by
    apply Polynomial.Nc.BlockLane.Point.ext
    · change
        KPointEquality.decodedLeft
            (KSplitNcNcEndpoint.blockEqualityInput localInput)
            localWitness =
          KPointEquality.decodedLeft
            (KSplitNcNcEndpoint.blockEqualityInput localInput)
            assignment
      rw [KSplitNcEndpoints.decodedLeft_eq_decodedPointOf,
        KSplitNcEndpoints.decodedLeft_eq_decodedPointOf]
      exact decodedPointOf_eq_of_preserved assignment localWitness
        (KSplitNcEndpoints.ncBlockPoint input) input.frameBase
        sources.ncPointBlock preserved
    · change
        KPointEquality.decodedLeft
            (KSplitNcNcEndpoint.laneEqualityInput localInput)
            localWitness =
          KPointEquality.decodedLeft
            (KSplitNcNcEndpoint.laneEqualityInput localInput)
            assignment
      rw [KSplitNcEndpoints.decodedLeft_eq_decodedPointOf,
        KSplitNcEndpoints.decodedLeft_eq_decodedPointOf]
      exact decodedPointOf_eq_of_preserved assignment localWitness
        (KSplitNcEndpoints.ncLanePoint input) input.frameBase
        sources.ncPointLane preserved
  have pointSemantic :
      KSplitNcNcEndpoint.decodedPoint calculatedInput localWitness =
        (KSplitNcTranscriptPhases.semanticNcExecution
          profile constants assignment input.transcript).challengePoint := by
    rw [pointPreserved,
      KSplitNcEndpoints.ncPoint_eq input assignment,
      ncReplay.point]
  have messageYZcol :
      ∀ source lane,
        (KSplitNcNcEndpoint.decodedMessage
            calculatedInput localWitness).yZcol source lane =
          message.yZcol source lane := by
    intro source lane
    change KPointEquality.decoded localWitness
      (input.authority.outputYZcol source lane) =
        message.yZcol source lane
    rw [decoded_eq_of_preserved assignment localWitness
      (input.authority.outputYZcol source lane) input.frameBase
      (sources.ncMessage source lane) preserved]
    exact authority.outputYZcol source lane
  have computedTerminal :
      KSplitNcNcEndpoint.decoded localWitness
          (KSplitNcNcEndpoint.terminalExpression localInput) =
        Polynomial.Nc.BlockLane.Terminal.terminalFromMessage message
          (KSplitNcTranscriptPhases.semanticPre
            constants assignment input.transcript).challenges.ncCoins
          (KSplitNcTranscriptPhases.semanticNcExecution
            profile constants assignment input.transcript).challengePoint := by
    change
      KSplitNcNcEndpoint.decoded localWitness calculatedInput.terminal = _
    rw [calculatedSound.2, coinsSemantic, pointSemantic]
    exact KSplitNcEndpoints.ncTerminal_eq_of_yZcol
      _ _ messageYZcol _ _
  have targetInitial :
      KSplitNcNcEndpoint.decoded localWitness localInput.initial =
        Polynomial.Nc.BlockLane.InitialSum.claimedInitial := by
    calc
      KSplitNcNcEndpoint.decoded localWitness localInput.initial =
          KPointEquality.decoded assignment localInput.initial :=
        decoded_eq_of_preserved assignment localWitness localInput.initial
          input.frameBase sources.ncInitialEndpoint preserved
      _ =
          KSplitNcTranscriptSemantics.decodedColumns
            assignment input.transcript.nc.initial :=
        KSplitNcEndpoints.decoded_carried
          assignment input.transcript.nc.initial
      _ = _ := endpoints.ncInitial
  have targetTerminal :
      KSplitNcNcEndpoint.decoded localWitness localInput.terminal =
        Polynomial.Nc.BlockLane.Terminal.terminalFromMessage message
          (KSplitNcTranscriptPhases.semanticPre
            constants assignment input.transcript).challenges.ncCoins
          (KSplitNcTranscriptPhases.semanticNcExecution
            profile constants assignment input.transcript).challengePoint := by
    calc
      KSplitNcNcEndpoint.decoded localWitness localInput.terminal =
          KPointEquality.decoded assignment localInput.terminal :=
        decoded_eq_of_preserved assignment localWitness localInput.terminal
          input.frameBase sources.ncTerminalEndpoint preserved
      _ =
          KSplitNcTranscriptSemantics.decodedColumns
            assignment input.transcript.nc.terminal :=
        KSplitNcEndpoints.decoded_carried
          assignment input.transcript.nc.terminal
      _ = _ := endpoints.ncTerminal
  have initialEqual :
      KPointEquality.decoded localWitness KLinear.zeroCarried =
        KPointEquality.decoded localWitness localInput.initial :=
    calculatedSound.1.trans targetInitial.symm
  have terminalEqual :
      KPointEquality.decoded localWitness
          (KSplitNcNcEndpoint.terminalExpression localInput) =
        KPointEquality.decoded localWitness localInput.terminal :=
    computedTerminal.trans targetTerminal.symm
  exact
    {
      initialLow := (coordinates_of_decoded_eq localWitness
        KLinear.zeroCarried localInput.initial initialEqual).1
      initialHigh := (coordinates_of_decoded_eq localWitness
        KLinear.zeroCarried localInput.initial initialEqual).2
      terminalLow := (coordinates_of_decoded_eq localWitness
        (KSplitNcNcEndpoint.terminalExpression localInput)
        localInput.terminal terminalEqual).1
      terminalHigh := (coordinates_of_decoded_eq localWitness
        (KSplitNcNcEndpoint.terminalExpression localInput)
        localInput.terminal terminalEqual).2
    }

/-- The unchanged deterministic endpoint relation constructs every temporary
endpoint equality required by the sequential physical witness. No equality is
accepted from the caller independently of that relation. -/
theorem bindings_of_endpointAgrees
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (profile : Polynomial.Fe.SupportedProfile shape domains.fe)
    (constants : Poseidon2Schedule.Constants)
    (assignment : Nat → Nat)
    (constantWire : assignment 0 = 1)
    (input : KSplitNcEndpoints.Input polynomialInput domains)
    (message : OutputMessage shape)
    (transcriptValid :
      SymbolicDuplexSemantics.Valid
        input.transcript.transcriptBase constants assignment
        (KSplitNcTranscript.outputBuilder input.transcript))
    (authority :
      KSplitNcEndpoints.DecodedAuthority input assignment message)
    (sources : KSplitNcEndpointsSupport.InputsBelow input)
    (endpoints :
      KSplitNcOperational.EndpointAgrees
        profile constants assignment input.transcript message)
    (positive : 0 < input.frameBase) :
    KSplitNcEndpointsHonest.Bindings input assignment where
  feInitial := feInitial_binding profile constants assignment constantWire
    input message transcriptValid authority sources endpoints positive
  feTerminal := feTerminal_binding profile constants assignment constantWire
    input message transcriptValid authority sources endpoints positive
  nc := nc_binding profile constants assignment constantWire input message
    transcriptValid authority sources endpoints positive

/-- Model-proved honest completeness for the exact FE-initial, FE-terminal,
and NC endpoint rows. The witness is constructed from authoritative decoded
inputs, transcript replay, and the unchanged endpoint relation. -/
theorem rows_honest_of_endpointAgrees
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (profile : Polynomial.Fe.SupportedProfile shape domains.fe)
    (constants : Poseidon2Schedule.Constants)
    (assignment : Nat → Nat)
    (constantWire : assignment 0 = 1)
    (input : KSplitNcEndpoints.Input polynomialInput domains)
    (message : OutputMessage shape)
    (transcriptValid :
      SymbolicDuplexSemantics.Valid
        input.transcript.transcriptBase constants assignment
        (KSplitNcTranscript.outputBuilder input.transcript))
    (authority :
      KSplitNcEndpoints.DecodedAuthority input assignment message)
    (sources : KSplitNcEndpointsSupport.InputsBelow input)
    (endpoints :
      KSplitNcOperational.EndpointAgrees
        profile constants assignment input.transcript message)
    (positive : 0 < input.frameBase) :
    Satisfies (KSplitNcEndpoints.rows input)
      (KSplitNcEndpointsHonest.witness input assignment) := by
  exact KSplitNcEndpointsHonest.rows_honest_of_bindings
    input assignment positive constantWire sources
      (bindings_of_endpointAgrees profile constants assignment constantWire
        input message transcriptValid authority sources endpoints positive)

end Nightstream.Implementation.R1CS.Canonical.KSplitNcEndpointsSemanticHonest
