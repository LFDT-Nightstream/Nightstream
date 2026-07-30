import Nightstream.Implementation.R1CS.Canonical.KSplitNcEndpointsSupport
import Nightstream.Implementation.R1CS.Canonical.KSplitNcFeInitialHonest
import Nightstream.Implementation.R1CS.Canonical.KSplitNcFeTerminalProductsHonest
import Nightstream.Implementation.R1CS.Canonical.KSplitNcNcEndpointHonest

/-!
Contract: constructive completeness for the three selected Split-NC endpoint
programs.

Owns the sequential witness and preservation proof for FE-initial,
FE-terminal, and NC endpoint rows in their compact contiguous allocation.
`Bindings` is only the local endpoint-equality interface needed while the
three calculations are composed.  A selected-verifier theorem must construct
it from the unchanged frozen endpoint equations; it is not an acceptance
premise and is not a public call contract.
-/

set_option autoImplicit false
set_option maxRecDepth 100000

namespace Nightstream.Implementation.R1CS.Canonical.KSplitNcEndpointsHonest

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.KCompositeRowSupport
open Nightstream.Implementation.R1CS.Canonical.KMul
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.TranscriptAuthority.BlockLane

def afterFeInitial
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (input : KSplitNcEndpoints.Input polynomialInput domains)
    (assignment : Nat → Nat) : Nat → Nat :=
  KSplitNcFeInitialHonest.witness
    (KSplitNcEndpoints.feInitialInput input) assignment

def afterFeTerminal
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (input : KSplitNcEndpoints.Input polynomialInput domains)
    (assignment : Nat → Nat) : Nat → Nat :=
  KSplitNcFeTerminalProductsHonest.witness
    (KSplitNcEndpoints.feTerminalInput input)
    (afterFeInitial input assignment)

def witness
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (input : KSplitNcEndpoints.Input polynomialInput domains)
    (assignment : Nat → Nat) : Nat → Nat :=
  KSplitNcNcEndpointHonest.witness
    (KSplitNcEndpoints.ncInput input)
    (afterFeTerminal input assignment)

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

private theorem feTerminal_le_ncBase
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (input : KSplitNcEndpoints.Input polynomialInput domains) :
    (KSplitNcEndpoints.feTerminalInput input).frameBase ≤
      (KSplitNcEndpoints.ncInput input).frameBase := by
  change KSplitNcEndpoints.feTerminalBase input ≤
    KSplitNcEndpoints.ncBase input
  unfold KSplitNcEndpoints.ncBase
  omega

private theorem rowsBelow_append
    {left right : List Row} {boundary : Nat}
    (leftBelow : RowsBelow left boundary)
    (rightBelow : RowsBelow right boundary) :
    RowsBelow (left ++ right) boundary := by
  intro row member column mentioned
  exact (List.mem_append.1 member).elim
    (fun inLeft => leftBelow row inLeft column mentioned)
    (fun inRight => rightBelow row inRight column mentioned)

private theorem satisfies_append
    {left right : List Row} {assignment : Nat → Nat}
    (leftSatisfied : Satisfies left assignment)
    (rightSatisfied : Satisfies right assignment) :
    Satisfies (left ++ right) assignment := by
  intro row member
  exact (List.mem_append.1 member).elim
    (leftSatisfied row) (rightSatisfied row)

private theorem feInitialRows_below_feTerminalBase
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (input : KSplitNcEndpoints.Input polynomialInput domains)
    (positive : 0 < input.frameBase)
    (sources : KSplitNcEndpointsSupport.InputsBelow input) :
    RowsBelow
      (KSplitNcFeInitial.rows (KSplitNcEndpoints.feInitialInput input))
      (KSplitNcEndpoints.feTerminalInput input).frameBase := by
  apply KSplitNcEndpointSupport.feInitial_rows_below
  · exact Nat.lt_of_lt_of_le positive (outer_le_feTerminalBase input)
  · exact sources.feInitialGamma
  · exact sources.feInitialAlpha
  · exact sources.feInitialClaims
  · exact carried_mono sources.feInitialEndpoint
      (outer_le_feTerminalBase input)
  · change input.frameBase +
        KSplitNcFeInitial.allocationWidth
          (KSplitNcEndpoints.feInitialInput input) ≤
      KSplitNcEndpoints.feTerminalBase input
    unfold KSplitNcEndpoints.feTerminalBase
    exact Nat.le_refl _

private theorem feInitialRows_below_ncBase
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (input : KSplitNcEndpoints.Input polynomialInput domains)
    (positive : 0 < input.frameBase)
    (sources : KSplitNcEndpointsSupport.InputsBelow input) :
    RowsBelow
      (KSplitNcFeInitial.rows (KSplitNcEndpoints.feInitialInput input))
      (KSplitNcEndpoints.ncInput input).frameBase := by
  intro row member column mentioned
  exact Nat.lt_of_lt_of_le
    (feInitialRows_below_feTerminalBase input positive sources
      row member column mentioned)
    (feTerminal_le_ncBase input)

private def feTerminalSources
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (input : KSplitNcEndpoints.Input polynomialInput domains)
    (sources : KSplitNcEndpointsSupport.InputsBelow input) :
    KSplitNcFeTerminalProductsHonest.SourceBounds
      (KSplitNcEndpoints.feTerminalInput input) where
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

private theorem feTerminalRows_below_ncBase
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (input : KSplitNcEndpoints.Input polynomialInput domains)
    (positive : 0 < input.frameBase)
    (sources : KSplitNcEndpointsSupport.InputsBelow input) :
    RowsBelow
      (KSplitNcFeTerminal.rows
        (KSplitNcEndpoints.feTerminalInput input))
      (KSplitNcEndpoints.ncInput input).frameBase := by
  apply KSplitNcEndpointSupport.feTerminal_rows_below
  · exact Nat.lt_of_lt_of_le positive (outer_le_ncBase input)
  · exact (feTerminalSources input sources).gamma
  · exact (feTerminalSources input sources).alpha
  · exact (feTerminalSources input sources).betaA
  · exact (feTerminalSources input sources).betaR
  · exact (feTerminalSources input sources).pointLane
  · exact (feTerminalSources input sources).pointRow
  · exact (feTerminalSources input sources).priorPoint
  · exact (feTerminalSources input sources).message
  · exact carried_mono sources.feTerminalEndpoint (outer_le_ncBase input)
  · change KSplitNcEndpoints.feTerminalBase input +
        KSplitNcFeTerminal.allocationWidth
          (KSplitNcEndpoints.feTerminalInput input) ≤
      KSplitNcEndpoints.ncBase input
    unfold KSplitNcEndpoints.ncBase
    exact Nat.le_refl _

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

/-- The three local endpoint equalities in the assignments where their
respective calculation witnesses have just been constructed. -/
structure Bindings
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (input : KSplitNcEndpoints.Input polynomialInput domains)
    (assignment : Nat → Nat) : Prop where
  feInitial :
    KSplitNcFeInitialHonest.Binding
      (KSplitNcEndpoints.feInitialInput input) assignment
  feTerminal :
    KSplitNcFeTerminalProductsHonest.TerminalBinding
      (KSplitNcEndpoints.feTerminalInput input)
      (afterFeInitial input assignment)
  nc :
    KSplitNcNcEndpointHonest.EndpointBinding
      (KSplitNcEndpoints.ncInput input)
      (afterFeTerminal input assignment)

theorem afterFeInitial_off_source
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (input : KSplitNcEndpoints.Input polynomialInput domains)
    (assignment : Nat → Nat)
    (column : Nat) (below : column < input.frameBase) :
    afterFeInitial input assignment column = assignment column := by
  unfold afterFeInitial
  exact KSplitNcFeInitialHonest.witness_off_block
    (KSplitNcEndpoints.feInitialInput input) assignment column below

theorem afterFeTerminal_off_source
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (input : KSplitNcEndpoints.Input polynomialInput domains)
    (assignment : Nat → Nat)
    (column : Nat) (below : column < input.frameBase) :
    afterFeTerminal input assignment column = assignment column := by
  unfold afterFeTerminal
  rw [KSplitNcFeTerminalProductsHonest.witness_off_source
    (KSplitNcEndpoints.feTerminalInput input)
    (afterFeInitial input assignment) column
    (Nat.lt_of_lt_of_le below (outer_le_feTerminalBase input))]
  exact afterFeInitial_off_source input assignment column below

theorem witness_off_source
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (input : KSplitNcEndpoints.Input polynomialInput domains)
    (assignment : Nat → Nat)
    (column : Nat) (below : column < input.frameBase) :
    witness input assignment column = assignment column := by
  unfold witness
  rw [KSplitNcNcEndpointHonest.witness_off_source
    (KSplitNcEndpoints.ncInput input)
    (afterFeTerminal input assignment) column
    (Nat.lt_of_lt_of_le below (outer_le_ncBase input))]
  exact afterFeTerminal_off_source input assignment column below

theorem rows_honest_of_bindings
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (input : KSplitNcEndpoints.Input polynomialInput domains)
    (assignment : Nat → Nat)
    (positive : 0 < input.frameBase)
    (constantWire : assignment 0 = 1)
    (sources : KSplitNcEndpointsSupport.InputsBelow input)
    (bindings : Bindings input assignment) :
    Satisfies (KSplitNcEndpoints.rows input) (witness input assignment) := by
  have feInitialSatisfied :
      Satisfies
        (KSplitNcFeInitial.rows (KSplitNcEndpoints.feInitialInput input))
        (afterFeInitial input assignment) := by
    exact KSplitNcFeInitialHonest.rows_honest_of_binding
      (KSplitNcEndpoints.feInitialInput input) assignment positive
      constantWire sources.feInitialGamma sources.feInitialAlpha
      sources.feInitialClaims bindings.feInitial
  have afterInitialOne :
      afterFeInitial input assignment 0 = 1 := by
    unfold afterFeInitial
    rw [KSplitNcFeInitialHonest.witness_off_block
      (KSplitNcEndpoints.feInitialInput input) assignment 0 positive]
    exact constantWire
  have feTerminalPositive :
      0 < (KSplitNcEndpoints.feTerminalInput input).frameBase :=
    Nat.lt_of_lt_of_le positive (outer_le_feTerminalBase input)
  have feTerminalSatisfied :
      Satisfies
        (KSplitNcFeTerminal.rows
          (KSplitNcEndpoints.feTerminalInput input))
        (afterFeTerminal input assignment) := by
    exact KSplitNcFeTerminalProductsHonest.rows_honest_of_binding
      (KSplitNcEndpoints.feTerminalInput input)
      (afterFeInitial input assignment) feTerminalPositive afterInitialOne
      (feTerminalSources input sources) bindings.feTerminal
  have feInitialPreserved :
      Satisfies
        (KSplitNcFeInitial.rows (KSplitNcEndpoints.feInitialInput input))
        (afterFeTerminal input assignment) := by
    apply KHornerSupport.satisfies_extend _
      (afterFeInitial input assignment) (afterFeTerminal input assignment)
    · intro row member column mentioned
      exact
        (KSplitNcFeTerminalProductsHonest.witness_off_source
          (KSplitNcEndpoints.feTerminalInput input)
          (afterFeInitial input assignment) column
          (feInitialRows_below_feTerminalBase input positive sources
            row member column mentioned)).symm
    · exact feInitialSatisfied
  have prefixSatisfied :
      Satisfies
        (KSplitNcFeInitial.rows (KSplitNcEndpoints.feInitialInput input) ++
          KSplitNcFeTerminal.rows
            (KSplitNcEndpoints.feTerminalInput input))
        (afterFeTerminal input assignment) :=
    satisfies_append feInitialPreserved feTerminalSatisfied
  have afterTerminalOne :
      afterFeTerminal input assignment 0 = 1 := by
    unfold afterFeTerminal
    exact KSplitNcFeTerminalProductsHonest.witness_constantWire
      (KSplitNcEndpoints.feTerminalInput input)
      (afterFeInitial input assignment) feTerminalPositive afterInitialOne
  have ncPositive :
      0 < (KSplitNcEndpoints.ncInput input).frameBase :=
    Nat.lt_of_lt_of_le positive (outer_le_ncBase input)
  have ncSatisfied :
      Satisfies
        (KSplitNcNcEndpoint.rows (KSplitNcEndpoints.ncInput input))
        (witness input assignment) := by
    exact KSplitNcNcEndpointHonest.rows_honest_of_binding
      (KSplitNcEndpoints.ncInput input)
      (afterFeTerminal input assignment) ncPositive afterTerminalOne
      (ncSources input sources) bindings.nc
  have prefixBelow :
      RowsBelow
        (KSplitNcFeInitial.rows (KSplitNcEndpoints.feInitialInput input) ++
          KSplitNcFeTerminal.rows
            (KSplitNcEndpoints.feTerminalInput input))
        (KSplitNcEndpoints.ncInput input).frameBase :=
    rowsBelow_append
      (feInitialRows_below_ncBase input positive sources)
      (feTerminalRows_below_ncBase input positive sources)
  have prefixPreserved :
      Satisfies
        (KSplitNcFeInitial.rows (KSplitNcEndpoints.feInitialInput input) ++
          KSplitNcFeTerminal.rows
            (KSplitNcEndpoints.feTerminalInput input))
        (witness input assignment) := by
    apply KHornerSupport.satisfies_extend _
      (afterFeTerminal input assignment) (witness input assignment)
    · intro row member column mentioned
      exact
        (KSplitNcNcEndpointHonest.witness_off_source
          (KSplitNcEndpoints.ncInput input)
          (afterFeTerminal input assignment) column
          (prefixBelow row member column mentioned)).symm
    · exact prefixSatisfied
  have combined :=
    satisfies_append prefixPreserved ncSatisfied
  simpa [KSplitNcEndpoints.rows, KSplitNcEndpoints.rowGroups,
    List.append_assoc] using combined

end Nightstream.Implementation.R1CS.Canonical.KSplitNcEndpointsHonest
