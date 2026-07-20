import Nightstream.Implementation.R1CS.Correspondence.PiCcsNc.Authority.DelayedResidual.FlatCombinedNc
import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.NcRefinement
import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.Primitives
import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.Transport
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Nc

/-!
Contract: execute the flat delayed Split-NC claimed-chain verifier with the
production Poseidon2 transcript machine and the authoritative raw running
assignments.

Assurance tier: model-level executable implementation semantics.

Owns: the delayed NC prologue with raw tags `8`, `9`, and `10`; transport of
the verifier-computed, generally nonzero delayed initial claim into that
prologue; exact five-extension-coefficient round serialization; derivation of
all fifteen fixed-270 model challenges from ordered transcript replay; the executable
combined-polynomial terminal check; and composition into the flat delayed-NC
soundness partition.

Does not own: derivation or domain separation of `producerBeta` and
`batchWeight` before this phase, generated-row enforcement of the ten parent
padding lanes, construction of `Sources.Data` from production witness columns,
recursive-state continuity, commitment-key coordinate alignment, Ajtai
binding, Rust/R1CS refinement, costs, or row-removal permission.

Emits constraints: no.

Authority boundary: the fixed-270 certificate contains only fifteen statically
five-coefficient messages. The verifier computes the delayed initial claim,
absorbs it in the NC prologue, derives every challenge, and recomputes the
terminal from `ProductionRawChildren.Fixed270.authoritativeRunningChildren`.
No challenge vector, child `CeClaim.y_zcol`, source-match proposition,
raw-child equality, or `ProjectionCheck.Accepted` premise crosses this API.

The current Rust path that enters NC with zero and reads a sidecar-derived
terminal does **not** refine this verifier. Rust conformance remains open until
that path uses this computed delayed initial and a raw-assignment terminal.

| Stage path | Mathematical obligation | Authority class | Outer boundary |
|---|---|---|---|
| `pi_ccs.nc.flat.delayed.verify.prologue` | absorb `[8]`, `[9]`, the computed initial pair, then `[10]` | computed | Poseidon2 row refinement |
| `pi_ccs.nc.flat.delayed.verify.round` | serialize exactly five extension coefficients and sample after absorption | verifier transcript | Rust transcript replay |
| `pi_ccs.nc.flat.delayed.verify.point.fixed270` | derive the fixed-270 model's fifteen ordered challenges | computed | live full-witness domain refinement |
| `pi_ccs.nc.flat.delayed.verify.terminal` | evaluate the combined polynomial over authoritative raw children | direct dataflow | witness-table decoder |
| `pi_ccs.nc.flat.delayed.verify.soundness` | acceptance yields NC truth and old-point binding or named events | security boundary | beta/weight schedule and padding rows |
-/

namespace Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual.FlatCombinedNc.Verifier

open Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedParentProjection
open Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual
open Nightstream.Implementation.R1CS.PiCcsTranscript
open Nightstream.Implementation.R1CS.PiCcsTranscript.Primitives
open Nightstream.Implementation.R1CS.PiCcsTranscript.Transport
open Nightstream.Implementation.R1CS.PiRlcChallenge.TranscriptMachine
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.Degree
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.MixingSoundness
open Nightstream.SuperNeo.SumCheck.Finite

private abbrev ops := ConcreteCarrier.extensionOps

private abbrev projectionOps :=
  Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism.Authority.DelayedPackedProjection.projectionOps

/-- The unweighted old-parent scalar is computed from the complete typed
parent vector. It is not a proof-carried sidecar scalar. -/
def parentProjection (parent : DelayedParent) (producerBeta : K) : K :=
  Nightstream.SuperNeo.ProjectionCheck.eval projectionOps
    (delayedParentActiveCoefficients parent) producerBeta

/-- The delayed NC initial claim checked by the claimed chain. Unlike the
ordinary production NC prologue, this value is not fixed to zero. -/
def claimedInitial
    (parent : DelayedParent) (producerBeta batchWeight : K) : K :=
  K.mul batchWeight (parentProjection parent producerBeta)

/-- Exact delayed NC prologue. `appendRaw` adds the authoritative length word
for each payload, so this is the current raw schedule with domain tag `8`,
initial tag `9`, the verifier-computed extension pair, and phase tag `10`. -/
def delayedNcPrologue (initialState : State) (initialClaim : K) : State :=
  let afterDomain := appendRaw initialState [wordField 8]
  let afterInitialTag := appendRaw afterDomain [wordField 9]
  let transported := toExtension initialClaim
  let afterInitial := appendRaw afterInitialTag
    [transported.c0, transported.c1]
  appendRaw afterInitial [wordField 10]

/-- The prologue exposes the exact tag and initial-pair order without treating
an incoming digest as authority. -/
theorem delayedNcPrologue_eq_raw_schedule
    (initialState : State) (initialClaim : K) :
    delayedNcPrologue initialState initialClaim =
      appendRaw
        (appendRaw
          (appendRaw
            (appendRaw initialState [wordField 8])
            [wordField 9])
          [(toExtension initialClaim).c0,
            (toExtension initialClaim).c1])
        [wordField 10] := by
  rfl

/-- Concrete delayed-NC transcript machine. Each round absorbs its complete
ten-base-field serialization before squeezing the next extension challenge. -/
def delayedMachine (initialClaim : K) :
    Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Transcript.Nc.Machine
      State where
  enterNc state := delayedNcPrologue state initialClaim
  absorbRound state message :=
    appendRaw state
      (PiCcsTranscript.SumCheck.roundFields
        (PiCcsTranscript.NcRefinement.toConcreteRound message))
  squeezeChallenge state :=
    let response := squeezeN state 2
    (toK (firstExtension response.2), response.1)

/-- Static NC messages contain exactly five extension coefficients. -/
@[simp] theorem serializedRound_extensionCount
    (message :
      Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Transcript.Nc.RoundMessage) :
    (PiCcsTranscript.NcRefinement.toConcreteRound message).coefficients.length =
      5 := by
  exact PiCcsTranscript.NcRefinement.toConcreteRound_coefficients_length message

/-- Five extension coefficients become exactly ten ordered base-field
elements before `appendRaw` contributes the message length word. -/
@[simp] theorem serializedRound_baseFieldCount
    (message :
      Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Transcript.Nc.RoundMessage) :
    (PiCcsTranscript.SumCheck.roundFields
      (PiCcsTranscript.NcRefinement.toConcreteRound message)).length = 10 := by
  exact PiCcsTranscript.NcRefinement.toConcreteRound_fields_length message

/-- The combined polynomial's old column is the parent point itself. The
length proof remains explicit because generated padding rows do not belong to
this transcript leaf. -/
def oldColumn
    (parent : DelayedParent)
    (pointLength : parent.sCol.length =
      ProductionRawChildren.Fixed270.domain.columnVariables) :
    CubePoint K ProductionRawChildren.Fixed270.domain.columnVariables :=
  { coordinates := parent.sCol, dimension := pointLength }

/-- Production flat combined polynomial over the authoritative raw running
assignments. No output-message evaluation vector is an input. -/
def combinedPolynomial
    {rowVariables freshCount runningCount matrixCount : Nat}
    (data : Data
      (ProductionRawChildren.Fixed270.semanticShape rowVariables freshCount
        runningCount matrixCount))
    (coins : Mixing.Coins ProductionRawChildren.Fixed270.domain)
    (radix : F) (parent : DelayedParent)
    (pointLength : parent.sCol.length =
      ProductionRawChildren.Fixed270.domain.columnVariables)
    (producerBeta batchWeight : K) : List K -> K :=
  FlatCombinedNc.sumcheckPolynomial
    (Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.PiCcsDomain.domain_covers
      rowVariables freshCount runningCount matrixCount)
    data coins radix
    (ProductionRawChildren.Fixed270.authoritativeRunningChildren data)
    producerBeta batchWeight (oldColumn parent pointLength)

/-- Verifier-derived flat NC point and successor state. The proof contains no
challenge vector field. -/
def execution
    (initialState : State) (parent : DelayedParent)
    (producerBeta batchWeight : K)
    (certificate :
      Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Transcript.Nc.Certificate
        ProductionRawChildren.Fixed270.domain) :
    Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Nc.Execution
      ProductionRawChildren.Fixed270.domain State :=
  Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Nc.derive
    (delayedMachine (claimedInitial parent producerBeta batchWeight))
    initialState certificate

/-- The transcript-derived point contains the fixed-270 model shape: nine
column coordinates followed by six lane coordinates. -/
theorem execution_challengeCount
    (initialState : State) (parent : DelayedParent)
    (producerBeta batchWeight : K)
    (certificate :
      Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Transcript.Nc.Certificate
        ProductionRawChildren.Fixed270.domain) :
    (execution initialState parent producerBeta batchWeight certificate).point.coordinates.length =
      15 := by
  calc
    _ = ProductionRawChildren.Fixed270.domain.columnVariables +
        ProductionRawChildren.Fixed270.domain.laneVariables :=
      (execution initialState parent producerBeta batchWeight certificate).point.coordinates_length
    _ = 15 := FlatCombinedNc.fixed270_roundCount

/-- The terminal is recomputed by evaluating the combined polynomial at the
point derived from these exact messages and this incoming transcript state. -/
def combinedTerminal
    {rowVariables freshCount runningCount matrixCount : Nat}
    (data : Data
      (ProductionRawChildren.Fixed270.semanticShape rowVariables freshCount
        runningCount matrixCount))
    (coins : Mixing.Coins ProductionRawChildren.Fixed270.domain)
    (radix : F) (parent : DelayedParent)
    (pointLength : parent.sCol.length =
      ProductionRawChildren.Fixed270.domain.columnVariables)
    (producerBeta batchWeight : K) (initialState : State)
    (certificate :
      Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Transcript.Nc.Certificate
        ProductionRawChildren.Fixed270.domain) : K :=
  combinedPolynomial data coins radix parent pointLength producerBeta
    batchWeight
    (execution initialState parent producerBeta batchWeight certificate).point.coordinates

/-- Logical acceptance of the transcript-bound delayed claimed chain. The
terminal is a verifier computation, while the certificate contributes only
its exact-width round messages. -/
def Accepted
    {rowVariables freshCount runningCount matrixCount : Nat}
    (data : Data
      (ProductionRawChildren.Fixed270.semanticShape rowVariables freshCount
        runningCount matrixCount))
    (coins : Mixing.Coins ProductionRawChildren.Fixed270.domain)
    (radix : F) (parent : DelayedParent)
    (pointLength : parent.sCol.length =
      ProductionRawChildren.Fixed270.domain.columnVariables)
    (producerBeta batchWeight : K) (initialState : State)
    (certificate :
      Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Transcript.Nc.Certificate
        ProductionRawChildren.Fixed270.domain) : Prop :=
  Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Transcript.Nc.Accepted
    (delayedMachine (claimedInitial parent producerBeta batchWeight))
    initialState (claimedInitial parent producerBeta batchWeight)
    (combinedTerminal data coins radix parent pointLength producerBeta
      batchWeight initialState certificate)
    certificate

/-- Executable verifier for the same transcript-derived point and raw-source
terminal. -/
def check
    {rowVariables freshCount runningCount matrixCount : Nat}
    (data : Data
      (ProductionRawChildren.Fixed270.semanticShape rowVariables freshCount
        runningCount matrixCount))
    (coins : Mixing.Coins ProductionRawChildren.Fixed270.domain)
    (radix : F) (parent : DelayedParent)
    (pointLength : parent.sCol.length =
      ProductionRawChildren.Fixed270.domain.columnVariables)
    (producerBeta batchWeight : K) (initialState : State)
    (certificate :
      Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Transcript.Nc.Certificate
        ProductionRawChildren.Fixed270.domain) : Bool :=
  Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Transcript.Nc.check
    (delayedMachine (claimedInitial parent producerBeta batchWeight))
    initialState (claimedInitial parent producerBeta batchWeight)
    (combinedTerminal data coins radix parent pointLength producerBeta
      batchWeight initialState certificate)
    certificate

/-- The executable delayed verifier is exactly its logical acceptance
relation. -/
theorem check_eq_true_iff_accepted
    {rowVariables freshCount runningCount matrixCount : Nat}
    (data : Data
      (ProductionRawChildren.Fixed270.semanticShape rowVariables freshCount
        runningCount matrixCount))
    (coins : Mixing.Coins ProductionRawChildren.Fixed270.domain)
    (radix : F) (parent : DelayedParent)
    (pointLength : parent.sCol.length =
      ProductionRawChildren.Fixed270.domain.columnVariables)
    (producerBeta batchWeight : K) (initialState : State)
    (certificate :
      Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Transcript.Nc.Certificate
        ProductionRawChildren.Fixed270.domain) :
    check data coins radix parent pointLength producerBeta batchWeight
        initialState certificate = true <->
      Accepted data coins radix parent pointLength producerBeta batchWeight
        initialState certificate := by
  exact
    Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Transcript.Nc.check_eq_true_iff_accepted
      (delayedMachine (claimedInitial parent producerBeta batchWeight))
      initialState (claimedInitial parent producerBeta batchWeight)
      (combinedTerminal data coins radix parent pointLength producerBeta
        batchWeight initialState certificate)
      certificate

/-- Transcript-bound logical acceptance refines the semantic fixed-phase
relation over the same combined polynomial. This theorem derives the
challenge vector; it does not accept one as a premise. -/
theorem accepted_implies_fixedPhaseAccepted
    {rowVariables freshCount runningCount matrixCount : Nat}
    (data : Data
      (ProductionRawChildren.Fixed270.semanticShape rowVariables freshCount
        runningCount matrixCount))
    (coins : Mixing.Coins ProductionRawChildren.Fixed270.domain)
    (radix : F) (parent : DelayedParent)
    (pointLength : parent.sCol.length =
      ProductionRawChildren.Fixed270.domain.columnVariables)
    (producerBeta batchWeight : K) (initialState : State)
    (certificate :
      Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Transcript.Nc.Certificate
        ProductionRawChildren.Fixed270.domain)
    (accepted : Accepted data coins radix parent pointLength producerBeta
      batchWeight initialState certificate) :
    FixedPhase.Accepted ops.toOps
      (combinedPolynomial data coins radix parent pointLength producerBeta
        batchWeight)
      (claimedInitial parent producerBeta batchWeight)
      (execution initialState parent producerBeta batchWeight certificate).point.coordinates
      certificate.toSumCheck := by
  have coordinatesEq :
      (execution initialState parent producerBeta batchWeight certificate).point.coordinates =
        (Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Transcript.Nc.derive
          (delayedMachine (claimedInitial parent producerBeta batchWeight))
          initialState certificate).challengePoint.coordinates := by
    exact
      Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Nc.derive_point_coordinates
        (delayedMachine (claimedInitial parent producerBeta batchWeight))
        initialState certificate
  unfold Accepted at accepted
  unfold Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Transcript.Nc.Accepted at accepted
  unfold Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.Nc.Accepted at accepted
  dsimp only at accepted
  rw [<- coordinatesEq] at accepted
  unfold FixedPhase.Accepted
  simpa [combinedTerminal] using accepted

/-- Residual-weight collision after excluding the explicit zero-weight
degeneration. -/
def NonzeroResidualWeightRoot
    {rowVariables freshCount runningCount matrixCount : Nat}
    (data : Data
      (ProductionRawChildren.Fixed270.semanticShape rowVariables freshCount
        runningCount matrixCount))
    (coins : Mixing.Coins ProductionRawChildren.Fixed270.domain)
    (radix : F) (parent : DelayedParent)
    (pointLength : parent.sCol.length =
      ProductionRawChildren.Fixed270.domain.columnVariables)
    (producerBeta batchWeight : K) : Prop :=
  FlatCombinedNc.ResidualWeightRoot
      (Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.PiCcsDomain.domain_covers
        rowVariables freshCount runningCount matrixCount)
      data coins radix
      (ProductionRawChildren.Fixed270.authoritativeRunningChildren data)
      producerBeta batchWeight (parentProjection parent producerBeta)
      (oldColumn parent pointLength) /\
    batchWeight ≠ K.zero

/-- Transcript-derived SumCheck collision for this exact combined polynomial,
initial claim, point, and certificate. -/
def SumCheckCollision
    {rowVariables freshCount runningCount matrixCount : Nat}
    (data : Data
      (ProductionRawChildren.Fixed270.semanticShape rowVariables freshCount
        runningCount matrixCount))
    (coins : Mixing.Coins ProductionRawChildren.Fixed270.domain)
    (radix : F) (parent : DelayedParent)
    (pointLength : parent.sCol.length =
      ProductionRawChildren.Fixed270.domain.columnVariables)
    (producerBeta batchWeight : K) (initialState : State)
    (certificate :
      Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Transcript.Nc.Certificate
        ProductionRawChildren.Fixed270.domain)
    (challengeSetSize : Nat) : Prop :=
  exists round, FixedPhase.BadChallenge ops.toOps
    (combinedPolynomial data coins radix parent pointLength producerBeta
      batchWeight)
    ncSumcheckDegreeBound challengeSetSize
    (claimedInitial parent producerBeta batchWeight)
    (execution initialState parent producerBeta batchWeight certificate).point.coordinates
    certificate.toSumCheck round

/-- Full transcript-bound flat delayed-NC soundness partition. The
`batchWeight = 0` degeneration is a dedicated top-level branch; the residual
root branch therefore records nonzeroness explicitly. `parentPadding` is the
remaining generated-row boundary and `producerBeta`/`batchWeight` remain
outer transcript-schedule inputs to this leaf. -/
theorem accepted_implies_truth_and_oldPointRelation_or_badEvent
    (baseNoZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    (sevenNonresidue : ConcreteCarrier.SevenProjectiveNonresidue)
    {rowVariables freshCount runningCount matrixCount : Nat}
    (data : Data
      (ProductionRawChildren.Fixed270.semanticShape rowVariables freshCount
        runningCount matrixCount))
    (coins : Mixing.Coins ProductionRawChildren.Fixed270.domain)
    (radix : F) (parent : DelayedParent)
    (pointLength : parent.sCol.length =
      ProductionRawChildren.Fixed270.domain.columnVariables)
    (parentPadding : forall lane, ringDegree <= lane ->
      lane < ProductionRawChildren.Fixed270.implementationShape.laneDomain ->
      parent.yZcol lane = K.zero)
    (producerBeta batchWeight : K) (initialState : State)
    (certificate :
      Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Transcript.Nc.Certificate
        ProductionRawChildren.Fixed270.domain)
    (challengeSetSize : Nat)
    (accepted : Accepted data coins radix parent pointLength producerBeta
      batchWeight initialState certificate) :
    (Semantics.Nc.Truth data /\
      OldPointSumcheckRelation ProductionRawChildren.Fixed270.implementationShape
        radix parent
        (ProductionRawChildren.Fixed270.authoritativeRunningChildren data)) \/
    SelectorRoot
      (Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.PiCcsDomain.domain_covers
        rowVariables freshCount runningCount matrixCount) data coins \/
    GammaRoot
      (Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.PiCcsDomain.domain_covers
        rowVariables freshCount runningCount matrixCount) data coins \/
    SplitV1GammaZero
      (Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.PiCcsDomain.domain_covers
        rowVariables freshCount runningCount matrixCount) data coins \/
    batchWeight = K.zero \/
    NonzeroResidualWeightRoot data coins radix parent pointLength producerBeta
      batchWeight \/
    Nightstream.SuperNeo.ProjectionCheck.BadRoot projectionOps
      (projectionIdentity ProductionRawChildren.Fixed270.implementationShape
        radix (ProductionRawChildren.Fixed270.authoritativeRunningChildren data)
        parent.sCol (delayedParentActiveCoefficients parent) producerBeta) \/
    SumCheckCollision data coins radix parent pointLength producerBeta
      batchWeight initialState certificate challengeSetSize := by
  by_cases batchWeightZero : batchWeight = K.zero
  · exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inl batchWeightZero))))
  · have fixedAccepted := accepted_implies_fixedPhaseAccepted data coins radix
      parent pointLength producerBeta batchWeight initialState certificate
      accepted
    rcases FlatCombinedNc.accepted_implies_truth_and_oldPointRelation_or_badEvent
        baseNoZeroDivisors sevenNonresidue data coins radix parent pointLength
        parentPadding producerBeta batchWeight
        (execution initialState parent producerBeta batchWeight certificate).point
        certificate.toSumCheck challengeSetSize fixedAccepted with
      semantic | selectorRoot | gammaRoot | gammaZero | residualRoot |
        projectionRoot | sumcheckCollision
    · exact Or.inl semantic
    · exact Or.inr (Or.inl selectorRoot)
    · exact Or.inr (Or.inr (Or.inl gammaRoot))
    · exact Or.inr (Or.inr (Or.inr (Or.inl gammaZero)))
    · exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inr
        (Or.inl <| And.intro residualRoot batchWeightZero)))))
    · exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inr
        (Or.inr (Or.inl projectionRoot))))))
    · exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inr
        (Or.inr (Or.inr sumcheckCollision))))))

end Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual.FlatCombinedNc.Verifier
