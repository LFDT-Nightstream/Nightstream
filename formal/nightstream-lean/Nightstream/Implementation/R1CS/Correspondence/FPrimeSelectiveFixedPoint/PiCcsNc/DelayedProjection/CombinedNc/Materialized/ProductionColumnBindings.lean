import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.ProductionMessageAcceptance
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.ProductionTerminalBridge
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.RoundArtifact

/-!
Generated source-column contract for production combined-NC messages.

Owns: the exact handoff from the generated post-`Pi_DEC` source columns to
the production claims-level NC message surface.  The contract reads the
boundary columns exported in `Metadata.boundary` and, for every generated
round, the five coefficient pairs, challenge pair, and claimed-value pairs in
its `RawRoundMap`.  Generic kernel lemmas use `roundMapValid` to connect those
named source columns to the isolated 43-column round program.  The resulting
theorem constructs `ProductionMessageAcceptance.ExactDataflow`.

Does not own: proof that an execution populates these columns, transcript
scheduling, source-row satisfaction, parent or raw-child authority,
commitment binding, digests, costs, or row removal.  Terminal result equality
is derived from source-row consequences and exact terminal input columns; no
digest or caller-supplied terminal result is semantic authority.

Emits constraints: none.

The sole executable certificate evaluates exactly the 25 proof-free
`RawRoundMap` records.  Each has 43 mapped columns, 28 allocated columns, and
five coefficient pairs.  It projects only the first claim pair, the ordered
25 challenge pairs, and the last claim pair, together with the proof-free
boundary record.  It evaluates no assignment, decoded value, or proof-bearing
structure.

Assurance tier: artifact-checked for the fixed generated column schedule;
all assignment-to-production-value equalities remain an explicit concrete
Rust/R1CS refinement contract.
-/

/-!
| Stable stage path | Obligation | Authority class |
|---|---|---|
| `f_prime.pi_ccs_nc.delayed.combined.production_column_bindings` | Bind production message, challenge, selector, and terminal values to their generated columns. | direct dataflow |

-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.ProductionColumnBindings

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.SumCheck.Finite
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.ProjectionProgram
open Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc
open Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized

universe uState

/-- Read one generated quadratic-extension source-column pair and transport
it to the independently named production semantic carrier. -/
def sourceValue (assignment : Nat -> Nat) (columns : RawKColumns) :
    Nightstream.SuperNeo.Concrete.K :=
  ProductionMessageAcceptance.toConcreteK
    (Semantics.rawKColumnsValue columns assignment)

/-- First claimed-value pair, kept as a proof-free column projection. -/
def firstClaimInColumns : List RawRoundMap -> Option RawKColumns
  | [] => none
  | round :: _ => some round.claimInColumns

/-- Last claimed-value pair, kept as a proof-free column projection. -/
def lastClaimOutColumns : List RawRoundMap -> Option RawKColumns
  | [] => none
  | [round] => some round.claimOutColumns
  | _ :: round :: rounds => lastClaimOutColumns (round :: rounds)

/-- The generated round schedule starts at the boundary claimed-in pair,
uses the boundary block challenges before the lane challenges, and terminates
at the boundary final-sum pair. -/
def GeneratedColumnSchedule : Prop :=
  firstClaimInColumns RoundMaps.values =
      some Metadata.boundary.claimedInitialColumns /\
    RoundMaps.values.map RawRoundMap.challengeColumns =
      Metadata.boundary.blockPointColumns ++
        Metadata.boundary.lanePointColumns /\
    lastClaimOutColumns RoundMaps.values =
      some Metadata.boundary.finalSumColumns

instance : Decidable GeneratedColumnSchedule := by
  unfold GeneratedColumnSchedule
  infer_instance

set_option maxRecDepth 100000 in
theorem generatedColumnSchedule : GeneratedColumnSchedule := by
  native_decide

/-- The named fields of a valid round map are exactly the columns obtained by
applying its complete affine map to the isolated round vocabulary. -/
structure NamedColumnsValid (round : RawRoundMap) : Prop where
  coefficients :
    round.coefficientColumns =
      Decoder.expectedCoefficientColumns round.columnMap
  challenge :
    round.challengeColumns =
      Decoder.expectedChallengeColumns round.columnMap
  claimIn :
    round.claimInColumns = Decoder.expectedClaimInColumns round.columnMap
  claimOut :
    round.claimOutColumns = Decoder.expectedClaimOutColumns round.columnMap

theorem namedColumnsValid_of_roundMapValid {round : RawRoundMap}
    (valid : Decoder.roundMapValid round) : NamedColumnsValid round := by
  rcases valid with
    ⟨schema, sourceRows, sourceColumns, mapsOne, roundIndex, rowRange,
      rowLength, firstAllocated, allocatedLength, allocatedNodup,
      allocatedInRange, firstAllocatedMem, coefficientLength,
      coefficientInRange, challengeInRange, claimInRange, claimOutInRange,
      columnMapLength, columnMapNodup, columnMapInRange, allocatedColumns,
      coefficientColumns, challengeColumns, claimInColumns,
      claimOutColumns⟩
  exact
    { coefficients := coefficientColumns
      challenge := challengeColumns
      claimIn := claimInColumns
      claimOut := claimOutColumns }

private theorem coefficientValues_eq_named
    (round : RawRoundMap) (assignment : Nat -> Nat)
    (valid : Decoder.roundMapValid round) :
    ProductionRound.coefficientValues
        (ClaimedChain.mappedAssignment round assignment) =
      round.coefficientColumns.map
        (fun columns => Semantics.rawKColumnsValue columns assignment) := by
  have named := namedColumnsValid_of_roundMapValid valid
  rw [named.coefficients]
  rfl

private theorem challengeValue_eq_named
    (round : RawRoundMap) (assignment : Nat -> Nat)
    (valid : Decoder.roundMapValid round) :
    ClaimedChain.challenge round assignment =
      Semantics.rawKColumnsValue round.challengeColumns assignment := by
  have named := namedColumnsValid_of_roundMapValid valid
  unfold ClaimedChain.challenge ClaimedChain.mappedAssignment
    ProductionRound.challengeValue
  rw [named.challenge]
  rfl

private theorem claimInValue_eq_named
    (round : RawRoundMap) (assignment : Nat -> Nat)
    (valid : Decoder.roundMapValid round) :
    ClaimedChain.claimIn round assignment =
      Semantics.rawKColumnsValue round.claimInColumns assignment := by
  have named := namedColumnsValid_of_roundMapValid valid
  unfold ClaimedChain.claimIn ClaimedChain.mappedAssignment
    ProductionRound.claimInValue
  rw [named.claimIn]
  rfl

private theorem claimOutValue_eq_named
    (round : RawRoundMap) (assignment : Nat -> Nat)
    (valid : Decoder.roundMapValid round) :
    ClaimedChain.claimOut round assignment =
      Semantics.rawKColumnsValue round.claimOutColumns assignment := by
  have named := namedColumnsValid_of_roundMapValid valid
  unfold ClaimedChain.claimOut ClaimedChain.mappedAssignment
    ProductionRound.claimOutValue
  rw [named.claimOut]
  rfl

private theorem transportedRoundCoefficients_eq_named
    (round : RawRoundMap) (assignment : Nat -> Nat)
    (valid : Decoder.roundMapValid round) :
    (ProductionMessageAcceptance.mapFixedPolynomial
        ProductionMessageAcceptance.toConcreteK
        (ClaimedChain.roundMessage round assignment)).coefficients =
      round.coefficientColumns.map (sourceValue assignment) := by
  change
    (ProductionRound.coefficientValues
      (ClaimedChain.mappedAssignment round assignment)).map
        ProductionMessageAcceptance.toConcreteK =
      round.coefficientColumns.map (sourceValue assignment)
  rw [coefficientValues_eq_named round assignment valid]
  rw [List.map_map]
  apply List.map_congr_left
  intro columns member
  rfl

private theorem fixedPolynomial_eq_of_coefficients_eq
    {left right : FixedPolynomial Nightstream.SuperNeo.Concrete.K
      ProductionRound.degree}
    (equal : left.coefficients = right.coefficients) : left = right := by
  cases left with
  | mk leftCoefficients leftLength =>
      cases right with
      | mk rightCoefficients rightLength =>
          simp only at equal
          cases equal
          rfl

private theorem transportedRoundMessages_eq
    {rounds : List RawRoundMap}
    {messages : List
      (FixedPolynomial Nightstream.SuperNeo.Concrete.K
        ProductionRound.degree)}
    (assignment : Nat -> Nat)
    (valid : forall round, round ∈ rounds -> Decoder.roundMapValid round)
    (bindings :
      rounds.map (fun round =>
          round.coefficientColumns.map (sourceValue assignment)) =
        messages.map FixedPolynomial.coefficients) :
    (rounds.map fun round =>
        ProductionMessageAcceptance.mapFixedPolynomial
          ProductionMessageAcceptance.toConcreteK
          (ClaimedChain.roundMessage round assignment)) = messages := by
  induction rounds generalizing messages with
  | nil =>
      cases messages with
      | nil => rfl
      | cons message messages => simp at bindings
  | cons round rounds inductionHypothesis =>
      cases messages with
      | nil => simp at bindings
      | cons message messages =>
          simp only [List.map_cons, List.cons.injEq] at bindings
          have roundValid : Decoder.roundMapValid round :=
            valid round (by simp)
          have tailValid :
              forall candidate, candidate ∈ rounds ->
                Decoder.roundMapValid candidate := by
            intro candidate member
            exact valid candidate (by simp [member])
          have headEquality :
              ProductionMessageAcceptance.mapFixedPolynomial
                  ProductionMessageAcceptance.toConcreteK
                  (ClaimedChain.roundMessage round assignment) =
                message := by
            exact fixedPolynomial_eq_of_coefficients_eq
              ((transportedRoundCoefficients_eq_named round assignment
                roundValid).trans bindings.1)
          simp only [List.map_cons]
          rw [headEquality,
            inductionHypothesis tailValid bindings.2]

private theorem transportedChallenges_eq_named
    (rounds : List RawRoundMap) (assignment : Nat -> Nat)
    (valid : forall round, round ∈ rounds -> Decoder.roundMapValid round) :
    (ClaimedChain.challenges rounds assignment).map
        ProductionMessageAcceptance.toConcreteK =
      rounds.map (fun round => sourceValue assignment round.challengeColumns) := by
  induction rounds with
  | nil => rfl
  | cons round rounds inductionHypothesis =>
      have roundValid : Decoder.roundMapValid round :=
        valid round (by simp)
      have tailValid :
          forall candidate, candidate ∈ rounds ->
            Decoder.roundMapValid candidate := by
        intro candidate member
        exact valid candidate (by simp [member])
      have headEquality :
          ProductionMessageAcceptance.toConcreteK
              (ClaimedChain.challenge round assignment) =
            sourceValue assignment round.challengeColumns :=
        congrArg ProductionMessageAcceptance.toConcreteK
          (challengeValue_eq_named round assignment roundValid)
      have tailEquality := inductionHypothesis tailValid
      unfold ClaimedChain.challenges at tailEquality
      simp only [ClaimedChain.challenges, List.map_cons]
      rw [headEquality, tailEquality]

private theorem transportedInitial_eq_named
    {rounds : List RawRoundMap} {columns : RawKColumns}
    (assignment : Nat -> Nat)
    (valid : forall round, round ∈ rounds -> Decoder.roundMapValid round)
    (first : firstClaimInColumns rounds = some columns) :
    ProductionMessageAcceptance.toConcreteK
        (ClaimedChain.initial rounds assignment) =
      sourceValue assignment columns := by
  cases rounds with
  | nil => simp [firstClaimInColumns] at first
  | cons round rounds =>
      have roundValid : Decoder.roundMapValid round :=
        valid round (by simp)
      have columnsEqual : round.claimInColumns = columns := by
        exact Option.some.inj (by simpa [firstClaimInColumns] using first)
      calc
        ProductionMessageAcceptance.toConcreteK
            (ClaimedChain.initial (round :: rounds) assignment) =
            ProductionMessageAcceptance.toConcreteK
              (ClaimedChain.claimIn round assignment) := rfl
        _ = sourceValue assignment round.claimInColumns :=
          congrArg ProductionMessageAcceptance.toConcreteK
            (claimInValue_eq_named round assignment roundValid)
        _ = sourceValue assignment columns := by rw [columnsEqual]

private theorem transportedTerminal_eq_named
    {rounds : List RawRoundMap} {columns : RawKColumns}
    (assignment : Nat -> Nat)
    (valid : forall round, round ∈ rounds -> Decoder.roundMapValid round)
    (last : lastClaimOutColumns rounds = some columns) :
    ProductionMessageAcceptance.toConcreteK
        (ClaimedChain.terminal rounds assignment) =
      sourceValue assignment columns := by
  induction rounds with
  | nil => simp [lastClaimOutColumns] at last
  | cons round rounds inductionHypothesis =>
      cases rounds with
      | nil =>
          have roundValid : Decoder.roundMapValid round :=
            valid round (by simp)
          have columnsEqual : round.claimOutColumns = columns := by
            exact Option.some.inj
              (by simpa [lastClaimOutColumns] using last)
          calc
            ProductionMessageAcceptance.toConcreteK
                (ClaimedChain.terminal [round] assignment) =
                ProductionMessageAcceptance.toConcreteK
                  (ClaimedChain.claimOut round assignment) := rfl
            _ = sourceValue assignment round.claimOutColumns :=
              congrArg ProductionMessageAcceptance.toConcreteK
                (claimOutValue_eq_named round assignment roundValid)
            _ = sourceValue assignment columns := by rw [columnsEqual]
      | cons next rest =>
          have tailValid :
              forall candidate, candidate ∈ next :: rest ->
                Decoder.roundMapValid candidate := by
            intro candidate member
            exact valid candidate (by simp [member])
          simpa [ClaimedChain.terminal, lastClaimOutColumns] using
            inductionHypothesis tailValid last

variable
  {shape : SemanticShape}
  {State : Type uState}
  {publicRingColumns verifierRows : Nat}
  {publicFits : ringDegree * publicRingColumns <= shape.carrierWidth}

/-- Concrete column-by-column binding supplied by the production encoder and
post-`Pi_DEC` assignment audit.  Every round relation names all five source
coefficient pairs.  Challenges are split into the generated block and lane
families so their serialization order is explicit.  The terminal witness
binds only generated input columns; its result follows from source rows. -/
structure ColumnBindings
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (certificate : FixedActive.Certificate context)
    (assignment : Nat -> Nat) : Prop where
  claimedInitial :
    sourceValue assignment Metadata.boundary.claimedInitialColumns =
      Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual.CombinedNc.ProductionPiCcs.rawInitial
        context
  roundMessages :
    RoundMaps.values.map (fun round =>
        round.coefficientColumns.map (sourceValue assignment)) =
      certificate.piCcs.nc.toSumCheck.rounds.map FixedPolynomial.coefficients
  blockChallenges :
    Metadata.boundary.blockPointColumns.map (sourceValue assignment) =
      (Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual.CombinedNc.ProductionPiCcs.ncPoint
        context certificate).block.coordinates
  laneChallenges :
    Metadata.boundary.lanePointColumns.map (sourceValue assignment) =
      (Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual.CombinedNc.ProductionPiCcs.ncPoint
        context certificate).lane.coordinates
  terminal :
    ∃ pending : ProductionDelayedBlockLane,
      ProductionTerminalBridge.TerminalColumnBindings
        context certificate pending assignment

/-- The concrete generated-column bindings construct the exact materialized
assignment-to-production-message dataflow package.  No acceptance predicate,
row-satisfaction proposition, projection result, or digest authority occurs
in the premises. -/
theorem columnBindings_imply_exactDataflow
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (certificate : FixedActive.Certificate context)
    (assignment : Nat -> Nat)
    (constantOne : assignment 0 = 1)
    (consequences : SourceRowsSoundness.Consequences assignment)
    (bindings : ColumnBindings context certificate assignment) :
    ProductionMessageAcceptance.ExactDataflow context certificate assignment := by
  rcases bindings.terminal with ⟨pending, terminalBindings⟩
  have allValid :
      forall round, round ∈ RoundMaps.values ->
        Decoder.roundMapValid round :=
    RoundArtifact.generatedRoundMapsValid.2.2
  refine
    { claimedInitial := ?_
      roundMessages := ?_
      challenges := ?_
      finalClaim := ?_ }
  · exact
      (transportedInitial_eq_named assignment allValid
        generatedColumnSchedule.1).trans bindings.claimedInitial
  · simpa only [ClaimedChain.certificate] using
      transportedRoundMessages_eq assignment allValid bindings.roundMessages
  · calc
      (ClaimedChain.challenges RoundMaps.values assignment).map
          ProductionMessageAcceptance.toConcreteK =
          RoundMaps.values.map
            (fun round => sourceValue assignment round.challengeColumns) :=
        transportedChallenges_eq_named RoundMaps.values assignment allValid
      _ = (Metadata.boundary.blockPointColumns ++
            Metadata.boundary.lanePointColumns).map
              (sourceValue assignment) := by
        simpa only [List.map_map, Function.comp_apply] using
          congrArg (fun columns => columns.map (sourceValue assignment))
            generatedColumnSchedule.2.1
      _ = Metadata.boundary.blockPointColumns.map (sourceValue assignment) ++
            Metadata.boundary.lanePointColumns.map (sourceValue assignment) := by
        rw [List.map_append]
      _ = (Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual.CombinedNc.ProductionPiCcs.ncPoint
              context certificate).block.coordinates ++
            (Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual.CombinedNc.ProductionPiCcs.ncPoint
              context certificate).lane.coordinates := by
        rw [bindings.blockChallenges, bindings.laneChallenges]
      _ = (Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual.CombinedNc.ProductionPiCcs.ncPoint
            context certificate).coordinates := rfl
  · calc
      ProductionMessageAcceptance.toConcreteK
          (ClaimedChain.terminal RoundMaps.values assignment) =
          sourceValue assignment Metadata.boundary.finalSumColumns :=
        transportedTerminal_eq_named assignment allValid
          generatedColumnSchedule.2.2
      _ = Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual.CombinedNc.ProductionPiCcs.messageTerminal
            context certificate := by
        change ProductionTerminalBridge.sourceValue assignment
            TerminalProgram.finalSumColumns =
          Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual.CombinedNc.ProductionPiCcs.messageTerminal
            context certificate
        exact ProductionTerminalBridge.computed_finalSum_eq_messageTerminal
          context certificate pending assignment constantOne
          consequences.terminal terminalBindings

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.ProductionColumnBindings
