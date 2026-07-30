import Nightstream.Implementation.R1CS.Canonical.KPiRlcSemanticBinding
import Nightstream.Implementation.R1CS.Canonical.SymbolicDuplexSemantics

/-!
Contract: derive the public PiRLC quotient-check point from a Lean-owned
Poseidon2 transcript.

The paper's `Pi_RLC` reduction samples its coefficient challenges before the
prover constructs the combined output.  The quotient identities are a later
implementation check: their Schwartz--Zippel point must therefore be sampled
after the bounded identity coefficients and quotient witnesses have been
absorbed.  This module fixes that suffix without importing Rust or generated
rows.

Owns:
- one length-delimited serialization of every bounded quotient identity in
  public-role order;
- one Poseidon2-duplex squeeze whose output ports are definitionally the
  `beta` columns consumed by `KTraceProgram`;
- concatenation of the transcript rows and the existing quotient rows; and
- the exact statement that a surviving `BatchBadRoot` is evaluated at the
  transcript-derived point.

Does not own the preceding PiCCS/PiRLC coefficient-sampler state, construction
of the prover quotient witnesses, a root-probability bound, or the complete
`nifsVerify` program.  The caller supplies a prior symbolic builder; this
module binds the suffix from that state onward.

The numeric tag is a local canonical-encoding choice, not a paper constant.
Assurance tier: canonical model/R1CS refinement.
-/

set_option autoImplicit false
set_option maxRecDepth 10000

namespace Nightstream.Implementation.R1CS.Canonical.KPiRlcProjectionTranscript

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.KPiRlcSemanticBinding
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Core
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Schedule
open Nightstream.Implementation.R1CS.Canonical.SymbolicDuplexSemantics
open Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper
open Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc
open Nightstream.Implementation.R1CS.ProjectionProgram
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding

/-- Domain tag for the bounded public-PiRLC quotient-identity batch. -/
def identityBatchTag : Nat := 48

/-- A verifier constant represented on the shared constant wire. -/
def word (value : Nat) : LinCombNormal.LinComb :=
  [(0, value % goldilocksP)]

/-- One authoritative field column. -/
def columnField (column : Nat) : LinCombNormal.LinComb := [(column, 1)]

/-- Length-delimited field-vector serialization.  The delimiter is needed even
though `Valid` later fixes the selected widths: transcript syntax must remain
unambiguous before semantic validation is invoked. -/
def vectorFields (columns : List Nat) : List LinCombNormal.LinComb :=
  word columns.length :: columns.map columnField

/-- One input claim in exact public-role order. -/
def inputFields
    {params : GlobalParams}
    {arity : BatchArity params}
    {matrixCount : Nat}
    (source : SourceColumns params arity matrixCount)
    (index : Fin arity.total) : List LinCombNormal.LinComb :=
  vectorFields (source.challenges index) ++
    (publicOrder matrixCount).flatMap fun role =>
      vectorFields ((source.inputs index).at role)

/-- The complete bounded identity statement.

The order is:
1. domain and shape;
2. every `(rho,input)` source pair, input-major then role-major;
3. every output vector in public-role order; and
4. every prover quotient in the same role order.

Consequently the point cannot be chosen before any coefficient occurring in
the checked identities. -/
def statementFields
    {params : GlobalParams}
    {arity : BatchArity params}
    {matrixCount : Nat}
    (source : SourceColumns params arity matrixCount)
    (quotients : PublicRole matrixCount → List Nat) :
    List LinCombNormal.LinComb :=
  [word identityBatchTag, word arity.total, word matrixCount,
    word (publicOrder matrixCount).length] ++
  (List.ofFn (inputFields source)).flatten ++
  (publicOrder matrixCount).flatMap
    (fun role => vectorFields (source.output.at role)) ++
  (publicOrder matrixCount).flatMap
    (fun role => vectorFields (quotients role))

/-- Inputs selected before lowering this transcript suffix.  `prior` is the
symbolic state produced by the preceding verified schedule, not a digest. -/
structure Input
    (params : GlobalParams)
    (arity : BatchArity params)
    (matrixCount : Nat) where
  transcriptBase : Nat
  prior : SymbolicDuplex.Builder
  source : SourceColumns params arity matrixCount
  quotients : PublicRole matrixCount → List Nat

/-- Every authoritative coefficient column is allocated before the transcript
suffix begins.  The transcript may read those columns but never owns them. -/
structure Input.ColumnsBelowTranscript
    {params : GlobalParams} {arity : BatchArity params}
    {matrixCount : Nat}
    (input : Input params arity matrixCount) : Prop where
  challenge :
    ∀ index column,
      column ∈ input.source.challenges index →
        column < input.transcriptBase
  sourceInput :
    ∀ index role column,
      column ∈ (input.source.inputs index).at role →
        column < input.transcriptBase
  output :
    ∀ role column,
      column ∈ input.source.output.at role →
        column < input.transcriptBase
  quotient :
    ∀ role column,
      column ∈ input.quotients role →
        column < input.transcriptBase

/-- Absorb the complete bounded batch before sampling its check point. -/
def absorbed
    {params : GlobalParams}
    {arity : BatchArity params}
    {matrixCount : Nat}
    (input : Input params arity matrixCount) : SymbolicDuplex.Builder :=
  SymbolicDuplex.absorbMany input.transcriptBase
    (statementFields input.source input.quotients) input.prior

/-- The call index of the forced gate permutation.  This definition follows
the builder operation itself, so it also covers the case where absorbing the
gate marker first flushes a full rate block. -/
def betaCall
    {params : GlobalParams}
    {arity : BatchArity params}
    {matrixCount : Nat}
    (input : Input params arity matrixCount) : Nat :=
  (SymbolicDuplex.absorb input.transcriptBase SymbolicDuplex.one
    (absorbed input)).entries.length

/-- The two exact output ports returned by the gate permutation. -/
def betaColumns
    {params : GlobalParams}
    {arity : BatchArity params}
    {matrixCount : Nat}
    (input : Input params arity matrixCount) : KColumns :=
  let layout := SymbolicDuplex.layoutAt input.transcriptBase (betaCall input)
  ⟨layout.outputPort ⟨0, by decide⟩,
    layout.outputPort ⟨1, by decide⟩⟩

/-- The transcript replay and its freshly sampled extension point. -/
def replay
    {params : GlobalParams}
    {arity : BatchArity params}
    {matrixCount : Nat}
    (input : Input params arity matrixCount) :
    KMul.Carried × SymbolicDuplex.Builder :=
  SymbolicDuplex.squeezeK input.transcriptBase (absorbed input)

/-- Source-bound quotient columns with no free `beta` field. -/
def projectionColumns
    {params : GlobalParams}
    {arity : BatchArity params}
    {matrixCount : Nat}
    (input : Input params arity matrixCount) :
    KPiRlcSemanticBinding.ProjectionColumns params arity matrixCount where
  source := input.source
  beta := betaColumns input
  quotients := input.quotients

/-- The quotient program begins after the complete transcript column span. -/
def projectionBase
    {params : GlobalParams}
    {arity : BatchArity params}
    {matrixCount : Nat}
    (input : Input params arity matrixCount) : Nat :=
  input.transcriptBase +
    (replay input).2.entries.length * SymbolicDuplex.stride

/-- The gate contributes the final entry of the replay. -/
theorem replay_entries_length
    {params : GlobalParams} {arity : BatchArity params}
    {matrixCount : Nat}
    (input : Input params arity matrixCount) :
    (replay input).2.entries.length = betaCall input + 1 := by
  unfold replay SymbolicDuplex.squeezeK SymbolicDuplex.gate betaCall
  exact SymbolicDuplex.permute_entries_length _ _

/-- Both beta ports precede the quotient program's fresh block. -/
theorem betaColumns_below_projectionBase
    {params : GlobalParams} {arity : BatchArity params}
    {matrixCount : Nat}
    (input : Input params arity matrixCount) :
    (betaColumns input).c0 < projectionBase input ∧
      (betaColumns input).c1 < projectionBase input := by
  rw [projectionBase, replay_entries_length, SymbolicDuplex.stride_eq]
  unfold betaColumns SymbolicDuplex.layoutAt
  dsimp only
  rw [SymbolicDuplex.stride_eq]
  omega

/-- Every column read by the quotient occurrence is below its fresh auxiliary
block.  In particular, beta is owned by the transcript rather than allocated
again by the quotient program. -/
theorem projectionColumns_belowBase
    {params : GlobalParams} {arity : BatchArity params}
    {matrixCount : Nat}
    (input : Input params arity matrixCount)
    (below : input.ColumnsBelowTranscript) :
    (projectionColumns input).toColumns.BelowBase
      (projectionBase input) := by
  have transcriptBefore :
      input.transcriptBase ≤ projectionBase input := by
    unfold projectionBase
    omega
  have betaBelow := betaColumns_below_projectionBase input
  refine
    { betaLow := betaBelow.1
      betaHigh := betaBelow.2
      challenge := ?_
      input := ?_
      output := ?_
      quotient := ?_ }
  · intro index column member
    exact Nat.lt_of_lt_of_le (below.challenge index column member)
      transcriptBefore
  · intro index role column member
    exact Nat.lt_of_lt_of_le (below.sourceInput index role column member)
      transcriptBefore
  · intro role column member
    exact Nat.lt_of_lt_of_le (below.output role column member)
      transcriptBefore
  · intro role column member
    exact Nat.lt_of_lt_of_le (below.quotient role column member)
      transcriptBefore

/-- The squeezed symbolic value is exactly the singleton expression over the
two columns handed to the quotient program. -/
theorem replay_point_eq_decodePoint
    {params : GlobalParams}
    {arity : BatchArity params}
    {matrixCount : Nat}
    (input : Input params arity matrixCount) :
    (replay input).1 =
      KTraceProgram.decodePoint (betaColumns input) := by
  rfl

/-- Decoding the physical beta columns is the same as decoding the symbolic
squeeze result. -/
theorem beta_value_eq_decoded
    {params : GlobalParams}
    {arity : BatchArity params}
    {matrixCount : Nat}
    (assignment : Nat → Nat)
    (input : Input params arity matrixCount) :
    (betaColumns input).value assignment =
      KFixedPhaseSumCheck.decodeCarried assignment (replay input).1 := by
  apply KBridge.toPair_injective
  rw [KFixedPhaseSumCheck.toPair_decodeCarried, replay_point_eq_decodePoint,
    KTraceProgram.carriedValue_decodePoint]

/-- Static width validity is preserved when the transcript supplies beta. -/
theorem projectionColumns_valid
    {params : GlobalParams}
    {arity : BatchArity params}
    {matrixCount : Nat}
    (input : Input params arity matrixCount)
    (sourceValid : input.source.Valid)
    (quotientWidth : ∀ role, (input.quotients role).length = 53) :
    (projectionColumns input).Valid :=
  ⟨sourceValid, quotientWidth⟩

/-- Complete transcript-plus-quotient row program for this bounded suffix. -/
def rows
    {params : GlobalParams}
    {arity : BatchArity params}
    {matrixCount : Nat}
    (constants : Constants)
    (input : Input params arity matrixCount)
    (valid : (projectionColumns input).Valid) : List Row :=
  SymbolicDuplex.rows input.transcriptBase constants (replay input).2 ++
    ((projectionColumns input).occurrence valid
      (projectionBase input)).rows

/-- Satisfaction restricts to the transcript program. -/
theorem transcript_satisfied
    {params : GlobalParams}
    {arity : BatchArity params}
    {matrixCount : Nat}
    (constants : Constants)
    (input : Input params arity matrixCount)
    (valid : (projectionColumns input).Valid)
    (assignment : Nat → Nat)
    (satisfied : Satisfies (rows constants input valid) assignment) :
    Satisfies
      (SymbolicDuplex.rows input.transcriptBase constants (replay input).2)
      assignment :=
  fun row member => satisfied row (List.mem_append_left _ member)

/-- Satisfaction restricts to the coefficient-identity occurrence. -/
theorem occurrence_satisfied
    {params : GlobalParams}
    {arity : BatchArity params}
    {matrixCount : Nat}
    (constants : Constants)
    (input : Input params arity matrixCount)
    (valid : (projectionColumns input).Valid)
    (assignment : Nat → Nat)
    (satisfied : Satisfies (rows constants input valid) assignment) :
    Satisfies
      ((projectionColumns input).occurrence valid
        (projectionBase input)).rows assignment :=
  fun row member => satisfied row (List.mem_append_right _ member)

/-- The beta consumed by every quotient row is the value-level Poseidon2
challenge derived after absorbing the complete bounded identity statement. -/
theorem beta_eq_transcript
    {params : GlobalParams}
    {arity : BatchArity params}
    {matrixCount : Nat}
    (constants : Constants)
    (input : Input params arity matrixCount)
    (valid : (projectionColumns input).Valid)
    (assignment : Nat → Nat)
    (residues : ∀ column, assignment column < goldilocksP)
    (constantWire : assignment 0 = 1)
    (satisfied : Satisfies (rows constants input valid) assignment) :
    (betaColumns input).value assignment =
      (squeezeKValue constants
        (decodedBuilder assignment (absorbed input))).1 := by
  rw [beta_value_eq_decoded assignment input]
  have allValid :=
    valid_of_satisfied input.transcriptBase constants (replay input).2
      assignment residues constantWire
      (transcript_satisfied constants input valid assignment satisfied)
  have squeeze :=
    decoded_squeezeK input.transcriptBase constants assignment
      (absorbed input) constantWire allValid
  exact congrArg Prod.fst squeeze

/-- Exact recurring-row count of the constructed suffix. -/
theorem rows_length
    {params : GlobalParams}
    {arity : BatchArity params}
    {matrixCount : Nat}
    (constants : Constants)
    (input : Input params arity matrixCount)
    (valid : (projectionColumns input).Valid) :
    (rows constants input valid).length =
      (replay input).2.entries.length * 352 +
        (23 + 2 * matrixCount) * (321 * arity.total + 482) := by
  unfold rows
  rw [List.length_append, SymbolicDuplex.rows_length]
  unfold KPiRlcSemanticBinding.ProjectionColumns.occurrence
  rw [KPiRlcTrace.occurrence_rows_length]

/-- Row satisfaction yields the paper PiRLC equations or the exact bounded
bad-root event, and the latter is explicitly tied to the transcript point.
There is no caller-provided beta or unnamed refinement failure. -/
theorem equations_or_transcriptBadRoot_of_rows
    {Assignment : Type}
    {params : GlobalParams}
    {arity : BatchArity params}
    {matrixCount : Nat}
    {semantics :
      RelationSemantics Unit Assignment PackedPublicInput
        FPrimeFullHistoryNifsPaper.Point
        FPrimeFullHistoryNifsPaper.Evaluation
        PackedCommitment}
    (algebra :
      PiRLC.Algebra Unit Assignment PackedPublicInput
        FPrimeFullHistoryNifsPaper.Point
        FPrimeFullHistoryNifsPaper.Evaluation
        PackedCommitment Ring semantics params)
    (codec : CarrierCodec matrixCount)
    (ring : RingAlgebra)
    (algebraRefinement :
      AlgebraRefinement algebra codec ring)
    (constants : Constants)
    (input : Input params arity matrixCount)
    (valid : (projectionColumns input).Valid)
    (assignment : Nat → Nat)
    (residues : ∀ column, assignment column < goldilocksP)
    (constantWire : assignment 0 = 1)
    (satisfied : Satisfies (rows constants input valid) assignment) :
    PiRLC.Equations algebra
        (attempt codec assignment input.source.toBatchColumns) ∨
      (((projectionColumns input).occurrence valid
          (projectionBase input)).BadRoot assignment ∧
        (betaColumns input).value assignment =
          (squeezeKValue constants
            (decodedBuilder assignment (absorbed input))).1) := by
  rcases KPiRlcSemanticBinding.equations_or_badRoot_of_rows
      algebra codec ring algebraRefinement assignment
      (projectionColumns input) valid (projectionBase input)
      constantWire
      (occurrence_satisfied constants input valid assignment satisfied) with
    equations | badRoot
  · exact Or.inl equations
  · exact Or.inr
      ⟨badRoot,
        beta_eq_transcript constants input valid assignment residues
          constantWire satisfied⟩

end Nightstream.Implementation.R1CS.Canonical.KPiRlcProjectionTranscript
