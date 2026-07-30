import Nightstream.Implementation.R1CS.Canonical.KSplitNcEndpoints

/-!
Contract: one physical row program for the selected operational Split-NC
verifier.

The program contains, in order:

* the exact Poseidon2 transcript replay;
* the FE and block×lane NC claimed-chain rows; and
* the four verifier-owned endpoint computations.

The transcript output columns are used definitionally by both later groups.
Satisfaction therefore reaches the unchanged operational verifier without a
caller-provided endpoint equation, challenge point, transcript state, or
acceptance proposition.

The direct public/message decoding relation remains explicit here only until
the selected fixed-one/plain/270 call-frame codec constructs it.
-/

set_option autoImplicit false
set_option maxRecDepth 100000

namespace Nightstream.Implementation.R1CS.Canonical.KSplitNcOperationalRows

open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.R1CS
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Folding.PiCCS.OutputClaims
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.TranscriptAuthority.BlockLane

/-- One operational occurrence before its direct public/message views are
bound to the selected outer call frame. -/
structure Input
    {shape : SemanticShape}
    (polynomialInput : PublicInput shape) (domains : Domains) where
  transcript : KSplitNcTranscript.Input polynomialInput domains
  authority : KSplitNcEndpoints.AuthorityColumns shape

/-- The first free column after every transcript permutation span.  This is a
span calculation, not the transcript's distinct-column count. -/
def numericBase
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (input : Input polynomialInput domains) : Nat :=
  input.transcript.transcriptBase +
    (KSplitNcTranscript.replay input.transcript).afterOutput.entries.length *
      SymbolicDuplex.stride

/-- Numeric claimed-chain auxiliaries are contiguous from `numericBase`. -/
def endpointBase
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (input : Input polynomialInput domains) : Nat :=
  numericBase input +
    (KSplitNcBlockLaneRows.cost
      (KSplitNcTranscript.numericColumns input.transcript)).auxiliaryColumns

def endpointInput
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (input : Input polynomialInput domains) :
    KSplitNcEndpoints.Input polynomialInput domains where
  transcript := input.transcript
  authority := input.authority
  frameBase := endpointBase input

/-- Exact compact allocation after the caller-owned four `K` chain endpoints:
transcript state, numeric claimed-chain frames, then endpoint frames. -/
def allocationWidth
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (input : Input polynomialInput domains) : Nat :=
  (KSplitNcTranscript.replay input.transcript).afterOutput.entries.length *
      SymbolicDuplex.stride +
    (KSplitNcBlockLaneRows.cost
      (KSplitNcTranscript.numericColumns input.transcript)).auxiliaryColumns +
    KSplitNcEndpoints.allocationWidth (endpointInput input)

/-- First free column after the complete operational ΠCCS occurrence. -/
def afterAllocation
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (input : Input polynomialInput domains) : Nat :=
  input.transcript.transcriptBase + allocationWidth input

def transcriptRows
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (constants : Poseidon2Schedule.Constants)
    (input : Input polynomialInput domains) : List Row :=
  KSplitNcTranscript.rows constants input.transcript

def numericRows
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (input : Input polynomialInput domains) : List Row :=
  KSplitNcBlockLaneRows.rows
    (KSplitNcTranscript.numericColumns input.transcript)
    (numericBase input)

def endpointRows
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (input : Input polynomialInput domains) : List Row :=
  KSplitNcEndpoints.rows (endpointInput input)

def rowGroups
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (constants : Poseidon2Schedule.Constants)
    (input : Input polynomialInput domains) : List (List Row) :=
  [transcriptRows constants input, numericRows input, endpointRows input]

def rows
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (constants : Poseidon2Schedule.Constants)
    (input : Input polynomialInput domains) : List Row :=
  (rowGroups constants input).flatten

theorem rows_length
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (constants : Poseidon2Schedule.Constants)
    (input : Input polynomialInput domains) :
    (rows constants input).length =
      (transcriptRows constants input).length +
        (numericRows input).length + (endpointRows input).length := by
  simp [rows, rowGroups]
  omega

theorem afterAllocation_eq_endpoint_end
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (input : Input polynomialInput domains) :
    afterAllocation input =
      endpointBase input +
        KSplitNcEndpoints.allocationWidth (endpointInput input) := by
  unfold afterAllocation allocationWidth endpointBase numericBase
  omega

/-- Exact ordered compact allocation, independent of row count. -/
def columns
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (input : Input polynomialInput domains) : List Nat :=
  (List.range (allocationWidth input)).map
    (fun offset => input.transcript.transcriptBase + offset)

theorem columns_length
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (input : Input polynomialInput domains) :
    (columns input).length = allocationWidth input := by
  simp [columns]

theorem columns_nodup
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (input : Input polynomialInput domains) :
    (columns input).Nodup := by
  unfold columns
  exact LinCombNormal.nodup_map _ _ (fun left right equal => by omega)
    List.nodup_range

theorem columns_lt_afterAllocation
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (input : Input polynomialInput domains)
    (column : Nat) (member : column ∈ columns input) :
    column < afterAllocation input := by
  unfold columns at member
  rcases List.mem_map.mp member with ⟨offset, inRange, rfl⟩
  have offsetLt := List.mem_range.mp inRange
  unfold afterAllocation
  omega

/-- Endpoint computation cost.  Its eight equality-only rows are present in
the row component and absent from the compact allocation component. -/
def endpointCost
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (input : Input polynomialInput domains) : Cost where
  recurringRows := (endpointRows input).length
  committedColumns := 0
  publicColumns := 0
  auxiliaryColumns :=
    KSplitNcEndpoints.allocationWidth (endpointInput input)

/-- Exact operational ΠCCS cost as a receipt fold over transcript, numeric
claimed-chain, and endpoint programs. -/
def cost
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (input : Input polynomialInput domains) : Cost :=
  KSplitNcTranscript.cost input.transcript +
    KSplitNcBlockLaneRows.cost
      (KSplitNcTranscript.numericColumns input.transcript) +
    endpointCost input

theorem rows_cost
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (constants : Poseidon2Schedule.Constants)
    (input : Input polynomialInput domains) :
    (rows constants input).length = (cost input).recurringRows := by
  rw [rows_length]
  unfold cost endpointCost transcriptRows numericRows
  simp only [Cost.add_recurringRows]
  rw [KSplitNcTranscript.rows_cost, KSplitNcBlockLaneRows.rows_cost]

theorem allocationWidth_eq_cost
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (input : Input polynomialInput domains) :
    allocationWidth input = (cost input).auxiliaryColumns := by
  unfold allocationWidth cost endpointCost KSplitNcTranscript.cost
    SymbolicDuplex.cost
  simp only [Cost.add_auxiliaryColumns, KSplitNcBlockLaneRows.cost]
  rw [SymbolicDuplex.stride_eq]

theorem satisfies_group
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (constants : Poseidon2Schedule.Constants)
    (input : Input polynomialInput domains)
    (assignment : Nat → Nat)
    (satisfied : Satisfies (rows constants input) assignment)
    (group : List Row) (member : group ∈ rowGroups constants input) :
    Satisfies group assignment := by
  intro row rowMember
  exact satisfied row (List.mem_flatten.2 ⟨group, member, rowMember⟩)

/-- Satisfaction of one Lean-owned row list implies the unchanged
deterministic operational Split-NC verifier relation. -/
theorem accepted_of_rows
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (profile : Polynomial.Fe.SupportedProfile shape domains.fe)
    (constants : Poseidon2Schedule.Constants)
    (input : Input polynomialInput domains)
    (message : OutputMessage shape)
    (assignment : Nat → Nat)
    (residues : ∀ column, assignment column < goldilocksP)
    (constantWire : assignment 0 = 1)
    (authority :
      KSplitNcEndpoints.DecodedAuthority
        (endpointInput input) assignment message)
    (satisfied : Satisfies (rows constants input) assignment) :
    Protocol.BlockLane.Accepted
      (fun _ : Unit => polynomialInput)
      (KSplitNcTranscriptSemantics.valueSchedule
        constants assignment input.transcript)
      (KSplitNcTranscriptSemantics.priorState assignment input.transcript)
      profile KSplitNcTranscriptSemantics.unitStatement
      (KSplitNcOperational.certificate
        assignment input.transcript message) := by
  have transcriptSatisfied :
      Satisfies (transcriptRows constants input) assignment :=
    satisfies_group constants input assignment satisfied _ (by
      simp [rowGroups])
  have numericSatisfied :
      Satisfies (numericRows input) assignment :=
    satisfies_group constants input assignment satisfied _ (by
      simp [rowGroups])
  have endpointSatisfied :
      Satisfies (endpointRows input) assignment :=
    satisfies_group constants input assignment satisfied _ (by
      simp [rowGroups])
  have transcriptValid :
      SymbolicDuplexSemantics.Valid
        input.transcript.transcriptBase constants assignment
        (KSplitNcTranscript.outputBuilder input.transcript) := by
    exact SymbolicDuplexSemantics.valid_of_satisfied
      input.transcript.transcriptBase constants
      (KSplitNcTranscript.outputBuilder input.transcript)
      assignment residues constantWire transcriptSatisfied
  have endpoints :=
    KSplitNcEndpoints.endpointAgrees_of_rows
      profile constants assignment constantWire (endpointInput input)
      message transcriptValid authority endpointSatisfied
  exact KSplitNcOperational.accepted_of_rows
    profile constants assignment constantWire input.transcript message
    transcriptValid endpoints (numericBase input) numericSatisfied

end Nightstream.Implementation.R1CS.Canonical.KSplitNcOperationalRows
