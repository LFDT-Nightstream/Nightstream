import Nightstream.Implementation.R1CS.Canonical.KFixedPhaseEndpointCoverage
import Nightstream.Implementation.R1CS.Canonical.KSplitNcOperationalRows

/-!
Contract: the five quadratic-extension claimed-chain values used by the
selected Split-NC occurrence are mentioned by its emitted fixed-phase rows.

The endpoints are shared reads owned by the enclosing `nifsVerify` frame, so
they are intentionally absent from `KSplitNcOperationalRows.columns`.
-/

set_option autoImplicit false
set_option maxRecDepth 100000

namespace Nightstream.Implementation.R1CS.Canonical.KSplitNcClaimedEndpointCoverage

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.AllocationCoverage
open Nightstream.Implementation.R1CS.Canonical.KFixedPhaseEndpointCoverage
open Nightstream.Implementation.R1CS.Canonical.KFixedPhaseSumCheck
open Nightstream.Implementation.R1CS.Canonical.KSplitNcBlockLaneRows
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.KFixedPhaseSemanticOccurrence
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Folding.PiCCS.OutputClaims
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.TranscriptAuthority.BlockLane

/-- The five caller-owned claimed values, in their selected frame order. -/
def columns
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (input : KSplitNcOperationalRows.Input polynomialInput domains) :
    List Nat :=
  let numeric := KSplitNcTranscript.numericColumns input.transcript
  KFixedPhaseEndpointCoverage.columns (carried numeric.fe.initial) ++
    KFixedPhaseEndpointCoverage.columns (carried numeric.fe.boundary) ++
    KFixedPhaseEndpointCoverage.columns (carried numeric.fe.terminal) ++
    KFixedPhaseEndpointCoverage.columns (carried numeric.nc.current) ++
    KFixedPhaseEndpointCoverage.columns (carried numeric.nc.terminal)

private theorem fe
    {rowDegree : Nat}
    (source : KSplitNcFeRows.Columns rowDegree)
    (base : Nat) :
    RowsCover
      (KSplitNcFeRows.rows source base)
      (KFixedPhaseEndpointCoverage.columns (carried source.initial) ++
        KFixedPhaseEndpointCoverage.columns (carried source.boundary) ++
        KFixedPhaseEndpointCoverage.columns (carried source.terminal)) := by
  intro column member
  rcases List.mem_append.1 member with inPrefix | inTerminal
  · have covered :=
      KFixedPhaseEndpointCoverage.chain
        (carried source.initial)
        source.rowSource.rowRounds
        source.rowSource.rowChallenges
        (carried source.boundary)
        base
        (by
          simpa [KSplitNcFeRows.Columns.rowSource,
            SourceColumns.rowRounds, SourceColumns.rowChallenges] using
            source.rowSameLength)
    rcases covered column inPrefix with ⟨row, rowMember, mentioned⟩
    exact
      ⟨row, List.mem_append_left _ rowMember, mentioned⟩
  · have covered :=
      KFixedPhaseEndpointCoverage.chain
        (carried source.boundary)
        source.laneSource.rowRounds
        source.laneSource.rowChallenges
        (carried source.terminal)
        (KSplitNcFeRows.laneBase source base)
        (by
          simpa [KSplitNcFeRows.Columns.laneSource,
            SourceColumns.rowRounds, SourceColumns.rowChallenges] using
            source.laneSameLength)
    rcases covered column
        (List.mem_append_right
          (KFixedPhaseEndpointCoverage.columns (carried source.boundary))
          inTerminal) with
      ⟨row, rowMember, mentioned⟩
    exact
      ⟨row, List.mem_append_right _ rowMember, mentioned⟩

private theorem nc
    (source : KSplitNcNcRows.Columns)
    (base : Nat) :
    RowsCover
      (KSplitNcNcRows.rows source base)
      (KFixedPhaseEndpointCoverage.columns (carried source.current) ++
        KFixedPhaseEndpointCoverage.columns
          (carried source.terminal)) :=
  KFixedPhaseEndpointCoverage.chain
    (carried source.current)
    source.rowRounds
    source.rowChallenges
    (carried source.terminal)
    base
    (by
      simpa [SourceColumns.rowRounds, SourceColumns.rowChallenges] using
        source.sameLength)

private theorem numeric
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (source : KSplitNcBlockLaneRows.Columns polynomialInput domains)
    (base : Nat) :
    RowsCover
      (KSplitNcBlockLaneRows.rows source base)
      (KFixedPhaseEndpointCoverage.columns (carried source.fe.initial) ++
        KFixedPhaseEndpointCoverage.columns (carried source.fe.boundary) ++
        KFixedPhaseEndpointCoverage.columns (carried source.fe.terminal) ++
        KFixedPhaseEndpointCoverage.columns (carried source.nc.current) ++
        KFixedPhaseEndpointCoverage.columns (carried source.nc.terminal)) := by
  exact AllocationCoverage.append
    (KSplitNcFeRows.rows source.fe base)
    (KSplitNcNcRows.rows source.nc
      (KSplitNcBlockLaneRows.ncBase source base))
    (KFixedPhaseEndpointCoverage.columns (carried source.fe.initial) ++
      KFixedPhaseEndpointCoverage.columns (carried source.fe.boundary) ++
      KFixedPhaseEndpointCoverage.columns (carried source.fe.terminal))
    (KFixedPhaseEndpointCoverage.columns (carried source.nc.current) ++
      KFixedPhaseEndpointCoverage.columns (carried source.nc.terminal))
    (fe source.fe base)
    (nc source.nc (KSplitNcBlockLaneRows.ncBase source base))

/-- Every caller-owned claimed endpoint is reached by the complete operational
row list. -/
theorem rows
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (constants : Poseidon2Schedule.Constants)
    (input : KSplitNcOperationalRows.Input polynomialInput domains) :
    RowsCover
      (KSplitNcOperationalRows.rows constants input)
      (columns input) := by
  let source := KSplitNcTranscript.numericColumns input.transcript
  have covered :=
    numeric source (KSplitNcOperationalRows.numericBase input)
  intro column member
  rcases covered column (by simpa [columns, source] using member) with
    ⟨row, rowMember, mentioned⟩
  refine
    ⟨row, ?_, mentioned⟩
  unfold KSplitNcOperationalRows.rows
  apply List.mem_flatten.2
  exact
    ⟨KSplitNcOperationalRows.numericRows input,
      by simp [KSplitNcOperationalRows.rowGroups],
      by
        simpa [KSplitNcOperationalRows.numericRows, source] using rowMember⟩

end Nightstream.Implementation.R1CS.Canonical.KSplitNcClaimedEndpointCoverage
