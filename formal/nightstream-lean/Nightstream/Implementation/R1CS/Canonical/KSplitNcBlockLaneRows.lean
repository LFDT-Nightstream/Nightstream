import Nightstream.Implementation.R1CS.Canonical.KSplitNcFeRows
import Nightstream.Implementation.R1CS.Canonical.KSplitNcNcRows
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.TranscriptAuthority.BlockLane

/-!
Lean-owned numeric row program for the exact operational Split-NC FE and
block×lane NC claimed chains.

Assurance tier: model-level.

Owns: one shared-domain column bundle, disjoint Horner allocation, exact
row/column cost, and composition of row satisfaction into the two unchanged
claimed-chain relations.

Does not own: Poseidon2 transcript replay, construction of the FE/NC entry or
terminal values, output authority, call-frame decoding, semantic soundness
events, Rust, or generated rows.

The FE successor transcript state is the NC predecessor transcript state, but
that state is not a claimed scalar and therefore is not represented by these
numeric rows. The later transcript program owns that handoff.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Canonical.KSplitNcBlockLaneRows

open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.TranscriptAuthority.BlockLane

/-- Exact numeric source columns for one operational FE→NC execution. The
shared `Domains` record makes divergent FE/NC lane dimensions
unrepresentable at this layer. -/
structure Columns
    {shape : SemanticShape}
    (input : PublicInput shape)
    (domains : Domains) where
  fe : KSplitNcFeRows.Columns (SumCheck.Fe.Drow input)
  nc : KSplitNcNcRows.Columns

/-- NC Horner auxiliaries begin immediately after the FE auxiliaries. -/
def ncBase
    {shape : SemanticShape}
    {input : PublicInput shape}
    {domains : Domains}
    (columns : Columns input domains)
    (base : Nat) : Nat :=
  base + (KSplitNcFeRows.cost columns.fe).auxiliaryColumns

/-- Exact numeric claimed-chain program: FE first, then NC. -/
def rows
    {shape : SemanticShape}
    {input : PublicInput shape}
    {domains : Domains}
    (columns : Columns input domains)
    (base : Nat) : List Row :=
  KSplitNcFeRows.rows columns.fe base ++
    KSplitNcNcRows.rows columns.nc (ncBase columns base)

/-- Only Horner auxiliaries are owned here. Every message, challenge, entry,
boundary, and terminal column is a shared read owned by the call layout. -/
def cost
    {shape : SemanticShape}
    {input : PublicInput shape}
    {domains : Domains}
    (columns : Columns input domains) : Cost :=
  KSplitNcFeRows.cost columns.fe + KSplitNcNcRows.cost columns.nc

theorem rows_length
    {shape : SemanticShape}
    {input : PublicInput shape}
    {domains : Domains}
    (columns : Columns input domains)
    (base : Nat) :
    (rows columns base).length =
      columns.fe.rowRounds.length *
          (3 * SumCheck.Fe.Drow input + 2) + 2 +
        (columns.fe.laneRounds.length * 8 + 2) +
        (columns.nc.rounds.length * 14 + 2) := by
  simp only [rows, List.length_append]
  rw [KSplitNcFeRows.rows_length, KSplitNcNcRows.rows_length]

theorem rows_cost
    {shape : SemanticShape}
    {input : PublicInput shape}
    {domains : Domains}
    (columns : Columns input domains)
    (base : Nat) :
    (rows columns base).length = (cost columns).recurringRows := by
  simp only [rows, cost, List.length_append, Cost.add_recurringRows]
  rw [KSplitNcFeRows.rows_cost, KSplitNcNcRows.rows_cost]

theorem auxiliary_count
    {shape : SemanticShape}
    {input : PublicInput shape}
    {domains : Domains}
    (columns : Columns input domains) :
    (cost columns).auxiliaryColumns =
      columns.fe.rowRounds.length * (3 * SumCheck.Fe.Drow input) +
        columns.fe.laneRounds.length * 6 +
        columns.nc.rounds.length * 12 := by
  simp only [cost, Cost.add_auxiliaryColumns]
  rw [KSplitNcFeRows.auxiliary_count, KSplitNcNcRows.auxiliary_count]

private theorem satisfies_left
    {left right : List Row}
    {assignment : Nat -> Nat}
    (satisfied : Satisfies (left ++ right) assignment) :
    Satisfies left assignment :=
  fun row member => satisfied row (List.mem_append_left _ member)

private theorem satisfies_right
    {left right : List Row}
    {assignment : Nat -> Nat}
    (satisfied : Satisfies (left ++ right) assignment) :
    Satisfies right assignment :=
  fun row member => satisfied row (List.mem_append_right _ member)

/-- Satisfaction of the combined numeric program derives both exact
claimed-chain relations. The projection equations contain no acceptance
proposition and must eventually be derived from call-frame and transcript
rows. -/
theorem accepted_of_rows
    {shape : SemanticShape}
    {input : PublicInput shape}
    {domains : Domains}
    (columns : Columns input domains)
    (base : Nat)
    (assignment : Nat -> Nat)
    (constantWire : assignment 0 = 1)
    (feInitial feTerminal ncInitial ncTerminal : K)
    (fePoint : Polynomial.Fe.Point shape domains.fe)
    (feCertificate : SumCheck.Fe.Certificate input domains.fe)
    (ncPoint : Polynomial.Nc.BlockLane.Point domains.nc)
    (ncCertificate : Transcript.Nc.BlockLane.Certificate domains.nc)
    (feAgrees :
      KSplitNcFeRows.Agrees columns.fe assignment
        feInitial feTerminal fePoint feCertificate)
    (ncAgrees :
      KSplitNcNcRows.Agrees columns.nc assignment
        ncInitial ncTerminal ncPoint ncCertificate)
    (satisfied : Satisfies (rows columns base) assignment) :
    SumCheck.Fe.Accepted feInitial feTerminal fePoint feCertificate ∧
      SumCheck.Nc.Accepted ncInitial ncPoint.coordinates ncTerminal
        ncCertificate.toSumCheck := by
  exact
    ⟨KSplitNcFeRows.accepted_of_rows columns.fe base assignment
        constantWire feInitial feTerminal fePoint feCertificate feAgrees
        (satisfies_left satisfied),
      KSplitNcNcRows.accepted_of_rows columns.nc (ncBase columns base)
        assignment constantWire ncInitial ncTerminal ncPoint ncCertificate
        ncAgrees (satisfies_right satisfied)⟩

end Nightstream.Implementation.R1CS.Canonical.KSplitNcBlockLaneRows
