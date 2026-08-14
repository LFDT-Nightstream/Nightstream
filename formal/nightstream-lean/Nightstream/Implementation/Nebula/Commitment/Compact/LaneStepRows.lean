import Nightstream.Implementation.Nebula.Commitment.Compact.ChainPoseidonRows
import Nightstream.Implementation.Nebula.Commitment.Compact.TokenRows

/-!
Contract: one exact V2 compact-chain update for one commitment-bundle lane.

Assurance tier: implementation-to-protocol bridge.

Owns one two-stage token call, one Poseidon2 leaf call, one indexed Poseidon2
link call, exact column reuse between calls, and the row-derived theorem that
the supplied after-root is the hash of the supplied before-root and the exact
bundle commitment.

Does not own bundle bit-to-field decoding, the three-lane composition,
prechallenge extraction, transcript rows, absolute generated columns, or
Rust conformance.

Emits constraints: yes.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Nebula.CompactLaneStepRows

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Poseidon2Sponge
open Nightstream.Implementation.Nebula.CompactChainHashFrame
open Nightstream.Implementation.Nebula.CompactChainHashFrameRows
open Nightstream.Implementation.Nebula.CompactChainPoseidonRows
open Nightstream.Protocol.Nebula
open Nightstream.Protocol.Nebula.CompactCommit
open Nightstream.Protocol.Nebula.Lifecycle

structure Layout where
  token : CompactTokenRows.Layout
  leafFrame : LeafLayout
  leafTrace : Trace
  linkFrame : LinkLayout
  linkTrace : Trace
  priorDigestColumn : Fin 4 → Nat
  afterDigestColumn : Fin 4 → Nat

def rows (manifest : SeedSchedule.Manifest) (role : Role)
    (layout : Layout) : List Row :=
  CompactTokenRows.rows manifest role layout.token ++
    Framed.rows (leafRows manifest role layout.leafFrame) layout.leafTrace ++
    Framed.rows (linkRows role layout.linkFrame) layout.linkTrace

/-- Pure column/schedule certificate. It contains no token, digest, or
row-satisfaction conclusion. -/
structure Layout.Valid
    (manifest : SeedSchedule.Manifest) (role : Role)
    (layout : Layout) : Prop where
  leafTokenColumns :
    layout.leafFrame.tokenColumn = layout.token.tokenOutputColumn
  linkPriorColumns :
    layout.linkFrame.priorDigestColumn = layout.priorDigestColumn
  linkLeafColumns :
    layout.linkFrame.leafDigestColumn =
      fun lane => layout.leafTrace.outputColumns.getD lane.val 0
  linkOutputColumns :
    (fun lane : Fin 4 => layout.linkTrace.outputColumns.getD lane.val 0) =
      layout.afterDigestColumn
  leafInputColumns :
    layout.leafTrace.inputColumns = layout.leafFrame.inputColumns
  leafSchedule :
    valueSchedules layout.leafTrace.rounds =
      compactSchedule
        (.leaf role manifest.profile manifest.plan (fun _ => ⟨0, by decide⟩))
  leafTraceValid :
    layout.leafTrace.Valid
      (Framed.rows (leafRows manifest role layout.leafFrame) layout.leafTrace)
  linkInputColumns :
    layout.linkTrace.inputColumns = layout.linkFrame.inputColumns
  linkSchedule :
    valueSchedules layout.linkTrace.rounds =
      compactSchedule
        (.link role ⟨0, by decide⟩
          { lanes := fun _ => ⟨0, by decide⟩ }
          { lanes := fun _ => ⟨0, by decide⟩ })
  linkTraceValid :
    layout.linkTrace.Valid
      (Framed.rows (linkRows role layout.linkFrame) layout.linkTrace)
  leafTraceRowsLength : layout.leafTrace.rows.length = 10266
  linkTraceRowsLength : layout.linkTrace.rows.length = 2413

theorem rows_length_exact
    {manifest : SeedSchedule.Manifest} {role : Role}
    {layout : Layout} (valid : layout.Valid manifest role) :
    (rows manifest role layout).length = 146773 := by
  simp [rows, CompactTokenRows.rows_length_exact,
    Framed.rows, leafRows_length, linkRows_length,
    valid.leafTraceRowsLength, valid.linkTraceRowsLength]

def outputDigest (trace : Trace) (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP) : Digest.Value :=
  { lanes := fun lane =>
      ⟨assignment (trace.outputColumns.getD lane.val 0), canonical _⟩ }

private theorem token_rows_hold
    {manifest : SeedSchedule.Manifest} {role : Role}
    {layout : Layout}
    {assignment : Nat → Nat}
    (holds : Satisfies (rows manifest role layout) assignment) :
    Satisfies (CompactTokenRows.rows manifest role layout.token) assignment := by
  intro row member
  exact holds row (by simp [rows, member])

private theorem leaf_rows_hold
    {manifest : SeedSchedule.Manifest} {role : Role}
    {layout : Layout}
    {assignment : Nat → Nat}
    (holds : Satisfies (rows manifest role layout) assignment) :
    Satisfies
      (Framed.rows (leafRows manifest role layout.leafFrame) layout.leafTrace)
      assignment := by
  intro row member
  exact holds row (by simp [rows, member])

private theorem link_rows_hold
    {manifest : SeedSchedule.Manifest} {role : Role}
    {layout : Layout}
    {assignment : Nat → Nat}
    (holds : Satisfies (rows manifest role layout) assignment) :
    Satisfies
      (Framed.rows (linkRows role layout.linkFrame) layout.linkTrace)
      assignment := by
  intro row member
  exact holds row (by simp [rows, member])

private theorem leafValid
    {manifest : SeedSchedule.Manifest} {role : Role}
    {layout : Layout}
    (valid : layout.Valid manifest role) (token : Token) :
    Framed.Valid (.leaf role manifest.profile manifest.plan token)
      (leafRows manifest role layout.leafFrame)
      layout.leafFrame.inputColumns layout.leafTrace where
  exactInputColumns := valid.leafInputColumns
  exactSchedule := by simpa [compactSchedule] using valid.leafSchedule
  traceValid := valid.leafTraceValid

private theorem linkValid
    {manifest : SeedSchedule.Manifest} {role : Role}
    {index : Fin claimsPerSegment} {layout : Layout}
    (valid : layout.Valid manifest role)
    (prior leaf : Digest.Value) :
    Framed.Valid (.link role index prior leaf)
      (linkRows role layout.linkFrame)
      layout.linkFrame.inputColumns layout.linkTrace where
  exactInputColumns := valid.linkInputColumns
  exactSchedule := by simpa [compactSchedule] using valid.linkSchedule
  traceValid := valid.linkTraceValid

/-- Main one-lane theorem. The leaf and after-root equations are conclusions
of the token, frame, and Poseidon2 rows. -/
theorem after_root_exact
    {manifest : SeedSchedule.Manifest} {role : Role}
    {index : Fin claimsPerSegment} {layout : Layout}
    {assignment : Nat → Nat} {commitment : CommitmentEncoding}
    {prior after : Digest.Value}
    (valid : layout.Valid manifest role)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (indexPlaced : assignment layout.linkFrame.indexColumn = index.val)
    (commitmentPlaced :
      CompactTokenRows.CommitmentPlaced layout.token assignment commitment)
    (priorPlaced :
      DigestPlaced layout.priorDigestColumn assignment prior)
    (afterPlaced :
      DigestPlaced layout.afterDigestColumn assignment after)
    (holds : Satisfies (rows manifest role layout) assignment) :
    let token := (CompactTokenRows.key manifest).token role commitment
    let leafDigest := outputDigest layout.leafTrace assignment canonical
    (∀ lane : Fin 4,
      (leafDigest.lanes lane).val =
        pureHash (.leaf role manifest.profile manifest.plan token) lane.val) ∧
    (∀ lane : Fin 4,
      (after.lanes lane).val =
        pureHash (.link role index prior leafDigest) lane.val) := by
  let token := (CompactTokenRows.key manifest).token role commitment
  let leafDigest := outputDigest layout.leafTrace assignment canonical
  have tokenExact := CompactTokenRows.token_exact canonical one
    commitmentPlaced (token_rows_hold holds)
  have tokenPlaced :
      TokenPlaced layout.leafFrame assignment token := by
    intro coordinate
    rw [valid.leafTokenColumns]
    exact tokenExact coordinate
  have leafFrameExact := leaf_input_exact canonical one tokenPlaced
    (Framed.frame_rows_hold (leaf_rows_hold holds))
  have leafExact := Framed.output_exact (leafValid valid token)
    canonical one leafFrameExact (leaf_rows_hold holds)
  have leafDigestExact : ∀ lane : Fin 4,
      (leafDigest.lanes lane).val =
        pureHash (.leaf role manifest.profile manifest.plan token) lane.val := by
    intro lane
    exact leafExact lane
  have linkPriorPlaced :
      DigestPlaced layout.linkFrame.priorDigestColumn assignment prior := by
    rw [valid.linkPriorColumns]
    exact priorPlaced
  have linkLeafPlaced :
      DigestPlaced layout.linkFrame.leafDigestColumn assignment leafDigest := by
    intro lane
    rw [valid.linkLeafColumns]
    rfl
  have linkFrameExact := link_input_exact canonical one indexPlaced linkPriorPlaced
    linkLeafPlaced (Framed.frame_rows_hold (link_rows_hold holds))
  have linkExact := Framed.output_exact (linkValid valid prior leafDigest)
    canonical one linkFrameExact (link_rows_hold holds)
  refine ⟨leafDigestExact, ?_⟩
  intro lane
  calc
    (after.lanes lane).val =
        assignment (layout.afterDigestColumn lane) :=
      (afterPlaced lane).symm
    _ = assignment (layout.linkTrace.outputColumns.getD lane.val 0) := by
      rw [← congrFun valid.linkOutputColumns lane]
    _ = pureHash (.link role index prior leafDigest) lane.val :=
      linkExact lane

end Nightstream.Implementation.Nebula.CompactLaneStepRows
