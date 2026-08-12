import Nightstream.Implementation.NebulaV2.CommitmentBundleCodec
import Nightstream.Implementation.R1CS.Core.EqualityPins

/-!
Contract: exact equality-row refinement for one mandatory V2 commitment bundle.

Assurance tier: implementation model.

Owns one R1CS equality row for each of the 248,832 canonical bundle bits,
the link from those row columns to the independently defined bundle codec,
and the proof that satisfying all rows forwards the complete four-component
bundle without change.

Does not own an absolute generated-column manifest, generated-row inclusion,
PiCCS arithmetic, PiRLC, PiDEC, Ajtai binding, or the terminal opening.

Emits constraints: yes, through `EqualityPins.rows`.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.NebulaV2.BundleForwardingRows

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.EqualityPins
open Nightstream.Implementation.NebulaV2.CommitmentBundleCodec
open Nightstream.Protocol.NebulaV2.MemoryWireGeometry

/-- Relative columns for the bundle before and after one forwarding step. -/
structure Layout where
  inputStart : Nat
  outputStart : Nat
deriving DecidableEq, Repr

def Layout.pairAt (layout : Layout) (index : Nat) : Nat × Nat :=
  (layout.outputStart + index, layout.inputStart + index)

def Layout.pairs (layout : Layout) : List (Nat × Nat) :=
  (List.range mandatoryBundleBits).map layout.pairAt

def rows (layout : Layout) : List Row :=
  EqualityPins.rows layout.pairs

theorem Layout.pairs_length (layout : Layout) :
    layout.pairs.length = mandatoryBundleBits := by
  simp [Layout.pairs]

theorem rows_length (layout : Layout) :
    (rows layout).length = mandatoryBundleBits := by
  simp [rows, EqualityPins.rows, layout.pairs_length]

def bitAt (value : Value) (index : Fin mandatoryBundleBits) : Nat :=
  (encode value).get
    ⟨index.val, by simpa [encode_length] using index.isLt⟩

/-- The assignment columns contain the independent bundle codec bits. -/
def Placed (layout : Layout) (assignment : Nat → Nat)
    (input output : Value) : Prop :=
  ∀ index : Fin mandatoryBundleBits,
    assignment (layout.inputStart + index.val) = bitAt input index ∧
      assignment (layout.outputStart + index.val) = bitAt output index

def RowsHold (layout : Layout) (assignment : Nat → Nat) : Prop :=
  Satisfies (rows layout) assignment

private theorem pairAt_mem (layout : Layout)
    (index : Fin mandatoryBundleBits) :
    layout.pairAt index.val ∈ layout.pairs := by
  exact List.mem_map.mpr
    ⟨index.val, List.mem_range.mpr index.isLt, rfl⟩

theorem bitAt_equal_of_rows
    {layout : Layout} {assignment : Nat → Nat} {input output : Value}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (placed : Placed layout assignment input output)
    (holds : RowsHold layout assignment) :
    ∀ index, bitAt output index = bitAt input index := by
  intro index
  have columnEqual := EqualityPins.rows_sound canonical one holds
    (layout.pairAt index.val) (pairAt_mem layout index)
  simp only [Layout.pairAt] at columnEqual
  rw [(placed index).2, (placed index).1] at columnEqual
  exact columnEqual

theorem encode_equal_of_rows
    {layout : Layout} {assignment : Nat → Nat} {input output : Value}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (placed : Placed layout assignment input output)
    (holds : RowsHold layout assignment) :
    encode output = encode input := by
  apply List.ext_get
  · rw [encode_length, encode_length]
  · intro index outputBound inputBound
    let bounded : Fin mandatoryBundleBits :=
      ⟨index, by simpa [encode_length] using outputBound⟩
    simpa [bitAt, bounded] using
      bitAt_equal_of_rows canonical one placed holds bounded

/-- Satisfying the exact forwarding rows forwards all four commitment
components. No digest or sidecar flag can replace this equality. -/
theorem exact_bundle_forwarding
    {layout : Layout} {assignment : Nat → Nat} {input output : Value}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (placed : Placed layout assignment input output)
    (holds : RowsHold layout assignment) :
    output = input := by
  exact CommitmentBundleCodec.encode_injective
    (encode_equal_of_rows canonical one placed holds)

/-- Equal input and output bundles satisfy the local forwarding block when
their codec bits are placed at the declared columns. -/
theorem rows_complete
    {layout : Layout} {assignment : Nat → Nat} {input output : Value}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (placed : Placed layout assignment input output)
    (equal : output = input) :
    RowsHold layout assignment := by
  apply EqualityPins.rows_complete canonical one
  intro pair member
  rcases List.mem_map.mp member with ⟨index, indexMember, pairEqual⟩
  subst pair
  let bounded : Fin mandatoryBundleBits :=
    ⟨index, List.mem_range.mp indexMember⟩
  calc
    assignment (layout.outputStart + index) = bitAt output bounded :=
      (placed bounded).2
    _ = bitAt input bounded := by rw [equal]
    _ = assignment (layout.inputStart + index) := (placed bounded).1.symm

/-- Artifact-facing certificate. It records exact row inclusion and exact
codec-bit placement. It does not let a caller assert bundle equality. -/
structure CallSite (programRows : List Row) (assignment : Nat → Nat)
    (input output : Value) where
  layout : Layout
  rowsIncluded : rowsIncluded (rows layout) programRows = true
  canonicalAssignment : ∀ column, assignment column < goldilocksP
  one : assignment 0 = 1
  placed : Placed layout assignment input output

theorem CallSite.sound
    {programRows : List Row} {assignment : Nat → Nat}
    {input output : Value}
    (site : CallSite programRows assignment input output)
    (satisfies : Satisfies programRows assignment) :
    output = input := by
  apply CommitmentBundleCodec.encode_injective
  apply List.ext_get
  · rw [encode_length, encode_length]
  · intro index outputBound inputBound
    let bounded : Fin mandatoryBundleBits :=
      ⟨index, by simpa [encode_length] using outputBound⟩
    have columnEqual := EqualityPins.sound site.rowsIncluded
      site.canonicalAssignment site.one satisfies
      (site.layout.pairAt bounded.val) (pairAt_mem site.layout bounded)
    simp only [Layout.pairAt] at columnEqual
    rw [(site.placed bounded).2, (site.placed bounded).1] at columnEqual
    simpa [bitAt, bounded] using columnEqual

end Nightstream.Implementation.NebulaV2.BundleForwardingRows
