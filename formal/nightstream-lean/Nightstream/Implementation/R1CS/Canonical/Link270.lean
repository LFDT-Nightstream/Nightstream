import Nightstream.Implementation.R1CS.Core.Semantics
import Nightstream.Implementation.Lowering.Typed.Cost

/-!
Contract: the canonical Lean-owned encoding of the fixed-active 270-coordinate
terminal link.

Owns: the paper-level link relation; one canonical coordinatewise-affine
encoding whose every coefficient and column identity is computed here; the
row count derived from that construction rather than declared; exact row and
column ownership; and per-coordinate necessity.

Does not own: any Rust artifact, generated row, captured coefficient, or
measured dimension.  No value in this module is copied from an emitter.  The
comparison against production lives in `Link270Production` and may not feed
back into these definitions.

Authority boundary: the count `270` is a consequence of `carrierWidth`, which
is `ringDegree * publicRingColumns` for the selected Phi81 profile.  It is
never read from an artifact.

| Obligation | Lean owner |
|---|---|
| link relation | `Link270Holds` |
| canonical encoding | `canonicalRows` |
| soundness and completeness | `canonicalRows_holds_iff` |
| derived row count | `canonicalRows_length` |
| column ownership | `sourceColumn_injective`, `destinationColumn_injective`, `columns_disjoint` |
| per-coordinate necessity | `dropCoordinate_admits_violation` |
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Canonical.Link270

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.Lowering

/-! ## Profile

The width is derived from the selected Phi81 ring shape, not asserted. -/

/-- Degree of the Phi81 cyclotomic ring, `Φ = X^54 + X^27 + 1`. -/
def ringDegree : Nat := 54

/-- Public ring columns in the fixed-active carrier. -/
def publicRingColumns : Nat := 5

/-- The complete paper carrier width.  `270` is a computed consequence. -/
def carrierWidth : Nat := ringDegree * publicRingColumns

theorem carrierWidth_eq : carrierWidth = 270 := by decide

/-! ## Semantics

The link asserts coordinatewise equality of two complete carriers.  Every
coordinate is authoritative, including the thirteen beyond the 257 external
ones: a running carrier is live on all of them. -/

/-- Paper-level link relation. -/
def Link270Holds (source destination : Fin carrierWidth → Nat) : Prop :=
  ∀ i : Fin carrierWidth, destination i = source i

instance (source destination : Fin carrierWidth → Nat) :
    Decidable (Link270Holds source destination) := by
  unfold Link270Holds; infer_instance

/-! ## Canonical column allocation

Source and destination occupy separately owned preallocated blocks.  Column 0
is the constant-one wire, so both blocks start after it. -/

/-- First column of the source block. -/
def sourceBase : Nat := 1

/-- First column of the destination block. -/
def destinationBase : Nat := sourceBase + carrierWidth

def sourceColumn (i : Fin carrierWidth) : Nat := sourceBase + i.val

def destinationColumn (i : Fin carrierWidth) : Nat := destinationBase + i.val

theorem sourceColumn_injective :
    Function.Injective sourceColumn := by
  intro left right equal
  apply Fin.ext
  simpa [sourceColumn, sourceBase] using equal

theorem destinationColumn_injective :
    Function.Injective destinationColumn := by
  intro left right equal
  apply Fin.ext
  simpa [destinationColumn, destinationBase, sourceBase] using equal

/-- The two blocks never collide, and neither touches the constant wire. -/
theorem columns_disjoint (i j : Fin carrierWidth) :
    sourceColumn i ≠ destinationColumn j := by
  have bound := i.isLt
  simp only [sourceColumn, destinationColumn, destinationBase, sourceBase,
    carrierWidth, ringDegree, publicRingColumns] at bound ⊢
  omega

theorem sourceColumn_ne_constant (i : Fin carrierWidth) :
    sourceColumn i ≠ 0 := by
  simp [sourceColumn, sourceBase]

theorem destinationColumn_ne_constant (i : Fin carrierWidth) :
    destinationColumn i ≠ 0 := by
  simp [destinationColumn, destinationBase, sourceBase]

/-! ## Canonical encoding

Normal form: one affine equality per coordinate, no packing and no
cross-coordinate compression.  Each row is `(dest_i - src_i) * 1 = 0`,
emitted as `A = [(dest_i, 1), (src_i, p-1)]`, `B = [(0, 1)]`, `C = []`.

Every coefficient here is chosen by this module. -/

/-- The canonical row for one coordinate. -/
def coordinateRow (i : Fin carrierWidth) : Row where
  a := [(destinationColumn i, 1), (sourceColumn i, goldilocksP - 1)]
  b := [(0, 1)]
  c := []

/-- The canonical encoding of the complete link. -/
def canonicalRows : List Row :=
  (List.finRange carrierWidth).map coordinateRow

/-- The row count is *derived* from the construction. -/
theorem canonicalRows_length : canonicalRows.length = carrierWidth := by
  simp [canonicalRows]

/-- Hence the concrete count, as a consequence rather than an input. -/
theorem canonicalRows_length_eq : canonicalRows.length = 270 := by
  rw [canonicalRows_length]; exact carrierWidth_eq

/-! ## Canonical assignments

An assignment is *canonical for the link* when the constant wire is one, every
value is a canonical residue, and the two blocks read the two carriers. -/

structure CanonicalAssignment
    (source destination : Fin carrierWidth → Nat) (z : Nat → Nat) : Prop where
  constantOne : z 0 = 1
  residues : ∀ column, z column < goldilocksP
  readsSource : ∀ i, z (sourceColumn i) = source i
  readsDestination : ∀ i, z (destinationColumn i) = destination i

/-! ## The affine-equality core -/

/-- For canonical residues, the affine combination vanishes exactly on
equality.  This is where `goldilocksP - 1` acts as `-1`. -/
theorem affine_zero_iff (d s : Nat)
    (dBound : d < goldilocksP) (sBound : s < goldilocksP) :
    (d + (goldilocksP - 1) * s) % goldilocksP = 0 ↔ d = s := by
  have positive : 0 < goldilocksP := by decide
  have expand : (goldilocksP - 1) * s = goldilocksP * s - s := by
    rw [Nat.sub_one_mul]
  rw [expand]
  by_cases le : s ≤ d
  · have sourceLe : s ≤ goldilocksP * s := Nat.le_mul_of_pos_left s positive
    have rearrange : d + (goldilocksP * s - s) = (d - s) + goldilocksP * s := by
      omega
    rw [rearrange, Nat.add_mul_mod_self_left]
    have small : d - s < goldilocksP := by omega
    rw [Nat.mod_eq_of_lt small]
    omega
  · have le : d < s := Nat.lt_of_not_le le
    have rearrange : d + (goldilocksP * s - s) =
        goldilocksP * (s - 1) + (goldilocksP - (s - d)) := by
      have step : goldilocksP * s = goldilocksP * (s - 1) + goldilocksP := by
        cases s with
        | zero => omega
        | succ n => simp [Nat.mul_succ]
      omega
    rw [rearrange, Nat.mul_add_mod]
    have small : goldilocksP - (s - d) < goldilocksP := by omega
    rw [Nat.mod_eq_of_lt small]
    omega

/-! ## Soundness and honest completeness -/

/-- One canonical row holds exactly when its coordinate agrees. -/
theorem coordinateRow_holds_iff
    {source destination : Fin carrierWidth → Nat} {z : Nat → Nat}
    (canonical : CanonicalAssignment source destination z)
    (i : Fin carrierWidth) :
    RowHolds z (coordinateRow i) ↔ destination i = source i := by
  unfold RowHolds coordinateRow lcEval
  simp only [List.foldl, Nat.zero_add, canonical.constantOne, Nat.mul_one,
    Nat.one_mul, canonical.readsSource, canonical.readsDestination]
  have destinationBound : destination i < goldilocksP := by
    rw [← canonical.readsDestination i]; exact canonical.residues _
  have sourceBound : source i < goldilocksP := by
    rw [← canonical.readsSource i]; exact canonical.residues _
  rw [Nat.mod_eq_of_lt (by decide : (1 : Nat) < goldilocksP), Nat.mul_one,
    Nat.mod_mod, Nat.zero_mod]
  exact affine_zero_iff (destination i) (source i) destinationBound sourceBound

/-- **Soundness and honest completeness.**  The canonical encoding is
satisfied exactly by the assignments whose carriers are linked. -/
theorem canonicalRows_holds_iff
    {source destination : Fin carrierWidth → Nat} {z : Nat → Nat}
    (canonical : CanonicalAssignment source destination z) :
    Satisfies canonicalRows z ↔ Link270Holds source destination := by
  constructor
  · intro satisfies i
    refine (coordinateRow_holds_iff canonical i).1 (satisfies _ ?_)
    exact List.mem_map.mpr ⟨i, List.mem_finRange i, rfl⟩
  · intro link row member
    rcases List.mem_map.1 member with ⟨i, _, rfl⟩
    exact (coordinateRow_holds_iff canonical i).2 (link i)

/-! ## Ownership and conservation

Each coordinate owns exactly one row, and every emitted row is owned. -/

/-- Every emitted row is some coordinate's row: no row exists outside the
per-coordinate receipts. -/
theorem canonicalRows_owned (row : Row) (member : row ∈ canonicalRows) :
    ∃ i : Fin carrierWidth, row = coordinateRow i := by
  rcases List.mem_map.1 member with ⟨i, _, rfl⟩
  exact ⟨i, rfl⟩

/-- Distinct coordinates emit distinct rows, so ownership is exact rather than
merely surjective. -/
theorem coordinateRow_injective : Function.Injective coordinateRow := by
  intro left right equal
  have columns : destinationColumn left = destinationColumn right := by
    have := congrArg Row.a equal
    simpa [coordinateRow] using congrArg (fun l => l.headD (0, 0)) this
  exact destinationColumn_injective columns

/-! ## Per-coordinate necessity

Dropping any single coordinate's row admits an assignment that satisfies the
remaining rows while violating the link at exactly that coordinate.  This is
minimality *within the declared coordinatewise-affine normal form* — it is not
a claim that no other encoding uses fewer rows. -/

/-- The encoding with coordinate `i`'s obligation removed. -/
def rowsWithout (i : Fin carrierWidth) : List Row :=
  ((List.finRange carrierWidth).filter (fun j => j ≠ i)).map coordinateRow

/-- A witness that differs from the source at exactly coordinate `i`. -/
def violatingDestination
    (source : Fin carrierWidth → Nat) (i : Fin carrierWidth) :
    Fin carrierWidth → Nat :=
  fun j => if j = i then (source j + 1) % goldilocksP else source j

theorem violatingDestination_off
    (source : Fin carrierWidth → Nat) (i j : Fin carrierWidth) (ne : j ≠ i) :
    violatingDestination source i j = source j := by
  simp [violatingDestination, ne]

theorem violatingDestination_at
    (source : Fin carrierWidth → Nat) (i : Fin carrierWidth)
    (bound : source i < goldilocksP) :
    violatingDestination source i i ≠ source i := by
  have modulus : 1 < goldilocksP := by decide
  have reduce : violatingDestination source i i = (source i + 1) % goldilocksP := by
    simp [violatingDestination]
  rw [reduce]
  by_cases top : source i + 1 = goldilocksP
  · rw [top, Nat.mod_self]; omega
  · rw [Nat.mod_eq_of_lt (by omega)]; omega

/-- **Necessity.**  Every coordinate's row is required: without it the link can
be violated at exactly that coordinate while all remaining rows still hold. -/
theorem dropCoordinate_admits_violation
    {source destination : Fin carrierWidth → Nat} {z : Nat → Nat}
    (i : Fin carrierWidth)
    (canonical : CanonicalAssignment source destination z)
    (violates : destination = violatingDestination source i) :
    Satisfies (rowsWithout i) z ∧ ¬ Link270Holds source destination := by
  constructor
  · intro row member
    rcases List.mem_map.1 member with ⟨j, memberFiltered, rfl⟩
    have ne : j ≠ i := by
      have := (List.mem_filter.1 memberFiltered).2
      simpa using this
    refine (coordinateRow_holds_iff canonical j).2 ?_
    rw [violates, violatingDestination_off source i j ne]
  · intro link
    have bound : source i < goldilocksP := by
      rw [← canonical.readsSource i]; exact canonical.residues _
    refine violatingDestination_at source i bound ?_
    rw [← violates]; exact link i

/-! ## Carrier coordinates

The thirteen coordinates past the 257 external ones are ordinary authoritative
coordinates of this encoding.  Nothing here pins them to zero. -/

/-- The external prefix width of the fixed-active fresh input. -/
def legacyPublicWidth : Nat := 257

/-- The completion coordinates `257 .. 269`. -/
def firstTail : Fin carrierWidth := ⟨legacyPublicWidth, by decide⟩

def lastTail : Fin carrierWidth := ⟨carrierWidth - 1, by decide⟩

theorem firstTail_val : firstTail.val = 257 := by decide

theorem lastTail_val : lastTail.val = 269 := by decide

/-- Coordinate 257 is linked by a row of exactly the same shape as coordinate
0: the encoding gives the tail no special treatment. -/
theorem tail_row_is_ordinary :
    coordinateRow firstTail =
      { a := [(destinationColumn firstTail, 1),
              (sourceColumn firstTail, goldilocksP - 1)],
        b := [(0, 1)], c := [] } := rfl

/-- A carrier with a nonzero tail is linked exactly like any other: the
encoding never forces `destination i = 0`. -/
theorem nonzeroTail_linked
    {source destination : Fin carrierWidth → Nat} {z : Nat → Nat}
    (canonical : CanonicalAssignment source destination z)
    (satisfied : Satisfies canonicalRows z) :
    destination firstTail = source firstTail ∧
      destination lastTail = source lastTail :=
  ⟨(canonicalRows_holds_iff canonical).1 satisfied firstTail,
   (canonicalRows_holds_iff canonical).1 satisfied lastTail⟩

/-! ## Complete cost tuple

The encoding's cost is not only its row count.  It preallocates two column
blocks and reads the shared constant wire, and both must be accounted. -/

/-- Every column this encoding allocates: the source block then the
destination block.  Column `0` is the shared constant wire and is *read*, not
allocated, so it does not appear here. -/
def allocatedColumns : List Nat :=
  (List.range (2 * carrierWidth)).map (fun offset => sourceBase + offset)

theorem allocatedColumns_length :
    allocatedColumns.length = 2 * carrierWidth := by
  simp [allocatedColumns]

/-- The source block is allocated. -/
theorem sourceColumn_allocated (i : Fin carrierWidth) :
    sourceColumn i ∈ allocatedColumns := by
  refine List.mem_map.2 ⟨i.val, List.mem_range.2 ?_, rfl⟩
  have := i.isLt; omega

/-- The destination block is allocated: it is the second half of the same
consecutive block, since `destinationBase = sourceBase + carrierWidth`. -/
theorem destinationColumn_allocated (i : Fin carrierWidth) :
    destinationColumn i ∈ allocatedColumns := by
  refine List.mem_map.2 ⟨carrierWidth + i.val, List.mem_range.2 ?_, ?_⟩
  · have := i.isLt; omega
  · simp [destinationColumn, destinationBase, Nat.add_assoc]

/-- The derived column cost: `540`. -/
theorem allocatedColumns_length_eq : allocatedColumns.length = 540 := by
  rw [allocatedColumns_length]; decide

/-- Allocation is exact: distinct coordinates own distinct source columns,
distinct destination columns, and the two blocks never collide.  This is the
ownership content that a `Nodup` restatement would carry. -/
theorem allocation_exact :
    Function.Injective sourceColumn ∧
      Function.Injective destinationColumn ∧
      (∀ i j : Fin carrierWidth, sourceColumn i ≠ destinationColumn j) :=
  ⟨sourceColumn_injective, destinationColumn_injective, columns_disjoint⟩

/-- The shared constant wire is read but never allocated. -/
theorem constantWire_not_allocated : (0 : Nat) ∉ allocatedColumns := by
  intro member
  rcases List.mem_map.1 member with ⟨offset, _, equal⟩
  simp [sourceBase] at equal

/-- Columns touched by one row. -/
def rowColumns (row : Row) : List Nat :=
  (row.a ++ row.b ++ row.c).map Prod.fst

/-- **Conservation.**  Every column any emitted row touches is either the
shared constant wire or one this encoding allocated.  Nothing is read or
written outside the receipts. -/
theorem rowColumns_accounted
    (row : Row) (member : row ∈ canonicalRows) (column : Nat)
    (touched : column ∈ rowColumns row) :
    column = 0 ∨ column ∈ allocatedColumns := by
  rcases canonicalRows_owned row member with ⟨i, rfl⟩
  simp [rowColumns, coordinateRow] at touched
  rcases touched with rfl | rfl | rfl
  · exact Or.inr (destinationColumn_allocated i)
  · exact Or.inr (sourceColumn_allocated i)
  · exact Or.inl rfl

/-! ### Referenced versus allocated

`allocatedColumns` above names the 540 columns this encoding *reads and
writes*.  It does **not** allocate them: both blocks are preallocated inputs
supplied by the surrounding frame, and the constant wire is a shared read.
The link instruction introduces no temporaries of its own.

Distinguishing these matters for composition.  If every instruction that
touches the carrier reported 540 columns, a program containing several such
instructions would count the same columns repeatedly. -/

/-- The columns this encoding references.  Named to prevent it being read as
an allocation. -/
abbrev referencedColumns : List Nat := allocatedColumns

theorem referencedColumns_length_eq : referencedColumns.length = 540 :=
  allocatedColumns_length_eq

/-- **Incremental cost in the project's canonical `Typed.Cost`.**  The link
contributes its rows and *no* new columns, because both carrier blocks are
preallocated inputs and it introduces no temporaries.  Composing this receipt
therefore cannot double-count a column some other instruction owns. -/
def link270Cost : Typed.Cost :=
  ⟨canonicalRows.length, 0, 0, 0⟩

/-- The derived incremental cost: 270 rows, zero newly allocated columns. -/
theorem link270Cost_eq : link270Cost = ⟨270, 0, 0, 0⟩ := by
  unfold link270Cost
  rw [canonicalRows_length_eq]

/-- The two figures are different quantities and must not be confused: 270
rows and 0 allocated columns is the *incremental* cost; 540 is the number of
preallocated columns *referenced*. -/
theorem cost_references_without_allocating :
    link270Cost = ⟨270, 0, 0, 0⟩ ∧ referencedColumns.length = 540 :=
  ⟨link270Cost_eq, referencedColumns_length_eq⟩

end Nightstream.Implementation.R1CS.Canonical.Link270
