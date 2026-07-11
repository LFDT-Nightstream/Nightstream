/-!
Contract: executable R1CS satisfaction semantics for exported artifacts.

Owns: the sparse-row representation matching the Rust `R1csBuilder` triplet
emission (`crates/neo-fold-clean/src/engine/r1cs_circuit/builder.rs`), the
Goldilocks modulus, satisfaction as a decidable predicate, and the generic
soundness lemma for the `enforce_bit` row shape.

Assumes: assignments are canonical residues (`z i < goldilocksP`) with the
constant-one wire at column 0. The Euclid divisor property of the modulus is
a named hypothesis (`EuclidPrime`), never asserted axiomatically: it follows
from primality of `2^64 - 2^32 + 1`, which is not yet reconstructed locally.

Non-goals: commitment openings, witness extraction, and any claim about rows
this module was not instantiated with.
-/

namespace Nightstream.Implementation.R1CS

/-- Goldilocks modulus `2^64 - 2^32 + 1`. -/
def goldilocksP : Nat := 18446744069414584321

/-- One R1CS row: sparse `(column, coefficient)` terms for A, B, C, in the
Rust builder's emission order. Coefficients are canonical residues; the
builder folds row constants into column 0 (the constant-one wire). -/
structure Row where
  a : List (Nat × Nat)
  b : List (Nat × Nat)
  c : List (Nat × Nat)
deriving DecidableEq, Repr

/-- Canonical value of a sparse linear combination at assignment `z`. -/
def lcEval (z : Nat → Nat) (terms : List (Nat × Nat)) : Nat :=
  (terms.foldl (fun acc t => acc + t.2 * z t.1) 0) % goldilocksP

/-- `(A·z) · (B·z) = (C·z)` in the field, for one row. -/
def RowHolds (z : Nat → Nat) (r : Row) : Prop :=
  lcEval z r.a * lcEval z r.b % goldilocksP = lcEval z r.c

instance (z : Nat → Nat) (r : Row) : Decidable (RowHolds z r) := by
  unfold RowHolds; infer_instance

/-- The whole exported constraint block is satisfied. -/
def Satisfies (rows : List Row) (z : Nat → Nat) : Prop :=
  ∀ r ∈ rows, RowHolds z r

instance (rows : List Row) (z : Nat → Nat) : Decidable (Satisfies rows z) := by
  unfold Satisfies; infer_instance

/-- Satisfaction of an exact row schedule is equivalent to satisfaction of
every compact piece.  This lets generated owners interleave ordinary row
segments with compact compiler blocks without densifying either one. -/
theorem satisfies_flatten_iff
    (pieces : List (List Row)) (z : Nat → Nat) :
    Satisfies pieces.flatten z ↔
      ∀ piece ∈ pieces, Satisfies piece z := by
  constructor
  · intro satisfies piece pieceMember row rowMember
    exact satisfies row
      (List.mem_flatten.mpr ⟨piece, pieceMember, rowMember⟩)
  · intro piecesSatisfy row rowMember
    rcases List.mem_flatten.mp rowMember with
      ⟨piece, pieceMember, memberInPiece⟩
    exact piecesSatisfy piece pieceMember row memberInPiece

/-- Assignment view of an exported witness vector (zero beyond its length). -/
def assignmentOf (w : List Nat) : Nat → Nat := fun i => w.getD i 0

/-- Pull an assignment back along a column-renaming map. -/
def pullAssignment (z : Nat → Nat) (f : Nat → Nat) : Nat → Nat :=
  fun i => z (f i)

/-- Rename every column referenced by one sparse linear combination. -/
def renameTerms (f : Nat → Nat) (terms : List (Nat × Nat)) :
    List (Nat × Nat) :=
  terms.map (fun term => (f term.1, term.2))

/-- Rename all columns referenced by an R1CS row. -/
def renameRow (f : Nat → Nat) (r : Row) : Row :=
  ⟨renameTerms f r.a, renameTerms f r.b, renameTerms f r.c⟩

theorem lcEval_pull (z : Nat → Nat) (f : Nat → Nat)
    (terms : List (Nat × Nat)) :
    lcEval (pullAssignment z f) terms = lcEval z (renameTerms f terms) := by
  simp [lcEval, pullAssignment, renameTerms, List.foldl_map]

theorem rowHolds_pull_iff (z : Nat → Nat) (f : Nat → Nat) (r : Row) :
    RowHolds (pullAssignment z f) r ↔ RowHolds z (renameRow f r) := by
  simp only [RowHolds, renameRow, lcEval_pull]

/-- Executable certificate that every row in `small` occurs in `large`. -/
def rowsIncluded (small large : List Row) : Bool :=
  small.all (fun row => decide (row ∈ large))

theorem rowsIncluded_sound {small large : List Row}
    (h : rowsIncluded small large = true) :
    ∀ row ∈ small, row ∈ large := by
  intro row hrow
  have hdecide := (List.all_eq_true.mp h) row hrow
  exact of_decide_eq_true hdecide

/-- A checked row-inclusion certificate transports satisfaction through an
arbitrary column projection. This is the composition rule used to prove a
large production block from its already-proved gadget row programs. -/
theorem satisfies_pull_of_rowsIncluded
    {small large : List Row} {z : Nat → Nat} (f : Nat → Nat)
    (hrows : rowsIncluded (small.map (renameRow f)) large = true)
    (hsat : Satisfies large z) :
    Satisfies small (pullAssignment z f) := by
  intro row hrow
  apply (rowHolds_pull_iff z f row).mpr
  apply hsat
  apply rowsIncluded_sound hrows
  exact List.mem_map.mpr ⟨row, hrow, rfl⟩

/-- The Euclid divisor property of the modulus. For `goldilocksP` this is a
consequence of primality — an explicitly named mathematical boundary passed
to every theorem that needs it (spec §9). -/
def EuclidPrime (q : Nat) : Prop :=
  ∀ a b : Nat, a * b % q = 0 → a % q = 0 ∨ b % q = 0

/-- The exact row shape `enforce_bit` emits: `v · (v - 1) = 0`. -/
def bitRow (c : Nat) : Row :=
  ⟨[(c, 1)], [(c, 1), (0, goldilocksP - 1)], []⟩

/-- A satisfied bit row bounds its wire by 1. -/
theorem bitRow_le_one (hq : EuclidPrime goldilocksP)
    {z : Nat → Nat} {c : Nat}
    (hlt : z c < goldilocksP) (hone : z 0 = 1)
    (h : RowHolds z (bitRow c)) : z c ≤ 1 := by
  simp only [RowHolds, bitRow, lcEval, List.foldl, hone, goldilocksP] at h hlt
  rcases hq _ _ h with h' | h' <;> simp only [goldilocksP] at h' <;> omega

end Nightstream.Implementation.R1CS
