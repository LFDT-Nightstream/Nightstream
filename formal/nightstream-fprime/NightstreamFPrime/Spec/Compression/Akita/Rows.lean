import Mathlib.Data.List.Basic

/-! The rows of the Akita schedule files `crates/ns-compress/artifacts/key20.aks` and
`crates/ns-compress/artifacts/key30.aks` (SPEC.md section 3), and the conditions of Akita
(ePrint 2026/1983, revision of 2026-10-15) Section 9.3 that a file decides.

- `Row` holds the fields of a file row that these conditions read. `tests/AkitaSchedules.lean`
  reads both files and fails when a row of a file is not its Lean row.
- `Row.Methods`, `Row.Compressed` and `Row.RingChecks` are the family F1 to F3. `Row.Interface` is
  I2, I3 and the second part of I1. `Row.Chained` checks the witness lengths in C3.
- The file of the key of 20 variables has these conditions (`key20_conditions`). Each row of the
  file of the key of 30 variables has them without F1 (`key30W27_conditions`,
  `key30W26_conditions`): it uses coefficient packing at levels 2 and 3, and Akita (127) gives
  these levels evaluation trace. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita

/-- The opening method of a fold (Akita (127)): `SubringCoefficientPacking` or `EvaluationTrace`
in the field `opening_method`. -/
inductive Opening
  | packing
  | trace
  deriving DecidableEq, Repr

/-- The norm certificate of a response (Akita (53), `Norm_h`): the coefficient-wise bound or the
squared-norm certificate, which are `Linf` and `L2` in the field `security_route` of the inner
matrix. -/
inductive Norm
  | coefficient
  | euclidean
  deriving DecidableEq, Repr

/-- The ring check of a fold (Akita (53), `RingCheck_h`): `QuotientLift`, or `ReducedEvaluation`
for the quotient-free check, in the field `ring_relation_mode`. -/
inductive RingCheck
  | quotientLift
  | quotientFree
  deriving DecidableEq, Repr

/-- A source group of a level. `setup` is true for the setup-prefix group `S_{h−1}` of Akita (30),
which has the field `setup_natural_len`. -/
structure Group where
  setup : Bool
  opening : Opening
  norm : Norm
  deriving DecidableEq, Repr

/-- A level of a schedule (Akita (33)): its source groups in order, `payload_mode = Compressed`,
the ring check, `num_chunks`, `input_witness_len` and `output_witness_len`. -/
structure Level where
  groups : List Group
  compressed : Bool
  ringCheck : RingCheck
  chunks : ℕ
  source : ℕ
  target : ℕ
  deriving DecidableEq, Repr

/-- A row of a schedule file: the root and the recursive folds in order, which are levels 0, 1 and
so on, and `input_witness_len` of the terminal. A row has one terminal, which is the edge
`TerminalInnerState` of I3. -/
structure Row where
  levels : List Level
  terminal : ℕ
  deriving DecidableEq, Repr

namespace Row

variable (row : Row)

/-- Akita (127): the opening method of a nonterminal fold at `level`. -/
def method (level : ℕ) : Opening := if level < 2 then .packing else .trace

/-- From `level` on, no fold reads a setup group and no fold has response chunks. -/
def Settled (level : ℕ) : Prop :=
  ∀ later ∈ row.levels.drop level, later.chunks = 1 ∧ ∀ group ∈ later.groups, group.setup = false

/-- F1: each fold has the method of Akita (127), a packing fold has the coefficient bound, and the
squared-norm certificate is for the one group of a fold after the setup openings and the response
chunks end. -/
def Methods : Prop :=
  ∀ entry ∈ row.levels.zipIdx, ∀ group ∈ entry.1.groups,
    group.opening = method entry.2 ∧ (group.opening = .packing → group.norm = .coefficient) ∧
      (group.norm = .euclidean → entry.1.groups.length = 1 ∧ row.Settled entry.2)

/-- F2: the root and each fold with a setup group are compressed, and compression does not start
again. -/
def Compressed : Prop :=
  (∀ level ∈ row.levels.zipIdx,
      level.2 = 0 ∨ (∃ group ∈ level.1.groups, group.setup = true) → level.1.compressed = true) ∧
    row.levels.Pairwise fun early late => late.compressed = true → early.compressed = true

/-- F3: the quotient-lift folds are a prefix, and a quotient-free fold is at level 2 or later, uses
evaluation trace, reads no setup group and makes none for the next fold. -/
def RingChecks : Prop :=
  (row.levels.Pairwise fun early late =>
      late.ringCheck = .quotientLift → early.ringCheck = .quotientLift) ∧
    (∀ entry ∈ row.levels.zipIdx, entry.1.ringCheck = .quotientFree →
      2 ≤ entry.2 ∧ ∀ group ∈ entry.1.groups, group.opening = .trace ∧ group.setup = false) ∧
    ∀ edge ∈ row.levels.zip row.levels.tail, edge.1.ringCheck = .quotientFree →
      ∀ group ∈ edge.2.groups, group.setup = false

/-- The family F1 to F3 of Akita Section 9.3. -/
def Family : Prop := row.Methods ∧ row.Compressed ∧ row.RingChecks

/-- The second part of I1, I2 and I3: the root source is a nonempty list of application groups
(Akita (29)), and each later source is `(W_h)` or `(S_{h−1}, W_h)` (Akita (30)). -/
def Interface : Prop :=
  (∃ root ∈ row.levels.head?, root.groups ≠ [] ∧ ∀ group ∈ root.groups, group.setup = false) ∧
    ∀ level ∈ row.levels.tail,
      level.groups.map (·.setup) = [false] ∨ level.groups.map (·.setup) = [true, false]

/-- C3 for the witness lengths: the target length of each fold is the source length of the next
fold or of the terminal. Akita Section 5.1 also requires the successor commitment and ordered
witness segments to match. -/
def Chained : Prop :=
  (∀ edge ∈ row.levels.zip row.levels.tail, edge.1.target = edge.2.source) ∧
    row.levels.getLast?.map (·.target) = some row.terminal

/-- The conditions of Akita Definition 9.2 that a file decides. -/
def Conditions : Prop := row.Family ∧ row.Interface ∧ row.Chained

instance : Decidable row.Methods := by unfold Methods Settled; infer_instance
instance : Decidable row.Compressed := by unfold Compressed; infer_instance
instance : Decidable row.RingChecks := by unfold RingChecks; infer_instance
instance : Decidable row.Interface := by unfold Interface; infer_instance
instance : Decidable row.Chained := by unfold Chained; infer_instance
instance : Decidable row.Conditions := by unfold Conditions Family; infer_instance

end Row

/-- The row of `key20.aks`, for the polynomials of 20, 16 and 16 variables. -/
def key20 : Row where
  levels := [
    ⟨[⟨false, .packing, .coefficient⟩, ⟨false, .packing, .coefficient⟩,
      ⟨false, .packing, .coefficient⟩], true, .quotientLift, 1, 65536, 3659776⟩,
    ⟨[⟨false, .packing, .coefficient⟩], true, .quotientLift, 1, 3659776, 1370752⟩,
    ⟨[⟨false, .trace, .coefficient⟩], true, .quotientFree, 1, 1370752, 330368⟩,
    ⟨[⟨false, .trace, .euclidean⟩], true, .quotientFree, 1, 330368, 214400⟩,
    ⟨[⟨false, .trace, .euclidean⟩], false, .quotientFree, 1, 214400, 109888⟩,
    ⟨[⟨false, .trace, .euclidean⟩], false, .quotientFree, 1, 109888, 77120⟩]
  terminal := 77120

/-- The levels 4 to 6 of each row of `key30.aks`. -/
def key30Tail : List Level := [
  ⟨[⟨false, .trace, .coefficient⟩], true, .quotientFree, 1, 523264, 222208⟩,
  ⟨[⟨false, .trace, .euclidean⟩], false, .quotientFree, 1, 222208, 109888⟩,
  ⟨[⟨false, .trace, .euclidean⟩], false, .quotientFree, 1, 109888, 77120⟩]

/-- The first row of `key30.aks`, for the polynomials of 30, 27 and 28 variables. -/
def key30W27 : Row where
  levels := [
    ⟨[⟨false, .packing, .coefficient⟩, ⟨false, .packing, .coefficient⟩,
      ⟨false, .packing, .coefficient⟩], true, .quotientLift, 1, 268435456, 174909440⟩,
    ⟨[⟨true, .packing, .coefficient⟩, ⟨false, .packing, .coefficient⟩], true, .quotientLift, 1,
      174909440, 11424256⟩,
    ⟨[⟨false, .packing, .coefficient⟩], true, .quotientLift, 1, 11424256, 1463296⟩,
    ⟨[⟨false, .packing, .coefficient⟩], true, .quotientLift, 1, 1463296, 523264⟩] ++ key30Tail
  terminal := 77120

/-- The second row of `key30.aks`, for the polynomials of 30, 26 and 28 variables. -/
def key30W26 : Row where
  levels := [
    ⟨[⟨false, .packing, .coefficient⟩, ⟨false, .packing, .coefficient⟩,
      ⟨false, .packing, .coefficient⟩], true, .quotientLift, 1, 268435456, 165472256⟩,
    ⟨[⟨true, .packing, .coefficient⟩, ⟨false, .packing, .coefficient⟩], true, .quotientLift, 1,
      165472256, 11239936⟩,
    ⟨[⟨false, .packing, .coefficient⟩], true, .quotientLift, 1, 11239936, 1448960⟩,
    ⟨[⟨false, .packing, .coefficient⟩], true, .quotientLift, 1, 1448960, 523264⟩] ++ key30Tail
  terminal := 77120

/-- The row of `key20.aks` is in the family F1 to F3 and has I2, I3, the second part of I1 and
the witness lengths of C3. -/
theorem key20_conditions : key20.Conditions := by decide

/-- The first row of `key30.aks` is not in F1: its folds at levels 2 and 3 use coefficient
packing. It has each other condition of `Row.Conditions`. -/
theorem key30W27_conditions : ¬ key30W27.Methods ∧ key30W27.Compressed ∧ key30W27.RingChecks ∧
    key30W27.Interface ∧ key30W27.Chained := by decide

/-- The second row of `key30.aks` is not in F1, for the same levels. It has each other condition
of `Row.Conditions`. -/
theorem key30W26_conditions : ¬ key30W26.Methods ∧ key30W26.Compressed ∧ key30W26.RingChecks ∧
    key30W26.Interface ∧ key30W26.Chained := by decide

end NightstreamFPrime.Spec.Compression.Akita
