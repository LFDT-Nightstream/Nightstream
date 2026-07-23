import Nightstream.Implementation.R1CS.Core.ProjectionProgram
import Nightstream.Implementation.R1CS.Core.Relabel

/-!
Exact semantics of the isolated five-coefficient production SumCheck round.

Owns: the normalized 43-column layout, the two claimed-sum checks, the exact
28-definition quartic Horner program, and soundness under an exact affine
column renaming.

Does not own: the generated production column maps, the fact that any emitted
row interval equals this program, transcript sampling, chain forwarding, or
the terminal NC formula.

Emits constraints: 30 rows: two retained linear checks and four seven-row
quadratic-extension Horner steps.
-/

/-!
| Stable stage path | Obligation | Authority class |
|---|---|---|
| `f_prime.pi_ccs_nc.delayed.combined.production_round` | Give exact semantics for one five-coefficient production SumCheck round. | derived |

-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.ProductionRound

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.CheckedProgram
open Nightstream.Implementation.R1CS.ProjectionProgram

def degree : Nat := 4

/-- Constant-first quartic coefficient columns in the normalized isolated
layout used by the production audit. -/
def coefficientColumns : List (Nat × Nat) :=
  [(2, 8), (3, 9), (4, 10), (5, 11), (6, 12)]

def challengeColumns : Nat × Nat := (13, 15)
def claimInColumns : Nat × Nat := (1, 7)
def claimOutColumns : Nat × Nat := (41, 42)

/-- Columns allocated by `enforce_sumcheck_round` itself. Column 15 predates
the round: it is the high challenge limb, so the first Karatsuba product at 14
is followed by the remaining contiguous allocation interval 16 through 42. -/
def allocatedColumns : List Nat := 14 :: (List.range 43).drop 16

def columns (pair : Nat × Nat) : KColumns := ⟨pair.1, pair.2⟩

def coefficientValues (assignment : Nat → Nat) : List K :=
  coefficientColumns.map fun pair => (columns pair).value assignment

def challengeValue (assignment : Nat → Nat) : K :=
  (columns challengeColumns).value assignment

def claimInValue (assignment : Nat → Nat) : K :=
  (columns claimInColumns).value assignment

def claimOutValue (assignment : Nat → Nat) : K :=
  (columns claimOutColumns).value assignment

def polynomial (assignment : Nat → Nat) (point : K) : K :=
  Nightstream.SuperNeo.ProjectionCheck.eval K.ops
    (coefficientValues assignment) point

structure Accepted (assignment : Nat → Nat) : Prop where
  initial : claimInValue assignment =
    K.add (polynomial assignment K.zero) (polynomial assignment K.one)
  terminal : claimOutValue assignment =
    polynomial assignment (challengeValue assignment)

private def lowTerms : List (Nat × Nat) :=
  [(2, 2), (3, 1), (4, 1), (5, 1), (6, 1)]

private def highTerms : List (Nat × Nat) :=
  [(8, 2), (9, 1), (10, 1), (11, 1), (12, 1)]

private def lowCheck : Row := builderLinearRow 1 lowTerms
private def highCheck : Row := builderLinearRow 7 highTerms

/-- The first multiplication straddles the already allocated high challenge
limb at column 15. All later multiplication outputs use the ordinary
contiguous five-column constructor. -/
private def mul0 : KMulTrace where
  left := KTerms.ofColumns ⟨6, 12⟩
  right := KTerms.ofColumns ⟨13, 15⟩
  sumLeft := [(6, 1), (12, 1)]
  sumRight := [(13, 1), (15, 1)]
  productC0 := 14
  productC1 := 16
  productSum := 17
  output := ⟨18, 19⟩

private def mul1 : KMulTrace :=
  KMulTrace.ofColumns ⟨20, 21⟩ ⟨13, 15⟩ ⟨25, 26⟩

private def mul2 : KMulTrace :=
  KMulTrace.ofColumns ⟨27, 28⟩ ⟨13, 15⟩ ⟨32, 33⟩

private def mul3 : KMulTrace :=
  KMulTrace.ofColumns ⟨34, 35⟩ ⟨13, 15⟩ ⟨39, 40⟩

private def addDefinitions
    (product coefficient output : KColumns) : List Definition :=
  [⟨output.c0, .linear [(product.c0, 1), (coefficient.c0, 1)]⟩,
   ⟨output.c1, .linear [(product.c1, 1), (coefficient.c1, 1)]⟩]

private def add0 : List Definition :=
  addDefinitions ⟨18, 19⟩ ⟨5, 11⟩ ⟨20, 21⟩

private def add1 : List Definition :=
  addDefinitions ⟨25, 26⟩ ⟨4, 10⟩ ⟨27, 28⟩

private def add2 : List Definition :=
  addDefinitions ⟨32, 33⟩ ⟨3, 9⟩ ⟨34, 35⟩

private def add3 : List Definition :=
  addDefinitions ⟨39, 40⟩ ⟨2, 8⟩ ⟨41, 42⟩

private def roundDefinitionSegments : List (List Definition) :=
  [mul0.definitions, add0,
   mul1.definitions, add1,
   mul2.definitions, add2,
   mul3.definitions, add3]

private def roundDefinitions : List Definition :=
  roundDefinitionSegments.flatten

/-- Exact normalized instruction stream of the five-coefficient production
round. The order matches `enforce_sumcheck_round`: two checks followed by the
four Horner steps. -/
def instructions : List Instruction :=
  [.check lowCheck, .check highCheck] ++
    roundDefinitions.map .define

def rows : List Row := CheckedProgram.rows instructions

theorem coefficient_count : coefficientColumns.length = degree + 1 := by
  decide

theorem allocated_column_count : allocatedColumns.length = 28 := by
  decide

theorem row_count : rows.length = 30 := by
  decide

private theorem checked_definitions_eq :
    CheckedProgram.definitions instructions = roundDefinitions := by
  simp [instructions, CheckedProgram.definitions, Function.comp_def]

private theorem checked_checks_eq :
    CheckedProgram.checks instructions = [lowCheck, highCheck] := by
  simp [instructions, CheckedProgram.checks, Function.comp_def]

private theorem roundDefinitions_canonical :
    ∀ definition ∈ roundDefinitions, definition.Canonical := by
  decide

theorem definitions_canonical :
    ∀ definition ∈ CheckedProgram.definitions instructions,
      definition.Canonical := by
  rw [checked_definitions_eq]
  exact roundDefinitions_canonical

private theorem segmentDefinitionsHold
    {assignment : Nat → Nat}
    (definitionsHold : DefinitionsHold assignment roundDefinitions)
    (segment : List Definition)
    (segmentMember : segment ∈ roundDefinitionSegments) :
    DefinitionsHold assignment segment := by
  intro definition definitionMember
  exact definitionsHold definition
    (List.mem_flatten.mpr ⟨segment, segmentMember, definitionMember⟩)

private theorem mul0_layout : mul0.SumLayoutValid := by
  simp [mul0, KMulTrace.SumLayoutValid, KTerms.ofColumns]

private theorem ofColumns_layout
    (left right output : KColumns) :
    (KMulTrace.ofColumns left right output).SumLayoutValid := by
  simp [KMulTrace.ofColumns, KMulTrace.SumLayoutValid, KTerms.ofColumns]

private theorem addOutput_sound
    {assignment : Nat → Nat}
    (product coefficient output : KColumns)
    (definitionsHold :
      DefinitionsHold assignment
        (addDefinitions product coefficient output)) :
    output.value assignment =
      K.add (product.value assignment) (coefficient.value assignment) := by
  have low := definitionsHold
    ⟨output.c0, .linear [(product.c0, 1), (coefficient.c0, 1)]⟩
    (by simp [addDefinitions])
  have high := definitionsHold
    ⟨output.c1, .linear [(product.c1, 1), (coefficient.c1, 1)]⟩
    (by simp [addDefinitions])
  simp only [KColumns.value, K.add, K.mk.injEq]
  constructor
  · apply Fin.ext
    simpa [Definition.Holds, Rhs.eval, KColumns.value, K.add, baseAt,
      residue, lcEval, Fin.val_add] using
        congrArg (fun value => value % goldilocksP) low
  · apply Fin.ext
    simpa [Definition.Holds, Rhs.eval, KColumns.value, K.add, baseAt,
      residue, lcEval, Fin.val_add] using
        congrArg (fun value => value % goldilocksP) high

private theorem terminal_sound
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies rows assignment) :
    claimOutValue assignment =
      polynomial assignment (challengeValue assignment) := by
  have definitionsHold := CheckedProgram.definitionsHold_of_satisfies
    definitions_canonical canonical one satisfies
  rw [checked_definitions_eq] at definitionsHold
  have m0 := mul0.sound assignment mul0_layout
    (segmentDefinitionsHold definitionsHold mul0.definitions (by
      simp [roundDefinitionSegments]))
  have a0 := addOutput_sound ⟨18, 19⟩ ⟨5, 11⟩ ⟨20, 21⟩
    (segmentDefinitionsHold definitionsHold add0 (by
      simp [roundDefinitionSegments]))
  have m1 := mul1.sound assignment
    (ofColumns_layout ⟨20, 21⟩ ⟨13, 15⟩ ⟨25, 26⟩)
    (segmentDefinitionsHold definitionsHold mul1.definitions (by
      simp [roundDefinitionSegments]))
  have a1 := addOutput_sound ⟨25, 26⟩ ⟨4, 10⟩ ⟨27, 28⟩
    (segmentDefinitionsHold definitionsHold add1 (by
      simp [roundDefinitionSegments]))
  have m2 := mul2.sound assignment
    (ofColumns_layout ⟨27, 28⟩ ⟨13, 15⟩ ⟨32, 33⟩)
    (segmentDefinitionsHold definitionsHold mul2.definitions (by
      simp [roundDefinitionSegments]))
  have a2 := addOutput_sound ⟨32, 33⟩ ⟨3, 9⟩ ⟨34, 35⟩
    (segmentDefinitionsHold definitionsHold add2 (by
      simp [roundDefinitionSegments]))
  have m3 := mul3.sound assignment
    (ofColumns_layout ⟨34, 35⟩ ⟨13, 15⟩ ⟨39, 40⟩)
    (segmentDefinitionsHold definitionsHold mul3.definitions (by
      simp [roundDefinitionSegments]))
  have a3 := addOutput_sound ⟨39, 40⟩ ⟨2, 8⟩ ⟨41, 42⟩
    (segmentDefinitionsHold definitionsHold add3 (by
      simp [roundDefinitionSegments]))
  have m0' : (⟨18, 19⟩ : KColumns).value assignment =
      K.mul ((⟨6, 12⟩ : KColumns).value assignment)
        ((⟨13, 15⟩ : KColumns).value assignment) := by
    simpa [mul0] using m0
  have m1' : (⟨25, 26⟩ : KColumns).value assignment =
      K.mul ((⟨20, 21⟩ : KColumns).value assignment)
        ((⟨13, 15⟩ : KColumns).value assignment) := by
    simpa [mul1, KMulTrace.ofColumns] using m1
  have m2' : (⟨32, 33⟩ : KColumns).value assignment =
      K.mul ((⟨27, 28⟩ : KColumns).value assignment)
        ((⟨13, 15⟩ : KColumns).value assignment) := by
    simpa [mul2, KMulTrace.ofColumns] using m2
  have m3' : (⟨39, 40⟩ : KColumns).value assignment =
      K.mul ((⟨34, 35⟩ : KColumns).value assignment)
        ((⟨13, 15⟩ : KColumns).value assignment) := by
    simpa [mul3, KMulTrace.ofColumns] using m3
  unfold claimOutValue polynomial coefficientValues challengeValue
  simp only [coefficientColumns, List.map_cons, List.map_nil,
    Nightstream.SuperNeo.ProjectionCheck.eval, List.foldr_cons,
    List.foldr_nil, K.ops, columns, claimOutColumns, challengeColumns]
  rw [a3, m3', a2, m2', a1, m1', a0, m0']
  simp only [K.mul_zero, K.add_zero]
  simp only [K.add_comm, K.mul_comm]

private theorem initial_sound
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies rows assignment) :
    claimInValue assignment =
      K.add (polynomial assignment K.zero) (polynomial assignment K.one) := by
  have checks := CheckedProgram.checksSatisfy_of_satisfies satisfies
  have lowRow : RowHolds assignment lowCheck := by
    apply checks
    rw [checked_checks_eq]
    simp
  have highRow : RowHolds assignment highCheck := by
    apply checks
    rw [checked_checks_eq]
    simp
  have low := builderLinearRow_sound canonical one 1 lowTerms (by
      decide) lowRow
  have high := builderLinearRow_sound canonical one 7 highTerms (by
      decide) highRow
  have atZero : polynomial assignment K.zero =
      (columns (2, 8)).value assignment := by
    simp [polynomial, coefficientValues, coefficientColumns,
      Nightstream.SuperNeo.ProjectionCheck.eval, K.ops]
  have atOne : polynomial assignment K.one =
      K.add ((columns (2, 8)).value assignment)
        (K.add ((columns (3, 9)).value assignment)
          (K.add ((columns (4, 10)).value assignment)
            (K.add ((columns (5, 11)).value assignment)
              ((columns (6, 12)).value assignment)))) := by
    simp [polynomial, coefficientValues, coefficientColumns,
      Nightstream.SuperNeo.ProjectionCheck.eval, K.ops]
  rw [atZero, atOne]
  simp only [claimInValue, columns, claimInColumns, KColumns.value, K.add,
    K.mk.injEq]
  constructor
  · apply Fin.ext
    have lowMod : assignment 1 =
        (2 * assignment 2 + assignment 3 + assignment 4 + assignment 5 +
          assignment 6) % goldilocksP := by
      simpa [lowTerms, lcEval, Nat.mod_eq_of_lt (canonical 1)] using
        congrArg (fun value => value % goldilocksP) low
    simp only [baseAt, residue, Fin.val_add]
    rw [Nat.mod_eq_of_lt (canonical 1), lowMod]
    simp only [Nat.mod_eq_of_lt (canonical 2),
      Nat.mod_eq_of_lt (canonical 3), Nat.mod_eq_of_lt (canonical 4),
      Nat.mod_eq_of_lt (canonical 5), Nat.mod_eq_of_lt (canonical 6),
      Nat.add_mod_mod]
    congr 1
    omega
  · apply Fin.ext
    have highMod : assignment 7 =
        (2 * assignment 8 + assignment 9 + assignment 10 + assignment 11 +
          assignment 12) % goldilocksP := by
      simpa [highTerms, lcEval, Nat.mod_eq_of_lt (canonical 7)] using
        congrArg (fun value => value % goldilocksP) high
    simp only [baseAt, residue, Fin.val_add]
    rw [Nat.mod_eq_of_lt (canonical 7), highMod]
    simp only [Nat.mod_eq_of_lt (canonical 8),
      Nat.mod_eq_of_lt (canonical 9), Nat.mod_eq_of_lt (canonical 10),
      Nat.mod_eq_of_lt (canonical 11), Nat.mod_eq_of_lt (canonical 12),
      Nat.add_mod_mod]
    congr 1
    omega

/-- Exact-row soundness for the normalized production quartic SumCheck
round. -/
theorem sound
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies rows assignment) :
    Accepted assignment :=
  ⟨initial_sound canonical one satisfies,
    terminal_sound canonical one satisfies⟩

/-- Exact affine renaming rule used by production round maps. Generated data
must separately prove that its concrete 30-row interval is this mapped list. -/
theorem mapped_sound
    (columnMap : List Nat)
    (mapsOne : Relabel.column columnMap 0 = 0)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies (rows.map (Relabel.row columnMap)) assignment) :
    Accepted (Relabel.assignment columnMap assignment) := by
  apply sound (Relabel.canonical canonical)
    (Relabel.constantOne mapsOne one)
  exact (Relabel.satisfies_mapped_iff rows columnMap assignment).mp satisfies

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.ProductionRound
