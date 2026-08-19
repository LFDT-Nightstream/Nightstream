import Nightstream.Implementation.R1CS.Ownership.Sumcheck.SumcheckRoundArtifact
import Nightstream.Implementation.R1CS.Core.ProjectionProgram
import Nightstream.Implementation.R1CS.Core.Relabel

/-!
Exact compiler correspondence for one production degree-four SumCheck round.

The two assertion rows enforce `g(0) + g(1) = claimIn`.  The remaining
twenty-eight exact rows are the four-step quadratic-extension Horner program
and determine `claimOut = g(challenge)` for every canonical satisfying
assignment.
-/

namespace Nightstream.Implementation.R1CS.SumcheckRoundSound

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.ProjectionProgram
open Nightstream.Implementation.R1CS.SumcheckRoundArtifact

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

private def mul0 : KMulTrace where
  left := KTerms.ofColumns ⟨6, 12⟩
  right := KTerms.ofColumns ⟨13, 15⟩
  sumLeft := [(6, 1), (12, 1)]
  sumRight := [(13, 1), (15, 1)]
  productC0 := 14
  productC1 := 16
  productSum := 17
  output := ⟨18, 19⟩

private def mul1 : KMulTrace where
  left := KTerms.ofColumns ⟨20, 21⟩
  right := KTerms.ofColumns ⟨13, 15⟩
  sumLeft := [(20, 1), (21, 1)]
  sumRight := [(13, 1), (15, 1)]
  productC0 := 22
  productC1 := 23
  productSum := 24
  output := ⟨25, 26⟩

private def mul2 : KMulTrace where
  left := KTerms.ofColumns ⟨27, 28⟩
  right := KTerms.ofColumns ⟨13, 15⟩
  sumLeft := [(27, 1), (28, 1)]
  sumRight := [(13, 1), (15, 1)]
  productC0 := 29
  productC1 := 30
  productSum := 31
  output := ⟨32, 33⟩

private def mul3 : KMulTrace where
  left := KTerms.ofColumns ⟨34, 35⟩
  right := KTerms.ofColumns ⟨13, 15⟩
  sumLeft := [(34, 1), (35, 1)]
  sumRight := [(13, 1), (15, 1)]
  productC0 := 36
  productC1 := 37
  productSum := 38
  output := ⟨39, 40⟩

private theorem multiplication_sound
    {assignment : Nat → Nat}
    (definitionsHold : DefinitionsHold assignment
      (CheckedProgram.definitions instructions))
    (trace : KMulTrace)
    (traceMember : ∀ definition ∈ trace.definitions,
      definition ∈ CheckedProgram.definitions instructions)
    (layout : trace.SumLayoutValid) :
    trace.output.value assignment =
      K.mul (trace.left.value assignment) (trace.right.value assignment) := by
  apply trace.sound assignment layout
  intro definition member
  exact definitionsHold definition (traceMember definition member)

private theorem addOutput_sound
    {assignment : Nat → Nat}
    {product coefficient output : KColumns}
    (definitionsHold : DefinitionsHold assignment
      (CheckedProgram.definitions instructions))
    (lowMember :
      (⟨output.c0, .linear [(product.c0, 1), (coefficient.c0, 1)]⟩ : Definition) ∈
        CheckedProgram.definitions instructions)
    (highMember :
      (⟨output.c1, .linear [(product.c1, 1), (coefficient.c1, 1)]⟩ : Definition) ∈
        CheckedProgram.definitions instructions) :
    output.value assignment = K.add (product.value assignment)
      (coefficient.value assignment) := by
  have low := definitionsHold _ lowMember
  have high := definitionsHold _ highMember
  simp only [KColumns.value, K.add, K.mk.injEq]
  constructor
  · apply Fin.ext
    simpa [Definition.Holds, Rhs.eval, KColumns.value, K.add, baseAt,
      residue, lcEval, Fin.val_add] using congrArg (fun value => value % goldilocksP) low
  · apply Fin.ext
    simpa [Definition.Holds, Rhs.eval, KColumns.value, K.add, baseAt,
      residue, lcEval, Fin.val_add] using congrArg (fun value => value % goldilocksP) high

private theorem terminal_sound
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies rows assignment) :
    claimOutValue assignment = polynomial assignment (challengeValue assignment) := by
  have definitionsHold := CheckedProgram.definitionsHold_of_satisfies
    definitions_canonical canonical one satisfies
  have m0 := multiplication_sound definitionsHold mul0 (by native_decide)
    (by native_decide)
  have a0 := addOutput_sound definitionsHold
    (product := ⟨18, 19⟩) (coefficient := ⟨5, 11⟩)
    (output := ⟨20, 21⟩) (by native_decide) (by native_decide)
  have m1 := multiplication_sound definitionsHold mul1 (by native_decide)
    (by native_decide)
  have a1 := addOutput_sound definitionsHold
    (product := ⟨25, 26⟩) (coefficient := ⟨4, 10⟩)
    (output := ⟨27, 28⟩) (by native_decide) (by native_decide)
  have m2 := multiplication_sound definitionsHold mul2 (by native_decide)
    (by native_decide)
  have a2 := addOutput_sound definitionsHold
    (product := ⟨32, 33⟩) (coefficient := ⟨3, 9⟩)
    (output := ⟨34, 35⟩) (by native_decide) (by native_decide)
  have m3 := multiplication_sound definitionsHold mul3 (by native_decide)
    (by native_decide)
  have a3 := addOutput_sound definitionsHold
    (product := ⟨39, 40⟩) (coefficient := ⟨2, 8⟩)
    (output := ⟨41, 42⟩) (by native_decide) (by native_decide)
  have m0' : (⟨18, 19⟩ : KColumns).value assignment =
      K.mul ((⟨6, 12⟩ : KColumns).value assignment)
        ((⟨13, 15⟩ : KColumns).value assignment) := by
    simpa [mul0] using m0
  have m1' : (⟨25, 26⟩ : KColumns).value assignment =
      K.mul ((⟨20, 21⟩ : KColumns).value assignment)
        ((⟨13, 15⟩ : KColumns).value assignment) := by
    simpa [mul1] using m1
  have m2' : (⟨32, 33⟩ : KColumns).value assignment =
      K.mul ((⟨27, 28⟩ : KColumns).value assignment)
        ((⟨13, 15⟩ : KColumns).value assignment) := by
    simpa [mul2] using m2
  have m3' : (⟨39, 40⟩ : KColumns).value assignment =
      K.mul ((⟨34, 35⟩ : KColumns).value assignment)
        ((⟨13, 15⟩ : KColumns).value assignment) := by
    simpa [mul3] using m3
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
  have lowRow : RowHolds assignment
      ⟨[(1, 1), (2, goldilocksP - 2), (3, goldilocksP - 1),
        (4, goldilocksP - 1), (5, goldilocksP - 1),
        (6, goldilocksP - 1)], [(0, 1)], []⟩ := by
    apply checks
    native_decide
  have highRow : RowHolds assignment
      ⟨[(7, 1), (8, goldilocksP - 2), (9, goldilocksP - 1),
        (10, goldilocksP - 1), (11, goldilocksP - 1),
        (12, goldilocksP - 1)], [(0, 1)], []⟩ := by
    apply checks
    native_decide
  have low := builderLinearRow_sound canonical one 1
    [(2, 2), (3, 1), (4, 1), (5, 1), (6, 1)] (by
      simp [CanonicalTerms, goldilocksP]) (by
      simpa [builderLinearRow, negateTerms, negCoeff, goldilocksP] using lowRow)
  have high := builderLinearRow_sound canonical one 7
    [(8, 2), (9, 1), (10, 1), (11, 1), (12, 1)] (by
      simp [CanonicalTerms, goldilocksP]) (by
      simpa [builderLinearRow, negateTerms, negCoeff, goldilocksP] using highRow)
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
      simpa [columns, claimInColumns, KColumns.value, baseAt, residue, K.add,
      K.mul, K.zero, K.one, lcEval, Fin.val_add, Fin.val_mul,
      Nat.mod_eq_of_lt (canonical 1)] using congrArg (fun value => value % goldilocksP) low
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
      simpa [columns, claimInColumns, KColumns.value, baseAt, residue, K.add,
      K.mul, K.zero, K.one, lcEval, Fin.val_add, Fin.val_mul,
      Nat.mod_eq_of_lt (canonical 7)] using congrArg (fun value => value % goldilocksP) high
    simp only [baseAt, residue, Fin.val_add]
    rw [Nat.mod_eq_of_lt (canonical 7), highMod]
    simp only [Nat.mod_eq_of_lt (canonical 8),
      Nat.mod_eq_of_lt (canonical 9), Nat.mod_eq_of_lt (canonical 10),
      Nat.mod_eq_of_lt (canonical 11), Nat.mod_eq_of_lt (canonical 12),
      Nat.add_mod_mod]
    congr 1
    omega

/-- Exact-row CIR-SOUND for the production claimed-chain round. -/
theorem sound
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies rows assignment) :
    Accepted assignment :=
  ⟨initial_sound canonical one satisfies,
    terminal_sound canonical one satisfies⟩

/-- Any exact affine renaming of the isolated round has the same semantics.
This is the call-site compiler rule used by generated FE/NC maps. -/
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

end Nightstream.Implementation.R1CS.SumcheckRoundSound
