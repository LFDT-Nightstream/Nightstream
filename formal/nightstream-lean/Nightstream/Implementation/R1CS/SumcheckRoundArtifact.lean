import Nightstream.Implementation.R1CS.CheckedProgram

/-! Exact isolated production `enforce_sumcheck_round` artifact. -/

namespace Nightstream.Implementation.R1CS.SumcheckRoundArtifact

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.CheckedProgram

def degree : Nat := 4
def coefficientColumns : List (Nat × Nat) := [(2, 8), (3, 9), (4, 10), (5, 11), (6, 12)]
def challengeColumns : Nat × Nat := (13, 15)
def claimInColumns : Nat × Nat := (1, 7)
def claimOutColumns : Nat × Nat := (41, 42)
def inputColumns : List Nat := [0, 2, 8, 3, 9, 4, 10, 5, 11, 6, 12, 13, 15, 1, 7]
def instructions : List Instruction :=
   [.check ⟨[(1, 1), (2, 18446744069414584319), (3, 18446744069414584320), (4, 18446744069414584320), (5, 18446744069414584320), (6, 18446744069414584320)], [(0, 1)], []⟩,
   .check ⟨[(7, 1), (8, 18446744069414584319), (9, 18446744069414584320), (10, 18446744069414584320), (11, 18446744069414584320), (12, 18446744069414584320)], [(0, 1)], []⟩,
   .define ⟨14, .product [(6, 1)] [(13, 1)]⟩,
   .define ⟨16, .product [(12, 1)] [(15, 1)]⟩,
   .define ⟨17, .product [(6, 1), (12, 1)] [(13, 1), (15, 1)]⟩,
   .define ⟨18, .linear [(14, 1), (16, 7)]⟩,
   .define ⟨19, .linear [(17, 1), (14, 18446744069414584320), (16, 18446744069414584320)]⟩,
   .define ⟨20, .linear [(18, 1), (5, 1)]⟩,
   .define ⟨21, .linear [(19, 1), (11, 1)]⟩,
   .define ⟨22, .product [(20, 1)] [(13, 1)]⟩,
   .define ⟨23, .product [(21, 1)] [(15, 1)]⟩,
   .define ⟨24, .product [(20, 1), (21, 1)] [(13, 1), (15, 1)]⟩,
   .define ⟨25, .linear [(22, 1), (23, 7)]⟩,
   .define ⟨26, .linear [(24, 1), (22, 18446744069414584320), (23, 18446744069414584320)]⟩,
   .define ⟨27, .linear [(25, 1), (4, 1)]⟩,
   .define ⟨28, .linear [(26, 1), (10, 1)]⟩,
   .define ⟨29, .product [(27, 1)] [(13, 1)]⟩,
   .define ⟨30, .product [(28, 1)] [(15, 1)]⟩,
   .define ⟨31, .product [(27, 1), (28, 1)] [(13, 1), (15, 1)]⟩,
   .define ⟨32, .linear [(29, 1), (30, 7)]⟩,
   .define ⟨33, .linear [(31, 1), (29, 18446744069414584320), (30, 18446744069414584320)]⟩,
   .define ⟨34, .linear [(32, 1), (3, 1)]⟩,
   .define ⟨35, .linear [(33, 1), (9, 1)]⟩,
   .define ⟨36, .product [(34, 1)] [(13, 1)]⟩,
   .define ⟨37, .product [(35, 1)] [(15, 1)]⟩,
   .define ⟨38, .product [(34, 1), (35, 1)] [(13, 1), (15, 1)]⟩,
   .define ⟨39, .linear [(36, 1), (37, 7)]⟩,
   .define ⟨40, .linear [(38, 1), (36, 18446744069414584320), (37, 18446744069414584320)]⟩,
   .define ⟨41, .linear [(39, 1), (2, 1)]⟩,
   .define ⟨42, .linear [(40, 1), (8, 1)]⟩]
def rows : List Row := CheckedProgram.rows instructions
def honestAssignment : Nat → Nat
  | 0 => 1
  | 1 => 60
  | 2 => 2
  | 3 => 5
  | 4 => 11
  | 5 => 17
  | 6 => 23
  | 7 => 74
  | 8 => 3
  | 9 => 7
  | 10 => 13
  | 11 => 19
  | 12 => 29
  | 13 => 31
  | 14 => 713
  | 15 => 37
  | 16 => 1073
  | 17 => 3536
  | 18 => 8224
  | 19 => 1750
  | 20 => 8241
  | 21 => 1769
  | 22 => 255471
  | 23 => 65453
  | 24 => 680680
  | 25 => 713642
  | 26 => 359756
  | 27 => 713653
  | 28 => 359769
  | 29 => 22123243
  | 30 => 13311453
  | 31 => 72992696
  | 32 => 115303414
  | 33 => 37558000
  | 34 => 115303419
  | 35 => 37558007
  | 36 => 3574405989
  | 37 => 1389646259
  | 38 => 10394576968
  | 39 => 13301929802
  | 40 => 5430524720
  | 41 => 13301929804
  | 42 => 5430524723
  | _ => 0

theorem coefficient_count : coefficientColumns.length = degree + 1 := by native_decide
theorem input_has_one : 0 ∈ inputColumns := by native_decide
theorem definitions_wellFormed : Program.WellFormed inputColumns
    (CheckedProgram.definitions instructions) := by native_decide
theorem definitions_canonical : ∀ definition ∈ CheckedProgram.definitions instructions,
    definition.Canonical := by native_decide
theorem checks_reference : CheckedProgram.ChecksReference
    (Program.knownAfter inputColumns (CheckedProgram.definitions instructions))
    instructions := by native_decide

end Nightstream.Implementation.R1CS.SumcheckRoundArtifact
