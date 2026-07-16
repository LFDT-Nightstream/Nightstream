import Nightstream.Implementation.R1CS.Correspondence.FieldEncoding.Refinement.OrdinaryPrivateField
import Nightstream.Implementation.R1CS.Correspondence.FieldEncoding.FreshAssignmentPacking

/-! Narrow compile-time checks for the ordinary-private 41-coordinate
refinement and its still-conditional CE authority boundary. -/

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.CenteredTernaryField
open Nightstream.Implementation.R1CS.CenteredTernaryNormDischarged
open Nightstream.Implementation.R1CS.CenteredTernaryLinearCompiler
open Nightstream.Implementation.R1CS.FPrimeFieldLayout
open Nightstream.Implementation.R1CS.FreshAssignmentPacking
open Nightstream.Implementation.R1CS.OrdinaryPrivateFieldRefinement
open Nightstream.Implementation.R1CS.ShiftedTernaryCompiler
open Nightstream.SuperNeo

example (source : Nat) (index : Fin digitCount) :
    materializeWord source index =
      match ((source + shift) % goldilocksP / 3 ^ index.val) % 3 with
      | 0 => goldilocksP - 1
      | 1 => 0
      | _ => 1 :=
  materializeWord_coordinate source index

example {source : Nat} (canonical : source < goldilocksP) :
    decodeFiniteWord (materializeWord source) = source ∧
      FiniteAlphabetWord (materializeWord source) :=
  materializeWord_represents canonical

example : 3 ^ 40 < goldilocksP ∧ goldilocksP < 3 ^ 41 :=
  threeSymbol_width_boundary

example (fieldCount : Nat) :
    centeredLogicalObligationCount fieldCount = fieldCount * 41 :=
  centeredLogicalObligationCount_eq fieldCount

example (prime : EuclidPrime goldilocksP)
    {fieldCount : Nat}
    (layout : CenteredTernaryLinearCompiler.Layout fieldCount)
    (sourceRows : List Row) (encoded : Nat → Nat)
    (canonical : ∀ column, encoded column < goldilocksP) :
    SafeAccepts layout sourceRows encoded ↔
      Satisfies sourceRows (decodedAssignment layout encoded) ∧
        PrivateCoordinatesCentered layout encoded :=
  safeAccepts_iff prime layout sourceRows encoded canonical

example (prime : EuclidPrime goldilocksP)
    {fieldCount : Nat}
    {layout : CenteredTernaryLinearCompiler.Layout fieldCount}
    (materializer : CenteredTernaryLinearCompiler.HonestMaterializer layout)
    (sourceRows : List Row) {source : Nat → Nat}
    (canonical : ∀ column, source column < goldilocksP)
    (accepted : Satisfies sourceRows source) :
    ∃ encoded,
      SafeAccepts layout sourceRows encoded ∧
        decodedAssignment layout encoded = source :=
  honest_safe_complete prime materializer sourceRows canonical accepted

example {fieldCount : Nat}
    (layout : CenteredTernaryLinearCompiler.Layout fieldCount)
    (sourceRows : List Row) {encoded : Nat → Nat}
    (norm : PrivateCoordinatesNormBoundTwo layout encoded)
    (satisfies : Satisfies (loweredRows layout sourceRows) encoded) :
    Satisfies sourceRows (decodedAssignment layout encoded) ∧
      PrivateCoordinatesCentered layout encoded :=
  normDischargedLowering_sound layout sourceRows norm satisfies

example
    (artifact : GeneratedArtifact)
    (compiler : CompilerBinding artifact)
    (context : Nightstream.SuperNeo.Concrete.Context)
    (params : GlobalParams)
    (statement : Nightstream.SuperNeo.Concrete.CCSStatement)
    (encodedValues : List Nat)
    (assignment : Nightstream.SuperNeo.Concrete.Assignment)
    (authority : FreshCcsNormDischargeAuthority artifact
      context params statement encodedValues assignment) :
    PrivateCoordinatesNormBoundTwo compiler.layout
      (assignmentOf encodedValues) :=
  freshCcsAuthority_privateNorm artifact compiler context params
    statement encodedValues assignment authority

example (z : List Nightstream.SuperNeo.Concrete.F) (i : Nat)
    (iLt : i < z.length) :
    (Nightstream.SuperNeo.Concrete.packAssignment z).getD
        (i / Nightstream.SuperNeo.Concrete.ringDegree)
        Nightstream.SuperNeo.Concrete.ringFZero
        ⟨i % Nightstream.SuperNeo.Concrete.ringDegree,
          Nat.mod_lt _ (by simp [Nightstream.SuperNeo.Concrete.ringDegree])⟩ =
      z.getD i 0 :=
  packAssignment_coordinate z i iLt

example (publicWidth : Nat) (z : List Nightstream.SuperNeo.Concrete.F) :
    Nightstream.SuperNeo.Concrete.projectPublicInput publicWidth z =
      z.take publicWidth :=
  projectPublicInput_eq_take publicWidth z

example (z : List Nightstream.SuperNeo.Concrete.F) (block : Nat)
    (rho : Fin Nightstream.SuperNeo.Concrete.ringDegree)
    (blockLt : block <
      (Nightstream.SuperNeo.Concrete.packAssignment z).length)
    (padding : z.length ≤
      block * Nightstream.SuperNeo.Concrete.ringDegree + rho.val) :
    (Nightstream.SuperNeo.Concrete.packAssignment z).getD block
        Nightstream.SuperNeo.Concrete.ringFZero rho = 0 :=
  packAssignment_padding_zero z block rho blockLt padding

example {left right : List Nightstream.SuperNeo.Concrete.F}
    (sameLength : left.length = right.length)
    (samePacked : Nightstream.SuperNeo.Concrete.packAssignment left =
      Nightstream.SuperNeo.Concrete.packAssignment right) :
    left = right :=
  packAssignment_injective_of_length_eq sameLength samePacked

/-- Equal scalar width is a real authority premise: a trailing scalar zero is
indistinguishable from padding inside the same ring block. -/
example :
    Nightstream.SuperNeo.Concrete.packAssignment
        ([1] : List Nightstream.SuperNeo.Concrete.F) =
      Nightstream.SuperNeo.Concrete.packAssignment
        ([1, 0] : List Nightstream.SuperNeo.Concrete.F) := by
  change
    [fun rho => Nightstream.SuperNeo.Concrete.packedCoeff
      ([1] : List Nightstream.SuperNeo.Concrete.F) 0 rho] =
    [fun rho => Nightstream.SuperNeo.Concrete.packedCoeff
      ([1, 0] : List Nightstream.SuperNeo.Concrete.F) 0 rho]
  congr 1
  funext rho
  rcases rho with ⟨(_ | _ | n), bound⟩
  · rfl
  · rfl
  · rfl
