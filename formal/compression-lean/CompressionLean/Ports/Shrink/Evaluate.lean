import CompressionLean.Ports.Shrink.BlockPart
import CompressionLean.Lib.List

namespace CompressionLean.Shrink

open CompressionLean.Lib

def splitEq {K : Type} [CommRing K] (point : List K) (index : ℕ) : K :=
  eqAt (point.take (point.length / 2)) (index % 2 ^ (point.length / 2)) *
    eqAt (point.drop (point.length / 2)) (index / 2 ^ (point.length / 2))

section Glue

variable {F K : Type} [Field F] [Field K] [Algebra F K]

local notation "ι" => algebraMap F K

variable (T : Template F) (s : Counts) (rx ry : List K) (ρ : K)

def sideValue (terms : List (Wire × F)) : K :=
  (terms.map fun t => splitEq ry (Layout.column T s t.1) * ι t.2).sum

def productsValue : ℕ → List (List (Wire × F) × List (Wire × F)) → K
  | _, [] => 0
  | i, (a, b) :: rest =>
      ρ ^ (1 + 2 * i) * sideValue T s ry a + ρ ^ (2 + 2 * i) * sideValue T s ry b + productsValue (i + 1) rest

def rowCallValue (products : List (List (Wire × F) × List (Wire × F))) (linear : List (Wire × F)) : K :=
  ρ ^ 7 * sideValue T s ry linear + productsValue T s ry ρ 0 products

open Classical in
noncomputable def formValue (form : Form F) : K :=
  form.terms.foldl (fun sum t => sum + splitEq ry (Layout.column T s t.1) * ι t.2)
    (if form.constant = 0 then 0 else splitEq ry (Layout.column T s (.pub 0)) * ι form.constant)

noncomputable def r1csSideValue (columns : List (Form F)) (terms : List (ℕ × F)) : K :=
  (terms.map fun t => formValue T s ry (columns.getD t.1 ⟨[], 0⟩) * ι t.2).sum

noncomputable def r1csRowValue (columns : List (Form F)) (row : R1csRow F) : K :=
  ρ ^ 1 * r1csSideValue T s ry columns row.a + ρ ^ 2 * r1csSideValue T s ry columns row.b +
    ρ ^ 7 * r1csSideValue T s ry columns row.c

noncomputable def r1csValue (columns : List (Form F)) : ℕ → List (R1csRow F) → K
  | _, [] => 0
  | g, row :: rest =>
      splitEq rx (Layout.glueRow T s g) * r1csRowValue T s ry ρ columns row + r1csValue columns (g + 1) rest

noncomputable def glueTotal : ℕ → List (Call F) → K
  | _, [] => 0
  | g, .row products linear _ :: rest =>
      splitEq rx (Layout.glueRow T s g) * rowCallValue T s ry ρ products linear + glueTotal (g + 1) rest
  | g, .r1cs rows columns :: rest => r1csValue T s rx ry ρ columns g rows + glueTotal (g + rows.length) rest

end Glue

section Matrices

variable {F K : Type} [Field F] [Field K] [Algebra F K] [DecidableEq K]

noncomputable def verifierMatrices (T : Template F) (s : Counts) (run : Run F) (rx ry : List K) (ρ : K) : K :=
  glueTotal T s rx ry ρ 0 run.calls + blockPart T s rx ry ρ

end Matrices

end CompressionLean.Shrink
