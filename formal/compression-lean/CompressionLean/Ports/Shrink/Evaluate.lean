import CompressionLean.Ports.Shrink.BlockPart
import CompressionLean.Lib.List

/-!
Owns the port of the verifier's value of `Σ_j ρ^j·M̃_j(rx, ry)` (`matrices`,
`crates/neo-spartan/src/shrink/mod.rs::verify`): the total that the shape run's
sink accumulates (`evaluate.rs`) plus the closed form of `block_part`
(`verifierMatrices`).

- `splitEq` is `SplitEq::at`, the eq weight from a low and a high table.
  `Evaluate::new` builds them for `rx` and `ry` (`evaluate.rs::Evaluate::new`).
- A row call adds `eq(rx, row)·(ρ^7·C + Σ_i ρ^(1+2i)·A_i + ρ^(2+2i)·B_i)`
  over its sides (`Sink::row`). An R1CS call adds the same for each of its
  rows, from each column form's value (`Sink::r1cs`).
- `rho[j]` is `ρ^j` (`powers`, `mod.rs::powers`).

`Shrink/Evaluate.lean` proves its value.
-/

namespace CompressionLean.Shrink

open CompressionLean.Lib

/-- `SplitEq::at` (`evaluate.rs::SplitEq`): the eq table of the low half of
`point` at the low bits of `index`, times the high half's at the rest. -/
def splitEq {K : Type} [CommRing K] (point : List K) (index : ℕ) : K :=
  eqAt (point.take (point.length / 2)) (index % 2 ^ (point.length / 2)) *
    eqAt (point.drop (point.length / 2)) (index / 2 ^ (point.length / 2))

section Glue

variable {F K : Type} [Field F] [Field K] [Algebra F K]

local notation "ι" => algebraMap F K

variable (T : Template F) (s : Counts) (rx ry : List K) (ρ : K)

/-- `Evaluate::side` (`evaluate.rs::Evaluate::side`): `Σ eq(ry, column)·coefficient`. -/
def sideValue (terms : List (Wire × F)) : K :=
  (terms.map fun t => splitEq ry (Layout.column T s t.1) * ι t.2).sum

/-- The products of a row call from product `i` on (`evaluate.rs::Evaluate::row`). -/
def productsValue : ℕ → List (List (Wire × F) × List (Wire × F)) → K
  | _, [] => 0
  | i, (a, b) :: rest =>
      ρ ^ (1 + 2 * i) * sideValue T s ry a + ρ ^ (2 + 2 * i) * sideValue T s ry b + productsValue (i + 1) rest

/-- `Evaluate::row`'s sum (`evaluate.rs::Evaluate::row`): `ρ^7` times the linear
side, plus the products. -/
def rowCallValue (products : List (List (Wire × F) × List (Wire × F))) (linear : List (Wire × F)) : K :=
  ρ ^ 7 * sideValue T s ry linear + productsValue T s ry ρ 0 products

open Classical in
/-- `Form::evaluate` at `ry`'s eq weights (`record.rs::Form::evaluate`), as
`Evaluate::r1cs` calls it (`evaluate.rs::Evaluate::r1cs`). -/
noncomputable def formValue (form : Form F) : K :=
  form.terms.foldl (fun sum t => sum + splitEq ry (Layout.column T s t.1) * ι t.2)
    (if form.constant = 0 then 0 else splitEq ry (Layout.column T s (.pub 0)) * ι form.constant)

/-- One side of an R1CS row from the forms' values (`side`,
`evaluate.rs::Evaluate::r1cs`). -/
noncomputable def r1csSideValue (columns : List (Form F)) (terms : List (ℕ × F)) : K :=
  (terms.map fun t => formValue T s ry (columns.getD t.1 ⟨[], 0⟩) * ι t.2).sum

/-- One R1CS row's sum (`evaluate.rs::Evaluate::r1cs`): `ρ·a + ρ²·b + ρ^7·c`. -/
noncomputable def r1csRowValue (columns : List (Form F)) (row : R1csRow F) : K :=
  ρ ^ 1 * r1csSideValue T s ry columns row.a + ρ ^ 2 * r1csSideValue T s ry columns row.b +
    ρ ^ 7 * r1csSideValue T s ry columns row.c

/-- An R1CS call's rows from glue row `g` on (`evaluate.rs::Evaluate::r1cs`). -/
noncomputable def r1csValue (columns : List (Form F)) : ℕ → List (R1csRow F) → K
  | _, [] => 0
  | g, row :: rest =>
      splitEq rx (Layout.glueRow T s g) * r1csRowValue T s ry ρ columns row + r1csValue columns (g + 1) rest

/-- `total` after the calls from glue row `g` on (`Evaluate`'s `Sink::row`
and `Sink::r1cs`, `evaluate.rs`). -/
noncomputable def glueTotal : ℕ → List (Call F) → K
  | _, [] => 0
  | g, .row products linear _ :: rest =>
      splitEq rx (Layout.glueRow T s g) * rowCallValue T s ry ρ products linear + glueTotal (g + 1) rest
  | g, .r1cs rows columns :: rest => r1csValue T s rx ry ρ columns g rows + glueTotal (g + rows.length) rest

end Glue

section Matrices

variable {F K : Type} [Field F] [Field K] [Algebra F K] [DecidableEq K]

/-- The verifier's `Σ_j ρ^j·M̃_j(rx, ry)` (`matrices`, `mod.rs::verify`): the
shape run's total plus `block_part`. -/
noncomputable def verifierMatrices (T : Template F) (s : Counts) (run : Run F) (rx ry : List K) (ρ : K) : K :=
  glueTotal T s rx ry ρ 0 run.calls + blockPart T s rx ry ρ

end Matrices

end CompressionLean.Shrink
