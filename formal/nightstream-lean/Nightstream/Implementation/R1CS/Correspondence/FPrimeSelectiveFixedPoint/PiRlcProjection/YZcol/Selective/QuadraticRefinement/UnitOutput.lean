import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.QuadraticRefinement.StateTransport

/-!
Unit-output decoding for the focused compact `y_zcol` quadratic refinement.

Owns: the exact value of a decoded source output whose authoritative program
form is one unit-coefficient column.

Does not own: terminal-form equivalence, listwise terminal matching,
selected-row materialization, protocol authority, or security events.

Emits constraints: no.

| Decode leaf | Mathematical obligation | Authority class |
|---|---|---|
| unit program terms | the decoded source output is exactly one unit-coefficient column | checked premise |
| source value | evaluating that unit output equals the selected source-field coordinate | derived |
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.QuadraticRefinement

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.ProjectionProgram
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.SourceDecode
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.RewriteBridge

theorem sourceValue_of_unitOutput (assignment : Nat → Nat)
    (output : DecodedSourceLinearCombination) (column : Nat)
    (unit : output.programTerms = [(column, 1)]) :
    sourceValue assignment output = sourceFieldAssignment assignment column := by
  unfold sourceValue sourceFieldAssignment
  rw [unit]
  exact fieldResidue_lcEval_unit (compilerAssignment assignment) column

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.QuadraticRefinement
