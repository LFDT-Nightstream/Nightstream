import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.KPiRlcPhysicalOccurrence
import Nightstream.Implementation.R1CS.Canonical.KPiRlcSemanticBinding

/-!
Contract: compose the typed physical public-PiRLC occurrence with its selected
paper semantics.

Typed row satisfaction is first pulled back through the occurrence's explicit
column map.  The canonical quotient program then yields either the exact
`PiRLC.Equations` for the same source columns or the occurrence-bound
projection-root event.

This module does not derive the projection challenge from a transcript, bound
the named event, or claim that the public PiRLC occurrence is the complete
NIFS verifier.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.KPiRlcPhysicalSemantic

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Folding
open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.Goldilocks.NumericRowBridge
open Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper
open Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc

namespace Semantic

abbrev ProjectionColumns
    (params : GlobalParams) (arity : BatchArity params)
    (matrixCount : Nat) :=
  Nightstream.Implementation.R1CS.Canonical.KPiRlcSemanticBinding.ProjectionColumns
    params arity matrixCount

end Semantic

namespace Physical

abbrev Occurrence
    {arity matrixCount : Nat}
    (base : Nat)
    (columns :
      Nightstream.Implementation.R1CS.Canonical.KPiRlcTrace.Columns
        arity matrixCount)
    (valid : columns.Valid) :=
  Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.KPiRlcPhysicalOccurrence.PhysicalOccurrence
    base columns valid

end Physical

/-- The typed constant wire induces the numeric representative required by
the canonical quotient program. -/
private theorem numeric_constant
    {arity matrixCount base : Nat}
    {columns :
      Nightstream.Implementation.R1CS.Canonical.KPiRlcTrace.Columns
        arity matrixCount}
    {valid : columns.Valid}
    (physical : Physical.Occurrence base columns valid)
    (assignment : ColumnId → Nightstream.SuperNeo.Concrete.F)
    (constantWire : assignment (physical.map 0) = 1) :
    numericAssignment physical.map assignment 0 = 1 := by
  unfold numericAssignment
  have values := congrArg Fin.val constantWire
  simpa using values

/-- Complete public-PiRLC physical/semantic refinement.

The conclusion names the exact paper equations and the exact bad-root event
bound to this occurrence.  No public equation, acceptance bit, source-binding
escape, or unrelated bad event is supplied by the caller. -/
theorem equations_or_badRoot_of_typed_rows
    {Assignment : Type}
    {params : GlobalParams} {arity : BatchArity params}
    {matrixCount base : Nat}
    {semantics :
      RelationSemantics Unit Assignment PackedPublicInput
        Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.Point
        Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.Evaluation
        PackedCommitment}
    (algebra :
      PiRLC.Algebra Unit Assignment PackedPublicInput
        Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.Point
        Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.Evaluation
        PackedCommitment Ring semantics params)
    (codec : CarrierCodec matrixCount)
    (ring : RingAlgebra)
    (algebraRefinement : AlgebraRefinement algebra codec ring)
    (columns : Semantic.ProjectionColumns params arity matrixCount)
    (valid : columns.Valid)
    (physical :
      Physical.Occurrence base columns.toColumns
        (columns.toColumns_valid valid))
    (assignment : ColumnId → Nightstream.SuperNeo.Concrete.F)
    (constantWire : assignment (physical.map 0) = 1)
    (satisfied : Satisfies physical.rows assignment) :
    PiRLC.Equations algebra
        (attempt codec
          (numericAssignment physical.map assignment)
          columns.source.toBatchColumns) ∨
      physical.numeric.BadRoot
        (numericAssignment physical.map assignment) := by
  have numericSatisfied :
      Nightstream.Implementation.R1CS.Satisfies
        physical.numeric.rows
        (numericAssignment physical.map assignment) :=
    (Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.KPiRlcPhysicalOccurrence.satisfies_iff
      physical assignment).1 satisfied
  have result :=
    Nightstream.Implementation.R1CS.Canonical.KPiRlcSemanticBinding.equations_or_badRoot_of_rows
      algebra codec ring algebraRefinement
      (numericAssignment physical.map assignment)
      columns valid base
      (numeric_constant physical assignment constantWire)
      numericSatisfied
  exact result

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.KPiRlcPhysicalSemantic
