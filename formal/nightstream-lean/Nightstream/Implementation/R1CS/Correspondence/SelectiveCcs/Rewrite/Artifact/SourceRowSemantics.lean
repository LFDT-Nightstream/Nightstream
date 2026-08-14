import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.SelectiveGroupedProductRewriteSchema
import Nightstream.Implementation.Lowering.Goldilocks.NumericRowBridge
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteCarrier.Algebra

/-!
Contract: direct field semantics for source R1CS rows exported with a grouped
product rewrite.

Assurance tier: model-level artifact interpreter.

Owns: evaluation of raw source linear combinations and the active R1CS row
equation `A(z) * B(z) = C(z)`.

Does not own: a generated artifact, coefficient canonicity, source-column
bounds, selector authority, source reconstruction, or production coverage.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.SourceRowSemantics

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.Lowering.Goldilocks.NumericRowBridge
open Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.Wire

/-- Evaluate sparse source terms. The assignment uses natural indices so one
reconstruction can also describe source-only temporary columns. -/
def evalTerms (assignment : Nat → F) : List RawSourceTerm → F
  | [] => 0
  | term :: tail =>
      residue term.coefficient * assignment term.column +
        evalTerms assignment tail

/-- The raw constant is the coefficient of the source constant-one wire. -/
def evalLinearCombination (constantWire : F) (assignment : Nat → F)
    (value : RawSourceLinearCombination) : F :=
  residue value.constant * constantWire + evalTerms assignment value.terms

/-- Active source R1CS semantics. -/
def Holds (constantWire : F) (assignment : Nat → F)
    (row : RawSourceR1csRow) : Prop :=
  evalLinearCombination constantWire assignment row.a *
      evalLinearCombination constantWire assignment row.b =
    evalLinearCombination constantWire assignment row.c

end Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.SourceRowSemantics
