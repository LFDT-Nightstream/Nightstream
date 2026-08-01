import Nightstream.Implementation.Lowering.Goldilocks.Compiler
import Nightstream.Implementation.Lowering.Goldilocks.Optimization.Composition

/-!
Contract: expose canonical Goldilocks R1CS encodings through the generic
replacement specification.

Assurance tier: model-level.

Owns: the degree-two system adapter and its identity replacement.

Does not own: protocol observables, an optimization pass, native CCS, a
manifest, or Rust.

Emits constraints: none.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Lowering.Goldilocks.Optimization.R1CS

open Nightstream.Implementation.Lowering.Goldilocks

universe u

private abbrev Field := Nightstream.SuperNeo.Concrete.F
abbrev Assignment := ColumnId -> Field

/-- Canonical R1CS has total degree two. -/
def degree : Nat := 2

/-- Adapt one exact row list and constant-one coordinate to the replacement
interface. The caller owns the observation projection. -/
def system
    {Observable : Type u}
    (one : ColumnId)
    (rows : List OwnedRow)
    (observe : Assignment -> Observable) :
    Optimization.System Assignment Observable where
  Accepts := fun assignment =>
    assignment one = 1 /\ Goldilocks.Satisfies rows assignment
  observe := observe
  degree := degree

/-- The physical encoding adapter uses the exact receipt-derived rows. -/
def ofEncoding
    {Observable : Type u}
    {signature : Nightstream.Implementation.Lowering.Typed.Signature}
    {input output : Nightstream.Implementation.Lowering.Typed.Schema
      signature.types}
    {source :
      Nightstream.Implementation.Lowering.Typed.Program
        signature input output}
    (encoding : Encoding source)
    (observe : Assignment -> Observable) :
    Optimization.System Assignment Observable :=
  system encoding.one encoding.rows observe

theorem accepts_ofEncoding_iff
    {Observable : Type u}
    {signature : Nightstream.Implementation.Lowering.Typed.Signature}
    {input output : Nightstream.Implementation.Lowering.Typed.Schema
      signature.types}
    {source :
      Nightstream.Implementation.Lowering.Typed.Program
        signature input output}
    (encoding : Encoding source)
    (observe : Assignment -> Observable)
    (assignment : Assignment) :
    (ofEncoding encoding observe).Accepts assignment <->
      encoding.PhysicalSatisfies assignment :=
  Iff.rfl

/-- Identity replacement for a canonical R1CS encoding under any selected
degree limit of at least two. -/
def identity
    {Observable : Type u}
    {signature : Nightstream.Implementation.Lowering.Typed.Signature}
    {input output : Nightstream.Implementation.Lowering.Typed.Schema
      signature.types}
    {source :
      Nightstream.Implementation.Lowering.Typed.Program
        signature input output}
    (encoding : Encoding source)
    (observe : Assignment -> Observable)
    (degreeLimit : Nat)
    (withinLimit : degree <= degreeLimit) :
    Optimization.Replacement
      (ofEncoding encoding observe)
      (ofEncoding encoding observe)
      degreeLimit :=
  Optimization.Replacement.identity
    (ofEncoding encoding observe) degreeLimit withinLimit

end Nightstream.Implementation.Lowering.Goldilocks.Optimization.R1CS
