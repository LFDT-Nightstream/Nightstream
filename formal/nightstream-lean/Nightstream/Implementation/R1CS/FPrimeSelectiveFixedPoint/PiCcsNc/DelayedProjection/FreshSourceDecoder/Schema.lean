/-!
Neutral schema for the fixed-active fresh public-`X` decoder artifact.

Assurance tier: model-level schema only.

Owns: the exact one-source, 270-coordinate public domain and a proof-free
record of each coordinate's normalized source-arm column and selective
lowering disposition.

Does not own: any coordinate value, a complete fresh witness `Z`, source-row
satisfaction, public-input binding, commitment binding, or row removal.

Emits constraints: none.
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.FreshSourceDecoder

/-- Schema revision emitted by the owning Rust artifact generator. -/
def schemaVersion : Nat := 1

/-- The stabilized steady-recursive arm. -/
def sourceArm : Nat := 2

/-- Fixed-active recursion carries exactly one fresh source. -/
def sourceCount : Nat := 1

/-- Five complete Phi81 public rings contain 270 field coordinates. -/
def logicalColumnCount : Nat := 270

abbrev LogicalColumn := Fin logicalColumnCount

/-- Exact selective-lowering disposition of one normalized source column.

These constructors describe provenance only. In particular,
`constantOne`, `linearDefinition`, and `traceEliminated` do not assert a
field value. -/
inductive Resolution where
  | constantOne
  | direct (start width : Nat) (centered : Bool)
  | decompositionAlias (source digit start : Nat) (centered : Bool)
  | equalityAlias (source start width : Nat) (centered : Bool)
  | linearDefinition
  | traceEliminated
deriving DecidableEq, Repr

namespace Resolution

/-- Every referenced final-assignment interval is nonempty and in range.
Disposition variants without a final interval are structurally valid but
still carry no value semantics. -/
def RangeValid (resolution : Resolution) (finalColumnCount : Nat) : Prop :=
  match resolution with
  | .constantOne => True
  | .direct start width _ => 0 < width /\ start + width <= finalColumnCount
  | .decompositionAlias _ _ start _ => start < finalColumnCount
  | .equalityAlias _ start width _ =>
      0 < width /\ start + width <= finalColumnCount
  | .linearDefinition => True
  | .traceEliminated => True

/-- The generated range certificate is executable over proof-free records. -/
instance (resolution : Resolution) (finalColumnCount : Nat) :
    Decidable (resolution.RangeValid finalColumnCount) := by
  cases resolution <;> simp only [RangeValid] <;> infer_instance

end Resolution

/-- Compact generated datum for one fresh public coordinate. -/
structure SourceColumnRecord where
  logicalColumn : Nat
  sourceArmColumn : Nat
  resolution : Resolution
deriving DecidableEq, Repr

namespace SourceColumnRecord

/-- Coordinate order and final-range validity, deliberately excluding any
claim about the source column's field value. -/
def WellFormed (record : SourceColumnRecord)
    (finalColumnCount : Nat) : Prop :=
  record.logicalColumn < logicalColumnCount /\
    record.resolution.RangeValid finalColumnCount

end SourceColumnRecord

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.FreshSourceDecoder
