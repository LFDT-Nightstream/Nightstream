/-!
Contract: the implementation-independent correctness condition for a
constraint-system replacement.

Assurance tier: model-level.

Owns: functional witness recovery, functional optimized-witness derivation,
exact preservation of caller-selected observables, and one shared degree
limit.

Does not own: a concrete constraint format, optimization pass, transcript
policy, manifest, Rust, or a security reduction.

Emits constraints: none.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Lowering.Goldilocks.Optimization

universe u v

/-- One constraint system with a caller-selected observation boundary.

`observe` is deliberately abstract. A protocol integration must include every
authoritative input, output, ordered transcript action, and named event that a
replacement is required to preserve. -/
structure System (Assignment : Type u) (Observable : Type v) where
  Accepts : Assignment -> Prop
  observe : Assignment -> Observable
  degree : Nat

/-- A proof-carrying replacement of `source` by `target`.

The two witness maps are data, not existential claims. Consequently a later
manifest or compiler can use the same maps that the proof checks.

The degree condition uses one shared deployment limit. This permits a valid
replacement such as degree-two selected R1CS by degree-three native CCS when
the selected CCS relation already permits degree three. -/
structure Replacement
    {Assignment : Type u}
    {Observable : Type v}
    (source target : System Assignment Observable)
    (degreeLimit : Nat) where
  recover : Assignment -> Assignment
  derive : Assignment -> Assignment
  sound :
    forall assignment,
      target.Accepts assignment ->
        source.Accepts (recover assignment)
  complete :
    forall assignment,
      source.Accepts assignment ->
        target.Accepts (derive assignment)
  recover_observes :
    forall assignment,
      target.Accepts assignment ->
        source.observe (recover assignment) =
          target.observe assignment
  derive_observes :
    forall assignment,
      source.Accepts assignment ->
        target.observe (derive assignment) =
          source.observe assignment
  source_degree : source.degree <= degreeLimit
  target_degree : target.degree <= degreeLimit

namespace Replacement

/-- A replacement preserves acceptance in both directions after applying the
two explicit witness maps. -/
theorem accepts_iff_after_maps
    {Assignment : Type u}
    {Observable : Type v}
    {source target : System Assignment Observable}
    {degreeLimit : Nat}
    (replacement : Replacement source target degreeLimit)
    (sourceAssignment targetAssignment : Assignment)
    (targetFromSource :
      targetAssignment = replacement.derive sourceAssignment)
    (sourceFromTarget :
      sourceAssignment = replacement.recover targetAssignment) :
    source.Accepts sourceAssignment <-> target.Accepts targetAssignment := by
  subst targetAssignment
  constructor
  · exact replacement.complete sourceAssignment
  · intro targetHolds
    have recovered := replacement.sound _ targetHolds
    rw [← sourceFromTarget] at recovered
    exact recovered

end Replacement

end Nightstream.Implementation.Lowering.Goldilocks.Optimization
