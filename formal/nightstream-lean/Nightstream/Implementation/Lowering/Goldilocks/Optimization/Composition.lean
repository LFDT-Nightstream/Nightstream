import Nightstream.Implementation.Lowering.Goldilocks.Optimization.Spec

/-!
Contract: identity and sequential composition for certified replacements.

Assurance tier: model-level.

Owns: the proof that replacement passes compose without weakening witness,
observable, or degree obligations.

Does not own: a concrete pass, pass search, cost selection, or a manifest.

Emits constraints: none.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Lowering.Goldilocks.Optimization

universe u v

namespace Replacement

/-- The no-op replacement. It is the first end-to-end integration target for
every concrete constraint format. -/
def identity
    {Assignment : Type u}
    {Observable : Type v}
    (system : System Assignment Observable)
    (degreeLimit : Nat)
    (withinLimit : system.degree <= degreeLimit) :
    Replacement system system degreeLimit where
  recover := fun assignment => assignment
  derive := fun assignment => assignment
  sound := fun _ accepted => accepted
  complete := fun _ accepted => accepted
  recover_observes := fun _ _ => rfl
  derive_observes := fun _ _ => rfl
  source_degree := withinLimit
  target_degree := withinLimit

/-- Sequential replacement composition.

If `middle` replaces `source` and `target` replaces `middle`, then `target`
replaces `source`. Recovery runs in reverse pass order. Derivation runs in
forward pass order. -/
def compose
    {Assignment : Type u}
    {Observable : Type v}
    {source middle target : System Assignment Observable}
    {degreeLimit : Nat}
    (first : Replacement source middle degreeLimit)
    (second : Replacement middle target degreeLimit) :
    Replacement source target degreeLimit where
  recover := fun assignment =>
    first.recover (second.recover assignment)
  derive := fun assignment =>
    second.derive (first.derive assignment)
  sound := by
    intro assignment accepted
    exact first.sound _ (second.sound assignment accepted)
  complete := by
    intro assignment accepted
    exact second.complete _ (first.complete assignment accepted)
  recover_observes := by
    intro assignment accepted
    calc
      source.observe
          (first.recover (second.recover assignment)) =
        middle.observe (second.recover assignment) :=
          first.recover_observes _
            (second.sound assignment accepted)
      _ = target.observe assignment :=
          second.recover_observes assignment accepted
  derive_observes := by
    intro assignment accepted
    calc
      target.observe
          (second.derive (first.derive assignment)) =
        middle.observe (first.derive assignment) :=
          second.derive_observes _
            (first.complete assignment accepted)
      _ = source.observe assignment :=
          first.derive_observes assignment accepted
  source_degree := first.source_degree
  target_degree := second.target_degree

end Replacement

end Nightstream.Implementation.Lowering.Goldilocks.Optimization
