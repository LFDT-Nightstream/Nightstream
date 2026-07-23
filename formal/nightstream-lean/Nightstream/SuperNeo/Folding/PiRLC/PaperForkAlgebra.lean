/-!
Generic fork algebra for the paper `Pi_RLC` extractor (Appendix D.5).

Protocol: SuperNeo `Pi_RLC` weak reduction.
Phase: algebra of a complete coordinate-wise fork.
Constraint family: none; this file emits no rows.

Owns: an artifact-independent commutative-ring/module vocabulary, the
canonical head-first finite linear combination, isolation of the one
coordinate changed by a fork, and cancellation of an invertible scalar
action.

Does not own: transcript rewinding, fork probabilities, challenge-set
security, source-relation validity, commitment binding, concrete Phi81
associativity, Rust, R1CS, row removal, or constraint counts.

Authority boundary: all coefficients, values, and the changed coordinate are
explicit inputs.  No extractor result, source witness, uniqueness bridge, or
desired extraction statement is accepted as a premise.
-/

namespace Nightstream.SuperNeo.Folding.PiRLC.PaperForkAlgebra

universe uScalar uValue

/-- Operations of the commutative scalar ring used by Definition 15.  An
inverse is deliberately not a total operation: Appendix D.5 receives a unit
witness only for a forked challenge difference. -/
structure CommutativeRingOps (Scalar : Type uScalar) where
  zero : Scalar
  one : Scalar
  add : Scalar -> Scalar -> Scalar
  mul : Scalar -> Scalar -> Scalar
  neg : Scalar -> Scalar

namespace CommutativeRingOps

/-- Subtraction derived from the paper ring operations. -/
def sub {Scalar : Type uScalar} (ops : CommutativeRingOps Scalar)
    (left right : Scalar) : Scalar :=
  ops.add left (ops.neg right)

end CommutativeRingOps

/-- Explicit commutative-ring laws needed by the paper-owned algebra. -/
structure CommutativeRingLaws
    {Scalar : Type uScalar}
    (ops : CommutativeRingOps Scalar) : Prop where
  add_assoc : forall left middle right,
    ops.add (ops.add left middle) right =
      ops.add left (ops.add middle right)
  add_comm : forall left right, ops.add left right = ops.add right left
  zero_add : forall value, ops.add ops.zero value = value
  add_zero : forall value, ops.add value ops.zero = value
  add_neg : forall value, ops.add value (ops.neg value) = ops.zero
  mul_assoc : forall left middle right,
    ops.mul (ops.mul left middle) right =
      ops.mul left (ops.mul middle right)
  mul_comm : forall left right, ops.mul left right = ops.mul right left
  one_mul : forall value, ops.mul ops.one value = value
  mul_one : forall value, ops.mul value ops.one = value
  left_distrib : forall left middle right,
    ops.mul left (ops.add middle right) =
      ops.add (ops.mul left middle) (ops.mul left right)
  right_distrib : forall left middle right,
    ops.mul (ops.add left middle) right =
      ops.add (ops.mul left right) (ops.mul middle right)

/-- Operations of the additive module carrying the extracted assignment. -/
structure ModuleOps (Scalar : Type uScalar) (Value : Type uValue) where
  zero : Value
  add : Value -> Value -> Value
  neg : Value -> Value
  smul : Scalar -> Value -> Value

namespace ModuleOps

/-- Subtraction derived from the module's additive operations. -/
def sub {Scalar : Type uScalar} {Value : Type uValue}
    (ops : ModuleOps Scalar Value) (left right : Value) : Value :=
  ops.add left (ops.neg right)

end ModuleOps

/-- Explicit module laws over the same scalar and value operations used by
the finite combination. -/
structure ModuleLaws
    {Scalar : Type uScalar}
    {Value : Type uValue}
    (ring : CommutativeRingOps Scalar)
    (module : ModuleOps Scalar Value) : Prop where
  add_assoc : forall left middle right,
    module.add (module.add left middle) right =
      module.add left (module.add middle right)
  add_comm : forall left right,
    module.add left right = module.add right left
  zero_add : forall value, module.add module.zero value = value
  add_zero : forall value, module.add value module.zero = value
  add_neg : forall value,
    module.add value (module.neg value) = module.zero
  zero_smul : forall value, module.smul ring.zero value = module.zero
  add_smul : forall left right value,
    module.smul (ring.add left right) value =
      module.add (module.smul left value) (module.smul right value)
  one_smul : forall value, module.smul ring.one value = value
  mul_smul : forall left right value,
    module.smul (ring.mul left right) value =
      module.smul left (module.smul right value)
  smul_zero : forall scalar, module.smul scalar module.zero = module.zero
  smul_add : forall scalar left right,
    module.smul scalar (module.add left right) =
      module.add (module.smul scalar left) (module.smul scalar right)

/-- A concrete inverse for one scalar.  Challenge-set security supplies this
for the difference between the original and forked coordinate. -/
structure UnitWitness
    {Scalar : Type uScalar}
    (ring : CommutativeRingOps Scalar)
    (value : Scalar) where
  inverse : Scalar
  inverse_mul : ring.mul inverse value = ring.one
  mul_inverse : ring.mul value inverse = ring.one

/-- Canonical head-first finite linear combination used by a complete
coordinate fork.  It is independent of every executable artifact. -/
def linearCombination
    {Scalar : Type uScalar}
    {Value : Type uValue}
    (ring : CommutativeRingOps Scalar)
    (module : ModuleOps Scalar Value) :
    {count : Nat} ->
      (Fin count -> Scalar) -> (Fin count -> Value) -> Value
  | 0, _, _ => module.zero
  | _ + 1, coefficients, values =>
      module.add
        (module.smul (coefficients 0) (values 0))
        (linearCombination ring module
          (fun index => coefficients index.succ)
          (fun index => values index.succ))

/-- Two coefficient vectors agree away from the coordinate rewound by the
fork. -/
def AgreeExcept
    {Scalar : Type uScalar}
    {count : Nat}
    (coordinate : Fin count)
    (left right : Fin count -> Scalar) : Prop :=
  forall index, index ≠ coordinate -> left index = right index

private theorem neg_eq_of_add_eq_zero
    {Scalar : Type uScalar}
    {Value : Type uValue}
    (ring : CommutativeRingOps Scalar)
    (module : ModuleOps Scalar Value)
    (laws : ModuleLaws ring module)
    (left right : Value)
    (equal : module.add left right = module.zero) :
    module.neg left = right := by
  calc
    module.neg left = module.add (module.neg left) module.zero :=
      (laws.add_zero _).symm
    _ = module.add (module.neg left) (module.add left right) := by
      rw [equal]
    _ = module.add (module.add (module.neg left) left) right :=
      (laws.add_assoc _ _ _).symm
    _ = module.add module.zero right := by
      rw [laws.add_comm (module.neg left) left, laws.add_neg]
    _ = right := laws.zero_add right

private theorem module_neg_add
    {Scalar : Type uScalar}
    {Value : Type uValue}
    (ring : CommutativeRingOps Scalar)
    (module : ModuleOps Scalar Value)
    (laws : ModuleLaws ring module)
    (left right : Value) :
    module.neg (module.add left right) =
      module.add (module.neg left) (module.neg right) := by
  apply neg_eq_of_add_eq_zero ring module laws
  change module.add (module.add left right)
      (module.add (module.neg left) (module.neg right)) = module.zero
  letI : Std.Associative module.add := ⟨laws.add_assoc⟩
  letI : Std.Commutative module.add := ⟨laws.add_comm⟩
  calc
    module.add (module.add left right)
        (module.add (module.neg left) (module.neg right)) =
      module.add
        (module.add left (module.neg left))
        (module.add right (module.neg right)) := by ac_rfl
    _ = module.add module.zero module.zero := by
      rw [laws.add_neg left, laws.add_neg right]
    _ = module.zero := laws.zero_add module.zero

private theorem module_add_sub_add
    {Scalar : Type uScalar}
    {Value : Type uValue}
    (ring : CommutativeRingOps Scalar)
    (module : ModuleOps Scalar Value)
    (laws : ModuleLaws ring module)
    (left₁ left₂ right₁ right₂ : Value) :
    module.sub (module.add left₁ left₂) (module.add right₁ right₂) =
      module.add (module.sub left₁ right₁) (module.sub left₂ right₂) := by
  unfold ModuleOps.sub
  rw [module_neg_add ring module laws]
  letI : Std.Associative module.add := ⟨laws.add_assoc⟩
  letI : Std.Commutative module.add := ⟨laws.add_comm⟩
  ac_rfl

private theorem module_sub_add_same_right
    {Scalar : Type uScalar}
    {Value : Type uValue}
    (ring : CommutativeRingOps Scalar)
    (module : ModuleOps Scalar Value)
    (laws : ModuleLaws ring module)
    (left right suffix : Value) :
    module.sub (module.add left suffix) (module.add right suffix) =
      module.sub left right := by
  rw [module_add_sub_add ring module laws]
  unfold ModuleOps.sub
  rw [laws.add_neg, laws.add_zero]

private theorem module_sub_add_same_left
    {Scalar : Type uScalar}
    {Value : Type uValue}
    (ring : CommutativeRingOps Scalar)
    (module : ModuleOps Scalar Value)
    (laws : ModuleLaws ring module)
    (common left right : Value) :
    module.sub (module.add common left) (module.add common right) =
      module.sub left right := by
  rw [module_add_sub_add ring module laws]
  unfold ModuleOps.sub
  rw [laws.add_neg, laws.zero_add]

private theorem smul_neg
    {Scalar : Type uScalar}
    {Value : Type uValue}
    (ring : CommutativeRingOps Scalar)
    (module : ModuleOps Scalar Value)
    (ringLaws : CommutativeRingLaws ring)
    (moduleLaws : ModuleLaws ring module)
    (scalar : Scalar)
    (value : Value) :
    module.smul (ring.neg scalar) value =
      module.neg (module.smul scalar value) := by
  have inverse :
      module.add (module.smul scalar value)
          (module.smul (ring.neg scalar) value) = module.zero := by
    calc
      module.add (module.smul scalar value)
          (module.smul (ring.neg scalar) value) =
        module.smul (ring.add scalar (ring.neg scalar)) value :=
          (moduleLaws.add_smul _ _ _).symm
      _ = module.smul ring.zero value := by rw [ringLaws.add_neg]
      _ = module.zero := moduleLaws.zero_smul value
  exact (neg_eq_of_add_eq_zero ring module moduleLaws _ _ inverse).symm

private theorem smul_sub
    {Scalar : Type uScalar}
    {Value : Type uValue}
    (ring : CommutativeRingOps Scalar)
    (module : ModuleOps Scalar Value)
    (ringLaws : CommutativeRingLaws ring)
    (moduleLaws : ModuleLaws ring module)
    (left right : Scalar)
    (value : Value) :
    module.smul (ring.sub left right) value =
      module.sub (module.smul left value) (module.smul right value) := by
  unfold CommutativeRingOps.sub ModuleOps.sub
  rw [moduleLaws.add_smul, smul_neg ring module ringLaws moduleLaws]

/-- If two finite coefficient vectors differ at only one coordinate, the
difference of their canonical combinations is exactly the changed scalar
acting on that coordinate's value. -/
theorem coordinateIsolation
    {Scalar : Type uScalar}
    {Value : Type uValue}
    (ring : CommutativeRingOps Scalar)
    (module : ModuleOps Scalar Value)
    (ringLaws : CommutativeRingLaws ring)
    (moduleLaws : ModuleLaws ring module) :
    forall {count : Nat}
      (left right : Fin count -> Scalar)
      (values : Fin count -> Value)
      (coordinate : Fin count),
      AgreeExcept coordinate left right ->
      module.sub
          (linearCombination ring module left values)
          (linearCombination ring module right values) =
        module.smul (ring.sub (left coordinate) (right coordinate))
          (values coordinate)
  | 0, _, _, _, coordinate, _ => Fin.elim0 coordinate
  | count + 1, left, right, values, coordinate, agree => by
      revert agree
      refine Fin.cases ?_ (fun tail => ?_) coordinate
      · intro agree
        have tailsEqual :
            (fun index : Fin count => left index.succ) =
              (fun index : Fin count => right index.succ) := by
          funext index
          exact agree index.succ (Fin.succ_ne_zero index)
        simp only [linearCombination]
        rw [tailsEqual]
        rw [module_sub_add_same_right ring module moduleLaws]
        exact (smul_sub ring module ringLaws moduleLaws _ _ _).symm
      · intro agree
        have headEqual : left 0 = right 0 := by
          apply agree 0
          intro equal
          exact Fin.succ_ne_zero tail equal.symm
        have tailsAgree : AgreeExcept tail
            (fun index : Fin count => left index.succ)
            (fun index : Fin count => right index.succ) := by
          intro index different
          apply agree index.succ
          intro equal
          exact different (Fin.succ_inj.mp equal)
        simp only [linearCombination, headEqual]
        rw [module_sub_add_same_left ring module moduleLaws]
        exact coordinateIsolation ring module ringLaws moduleLaws
          (fun index => left index.succ)
          (fun index => right index.succ)
          (fun index => values index.succ)
          tail tailsAgree

/-- Applying the inverse of a unit scalar cancels its action.  This is the
algebraic final step in Appendix D.5's coordinate extractor. -/
theorem inverseActionCancellation
    {Scalar : Type uScalar}
    {Value : Type uValue}
    (ring : CommutativeRingOps Scalar)
    (module : ModuleOps Scalar Value)
    (moduleLaws : ModuleLaws ring module)
    (scalar : Scalar)
    (unit : UnitWitness ring scalar)
    (value : Value) :
    module.smul unit.inverse (module.smul scalar value) = value := by
  rw [<- moduleLaws.mul_smul]
  rw [unit.inverse_mul, moduleLaws.one_smul]

end Nightstream.SuperNeo.Folding.PiRLC.PaperForkAlgebra
