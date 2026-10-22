
/-!
Finite verifier-visible SumCheck polynomials.

Owns: one coefficient-list message format, a degree upper bound derived from
that list, canonical-shape validation, and executable Horner evaluation.

Does not own: verifier challenges, claimed initial or terminal values,
semantic expected polynomials, root counting, transcript encoding, PiCCS, or
any implementation refinement.
-/

namespace NightstreamFPrime.Spec.SumCheck.Finite

universe uField

structure Ops (Field : Type uField) where
  zero : Field
  one : Field
  add : Field -> Field -> Field
  mul : Field -> Field -> Field

structure Message (Field : Type uField) where
  coefficients : List Field

namespace Message

def Canonical
    {Field : Type uField}
    (ops : Ops Field)
    (message : Message Field) : Prop :=
  message.coefficients ≠ [] ∧
    (message.coefficients.length = 1 \/
      message.coefficients.getLast? ≠ some ops.zero)

def degreeUpperBound
    {Field : Type uField}
    (message : Message Field) : Nat :=
  message.coefficients.length - 1

def evaluateCoefficients
    {Field : Type uField}
    (ops : Ops Field)
    (point : Field) : List Field -> Field
  | [] => ops.zero
  | coefficient :: rest =>
      ops.add coefficient
        (ops.mul point (evaluateCoefficients ops point rest))

def evaluate
    {Field : Type uField}
    (ops : Ops Field)
    (message : Message Field)
    (point : Field) : Field :=
  evaluateCoefficients ops point message.coefficients

def canonicalCheck
    {Field : Type uField}
    [DecidableEq Field]
    (ops : Ops Field)
    (message : Message Field) : Bool :=
  if message.coefficients = [] then false
  else if message.coefficients.length = 1 then true
  else decide (message.coefficients.getLast? ≠ some ops.zero)

@[simp] theorem canonicalCheck_eq_true_iff
    {Field : Type uField}
    [DecidableEq Field]
    (ops : Ops Field)
    (message : Message Field) :
    canonicalCheck ops message = true ↔ Canonical ops message := by
  unfold canonicalCheck Canonical
  by_cases empty : message.coefficients = []
  · simp [empty]
  · by_cases singleton : message.coefficients.length = 1
    · simp [empty, singleton]
    · simp [empty, singleton]

def zero
    {Field : Type uField}
    (ops : Ops Field) : Message Field where
  coefficients := [ops.zero]

theorem zero_canonical
    {Field : Type uField}
    (ops : Ops Field) :
    Canonical ops (zero ops) := by
  simp [Canonical, zero]

end Message

end NightstreamFPrime.Spec.SumCheck.Finite
