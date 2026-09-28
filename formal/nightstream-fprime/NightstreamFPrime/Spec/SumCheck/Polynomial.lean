
/-! Provenance: copied from `formal/nightstream-lean/Nightstream/SuperNeo/SumCheck/Polynomial.lean`
at commit `fb7a8a99aefbb8ebb5474681ecf80f1b95a1b7a2`; namespace renamed, otherwise unchanged. -/
/-!
Finite verifier-visible SumCheck polynomials.

Owns: one coefficient-list message format, a degree upper bound derived from
that list, canonical-shape validation, and executable Horner evaluation.

Does not own: verifier challenges, claimed initial or terminal values,
semantic expected polynomials, root counting, transcript encoding, PiCCS, or
any implementation refinement.

Emits constraints: no.

Authority boundary: a prover supplies only coefficients. Degree is always
computed as `coefficients.length - 1`; it is never a prover-carried field.
Canonical validation rejects the empty list and redundant high zero
coefficients, except that the zero polynomial has the unique representation
`[zero]`.

| Mathematical object | Verifier-visible data | Computed value |
|---|---|---|
| round polynomial | constant-first coefficient list | `Message.evaluate` |
| degree upper bound | canonical coefficient-list length | `Message.degreeUpperBound` |
| canonical encoding shape | nonempty, no redundant high zero | `Message.Canonical` |
-/

namespace NightstreamFPrime.Spec.SumCheck.Finite

universe uField

/-- Computational operations needed by constant-first Horner evaluation.

This is deliberately one carrier: coefficients, evaluation points, and values
all live in `Field`. No caller-supplied polynomial evaluator or degree oracle
is accepted. Algebraic field laws are not needed for finite verifier replay;
they belong to later semantic and root-counting theorems. -/
structure Ops (Field : Type uField) where
  zero : Field
  one : Field
  add : Field -> Field -> Field
  mul : Field -> Field -> Field

/-- Raw verifier-visible polynomial message, in constant-first order. -/
structure Message (Field : Type uField) where
  coefficients : List Field

namespace Message

/-- A unique finite shape for degree accounting.

The singleton list represents every constant, including zero. Longer lists
must end in a nonzero coefficient. -/
def Canonical
    {Field : Type uField}
    (ops : Ops Field)
    (message : Message Field) : Prop :=
  message.coefficients ≠ [] ∧
    (message.coefficients.length = 1 \/
      message.coefficients.getLast? ≠ some ops.zero)

/-- Degree upper bound derived solely from the coefficient-list length. -/
def degreeUpperBound
    {Field : Type uField}
    (message : Message Field) : Nat :=
  message.coefficients.length - 1

/-- Constant-first Horner evaluation. The empty-list branch is total so raw
messages can be parsed before canonical validation; acceptance rejects it. -/
def evaluateCoefficients
    {Field : Type uField}
    (ops : Ops Field)
    (point : Field) : List Field -> Field
  | [] => ops.zero
  | coefficient :: rest =>
      ops.add coefficient
        (ops.mul point (evaluateCoefficients ops point rest))

/-- Evaluate one verifier-visible message at a field point. -/
def evaluate
    {Field : Type uField}
    (ops : Ops Field)
    (message : Message Field)
    (point : Field) : Field :=
  evaluateCoefficients ops point message.coefficients

/-- Executable canonical-shape check. -/
def canonicalCheck
    {Field : Type uField}
    [DecidableEq Field]
    (ops : Ops Field)
    (message : Message Field) : Bool :=
  if message.coefficients = [] then false
  else if message.coefficients.length = 1 then true
  else decide (message.coefficients.getLast? ≠ some ops.zero)

/-- The executable check is exact for the mathematical canonical-shape
predicate. -/
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

/-- Canonical verifier-visible zero polynomial. -/
def zero
    {Field : Type uField}
    (ops : Ops Field) : Message Field where
  coefficients := [ops.zero]

/-- The canonical zero message satisfies the finite-shape contract. -/
theorem zero_canonical
    {Field : Type uField}
    (ops : Ops Field) :
    Canonical ops (zero ops) := by
  simp [Canonical, zero]

end Message

end NightstreamFPrime.Spec.SumCheck.Finite
