import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.CCSResidualTable

/-! Provenance: copied from `formal/nightstream-lean/Nightstream/SuperNeo/Folding/PiCCS/PaperJoint/ProtocolVerifier/Types.lean`
at commit `fb7a8a99aefbb8ebb5474681ecf80f1b95a1b7a2`; namespaces renamed, otherwise unchanged. -/

/-!
Verifier-visible carriers for canonical SuperNeo v1.1 `Pi_CCS` acceptance.

Protocol: SuperNeo v1.1 `Pi_CCS` (Section 7.3 / Appendix B.2).
Phase: public verifier input and raw post-SumCheck output message.
Constraint family: typed authority boundaries only; this file emits no rows.

Owns: the minimal data available to executable acceptance and the complete
typed output-evaluation payload supplied by the prover.

Does not own: hidden assignments, matrix-image tables, carried-image tables,
semantic truth, transcript derivation, degree bounds, concrete refinement,
Poseidon2, Rust/R1CS, row removal, or constraint counts.

Emits constraints: no.

Authority boundary: `VerifierInput` contains only the verifier-owned sparse
constraint polynomial, the public prior point, and the separate public Pad
and matrix coefficients. `OutputMessage` contains values but no point,
challenge, terminal, degree, or transcript state.

| Stage path | Carrier | Authority class | Excluded semantic data |
|---|---|---|---|
| `nifs.pi_ccs.verify.input` | `VerifierInput` | verifier-owned / public | assignments and image tables |
| `nifs.pi_ccs.verify.degree` | `VerifierInput.sumcheckDegreeBound` | computed from explicit monomial syntax | declared degree metadata |
| `nifs.pi_ccs.verify.output` | `OutputMessage` | checked prover payload | point, challenges, terminal, state |
-/

namespace NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ProtocolPolynomial

universe uField

/-- Exact input surface read by executable paper-joint acceptance. -/
structure VerifierInput (Field : Type uField) (shape : Shape) where
  constraintPolynomial :
    CCSResidualTable.ConstraintPolynomial Field shape.matrixCount
  priorPoint : CubePoint Field shape.cubeVariables
  claimedPadCoefficient : PadCoordinate shape -> Field
  claimedMatrixCoefficient : MatrixCoordinate shape -> Field

namespace VerifierInput

/-- Protocol-selected per-round degree ceiling for strict-`b = 2` paper
`Pi_CCS`. The CCS branch is computed from the explicit monomial list; the
equality-gated cubic norm branch contributes four. Declared degree metadata
cannot inflate verifier acceptance. A later theorem must still prove that the
actual nonlinear polynomial has this bound. -/
def sumcheckDegreeBound
    {Field : Type uField}
    {shape : Shape}
    (input : VerifierInput Field shape) : Nat :=
  Nat.max
    input.constraintPolynomial.canonicalEqualityGatedDegreeBound 4

/-- Corrected Appendix D.4 per-round degree ceiling
`max(D_f + 1, 2b, 2)`. The first entry is computed from the explicit sparse
syntax. For an empty zero polynomial it is zero; the final `2` still gives the
same paper bound. Declared degree metadata is not verifier authority. -/
def paperRoundDegreeCeiling
    {Field : Type uField}
    {shape : Shape}
    (input : VerifierInput Field shape)
    (b : Nat) : Nat :=
  Nat.max input.constraintPolynomial.canonicalEqualityGatedDegreeBound
    (Nat.max (2 * b) 2)

/-- For the frozen strict-`b = 2` specialization, the verifier ceiling is no
larger than the corrected Appendix D.4 ceiling. -/
theorem sumcheckDegreeBound_le_paperRoundDegreeCeiling_of_b_eq_two
    {Field : Type uField}
    {shape : Shape}
    (input : VerifierInput Field shape)
    {b : Nat}
    (b_eq : b = 2) :
    input.sumcheckDegreeBound <= input.paperRoundDegreeCeiling b := by
  subst b
  unfold sumcheckDegreeBound paperRoundDegreeCeiling
  exact (Nat.max_le).2 ⟨
    Nat.le_max_left _ _,
    Nat.le_trans (by decide : 4 <= Nat.max (2 * 2) 2)
      (Nat.le_max_right _ _)⟩

/-- Two verifier inputs whose sparse monomial syntax agrees have the same
degree ceiling even when their declared polynomial metadata differs. -/
theorem sumcheckDegreeBound_eq_of_terms_eq
    {Field : Type uField}
    {shape : Shape}
    (left right : VerifierInput Field shape)
    (terms : left.constraintPolynomial.terms =
      right.constraintPolynomial.terms) :
    left.sumcheckDegreeBound = right.sumcheckDegreeBound := by
  unfold sumcheckDegreeBound
  exact congrArg (fun degree => Nat.max degree 4)
    (CCSResidualTable.ConstraintPolynomial.canonicalEqualityGatedDegreeBound_eq_of_terms_eq
        left.constraintPolynomial right.constraintPolynomial terms)

/-- Equality of the four authoritative fields is equality of the complete
executable verifier input. -/
@[ext] theorem ext
    {Field : Type uField}
    {shape : Shape}
    (left right : VerifierInput Field shape)
    (constraintPolynomial :
      left.constraintPolynomial = right.constraintPolynomial)
    (priorPoint : left.priorPoint = right.priorPoint)
    (claimedPadCoefficient :
      left.claimedPadCoefficient = right.claimedPadCoefficient)
    (claimedMatrixCoefficient :
      left.claimedMatrixCoefficient = right.claimedMatrixCoefficient) :
    left = right := by
  cases left
  cases right
  simp_all

end VerifierInput

/-- The post-SumCheck values sent by the prover at the verifier-derived point.
There is no point, terminal, alpha, gamma, challenge, or state field. -/
structure OutputMessage (Field : Type uField) (shape : Shape) where
  freshMatrixImage : Fin shape.freshCount -> Fin shape.matrixCount -> Field
  sourceAssignment : Fin shape.sourceCount -> Field
  padImage : PadCoordinate shape -> Field
  matrixImage : MatrixCoordinate shape -> Field

namespace OutputMessage

/-- Pointwise equality of all four typed value families is equality of the
complete output message. -/
@[ext] theorem ext
    {Field : Type uField}
    {shape : Shape}
    (left right : OutputMessage Field shape)
    (freshMatrixImage : left.freshMatrixImage = right.freshMatrixImage)
    (sourceAssignment : left.sourceAssignment = right.sourceAssignment)
    (padImage : left.padImage = right.padImage)
    (matrixImage : left.matrixImage = right.matrixImage) :
    left = right := by
  cases left
  cases right
  simp_all

end OutputMessage

end NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ProtocolPolynomial
