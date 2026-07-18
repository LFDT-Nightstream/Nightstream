import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.BooleanEvaluation

/-!
A concrete necessity witness for the nonlinear paper terminal in joint
`Pi_CCS`.

Protocol: SuperNeo `Pi_CCS` (Section 7.3 / Appendix D.4).
Phase: SumCheck terminal evaluation away from the Boolean cube.
Constraint family: semantic polynomial construction only; this file emits no
rows.

Owns: a one-variable finite-field counterexample showing that interpolating a
table of already-composed nonlinear residual leaves is not equivalent to
interpolating the underlying image and then applying the nonlinear protocol
formula.

Does not own: the full CCS or norm polynomial, transcript derivation,
probability, Rust, R1CS, constraint counts, or a claim that any particular
production row is removable.

Emits constraints: no.

Authority boundary: this is a kernel-checked countermodel, not an empirical
comparison with the existing circuit. It proves that Boolean-cube agreement
alone cannot justify using a residual-table MLE as the paper's off-cube
terminal polynomial.

| Protocol | Phase | Family | Mathematical obligation |
|---|---|---|---|
| `Pi_CCS` | source image | one Boolean variable | raw table is `0` on `false`, `1` on `true` |
| `Pi_CCS` | nonlinear residual | square | compare `MLE(x -> z(x)^2)` with `MLE(z)(r)^2` |
| `Pi_CCS` | off-cube terminal | challenge `r = 2` in `Fin 5` | residual-table path gives `2`, protocol order gives `4` |
| assurance | necessity | construction order | the two terminal constructions are unequal |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ProtocolPolynomial.Necessity.NonlinearTerminal

/-- A concrete five-element field carrier. Only its additive and
multiplicative operations are needed by this counterexample. -/
abbrev Field := Fin 5

/-- Canonical modular operations used by both interpolation paths. -/
def ops : InterpolationOps Field where
  zero := 0
  one := 1
  add := (· + ·)
  mul := (· * ·)
  neg := fun value => 0 - value

/-- The underlying one-variable image: `z(false) = 0`, `z(true) = 1`. -/
def sourceImage : BooleanTable Field 1 :=
  .branch (.leaf 0) (.leaf 1)

/-- The nonlinear residual computed only at Boolean leaves. Squaring `0` and
`1` produces the same leaf table, which makes the off-cube failure easy to
audit. -/
def squaredResidualLeaves : BooleanTable Field 1 :=
  .branch (.leaf (ops.mul 0 0)) (.leaf (ops.mul 1 1))

/-- A verifier challenge outside the Boolean cube. -/
def offCubePoint : CubePoint Field 1 where
  coordinates := [2]
  dimension := rfl

/-- Incorrect construction: compose the nonlinear residual on Boolean leaves,
then interpolate that residual table. -/
def residualTableTerminal : Field :=
  squaredResidualLeaves.evaluate ops offCubePoint

/-- Paper construction: interpolate the underlying image first, then apply the
nonlinear formula at the verifier point. -/
def protocolTerminal : Field :=
  let interpolated := sourceImage.evaluate ops offCubePoint
  ops.mul interpolated interpolated

/-- The residual-table MLE evaluates to `2` at the chosen off-cube point. -/
theorem residualTableTerminal_eq_two : residualTableTerminal = 2 := by
  decide

/-- The actual nonlinear protocol polynomial evaluates to `4` there. -/
theorem protocolTerminal_eq_four : protocolTerminal = 4 := by
  decide

/-- Inclusion-necessity witness: replacing the nonlinear paper terminal with
the MLE of already-composed residual leaves changes the verified polynomial,
even though both constructions agree at every Boolean vertex. -/
theorem residualTableTerminal_ne_protocolTerminal :
    residualTableTerminal ≠ protocolTerminal := by
  decide

end Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ProtocolPolynomial.Necessity.NonlinearTerminal
