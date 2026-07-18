import Nightstream.Implementation.R1CS.Correspondence.PiCcsNc.Semantics.RangePolynomial
import Nightstream.Implementation.R1CS.Correspondence.PiCcsNc.Semantics.MixedPolynomial

/-!
Owns: the mathematical meaning of Π_CCS's norm-check channel before any
production-row correspondence is claimed.

Does not own: transcript derivation, SumCheck compiler rows, terminal wiring,
or permission to remove local low-norm gates.

Emits constraints: no.

Authority boundary: semantic polynomials must be evaluated on the exact outer
fresh CCS assignment. Claimed SumCheck polynomials and terminal evaluations
are not assignment authority.

| Constraint family | Mathematical obligation | Lean owner | Refinement status |
|---|---|---|---|
| centered range | `(z+1)z(z-1)=0` iff embedded `z` has strict norm `< 2` | `RangePolynomial` | model-level complete; production-row bridge open |
| direct packing | place raw `assignment[column]` only at lane `column mod 54` | `MixedPolynomial` | model-level complete; production source bridge open |
| batch mixing | combine `R_2(zTilde_i)` with `gamma^(i+1)` | `MixedPolynomial` | model-level complete; bad-mixing reduction separate |
| NC SumCheck | claimed chain equals the true mixed range polynomial or exposes a bad challenge | future child | open |
| zero initial claim | authoritative strict norms imply the true Boolean-cube sum is zero | `MixedPolynomial` | one-way model-level theorem |
| terminal identity | final claimed value equals the true assignment evaluation | `PiCcsNc.Terminal` | requires explicit `YZcolBound`; verifier derivation open |
-/
