import Nightstream.SuperNeo.Folding.PiDEC.Necessity.DigitAuthorization
import Nightstream.SuperNeo.Folding.PiDEC.Necessity.AggregateAuthorization

/-!
Curated `Pi_DEC` semantic-necessity surface.

Owns: imports of model-level counterexamples grouped by concrete omitted
obligation.

Does not own: `Pi_DEC` semantics, implementation refinement, constraint rows,
or permission to remove rows.

| Protocol | Phase | Family | Child owner |
|---|---|---|---|
| `Pi_DEC` | child authorization | signed digit / arity / range | `Necessity.DigitAuthorization` |
| `Pi_DEC` | child authorization | pointwise values / norms | `Necessity.AggregateAuthorization` |
-/
