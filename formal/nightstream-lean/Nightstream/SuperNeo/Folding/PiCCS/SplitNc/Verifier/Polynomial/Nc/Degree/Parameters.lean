/-!
Static width parameters shared by the NC polynomial degree proof and its
exact-width SumCheck interface.

Owns: the degree-four ceiling and its five-coefficient physical width.

Does not own: the proof that the NC polynomial satisfies this ceiling,
SumCheck replay, transcript encoding, Rust, R1CS, rows, or costs.

Emits constraints: no.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_ccs.nc.degree.bound` | one NC round has degree at most four | fixed protocol parameter | `ncSumcheckDegreeBound` |
| `nifs.pi_ccs.nc.sumcheck.message.width` | degree four uses five constant-first slots | computed | `ncMessageWidth` |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.Degree

/-- The concrete strict-`b = 2` NC per-variable degree ceiling. -/
def ncSumcheckDegreeBound : Nat := 4

/-- A degree-four polynomial has five constant-first coefficients. -/
def ncMessageWidth : Nat := ncSumcheckDegreeBound + 1

end Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.Degree
