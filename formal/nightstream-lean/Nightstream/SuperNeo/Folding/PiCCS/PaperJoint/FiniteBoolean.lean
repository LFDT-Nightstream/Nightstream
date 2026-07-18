import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.BooleanDomain
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.NumericBooleanDomain
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.BooleanTable
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.BooleanEvaluation
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.BooleanHypercubeSum
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.BooleanReproduction
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.BooleanProduct
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.FiniteSumAlgebra
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

/-!
Finite Boolean-algebra foundation for the paper-joint `Pi_CCS` model.

Owns: only the finite domains, table/evaluation correspondence, equality
weights, reproduction, hypercube sums, and finite linear-algebra lemmas used
by later paper phases.

Does not own: CCS or norm predicates, joint-polynomial construction,
transcript execution, concrete Phi81 placement, or implementation refinement.

Emits constraints: no.

| Child family | Mathematical obligation |
|---|---|
| domains, tables, and products | finite Boolean and numeric indices enumerate the intended cube and structured prefix/suffix products |
| evaluation and reproduction | canonical multilinear evaluation reproduces every Boolean leaf |
| hypercube and finite sums | explicit finite sums agree with the table and linear operations |
| paper linear algebra | reusable additive/scalar laws required by residual construction |
-/
