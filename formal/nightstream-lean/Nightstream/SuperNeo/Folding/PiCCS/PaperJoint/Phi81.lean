import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Phi81ColumnLayout
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Phi81CarrierLayout
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Phi81CoefficientKernel
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Phi81MatrixSource
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Phi81Evaluation
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteCarrier

/-!
Concrete Phi81 carrier and coefficient refinement for paper `Pi_CCS`.

Owns: only logical-column and completed-carrier layouts, the closed-form
Phi81 coefficient kernel, matrix-source placement, evaluation, and concrete
carrier algebra.

Does not own: production Rust cache construction, concrete CCS relation
closure, Split-NC verifier acceptance, transcript execution, or R1CS rows.

Emits constraints: no.

Authority boundary: coefficient lanes are derived from authoritative matrices
and assignments through an explicit kernel. A caller-supplied packed view is
not accepted as authority.

| Refinement family | Mathematical obligation |
|---|---|
| column/carrier layout | logical coordinates, completed coordinates, lanes, and padding have explicit maps |
| coefficient kernel | Phi81 multiplication determines every packed coefficient |
| matrix source | concrete coefficient matrices derive from the authoritative field matrices |
| evaluation/carrier algebra | packed evaluation agrees with the independently defined carrier operations |
-/
