import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.QuadraticRefinement.Aggregation
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.QuadraticRefinement.TerminalTransport

/-!
Public aggregation and semantic transport for the focused compact `y_zcol`
quadratic refinement.

Owns: the public import boundary composing bounded evaluation/product
certificates with field evaluation and symbolic terminal transport.

Does not own: native certificate computation, source-program execution,
selected-row materialization, protocol authority, security events, or
permission to remove rows.

Emits constraints: no.

| Refinement leaf | Mathematical obligation | Authority class |
|---|---|---|
| evaluation aggregation | bounded certificate slices cover every evaluation pair | derived |
| product aggregation | the product certificate covers every product pair | derived |
| symbolic transport | compact steps imply independent quadratic terminals | derived |
-/
