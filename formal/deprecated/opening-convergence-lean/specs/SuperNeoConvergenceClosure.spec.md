# SuperNeoConvergenceClosure Specification

## Goal

State the concrete end-to-end closure theorem for the current
`opening-convergence-lean` package instantiated at the real extension-field
claim boundary:

- field carrier: `SuperNeo.KExt`
- PCS verifier boundary: `OpeningConvergence.SuperNeoExtensionBridge.boundaryK registry`

This module closes the gap between:

1. the generic convergence theorems proved over an abstract
   `AjtaiPCSBoundary K`, and
2. the concrete extension-field SuperNeo boundary used by Nightstream's
   opening-convergence design.

## Theorem Target

For any explicit SuperNeo opened-object registry `registry`, if the concrete v1
convergence verifier accepts a pipeline over
`boundaryK registry`, then:

1. every original extension-field family evaluation claim in the pipeline is
   satisfied by the concrete split SuperNeo boundary, and
2. the verifier's failure probability is bounded by the frozen total v1 error
   bound obtained from the bucket parameters of that pipeline.

Formally, this module must expose a theorem of the shape:

```lean
accepted_v1_convergence_over_boundaryK
  -> all original claims satisfy boundaryK
  /\ failure_probability <= frozen_total_error_bound
```

where the claim-satisfaction predicate is the concrete one induced by
`boundaryK registry`, not an abstract placeholder.

## Boundary

This module does not re-prove the convergence algebra or the split-encoding
lemmas. It specializes the already-proved generic theorem package to the
concrete boundary from `SuperNeoExtensionBridge`.
