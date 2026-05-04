# Direct Parent Only Stage Semantics

`DirectParentOnlyStageSemantics` specifies the parent-only terminal direct CCS
`F'` theorem boundary after the SuperNeo stage computations are packaged as
verified stage semantics.

The public accumulator handle carries only:

```text
parentSource
```

The private post-DEC child table is authorized pointwise from the prior parent
source, then reused exactly as the child input table for the next `Pi_CCS`
stage. The accepted stage relations are not arbitrary predicates: they are the
verified `Pi_CCS` and `Pi_RLC` relations induced by
`DirectStageSemantics.VerifiedStageComputations`, or by the contextual reused
stage package after forgetting to that verified stage object.

An accepted parent-source relation for a verified stage computes exactly the
stage package's deterministic parent source:

```text
source = computePiRLC(i, computePiCCS(i, prior parent source, private children))
```

The theorem composes:

```text
contextual reused SuperNeo Pi_CCS/Pi_RLC stage semantics
sound compressed prior F' verifier
parent-only latest direct F' transition
canonical parent CE(B) hash binding
deterministic parent-statement commitment encoding
MSIS-to-Ajtai reduction
MSIS hardness
```

and proves:

```text
final public image is reachable from the initial image
latest parent CE(B) source is unique for the fixed prior image
the accepted and alternate latest steps share one pointwise-authorized child table
the shared table feeds the deterministic computePiCCS/computePiRLC parent source
final.step = priorSteps + 1
final.vkDigest = initial.vkDigest
final.initialBoundary = initial.initialBoundary
final image is well formed
```

The parent-hash theorem consumes the canonical
`ParentCEBHashBinding.ParentCEBHash` object directly, keeping the hash function
and its parent-encoding binding property packaged together. The
Poseidon2-facing theorem is the implementation wrapper around that same
boundary. Poseidon2 itself remains outside this module; the theorem requires
only the binding property for the canonical parent `CE(B)` encoding. The digest
is compression, not authority. Prior authority comes from
`CompressedFPrimeAuthority.SoundVerifier`, whose acceptance implies folded `F'`
reachability under the same transition and initial image.
