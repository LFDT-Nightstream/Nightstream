# Direct Parent Only Production Soundness

`DirectParentOnlyProductionSoundness` specifies the production-facing theorem
surface for the optimized direct CCS `F'` terminal path whose public accumulator
carries only the parent `CE(B)` source.

The context fixes the actual implementation obligations:

```text
canonical concrete CE data
fixed child CE relation and Ajtai parameters
contextual reused SuperNeo Pi_CCS/Pi_RLC stage semantics
Poseidon2 parent CE(B) hash binding object
deterministic boundary update
deterministic parent-statement commitment encoding
initial public image and public-image well-formedness
MSIS-to-Ajtai reduction
MSIS hardness
sound compressed prior F' verifier
or a concrete prior verifier whose accepted proofs open to proof-carrying
  folded F' authority for the same `(steps, image)`
```

The latest verifier and transition are derived from that single context. A
terminal proof is accepted only against those fixed objects, and any alternate
latest transition used for adversarial comparison is checked against the same
prior image and the same context-fixed transition.

The proof-carrying prior verifier is the baseline authority instance. It
accepts exactly a `FoldedFPrimeAuthority` object for the context's transition
and initial image. Every accepted proof by any `SoundPriorVerifier` reaches the
claimed prior image; therefore a verifier that accepts an unreachable prior
image cannot inhabit the sound prior-verifier boundary.

A concrete compressed verifier may also be supplied by giving its verifier
predicate together with the opening theorem:

```text
VerifyPrior(steps, proof, image)
=>
exists authority:
  FoldedFPrimeAuthority.Accepts(steps, authority, image)
```

That theorem constructs the production `SoundPriorVerifier` object and exposes
the same unreachable-prior rejection and computed-latest-image terminal
soundness conclusions. The strongest raw-verifier entry point also exposes
private-child uniqueness for the same parent source. The packaged endpoint
module combines that private-child uniqueness with prior-image reachability.

The theorem proves:

```text
final public image is reachable from the initial image
accepted prior proof reaches the claimed prior public image
accepted latest public image equals any alternate latest public image from the
  same prior image and context
latest parent CE(B) source is unique for the fixed prior image
the accepted and alternate latest steps share one pointwise-authorized child table
both latest parent sources equal the context's deterministic computePiRLC result
  over computePiCCS applied to that shared child table
both latest public computation boundaries equal the context's deterministic
  computeBoundary result for the fixed prior image
both latest public images equal the context's deterministic ComputedNextImage
  record built from the fixed prior image and shared child table
any other pointwise-valid private DEC child table for the same parent source
  equals the shared child table
final.step = priorSteps + 1
final.vkDigest = initial.vkDigest
final.initialBoundary = initial.initialBoundary
final image is well formed
```

The shared child table conclusion is intentionally pointwise. It relies on the
canonical private `Pi_DEC` obligations for the exact context-fixed child CE
relation and Ajtai parameters, binary digits, exact length, per-column
Goldilocks recomposition, child witness-table identity, and wire identity into
`Pi_CCS`. Under the context's MSIS-backed Ajtai binding and parent-hash binding,
two sets of such pointwise obligations for the same parent source authorize the
same child table. The theorem does not accept aggregate norm checks as a
substitute for those obligations.

Poseidon2 remains an external implementation primitive. The Lean boundary
requires only the parent-encoding binding assumption packaged by the
implementation hash object; digest chains are not treated as folded `F'`
authority.
