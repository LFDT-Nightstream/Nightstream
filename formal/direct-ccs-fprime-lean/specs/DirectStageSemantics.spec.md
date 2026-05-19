# Direct Stage Semantics

`DirectStageSemantics` specifies the theorem boundary that connects direct CCS
`F'` stage computations to the imported SuperNeo `Pi_CCS` and `Pi_RLC`
theorem surfaces.

A verified direct stage computation consists of:

```text
computePiCCS(i, prior_accumulator) -> pi_ccs_output
computePiRLC(i, pi_ccs_output) -> parent_CE_B_source
```

together with:

```text
Pi_CCS protocol target context
Pi_CCS theorem statement for that context
proof that the computed Pi_CCS output matches that context

Pi_RLC protocol target context
Pi_RLC theorem statement for that context
proof that the computed Pi_RLC parent source matches that context
```

The imported SuperNeo theorem surfaces must derive:

```text
Pi_CCS strong statement for the packaged Pi_CCS context
Pi_RLC weak statement for the packaged Pi_RLC context
```

A frontend may instantiate those theorem statements through the existing
SuperNeo CE-relation authority:

```text
ceRelation(ctx) => Pi_CCS strong statement for ctx
ceRelation(ctx) => Pi_RLC weak statement for ctx
```

A frontend may also use the context-carried reused-stage form. In that form,
the computed `Pi_CCS` output contains the SuperNeo context it is about, and
the `Pi_RLC` context is computed from the accepted `Pi_CCS` output and parent
source. The bridge predicates are then definitional equalities rather than
free proof obligations.

The reused-stage terminal theorem applies that CE-relation authority directly
to the strict terminal statement with public-image invariants. The prior
authority may be any predicate whose acceptance implies F' reachability. The
compressed-prior verifier and proof-carrying folded-authority statements are
specializations of that same boundary.

The context-carried reused-stage terminal theorem applies the same terminal
statements to the context-carried stage object. This is the preferred direct
frontend boundary: accepted stage data determines its SuperNeo context, rather
than relying on caller-selected output/source predicates.

The production compressed-prior entry point consumes a
`CompressedFPrimeAuthority.SoundVerifier`, so the prior compressed proof is
accepted only through a verifier whose accepted proofs open to folded `F'`
authority for the same public image.

The reduced-handle production entry point also consumes
`ParentCEBHashBinding.ParentCEBHash`, so the parent `CE(B)` hash binding
assumption is localized to the canonical parent-handle object rather than
passed as an unstructured digest-binding premise.

The accepted stage relations are:

```text
VerifiedPiCCS(i, prior, out)
  := out = computePiCCS(i, prior)
   + Pi_CCS output-soundness bridge
   + imported Pi_CCS strong statement

VerifiedPiRLC(i, out, source)
  := source = computePiRLC(i, out)
   + Pi_RLC source-soundness bridge
   + imported Pi_RLC weak statement
```

These accepted relations are functional because they include equality to the
deterministic computed output/source.

The main terminal theorem combines a verified stage package with deterministic
direct-program boundary semantics, concrete Ajtai-backed CE data, encoded
parent `CE(B)` digest binding, MSIS-to-Ajtai reductions, MSIS hardness, and a
sound prior folded `F'` authority predicate.

The prior authority predicate must satisfy:

```text
AuthorityAccepts(steps, authority, image)
=>
Reachable(F_prime_transition, initial, steps, image)
```

A proof-carrying folded authority is one valid instantiation of this predicate,
but a compressed proof verifier may also instantiate it by proving the same
soundness implication. The `SoundVerifier` entry point packages that implication
as a typed verifier object.

The compressed-prior verifier specialization uses:

```text
VerifyPrior(steps, proof, image)
=>
Reachable(F_prime_transition, initial, steps, image)
```

This implication is the only way a compressed proof or digest handle becomes
prior `F'` authority for the terminal theorem.

The theorem target is:

```text
accepted terminal compression
alternate latest transition from the same prior public image
=>
the final image is reachable
and the latest public image is unique
```

For a compressed-prior verifier from a well-formed zero-step base image, the
strict terminal theorem also derives the public Construction-2 image
invariants:

```text
final.step = priorSteps + 1
final.vkDigest = initial.vkDigest
final.initialBoundary = initial.initialBoundary
final.pc = 1
```

This module does not define the frontend encoding of `PiCCSOut` or the reduced
parent source. A frontend supplies the exact output/source soundness predicates
for its encoding.
