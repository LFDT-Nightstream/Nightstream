# Direct Parent Only Production Concrete FPrime Prior Terminal IO Opening

`DirectParentOnlyProductionConcreteFPrimePriorTerminalIOOpening` specifies the
opening-level terminal public-IO verifier boundary for the concrete prior `F'`
path.

Verifier acceptance consists only of verifier-visible facts:

```text
compactImageReplay(steps, proof, image)
construction2BoundaryReplay(steps, proof, image)
transcriptReplay(steps, proof, image)
statementPublicValid(canonicalStatement(steps, image))
proofBoundary(proof) = statementBoundary(canonicalStatement(steps, image))
exists publicIO,
  terminalVerifierPublicIO(terminalCommittedProof(proof)) = some publicIO
  terminalPublicValuesPrefix(terminalPublicValues(canonicalStatement), publicIO)
  terminalBoundaryValuesSuffix(statementBoundary(canonicalStatement), publicIO)
```

Authority is separate from verifier acceptance. The surface requires accepted
bound terminal public-IO evidence to open through the fixed authority opener,
and requires any opened authority for that evidence to carry the identical
`(steps, image)` pair.

From those obligations the module derives:

```text
PriorVerifierAuthorityOpening
CertifiedPriorVerifier
SoundVerifier
Reachable(Transition(ctx.toProductionContext), ctx.initial, steps, image)
AcceptedPriorPublicImageInvariants(ctx, steps, image)
```

It also derives rejection of unreachable prior images, same-proof
functionality, and the latest-step terminal end-to-end theorem.

The terminal public-IO prefix/suffix checks are binding checks, not digest
authority. The authority object remains the opened proof-carrying folded `F'`
reachability witness for the same public pair.
