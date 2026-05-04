# Direct Parent Only Production Concrete FPrime Prior Terminal IO

`DirectParentOnlyProductionConcreteFPrimePriorTerminalIO` is the
production-facing terminal committed public-IO binding layer for the concrete
compressed prior `F'` verifier.

Verifier acceptance is the public terminal-IO check:

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

The prefix check binds the verifier-returned public values to the expected
folded `F'` terminal public values. The suffix check binds those same public
values to the expected Construction-2 public boundary.

Folded `F'` authority is not part of verifier acceptance. It is derived from
the opening-level obligation that accepted terminal public-IO evidence opens
through the fixed authority opener and that any opened authority for that
evidence carries the identical `(steps, image)` pair.

The module exposes the resulting accepted-opens certificate, certified prior
verifier, same-proof functionality, public-image invariants, unreachable-prior
rejection, latest-step terminal end-to-end theorem, and non-aggregate private
DEC/stage projections.
