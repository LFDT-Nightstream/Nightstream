# Reduced Handle Terminal

`ReducedHandleTerminal` specifies the terminal theorem shape for the direct
CCS reduced parent-handle path.

The terminal proof consumes:

```text
ParentCEBHashBinding.ParentCEBHash
ConcreteCEData
ContextualReusedStageComputations
MSIS-to-Ajtai reduction assumption
MSIS hardness assumption
prior F' authority
sound prior F' authority predicate
accepted latest Construction-2/F' step
```

The parent-hash object supplies the exact binding premise for:

```text
Hash(encodeSomeParentCEB(parent_CE_B))
```

so the terminal theorem does not accept an unstructured digest-binding premise.

The primary theorem requires:

```text
AuthorityAccepts(steps, authority, image)
=>
Reachable(F', initial, steps, image)
```

The compressed-prior theorem is the specialization where the authority
predicate is verifier acceptance:

```text
VerifyPrior(steps, proof, image)
=>
Reachable(F', initial, steps, image)
```

The production-shaped compressed-prior theorem may discharge that verifier
soundness by opening each accepted compressed proof to proof-carrying folded
authority:

```text
VerifyPrior(steps, proof, image)
=>
exists authority.
  FoldedFPrimeAuthority.Accepts(steps, authority, image)
```

The same terminal theorem may consume a `CompressedFPrimeAuthority.SoundVerifier`
object that packages the verifier predicate with that opening theorem.

The proof-carrying theorem uses a prior authority that already contains the
reachability proof. This is the theorem-level reference path for the same
reduced-handle statement.

The negative theorem target is:

```text
AuthorityAccepts(steps, authority, image)
and not Reachable(F', initial, steps, image)
=>
PriorAuthoritySound(AuthorityAccepts) is false
```

Both terminal theorem forms prove:

```text
Reachable(F', initial, priorSteps + 1, nextImage)
nextImage = altNext
nextImage.step = priorSteps + 1
nextImage.vkDigest = initial.vkDigest
nextImage.initialBoundary = initial.initialBoundary
WellFormed(nextImage)
```

The theorem relies on `ContextualReusedStageComputations`, so accepted Pi_CCS
and Pi_RLC data determines the SuperNeo context used by the imported stage
theorem statements.
