# Direct Parent Only Terminal Soundness

`DirectParentOnlyTerminalSoundness` specifies the terminal direct CCS `F'`
theorem boundary whose Construction-2 public accumulator carries only the
parent `CE(B)` source.

The public accumulator handle is:

```text
parentSource
```

The post-DEC `CE(b)^k` child table is private proof advice. An accepted latest
step must nevertheless prove:

```text
AuthorizedNextPiCCSInputs(prior.parentSource, priorInputs)
ParentSourceFromPiStages(i, prior, priorInputs, next.parentSource)
```

where `AuthorizedNextPiCCSInputs` is the canonical private `Pi_DEC` relation
and `ParentSourceFromPiStages` places the same hidden child table into the
`Pi_CCS` input handle before applying `Pi_CCS -> Pi_RLC`.

The private authorization is intentionally pointwise, not aggregate. It exposes:

```text
child bundle CE relation equals the fixed stage CE relation
child bundle Ajtai parameters equal the fixed stage Ajtai parameters
binary digits for every column
exact DEC length 14 for every column
Goldilocks recomposition for every column
child witness-table identity
wire identity into the next Pi_CCS inputs
```

For two accepted latest transitions from the same prior image, the theorem
requires the same hidden child table to satisfy both parent-source computations.
Thus an alternate accepted transition cannot substitute different DEC children
or altered norms while preserving only an aggregate checksum or final parent
handle, nor can it validate those children against a different local CE
relation or Ajtai parameter set.

The terminal soundness theorem composes:

```text
sound prior folded F' authority
accepted latest parent-only direct F' transition
encoded parent CE(B) digest binding
deterministic parent-statement commitment encoding
MSIS-to-Ajtai reduction
MSIS hardness
functional Pi_CCS stage relation
functional Pi_RLC stage relation
```

and proves:

```text
final public image is reachable from the initial image
latest parent CE(B) source is unique for the fixed prior image
under a deterministic boundary update, the accepted latest public image is unique
the accepted and alternate latest steps share one pointwise-authorized child table
final.step = priorSteps + 1
final.vkDigest = initial.vkDigest
final.initialBoundary = initial.initialBoundary
final image is well formed
```

The compressed-prior theorem consumes `CompressedFPrimeAuthority.SoundVerifier`
instead of a loose digest-chain premise. Verifier acceptance is authority only
when it implies folded `F'` reachability under the same transition and initial
image.
