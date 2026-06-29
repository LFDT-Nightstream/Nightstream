# Compressed F Prime Authority

`CompressedFPrimeAuthority` specifies the theorem boundary for using a
compressed folded `F'` proof as prior authority in terminal direct CCS
compression.

The mathematical objects are:

```text
Image                     compact public F' image
Transition(i,a,b)         one valid F' transition
initial                   base public image
PriorProof                compressed proof or verifier artifact
VerifyPrior               verifier predicate for prior authority
```

The verifier-soundness requirement is:

```text
VerifyPrior(steps, proof, image)
=>
Reachable(Transition, initial, steps, image)
```

The induced authority predicate is exactly verifier acceptance:

```text
Accepts(steps, proof, image) := VerifyPrior(steps, proof, image)
```

The typed verifier object packages the verifier predicate with its required
opening theorem:

```text
SoundVerifier.verify(steps, proof, image)
=>
exists authority.
  FoldedFPrimeAuthority.Accepts(steps, authority, image)
```

The theorem target is:

```text
VerifierSound(VerifyPrior)
=>
PriorAuthoritySound(Accepts)
```

A verifier can also discharge `VerifierSound` by proving that every accepted
compressed proof opens to proof-carrying folded authority for the same
`(steps, image)`:

```text
VerifyPrior(steps, proof, image)
=>
exists authority.
  FoldedFPrimeAuthority.Accepts(steps, authority, image)
```

and, when combined with a sound latest-step verifier:

```text
VerifierSound(VerifyPrior)
+ LatestStepSound(VerifyLatestStep)
+ accepted terminal compression
=>
Reachable(Transition, initial, steps + 1, next_image)
```

Equivalently, a `SoundVerifier` object can be used directly as prior
authority:

```text
SoundVerifier
+ LatestStepSound(VerifyLatestStep)
+ accepted terminal compression
=>
Reachable(Transition, initial, steps + 1, next_image)
```

This component does not prove the cryptographic soundness of a concrete proof
system. It fixes the exact premise that such a proof system must satisfy. A
compressed proof, digest, or handle is not prior `F'` authority unless its
verifier acceptance implies reachability under the same transition and base
image.

The negative theorem target is:

```text
VerifyPrior accepts an unreachable image
=>
VerifierSound(VerifyPrior) is false
```

For replay-stable terminal compression, a stronger property is needed:

```text
ProofFunctional(VerifyPrior)
:= same proof accepted for two prior pairs
   => those prior pairs are equal
```

Lean also records the separation between these two requirements:

```text
exists SoundVerifier such that
  the same opaque proof is accepted for two different reachable prior pairs
```

Thus a production replay theorem cannot use `SoundVerifier` alone as the
same-proof anti-retargeting premise. It needs a fixed proof-to-authority
opening, as represented by the prior-opening modules.
