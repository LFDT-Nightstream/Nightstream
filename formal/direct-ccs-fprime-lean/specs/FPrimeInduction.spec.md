# F Prime Induction

`FPrimeInduction` specifies the minimal Construction-2 induction authority
boundary for direct CCS `F'`.

The mathematical objects are:

```text
Image                 compact public F' image
Transition(i,a,b)     one valid F' transition from image a to image b
Authority             folded prior-authority object
AuthorityAccepts      verifier predicate for prior authority
VerifyLatestStep      verifier predicate for the latest F' step
```

Reachability is the inductive statement:

```text
Reachable(0, initial)
Reachable(i, a) and Transition(i, a, b) => Reachable(i + 1, b)
```

Prior authority is sound only when:

```text
AuthorityAccepts(i, authority, image) => Reachable(i, image)
```

The latest-step verifier is sound only when:

```text
VerifyLatestStep(i, authority, prior_image, next_image, proof)
=>
Transition(i, prior_image, next_image)
```

The theorem target is:

```text
sound prior authority
+ sound latest-step verifier
+ terminal compression acceptance
=>
Reachable(i + 1, next_image)
```

This component forbids treating a self-consistent digest as induction
authority. A digest may identify an authority object, but the proof must supply
a soundness theorem connecting accepted authority to reachability.
