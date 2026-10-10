# Random-oracle knowledge-soundness target

This registration records the one-fold knowledge criterion of the production
NIFS in the random-oracle model. It is adversary 3 of
`formal/nightstream-fprime/SECURITY_MODEL.md`, which also maps the proof to
its Lean modules and states what the criterion trusts.

The final declaration is
`NightstreamFPrime.Export.Stage1.RandomOracleSetup.production_knowledge_error_lt`.
The literal criterion is `LeanGraph.Targets.RomKnowledgeSoundness` in
`tests/EvidenceTargets.lean`; `romKnowledgeSoundness` proves it.

The required conclusion is

```
𝔼 chunks, Pr_H[the verifier accepts the linked claim]
  < 𝔼 chunks, (Pr[the extractor returns a source-valid witness] + hashCollisions)
    + 17 (Q + 17) ε_sample + (Q + 74) ε_test
    + msisAdvantage + 2^-190
```

for every application that fits the `2^28` profile, at its own relation, for
every adversary that is a function of the setup chunks, makes at most `Q`
Fiat–Shamir oracle queries, and outputs a claim and a prior preimage, and for
every context digest.

The verifier is the Lean NIFS verifier with its coins read from `H`. The
deployed system checks every fold only inside the step circuit, and no
random-oracle model covers a hash inside a circuit. So the criterion bounds no
deployed verifier check; for the history it only motivates the error of
Assumption 1 (adversary 3 of the security model).

The game:

- `chunks` are uniform 256-bit setup chunks, one for each key coefficient. The
  coefficient is the chunk reduced modulo the Goldilocks prime. This is
  premise P1, SHAKE128 as a random oracle (`Spec.AjtaiSetupV1.Programming`).
- `H` is a uniform random function on the bounded challenge call lists, the
  random-oracle model of the Fiat–Shamir reads.
- The adversary is deterministic. It is chosen before the chunks and may read
  all of them, so it cannot contain a kernel vector of the key. An adversary
  precomputed for the fixed deployed seed is outside the game.
- The extractor and the binding reduction rerun the adversary with a changed
  `H`, so the oracle is programmable.
- The claim is linked: the verifier also checks that the prior preimage
  hashes to the digest in the fresh public input and is well formed
  (`PriorLink`). Well formed includes that the running children are the
  canonical split of the parent that the state hash stores.

The terms:

- `hashCollisions`: the chance that a retry or a rerun of the adversary gives
  two different prior preimages with one state hash. Both preimages are the
  adversary's own outputs.
- `17 (Q + 17) ε_sample + (Q + 74) ε_test`: `RandomOracleKnowledge.statisticalError`.
- `msisAdvantage`: the success of an explicit MSIS solver. For a uniform
  matrix, it draws chunks uniformly among the matrix's preimages, runs the
  adversary for those chunks, and returns the output of the binding
  reduction `RandomOracleBinding.rerunKernel`. That output is a nonzero
  integer kernel vector of the matrix's key with every coordinate below `8TB`.
  The vector is a function of the two runs; the definition is
  `noncomputable`.
- `2^-190`: the error of drawing a uniform matrix instead of reduced chunks
  for every application that fits
  (`RandomOracleSetup.programmingError_lt_of_fits`).

Required premises: the query bound only. MSIS hardness, the collision
resistance of the state hash, and the running time of the solver are not
premises; a numerical bound on `msisAdvantage` or `hashCollisions` needs them.

Dependencies, with namespace prefix `NightstreamFPrime`:

| Declaration | Required use |
| --- | --- |
| `Export.Stage1.RandomOracleSetup.programmingError_lt_of_fits` | Bound the programming error below `2^-190` for every application that fits. |
| `Export.Stage1.RandomOracleSetup.knowledge_error_le_setup` | Average the linked bound over the chunks and replace the binding term. |
| `Export.Stage1.RandomOracleLink.knowledge_error_le_linked` | The linked one-fold bound for one key. |
| `Lifecycle.RandomOracleBinding.rerunKernel` | The binding reduction as a function of the two runs. |
| `Lifecycle.RandomOracleBinding.collisionChance_le_kernelChance` | Bound the binding term by the reduction's success. |
| `Spec.AjtaiSetupV1.Programming.expect_le_programmed` | Move from reduced chunks to a uniform matrix. |
| `Export.Stage1.RandomOracleSetup.contract` | State the criterion as a `Spec.KnowledgeContract`. |

Use `explain rom-knowledge-soundness` for remaining validation and review.
The registered gate runs static, build, axioms, the exact target check and
declaration export, in order.
