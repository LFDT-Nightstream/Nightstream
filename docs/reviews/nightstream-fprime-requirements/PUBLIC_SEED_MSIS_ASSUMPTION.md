# Approved Nightstream public-seed MSIS assumption

The owner approved this assumption on 2026-09-08 with the six conditions below. It is a Nightstream-specific cryptographic premise. It is not a consequence of SuperNeo's MSIS assumption for uniformly sampled matrices and is not a proved Lean fact.

## Exact selected setup

The current compiled package uses a **fixed constant seed**. Its runtime `SetupSeed` distribution is the point mass at `Poseidon2HashChainV1Setup.productionSeed`; it does not draw a new seed for each proof or verification. The earlier comment recording an owner-approved operating-system CSPRNG output describes provenance. It does not provide an average-case security guarantee for this frozen matrix.

Seed bytes, in the exact stored order:

`fc404984d44c1b878d68a6a80092d7d7ab44d81ac17b45a8e7bd4c1f1e371702`

The matrix is `A := Expand(seed, params)` with the following fixed inputs.

| Input | Selected value |
|---|---|
| Expansion algorithm/version | `nightstream-ajtai-chacha20-wide256-v1` |
| Block function | RFC 8439 ChaCha20: ten double rounds and feed-forward |
| Indexed encoding | Counter = 32-bit coefficient lane; nonce = 32-bit row followed by 64-bit block index; little-endian words |
| Coefficient conversion | First 256 block-output bits, interpreted little-endian, reduced modulo `q` |
| Modulus `q` | 18446744069414584321 |
| Ring | `F[X]/(X^54 + X^27 + 1)`, degree 54 |
| Matrix shape | 22 rows × 4708530 ring columns |
| Integer witness length | 254260620 scalar coefficients |
| Profile | `b=2`, `k_rho=16`, `B=65536`, `T=216` |
| Required MSIS norm | Strict coefficient bound `8TB=113246208` |

The implementation sources are `Spec/AjtaiSetupV1.lean`, `Spec/AjtaiSetupV1/ChaCha20.lean`, `Spec/Profile.lean`, and the selected `Export/Stage1/Poseidon2HashChainV1Setup*.lean` modules. Their deterministic execution, framing, coefficient ranges and selected dimensions have checked Lean evidence.

## Hardness premise and seed independence

For this compiled package, assume that finding a nonzero integer vector of the stated length, with every absolute coefficient below 113246208 and `A*z=0 mod q`, is computationally hard for the **specific fixed matrix above**, even with the seed and complete matrix-generation procedure public. This is materially stronger than a setup-average claim for random seeds or the paper's uniform matrices. Any concrete attack budget must account for preprocessing against this fixed setup. No numerical success bound is introduced here.

A future package that samples its seed must state its concrete `SetupSeed` rule and analyze the exact distribution `seed ← SetupSeed; A := Expand(seed, params)`. That sampling must be independent of the prover, witness, and prover-controlled transcript data. An assumption about that sampled distribution must not be presented as a per-seed guarantee. Changing to that mode or changing the expansion/version/profile requires a separately identified package and corresponding assumption; this approval does not silently cover it.

## Verifier authority and substitution

The seed, expansion identifier, dimensions and byte framing are part of the 73-word setup authority. The verifier-context encoding also contains the fixed modulus limbs and the Nightstream profile, including `b`, `k_rho`, `B`, and ring degree; the norm bound follows from the proved selected profile. The NIFS key authority and final package identity bind that context and the canonical package.

The verifier must pin or allowlist the expected package identity from its own configuration. It must recompute authority from the selected setup; a prover-supplied identity or a self-consistent digest chain is not authority.

`PerApplicationSecurity.packageIdentity_identifies_package_authority_or_collision` identifies all raw authority words and the canonical package unless a named Poseidon2 collision occurs. `SetupSecurity.packageIdentity_identifies_selected_setup_or_collision` then identifies the exact selected setup, using the proved injectivity of its full authority encoding. Thus a different prover-selected seed or matrix cannot satisfy the expected identity without an explicit hash-collision event. This is a conditional binding reduction, not a proof of Poseidon2 collision resistance or of the separate production-loader gate.

## Security estimates and final claim

Do not inherit SuperNeo's concrete MSIS estimate for uniform matrices. `FOUNDATION_SECURITY_PARAMETERS.json` records only named generic lattice-cost model calculations. These are not a reduction from this fixed matrix to a uniform matrix, an indistinguishability proof, or a production security certificate.

The final Stage 1 security claim must say:

> Secure assuming public-seed MSIS hardness for the selected setup.

For the current package, that phrase includes hardness for the specific frozen matrix. The claim must also retain its other explicit hash, Fiat–Shamir, extraction, sampling and scope premises. It must not be shortened to “assuming MSIS” or presented as a completed Stage 1 security theorem while the corresponding integration obligations remain open.

The checked ordinary binding reduction yields a same-key integer kernel vector below `2B`; the relaxed binding reduction yields one below `8TB`. The selected-key wrappers use the exact seed, dimensions and norm above. These reductions connect the approved search-problem assumption to both binding properties without assuming injectivity or asserting that kernel vectors do not exist.

A future proved reduction or indistinguishability argument for the exact generated-matrix distribution may replace or discharge this local assumption. Until then it stays visible as an assumption, separate from proved mathematics.
