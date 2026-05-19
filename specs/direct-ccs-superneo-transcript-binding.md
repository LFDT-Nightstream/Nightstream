# Direct CCS SuperNeo Transcript Binding Spec

This spec defines the Fiat-Shamir boundary for the direct CCS/R1CS SuperNeo
path. It is not a generic hash policy. It answers one concrete question:

```text
When direct CCS F' verifies a SuperNeo NIFS.V transition, exactly what data must
be bound before each verifier challenge is derived?
```

## Protocol Boundary

Fiat-Shamir is used to make the public-coin verifier messages in SuperNeo's
interactive reductions non-interactive. In the direct CCS IVC path, those
non-interactive checks are executed by `F'` because HyperNova Construction 2
defines `F'` to run `NIFS.V` on the prior running accumulator and the prior
committed instance.

There are three distinct layers:

| Layer | What It Does | Fiat-Shamir Role |
|---|---|---|
| SuperNeo chunk | Runs `Pi_CCS -> Pi_RLC -> Pi_DEC` on `CCS^K + CE(b)^k`. | Replaces verifier-sampled `Pi_CCS` and `Pi_RLC` challenges with Poseidon2-derived challenges. |
| Direct CCS `F'` | Executes the verifier side of the latest SuperNeo NIFS transition and updates the Construction-2 public image. | Recomputes the SuperNeo transcript for `NIFS.V`; separately hashes the compact Construction-2 image. |
| Final Spartan compression | Proves the latest direct CCS `F'` step and terminal private CE consistency. | Does not invent new SuperNeo challenges; it proves the `F'` transcript and terminal relation checks were satisfied. |

So yes, this spec is about `F'`, but only the part of `F'` that verifies
`NIFS.V`. The Construction-2 public image hash is a linkage hash. It is not a
replacement for the SuperNeo transcript that derives folding challenges.

## Security Goal

The non-interactive proof must have the same challenge dependencies as the
interactive protocol:

```text
the verifier challenge must be unpredictable until all public inputs and prover
messages that precede that challenge have been fixed.
```

That means the transcript must be verifier-driven. A prover must not be able to:

- choose or mutate incoming CE data after seeing `Pi_CCS` challenges;
- choose or mutate `Pi_CCS` output CE claims after seeing `Pi_RLC` challenges;
- mutate a carried digest and recompute downstream digests to make a fake chain
  self-consistent;
- use a compact handle as authority unless the omitted data is proven elsewhere
  before the challenge is derived.

Poseidon2 is the only approved hash for proof, transcript, and public-digest
paths unless a separate protocol change explicitly approves otherwise.

## What Fiat-Shamir Compiles

SuperNeo section 7 is written as interactive reductions. The verifier sends
random challenges at these points:

| Reduction | Interactive Verifier Message | Non-Interactive Replacement |
|---|---|---|
| `Pi_CCS` | Initial `alpha` and `gamma` for the CCS/evaluation/norm sumcheck. | Poseidon2 challenge derived after binding the chunk parameters, fresh CCS public instances, and incoming CE public instances. |
| `Pi_CCS` | Sumcheck round challenges. | Poseidon2 challenges derived round-by-round after each prior prover sumcheck message. |
| `Pi_RLC` | `rho_1, ..., rho_{K+k}` from the strong sampling set. | Poseidon2 challenges derived after binding the `Pi_CCS` output CE claims. |
| `Pi_DEC` | No verifier randomness. | No Fiat-Shamir challenge; only deterministic recomposition checks. |

Construction-2/F' also uses hashes, but those hashes have a different job:

| Construction-2 Hash | Purpose |
|---|---|
| `hash(vk_fs, i, z0, zi, U_i, pc_i)` | Bind the committed instance image for the current IVC state. |
| `hash(vk_fs, i+1, z0, z_{i+1}, U_{i+1}, pc_{i+1})` | Output the next compact public image. |

Those Construction-2 hashes bind the IVC state transition. They do not replace
the SuperNeo `Pi_CCS`/`Pi_RLC` transcript inside `NIFS.V`.

## Required SuperNeo Chunk Transcript

For a chunk with `K` fresh CCS claims and `k` incoming CE claims:

```text
input:
  CCS(b)^K + CE(b)^k

Pi_CCS:
  CCS(b)^K + CE(b)^k -> CE(b)^(K+k)

Pi_RLC:
  CE(b)^(K+k) -> CE(B)

Pi_DEC:
  CE(B) -> CE(b)^k
```

The Fiat-Shamir transcript must follow this order:

```text
T0 = domain(
  "direct-ccs-superneo-nifs-v",
  params_digest,
  structure_digest,
  chunk_index,
  K,
  k
)

T1 = absorb(T0, fresh_CCS_public_instances)
T2 = absorb(T1, incoming_CE_public_instances)

Pi_CCS alpha,gamma = challenge(T2)

for each Pi_CCS sumcheck round:
  T_round = absorb(T_round, prover_round_message)
  round_challenge = challenge(T_round)

T3 = absorb(T2, all_Pi_CCS_sumcheck_messages, Pi_CCS_terminal_messages)
T4 = absorb(T3, Pi_CCS_output_CE_claims)

Pi_RLC rho_1..rho_(K+k) = challenge(T4)

parent_CE = RLC(Pi_CCS_output_CE_claims, rho)

Pi_DEC children are deterministic outputs checked against parent_CE:
  parent.c == sum_i b^i * child_i.c
  parent.x == sum_i b^i * child_i.x
  parent.y == sum_i b^i * child_i.y
```

Two ordering rules are mandatory:

- `Pi_CCS` challenges bind the inputs to `Pi_CCS`.
- `Pi_RLC` challenges bind the outputs of `Pi_CCS`.

Using `Pi_CCS` output CE claims to derive `Pi_RLC` challenges is correct.
Using `Pi_CCS` outputs to derive the initial `Pi_CCS` challenges is circular and
forbidden.

## CE Public Instance Binding

A CE public instance is:

```text
(c, x, r, {y_j})
```

where:

- `c` is the Ajtai commitment;
- `x` is the public/input projection;
- `r` is the evaluation point;
- `{y_j}` are matrix-evaluation claims.

For generic carried accumulators, the `Pi_CCS` transcript must bind the full CE
public instance:

```text
H(c, x, r, y, metadata)
```

Binding only `c`, only a commitment projection, or only
`claim_count + accumulator_handle` is not a generic replacement for binding the
CE instance.

## Accumulator Binding Strategy

The expensive object is the carried SuperNeo accumulator:

```text
U_i = CE(b)^k
```

For the current Goldilocks profile, `k = 14`. Hashing all 14 CE public
instances inside `F'` is straightforward but expensive because each CE contains
an Ajtai commitment plus `x`, `r`, and `y` evaluation data.

The preferred reusable accumulator strategy is to bind the single `Pi_RLC`
parent claim:

```text
parent_i = CE(B)^1
```

and prove inside `F'` that the private `CE(b)^k` children used as the next
incoming accumulator are exactly the valid low-norm `Pi_DEC` decomposition of
that parent.

That means the transcript/public image may bind:

```text
H(parent_i)
```

instead of:

```text
H(child_0, ..., child_{k-1})
```

but only if `F'` also proves all of the following over the same child wires that
are passed into the next `Pi_CCS` verifier:

```text
same structure and evaluation point:
  child_i.s == parent.s
  child_i.r == parent.r

DEC recomposition:
  parent.c = sum_i b^i * child_i.c
  parent.x = sum_i b^i * child_i.x
  parent.y = sum_i b^i * child_i.y

child CE(b) membership:
  child_i.c = Commit(child_i.z)
  child_i.x = L_in(child_i.z)
  child_i.y_j = M_j child_i.z evaluated at child_i.r, for every j
  ||child_i.z||_infty < b

canonical decomposition authority:
  child_i.z is the canonical base-b digit decomposition of parent.z,
  or the circuit proves a theorem-strength equivalent uniqueness condition

wire identity:
  the child_i CE instances proven above are exactly the incoming CE instances
  consumed by the next Pi_CCS transcript/verifier
```

This is not a commitment-only shortcut. The bound `CE(B)` parent is authority
only because the circuit proves the full child CE membership and
`Pi_DEC(parent -> children)` relation that authorizes the private low-norm
children.

For `b = 2`, signed low-norm alone is not enough. The decompositions

```text
1 = 1 + 2*0
1 = -1 + 2*1
```

both use digits with absolute value `< 2`. Therefore the circuit must not treat
`DEC recomposition + ||child.z|| < 2` as a uniqueness proof. It must enforce
canonical `{0,1}` digit construction/bitness, or prove an equivalent production
constraint that rules out signed alternate decompositions.

If the prover can choose different private children after the parent has been
bound, then those children can change the next `Pi_CCS` challenge input. That is
unsound. Therefore the parent-binding strategy is valid only when the child
claims and child witnesses are constrained before challenge derivation by the
membership, canonical-decomposition authority, recomposition, and wire-identity
checks above.

### Sumcheck-Like Child Authorization

A Twist/Shout-style sumcheck can be useful here, but only as a proof of the
private DEC child relation. It is not sound to merely put a sumcheck output into
the Fiat-Shamir input and treat that as authorization.

The acceptable shape is:

```text
public before challenge:
  parent CE(B) handle/claim
  structure/params digest
  child table commitment or equivalent binding handle

private witness:
  child CE(b)^k instances
  child witnesses

proof checked by F':
  child table opens to the same child wires used by next Pi_CCS
  parent = Pi_DEC_recompose(children)
  each child satisfies CE(b) membership
  child witnesses are canonical base-b digits, or satisfy an equivalent
  uniqueness theorem
```

The Fiat-Shamir challenges for that proof must be derived after the public
parent and child-table binding are fixed, and the verified proof must constrain
the exact child wires consumed by the next `Pi_CCS` transcript. If the proof
only checks a digest, a sample, or a self-consistent recomputation that can be
mutated with the children, it is not authority.

Equivalently, for any reduced public challenge source `source`, the required
proof obligation is:

```text
same source + same parent + accepted proof authorization
  => same private child CE(b)^k accumulator
  => same next Pi_CCS input accumulator
```

Only after that implication is proved may `source` replace the full child
accumulator in the challenge input. This is the exact condition formalized in
the separate direct-CCS F' Lean project.

For the current `b = 2` profile, the concrete sufficient condition is stronger:
the proof checked by `F'` must imply base-2 recomposition and canonical child
membership for the same parent/source. Under that condition, the Lean model
proves that the same reduced source cannot feed different next `Pi_CCS`
accumulators.

A lower-level sufficient condition is also available for a future
sumcheck/table proof: for every coefficient column, prove fixed-length binary
digits and column-wise recomposition to the parent coefficient. The Lean model
proves those constraints make the hidden child table unique. This is closer to
what an arithmetized proof can actually check than an opaque "canonical
membership" predicate.

The implementation-facing version is Goldilocks modular recomposition:

```text
for every coefficient column j:
  child_digits[j][0..13] are bits
  recompose(child_digits[j]) mod q = parent_residue[j] mod q
  next Pi_CCS input column j is exactly child_digits[j]
```

The Lean model proves this version too. The proof relies on the exact length
`14` and the Goldilocks no-wrap fact below.

Two requirements are not optional:

- the reduced source must bind the parent residues functionally, meaning one
  source authorizes at most one parent-residue vector;
- the next `Pi_CCS` input must be the same child table checked by the proof.

The Lean model includes concrete counterexamples showing that if either
requirement is removed, the same deterministic challenge source can feed
different hidden child accumulators.

With functional source-to-parent binding restored, the Lean model proves the
implementation-facing statement: the same reduced source cannot authorize two
different next `Pi_CCS` child accumulators.

A minimal concrete source that satisfies this condition carries the parent
residue vector directly:

```text
source.parent_residues = flatten(parent CE(B) residues)
SourceBindsParent(source, parent) := source.parent_residues == parent
```

The Lean module `DirectCcsFPrime.ParentBoundSource` proves this binding is
functional. A Poseidon2 digest may compress this source, but the verifier must
recompute that digest from the authoritative parent residue data; a digest that
is merely supplied as advice is not authority.

The digest-only source variant has binding relation:

```text
SourceBindsParent(source_digest, parent) :=
  source_digest == Poseidon2(parent_residues(parent))
```

The Lean module `DirectCcsFPrime.DigestParentBinding` proves this relation is
functional only under an injectivity/collision-resistance assumption for the
parent hash, and also proves that a non-injective digest such as a constant hash
is not functional binding. This is the formal reason a digest can compress an
authoritative source but cannot become authority by itself.

The fixed-length part is not cosmetic. Binary digits without a fixed length are
not unique because leading zero rows can be added. Therefore the proof must bind
the exact SuperNeo decomposition length `k_dec` as well as bitness and
recomposition.

The recomposition equality must also be interpreted with care. If the circuit
checks recomposition as field equality modulo `q`, it must prove a no-wrap
condition. For the current Goldilocks profile, fixed-length binary columns of
length `k_dec=14` recompose below `2^14`, and `2^14 < q`; that kind of range
fact is what lets field equality stand for integer recomposition. Without a
no-wrap proof, modular equality can identify different decompositions.
The separate Lean module `DirectCcsFPrime.GoldilocksNoWrap` proves this
Goldilocks `k_dec=14` no-wrap fact for column tables.

| Approach | Hash Cost | Extra Algebra | Status |
|---|---:|---:|---|
| Hash full `CE(b)^k` | Very high | Low | Conservative but too expensive for recursive `F'`. |
| Hash `CE(B)^1` + prove `Pi_DEC` + canonical child `CE(b)^k` membership | Much lower | Adds DEC recomposition, child CE checks, and canonical digit/range checks | Preferred reusable strategy. |
| Hash commitment-only handle | Low | Missing `x`, `r`, `y` authority unless separately proved | Not sound generically. |
| Terminal private `CE(B)^1` | Low | Prove terminal `CE(B)` only | Good for final-only proofs, but not reusable for another fold. |

For intermediate recursive steps, use:

```text
Hash CE(B)^1 + prove Pi_DEC + canonical child CE(b)^k membership
```

For final terminal proofs that will not be continued, the prover may stop before
the final `Pi_DEC` and prove private terminal `CE(B)^1` directly. That terminal
pre-DEC mode is a separate optimization and does not provide a reusable
`CE(b)^k` accumulator for another SuperNeo fold.

## Direct CCS F' Requirements

The direct CCS `F'` relation must prove the verifier-side transition:

```text
input image:
  vk_fs_digest
  step/chunk counter i
  initial boundary digest z0
  current boundary digest zi
  prior running accumulator U_i
  prior committed instance u_i
  fixed pc = 1

advice:
  current direct CCS witness/boundary
  NIFS.V proof/advice pi

checks:
  base case uses the default accumulator/default committed instance;
  inductive case parses u_i and checks its Construction-2 image;
  NIFS.V verifies the SuperNeo transcript above;
  U_{i+1} is the NIFS.V output accumulator;
  z_{i+1} is the direct CCS step output boundary;
  x_{i+1} = hash(vk_fs, i+1, z0, z_{i+1}, U_{i+1}, pc=1)
```

The final Spartan proof may keep terminal semantic CE claims private, but then
it must prove their CE relation and wire them to the `NIFS.V` output. It must
not accept a final accumulator digest as authority by itself.

## Digest Authority Rule

A digest may appear in the transcript only if one of these is true:

- it is recomputed from the authoritative data inside the verified relation;
- it is opened/proved by a relation whose soundness is already part of the
  statement;
- it is explicitly non-authoritative diagnostic metadata and is not used to
  derive proof challenges.

Self-consistent digest chains are not authority. If an attacker can mutate data
and recompute all downstream digests, verification must still fail.

## Commitment Projection Handles

The SuperNeo paper proves strong/weak properties for the commitment projection
function `phi`, which projects commitments from CE instances. This makes
commitments security-relevant, but it does not by itself authorize omitting
`x`, `r`, or `y` from the Fiat-Shamir transcript.

A compact commitment-projection handle may be used only in these cases:

- the incoming accumulator is canonical and fixed by the verifier, such as the
  base/default accumulator;
- the handle is an implementation optimization inside a relation that separately
  proves all omitted CE fields were already fixed before the challenge;
- a dedicated proof establishes that the reduced transcript is equivalent for
  the target relation.

Otherwise, commitment-projection-only binding is not sound for generic carried
CE accumulators.

## Forbidden Shortcuts

These are soundness bugs unless accompanied by a separate proof and explicit
spec update:

- deriving `Pi_CCS` challenges from only `H(commitments)`;
- deriving `Pi_CCS` challenges from only `claim_count + accumulator_handle`;
- treating final CE projection digests as proof authority;
- proving only the latest chunk without a folded `F'` induction authority chain;
- accepting public image hashes that are not recomputed or opened against the
  authoritative instance data.

## Current Implementation Requirement

Until the reduced-handle path has a formal proof or equivalent circuit
authority, the generic direct CCS multi-step path must use full CE public
instance binding for `Pi_CCS`, or it must refuse to produce a standalone proof.
