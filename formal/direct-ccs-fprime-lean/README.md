# Direct CCS F' Lean

This is a small, independent Lean project for direct CCS `F'` protocol-boundary
checks. It is intentionally separate from `formal/superneo-lean`.

The first module, `DirectCcsFPrime.DecAuthorization`, proves the wiring theorem
needed for the planned accumulator-binding strategy:

```text
Hash CE(B)^1 + prove Pi_DEC + child CE(b)^k membership
```

The module does not prove Poseidon security, Ajtai binding, or concrete
base-`b` arithmetic. Those are represented as explicit predicates or obligations.
The theorem closes the local protocol mistake: the `CE(b)^k` children proven by
DEC and CE membership must be the same children consumed by the next `Pi_CCS`.

`DirectCcsFPrime.DecBase2Authorization` imports the existing
`formal/superneo-lean` decomposition surface and discharges the uniqueness
obligation for canonical base-2 split children. The remaining production-level
refinement is to derive canonical child equality from the full CE relation and
cryptographic binding assumptions rather than assuming canonical membership as
the child-membership predicate.

`DirectCcsFPrime.DecDigitUniqueness` records the important negative result:
signed low-norm base-2 digits are not unique. In particular,
`1 + 2*0 = -1 + 2*1`, and both digit pairs satisfy `|digit| < 2`. This means
`Pi_DEC` recomposition plus a signed low-norm check is not enough by itself to
authorize hidden children from a bound parent.

`DirectCcsFPrime.DecProofAuthorization` captures the boundary for a
sumcheck-like proof of the private child relation. Such a proof is acceptable
only when its verifier soundly implies DEC recomposition and child membership
over the same child wires that feed the next `Pi_CCS` input.

`DirectCcsFPrime.TranscriptAuthorization` proves the reduced-transcript wiring
claim: if a reduced public challenge source is accepted by a sound DEC proof
verifier and the child decomposition is unique, then that source cannot
authorize two different hidden child accumulators for the next `Pi_CCS`.

`DirectCcsFPrime.Base2TranscriptAuthorization` instantiates that claim for the
canonical base-2 split surface: if a local proof verifier implies base-2
recomposition and canonical child membership for the same parent/source, then
the same reduced source cannot feed different next `Pi_CCS` accumulators.

`DirectCcsFPrime.BinaryChildTableAuthorization` lowers the target one level:
if a proof verifier implies fixed-length binary digit columns that recompose to
the parent coefficient vector, the accepted hidden child table is unique. This
is the arithmetic core for the implementation-facing sumcheck/table proof.

`DirectCcsFPrime.AggregateChildTableNecessity` records why aggregate summaries
are not a valid substitute for that pointwise target. It proves a concrete
one-column, length-14 counterexample: `[1, 0, ..., 0]` and `[0, 1, ..., 0]`
are both binary fixed-length child columns with the same aggregate digit sum,
but they are different child tables and have different base-2 recompositions.
This is the formal guardrail against validating only a sum of child norms or
similar aggregate summaries. The module also proves the direct length-14 norm
case: two child-norm vectors can have the same total norm while assigning the
nonzero norm to different children.

`DirectCcsFPrime.GoldilocksNoWrap` instantiates the no-wrap side condition for
the concrete SuperNeo Goldilocks profile: binary columns of exact length
`k_dec = 14` recompose below the field modulus, so Goldilocks modular equality
lifts back to integer equality.

`DirectCcsFPrime.GoldilocksChildTableAuthorization` applies that no-wrap fact to
the implementation-facing proof boundary where recomposition is checked modulo
Goldilocks. It proves that same source and same parent residues cannot authorize
different next `Pi_CCS` child tables if the verifier proves bitness, exact
length, modular recomposition, and wire identity. It also proves the stronger
same-source result when the source-to-parent binding is functional.

`DirectCcsFPrime.ReducedSourceNecessity` records negative counterexamples:
the reduced source must bind parent residues, and the next `Pi_CCS` input must
be wire-identical to the proof-checked children. Without either condition, the
same challenge source can feed different hidden accumulators.

`DirectCcsFPrime.ParentBoundSource` gives the concrete positive source shape:
the source carries the parent residues and therefore binds them functionally.
This is the minimal implementation target before replacing raw residues with a
hash-compressed public representation.

`DirectCcsFPrime.DigestParentBinding` models that hash-compressed public
representation abstractly. It proves digest parent binding is functional if the
parent hash is injective, and proves a constant digest is not functional. This
keeps digest use in the right category: compression under an explicit binding
assumption, never authority by itself.

`DirectCcsFPrime.ParentEncoding` defines the canonical encoded parent sources
used by the reduced-handle strategy. It proves injectivity for parent residue
vectors and for shape-tagged flattened parent `CE(B)` handles. Concrete hash
security is an external cryptographic assumption; Lean separates that assumption
from the canonical encoding proof and does not implement the hash permutation.

`DirectCcsFPrime.ParentCEBHashBinding` packages that external assumption at
the exact parent-handle boundary. A `ParentCEBHash` supplies an abstract
field-list hash together with the theorem-facing binding premise over
`encodeSomeParentCEB`. From that premise, Lean proves equal parent-handle
digests recover the same parent `CE(B)` and every deterministic DEC residue
projection is functionally bound.

`DirectCcsFPrime.Poseidon2ParentCEBHash` is the implementation-facing version
of that boundary. It does not implement Poseidon2; it packages the verifier's
Poseidon2 parent-handle hash function together with the binding assumption over
canonical `encodeSomeParentCEB` preimages and converts it into
`ParentCEBHashBinding.ParentCEBHash`. The parent-only terminal theorem has a
specialization that consumes this object directly, so callers do not pass a
loose digest-binding premise.

`DirectCcsFPrime.ReducedHandleTerminal` composes that `ParentCEBHash` object
with the contextual terminal theorem. The top-level reduced-handle terminal
statement consumes the structured parent-hash boundary directly,
rather than accepting an unstructured digest-binding predicate as a loose
premise. Its primary theorem is stated at the exact induction boundary:
accepted prior authority must imply F' reachability under the same transition;
compressed-verifier and proof-carrying variants are specializations. The module
also proves the corresponding negative statement: an authority predicate that
accepts an unreachable prior image cannot satisfy that induction boundary.

`DirectCcsFPrime.ParentOpeningAuthorization` closes the next theory boundary:
parent residues used by private `Pi_DEC` must come from an accepted opening of
the same digest-bound parent `CE(B)` handle. The preferred theorem shape fixes
one CE relation and reduces the remaining cryptographic obligation to
commitment-map residue binding:

```text
ce.commitMap(assignmentA) = ce.commitMap(assignmentB)
=>
residues(assignmentA) = residues(assignmentB)
```

This is where the concrete Ajtai binding theorem must be applied. Digest
binding alone does not prove it, and a theorem over arbitrary unrelated CE
relations is not the target protocol statement.

The module also provides the deterministic statement-encoding relation needed
by the direct terminal theorem:

```text
StatementEncodes(stmt, parent_CE_B)
  := stmt.commitment = commitmentOfParent(parent_CE_B)
```

Lean proves this relation satisfies statement-commitment consistency by
construction. That removes the serializer-consistency premise for the intended
implementation shape; the remaining parent-opening obligations are digest
binding and Ajtai-backed opening binding.

`DirectCcsFPrime.AjtaiResidueBinding` connects that local commitment-map
obligation to the concrete Ajtai opening surface from `formal/superneo-lean`.
It proves that a CE `commitMap` is residue-functional when it is backed by
accepted bounded Ajtai openings and concrete Ajtai binding has no collision:

```text
NoAjtaiBindingCollision(params)
+ opensTo(params, commitMap(assignment), toOpening(assignment))
+ witness-residue adapter correctness
=>
CommitMapResiduesFunctional(commitMap)
```

This is still a cryptographic boundary: the module does not prove MSIS
hardness or the Ajtai advantage bound. It proves the exact deterministic bridge
needed once that binding assumption is supplied.

The module also proves that the theorem-facing `AjtaiBindingAssumption` from
`formal/superneo-lean` rules out concrete binding collisions in this Prop-level
model. This removes `NoAjtaiBindingCollision` as an independent terminal
premise; the remaining boundary is the standard Ajtai binding assumption
itself.

The module now also reuses the SuperNeo MSIS reduction package directly:

```text
MSISToAjtaiReductions(params)
+ MSISHardnessAssumption(params)
=>
NoAjtaiBindingCollision(params)
```

So the strongest terminal theorem can be stated against the same MSIS hardness
and MSIS-to-Ajtai reduction surface used by `formal/superneo-lean`, rather than
requiring a separate direct-project Ajtai premise.

The implementation-facing theorem is narrower than global `commitMap` binding:
it only requires a `CEOpeningAdapter` for witnesses that satisfy the fixed
`CE.Holds` relation. This matches the terminal proof obligation more closely:
private `Pi_DEC` may use parent residues only from CE openings that the proof
actually accepts.

The module also proves that an assignment-level adapter for the fixed
`ce.commitMap` induces that local `CEOpeningAdapter`. The bridge uses the
commitment equality already contained in `CE.Holds`:

```text
stmt.commitment = ce.commitMap(wit.assignment)
```

So the most concrete current terminal theorem does not need an independent
`CE.Holds -> opensTo` adapter premise; it needs the actual assignment-opening
adapter for the CE commitment map, plus the Ajtai binding assumption.

The module also provides the canonical adapter target for that CE commitment
map. An `AjtaiBackedCommitMap` states that `ce.commitMap` is exactly the
canonical Ajtai commitment induced by a fixed public matrix:

```text
commitment.payload = M || Mz
```

Lean proves that this canonical payload opens to the chosen Ajtai opening, and
therefore derives the assignment-opening adapter from
`AjtaiBackedCommitMap`. This removes the free adapter premise from the strongest
terminal theorem; the remaining implementation obligation is to instantiate
`AjtaiBackedCommitMap` for the actual SuperNeo CE commitment map.

The module now also defines the concrete canonical commitment map:

```text
ajtaiCommitMap(params, M, toOpening)(assignment) = M || Mz
```

and proves that it satisfies `AjtaiBackedCommitMap` from only the necessary
well-formedness, norm, bound, and residue-projection obligations. This is the
bridge needed when the concrete SuperNeo CE relation is built with the actual
Ajtai commitment map rather than an arbitrary abstract commitment function.

`DirectCcsFPrime.SuperNeoBridge` connects the reduced-handle theorem to the
existing `formal/superneo-lean` CE/Ajtai/`Pi_CCS`/`Pi_RLC`/`Pi_DEC` surfaces.
Its implementation-facing theorem says: encoded parent digest binding, a
deterministic statement commitment encoder, concrete Ajtai no-collision plus a
local `CE.Holds -> opensTo` adapter, Goldilocks child-table proof soundness, and
wire identity imply that the same source cannot authorize different next
`Pi_CCS` child inputs. The bridge still exposes lower-level variants that take
commitment-map residue binding directly, but the local Ajtai-backed
CE-opening theorem with `StatementEncodesByCommitment` is the direct terminal
target.

`DirectCcsFPrime.PrivatePiDecSoundness` narrows the remaining `Pi_DEC` verifier
boundary to the actual SuperNeo child bundle. The verifier is no longer modeled
as accepting a standalone child table; it accepts the child bundle whose fields
include child `CE.Holds`, child Ajtai openings, and the equality between the
checked digit table and the CE witness-derived digits. The minimal verifier
soundness obligation is now:

```text
Verify(source, parent_residues, child_bundle, proof)
=>
binary digits
+ fixed length 14
+ Goldilocks modular recomposition to parent_residues
```

Together with bundle wire identity, this proves the same reduced source cannot
feed different next `Pi_CCS` child inputs.

`DirectCcsFPrime.CanonicalPrivatePiDecVerifier` instantiates that boundary with
the minimal terminal relation itself. Its verifier accepts the actual child
bundle exactly when the bundle's digit table is binary, has fixed length 14, and
recomposes modulo Goldilocks to the opened parent residues. This removes the
abstract private-`Pi_DEC` verifier-soundness hypothesis for the terminal
relation case; the remaining proof obligations are the parent-opening binding
and the larger `F'` induction theorem.

It also packages the implementation-facing existential relation
`AuthorizedNextPiCCSInputs(source, next_inputs)`: there exists an opened parent,
an accepted child CE bundle, and a canonical private `Pi_DEC` proof authorizing
`next_inputs`. The accepted bundle must use the exact fixed CE relation and
Ajtai parameters from the stage context; it cannot satisfy the private DEC
check against an attacker-chosen local relation. Lean proves this relation is
functional for a fixed source under the encoded-parent digest binding,
deterministic parent-statement commitment encoding, Ajtai no-collision, and
local CE-opening adapter assumptions. This is the precise claim needed for the
reduced `CE(B)^1` handle to feed the next `Pi_CCS` without allowing hidden
child substitution.

`DirectCcsFPrime.ReducedAccumulatorStep` lifts that local authorization theorem
into the direct Construction-2 accumulator update. The reduced accumulator
handle carries a compact parent source and the authorized next `Pi_CCS` input
table. If the parent-source derivation is functional, and private `Pi_DEC`
authorization is functional, Lean proves two accepted latest `F'` transitions
from the same prior image cannot produce different authorized next accumulator
children. The remaining parent-source obligation is exactly the concrete
`Pi_CCS -> Pi_RLC` parent `CE(B)` source computation.

`DirectCcsFPrime.ParentOnlyAccumulatorStep` states the stricter optimized
public-handle shape: the accumulator handle carries only the parent `CE(B)`
source, while the post-DEC `CE(b)^k` children are private advice. An accepted
step must first authorize those private children from the prior parent source
and then use that exact table in the `Pi_CCS -> Pi_RLC` parent-source
computation. The canonical theorem exposes the non-aggregate obligations:
binary digits, exact DEC length 14, per-column Goldilocks recomposition,
the exact context-fixed child CE relation and Ajtai parameters,
child witness-table identity, and wire identity into the `Pi_CCS` inputs.
These obligations are packaged as named certificates
`PointwisePrivateDecCertificate` and `FixedCEChildMembershipCertificate`, so the
audit surface can be read field-by-field instead of as an anonymous existential
tuple.
It also proves the anti-substitution form needed for the parent-only handle:
two accepted steps from the same prior parent source share one common
authorized hidden child table.

`DirectCcsFPrime.DirectParentOnlyTerminalSoundness` lifts that parent-only
handle into the terminal Construction-2 theorem path. The public image carries
only the parent source, but the terminal theorem still extracts the private
pointwise DEC obligations and proves that any alternate accepted latest step
from the same prior image must use the same hidden child table in
`Pi_CCS -> Pi_RLC`. Its strongest implementation-shaped theorem consumes
MSIS-to-Ajtai reductions, MSIS hardness, deterministic parent-statement
encoding, stage functionality, and either arbitrary sound prior authority or a
`CompressedFPrimeAuthority.SoundVerifier`.

`DirectCcsFPrime.DirectParentOnlyStageSemantics` closes the next boundary in
that same parent-only path: instead of asking callers for bare functional
`Pi_CCS` and `Pi_RLC` relations, it consumes the existing verified
`DirectStageSemantics` stage package. Its production-shaped theorem uses a
contextual reused SuperNeo stage object,
`ParentCEBHashBinding.ParentCEBHash`, and a
`CompressedFPrimeAuthority.SoundVerifier`; the conclusion still extracts one
shared pointwise-authorized hidden child table for the accepted and alternate
latest steps. The Poseidon2-facing theorem is only the implementation wrapper
around that same parent-hash object. It also exposes that an accepted
parent-source relation computes exactly
`computePiRLC(i, computePiCCS(i, prior, children))` for that shared table. This
rules out aggregate-only validation surfaces for mutated DEC children while
keeping the terminal public state to the parent `CE(B)` source.

`DirectCcsFPrime.DirectParentOnlyProductionSoundness` packages the optimized
terminal verifier context into one theorem-facing object. The context fixes the
Poseidon2 parent hash boundary, concrete Ajtai-backed CE data, contextual
reused SuperNeo stages, deterministic boundary update, deterministic parent
commitment encoding, initial public image, MSIS assumptions, and the sound
compressed prior verifier. The terminal theorem then derives the exact
optimized conclusion from accepted terminal compression and any alternate
latest transition against the same context: reachability, full latest public
image equality, public-image invariants, parent-source uniqueness, and one
shared pointwise-authorized private DEC child table. The exact-compute theorem
further shows that both latest parent sources equal the context's deterministic
`computePiRLC` result over `computePiCCS` applied to that shared table, both
latest public boundaries equal the context's deterministic `computeBoundary`
result for the fixed prior image, and both latest public images equal the
canonical `ComputedNextImage` record built from those deterministic values. The
shared table is authorized under the same context-fixed CE relation and Ajtai
parameters, not merely under a table whose aggregate checks match. The
production theorem also proves that any other pointwise-valid private DEC table
for the same parent source equals that shared table. This is the compact audit
surface for the no-raw `CE(b)^14` public hash optimization.

The same module also instantiates the proof-carrying prior-authority baseline
for that production context. A `FoldedFPrimeAuthority` over the context's exact
transition induces a `SoundPriorVerifier`, and Lean proves that any accepted
sound prior proof reaches its claimed prior image. This makes the compressed
verifier's job explicit: accepted compressed proofs must open to that
proof-carrying authority shape, not merely to a self-consistent digest chain.
The module also provides the raw-verifier adapter: a concrete verifier
predicate plus the theorem that every accepted proof opens to proof-carrying
folded `F'` authority constructs the production `SoundPriorVerifier` and gets
the strongest local terminal conclusion: computed latest public images and
unique pointwise private children.

`DirectCcsFPrime.DirectParentOnlyProductionEndpoint` packages the final
production endpoint over that context. It combines accepted prior-image
reachability with the unique pointwise private-child conclusion, both for an
already packaged `SoundPriorVerifier` and for a raw compressed prior verifier
once the verifier-opening theorem is supplied. This is the theorem surface for
callers that need one conclusion saying the compressed prior proof is real
folded `F'` authority and the hidden post-DEC children cannot be swapped.
The same endpoint also exposes the audit-facing conclusion directly:
prior-image reachability, terminal soundness, and the pointwise child audit
trail covering accepted private `Pi_DEC`, fixed CE/Ajtai parameters, binary
fixed-length columns, per-column recomposition, witness-table identity, wire
identity into `Pi_CCS`, and uniqueness for the same parent source.
Its flattened `AuditedPublicEndpoint` conclusion additionally exposes final
Construction-2 reachability, `step`, `vkDigest`, `initialBoundary`,
well-formedness, deterministic boundary update, and equality with any
alternate latest transition checked from the same prior image.

`DirectCcsFPrime.DirectParentOnlyProductionChildMembership` keeps that same
production theorem surface but also exposes fixed-CE child membership for the
unique private table. The extracted children satisfy the context-fixed CE
relation, use the context-fixed Ajtai parameters, open under those parameters,
and wire the next `Pi_CCS` inputs to the CE witness-derived child table. This
is the explicit `ChildCEMembership(children)` conclusion for the parent-only
public handle path. The module also exposes a pointwise child audit trail:
accepted private `Pi_DEC`, binary child digits, fixed child-column length 14,
per-column Goldilocks recomposition to the opened parent residues, witness-table
identity, and next-`Pi_CCS` wire identity. The module exposes that conclusion
for both the proof-carrying folded-prior reference path and the raw
compressed-prior path once the verifier-opening theorem is supplied. It also
provides named projections for the exact private DEC acceptance,
fixed-CE membership, non-aggregate DEC table facts, and the equality between
next-`Pi_CCS` input wires and the CE witness-derived child digit table. The
production audit itself is packaged as `PointwiseChildAuditCertificate`, which
contains the lower-level private DEC certificate plus fixed child membership.

`DirectCcsFPrime.DirectParentOnlyProductionPriorOpening` packages the concrete
compressed-prior authority shape without modeling the compression scheme. An
opaque prior proof is accepted only through an opener that returns
proof-carrying folded `F'` authority for the exact `(steps, image)` consumed by
terminal compression. The same opaque proof cannot be reused to authorize a
different terminal prior step count or prior public image under the same opener
or opening certificate. The module also constructs the canonical
`CompressedFPrimeAuthority.SoundVerifier` object for both the opener-induced
predicate and a concrete verifier with an opening certificate, with acceptance
definitionally equal to the intended verifier predicate. That verifier composes
directly with the strongest parent-only theorem: reachable latest image,
computed public images, unique pointwise private children, and fixed-CE child
membership. The terminal replay guard also reaches the final public image:
after one opaque prior proof fixes the prior `(steps, image)` pair, any second
accepted latest step is forced to the same computed `nextImage` under the same
production context. The module now also exposes the generic
`ProofFunctional` fact for both opener-based and concrete verifier opening
certificates, making explicit that the same-proof replay guarantee comes from
the fixed opener, not from `SoundVerifier` alone. The same prior-opening
boundary also exposes the pointwise
child audit trail, so opener-based callers see the accepted private `Pi_DEC`,
fixed-length binary child columns, per-column recomposition, witness-table
identity, and next-`Pi_CCS` wire identity directly. Its strongest endpoint also
returns prior-image reachability and terminal-image reachability together, so a
compressed proof cannot be treated as authority through a digest-only path. The
module also exposes the flattened `AuditedPublicEndpoint` for both opener-based
and concrete-verifier opening certificates, including public-image invariants
and deterministic boundary update in the same conclusion as the child audit
trail. It also exposes direct opener projections: accepted verification rules
out an absent opened authority, and if the opener returns `some authority`, that
exact authority accepts the same `(steps, image)`.

`DirectCcsFPrime.DirectParentOnlyProductionStageAudit` extends that endpoint
with the contextual `Pi_CCS -> Pi_RLC` stage audit. The shared private child
table is placed into the child-carrying prior handle, the contextual `Pi_CCS`
output is proved to carry the current step, the parent source is the
deterministic `Pi_RLC` output over that exact `Pi_CCS` output, and the imported
SuperNeo `Pi_CCS` strong, `Pi_RLC` weak, and `Pi_DEC` knowledge theorem
statements are exposed for the computed parent-source context. This is the
audit surface for ruling out cross-round
or aggregate-only child substitution while still keeping raw post-DEC children
out of the public terminal hash. The prior-opening module exposes the same
stage-audited endpoint for both opener-based and concrete-verifier opening
certificates. The stage-audit module also exposes a flattened computed-stage
evidence projection: one pointwise-valid private child table, both terminal
parent sources equal to `computePiRLC(priorSteps, computePiCCS(...))` for that
table, uniqueness against every other pointwise-valid table for the same parent
source, and the imported `Pi_CCS`, `Pi_RLC`, and `Pi_DEC` statements for the
exact computed contexts. The DEC knowledge statement is attached to both the
computed `Pi_CCS` context that consumes the private children and the computed
`Pi_RLC` context that produces the compact parent handle.

`DirectCcsFPrime.DirectParentOnlyProductionSuperNeoReuse` packages the same
production context with the stronger Section 7.1-backed stage input. Its stage
field is `DirectStageSuperNeoReuse.Section71ContextualStageComputations`, so
the computed `Pi_CCS` and `Pi_RLC` contexts must be the exact targets of
upstream SuperNeo `ProtocolSection71Context` objects. The module converts that
package into the existing production context and re-exposes the stage-audited
endpoint theorem. It also exposes the terminal Section 7.1 owner-target audit:
the shared pointwise-valid private child table feeds `computePiCCS`, both
terminal parent sources equal `computePiRLC` over that exact output, the
computed `Pi_CCS` and `Pi_RLC` contexts are the targets of the wrapper's
Section 7.1 owner objects, and those owner targets carry `ceRelation` plus the
imported `Pi_CCS`, `Pi_RLC`, and `Pi_DEC` statements. This is the intended
implementation-facing constructor for the direct parent-only proof path when
paired with the concrete direct CCS application transition and the compressed
prior-verifier ABI contract.

`DirectCcsFPrime.DirectParentOnlyProductionSuperNeoReusePriorOpening` composes
that Section 7.1-backed production context with the concrete compressed-prior
opening boundary. An opaque prior proof is accepted only if it opens to
proof-carrying folded `F'` authority for the induced transition and the same
`(steps, image)` pair. The endpoint theorem then exposes the same flattened
public-image facts, pointwise child audit trail, and contextual stage audit.
It also gives the replay guard for a concrete verifier: the same prior proof
cannot authorize a different terminal public image under the same
Section 7.1-backed context. The Section 7.1 wrapper also exposes the direct
opener projections, so accepted compressed verification cannot hide a missing
opened authority or substitute a different authority for the same proof. It
also exposes prior-image reachability and the corresponding
cannot-accept-unreachable-prior theorem at this same Section 7.1-backed layer.
It also re-exposes `ProofFunctional` for the Section 7.1-backed opener and
concrete-verifier opening paths.

`DirectCcsFPrime.DirectParentOnlyProductionSuperNeoReuseReplayEndpoint`
packages that replay guard as endpoint theorems for both the authority-opener
path and the concrete compressed verifier. Given two terminal acceptances under
the same Section 7.1-backed context and the same opaque prior proof, Lean
proves the prior step count, prior public image, and terminal public image are
the same. The concrete-verifier theorem additionally requires the same
`VerifyPrior` predicate and prior-authority opening certificate. The conclusion
also includes the full stage-audited endpoint, so the private child table is
still authorized pointwise and fed into the contextual `Pi_CCS -> Pi_RLC`
computation. The pointwise child replay binding projection exposes that table
directly: both replayed terminal images are computed from the same private
children, and every other pointwise-valid table for the same parent source is
equal to that table. The direct no-swap theorem is phrased against
`PointwisePrivateDecRequirements` for the same parent source, so it requires the
full private `Pi_DEC` authorization, fixed CE/Ajtai membership, binary fixed
length columns, recomposition, witness-table identity, and next-`Pi_CCS` wire
identity before concluding equality with the audited table. The replay audit
package also returns the prior-opening facts, same-proof replay endpoint, and
computed-stage replay evidence together, so the concrete verifier path has one
theorem tying the accepted compressed proof to the exact `Pi_CCS -> Pi_RLC`
parent source and pointwise private child table. The computed-stage evidence
also has named projections for the exact computed `Pi_CCS` context that
consumes the private child table and the exact computed `Pi_RLC` context that
produces the compact parent source.

`DirectCcsFPrime.DirectParentOnlyProductionSuperNeoReuseCertifiedVerifier`
turns that concrete-verifier path into the intended implementation-facing
object: a verifier predicate packaged together with its fixed
`PriorVerifierAuthorityOpening` certificate. The object induces the strict
`CompressedFPrimeAuthority.SoundVerifier`, proves same-proof functionality from
the fixed opener, exposes the exact opened folded authority for every accepted
prior proof, proves that accepted proofs cannot open to `none`, exposes prior
reachability, and rules out accepting an unreachable prior public image. It
also proves that if the fixed opener returns a concrete authority for an
accepted proof, that exact authority accepts the same `(steps, image)` pair. It
exposes the single-acceptance audited endpoint directly:
prior and next folded `F'` reachability, terminal public-image equality, public
invariants, pointwise child audit, and contextual stage audit. It also exposes
the single-terminal audit package, flattened computed-stage endpoint evidence,
single-terminal pointwise child-table uniqueness, replay audit package,
computed-stage replay evidence, and the pointwise no-swap theorem: any
alternate child table must satisfy the full private `Pi_DEC` requirements for
the relevant parent source before Lean concludes equality with the audited table
used by the terminal `Pi_CCS -> Pi_RLC` computations. This keeps the final
endpoint from consuming a bare `SoundVerifier` where replay-stability is
needed.
The implementation is split by responsibility. The `Core` module owns the
verifier object and folded `F'` authority-opening boundary. The `Terminal`
module owns the one-terminal audit package. The `Replay` module owns
same-proof replay and explicit no-swap projections. The unsuffixed module is a
compatibility facade that imports those pieces.

The same module exposes the raw-verifier certification adapter. A compressed
proof implementation supplies its `verify` predicate, fixed opener, and the
accepted-opens theorem:

```text
verify(steps, proof, image)
=>
openAuthority(proof) = some authority
and authority accepts folded F' reachability for the same (steps, image)
```

Lean then builds the certified verifier object and exposes the same terminal
audit package, single-terminal pointwise child-table uniqueness, same-proof
replay audit package, and same-proof replay pointwise no-swap theorem from
those raw obligations. The raw replay package also has an explicit no-swap
variant whose conclusion names the audited child table and states the universal
pointwise uniqueness condition directly, so implementers do not need to infer
that guarantee from a nested computed-stage evidence predicate.

`DirectCcsFPrime.ParentSourceStep` decomposes that parent-source obligation into
the two SuperNeo stages that produce the reduced parent handle:

```text
Pi_CCS: prior accumulator + fresh CCS -> CE(b)^(K+k)
Pi_RLC: CE(b)^(K+k) -> one parent CE(B) source
```

Lean proves that if the accepted `Pi_CCS` stage is functional for a fixed prior
state, and the accepted `Pi_RLC` stage is functional for fixed `Pi_CCS`
outputs, then the parent-source relation required by
`ReducedAccumulatorStep` is functional. It also composes that result with the
Ajtai-backed canonical private `Pi_DEC` theorem, so the remaining obligation is
now the concrete functionality of the accepted `Pi_CCS` transcript output and
`Pi_RLC` parent computation, not an opaque reduced-handle assumption.

`DirectCcsFPrime.FPrimeInduction` states the Construction-2 induction authority
boundary directly. The terminal proof may check only the latest `F'` step, but
that latest step must consume a prior-authority object whose acceptance implies
actual reachability from the base image. Lean proves:

```text
sound prior authority
+ sound latest-step verifier
+ accepted terminal compression
=>
final image is reachable from the base image
```

It also proves the required negative form: an acceptance predicate that accepts
an unreachable image cannot be sound induction authority. This keeps digests in
the compression role only; a digest may name an authority object, but it cannot
replace a reachability/soundness theorem.

`DirectCcsFPrime.FoldedFPrimeAuthority` instantiates that prior-authority
boundary with the minimal proof-carrying object:

```text
steps
image
reachable : Reachable(F' transition, initial, steps, image)
```

Acceptance for a requested `(steps, image)` is just field equality, and Lean
proves this authority is sound because it carries the reachability proof. The
module also provides base and one-step extension constructors, proves that
each constructor authorizes exactly its own `(steps, image)` pair, and gives a
direct Construction-2 terminal theorem with the `PriorAuthoritySound` premise
already discharged. This is the formal line we need for the Rust/protocol
design: terminal compression can consume a compact handle to prior folded
authority, but the handle is not evidence unless it opens to an accepted
authority whose acceptance implies reachability.

`DirectCcsFPrime.CompressedFPrimeAuthority` states the equivalent theorem
boundary for a terminal proof that consumes a compressed prior proof instead of
a Lean proof-carrying authority object. The verifier predicate is acceptable
only under this premise:

```text
VerifyPrior(steps, proof, image)
  => Reachable(F' transition, initial, steps, image)
```

Lean proves that this verifier-soundness premise is exactly enough to supply
`PriorAuthoritySound`, and therefore enough to compose with a sound latest-step
verifier. It also proves the negative form: any compressed verifier that accepts
an unreachable image is not sound prior authority.

The module also provides the concrete bridge needed for a compressed verifier:
`SoundVerifier` packages a verifier predicate with the theorem that every
accepted compressed prior proof opens to a proof-carrying
`FoldedFPrimeAuthority` for the same `(steps, image)`. Lean derives both
`VerifierSound` and `PriorAuthoritySound` from that object. This keeps the
proof-system object opaque while making the production obligation exact. The
module also records the replay separation explicitly: `SoundVerifier` alone
does not imply that one opaque proof is functional for a single prior
`(steps, image)` pair. A replay-stable terminal theorem needs the stronger
fixed-opener surface supplied by the prior-opening modules.

`DirectCcsFPrime.DirectTerminalSoundness` composes the required terminal direct
CCS `F'` theorem boundary. Its canonical accumulator update is:

```text
ParentSourceStep:
  Pi_CCS output claims -> Pi_RLC parent CE(B) source

AuthorizedNextPiCCSInputs:
  digest-bound parent CE(B)
  opened parent residues
  canonical private Pi_DEC
  child CE(b)^k witness table
  wire identity into next Pi_CCS
```

Lean proves that accepted terminal compression with proof-carrying prior
authority reaches the final public image. It also proves the strengthened
latest-step uniqueness property: under `Pi_CCS` stage functionality, `Pi_RLC`
stage functionality, encoded parent digest binding, deterministic
parent-statement commitment encoding, Ajtai no-collision, and the CE-opening
adapter, an accepted latest transition cannot be replaced by another transition
from the same prior image that changes the parent `CE(B)` source or authorized
next child inputs.

For the implementation-facing deterministic path, the module also provides a
computed-stage specialization. If the accepted `Pi_CCS` and `Pi_RLC` stages are
represented as:

```text
out = computePiCCS(i, prior)
source = computePiRLC(i, out)
```

then Lean discharges the two stage-functionality premises by construction. The
stronger deterministic statement-commitment specialization also discharges the
non-cryptographic statement-encoding premise. The remaining assumptions are the
real security boundaries: encoded parent digest binding, Ajtai no-collision,
and the CE-opening adapter.

The narrowest specialization replaces the CE-opening adapter with a canonical
Ajtai-backed commitment map for the fixed CE commitment map and replaces the
local no-collision premise with the theorem-facing MSIS boundary. That makes
the remaining terminal opening/security premises implementation-shaped:

```text
AjtaiBackedCommitMap(params, ce.commitMap)
MSISToAjtaiReductions(params)
MSISHardnessAssumption(params)
```

instead of separate local `CEOpeningAdapter` and `NoAjtaiBindingCollision`
premises.

`DirectCcsFPrime.DirectConcreteInstantiation` removes the remaining abstract
CE commitment-map freedom from that terminal theorem. It builds the terminal
CE relation from concrete CE data whose commitment map is exactly:

```text
assignment -> M || M * toOpening(assignment).witness
```

and proves that this concrete CE data supplies the required
`AjtaiBackedCommitMap`. Its terminal theorem then combines canonical Ajtai CE
data, encoded parent `CE(B)` digest binding, MSIS-to-Ajtai reductions, MSIS
hardness, proof-carrying folded F' authority, and the accepted latest direct
F' step. The theorem still deliberately leaves `computePiCCS`, `computePiRLC`,
the direct boundary relation, and folded F' authority as concrete protocol
objects to instantiate.

`DirectCcsFPrime.DirectProgramStep` removes the abstract latest direct boundary
relation from that theorem shape. It fixes the public computation boundary as
a deterministic function:

```text
nextBoundary = computeBoundary(step, priorBoundary)
```

Lean proves that two accepted latest direct `F'` transitions from the same
prior image cannot disagree on the public computation boundary, and that each
accepted transition exposes the exact `computeBoundary(step, priorBoundary)`
value. Composed with the reduced accumulator-field functionality, the strongest
theorem proves the entire latest public image is unique:

```text
final image is reachable
and nextImage = altNext
```

The concrete frontend still owns the actual `computeBoundary` function. This
module closes the proof-shape gap where an arbitrary boundary relation could
hide a different public output for the same prior image.

`DirectCcsFPrime.DirectStageSemantics` removes the bare-stage-computation
boundary from the strongest direct theorem. The terminal theorem no longer has
to be stated directly over naked `computePiCCS` and `computePiRLC` functions.
Instead, those functions are packaged as verified stage computations:

```text
computePiCCS(i, prior_accumulator)
computePiRLC(i, pi_ccs_output)
```

along with the imported SuperNeo theorem statements for `Pi_CCS` and `Pi_RLC`,
and explicit frontend predicates saying that the encoded direct outputs match
those theorem contexts. The accepted stage relations themselves require equality
to the deterministic computed value, the frontend bridge predicate, and the
imported SuperNeo theorem statement:

```text
VerifiedPiCCS = computed output + output bridge + Pi_CCS strong theorem
VerifiedPiRLC = computed source + source bridge + Pi_RLC weak theorem
```

The strongest terminal theorem uses those verified accepted relations directly.
This prevents the terminal proof boundary from silently ignoring the imported
SuperNeo stage evidence.

The stage package also has a direct reuse adapter for the existing SuperNeo
CE-relation surface:

```text
ceRelation(ctx) -> Pi_CCS strong theorem
ceRelation(ctx) -> Pi_RLC weak theorem
```

so a direct CCS frontend can reuse the already-formalized SuperNeo stage
authority instead of inventing a parallel theorem route.

The module also provides a context-carried reused-stage shape for direct
frontends. In that shape, the computed `Pi_CCS` output carries the exact
SuperNeo context it is about, and the `Pi_RLC` context is computed from the
accepted `Pi_CCS` output and parent source. This makes the output/source bridge
predicates definitional equalities in the adapter, instead of caller-chosen
predicates.

The strict compressed-prior terminal theorem is also exposed directly for this
reuse adapter, so the production-shaped theorem path consumes the imported
SuperNeo CE-relation authority without a manual conversion step.

The proof-carrying folded-authority terminal theorem is exposed directly for
the same reuse adapter. This gives the theorem-level induction path before a
concrete compressed verifier is instantiated.

The context-carried reuse adapter exposes the same compressed-prior and
proof-carrying terminal theorem shapes. That is the tighter direct frontend
entry point because accepted Pi_CCS/Pi_RLC data determines the SuperNeo context
used by the imported theorem statements.

It also exposes the `SoundVerifier` entry point, where the compressed prior
verifier is packaged together with the theorem that every accepted proof opens
to folded `F'` authority for the same public image.

The reduced-handle production entry point additionally consumes
`ParentCEBHashBinding.ParentCEBHash` directly. That means callers supply the
canonical parent `CE(B)` hash-binding object, not a loose digest-binding proof
threaded separately from the hash function.

`DirectCcsFPrime.DirectStageSuperNeoReuse` is the reuse adapter for the
existing `formal/superneo-lean` Section 7.1 theorem-native context. A direct
contextual stage can be constructed from deterministic `computePiCCS` and
`computePiRLC` functions plus upstream `ProtocolSection71Context` owner objects
whose targets are exactly the contexts carried by the computed `Pi_CCS` output
and computed for the `Pi_RLC` parent source. The adapter derives the direct
`ReusedStageAuthority` fields from upstream `ceRelation`, exposing the imported
`Pi_CCS`, `Pi_RLC`, and `Pi_DEC` theorem surfaces without creating a parallel
authority path.
`DirectParentOnlyProductionSuperNeoReuse` exposes those same imported
statements directly for the production context's actual computed stage
contexts: `computePiCCS(i, prior).ctx` and
`piRLCContext(out, computePiRLC(i, out))`. This keeps the implementation-facing
context anchored to Section 7.1 stage authority rather than a loose pair of
stage functions.

The main terminal theorem also no longer requires the prior folded `F'`
authority to be represented as a Lean proof-carrying object. It accepts any
authority predicate whose acceptance implies reachability of the prior public
image under the direct `F'` transition:

```text
AuthorityAccepts(steps, authority, image)
  => Reachable(F', initial, steps, image)
```

The proof-carrying `FoldedFPrimeAuthority` remains as a convenience
instantiation. A compressed-proof implementation must instantiate this same
soundness implication for its verifier.

The module also provides the production-shaped compressed-prior specialization.
It combines verified direct stages with a compressed prior verifier satisfying:

```text
VerifyPrior(steps, proof, image)
  => Reachable(F', initial, steps, image)
```

This removes the need for callers to manually compose the generic authority
theorem with `CompressedFPrimeAuthority`.

The strict compressed-prior specialization additionally derives the terminal
public-image invariants from that same accepted chain:

```text
final.step = priorSteps + 1
final.vkDigest = initial.vkDigest
final.initialBoundary = initial.initialBoundary
final.pc = 1
```

So the production theorem surface proves both the recursive authority claim and
the exposed Construction-2 public-state shape.

`DirectCcsFPrime.Construction2DirectFPrime` instantiates the latest-step side
of that induction theorem for the direct CCS public image:

```text
(vkDigest, step, initialBoundary, currentBoundary, accumulator, pc)
```

The direct path fixes `pc = 1`. A latest-step transition advances `step` by
one, preserves `vkDigest` and `initialBoundary`, updates the computation
boundary through `BoundaryStep`, and updates the folded F' accumulator handle
through `AccumulatorStep`. The canonical latest-step verifier accepts exactly
that transition, so the Lean theorem discharges `LatestStepSound` for this
public-image shape.

The module also proves the public-image invariants inherited by any accepted
reachable chain from a well-formed zero-step base:

```text
final.step = number_of_accepted_F'_steps
final.vkDigest = initial.vkDigest
final.initialBoundary = initial.initialBoundary
final.pc = 1
```

These facts are derived from reachability, not from independently trusted
digests. The remaining obligations are to instantiate
`BoundaryStep` and `AccumulatorStep` with the actual direct CCS application
step and NIFS.V accumulator update, and to supply the folded prior-authority
soundness theorem.

`DirectParentOnlyProductionSuperNeoReuseFinalEndpoint` is the internal
replay-stable terminal call surface. Its production-facing path consumes the
canonical `PriorVerifierAuthorityOpening` certificate directly and returns
opened folded `F'` authority, same-proof replay stability, computed stage
evidence, and explicit pointwise no-swap evidence in one package. The module
also provides named projections for opened prior authority, replay stability,
computed-stage evidence, and the pointwise no-swap witness, so consumers do
not rely on conjunction layout or digest self-consistency. It also exposes
direct projections for the computed `Pi_CCS` and `Pi_RLC` DEC knowledge facts
from that computed-stage evidence.
Those projections are also available directly from
`PriorVerifierAuthorityOpening`, which keeps the production call path on the
canonical verifier-opening certificate once the concrete verifier soundness
proof exists.
Its `pointwiseChildTableNoSwap_ofPriorVerifierAuthorityOpening` theorem is the
adversarial-child-table form: an alternate private DEC table must satisfy the
full `PointwisePrivateDecRequirements` predicate for the same parent source,
and then Lean identifies it with the audited table used by both terminal
images and returns the exact `Pi_CCS -> Pi_RLC` parent-source audits.
It also exposes the final Construction-2 public-image facts directly:
reachability of the accepted final image, next-step value, verifier-key digest
preservation, initial-boundary preservation, and final well-formedness.

`DirectParentOnlyProductionSuperNeoReuseEndToEnd` is the final composed theorem
surface for the Section 7.1-backed parent-only path. The production interface
advertises the exact-runtime prior-verifier entry point from
`DirectParentOnlyProductionFPrimePriorVerifier`:

```text
RuntimeExactSurface
RuntimeExactLayout
RuntimeExactVerify priorSteps priorProof priorImage
VerifyLatestStep priorSteps priorProof priorImage nextImage latestProof
=>
CertifiedTerminalEndToEnd
```

The canonical theorem is `endToEndOfRuntimeExact`. It returns one named package
containing opened folded `F'` authority, same-proof replay stability, computed
stage evidence, exact computed `Pi_CCS` DEC knowledge over the pointwise private
child table, exact computed `Pi_RLC` DEC knowledge for both replayed parent
sources, explicit no-swap evidence, quantified no-swap for every alternate table
satisfying full pointwise private DEC requirements, the flattened
non-aggregate private DEC and stage facts, the flattened public endpoint, the
terminal stage audit, the Section 7.1 owner-target audit, and final public-image
invariants.

The exact-runtime prior-verifier surface carries verifier replay, terminal
public-IO layout, the fixed authority opener, and the backend soundness theorem
that turns acceptance into folded `F'` authority for the same `(steps, image)`
pair. Digest replay remains verifier binding data; authority comes from the
opened folded reachability object.

The production projections are:

| Alias | Meaning |
|---|---|
| `endToEndOfRuntimeExact` / `terminalEndToEnd` | Exact-runtime prior verifier acceptance plus latest step gives the terminal end-to-end package. |
| `privateDecFactsOfRuntimeExact` / `privateDecFacts` | Extracts the non-aggregate private DEC stage certificate. |
| `privateDecNoSwapAuditOfRuntimeExact` / `privateDecNoSwapAudit` | Extracts the alternate-child no-swap audit. |
| `stageAuditOfRuntimeExact` / `section71StageAudit` | Extracts the Section 7.1 owner-target audit. |
| `privateDecCertificate_of_nonAggregatePrivateDecStageFacts` | Extracts the accepted private `Pi_DEC` certificate. |
| `uniquePrivateChildren_of_nonAggregatePrivateDecStageFacts` | Extracts pointwise private-child no-swap uniqueness. |
| `nextPiCCSInputs_eq_childWitnessDigitTable_of_nonAggregatePrivateDecStageFacts` | Extracts exact child witness-table to `Pi_CCS` wire identity. |

`DirectCcsFPrime.DirectParentOnlyProductionExactRuntimeInstantiation` is the
short production instantiation over exact verifier checks:

| Alias | Meaning |
|---|---|
| `productionVerifyPriorOpens` | Exact prior verifier acceptance opens folded `F'` authority for the same `(steps, image)` pair. |
| `productionVerifyPriorReaches` | Exact prior verifier acceptance proves prior folded reachability. |
| `productionVerifyPriorRejectsUnreachable` | Exact prior verifier acceptance cannot authorize an unreachable prior image. |
| `productionTerminalSoundness` | Production exact checks plus runtime authority soundness and latest-step evidence give the parent-only terminal package. |
| `productionPrivateDecFacts` / `productionPrivateDecNoSwapAudit` / `productionSection71StageAudit` | Production exact projections for private DEC facts, no-swap audit, and Section 7.1 audit. |

`DirectCcsFPrime.DirectParentOnlyProductionConcreteFPrimePrior` focuses the
compressed prior `F'` authority obligation into one verifier-body object. Its
checks correspond to compact-image replay, Construction-2 public-boundary
replay, Fiat-Shamir/transcript replay, committed-step verifier acceptance, and
a fixed authority opener. The production exact verifier surface proves that the
fixed opener is nonempty, that any opened authority binds to the same
`(steps, image)`, that unreachable prior images are rejected, and that the
terminal end-to-end package follows from exact-runtime verifier acceptance.

### Soundness Boundary Map

The parent-only optimization relies on these boundaries:

| Boundary | Lean object | Security role |
| --- | --- | --- |
| Parent `CE(B)` binding | `Poseidon2ParentCEBHash.Hash` and `ParentCEBHash` | Binds the public parent handle to one canonical parent encoding, assuming Poseidon2 binding. |
| Opened parent residues | `ParentOpeningAuthorization.BindsOpenedParentCEBResiduesFor` | Connects the parent handle to per-column residues used by private `Pi_DEC`. |
| Private DEC child correctness | `PointwisePrivateDecCertificate` | Requires accepted private `Pi_DEC`, binary length-14 digits, and per-column recomposition, not an aggregate checksum. |
| Fixed child CE membership | `FixedCEChildMembershipCertificate` and `PointwiseChildAuditCertificate` | Fixes CE/Ajtai parameters, CE satisfaction, openings, witness-table identity, and next-`Pi_CCS` wire identity. |
| Child reuse/no-swap | `CertifiedTerminalNonAggregatePrivateDecStageCertificate` | States that every alternate child table must satisfy the full pointwise private DEC requirements before it is identified with the audited table. |
| Prior folded `F'` authority | `CertifiedPriorVerifier` and `PriorVerifierAuthorityOpening` | Turns accepted compressed prior proofs into opened folded reachability authority for the same `(steps, image)`. |
| Public image invariants | `CertifiedTerminalEndToEnd` | Preserves Construction-2 reachability, step, verifier-key digest, initial boundary, and well-formedness. |

### Production F' Reader Map

The production prior-verifier path is organized by proof responsibility:

| Module | Responsibility |
| --- | --- |
| `DirectParentOnlyProductionConcreteFPrimePriorRawProductionRaw` | Raw verifier checks: compact image replay, Construction-2 replay, Poseidon2 transcript replay, raw public-vector acceptance, raw opening, and raw certified-verifier consequences. |
| `DirectParentOnlyProductionConcreteFPrimePriorRawProductionExactAccepted` | Structured terminal/boundary public-IO acceptance, bound-statement acceptance, exact opening surface, exact certified verifier, and exact terminal theorem. |
| `DirectParentOnlyProductionConcreteFPrimePriorRawProduction` | Compatibility facade for the complete production verifier surface. |
| `DirectParentOnlyProductionConcreteFPrimePriorBackendBase` | Generic runtime backend verifier surface and authority consequences. |
| `DirectParentOnlyProductionConcreteFPrimePriorBackendExactPublicIO` | Exact terminal/boundary public-IO adapter into the generic backend surface. |
| `DirectParentOnlyProductionConcreteFPrimePriorBackendRawPublicIO` | Raw public-vector adapter into the generic backend surface. |
| `DirectParentOnlyProductionConcreteFPrimePriorBackend` | Compatibility facade for the complete backend surface. |

The checks-first constructor, `ConcreteVerifierBodyChecks`, splits the
implementation-facing verifier obligation into two narrower claims: replay binds a verifier
statement, and committed-step verifier soundness for that statement opens the
same proof to folded `F'` authority. This is the intended continuation point
for the Rust verifier-body ABI represented by `direct_ccs/f_prime.rs`,
`direct_ccs/f_prime_verifier_body.rs`, `direct_ccs/terminal_committed.rs`, and
`direct_ccs/verify.rs`.
`ConcreteVerifierStatementBinding` tightens the replay side again by requiring
the opaque proof's verifier statement to equal `canonicalStatement steps image`.
This specifies the Rust verifier ABI shape where the caller's public
`(steps, image)` pair determines the exact statement checked by the committed
verifier body.
`ConcreteVerifierStatementSurface` then names the concrete direct CCS verifier
checks in that statement: final public statement validity, exact equality
between the proof boundary and the statement's Construction-2 boundary, and
terminal committed proof verification against both expected terminal public
values and that boundary. Its one backend soundness obligation says those
checks, plus the fixed opener, produce folded `F'` reachability authority for
the same `(steps, image)` pair.
The base concrete verifier module now exposes the direct F' consequences at
every layer: `verifyPrior_reaches_prior`,
`verifyPrior_openedAuthority_accepts_of_open`,
`verifyPriorOfChecks_reaches_prior`,
`verifyPriorOfChecks_openedAuthority_accepts_of_open`, and the corresponding
canonical-statement and statement-surface variants. These make the authority
movement explicit before the terminal-IO specialization: accepted concrete
prior verifier evidence reaches the claimed prior image, and any authority
returned by the fixed opener accepts exactly the same `(steps, image)` pair.
`DirectCcsFPrime.DirectParentOnlyProductionConcreteFPrimePriorTerminalIO`
refines that terminal committed verifier check into the public-IO checks made
by `verify_direct_ccs_terminal_committed_relation`: the decoded-and-verified
terminal proof public values must have the expected folded `F'` terminal public
values as a prefix and the expected Construction-2 public boundary encoding as
a suffix. Its `AcceptedTerminalIOEvidence` eliminator exposes the full
non-digest evidence behind verifier acceptance: compact-image replay,
Construction-2 boundary replay, transcript replay, proof/canonical statement
equality, statement validity, proof/statement boundary equality, decoded
terminal public IO prefix/suffix checks, and the opened folded `F'` authority
for the same `(steps, image)` pair.
The module also exposes `AcceptedPriorPublicImageInvariants`, proving that
accepted concrete prior proofs force `image.step = steps`, preserve the initial
verifier digest and boundary, and keep the prior image well formed. The
terminal-IO surface also supplies same-proof functionality plus direct
latest-step projections to the non-aggregate private DEC/stage facts and the
Section 7.1 owner-target audit. Its direct folded-authority theorems,
`verifyPriorOfTerminalIO_reaches_prior` and
`verifyPriorOfTerminalIO_openedAuthority_accepts_of_open`, make the F'
authority consequence explicit: terminal public-IO acceptance reaches the prior
image, and any concrete authority returned by the fixed opener accepts exactly
that same `(steps, image)` pair.
`DirectCcsFPrime.DirectParentOnlyProductionConcreteFPrimePriorBackend`
separates runtime verifier acceptance from authority extraction. Its
`RuntimeVerifyPrior` predicate contains only verifier-visible replay,
public-boundary, and terminal public-IO checks; it does not include an
`openAuthority` existence premise. The single backend soundness field turns
those checks into an opened folded `F'` authority for the same `(steps, image)`,
which Lean packages as `certifiedPriorVerifierOfRuntimeBackend`. Its direct
consequences, `runtimeVerifyPrior_reaches_prior` and
`runtimeVerifyPrior_openedAuthority_accepts_of_open`, make the authority bridge
auditable without treating digest replay or public-IO shape as authority. The
`runtimeVerifyPrior_evidence` eliminator additionally exposes the replay
checks, proof/canonical-statement equality, terminal public-IO prefix/suffix
checks, and opened folded authority in one package. The backend soundness field
itself receives the replay checks and replay-derived statement equality, so a
concrete instantiation cannot ignore compact-image replay or transcript replay
while claiming folded `F'` authority. The latest-step projections return the
same parent-only CE binding, no-swap, private DEC/stage facts, Section 7.1
audit, and public-image invariant package from runtime verifier acceptance.
The same backend module also defines `ConcreteRuntimeExactPublicIOSurface`,
matching the implementation-facing committed-step verifier ABI where Spartan
returns one public vector and the verifier checks exact equality with
`terminal F' public values ++ Construction-2 boundary public values`. Its
`AcceptedExactPublicIOEvidence` eliminator records the raw concatenation
equality, the terminal split, the boundary split, replay-derived statement
equality, and the opened folded authority. This is the production-shaped path
for avoiding a weak prefix/suffix-only public-IO model.
For the implementation-facing raw verifier ABI, the module exposes
`ConcreteRuntimeRawPublicIOSurface`: the terminal committed verifier returns a
raw public vector, and `RuntimeVerifyPriorOfRawPublicIO` accepts only when that
raw vector is exactly
`terminal F' public values ++ Construction-2 boundary public values`. Its
`certifiedPriorVerifierOfRawPublicIO` packages that raw-vector verifier into
the same certified prior verifier consumed by the end-to-end parent-only CE
binding theorems.
`DirectCcsFPrime.DirectParentOnlyProductionConcreteFPrimePriorRawIO` exposes
the audit-facing consequences of that raw-vector path: accepted raw verifier
results open authority, cannot authorize unreachable prior images, force the
prior public-image invariants, are same-proof functional, and feed the certified
terminal end-to-end package plus the non-aggregate and Section 7.1 stage-audit
projections.
`DirectCcsFPrime.DirectParentOnlyProductionConcreteFPrimePriorRawIOSoundness`
splits that raw-vector authority boundary into verifier-visible checks,
replay-to-canonical-statement binding, and one trusted
`rawBoundStatementAuthoritySound` theorem. This keeps the Poseidon2/compressed
verifier assumption at the exact point where an accepted raw public vector for a
bound `(steps, image)` statement opens folded `F'` authority.
`DirectCcsFPrime.DirectParentOnlyProductionConcreteFPrimePriorRawIOOpening`
refines that boundary by splitting the authority assumption into opener
existence and opened-authority statement binding. Accepted bound raw public IO
must open through the fixed opener, and the opened proof-carrying authority
must carry the same `(steps, image)` fields as the bound statement. Lean then
derives `FoldedFPrimeAuthority.Accepts`, packages the certified prior verifier,
and exposes the reachability, public-image invariant, same-proof functionality,
and unreachable-prior consequences.
`DirectCcsFPrime.DirectParentOnlyProductionConcreteFPrimePriorExactIOOpening`
lifts the opening-level boundary to the exact public-IO verifier shape without
using the older monolithic exact-runtime soundness field. Exact public IO is
projected to the raw Spartan public vector, terminal-length binding turns raw
concatenation equality into exact terminal and boundary slices, and the trusted
cryptographic obligations are only opener existence plus opened-authority
statement binding for exact bound statements. Lean derives folded `F'`
acceptance, certified prior verification, reachability, public-image
invariants, same-proof functionality, unreachable-prior rejection, and the
parent-only terminal end-to-end and audit projections.
`DirectCcsFPrime.DirectParentOnlyProductionConcreteFPrimePriorBackendOpening`
packages the same opening discipline in a backend-shaped exact public-IO
surface. The verifier body exposes compact-image replay, Construction-2
boundary replay, transcript replay, statement-boundary equality, terminal
committed-proof public IO, and a fixed authority opener. Lean proves that direct
backend checks lift to the certified opening verifier, open folded `F'`
authority for the same `(steps, image)` pair, reach the prior image, preserve
public-image invariants, reject unreachable prior images, are same-proof
functional, and make the fixed opener value authoritative: `some authority`
must accept the same public pair, while `none` cannot verify. The module also
feeds the parent-only terminal end-to-end theorem together with the
non-aggregate private DEC/stage facts and Section 7.1 stage audit. Existing
exact-runtime verifier surfaces that package backend SNARK soundness as
`exactRuntimeSound` can be reused through this stronger fixed-opener boundary
only when paired with exact public-IO layout binding, so raw concatenation is
not treated as terminal/boundary split authority.
`DirectCcsFPrime.DirectParentOnlyProductionConcreteFPrimePriorExactIOLayout`
turns the exact-public-IO layout obligation into an implementation-facing
terminal-length condition. If the verifier output's raw vector equals
`terminal F' public values ++ Construction-2 boundary public values` and its
terminal slice has the canonical terminal length, Lean proves the terminal and
boundary slices are exactly the canonical slices. The canonical-prefix adapter
derives that length condition when the implementation exposes
`terminal = raw.take(canonical terminal length)`. The module exposes the
terminal-length induced certified prior verifier and direct `F'` consequences:
accepted verification opens folded authority for the same `(steps, image)`,
reaches the prior image, preserves public-image invariants, rejects unreachable
prior images, and feeds the parent-only terminal end-to-end and audit
projections.

The project remains short of an end-to-end direct CCS `F'` proof. The
theorem-level proof-carrying induction path discharges `PriorAuthoritySound`.
The remaining necessary theory blockers are:

```text
1. instantiate `ConcreteRuntimeExactPublicIOOpeningSurface`, or reuse an
   exact-runtime verifier surface through
   `runtimeExactPublicIOOpeningSurfaceOfExactRuntimeSoundAndLayout`, for the
   intended compressed prior-verifier ABI. The connection must cover the
   compact-image replay, Construction-2 boundary replay, transcript replay,
   statement-boundary equality, terminal committed-proof public IO, exact
   terminal/boundary layout binding, and fixed authority opener. The bridge
   must prove that accepted runtime checks open every proof to
   `FoldedFPrimeAuthority` for the same `(steps, image)` pair;
2. instantiate `DirectParentOnlyProductionSuperNeoReuse.ProductionContext` for
   the concrete direct CCS application transition, NIFS.V parent `CE(B)` source
   update, implementation parent hash object, concrete initial public image,
   and Section 7.1-backed stage contexts accepted by
   `DirectStageSuperNeoReuse`.
```
