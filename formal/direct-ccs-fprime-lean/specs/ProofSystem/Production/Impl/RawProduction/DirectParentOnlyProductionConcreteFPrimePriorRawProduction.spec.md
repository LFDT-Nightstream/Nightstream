# DirectParentOnlyProductionConcreteFPrimePriorRawProduction

This module specifies the production-shaped raw public-IO bridge for the
compressed prior `F'` verifier used by the parent-only terminal path.

The verifier surface is the RV64IM compressed-main proof view: a published
statement, an expected IVC public image derived from that statement, a
SNARK-carried IVC public image, Poseidon2 proof-binding digests, the terminal
`F'` committed-step proof public vector, final carried CE claim checks, and a
fixed authority opener.

The verifier predicate is authority-free. It consists of:

- compressed-public-image replay for the same `(steps, image)` pair;
- published-statement and Construction-2 boundary replay;
- final carried CE canonicality, binding, and verifier acceptance;
- Poseidon2 transcript replay for the proof digest and binding digest;
- terminal committed-step verifier public IO equal to the exact concatenation
  of terminal `F'` public values and Construction-2 boundary public values.

The trusted backend boundary is restricted to the production view. Once the
above verifier checks bind the proof-carried IVC image to the canonical
statement, the backend must open the fixed authority opener and the opened
authority must bind the same `(steps, image)` pair. Lean derives
`FoldedFPrimeAuthority.Accepts` from those two facts and packages the result as
a certified prior verifier and as a strict `SoundVerifier`.

The module also exposes a structured exact-IO production verifier surface. In
that surface, the terminal committed proof returns explicit terminal `F'` and
Construction-2 boundary public-IO slices, and acceptance requires pointwise
equality to the canonical terminal and boundary values. The raw vector remains
available as the concatenation carried by the exact public-IO object, but
authority is still derived only through the fixed backend opener.

The structured exact-IO production verifier instantiates the backend-shaped
exact public-IO opening surface once the terminal committed proof binds its
terminal slice to the canonical terminal length. This layout binding is a
verifier requirement on the exposed public-IO split; it is separate from
Poseidon2 soundness and from backend SNARK soundness.

The verifier may satisfy this layout requirement by exposing the terminal slice
as the canonical-length prefix of the raw terminal proof public vector. This
canonical-slice discipline induces the terminal-length binding and forces the
terminal and boundary slices through exact list positions, not through an
aggregate relation.

The production-facing exact theorem surface can consume this canonical-slice
discipline directly. It packages the induced backend exact public-IO opening
surface, certified prior verifier, strict `SoundVerifier`, authority opening,
terminal acceptance, and terminal end-to-end projections without requiring
callers to pass a separate terminal-length object.

A single production exact authority certificate is the trusted backend boundary
for this surface. The certificate states that accepted bound verifier evidence
opens the fixed authority object and that the opened authority proves folded
`F'` reachability for the same `(steps, image)` pair. Lean derives the
backend-opening and opened-authority binding obligations from this one
certificate rather than accepting them as independent caller-supplied facts.

The certificate-native terminal theorem surface consumes production exact
checks, this single authority certificate, and canonical terminal-slice binding
directly. It derives the certified prior verifier, strict `SoundVerifier`,
terminal acceptance, terminal end-to-end package, non-aggregate private
DEC/stage facts, and Section 7.1 stage audit from those verifier-owned inputs.

The runtime authority-soundness surface is the concrete backend obligation that
instantiates that certificate. It consumes verifier-replayed compact-image,
Construction-2 boundary, Poseidon2 transcript, canonical-statement, and exact
terminal/boundary public-IO facts, then opens the fixed authority object to
folded `F'` authority for the same `(steps, image)` pair. This keeps backend
SNARK soundness as the cryptographic boundary while making the certificate an
artifact of concrete verifier facts.

Those same verifier facts induce the generic exact-runtime backend verifier
surface. Production exact verification can therefore be viewed as the existing
backend exact public-IO verifier, and its accepted evidence includes the
replayed statement, exact terminal/boundary public IO, and opened folded `F'`
authority.

The generic exact-runtime surface also packages directly as a certified prior
verifier and strict `SoundVerifier`. This gives the production exact runtime
path a theorem-facing prior-verifier object from concrete verifier facts alone,
with terminal end-to-end, non-aggregate private DEC/stage, and Section 7.1
audit projections following through the existing backend exact-public-IO
theorems.

The production exact compressed-verifier soundness surface states the same
authority boundary over `ProductionExactVerifierAccepted` itself. Accepted
production exact verifier evidence opens the fixed authority object to folded
`F'` reachability for the same `(steps, image)` pair, and Lean derives the
runtime authority-soundness surface, generic exact-runtime evidence, direct
certified prior verifier, strict `SoundVerifier`, same-proof functionality,
terminal end-to-end package, non-aggregate private DEC/stage facts, and Section
7.1 audit projections from that single verifier-acceptance boundary.

The exported theorems provide:

- accepted production raw verifier checks open folded `F'` authority;
- strict production raw `SoundVerifier` acceptance opens fixed folded `F'`
  authority for the same public pair and rejects missing or unreachable
  openings;
- accepted structured production exact-IO verifier checks open folded `F'`
  authority;
- structured production exact-IO verification maps into the backend-shaped
  exact public-IO opening verifier and opens authority through that bridge;
- canonical production exact terminal-slice binding induces the terminal
  length binding consumed by the backend-shaped bridge;
- the backend-shaped production exact bridge packages as a certified prior
  verifier, a strict `SoundVerifier`, and a terminal end-to-end theorem;
- the backend-shaped production exact bridge exposes the same terminal,
  non-aggregate private DEC/stage, and Section 7.1 audit projections as the
  rest of the certified verifier stack;
- canonical production exact terminal-slice binding exposes the same certified
  verifier, strict `SoundVerifier`, authority opening, terminal, non-aggregate
  private DEC/stage, and Section 7.1 audit projections directly;
- a single production exact authority certificate instantiates the production
  exact opening surface and exposes accepted verifier evidence as folded `F'`
  authority for the same public pair;
- production exact checks plus the single authority certificate and canonical
  terminal-slice binding expose the certified verifier, strict `SoundVerifier`,
  terminal acceptance, terminal end-to-end package, non-aggregate private
  DEC/stage facts, and Section 7.1 audit projections;
- production exact runtime authority soundness instantiates the single
  authority certificate and exposes the same certified verifier, strict
  `SoundVerifier`, terminal acceptance, terminal end-to-end package,
  non-aggregate private DEC/stage facts, and Section 7.1 audit projections;
- production exact runtime authority soundness also induces the generic
  exact-runtime backend surface and exposes accepted production verification
  as backend exact public-IO evidence;
- the induced generic exact-runtime backend surface packages directly as a
  certified prior verifier, strict `SoundVerifier`, terminal acceptance,
  terminal end-to-end package, non-aggregate private DEC/stage facts, and
  Section 7.1 audit projections;
- production exact compressed-verifier soundness over
  `ProductionExactVerifierAccepted` induces the runtime authority-soundness
  surface, generic exact-runtime evidence, direct certified prior verifier,
  strict `SoundVerifier`, same-proof functionality, terminal end-to-end
  package, non-aggregate private DEC/stage facts, and Section 7.1 audit
  projections;
- production exact opening surfaces and single authority certificates induce
  the same compressed-verifier soundness object over
  `ProductionExactVerifierAccepted`;
- compressed-verifier soundness induces the single production exact authority
  certificate consumed by the terminal-slice theorem surfaces;
- accepted `ProductionExactVerifierAccepted` evidence opens folded `F'`
  authority and is accepted directly by the induced certified prior verifier
  and strict `SoundVerifier`;
- strict structured production exact-IO `SoundVerifier` acceptance opens folded
  `F'` authority for the same public pair and rejects missing or unreachable
  openings;
- accepted production raw verifier checks reach the claimed prior image;
- unreachable prior images cannot be accepted;
- accepted prior public-image invariants are exposed;
- same-proof functionality is inherited by strict verifiers;
- exact production runtime verification composes directly with the latest-step
  verifier into the terminal end-to-end package, without a caller-supplied
  authority-opening premise;
- production raw and structured exact verifiers compose with the latest-step verifier into the
  existing terminal end-to-end package, including parent-only CE binding,
  no-swap evidence, non-aggregate private DEC/stage facts, and the Section 7.1
  stage audit.

Poseidon2 collision/binding soundness and backend SNARK soundness are external
cryptographic assumptions at the production backend boundary. The module does
not implement Poseidon2 and does not replace pointwise terminal public-IO
equality with aggregate checks.
