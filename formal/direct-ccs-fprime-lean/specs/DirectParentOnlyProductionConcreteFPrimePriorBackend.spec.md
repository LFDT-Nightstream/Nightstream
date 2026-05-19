# DirectParentOnlyProductionConcreteFPrimePriorBackend

`DirectParentOnlyProductionConcreteFPrimePriorBackend` specifies the runtime
backend authority bridge for the production concrete prior `F'` verifier.

The runtime verifier predicate contains only verifier-visible checks:

```text
compact image replay
Construction-2 boundary replay
transcript replay
canonical statement public validity
proof boundary = canonical statement boundary
terminal committed proof verifies to public IO
terminal public IO has expected folded F' values as a prefix
terminal public IO has expected Construction-2 boundary values as a suffix
```

The raw public-IO specialization models the production committed-step verifier
shape where the verifier returns one public vector and accepts only when that
vector is the exact concatenation:

```text
terminal_public_values(statement)
+ boundary_public_values(statement_boundary(statement))
```

Accepted raw-layout verification exposes this raw-vector equality together
with the replay facts, proof/canonical-statement equality, and opened folded
authority. This rules out a verifier model that validates independent prefix
and suffix predicates while leaving room for unaccounted public fields between
them.

The exact split specialization is an optional structured view of the same
terminal/boundary layout for backends that expose the split in addition to the
raw vector.

Authority extraction is separate from that predicate. The backend surface has a
single soundness obligation:

```text
runtime checks for (steps, proof, image)
and replay-derived proof statement = canonical statement
  => exists authority:
       openAuthority(proof) = some authority
       and FoldedFPrimeAuthority.Accepts(steps, authority, image)
```

This obligation is the cryptographic backend boundary. It may rely on
Poseidon2 binding, Fiat-Shamir replay, and compressed terminal proof soundness,
but the Lean verifier predicate itself does not treat digest consistency or
public-IO shape as folded `F'` authority.

The certified-prior-verifier constructor packages the runtime verifier
predicate, the fixed authority opener, and the accepted-opens theorem into the
same `CertifiedPriorVerifier` shape consumed by the production end-to-end
theorems.

Verifier acceptance has direct folded-authority consequences:

```text
RuntimeVerifyPrior(surface, steps, proof, image)
=> Reachable(initial, steps, image)

RuntimeVerifyPrior(surface, steps, proof, image)
and openAuthority(proof) = some authority
=> FoldedFPrimeAuthority.Accepts(steps, authority, image)
```

The evidence eliminator for runtime acceptance exposes the replay checks,
the proof/canonical-statement equality derived from replay, the terminal
public-IO prefix/suffix checks, and the opened folded authority in one audit
package. This ensures compact-image replay, Construction-2 boundary replay, and
transcript replay remain visible as statement binding rather than disappearing
inside the backend soundness boundary.

The production-facing latest-step theorem consumes runtime verifier acceptance
and the accepted latest Construction-2 step, then returns the terminal
end-to-end package with parent-only CE binding, no-swap evidence, exact private
DEC/stage facts, Section 7.1 audit data, and public-image invariants.
