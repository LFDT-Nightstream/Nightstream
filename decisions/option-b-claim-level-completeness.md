# Option B completeness at the claim level

## Status

Accepted.

## Problem

The Lean results must show that an honest option-B run gives a final proof
that the verifier accepts. The final proof is one layer-1 argument per slot
and one shrink argument over the final program. Each is a sum-check and WHIR
argument. Lean models these arguments as games for soundness, with WHIR
idealized, and has no honest prover for them.

## SuperNeo

SuperNeo v1.2 is a folding scheme. It does not fix a compression layer for the
folded claims, so it does not fix what the completeness of compression must
state.

## Decision

Completeness of compression is proved at the claim level. For every valid run
of at least one step, the claim that the final verifier checks holds for the
honest statement (`CompressionLean.Claims.option_b_complete`,
[SPEC.md](../SPEC.md) section 1). This needs no cryptographic assumption.

The completeness of the arguments themselves is an assumption: the honest
layer-1 and shrink provers produce proofs that the verifiers accept. The Rust
tests of SPEC.md lifecycle rule 2 are its evidence
([SECURITY_MODEL.md](../formal/nightstream-fprime/SECURITY_MODEL.md), Out of
scope). No theorem states it with a probability. SPEC.md Algorithm 7 names the
events that make an honest `compress` fail: a WHIR query sample of `p − 1` and
the layer-1 poles.

The two premises that join the slot witnesses to layer 1 are checks of Rust's
key files, with a soundness theorem
(`CompressionLean.Claims.option_b_covers_of_check`). The norm cover requires
that the depth-64 accumulator bound is at most the layer-1 norm bound. The
block cover requires that each package's logical width fits in the blocks that
layer 1 commits.
