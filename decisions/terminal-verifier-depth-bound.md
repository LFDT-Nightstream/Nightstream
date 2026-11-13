# The terminal verifier holds a slot to the bound of its depth

## Status

Accepted.

## Problem

HyperNova Definition 11 (i) requires that the prover can extend each proof
that the verifier accepts. With bounded-depth folds, a light fold adds
`T(b − 1)` to the norm of a slot witness. A verifier that admits each slot
witness below `B` accepts proofs that the prover cannot extend. Such
a proof needs a second opening of a slot commitment, so no honest run and no
known attack makes one, but the definition has probability 1.

## SuperNeo

SuperNeo v1.2 Definition 21 gives a CE witness a norm below `B`, and each of
its folds decomposes the accumulator, so one bound is sufficient there. It has
no light fold. Cyclo, ePrint 2026/359, Theorem 3 item 1 gives the accumulator
relation a bound that grows by a fixed amount at each fold.

## Decision

The terminal verifier requires, for each slot, a depth of at most the light
cap and a witness within `accumulatorBound` of that depth
(`Export.BoundedDepth.RunCompleteness.Verifies`, [SPEC.md](../SPEC.md) Algorithm 6).
One valid step extends each proof that this verifier accepts
(`Export.BoundedDepth.RunCompleteness.verifies_step`).

The terminal relation of the security results keeps `B`
(`Lifecycle.BoundedDepth.Terminal.HoldsFor`). Extraction reaches only that bound,
because a refresh fold recovers its input below `B`. The verifier accepts only
what the terminal relation accepts
(`Export.BoundedDepth.RunCompleteness.holds_of_verifies`), so each security bound
applies to it. The step circuits, the packages and the proof bytes do not
change.
