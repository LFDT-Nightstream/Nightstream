# Two-result WASM exports

## Status

Accepted.

## Problem

WASM permits an export to return two values. A single output pair cannot
represent both in the public proof state. A clean return with two values
therefore had no valid witness.

## SuperNeo

SuperNeo v1.2 defines the folding relation, not the WASM export result ABI.

## Decision

The public state carries the last result and, when present, the result before
it. Each pair is bound to its own stack-memory read at the clean return.
The second capture implies the first. A disabled pair has zero value limbs.
Exports with more than two results are
unsupported. Host-event output references still require one result.
SPEC.md section 7.7 defines the accepted relation.
