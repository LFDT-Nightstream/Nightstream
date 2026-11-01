# WHIR query redraws

## Status

Accepted.

## Problem

Plonky3 redraws a WHIR query sample when it equals `p − 1` at the selected
Goldilocks query widths. The compression circuit has a fixed shape, so it
cannot follow an unbounded loop whose length depends on private samples.

## SuperNeo

SuperNeo v1.2 does not select the WHIR sampler for Nightstream's compression
layer.

## Decision

Allow one initial sample and at most two redraws per query. Check each
rejection and use the first admissible sample. A disabled redraw must
preserve the complete duplex state, including buffered output words.
The circuit shape must be independent of the rejection pattern.

Three consecutive rejected samples remain a completeness failure. This
policy does not provide Plonky3's unlimited redraw behavior. The limit is
per query, not per group of queries. SPEC.md Algorithm 8 defines the rule.
`formal/nightstream-fprime/SECURITY_MODEL.md` states the remaining limit.
