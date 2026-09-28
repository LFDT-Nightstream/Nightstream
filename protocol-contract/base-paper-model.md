# Base SuperNeo paper model

Status: **historical, non-normative source extraction**.

This file records the relevant pre-errata rules reconstructed by exact reverse
application of errata v4. It exists to show what the reviewed patch changed. A
base rule cannot override a reviewed erratum.

## Base algebra and relations

`BASE-FND-001` The base paper defines `n_F=d*n_R` and
`n_F,in=d*n_R,in`, with protocol bounds `B=b^k<q/2`. It does not define the
integer universal ambient bound `B_amb`.

`BASE-REL-001` The base Structure writes `f` with an individual-degree-looking
annotation and calls it a degree-`u` polynomial. Its CCS relation contains the
incorrect integer-set expression instead of the Boolean zero set. Its `L_in`
domain is the input width instead of the full witness width. Its CE witness and
transformed evaluation notation are also inconsistent.

Source: reconstructed base hashes for `SRC-PAPER-04` and `SRC-PAPER-07`.

## Base joint PiCCS

`BASE-PICCS-001` The base protocol assumes `m=n_F`, one power-of-two cube, and
an identity first matrix. It defines the norm product with the same lower and
upper endpoint, so it does not test all centered values in the strict range.
It shifts the carried polynomial by `gamma^(2K+k)` but gives SumCheck the
unshifted local target. It does not state the corrected
`D_Q=max(D_f+1,2b,2)` individual-degree bound.

Source: reconstructed base hashes for `SRC-PAPER-07` and `SRC-PAPER-13`.

## Base extraction and PiRLC

`BASE-SEC-001` The base strong-extraction proof uses the old ambient
`CE(q/2,L)` notation and does not contain the reviewed success-gated global
`sqrt(delta)` argument with separate SumCheck and Schwartz--Zippel terms.

`BASE-PIRLC-001` The base PiRLC text has inconsistent challenge indexing and
does not state the corrected per-coordinate fork loss. Its proof also contains
the old projection and ambient-relation errors.

Source: reconstructed base hashes for `SRC-PAPER-07`, `SRC-PAPER-12`, and
`SRC-PAPER-13`.

## Base PiDEC and concrete profile

`BASE-PIDEC-001` The base PiDEC display omits the public-input recomposition
equation and uses inconsistent transformed-matrix evaluation notation.

`BASE-PROFILE-001` The base Appendix B.2 still gives `d=54` and `n_F=2^30`.
It therefore has the same unresolved dimension defect as the reviewed paper.

Source: reconstructed base hashes for `SRC-PAPER-07` and `SRC-PAPER-11`.

## Assembly rule

The authority chain is:

```text
base paper model
  + reviewed errata v4
  = reviewed paper model in literal-paper-model.md
  + approved Nightstream decisions
  = normative contract.
```

The checker verifies the byte derivation. Independent review must verify this
semantic mapping.
