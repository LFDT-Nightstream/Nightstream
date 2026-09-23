# Wide sampler review

Reviewed on 2026-09-23 at `919cd12b4`, the five commits on `c9aa75b04` from
`origin/nico/pirlc-wide-sampler`. The source was unavailable initially and
was fetched after the owner confirmed its push.

Decision: merge as an unselected, proved component. No mathematical defect
was found in the stated component contracts. Production selection remains
open. The selected relation still has 3,595,629 logical CCS rows and
149,597,982 committed coordinates; this merge earns no additional reduction.

The combined production library and test target pass (4,250 jobs), including
all 71 new exported-theorem audits and one added audit of the `FormalCircuit`
record. They use only `propext`, `Classical.choice`, and `Quot.sound`.
The static boundary gate passes. The lakefile conflict was resolved by
retaining both the compact-PiCCS and wide-sampler test roots.

## Arithmetic checked independently

Let `p = 18446744069414584321`, `N = 5^54`, `M = p^4`, and
`r = M mod N`. Four independent uniform field values, interpreted as
base-`p` digits, give a uniform integer below `M`. Reduction modulo `N`
has total variation distance `r * (N-r) / (N*M)` from uniform on `Fin N`.
An exact rational calculation gives distance below `2^-132`; its binary
logarithm is approximately `-132.9796469221`. Summing the exact distances
for 17 scalar draws gives approximately `2^-128.8921840808`.

This is a sampler distance, not total protocol security. For a parameter
`q` counting applicable fresh sampler calls, the usual hybrid loss is at
most `q * distance` when each conditional draw meets the same bound.
The transcript and oracle model must establish those conditions.

The maximum quotient needs 131 bits. The reported variable partition sums
to 617: four 66-variable `CanonicalU64` children, 131 quotient bits,
162 digit bits, and 60 check-quotient bits. The reported 681 rows give
11,577 rows for 17 scalars before hashing. These arithmetic checks do not
establish the circuit's row count, committed width, or matrix entry count.

## Findings and remaining integration

The current `PiRlcSampler` specification selects the first 54 accepted
16-bit chunks and carries a bounded-shortfall outcome. Its production
schedule advances through eight complete digest blocks per scalar. Wide
reduction changes this deterministic map and its state transition. It can
be reviewed and merged as an unselected component without changing the
selected specification. Production selection requires the corresponding
specification, transcript, identity, and correctness/security links.

V6 is a valid comparison of sampled acceptance with the existing uniform
extractor. `CoordinateTerminalLaw.returningProbability` on the right side
still averages over uniform challenge vectors. It does not identify the
return probability of a different sampler-driven extractor or a Fiat-Shamir
execution. The theorem comment now states this boundary directly. The
vector bound uses independent four-field draws; the transcript integration
must supply the appropriate conditional law and cumulative query/use count.

The CRT source checks all six moduli and their coprimality. Soundness bounds
both sides of every check row below Goldilocks before lifting equality to
integers. It then combines the congruences and bounds both integers below
the modulus product. The digit range fixes `R < 5^54`. Completeness builds
the quotient, digits, and check quotients, completes the four canonical
children, and preserves all external variables. No extra semantic premise
or trusted witness value enters soundness.

The completion theorem is existential, with explicit arithmetic witnesses
inside its proof. It does not supply an executable witness program or prove
the hint interpreter computes that completion. The footprint theorem is
conditional on a program allocating exactly 353 new values. Its comment
now states that condition without implying such a program already exists.

The coordinate estimate needs a retained layout. A count of field-valued
DSL variables is not a low-norm coordinate count. Prove the chosen slot
kinds and witness mapping; count normalized matrix nonzero entries and
the actual Poseidon2 schedule separately. Do not report the estimated
0.13M coordinates as an achieved reduction.

## Solver controls

`wide_sampler_controls.py` reads the reviewed commit's actual gadget
parameters. Its seven cvc5 checks pass under the 300-second project cap.
They check digit range, field-wrap exclusion under the actual bit bounds,
and integer equality after the separately proved CRT congruence. Removing
the digit-range row accepts digit 5; removing the check-quotient range permits
field wrap. An explicit bounded false witness with `X = 0` and
`Q*N+R = product(first five moduli)` passes the first five check rows and is
rejected by the sixth. The trace is saved in `wide_sampler_controls.json`.
These local checks and mutation replays supplement the Lean proofs.

No package or fixture was regenerated. No Rust test or benchmark was run.

## Sources consulted

- [SuperNeo, September 4 revision](https://eprint.iacr.org/2026/242):
  Appendix B.3 uses uniform challenges and separates extraction loss.
  The local sectioned source is `docs/superneo-paper-v1_2`.
- [RFC 9380, Section 5](https://www.rfc-editor.org/rfc/rfc9380.html#section-5):
  wide reduction is an established method to control modular bias. Its
  byte-based construction is context, not a proof of this field-based map.
- [cvc5 finite-field theory](https://cvc5.github.io/docs/latest/theories/finite_field.html)
  and the [maintainer discussion of integer/field checks](https://github.com/cvc5/cvc5/discussions/11911):
  integer range and modular-lift checks need an explicit model. Solver
  counterexample checks must target the actual branch equations.
