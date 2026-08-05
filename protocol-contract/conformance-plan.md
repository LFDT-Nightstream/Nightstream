# Rust-origin conformance plan

Status: **evidence plan; no selected-profile Rust-origin suite supplied**.

## Evidence pipeline

```text
canonical input
    |
pinned Rust verifier -----> observed ordered trace and decision
    |                                      |
    +--------------------------------------+
                                           |
                              independent semantic checker
                                           |
                              contract-bound comparison result
```

The semantic checker computes the expected result from authoritative input.
Neither a vector nor a Rust artifact may provide a trusted expected Boolean.

## Vector families

| Family | Required cases |
|---|---|
| Decoding | noncanonical field, wrong magic or version, unknown or duplicate section, wrong order, truncation, trailing bytes |
| Shape | wrong row count, assignment width, sparse Structure order or duplicate, matrix count, M0, lifted polynomial, padding, or profile hash |
| Strict norm and split | boundary roots, rejected `-2` and `2`, signed-bit order, prover assignment error, verifier public-input error |
| Joint PiCCS | source order, alpha or gamma order, 24 round recurrences, degree, terminal order, absolute target |
| Norm binding | substitute the M0 terminal from another witness; change a zero-padded row |
| PiRLC | non-constant ring action, source order, guard boundary, sampler coefficient order |
| PiDEC | sign, digit order, child count, commitment, public input, and every `y_ring` recomposition |
| Transcript | tag, frame length, exact schedule count, event order, padding, ratchet, continuation, challenge decoder, verifier-key prehash |
| Sampler | `q-2` accepted, `q-1` rejected, first/second/third attempt, exhaustion |
| Forbidden v1 fields | FE/NC variant, column point, carrier replay, beta challenge, undeclared relation field |
| Circuit | public alias, omitted field, unconstrained hint, stale manifest, row-owner mismatch |
| Terminal | public substitution, backend mismatch, parser mutation, unsupported manifest |

Each applicable normative rejection rule needs at least one mutation. Each
protocol event needs at least one accepting trace that reaches it.

## Trace record

Each trace event contains:

```text
sequence number
protocol event ID
normative rule IDs
exact Rust source symbol
canonical input and output hashes
observed branch or first rejection
```

Positive traces must reach the expected terminal state for the tested
component. A trace with an event after rejection is invalid.

## Differential executions

For each applicable vector, run:

1. the pinned native Rust verifier;
2. the independent semantic checker;
3. the circuit witness generator and constraint checker for circuit rules;
4. the terminal verifier for public parsing and final acceptance rules.

Agreement between two implementations is evidence of agreement only. The
semantic theorem and circuit correspondence establish the stronger claims.

## Aggregation

The release evidence index must have one entry for each applicable rule and
profile. It binds every vector, execution, build, and trace hash. Coverage
closes only when all required positive and mutation families pass and the
independent checker recognizes the same contract and profile hashes.
