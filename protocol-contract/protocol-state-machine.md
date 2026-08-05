# Nightstream v1 protocol state machine

This document explains the authored records in `src/protocol/`. The generated
`protocol-events.json` is the machine-readable view. The selected profile is
`PaddedRowIdentity`; no rectangular column branch remains.

## State progression

```text
START
  -> DECODED
  -> PROFILE_BOUND
  -> PICCS_INPUT
  -> PICCS_COINS
  -> PICCS_SUMCHECK
  -> PICCS_OUTPUT
  -> PICCS_ACCEPTED
  -> PIRLC_SAMPLING
  -> PIRLC_ACCEPTED
  -> PIDEC_ACCEPTED
  -> FOLD_ACCEPTED
```

Any failed check changes the result to `REJECT`. Unknown variants, unknown or
noncanonical fields, wrong counts, wrong order, and sampler exhaustion reject.
The terminal backend is outside this one-fold state machine.

## Challenge order

The transcript first binds the session, verifier-key digest, statement, and
ordered PiCCS inputs. It then samples 24 `alpha` coordinates and one `gamma`.
For SumCheck round `j`, the verifier absorbs `j` and the ten ordered extension
coefficients, checks the round recurrence, and only then samples round
challenge `j`. There are exactly 24 rounds.

After the 210 ordered PiCCS ring outputs and the joint terminal check are
bound, the verifier starts PiRLC sampling. For each source, coefficient, and
attempt tuple, it absorbs the three indices and samples one base-field
candidate. It stops at the first accepted candidate. Three rejected attempts
for one coefficient reject the proof.

PiRLC output, the 14 PiDEC children, and fold finalization are absorbed in
that order. Each fold starts a new zero-state duplex. The final four-field
fold transcript digest is a verifier-derived receipt. It is not a CE field and
is not an input to the next fold. `src/protocol/transcript-schedule.json` fixes
each frame payload count, squeeze count, tag, and loop nesting. The generated
`protocol-events.json` includes the same schedule.

`src/protocol/rejections.jsonl` is the complete proof-failure registry. Every
code is used by an event and cites its normative rule. The sampler value `q-1`
is a local retry and is not a proof failure unless all three attempts exhaust.
A total verifier-derived transition can have no rejection code; the checker
does not require an artificial failure branch.

## Bounded repetition

Repetition is separate from the assurance DAG. Its counters are protocol
data, not workflow status.

| ID | Count or bound | Order | Failure |
|---|---:|---|---|
| `REP-PICCS-ROUNDS` | exactly 24 | round 0 through 23 | reject |
| `REP-RHO-SOURCES` | exactly 15 | source 0 through 14 | reject |
| `REP-RHO-COEFFICIENTS` | exactly 54 per source | coefficient 0 through 53 | reject |
| `REP-RHO-ATTEMPTS` | 1 through 3 per coefficient | attempt 0 through 2 | reject on three failures |
| `REP-FOLD-STEPS` | 1 through 64 | fold 0 through `fold_count-1` | reject an additional fold |

Thus one fold derives exactly 810 accepted strong-set digits and uses at most
2,430 sampler candidates.

A bounded sequence is an ordered array of one-fold statement-proof pairs. All
statements use the same `fold_count`; indices are exactly `0..fold_count-1`.
For each adjacent pair, the 14 ordered PiDEC children of fold `j` are the 14
ordered running claims of fold `j+1` as canonical typed CE claims. The first
running state is an input to this fold contract. The last child state is its
output. The application base case and the terminal decider remain separate
protocol layers.

## Ownership

The fold verifier key owns the profile, Structure, and setup identity. The
decider key owns its separate backend manifest. Proof fields are messages only.
A value becomes relation authority
only through the exact `c`, `x`, `r`, and `y_ring` CE fields. The verifier
recomputes all caches, public-carrier evaluations, and digests.

The Structure uses the canonical sparse stream in the profile. The verifier-key
digest uses a fresh selected field duplex over the domain, profile version,
setup code, dimensions, 32 seed-byte lanes, and that Structure stream. The
digest is transcript input only; it does not replace verifier-key authority.

The circuit and deployed verifier are assurance boundaries, not later events
in this verifier state machine. The circuit must implement this same fold
relation. The decider must prove the selected circuit relation. Their current
conformance evidence remains open until G4 and G5.
