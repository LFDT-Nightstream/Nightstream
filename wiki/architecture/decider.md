# Terminal proof and decider

The terminal path closes the final accumulator and can then compress that
relation with WIP Spartan. Four components have separate authority.

## Public statement

`paper/decider.rs` owns the public image and the prover witness used for the
decider statement. Its validation walk replays the lifecycle and checks that
the final witness matrices open the recorded commitments. This is a preflight,
not a replacement for the circuit relation.

## Direct terminal relation

`paper/decider_ce_relation/` emits the terminal committed-evaluation rows. It
checks:

- the Ajtai opening;
- the public-input projection;
- the low-norm alphabet;
- each ring evaluation at the joint point; and
- the constant term of each ring evaluation.

The lifecycle verifier also performs the native form of these checks.

## Full-history audit R1CS

`engine/decider.rs` emits an audit relation that replays the complete
history. Its size is linear in the number of steps. It is useful for audits and
red-team tests. It is not the constant-size recursive terminal relation.

## Terminal Spartan bridge

`frontends/r1cs_f_prime/terminal_r1cs/` compiles the combined terminal R1CS
statement and calls `wip-spartan`. The backend proves a direct sparse R1CS
over Goldilocks with a Poseidon2 transcript and WHIR openings.

The backend is connected, but it remains work in progress. It needs
cryptographic review, performance work, and deployment integration before it
can be treated as a production compression backend.
