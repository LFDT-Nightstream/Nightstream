# Decider (Terminal Compression)

The decider is the terminal check of the folded accumulator — the thing that makes the
whole chain's validity a single verifiable statement. It is split across four modules
with distinct authority. The compact proof itself ("PR5") and its backend connection
are the main open work items.

## paper/decider.rs — the contract

Owns the **statement** a compact terminal proof must bind, and a non-SNARK
`validate_witness` preflight.

- **Public**: `PublicImage` — the chain-binding coordinates a verifier recomputes from
  preprocessing: vk_fs digest, counters, `z_0`, `z_i`, `pc`, `acc_digest`, public
  trace, `x_out`.
- **Witness**: prover-side material — step proofs, public batches, the terminal fold
  proof, the post-finalization state with the final running accumulator and witness
  matrices.

`validate_witness` replays every step plus the final fold (the chain-replay authority
path — a superset of `verify_uncompressed`), recomputes the public image, and asserts
it matches the statement; it also checks the final witness matrices commit to the
claims' commitments. It is a preflight, not the proof relation: the terminal CE
obligations remain circuit rows. `decider::prove` / `verify` are `Unsupported`
placeholders.

## paper/decider_ce_relation — the sound direct verifier

The terminal CE relation, checked natively against the NIFS-produced children:
commitment opening, public-input projection `X` from `Z`, low-norm bound,
`y_ring = mle(M_j·Z)(r)`, and the implementation invariant `ct = lane0(y_ring)`. This
is what the production verification paths use today; isolation tests:
`tests/system/decider_ce_relation_isolation.rs`.

## paper/terminal_ce — the compact public boundary

The backend-neutral public statement shape (`TerminalCePublic`), a Merkle commitment
over claim material (`merkle.rs`), and a **fail-closed** circuit entrypoint for the
future compact terminal-CE proof. Explicitly *not* an accepting verifier yet: a
matching public digest is binding material, not authority, and the module doc says so.
Tests: `tests/system/terminal_ce_public.rs`, `terminal_ce_merkle.rs`.

## engine/decider.rs — the full-history audit R1CS

Packages a validated `decider::Statement` into a self-contained R1CS that **replays
every lifecycle/F′ step and the terminal fold** in-circuit: canonical base-state pins,
every base/recursive F′ step, adjacent state links, full CE continuity, terminal
NIFS.V, terminal latest-links, public-image pins, and terminal CE rows against the
NIFS-output children.

Scope warning from the module doc: this is an **audit artifact, linear in history
length** — useful for auditing the direct-CCS interim path, not the constant-size
HyperNova decider, and not a production-compression sizing reference. The
constant-size terminal decider belongs to the F′ frontend path, where each online step
folds `enc(F′)` and the final SNARK proves only the terminal accumulator.

Shape snapshot:
`cargo test -p neo-fold-clean --release --test perf_fibonacci_bits -- --ignored --nocapture fibonacci_decider_r1cs_shape_snapshot`
(chain length via `NEO_FOLD_FIB_DECIDER_VALUES`).

## In-circuit verifier gadgets

The R1CS gadget layer the decider composes lives in `engine/r1cs_circuit/` (builder,
booleans, u64s, mux, extension-field ops, Poseidon2, transcript, sum-check,
ring-action, alphabet sampling) and the per-reduction verifier circuits in
`paper/reductions/`: `pi_ccs_circuit/` (the one-joint PaddedRowIdentity Π_CCS
verifier), `pi_rlc_circuit/`, `pi_dec_circuit.rs`, plus
`paper/nifs/circuit/mod.rs` composing them, its `pi_rlc/` subtree owning the
PiRLC phases, and `paper/f_prime/r1cs.rs` owning the F′ step.
Each gadget family has a dedicated test target (`gadgets_*`, `reductions_*` — see
[Testing](../development/testing.md)).

## Toy Spartan — standalone candidate backend

`crates/toy-spartan` contains a standalone WHIR-backed Spartan engine, but it is not
connected to this decider or the lifecycle compression path. See
[Toy Spartan](../crates/toy-spartan.md) for the implemented boundary and limitations.
