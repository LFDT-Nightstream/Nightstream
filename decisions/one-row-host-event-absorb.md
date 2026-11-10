# One-Row Host-Event Absorb

**Status:** Accepted

## Problem

The WASM relation commits each host-event block to the event chain with one
width-12 Poseidon2 permutation (`neo_application::event_commitment`). A
trace row of the relation is the same for every opcode, so a design that
runs one permutation round on each row needs 19 full trace rows for each
block. In the Starstream test transactions, these rows are more than half of
all trace rows. Every row also pays for a gadget that it does not use.

## SuperNeo

SuperNeo v1.2 and Nebula do not define the host-event chain. The chain is the
commitment that the WASM relation and the Starstream interleaving proof share;
its hash and block layout are fixed by `neo_application::event_commitment`.

## Decision

The owner chose this design on 2026-11-09.

1. **Absorb on the gather row.** The gather row that stages slot seven of a
   block that is not advice folds the block into the chain on that row:
   `chain_after = commit_block(previous, [evbuf_before[0..7] | staged word])`.
   The row clears the buffer. There are no permutation rows.
2. **Same digest.** The hash, the block layout, and the order of blocks do
   not change, so the chain digest of a run is the digest of the earlier
   design. `previous` is the chain before the row, or the restored outer chain
   when the row also copies the last root lane of an opaque object.
3. **One gadget for each group of rows.** The owner chose this rule on
   2026-11-10; before it, each row held its own permutation. Each group of
   `gcd(k, 8)` consecutive rows of a step of `k` rows shares one block: the
   permutation's input lanes, its witness words and its output lanes
   (`SPEC.md` section 7.6, "Shared columns"). A row that absorbs requires
   the block's input to be its own input and lands on the block's output, so
   soundness needs no bound on the absorbs of a group. Two absorbs are at
   least eight rows apart (`Lifecycle.Nebula.SharedAbsorb.absorbs_apart`),
   so an honest group has at most one.
4. **The staged word in the trace.** A gather row records its word
   (`WasmVmStep::host_event_word`), because the absorb row clears the buffer
   that held it.
