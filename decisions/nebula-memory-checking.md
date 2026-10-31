# Nebula Memory Checking

**Status:** Accepted

## Problem

An application needs public ROM and RAM. Nightstream must add Nebula memory
checking without a change to the F′ relation, the SuperNeo fold or the
HyperNova state link, and must select a security level for the memory
fingerprint.

## SuperNeo

SuperNeo v1.2 defines no memory checking. HyperNova Construction 2 links the
steps through a hash of the state. Nebula (Sections 3 and 4) checks memory in
two layers: one proof of the steps, then proofs of `F_ops` and `F_scan` after
the challenges, joined by `F_final`. It takes the fingerprint challenges in
the base field and carries its commitment as a separate field of the state
hash.

## Decision

The owner authorized Stage 2 on 2026-10-07. `SPEC.md` section 7 states the
protocol, and section 8 lists the differences from Nebula.

1. **One F′ chain.** A memory application is a Stage 1 application. Each step
   adds its own packed records to three Poseidon2 hash chains (Nebula
   Section 3.1). The prover proposes the ops and final-snapshot roots when a
   segment opens, `η` comes from them, and the step that closes the segment
   checks the proposals. One chain cannot make a second pass over the steps.
2. **The carry is in the application-state digest** (owner, 2026-10-07). The
   four Stage 1 state words are a digest of the application words and the
   memory carry, so the F′ state hash, layout and theorems do not change.
3. **Challenges in `K`.** Over `F_q`, one pair gives only about 46 bits at
   the largest geometry. Over `K`, the fingerprint error is
   `2(R + M + N · B_ops)/q²`.
4. **Floor `2^−109.91`** (owner, 2026-10-05). Setup rejects a geometry with
   `R + M + N · B_ops > 139,509` (`SPEC.md` section 7.1, rule 6). A larger
   memory needs a new decision. A second, independent challenge pair would
   give about `2^−219.8` for about 2% more coordinates. It is not used.
5. **The memory round is a premise** (owner, 2026-10-07; form chosen
   2026-10-12). The relation computes `η` with Poseidon2 inside the step, so
   no random-oracle proof covers it. `SECURITY_MODEL.md` trust item 7 states
   it in the form of HyperNova errata Assumption 1, as the Stage 1 history
   premise. The adversary time and the query counts stay symbolic
   (2026-10-12).
6. **Packages.** Lean emits two test packages for the machine application.
   A package for a production geometry waits for the production application
   (2026-10-12).
7. **Several trace rows in one step** (owner, 2026-10-31). A catalog
   application can prove a fixed number of trace rows in each step, so that
   the rows share the fixed F′ cost of a fold. The step uses the batched
   catalog of `SPEC.md` section 7.6, and Lean proves that batched steps are
   memory consistent exactly when their rows are. The protocol, the F′
   relation and the memory rows do not change. On CPU, 8 WASM rows took
   68.7 s at one row in each step and 29.4 s at two rows in each step. On
   Metal, select the number of rows by a measurement: at 8 rows, a fold has
   about 62.3M coordinates, and above 60.5M the Metal prover does not keep
   the commitment key.
