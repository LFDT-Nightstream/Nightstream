# WASM Fixed Sizes

**Status:** Accepted

## Problem

A memory circuit has one geometry and one region for each memory. If the
prover makes the regions from its run, each run shape is a new circuit, the
verifier compiles a circuit for each shape, and data of the run shapes the
circuit of the verifier. One size for all runs makes each small run pay for
the largest memory, because each segment scans every cell of the ROM and the
RAM (`SPEC.md` section 7.1).

## SuperNeo

SuperNeo v1.2 defines no memory checking. Nebula fixes the geometry of one
circuit and leaves open how an application selects its regions and its
geometry.

## Decision

The owner asked on 2026-11-02 for this choice to be made without Enzo, and
approved it on 2026-11-03.

1. **Three sizes.** `neo_wasm::nebula::WasmSize` has Small, Medium and
   Large, with ROM and RAM of `2^14`, `2^15` and `2^16` cells. `2^16` is the
   largest that rule 6 permits (`SPEC.md` section 7.1). A size is a set of
   limits, a geometry, and the rows of one step.
2. **Limits by dimension.** Each address part of a WASM memory counts one
   thing: a program counter, a function reference, an event, a slot, a stack
   cell, a word of linear memory, and so on (`WasmLimits`). A size gives a
   width to each of these. Program counters have 8, 9 and 10 bits, and
   function references 5, 6 and 7 bits. Each size has 3 bits for the events
   of a function, 3 bits for the slots of an event, and 1 bit for a control
   choice. In the RAM, linear memory takes half of the space and the stack a
   quarter. The four smaller test transactions of Starstream `159bedc` fit
   Small, and the two larger fit Medium.
3. **Steps and segments.** Small and Medium prove 8 rows in each step, in
   segments of 64 and 128 steps. Large proves 2 rows in each step, in
   segments of 128 steps, because rule 6 permits few slots with `2^17` cells.
   Each size holds runs of up to 16,384 trace rows.
4. **The prover selects, the verifier takes.** The prover takes the
   smallest size that holds the program and its run. The size is part of the
   statement, and the verifier compiles the circuit of that size. It never
   takes a region or a geometry from the run.
5. **Security.** Each size keeps rule 6, so one segment's memory fingerprint
   error is at most `2^−109.91`. The memory gate charges every allowed
   segment and the final comparison. It permits at most 106 bits for Small
   or Medium, and 103 bits for Large. The folding gate can set a lower limit.
   The external challenge-query and collision premises remain in
   `formal/nightstream-fprime/SECURITY_MODEL.md`, trust item 7.
