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
   Large, with ROMs of `2^14`, `2^15` and `2^16` cells. The RAM of a size has
   a sixteenth of the cells of its ROM. A segment scans each cell of both
   memories, and the program tables in the ROM need far more cells than a
   run's stack and linear memory: the test transactions of Starstream
   `159bedc` use at most 106 RAM cells. A size is a set of limits, a
   geometry, and the rows of one step.
2. **Limits by dimension.** Each address part of a WASM memory counts one
   thing: a program counter, a function reference, an event, a slot, a stack
   cell, a word of linear memory, and so on (`WasmLimits`). A size gives a
   width to each of these. Program counters have 8, 9 and 10 bits, and
   function references 5, 6 and 7 bits. Each size has 3 bits for the events
   of a function, 3 bits for the slots of an event, and 1 bit for a control
   choice. In the RAM, linear memory takes half of the space, the stack a
   quarter, and each of the two global tables a thirty-second. The four smaller test transactions of Starstream `159bedc` fit
   Small, and the two larger fit Medium.
3. **Steps and segments.** Small and Medium prove 64 rows in each step, in
   segments of 8 and 16 steps. Large proves 32 rows in each step, in
   segments of 16 steps. Small has at most 10 segments, Medium 5 and Large
   3 (item 5), so Small and Medium hold runs of up to 5,120 trace rows and
   Large of up to 1,536.
   The fixed cost of a step changes little with its rows, so fewer and
   larger steps cost less for each row. The owner approved these shapes on
   2026-11-09 after CPU measurements on memories with the same step shape: a
   512-row Small segment took 89 s in place of 251 s, and a Large row took
   0.28 s in place of 1.35 s. Small and Medium keep 512 and 1,024 rows in
   each segment. Large has 512 rows in each segment, so a short Large run
   pads to 512 rows. Its rule 6 count is 77,312, below 139,509.
4. **The prover selects, the verifier takes.** The prover takes the
   smallest size that holds the program and its run. The size is part of the
   statement, and the verifier compiles the circuit of that size. It never
   takes a region or a geometry from the run.
5. **Security.** Each size keeps rule 6, so the memory fingerprint error of
   one segment is at most `2^−109.91`. A proof has up to `S_max` segments,
   and the runtime gate charges each one (`SPEC.md` section 7.1). Each size
   has the most segments for which `S_max · 2·m_mem/q²` is at most `2^−109`,
   so a caller asks for at most 109 bits of security. On 2026-11-13 the
   owner kept 109 bits with fewer segments, in place of 107, 107 and 105
   bits with the earlier 32, 16 and 32 segments.
