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

The owner approved this on 2026-11-03, the shapes of item 3 on 2026-11-09,
the segment counts of item 5 on 2026-11-14, and the four sizes of items 1
to 3 and 5 on 2026-11-16 (`wasm-light-compression.md`).

1. **Four sizes.** `neo_wasm::nebula::WasmSize` has Micro, Small, Medium
   and Large, with ROMs of `2^11`, `2^12`, `2^13` and `2^14` cells. The RAM
   of Micro has an eighth of the cells of its ROM, and the RAM of each other
   size a quarter. A segment scans each cell of both memories, and the
   program tables in the ROM need far more cells than a run's stack and
   linear memory: the test transactions of Starstream `159bedc` use at most
   106 RAM cells. A size is a set of limits, a geometry, and the rows of one
   step.
2. **Limits by dimension.** Each address part of a WASM memory counts one
   thing: a program counter, a function reference, an event, a slot, a stack
   cell, a word of linear memory, and so on (`WasmLimits`). A size gives a
   width to each of these. In the order program counter, function, event,
   slot, stack, linear memory, global, call depth, host depth, table and
   table index, the widths are 8 5 3 3 6 7 2 2 2 1 2 for Micro, 8 6 7 3 8
   9 5 4 4 1 4 for Small, 9 6 7 3 9 10 6 5 4 1 5 for Medium, and 10 7 8 3
   10 11 7 6 4 1 6 for Large. Each event slot has one descriptor and one 32-bit
   payload. Descriptor bits 0 through 4 encode the kind and advice flag,
   bits 5 through 12 the argument, bits 13 through 16 the variant, and
   bits 17 through 24 the type flag or memory-write input index. The payload
   holds a 32-bit constant, byte offset, or type flag. A 64-bit literal
   uses two slots in low-word-first order.
3. **Steps and segments.** Micro proves 800 rows in each step, Small 672,
   and Medium and Large 664. A segment of Micro and of Small has 1 step, of
   Medium 2 and of Large 4. Micro has at most 30 segments, Small 24, Medium
   12 and Large 6 (item 5), so Micro holds runs of up to 24,000 trace rows
   and each other size of up to about 16,000. The fixed cost of a step
   changes little with its rows, so fewer and larger steps cost less for
   each row. The rows of each size are a selected count, and the light
   relation of each size fits the Akita key of `2^19` blocks
   (`akita-compression.md`). Small fills the key most closely. The four
   short test transactions of Starstream `159bedc` fit Micro, and `score`
   and `distinct_utxo_types` fit Small.
4. **The prover selects, the verifier takes.** The prover takes the
   smallest size that holds the program and its run. The size is part of the
   statement, and the verifier compiles the circuit of that size. It never
   takes a region or a geometry from the run.
5. **Security.** Each size keeps rule 6, so the memory fingerprint error of
   one segment is at most `2^−109.91`. A proof has up to `S_max` segments,
   and the runtime gate charges each one (`SPEC.md` section 7.1). Each size
   has the most segments for which `S_max · 2·m_mem/q²` is at most `2^−109`,
   so a caller asks for at most 109 bits of security. With 8 memory slots
   in each row, `m_mem` is 8,704, 10,496, 20,864 and 41,728, so the counts
   are 30, 24, 12 and 6 (`Spec.Nebula.WasmSize.shape`,
   `Spec.Nebula.WasmSize.segmentBits_eq`,
   `Spec.Nebula.WasmSize.segmentBits_succ_lt`).
6. **No refresh fold.** A proof of a size has at most `N · S_max` steps,
   which is 30, 24, 24 and 24. Each is at most `cap + 1 = 65`, so no proof
   reaches the refresh fold of step 66, and the final proof has two claims
   (`Spec.Nebula.WasmSize.steps_le_light`, `wasm-light-compression.md`).
