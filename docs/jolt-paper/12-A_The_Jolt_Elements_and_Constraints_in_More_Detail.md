## A The Jolt Elements and Constraints in More Detail

The table below lists the elements committed by the prover per CPU step for instructions that are not loads or stores (see Table 2 for the elements involved there).

| ELEMENT                                | PURPOSE                                                                                                            | #SIG. BITS                              |
|----------------------------------------|--------------------------------------------------------------------------------------------------------------------|-----------------------------------------|
| opflags[14]                            | These are 1-bit elements used to guide the constraint system on if-else branches. (List given below.)              | 1 bit $\times 14$                       |
| opcode[8]                              | These bits constitute the 8-bit <b>opcode</b> for the instruction.                                                 | 1 bit $\times 8$                        |
| rs1, rs2, rd                           | Indices of the step instruction’s source and destination registers.                                                | 5 bits $\times 3$                       |
| PC                                     | The program counter (possible two for certain programs with virtual instructions).                                 | $\log( \text{code} ) \times 2$          |
| step_counter                           | The global timestamp incremented every step.                                                                       | $\log(\#\text{steps})$ bits             |
| read_ts_code                           | The timestamp passed as advice for memory-checking when reading the program code.                                  | $\log(\#\text{steps})$ bits             |
| read_ts_rs1,<br>read_ts_rs2            | The timestamps passed as advice for memory-checking when reading the values at registers <b>rs1</b> , <b>rs2</b> . | $\log(\#\text{steps})$ bits $\times 2$  |
| imm                                    | The step instruction’s immediate, appropriately sign or zero extended.                                             | W bits                                  |
| Values read at rs1,<br>rs2             | The actual values read from registers <b>rs1</b> , <b>rs2</b> .                                                    | W bits $\times 2$                       |
| Lookup result                          | The instruction’s output passed in as advice.                                                                      | W bits                                  |
| Extra advice                           | The non-deterministic advice element used only in division and remainder operations.                               | W bits                                  |
| Elements involved in<br>Lasso lookups. | Commitment to subtable outputs                                                                                     | 10 bits $\times 2c$                     |
|                                        | Commitment to the chunks $C[c]$ of the lookup query.                                                               | 22-bits $\times c$                      |
|                                        | Commitment to the step counter                                                                                     | $\log(\#\text{steps})$ bits $\times 2c$ |

Table 1: The basic elements involved in most instructions.

### A.1 List of Operation Flags employed:

1. This flag is 0 if the first operand is **rs1** and 1 if it is **rs2**.
2. This flag is 0 if the second operand is **rs2** and 1 if it is **imm**.
3. Is this a load instruction?
4. Is this a store instruction?
5. Is this a jump instruction?
6. Is this a branch instruction?
7. Does this instruction update **rd**?
8. Does this instruction involve adding the operands?

9. Does this instruction involve subtracting the operands?
10. Does this instruction involve multiplying the operands?
11. Does this instruction involve non-deterministic advice?
12. Does this instruction assert the lookup output to be false?
13. Does this instruction assert the lookup output to be true?
14. This flag is the sign bit of the immediate.

The memory flags are as follows: if the instruction is a load/store operation that reads/writes  $k$  bytes, then the memory flags will be of the form  $1^k \parallel 0^{W/8-k}$ .

### A.2 Supporting byte-addressable loads and stores

One challenge with implementing a zkVM for RISC-V is supporting byte-addressable memory. This requires performing up to  $W/8$  memory operations per load/store, one for each byte written or read. This section describes the memory-checking steps involved. Large lookup tables are not involved in these instructions.

**Stores.** Each store instruction reads the lower  $k$ -byte-suffix from `rs2` ( $k = 8, 4, 2, 1$  for instructions SD, SW, SH, SB, respectively) and writes the result into memory locations starting from  $\text{loc} = \text{rs1} + \text{imm}$  (see Section 5.8 for how this is calculated). There are two steps involved in stores:

1. The prover provides as advice the bytes-decomposition of the value in `rs2`. These values are range-checked and verified to be the correct decomposition.
2. Memory-checking can then write these bytes one by one to their memory locations (which starts at `rs1 + imm`). Offline memory-checking requires that the prover provide a timestamp of the latest write that occurred to that memory location. This timestamp must be range-checked and verified to be less than the current timestamp.

**Loads.** The load operations take  $k$  bytes of memory ( $k = 8, 4, 2, 1$  for instructions LD, LW, LH, LB, respectively) from location  $\text{loc} = \text{rs1} + \text{imm}$ , sign-extends it to  $W$  bits and then stores the result in `rd`.

1. The  $k$  bytes are first read using memory-checking. As with stores, range checks and less-than comparisons are performed on the timestamps provided. not required here, as they were enforced during the stores.
2. Jolt employs a small lookup table to get the sign-bit of the highest order byte. Sign-extension and concatenation can then be performed using constraints.

**Range-check costs.** The CPU circuit requires performing range-checks on inputs of the following bit-sizes: 8 bits (for the bytes in stores) and  $\log(\#\text{steps})$  bits (for the timestamps). Both these range checks can be performed using `Lasso` with parameter  $c = 1$ . This is a special case that requires the prover to commit to only one element of value bounded by the step counter.

Table 2 shows the overheads (on top of the basic non-lookup elements involved in all operations) for load and store operations per byte involved in the operation (up to 4 for RV32 and 8 for RV32). These elements are always 0 in other operations and hence only count towards the prover’s cost when performing loads and stores.

| Element           | Purpose                                                                                                                            | #Bits per byte                                                               |
|-------------------|------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------|
| memflags          | Unary vector indicating the byte positions that are active in the load or store.                                                   | 1 bit                                                                        |
| memory_v_byte     | The actual byte value read/stored                                                                                                  | 8 bits                                                                       |
| memory_timestamp  | The timestamps passed as advice for memory-checking.                                                                               | $\log(\#steps)$ bits                                                         |
| ts_range_check    | Element involved in the range checks of <code>memory_ts[k]</code> performed with <code>Lasso</code>                                | $\log(\#steps)$ bits                                                         |
| max_checks        | Elements involved in verifying that the read timestamp is less than the step counter.                                              | 3 bits $\times 2$<br>5 bits $\times 2$<br>( $\log \#steps$ ) bits $\times 4$ |
| byte_range_checks | <b>Stores only:</b> Element involved in the 8-bit-range checks of <code>memory_v_bytes[8]</code> performed with <code>Lasso</code> | ( $\log \#steps$ ) bits                                                      |

Table 2: The extra elements involved per-byte in loads and stores.

### A.3 Summary of CPU Step Constraints

Here, we go through the CPU steps outlined in Figure 1 and add more context in terms of the committed elements used and constraints involved.

1. Read the program code using `PC` to get the instructions details: `opflags[14]`, `opcode`, `rs1`, `rs2`, `imm`.
  - As discussed in Section 3.3, this involves six reads from program memory: one for each element of the tuple.
2. Read the source registers `rs1`, `rs2`. And then set operands  $x$  and  $y$ 
  - Reading the source registers involves two memory-checking updates. The locations are the registers themselves.
  - The values and timestamps involved are committed advice elements: (`read_val_rs1`, `read_ts_rs1`) and (`read_val_rs2`, `read_ts_rs2`), respectively.
  - The setting of operands  $x$  and  $y$  involves using `opflags[0]`, `opflags[1]` described earlier.
3. Perform loads and stores using memory-checking.
  - For both loads and stores, the memory location involved is `rs1 + imm`. This sum is calculated using constraints and also involves `opflag[14]` which holds the sign of `imm`.
  - This involves up to two range-checks and a less-than comparison per byte of memory read/written (see Appendix A.2).
  - As RISC-V memory is byte-addressable, the lookup argument involves  $W/8$  memory operations. The required bytes-decomposition are the elements in `memory_v_bytes[W/8]`.
4. Construct the lookup element.
  - This is structured as `opcode`  $\parallel$   $z$  where  $z$  could be  $x$   $\parallel$   $y$ ,  $x + y$ ,  $x - y$  or  $x * y$ .
  - The exact format chosen is guided by many `opflags`.
5. Updating the destination register.

- If the corresponding `opflag` is 1, the lookup result is stored in `rd`.
- The memory-checking write operation here involves the location `rd`, value `result` and timestamp being the current global step count.

6. Updating the PC.

- For Jump instructions,  $PC \leftarrow (\text{lookup } \mathbf{result})$ . For Branch instructions,  $PC \leftarrow PC + \mathbf{imm}$  (sum computed similar to Step 3 above) if and only if the lookup `result` was true. For other instructions,  $PC \leftarrow PC + 4$ . The right choice is guided by the corresponding `opcodes`.
