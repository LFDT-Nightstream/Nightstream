## 8 Qualitative Cost Estimation

In this section, we provide a qualitative evaluation of the prover cost involved in Jolt when using the Lasso lookup argument. As discussed in Section 1.3.1, the dominating cost is in producing commitments to the elements fed to the constraint system and lookup argument every step. Table 3 provides a upper bound on the elements committed per step grouped by their bit-lengths. We analyze bit-length as that is the main factor determining the commitment cost when using Pippenger’s multi-scalar multiplication algorithm.<sup>22</sup> We provide below a brief overview of the elements involved and leave a more detailed discussion to Appendix A.

**Elements involved in CPU Execution.** First, let’s look at the elements involved in satisfying the CPU step circuit’s constraints before looking at the elements needed for the Lasso argument. The smallest of these are the 1-bit ones constituting the bits of `opflags` [14] and `opcode` [8] and the 5-bit elements indexing the source (`rs1`, `rs2`) and destination (`rd`) registers read from the instruction. Slightly larger elements are the PC (which could be as large as  $\log|\text{program\_code}|$  bits) and the step counter. The latter starts small but grows over time: for example, it can reach 25 bits for a reasonably long execution of 30 million steps. Finally, the largest elements involved are the  $W$ -bit ones specifying the values stored in the two source registers, the sign-extended `imm` read from the program code, the lookup output, (which is generally stored in the destination register), and the advice element involved (only) in division and reminder operations. Note that most instructions do not involve all these elements (notably, the advice element is used rarely and the ones that do use advice never use `imm`) and thus, the numbers from Table 3 are a worst-case upper bound.

**Elements involved in Lasso** The cost of a lookup query with Lasso was discussed in Section 4.3. In brief, most lookups require the prover to commit to three groups of  $c$  elements each with the groups having bit-lengths  $W/c$ ,  $2W/c$  and  $\log(T)$ , respectively. The lookup costs figuring into Table 3 are for the worst-case scenario of lookups involving the less-than comparison table of Section 5.3 which commits to  $2c$  more elements than normal. Thus, most RISC-V instructions are actually slightly cheaper than the costs reported. In fact, many, like ADD and those related to it, require fewer than  $3c$  elements to be committed as their queries are smaller (only  $W + 1$  bits long).

As a quick note, using Generalized-Lasso for lookups would increase the bit complexity of  $c$  of the committed

<sup>22</sup>In Pippenger’s multi-scalar multiplication algorithm to commit to elements, committing to an  $N$ -bit element costs roughly  $\lceil N/22 \rceil$  group operations. This makes committing to a 32-bit element cost two group operations while a 256-bit element costs 12 group operations.

| Base Costs per Memory Instruction           |                    | Overhead per Byte |              |
|---------------------------------------------|--------------------|-------------------|--------------|
| Bit-length                                  | Number of Elements | for Loads         | for Stores   |
| 1                                           | 22                 | 1                 | 1            |
| [2, 8]                                      | 3                  | 1                 | 1            |
| $\log(T)/2$                                 | 3                  | 4                 | 4            |
| $\log(T)$                                   | 5                  | 6                 | 7            |
| $W$                                         | 3                  | -                 | -            |
| Total Elements                              | 36                 | 12                | 13           |
| In 256-bit equivalents<br>(both RV32, RV32) | 3.5 elements       | 1.5 elements      | 1.5 elements |

Figure 4: The spread of elements committed to per memory operation with the extra overhead elements per byte of load or store. That is, a load and store of  $k$  bytes involves the prover committing to  $36 + 12k$  elements, and  $36 + 13k$  elements, respectively. We approximate the per-step commitments costs in terms of the cost of committing to a 256-bit element when using Pippenger’s MSM algorithm, assuming that the program code is under  $2^{22}$  bytes long (placing it in the  $\log(T)$  category), and the program finishes in under  $2^{30}$  CPU steps.

field elements from  $2W/c$  to  $\log|\mathbb{F}|$ .

### 8.1 Cost of Memory Operations

We analyze the cost of load and store operations separately as these do not involve large lookups to perform the core instruction logic. Rather, the main cost here is performing memory-checking operations, one for each byte of memory involved in the load/store. This can be up to four for 32-bit processors and eight for 64-bit processors. The elements involved on top of the non-lookup elements of the non-memory instructions are the actual bytes read/written, the timestamps involved in memory-checking, and the cost of range-checks and computing max function (via small lookup tables) with the timestamps. Memory operations also commit to two fewer  $W$ -bit elements as they never involve the advice (used only in division and remainder operations) and lookup output elements.

Firstly, a minor cost involved in memory operations are extra 1-bit memory “flags” which act as a unary vector indicating the exact number of bytes read/stores. For both loads/stores, for each byte read/written, the prover commits to the actual byte value and provides a timestamp `ts_read` to be used in offline memory-checking (indicating when the byte was last written to). This must be range-checked and verified to be less than the current step counter. This range-check is very efficient in `Lasso` and requires committing to a single element of value bounded by the number of steps up to that point. Verifying that `ts_read` is less than the current step counter uses the less-than lookup table (of Section 5.3) and requires committing to four elements of bit-length bounded by that of the step counter and four more of half those many bits.

Stores, which are memory “writes”, require 8-bit range checks of the bytes written. These range-checks are again very efficient in `Lasso` and only involve committing to a single element of value at most the number of steps up to that point.
