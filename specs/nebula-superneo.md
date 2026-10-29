# Nebula memory phase on SuperNeo F′ — protocol specification

Status: **Approved for implementation.** The owner authorized Stage 2 on
2026-10-07.

Base revision: `nico/f-prime-constraints-cuda-formal` at `b2b23f50b`.

Security companion: [`nebula-superneo-security.md`](./nebula-superneo-security.md).
It holds the security goal, the assumptions, the lemmas, and the error bound.
This file holds the protocol only.

On acceptance, this pair replaces the August V2 draft set
(`docs/Nebula-on-Superneo/`) and the historical `specs/` files on
`nico/nebula-m0-frontend`.

The key words **MUST**, **MUST NOT**, and **MAY** state requirements.

## 1. Scope

This specification adds one Nebula memory phase to the Stage 1 F′ system, as
`FPRIME_LEAN_ARCHITECTURE_SPEC.md` §6 requires. The phase supplies:

- public ROM and RAM in one flat address space;
- fixed application memory ports;
- segmented offline memory checking with public-coin fingerprints;
- a memory carry in the application state;
- terminal memory acceptance and a final memory root.

An accepting proof attests one execution of the verifier-selected application
relation. The execution starts from the statement's initial application state
and the initial memory of the verifier-selected program. It ends in the
statement's final application state. Every memory access goes through a port
and is sequentially consistent. This specification uses *relation* for the
circuit that a package proves and *program* for the ROM and RAM images.

Out of scope: application semantics (for example WASM), stacks, private
initial memory, zero knowledge, more than one checked step per fresh claim, and
more than one fresh claim per fold.

### 1.1 Overview (informative)

This subsection explains how the phase joins the Stage 1 stack. The rules are
in §6–§13.

Three layers keep their own protocols:

- **SuperNeo** folds the F′ claims exactly as in Stage 1. The memory phase adds
  no commitment component, no claim, and no evaluation (§7).
- **HyperNova Construction 2** links each invocation to the next through the
  hash of the F′ state. The memory carry (§11) is part of the application
  state: the four Stage 1 application-state words are a digest of the
  application words and the carry. The F′ state layout does not change.
- **Nebula** checks memory with fingerprints of four multisets (§8). Each step
  adds its own records to three Poseidon2 chains (§9.2), as in the hash-chain
  commitment of Nebula §3.1. Each segment closes with the checks of Nebula's
  `F_final` (§11.2).

The layers meet at four joints:

| Joint | Mechanism | Spec |
|---|---|---|
| J1 | Each step packs its own records and adds them to three Poseidon2 chains. | §6.3, §9.1, §9.2, §11.2 |
| J2 | The memory rows open the authenticated application state to the carry, read `η`, `h`, `ts`, and `idx`, and write the new carry into the next state. | §8, §11.1 |
| J3 | The proposed roots and `D_mem` enter the `η` transcript. The close checks that the chains equal them. | §9.3, §11.2 |
| J4 | The Stage 1 terminal checks the last claim and its state link. The verifier opens the final state to the final carry and requires a closed carry. | §12, §13 |

One segment runs in four stages:

1. **Open.** The prover proposes the ops root and the FS root. The relation
   derives `η` from them, `D_mem`, `ts`, and `plan_digest`. To compute the
   roots, the prover must first run the whole segment natively.
2. **Step.** Each invocation runs the memory rows on its own assignment. The
   rows update the four products. The same invocation adds the step's records
   to the chains. SuperNeo then folds the invocation's claim, as in Stage 1.
3. **Close**, in the step that sets `idx = N`. The chains must equal the roots
   that entered `η`, and the product equation must hold. Then `D_mem` takes the
   FS root.
4. **Terminal.** The Stage 1 terminal checks the last claim. The verifier
   requires that the final carry is closed.

```mermaid
flowchart TB
  subgraph NEB["Nebula memory checking, inside each F′ step"]
    ROWS["memory rows on z_i (§8)"]
    CHAIN["three Poseidon2 chains over packed records (§9.2)"]
    ETA["eta transcript at open (§9.3)"]
    CLOSE["close when idx = N (§11.2)"]
  end
  subgraph HN["HyperNova Construction 2 state link"]
    CARRY["MemoryCarry in the application-state digest (§11.1)"]
  end
  subgraph SN["SuperNeo folding, unchanged from Stage 1"]
    FOLD["PiCCS, PiRLC, PiDEC on F′ claims"]
    TERM["Stage 1 terminal (§13)"]
  end
  ROWS -- "J1" --> CHAIN
  CHAIN --> CARRY
  ETA --> CARRY
  CARRY -- "J2" --> ROWS
  CARRY --> CLOSE
  CLOSE -. "J3" .-> ETA
  ROWS --> FOLD
  TERM -- "J4" --> CARRY
```

Security note §4 gives the argument for each joint and its status.

## 2. Sources and authority

The published papers are the source of truth:

- Nebula: §3 (§3.1 incremental commitments, §3.2, Constructions 1 and 2), §4.2,
  §4.3, Appendix A, in the published text pinned in
  `docs/nebula-paper-original.zip`. The corrected working copy in
  `docs/nebula-paper/` is not the authors' text and is not authority here.
- SuperNeo v1.2 (`docs/superneo-paper-v1_2/`): §5–§8 and Appendix B.
- HyperNova: Construction 2 (§6.3) and Lemma 4.
- Coral: reference material only. This specification requires no Coral
  construction.

`FPRIME_LEAN_ARCHITECTURE_SPEC.md` §2 lists the corrected copies
`docs/nebula-paper/` and `docs/hypernova-paper/`. This specification follows
the owner rule that the published text is the source of truth. It uses a
corrected copy only where that copy adds a premise: the HyperNova Lemma 4
premises (security note A4). It does not use the corrected `F_scan` timestamp
range checks, because security note Lemma 5 shows that they are not needed.
At the base revision, only `docs/superneo-paper-v1_2/` is in git. The other
paper paths are local working-tree copies.

This specification depends on these trunk documents and does not restate
them:

| Trunk document | What this specification uses |
|---|---|
| `FPRIME_LEAN_ARCHITECTURE_SPEC.md` §3, §4, §5, §6 | Stage 1 profile, authority model, lifecycle, Stage 2 phase rule |
| `decisions/padded-row-identity-piccs.md` | One-joint PiCCS; no column challenge, `s_col`, or `y_zcol` |
| `decisions/piccs-prior-state-digest.md` | PiCCS absorbs the prior-state digest, then the fresh statement |
| `decisions/fprime-stage1-main-ajtai-setup.md` | `κ = 22`, main seed, wide reduction |
| `decisions/fprime-ajtai-shake128-setup.md` | SHAKE128 key expander |
| `decisions/fprime-stage1-domain-2p28.md` | `2^28` joint row and carrier domain |
| `decisions/fprime-stage1-per-application-packages.md` | Verifier-owned application package |

If this specification and a trunk decision disagree, the trunk decision is
authority until this specification is amended.

Lean is the authority for logical builders, physical layout, and transcript
framing, as `FPRIME_LEAN_ARCHITECTURE_SPEC.md` §4 states. This specification
fixes the logical content and order. It does not fix column numbers.

## 3. Inherited base profile

| Symbol | Value | Source |
|---|---|---|
| `q` | `2^64 − 2^32 + 1` | Stage 1 profile |
| `F` | `GF(q)` | Stage 1 profile |
| `𝕂` | `F[U]/(U² − 7)`, `|𝕂| = q²` | `neo_math::K`; field facts in the security note A5 |
| `R_F` | `F[X]/(X^54 + X^27 + 1)`, `d = 54` | Stage 1 profile |
| `b`, `k_ρ`, `B` | `2`, `16`, `2^16` | `CLAUDE.md` policy; architecture spec §3 |
| fold arity | 1 fresh + 16 running = 17 PiRLC inputs | architecture spec §3 |
| PiDEC children | 16 | architecture spec §3 |
| CCS matrices | 4 | architecture spec §3 |
| `κ` | 22; one main commitment is `κ·d = 1,188` field words | main Ajtai setup decision |
| key expander | SHAKE128, `nightstream-ajtai-shake128-wide256-v1` | SHAKE128 decision |
| row domain | `2^28` | domain decision |
| `H` | Poseidon2, width 16, rate 12, capacity 4, digest 4 | `neo-params::poseidon2_goldilocks` |

A digest is four canonical field elements. Every digest and challenge in this
document uses `H` through four transcript primitives. The Stage 1 fold
transcript does not use them, so this document defines them:

- `reset_v1_1()`: the zero state.
- `absorb_block_v1_1(tr, block)`: absorb the block's length, then its words,
  in rate chunks, with one permutation after each chunk.
- `squeeze_digest_v1_1(tr)`: the first four lanes.
- `squeeze_extension_v1_1(tr)`: read lane 0 and permute, two times. The two
  words are one element of 𝕂.

`α` and `γ` always mean the PiCCS challenges, and `ρ` always means the PiRLC
challenges. The memory challenges are `η`.

## 4. Verifier-owned memory plan

### 4.1 Plan fields

A `NebulaPlan` is a geometry, every field except the two images, and a
program, the two images. The verifier key binds the geometry:

| Field | Meaning |
|---|---|
| `r`, `μ` | ROM has `R = 2^r` cells; RAM has `M = 2^μ` cells |
| `W_ts` | timestamp width in bits |
| `B_ops` | operation slots per step |
| `B_scan` | scan slots per step |
| `N` | steps per segment |
| `S_max` | maximum segments per proof |
| `rom_image` | `R` words of 32 bits |
| `ram_image` | `M` words of 32 bits |

A package is the relation of one application at one geometry, and a program
is the two images. The relation reads no image, so one package and one setup
serve every program of its geometry: setups are one per (application,
geometry). The key of a program is its `D_init` (§9.2), which the verifier
computes from the images. A cell value is one 32-bit word.
The verifier accepts at most `S_max · N` steps. This is the fixed iteration
bound of HyperNova Lemma 4 for this verifier key. Security note A4 states how the bound covers an adversary whose
outputs can be longer.

### 4.2 Validity rules

Setup MUST reject a plan that breaks any of these rules:

1. **Exact cover:** `N · B_scan = R + M`.
2. **Timestamp range:** `S_max · N · B_ops < 2^W_ts`.
3. **Field encoding:** `W_ts ≤ 62`. Then every timestamp and value is an
   integer below `q`, its field encoding is injective, and row O4 (§8.1)
   cannot wrap modulo `q`. Each lane has fewer than `q` bits:
   `B_ops · (3 + μ + 64 + W_ts) < q` and `B_scan · (32 + W_ts) < q`. Then the
   length prefix of each packed block (§9.1) is one canonical field element.
4. **Address width:** `r ≤ μ` and `R + M < q`. The address field has `μ` bits,
   and every global index is below `R + M`. Only ROM addresses need a range
   gate (§8.1, O6).
5. **Positive sizes:** `B_ops, B_scan, N, S_max ≥ 1`.
6. **Security:** the plan passes the setup check in the security note §5.

In Lean, every memory package that `lake exe emit` writes gets its relation
from `NebulaMemoryPackage.application`. That function takes the proofs of
rules 1–5 (`Geometry.Valid`) and rule 6 (`Geometry.Secure`), so `lake exe emit`
cannot write a package for a geometry that breaks a rule. The relation
`MemoryApp.program` takes a `Geometry`, which has no image field, so two plans
with one geometry have one relation (`MemoryApp.program_eq_of_geometry`).

**Program images.** The verifier MUST reject a program unless its ROM image
has exactly `R` words and its RAM image exactly `M` words, each below `2^32`.
It MUST make this check before it computes `D_init` (§9.2). A scan slot holds
32 value bits (§6.2), so a word `v ≥ 2^32` would get the `D_init` of a smaller
word while the machine reads `v`. `Plan.Valid` states the same rules in Lean.
The Rust `nebula::Program` holds `u32` words, and `Program::new` rejects other
image sizes.

The segment range of `open` (§11.2) decomposes `S_max − 1 − seg_idx` into
`W_seg` bits:

```text
W_seg = ⌈log2(S_max + 1)⌉    holds every value in [0, S_max]
```

An implementation MUST derive `W_seg` from the geometry. It MUST NOT use a fixed
width. The relation needs no width for `idx` or `cnt`. The arm and close tests
(§12) keep `idx` in `[0, N]`, and `cnt` is a field sum of at most `B_ops < q`
terms.

### 4.3 Plan digest

```text
plan_digest = Digest("Nightstream/Nebula/v3/plan",
                     [r, μ, W_ts, B_ops, B_scan, N, S_max])
```

Each integer enters as one canonical field element. The images do not enter
`plan_digest` (Lean `planDigest` takes a `Geometry`), so the package rows do
not depend on the program. The verifier computes `plan_digest`. A
prover-supplied value is invalid.

The verifier binds `plan_digest` and `D_init` (§9.2) without a separate
verifier-context field. `plan_digest` binds the geometry, and `D_init` binds
the program. `plan_digest` is a package constant: the package rows fix it, as
they fix the Stage 1 verifier-context digest, and the relation MUST read it as
a constant (the headers and the §9.3 transcript). A witness copy of
`plan_digest` is invalid. The verifier computes `D_init` and puts it in the
start carry (§11.1). The relation reads `D_mem` from its input carry, which
the state digest binds to the input state. So step 0 reads `D_init` because
the verifier computes the initial state from the start carry (§13). A
different initial memory opens a different initial state, which the Stage 1
state link rejects, except with a state-digest collision.

## 5. Memory model

The global index of a cell is:

```text
g(ROM, a) = a,        0 ≤ a < R
g(RAM, a) = R + a,    0 ≤ a < M
```

Each cell has a state `(v, t)`: a value and the timestamp of its last access.
At chain start, ROM cells hold `rom_image`, RAM cells hold `ram_image`, and
every timestamp is zero.

`ts` is the global timestamp: the number of active operations so far. It never
resets. Each active operation on a cell with old state `(v, rt)` does this:

```text
read(space, a):          RS ← (rt, g, v)      WS ← (ts+1, g, v)      cell ← (v, ts+1)
write(RAM, a, v_new):    RS ← (rt, g, v)      WS ← (ts+1, g, v_new)  cell ← (v_new, ts+1)
```

A write to ROM is invalid. At segment open, `IS` holds one tuple `(t, g, v)`
for every cell. At segment close, `FS` holds one tuple for every cell. For an
honest segment, `IS ∪ WS = RS ∪ FS` holds as multisets.

## 6. Records and lanes

### 6.1 Operation slot

An operation slot has these fields, in this order:

| Field | Bits | Meaning |
|---|---|---|
| `pad` | 1 | `1` = inactive slot |
| `is_write` | 1 | `0` = read, `1` = write |
| `is_ram` | 1 | `0` = ROM, `1` = RAM |
| `addr` | `μ` | address inside the namespace |
| `v_r` | 32 | value read (for a write: the old value) |
| `v_w` | 32 | value written back (for a read: equal to `v_r`) |
| `rt` | `W_ts` | timestamp of the previous access to the cell |

An inactive slot has every other field equal to zero. Inactive slots MAY
appear at any position, because application ports have fixed positions (§10).
Active slots keep their physical position.

For slot `j` of a step with incoming timestamp `ts`:

```text
cnt_j = Σ_{i ≤ j} (1 − pad_i)                (active count in slot order)
wt_j  = ts + cnt_j                           (write timestamp)
g_j   = addr_j + is_ram_j · R
RS_j  = (rt_j, g_j, v_r_j)                   for an active slot
WS_j  = (wt_j, g_j, v_w_j)                   for an active slot
```

### 6.2 Scan slot

A scan slot has `(v: 32 bits, t: W_ts bits)`. Each step has an initial-snapshot
(IS) chunk and a final-snapshot (FS) chunk of `B_scan` slots each. Slot `j` of
step `idx` describes the cell `g = idx · B_scan + j`. This index is structural.
The relation computes it from `idx` and the slot constant. Exact cover gives
one IS tuple and one FS tuple for every cell in every segment. There are no
scan pads.

### 6.3 Lanes

Three fixed column ranges of the step assignment `z` hold the records:

```text
ops lane: the B_ops operation slots
IS lane:  the B_scan IS slots
FS lane:  the B_scan FS slots
```

Encoding: slot-major, then field order, then little-endian bits. One
coordinate holds one bit. Rows O1 and S1 (§8) make every lane coordinate a bit.
The plan fixes the length of each lane:

```text
len_ops = B_ops · (3 + μ + 64 + W_ts)
len_is  = len_fs = B_scan · (32 + W_ts)
```

The IS and FS lanes MUST have the same width and the same encoding. Segment
continuity (§9.2) depends on this.

### 6.4 Hash residency

A fingerprint factor (§8.2) has tuple content `(t, g, v)` and, for an operation
slot, a pad gate. The relation MUST derive the tuple content and the gate only
from:

1. lane coordinates of the same step, which the step hashes (§9.2);
2. the carry fields `ts` and `idx` that the step reads (§8), and structural
   constants;
3. the counter values `cnt_j`, which the counter rows (O2) fix as functions of
   lane coordinates.

The challenges `η` and the product values MUST appear only as fingerprint
coefficients and accumulators. A tuple input that the step does not hash, or
that depends on `η`, lets a prover choose memory content after the challenge.

The memory rows (§8) are the same for every application, and they obey this
rule. The rows of an application read only its state words, its private words,
and the operation slots (§10). They have no variable for `η`, for a product,
or for a carry word. Thus every memory application obeys this rule, and no
static check is necessary. Lean proves the memory soundness for every
application (security note Ob4, Ob7, and Lemma 6).

## 7. Interface with Stage 1

The memory phase is part of the application relation. A memory application is
one Stage 1 application whose circuit contains the memory rows (§8), the record
chains (§9.2), the carry actions (§11.2), and the application's own step,
joined through the ports (§10). The memory phase adds no commitment component,
no claim, no evaluation, and no field to the F′ state. SuperNeo runs with the
Stage 1 commitment map. PiCCS, PiRLC, PiDEC, the rules of
`decisions/piccs-prior-state-digest.md`, the state preimage, and the Stage 1
terminal openings do not change.

The four Stage 1 application-state words are the state digest of §11.1. So the
F′ relation, its layout, and its composition, fixed-point, and domain theorems
are the Stage 1 ones for the memory application's package. The application
interface has one addition: a validity predicate. The memory rows and the
application's rows (§10) reject an invalid step, so the circuit holds exactly
for valid steps.

A Nightstream F′ package without a memory application is a Stage 1 package and
does not change.

## 8. Per-step memory relation

Each invocation runs one step (§12). The step evaluates these rows on the
invocation's own assignment `z`. The rows read the carry fields `η`, `h`,
`ts`, and `idx` after the invocation's `open` action, if any (§11.2). The rows
write the new `h`, `ts`, and `idx` into the output carry. The same step adds
its records to the chains (§9.2).

### 8.1 Operation rows

For each operation slot `j`:

| # | Constraint |
|---|---|
| O1 | every lane coordinate and auxiliary bit is in `{0, 1}` |
| O2 | `cnt_j = cnt_{j−1} + (1 − pad_j)`, with `cnt_{−1} = 0`, as a field sum (at most `B_ops < q`) |
| O3 | `(1 − is_write_j) · (v_w_j − v_r_j) = 0` |
| O4 | `(1 − pad_j) · (wt_j − rt_j − 1 − diff_j) = 0`, where `diff_j` is a `W_ts`-bit auxiliary word |
| O5 | `is_write_j · (1 − is_ram_j) = 0` |
| O6 | `(1 − is_ram_j) · addr_bit_{j,k} = 0` for each `k ∈ [r, μ)` |
| O7 | `pad_j · w = 0` for each word `w ∈ {is_write, is_ram, addr, v_r, v_w, rt}` |
| O8 | `h_rs_j = h_rs_{j−1} · (pad_j + (1 − pad_j) · f_η(RS_j))` |
| O9 | `h_ws_j = h_ws_{j−1} · (pad_j + (1 − pad_j) · f_η(WS_j))` |

O4 is an integer relation because `diff_j` is range-checked and §4.2 rule 3
keeps every term below `2^62`. It gives `rt_j < wt_j`. Each chain `h_*_{−1}`
starts from the carry value `h`.

### 8.2 Fingerprint

```text
f_η(t, g, v) = g + η1 · v + η1² · t − η2      in 𝕂
```

This is the `Hash` function of Nebula §4.3, with the address `a` replaced by
the global index `g`. The challenges are in `𝕂`, not in `F` (§15).

### 8.3 Scan rows

For each scan slot `j` with `g_p = idx · B_scan + j`:

| # | Constraint |
|---|---|
| S1 | every IS and FS coordinate and auxiliary bit is in `{0, 1}` |
| S2 | `h_is_j = h_is_{j−1} · f_η(t^IS_j, g_p, v^IS_j)` |
| S3 | `h_fs_j = h_fs_{j−1} · f_η(t^FS_j, g_p, v^FS_j)` |

### 8.4 Boundary rows

```text
out.ts  = ts + cnt_{B_ops−1},     out.ts < 2^W_ts
out.h   = the last value of each product chain
out.idx = idx + 1
```

## 9. Record chains and memory challenges

### 9.1 Digest primitive and packing

```text
Digest(tag, block_1, …, block_n):
  tr ← reset_v1_1()
  absorb_block_v1_1(tr, tag as field words, one per ASCII byte)
  for each block: absorb_block_v1_1(tr, block)
  return squeeze_digest_v1_1(tr)
```

`pack_l(z)` turns lane `l` of the step assignment `z` into field elements. It
splits the lane's bit string (§6.3) into consecutive chunks of 63 bits. Only
the last chunk can be shorter. A chunk with bits `b_0 … b_62` becomes the field
element `Σ_k 2^k · b_k`. So every packed element is an integer below
`2^63 < q`, and `pack_l` is injective on bit strings of the plan's length
(security note Lemma 1).

A packing that puts more than 63 bits into one element is invalid. For
example, `v · 2^32 + t` with 32-bit `v` and `t` can exceed `q`, so two
different tuples can give the same field element.

### 9.2 Chains

```text
chain_ops(j, D, P) = Digest("Nightstream/Nebula/v3/chain-ops", [j, D], P)
chain_mem(j, D, P) = Digest("Nightstream/Nebula/v3/chain-mem", [j, D], P)
header_ops         = Digest("Nightstream/Nebula/v3/header-ops", [plan_digest])
header_mem         = Digest("Nightstream/Nebula/v3/header-mem", [plan_digest])
```

A lane chain over the steps `0 … N−1` of one segment is

```text
D[0] = header_l,   D[j+1] = chain_l(j, D[j], pack_l(z_j)),   root = D[N]
```

where `z_j` is the assignment of step `j` of the segment, with `l = ops` for
the ops lane and `l = mem` for both the IS and the FS lane. This is the hash
chain `C_{i+1} ← H(C_i, ω_i)` of Nebula §3.1, with a domain tag, the step
index, and a header added. The F′ relation computes it, as in Nebula §3.2. The
IS and FS chains MUST use the same formulas. Then the FS root of segment `k`
equals the IS root of segment `k + 1` when the snapshots are equal.

`D_init` is the IS root over the initial snapshot: for each step `j`, the IS
slots of the cells `j · B_scan … (j+1) · B_scan − 1` with the image values and
`t = 0`. The verifier computes `D_init` from the program's images and binds it
through the start carry (§4.3). `D_init` is the only path by which the program
enters a proof.

### 9.3 Challenge transcript

At segment open, the prover proposes the ops root `D_pre.ops` and the FS root
`D_pre.fs` of the segment. The IS root of the segment MUST be `D_mem`: the FS
root of the previous segment, or `D_init`. The relation derives the
challenges. It evaluates this transcript in every invocation, and only an
opening invocation keeps the result:

```text
tr ← reset_v1_1()
absorb_block_v1_1(tr, "Nightstream/Nebula/v3/eta" as field words)
absorb_block_v1_1(tr, [plan_digest])
absorb_block_v1_1(tr, [ts])
absorb_block_v1_1(tr, [D_pre.ops, D_mem, D_pre.fs])
η1 ← squeeze_extension_v1_1(tr)
η2 ← squeeze_extension_v1_1(tr)
```

These inputs fix every tuple of the segment's four multisets. The three roots
fix the records (security note Lemmas 1 and 2). `ts` fixes the write
timestamps.
`plan_digest` fixes the tuple layout. `D_mem` MUST be absorbed: without it, a
prover could learn `η` first and then choose the previous segment's final
snapshot. The proposals get authority only from the close check (§11.2).

## 10. Application memory ports

The application relation and the operation lane share fixed physical port
positions. For each port, the application relation MUST constrain the active
flag, the read or write mode, the namespace, the address, the value returned to
the application, and the value that a write requests. The operation slot at
the same position MUST equal those values.

Every semantic memory access of the application MUST use one port. No other
memory path is allowed. An inactive port maps to a canonical pad slot. The
verifier key binds the port table.

**Idle step.** A segment has exactly `N` steps, because exact cover needs all
of them. The application relation MUST accept an idle step: every port is
inactive and the application state does not change. An idle step MAY occur at
any step. The prover uses idle steps to fill the last segment. The iteration
count `T` includes them.

**Application interface.** An application supplies its state words (a fixed
count), its private witness words (a fixed count), and its rows. Its rows read
only the input state words, the output state words, the private words, and the
words of each operation slot (its lane fields and its O4 difference word). The
memory rows and the carry are the same for every application. The application
MUST prove two facts about its rows (Lean `PortApp`):

1. Soundness (security note Ob7): when its rows hold and each operation slot
   obeys rows O1, O3, and O7, the decoded ports are one application step from
   the input state words to the output state words. The application state is
   its state words. An application with a richer state decodes it inside its
   step relation.
2. Completeness: for each application step, and for operation slots that obey
   rows O1, O3, and O7 and decode to the step's ports, there are private words
   that satisfy its rows.

The witness words of one step are the memory words, then the input state
words, the output state words, and the private words. The first test
application (Lean `MachineApp`) has two state words `(pc, acc)` and one
private word, the idle flag.

### 10.1 Memory events of catalog applications

A catalog application describes its memory in the form of the Rocq proof of
the WASM relation (`formal/wasm-rocq/MemoryTrace.v` on `main`). A catalog is a
list of memories. Each memory has an id, a kind (RAM or ROM), and ports. A
port has address columns, a value column, a kind (a read, or a write with an
optional prior column), and an activation. Each active port of a row gives one
event. The events of a trace come in this order: rows, then catalog memories,
then ports. The Rocq proof assumes that the events replay from the initial
state (`MemoryConsistent`). Lean copies the model in
`Spec/Nebula/MemoryTrace.lean`.

A binding maps the events to port accesses (Lean `MemoryTrace.Binding` in
`Spec/Nebula/EventBinding.lean`). These are its rules:

1. A bound memory has a region: a base address and one bit width per address
   part. A memory that the catalog declares RAM is in the RAM namespace. Every
   other memory is in the ROM namespace. An address packs little-endian, as
   `base + x₀ + 2^w₀ · (x₁ + 2^w₁ · …)`.
2. An address is in range when its memory is bound, it has one part per
   width, and each part is in `[0, 2^w)`.
3. Two different memories in one namespace have disjoint regions
   (`Binding.Disjoint`).
4. A read event is a read access. Its read value is the event value.
5. A write event is a write access. Its written value is the event value. When
   the event has a prior value, the read value is that value. When it has none,
   the read value is the current value of the cell, and Nebula checks it.
6. An inactive port has no access, so its slot is a pad slot. The active slots
   of all steps, in order, bind the events of the trace in order.

The free application (Lean `freeApp`) has a selected number of state words, no
private words, and no rows. Its model accepts every step of `B_ops` ports, so
the memory rows are its only checks. It is a `PortApp`, so each result for a
`PortApp` applies to it.

**Theorem** (`MemoryTrace.memoryConsistent_of_executes`). Let the binding obey
rule 3. Let every event of the trace be in range. Let the initial state hold
the program images on the in-range set (`Binding.Initial`): an address of a
RAM memory starts at the RAM image word of its packed address, and an address
of a ROM memory starts at the ROM image word. If an execution from the program
images has active ports that bind the events of the trace in order, then the
trace is `MemoryConsistent`. `memoryConsistent_of_accessAll` gives the same
result for each list of accesses that replays from the program images, and
`executes_accessAll` gives that replay for each execution.

The rule that each region is inside its namespace is for completeness only.
Soundness does not need it, because the access rule rejects an address outside
its namespace. For the same reason, an event value at or above `2^32` has no
valid access. A catalog with larger values needs to split them into 32-bit
memories.

**Rust backend.** `nebula::CatalogBinding` takes a catalog, one region for
each checked memory, and a slot plan. The plan gives, for each operation
slot, the checked ports that share the slot, by their rank: the position of
the port among the ports of the checked memories, in catalog order. A memory
without a region is host-checked. For a port `p`, `a_p` is its activation as
an expression: `1` for `Always`, `c` for `When(c)`, and `1 − c` for
`Unless(c)`. The backend MUST obey these rules. Rules I1 to I4 are rows of
the application relation, and the backend checks the other conditions when it
builds the binding:

- **I1. Slot.** A slot is active exactly when one of its ports is active, and
  then it shows that port: its namespace, mode, address, and values. The
  rows are `1 − pad_j = Σ a_p`, `is_write_j = Σ a_p` over the write ports,
  and `is_ram_j = Σ a_p` over the RAM ports, for the ports `p` of slot `j`,
  and `c · (c − 1) = 0` for each activation column `c`. So a port cannot be
  active while its slot is inactive or shows another port.
- **I2. Order.** The active slots, in slot order, list the active ports in
  catalog order, which is the event order of `eventsForRow`. For each port
  `p` that has ports `q` of a higher rank in earlier slots, the row is
  `a_p · Σ a_q = 0`.
- **I3. Address.** Each address part of an active port fits its width, and
  the slot address is the region base plus the little-endian parts. The base
  of a region MUST be a multiple of the region size `2^Σw`. Then the rows
  read each part from its run of the slot's address bits, and the bits above
  the region from the base, so a part needs no other range check. This is
  stricter than Lean `Region`, which allows any base, and the theorem applies
  to an aligned region. At build, the regions of one namespace MUST be
  disjoint and inside the namespace, and each port MUST have one address
  column for each width.
- **I4. Values.** For a read port, `v_r = value` (and `v_w = v_r` by O3). For
  a write port, `v_w = value`, and `v_r = prior` when the prior column is
  named.
- **I5. Host-checked memories.** A memory without a region has no slot and no
  row. The binding names it as host-checked.
- **I6. Plan.** The slot plan changes only completeness and cost, never
  soundness. A plan that puts two co-active ports in one slot, or co-active
  ports out of catalog order, makes the rows of that honest row fail. At
  build, the plan MUST have `B_ops` slots and name each checked port exactly
  once.

The rows of I2, I3, and I4 for port `p` have the form `a_p · e = 0`, so an
inactive port has no constraint. Each row is one rank-one constraint, and the
rows add no witness word. For each slot there are three rows. For each port
there are one row for each address part, one for the bits above the region
(when the region is smaller than its namespace), one for the value, one for
the prior column (when it is named), and one for the order (when I2 needs
it). Each activation column adds one row. When these rows hold, the active
slots of each step bind the events of the step in order, and each event is in
range. Then the theorem above applies to the catalog of the checked memories.
The rows are Rust only: no Lean proof covers them yet (security note §7).

## 11. Memory carry

### 11.1 MemoryCarry

The application state holds one memory carry:

```text
MemoryCarry {
  seg_idx:  integer in [0, S_max]   // segments closed so far
  idx:      integer in [0, N]       // steps run in the current segment
  ts:       W_ts bits        // global timestamp; never resets
  η1, η2:   𝕂                // challenges of the current segment
  h[4]:     𝕂                // running products (rs, ws, is, fs)
  D_pre:    2 digests        // proposed ops and FS roots of the current segment
  D_seen:   3 digests        // chains over the current segment's records (ops, is, fs)
  D_mem:    digest           // FS root of the last closed segment, or D_init
}
```

Between invocations, `1 ≤ idx ≤ N`. The value `idx = N` means that the segment
is closed: the step that set `idx = N` also ran `close`. Otherwise the segment
is open, and `D_seen` covers its first `idx` steps.

The carry words are the fields in the order above, each integer as one field
element, each `𝕂` element as `(c0, c1)`, and each digest as four field
elements: 39 words. The four Stage 1 application-state words are

```text
state_digest = Digest("Nightstream/Nebula/v3/state", app_words, carry_words)
```

where `app_words` are the words of the application's own state. Each step
receives the preimage of its input state as witness, and the relation checks
that its digest is the input state. The relation computes the output state as
the digest of the new application words and the new carry. There is no other
carry digest.

Chain start: the start carry is `seg_idx = 0`, `idx = N`, `ts = 0`,
`η1 = η2 = 0`, `h` all `1_𝕂`, `D_pre = (D_init, D_init)`,
`D_seen = (D_init, D_init, D_init)`, and `D_mem = D_init`, with `D_init` as
the verifier computes it (§4.3, §9.2). It is a closed carry, so the first invocation
reopens it (§12). The statement's initial state is the state digest of the
initial application words and the start carry.

### 11.2 Actions

```text
open(c, D_pre.ops, D_pre.fs):   // proposals from the prover
  require c.seg_idx < S_max
  c.D_pre  ← the proposals
  c.η      ← the §9.3 transcript
  c.h      ← all 1_𝕂
  c.D_seen ← (header_ops, header_mem, header_mem)
  c.idx    ← 0

step(c):                        // the §8 rows on this invocation's assignment z
  c.D_seen.ops ← chain_ops(c.idx, c.D_seen.ops, pack_ops(z))
  c.D_seen.is  ← chain_mem(c.idx, c.D_seen.is,  pack_is(z))
  c.D_seen.fs  ← chain_mem(c.idx, c.D_seen.fs,  pack_fs(z))
  c.ts, c.h    ← the §8.4 outputs
  c.idx        ← c.idx + 1

close(c):                       // exactly when step has set c.idx = N
  require c.D_seen.ops = c.D_pre.ops
  require c.D_seen.is  = c.D_mem                         (memory continuity)
  require c.D_seen.fs  = c.D_pre.fs
  require c.h.is · c.h.ws = c.h.rs · c.h.fs
  c.D_mem   ← c.D_pre.fs
  c.seg_idx ← c.seg_idx + 1
```

`step` MUST hash the lane coordinates of the invocation's own assignment, the
same coordinates that the memory rows read. The relation MUST run `close`
exactly when `step` sets `idx = N`. The product equation is a close condition
only. Interior steps carry the products forward.

## 12. Lifecycle

This is HyperNova Construction 2. Each fold has exactly one fresh claim. An
implementation MUST reject any other fold arity for a Nebula verifier key.

Let `S` be the public segment count, `1 ≤ S ≤ S_max`, and `T = S · N`. The
proof has `T` fresh claims `u_0 … u_{T−1}` and `T` invocations
`A[0] … A[T−1]`. Invocation `A[i]` runs step `i` and produces `u_i`. Each
invocation selects exactly one memory arm from its authenticated input carry
`c`:

| Arm | Selected when | Actions |
|---|---|---|
| continue | `c.idx ≠ N` | `step`; `close` if `idx = N` |
| reopen | `c.idx = N` | `open`; `step`; `close` if `idx = N` |

Between invocations `1 ≤ idx ≤ N` (§11.1), so `c.idx ≠ N` means `c.idx < N`.

Invocation `A[0]` is the Stage 1 base branch (no fold). Its input carry is the
start carry of §11.1, which is closed, so it reopens. Every later invocation
folds `u_{i−1}` as in Stage 1. The prover cannot choose the arm, because
`c.idx` comes from the authenticated input state. "`close` if `idx = N`" is not
optional: §11.2 requires it.

**Terminal.** The verifier keeps the Stage 1 terminal check
(`crates/nightstream/src/lifecycle/verify.rs`). It does no extra fold. It opens
the 16 running children and `u_{T−1}` directly and checks the state link of
`u_{T−1}`. The prover gives the final carry to the verifier with the proof; it
is a verifier input, not part of the Stage 1 proof bytes. The verifier computes
the final Stage 1 state as the state digest of the statement's final
application words and that carry. Only the state link authenticates the carry
fields that are not statement fields. The final carry is the one that
`A[T−1]` wrote after its own `step` and `close`. The verifier MUST then
require `idx = N` in the final carry. A proof that ends inside a segment fails
here.

## 13. Public statement and terminal acceptance

The public statement is the iteration count `T`, the initial and final
application words, and three memory fields:

```text
segment_count       S
final_timestamp     ts of the final carry
final_memory_root   D_mem of the final carry
```

The verifier key binds the geometry, and the verifier takes the program. It
checks the program images (§4.2) and computes `D_init` before the terminal
check.

For a Nebula verifier key, the verifier MUST reject the Stage 1 initial
envelope (`T = 0`). A proof has at least one segment.

The terminal check MUST:

1. compute the initial state as the state digest of the initial application
   words and the start carry, and the final state as the state digest of the
   final application words and the prover's final carry; then run the
   Stage 1 terminal check with these states;
2. require `idx = N` in the final carry;
3. require `1 ≤ S ≤ S_max`, `T = S · N`, and `seg_idx = S`;
4. require every statement field to equal the value in the final state.

Before these checks, the verifier MUST decode the statement canonically
(security note Ob1): it MUST reject an initial or final application word, or a
`final_timestamp`, that is not below `q`. Such a value has the field word of a
smaller one. Each other statement field and carry counter must equal a value
below `q`. This is a decoding rule, not a relation check: the relation has
field words, and a field word has no alias.

`final_memory_root` lets another proof refer to the final memory.

## 14. Rejection and conformance

The native verifier, the recursive relation, and the terminal check MUST
reject the same typed violations. A native check without a matching circuit
equation is a conformance defect.

An implementation conforms only when its tests run through the authoritative
verifier and cover at least these cases:

| Case | Required result |
|---|---|
| Segments with `N ≥ 2`, at production `κ` and production parameters | accept, with continue arms |
| `N = 1`: every invocation opens and closes a segment | accept |
| `S = S_max`, with every counter at its largest value | accept |
| A last segment filled with idle steps | accept |
| Two segments with RAM written in the first and read in the second | accept |
| Fresh memory at a segment boundary | reject at `D_seen.is = D_mem` |
| Segment-0 IS records that differ from the program images | reject at `D_seen.is = D_mem` |
| A second program of the package's geometry, verified with that program | accept |
| A proof of one program, verified with another program of the same package | reject at the Stage 1 state link |
| A ROM image without `R` words, a RAM image without `M` words, or an image word at or above `2^32` | reject before `D_init` (§4.2); a Rust `u32` word cannot be at or above `2^32` |
| An input carry or application words that do not open the input state | reject at the state digest check (§11.1) |
| A final carry that does not open the final state | reject at the Stage 1 state link |
| An initial or final application word, or a `final_timestamp`, at or above `q` | reject at canonical decoding (§13) |
| Read value differs from the last write | reject at the product equation |
| `rt ≥ wt`; write to ROM; ROM address out of range; nonzero pad | reject at O4; O5; O6; O7 |
| Change one record of a step after its segment opened | reject at a close equality |
| Swap the IS and FS records of one step, when they differ | reject at a close equality |
| A lane coordinate equal to `−1` | reject at O1 or S1 |
| A packing with more than 63 bits in one element | cannot be expressed: the package fixes the packing (§9.1; security note Ob2) |
| End inside a segment | reject at the terminal `idx = N` check |
| The Stage 1 initial envelope (`T = 0`) with any memory fields | reject (§13) |
| A fold with more than one fresh claim | reject |

Each rejection test MUST fail at the named check, not at a host replay.

## 15. Deviations from the papers

| Deviation | Paper | Reason |
|---|---|---|
| One F′ chain. At segment open, the prover proposes the ops and FS roots. The challenges come from these roots, `D_mem`, `plan_digest`, and `ts`. `ts` is absorbed because the write timestamps are computed (`ts + cnt`), not committed. The ops and scan products run in the same invocations as the application steps. The close, in the step that completes the segment, checks the proposals. | Nebula §4.3: Layer 1 proves `F`, then hashes the four carried commitments into `γ`, then proves `F_ops` and `F_scan`. Layer 2 folds the three proofs with `F_final`. | Trunk §6 requires one memory phase that reuses the Stage 1 core. One chain cannot make a second pass over the steps. Security note Lemma 2 binds the records to the proposed roots, and premise A6 gives the commit-before-challenge order; Lemma 3 motivates the value of A6. The close repeats the `F_final` checks: continuity, challenge binding, product equation, and memory handoff. |
| Fingerprint challenges `(η1, η2)` in `𝕂` | Nebula §4.2–§4.3: one pair `(γ1, γ2)` in `F` | `|F| = 2^64` is too small: the example geometry gives 45.91 bits over `F` and 109.91 bits over `𝕂` (security note §5). |
| The incremental commitment is the §3.1 hash chain over packed records, computed inside F′ by the generic construction at the start of §3.2 | Nebula Constructions 1 and 2 chain commitments to a split-committed witness. §3.2 notes that the generic construction costs in-circuit work linear in the size of the carried data `ω`. | Here `ω` is only the memory records, and this design pays that cost: about 360–370 Poseidon2 permutations per fold at the example geometry. SuperNeo claims and the terminal openings stay unchanged. Separate lane commitments at lane rank 2 would cost about 955 permutations, because all 16 running children would carry them in both state hashes. |
| Fixed segment length `N`, with idle steps (§10) | Nebula §4.3: a segment has as many steps as the proof runs, and the scan has that many chunks | One uniform F′ circuit needs a fixed scan width `B_scan = (R + M)/N`. Idle steps let an execution of any length fill the last segment. |
| ROM and RAM in one flat address space, with O5 (no ROM write) and O6 (ROM address range) | Nebula §4.2: one memory | The scope includes public ROM. The checker runs unchanged over all cells. O5 and O6 only restrict which operations are valid. |
| The carry is in the application-state digest (§11.1) | Nebula Construction 2 hashes the carried commitment `C` as its own field of the state hash | The Stage 1 F′ relation, state preimage, layout, and fixed-point and domain theorems stay unchanged. The digest uses the same Poseidon2 collision assumption (security note A3). The owner chose this design on 2026-10-07. |

No other protocol change is made to Nebula, SuperNeo, or HyperNova. The
`F_ops` checks (`rt < ts` and `wt = ts`) appear as O4 and the write-timestamp
formula. The `F_scan` address check (`a = a′ = i`) appears as the structural
index. The published `F_scan` has no timestamp range check, and this
specification adds none. The terminal is the Stage 1 terminal, which is the
HyperNova Construction 2 verifier.
