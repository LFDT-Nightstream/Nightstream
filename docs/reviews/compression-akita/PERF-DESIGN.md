# Faster compression: Metal now, CUDA later (design)

Owner request (2026-10-09): shrink rate 1/8, then make `finish_with_spartan`
faster, with Metal support and CUDA later, and use any cache the normal flow
can keep without making the normal flow slower.

## Grounding (measured, CPU, production fold, rate 1/8)

`finish_with_spartan` is about 54 s:

| Part | Time | Work |
| --- | --- | --- |
| Last fold (PiCCS + PiRLC) | 2.7 s | CPU for every engine today |
| Terminal package load | 5 s | once per process: JSON parse, identity hash, validation |
| Layer 1 | 38-42 s | see below |
| Shrink | 8 s | WHIR at 2^24 (rate 1/8), sum-checks, program recording |

Layer 1 by step (timers): P0 opening about 30%, early GKR about 30%, P0
commitment about 12%, P1 opening and commitment about 14%, ring weights and
quotient about 9%.

By function (`sample`, symbols, layer 1 and shrink): Poseidon2 about 50% of
busy CPU time, all in Merkle trees (leaf hashing and node compression);
generic map/fold loops 19%; Ajtai key expansion 7.5%; DFT butterflies 6.6%;
cubic-extension products 4.8%; eq tables 3.9%.

Throughput (2^22 rows of 16 words): Plonky3's CPU Merkle commit 14.8 M
permutations/s on 18 cores; the existing Metal Poseidon2 kernel 71 M/s
(scalar variant; the SIMD-group variant 32 M/s). Upload: 512 MB in 37 ms.

## What the normal flow can keep

Almost all of finish's cost depends on the last fold's parent witness, which
exists only after that fold. `extend` cannot do that work early without doing
the next fold. What a process can keep is fixed state: the terminal package
(5 s), the device and its pipelines, and later the expanded commitment key on
the device. None of it touches `extend`.

## Usage (caller's view)

```rust
// nightstream, finish: the engine chooses where the heavy loops run.
let accelerator = self.backend.accelerator();   // the CPU unless Metal
let layer1 = neo_spartan::prove(&relation, &setup.layer1, &accelerator, transcript, claim, witness)?;
let shrink = shrink::prove(&Shrink::new(key.shape, key.layer_bits)?, &program, &accelerator)?;
```

## Shape

- `neo_spartan::Device` (public trait, `u64` words, no Plonky3 type crosses):
  the batch operations a GPU runs. First: `hash_rows(words, width)`, the
  Merkle leaf digests (Plonky3 `PaddingFreeSponge`, width 16, rate 12, 4
  outputs, overwrite mode).
- `neo_spartan::Accelerator`: an optional shared device; the default is CPU.
- The Merkle leaf hasher declares `LANES > 1`, so Plonky3 0.8's tree hands it
  whole row batches (`hash_many`). With a device it hashes them on the
  device; without one it hashes row pairs with the packed NEON permutation,
  as Plonky3's packed path does. Tree layout, compression and proof formats
  stay Plonky3's.
- `neo-prover-metal`: one overwrite-sponge row kernel next to the existing
  Poseidon2 kernels.
- `nightstream`: `Engine::Metal` builds a Metal `Device`; every other engine
  passes none.
- CUDA later: one more `Device` implementation; nothing above changes.

Invariant: a device changes where work runs, never a byte of a proof. Layer
1 has no grinding, so its CPU and Metal proofs are byte-identical (test
`metal_layer1_proof_equals_cpu`). The shrink grinds 20 bits with Plonky3's
`find_any`, so its proof-of-work witness, and the bytes after it, change from
run to run on every engine; there the check is that the Metal final proof
verifies.

## Synthesis decision

- A (chosen): device behind Plonky3's seams and our own loops, one narrow
  trait. Smallest surface; proof bytes equal by construction; one place for
  CUDA.
- B (rejected): a whole GPU compression prover. It copies layer 1 and the
  shrink into shaders, and again for CUDA.
- C (kept as slice 0): caching and overlap. It cannot remove the
  witness-dependent cost, but it helps every engine.

## Slices

0. Load the terminal package in the background when finish starts, and wait
   for it only after layer 1. Hides about 5 s. (Opening a setup does not
   preload: that would put the load on the normal flow.)
1. The `Device` seam and the batched leaf hasher on CPU. Acceptance: roots and
   proofs equal Plonky3's; CPU finish not slower.
2. The Metal row kernel and the Metal device. Acceptance: Metal and CPU final
   proofs are byte-identical; finish faster on Metal.
3. GKR on every engine: no allocation per pair in the round message, the
   nodes 0, 2, 3, 4 by adding the line's slope, no products by one, and
   parallel `fold_low` and `eq_table`. Same messages (exact field
   arithmetic); the early GKR goes from 10.2 s to 3.7 s.
4. Shrink and sum-check on every engine: the shrink weights and products
   fill each block's own row and cell slots in parallel (the constant-one
   column and the glue rows as before); `prove_product` and `evaluate`
   run in parallel. A run that compared both with the serial code on the
   production program found them equal. Weights 1.38 s to 0.62 s, inner
   sum-check 0.48 s to 0.10 s.

Measured and rejected:

- One Plonky3 lane (its own packed path) against 2^16 lanes on CPU, same
  load: 48.5 s and 49.1 s; P0 commitment 7.84 s for both. Batching does not
  slow the CPU.
- A lock-free Metal session for leaf batches (many batches in flight): no
  gain. The row kernel is GPU-bound at about 30-44 M permutations/s for
  width-48 batches of 2^16 rows, serial, locked or concurrent.

Next by measured share (Metal): in the P0 opening each rayon thread waits
about 28% on the device, copies rows into Plonky3's batch buffer 25%,
compresses Merkle nodes 27% and runs the DFT 10%. Levers: a faster row
kernel (coalesced loads), Merkle node compression on the device (needs an
MMCS of our own: `MerkleTree` fields are crate-private in Plonky3 0.8), the
ring key weights from the commitment key that a Metal prover keeps on the
device (now SHAKE128 expansion on the CPU, about 2 s), the last fold on the
Metal NIFS prover (2.8 s), and Gruen's eq factoring in the GKR (about 25%
of its message work).

## Results (production fold, rate 1/8, other jobs on the machine)

Same binary, Metal then CPU, under the same load (load average 8-20):

| Step | CPU before | CPU | Metal |
| --- | --- | --- | --- |
| Last fold | 2.9 s | 2.8 s | 2.8 s |
| Wait for the terminal package | 0 | 0 | 0 |
| P0 commitment | 6.6 s | 6.5 s | 4.5 s |
| Early GKR | 9.9 s | 3.6 s | 3.8 s |
| Ring division | 3.2 s | 2.9 s | 2.9 s |
| P1 commitment and opening | 5.7 s | 5.9 s | 5.1 s |
| P0 opening | 11.2 s | 13.3 s | 8.1 s |
| Layer 1 | 38.6 s | 33.8 s | 25.9 s |
| Shrink | 8.9 s | 8.4 s | 7.0 s |
| `finish_with_spartan` | 50.3 s | 45.0 s | 35.6 s |

"CPU before" is slices 0-2 without slice 3. Before slice 0 the finish also
waited about 5 s for the package (M3: about 54 s). Final proofs verify on
CPU in about 320-360 ms on both engines.
