# Direct witness mask upload

Source: uncommitted worktree based on `b141ce1e677bd7ada72388330e7cc38d2de8534b`.
Host: Apple M1 Max, 64 GiB, macOS 26.6.2. No Lean command ran.

The allocation regression failed before the storage change and passed after it.
Both PiCCS and PiDEC now pack directly into the final shared Metal buffer.
Logical witness indices and counts remain unchanged. All thirteen joint device
checks and nine engine parity checks passed; two CUDA checks remain ignored.

The public uninterrupted Metal lifecycle passed in 183.62 s with
16,167,714,816 bytes peak RSS. The complete second-fold producer matched every
CPU proof byte and all sixteen returned matrices after the change, using
15,983,771,648 bytes peak RSS. Its reference is the CPU proof generated in the
previous validation slice, from identical step-2 inputs.

The standalone three-step Metal benchmark passed in 187.16 s, including
preparation, the base step, two active folds, and terminal verification.
Peak RSS was 16,594,255,872 bytes. The benchmark log records every phase,
fixed inputs, profile, circuit identity, and verified final state.
The full CPU timing and 5× comparison are still pending.

Every run kept the project's 300-second cap. Production tests and the
benchmark used the stated 16 GiB RSS guard under the owner's approximate
16 GB allowance. Physical footprint is diagnostic only. The former CPU
reference memory exception was not reused.

The complete input/output files and source snapshots are preserved under:

`/Users/nijaar/Library/Application Support/Nightstream/runs/metal-norm-20260921-bgylh03h`

The current output directory is `direct-mask-second-metal`. Snapshots are
`direct-masks-source.patch` and `direct-masks-new-source.tar.gz`.
The copied comparison script originally ran from that private directory.
The normal memory observer and test runner are preserved with the earlier
`nonzero-fold-20260921` evidence.
