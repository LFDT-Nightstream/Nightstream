# Second production fold and terminal checks

Host: Apple M1 Max, 64 GiB, macOS 26.6.2. Source: the uncommitted worktree
based on `b141ce1e677bd7ada72388330e7cc38d2de8534b`.
The release binary used `metal,cuda,neo-reductions/perf-timers`.
No Lean command ran, and production arithmetic did not change in this slice.

The complete Metal producer matched every byte of the new 945,983-byte CPU
proof. The separate output comparison also checked all sixteen returned
matrices, all claims and openings, parent, transcript, and identities:
585,569,271 reference bytes. The Metal successor then passed terminal
acceptance, wrong-state rejection, and rejection of a rehashed and recommitted
false opening. See `second-fold-output-comparison.json` and the phase logs.

All test invocations kept the repository's 300-second cap. RSS is the owner's
chosen memory measure. The normal guard was 16 GiB (17,179,869,184 bytes),
under the owner's approximate-16-GB allowance for this pass. Footprint is
recorded only as diagnostic data. The first CPU PiCCS run hit the RSS guard.
The owner approved one CPU reference invocation without that guard; it used
21,273,313,280 bytes and does not pass the production memory target.
`run_reference_test.py` records that one exception. It is not approval for
another run. All later phases used `run_rss_test.py` and the normal guard.

The CPU `openings` phase used one cache for all sixteen children. Before
writing the reference proof, `nifs` repeated the signed split, recomputed all
CPU commitments, compared every saved child matrix, and replayed the proof.
The complete Metal `prove` phase used the CPU proof only for its final
comparison. Its inputs were the newly generated step-2 source, with six
nonzero carried witnesses. Seven output witnesses are nonzero.

The private run directory is:

`/Users/nijaar/Library/Application Support/Nightstream/runs/metal-norm-20260921-bgylh03h`

It preserves full witnesses and proofs in `second-fold-cpu` and
`second-fold-metal`, the original requests, the memory observer library, and
the source snapshots `nonzero-fold-source.patch` and
`nonzero-fold-new-source.tar.gz`. The comparison script is copied here as
execution evidence; it originally ran from that private directory.

These are separate phase runs with repeated preparation and file input/output.
They do not establish an uninterrupted lifecycle benchmark, the 5× speed
target, or the memory bound for all supported circuits. Further memory tuning
is deferred under the owner's instruction.
