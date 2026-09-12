# Actual selected NIFS conformance evidence

2026-09-12 UTC. Source base: `1607d34fe12593b39cad6f5c5c1f8b1ce853f8ab`,
plus the Rust source snapshots in `NATIVE_NIFS_EVIDENCE.zip`.

The selected base input now passes actual-witness C → R → D production
operations, complete normal NIFS verification, and comparison with an
independently computed Lean C/R/D result. The native proof encoding contains
945,983 bytes. All 43 complete NIFS mutations and all 55 PiDEC mutations in
the recorded suites are rejected.

This is executed conformance for one complete selected base input. It is not
an arbitrary-input Rust proof, a later recursive-input execution result, a
full-history execution result, or a production proof-backend result. The owner
keeps performance separate; these elapsed times document the capped checks,
not a runtime theorem. The protocol, package bytes, pins and cryptographic
assumptions are unchanged.

## Authority and stage boundaries

The original base fixture and verifier-owned package reconstruct the source
witnesses. C uses the normal optimized producer and evaluator. R uses the
shared production prefix. Saved C/R values are replayed from those original
inputs; the complete sponge state and absorption position must match. The
saved R witness is checked by recomputing its fixed-key commitment.

The canonical radix-two split determines all 16 D witnesses. Each loaded
digit must equal that split, and its activity and commitment are recomputed.
For active children 0 through 5, separate capped commands call the production
cache and opening evaluator on the exact digit witness. Children 6 through 15
are zero; the assembler uses the normal zero-opening branch after checking
their exact zero witnesses. It then calls `pi_dec::prove_from_split_material`
and the same observation, full NIFS verifier, mutation and encoding path as
the normal producer.

Saved openings remain ordinary proof inputs. Their metadata is not a proof
that they are witness evaluations. The six recorded computations establish
the origin of these particular inputs; normal verification checks their
public relations. The comparison does not substitute or repair expected
values. Lean separately executes C and R, checks D, and emits the complete
result from the supplied proof inputs. The Rust comparison checks complete
phase results, final children, transcript state and proof encoding. The saved
regression also loads the current pinned package and runs the normal D check.

## Executed checks

All native invocations used the project timeout of 300 seconds. The Lean
check and ordered gates used a timeout of 1,500 seconds. Only one build or
execution process ran at a time. Times below are elapsed command times and
include compilation when applicable.

| Check | Result | Seconds |
| --- | --- | ---: |
| Actual C/R producer and saved parent | Passed | 229.96 |
| C/R replay, recomputed parent commitment and canonical split | Passed | 134.74 |
| Child 0 opening | Passed | 231.71 |
| Child 1 opening | Passed | 230.82 |
| Child 2 opening | Passed | 233.91 |
| Child 3 opening | Passed | 230.19 |
| Child 4 opening | Passed | 226.10 |
| Child 5 opening | Passed | 214.12 |
| Complete assembly and 43 NIFS rejection checks | Passed | 146.77 |
| Independent Lean C/R/D result | Passed | 15.97 |
| Saved wire/result comparison and 55 D rejection checks | Passed | 20.23 |
| Substituted digit rejection test | Passed, 1 test | 18.50 |
| Staged/normal D comparison, including zero and active digits | Passed, 1 test | 50.67 |
| Retained actual-result regression against current package | Passed, 1 test | 39.29 |
| Ordered static gate | Passed | 8.76 |
| Ordered library build | Passed, 3,911 jobs | 0.90 |
| Ordered axiom gate | Passed, 3,999 jobs | 0.99 |

The axiom gate reports only the allowed `propext`, `Classical.choice` and
`Quot.sound` dependencies. Formatting passed. The unbroken full producer
remains ignored because it exceeded the native cap; it was not run again to
obtain this result. Staging is validated by the comparison with normal D on
the smaller fixture and by the recorded full-profile stage executions.

## Package and retained evidence

The unchanged Nightstream Goldilocks profile uses `b = 2`, `k_rho = 16`,
`B = 65536`, 17 sources, 16 children, 14 matrices and 28 rounds.

- Package file: `formal/nightstream-fprime/artifacts/nightstream-fprime-stage1-poseidon2-hash-chain-v1.json`.
- Structural identity: `[12322613552781674794, 11618660216342328080, 8890015725622288919, 10115040070495147315]`.
- Package identity: `[9705822157724451396, 520958727644325895, 9285622073986934000, 874020794279380938]`.
- Package SHA-256: `043bce25083eb15c733a903f4df2958acbc31a59dbe17420cd084c8242bd4d1f`.
- Actual result SHA-256: `9f7dccf82567bef88e30d5b844ff0efaeed2ea62dc7ee963e4cd4470966ae834`.
- Proof bytes SHA-256: `9bad16146f8e1b35fdf463166b153975a94edb494aa595bd01c79c7453c027b6`.
- Independent Lean result SHA-256: `f451dd6c32e00b40085d1e0dc86292dffa2c5d1a81d520ad4bfe120165ddf17d`.

`NATIVE_NIFS_EVIDENCE.zip` contains the saved actual proof and results, original
base fixture, six computed openings, source snapshots, command recipe, logs
and a SHA-256 manifest. Its copied C/R manifest references the earlier
`NATIVE_PARENT_EVIDENCE.zip` (SHA-256
`3f172178747d6f68b322025d746cd23fd0ac97a59c376f3137aeee87dbf8f9dd`).
The 4.88 GB parent witness and large digit files are not duplicated. Their
sizes and hashes remain in that earlier manifest. These hashes identify
review evidence; they are not protocol authority.

The compact regression retains `actual_result.json` and `proof.bin` under
`crates/neo-fold-clean/tests/nifs/fixtures/stage1_actual_nifs`. Its independent
Lean expectation is
`formal/nightstream-fprime/artifacts/nightstream-fprime-stage1-base-nifs-result-v1.json`.
The fixture README gives the regression command. The archive recipe records
all stage commands and the independent Lean result command.

Evidence archive: 2,402,471 bytes; SHA-256
`2c14aefd8319331d62d12103b549f04978d45561cd4c62ced864981b6d1d6b91`.
