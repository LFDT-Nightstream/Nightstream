# Public active extension: prepared validation

The staged iteration-2 to iteration-3 execution passes complete native NIFS,
Lean C/R/D and proof-byte comparison, full successor construction, independent
physical/logical rows, terminal acceptance and the three terminal mutations.
The public active-call fixture compiles. Its rehashed nonunit-child input is
rejected with `PriorFamily(PiDec(ChildXLowNorm))` before proving.

Two full public calls remain. Each starts from the same actual iteration-2
claims and complete witnesses and calls `package.extend` once. Expected
results never enter that call. The returned state, all child claims and
complete matrices, parent cache, fresh claim and complete fresh witness must
equal the separately checked iteration-3 output.

| Case | Input change |
| --- | --- |
| `parent-absent` | Remove the parent cache; change carried frame digests and the redundant scalar witness cache. |
| `parent-changed` | Change the supplied parent commitment, public input, point, Pad/matrix evaluations and frame; also change carried frames and the redundant scalar witness cache. |

These two cases also exercise successful active extension. A separate
ordinary-cache run adds no necessary coverage to this check.

## Proposed allowance, not yet approved

The measured C, R, D-material, terminal-acceptance and successor-construction
stages took 267.8793, 89.0901, 141.9882, 298.2090 and 187.0493 seconds.
Their sum is 984.215845722 seconds. Rounding it up gives a proposed cap of
**985 seconds for each of the two named calls**. These separate stages repeat
loading, commitments and cache construction; this is a measured planning
allowance, not a formal runtime upper bound.

`AGENTS.md` requires each native test invocation to stay within 300 seconds
unless the owner explicitly approves a longer allowance for that invocation.
Both full calls remain unexecuted. No longer allowance has been granted.
The existing shared build lock and process-group cleanup must remain in use.
No new memory limit, protocol, profile, backend, feature or environment
variable is part of this request.

The exact fixture commands, from the proof worktree, are:

```text
cargo run -p neo-fold-clean --release --features perf-timers --bin generate_pi_ccs_fixture -- check-public-active parent-absent formal/nightstream-fprime/artifacts/nightstream-fprime-stage1-poseidon2-hash-chain-v1.json /tmp/nightstream-native-nonzero-sources-1 /tmp/nightstream-native-nonzero-envelope-1 /tmp/nightstream-native-nonzero-material-1
cargo run -p neo-fold-clean --release --features perf-timers --bin generate_pi_ccs_fixture -- check-public-active parent-changed formal/nightstream-fprime/artifacts/nightstream-fprime-stage1-poseidon2-hash-chain-v1.json /tmp/nightstream-native-nonzero-sources-1 /tmp/nightstream-native-nonzero-envelope-1 /tmp/nightstream-native-nonzero-material-1
```

The fixture owner is `crates/neo-fold-clean/tests/nifs/stage1_active.rs`.
Its first build used whole-record equality on `CcsClaim`, which has no such
instance. The corrected fixture compares all four claim fields explicitly;
build two and the short rejection execution pass. Production code is unchanged.
