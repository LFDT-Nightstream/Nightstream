# Profiling scripts

This directory has scripts that profile Rust tests. The root `AGENTS.md`
sets the time caps for each run.

## Profile one test target

```bash
./scripts/profile_any_test.sh <package> <test_target> [mode]

# Examples:
./scripts/profile_any_test.sh nightstream application_poseidon2 flamegraph
./scripts/profile_any_test.sh neo-ajtai basic_commit_tests instruments
./scripts/profile_any_test.sh neo-ccs ccs_property_tests samply
```

`<test_target>` is the name of a file in the `tests/` folder of the package.
The script builds that target with the `profiling` Cargo profile, which keeps
the debug symbols. It needs `jq` to find the test binary.

| Mode | Tool | Output | Needs |
|---|---|---|---|
| `flamegraph` (default) | `flamegraph` with DTrace | `flamegraph-<package>-<test_target>.svg` | `cargo install flamegraph` and `sudo` |
| `instruments` | `xcrun xctrace` with the Time Profiler template | `profile-<package>-<test_target>.trace` | Xcode Command Line Tools |
| `samply` | `samply record` | `samply-<package>-<test_target>.json`, which opens in the Firefox Profiler | `cargo install samply` |
| `bench` | `cargo test`, five times | Timing in the terminal | Nothing more |

The script installs `flamegraph` or `samply` with `cargo install` when it does
not find it.
`scripts/check_profiling_tools.sh` shows which tools are installed.

## Profile one test function

These scripts take a package, a test target and a test function. Each script
prints its options when you give fewer than three arguments.

- `scripts/profile_xctrace.sh` records an Instruments trace and writes a
  symbolicated text report.
- `scripts/profile_for_ai.sh` writes a text profile to `profile-output.txt`.
- `scripts/profile_memory_deep.sh` takes heap snapshots over time and writes
  `profile-memory-deep.txt`.

`scripts/gpuprof.sh` runs the GPU profiling tool in `scripts/gpuprof/`.

## Troubleshooting

- If macOS shows "Permission denied", note that the flamegraph mode needs
  `sudo` for DTrace. The script calls `sudo`.
- If the shell shows "jq: command not found", install `jq`, for example with
  `brew install jq`.
- If the script cannot find the test binary, make sure that the package and
  the test target exist. `cargo test -p <package> --release -- --list` lists
  the tests.
- If the profile is empty, make sure that the test runs for more than 100 ms.
