# Evidence storage and terminal replay

The owner selected GitHub release assets for evidence on September 13.
Reports and SHA-256 sums stay in Git. New generated witnesses and graph
metadata do not enter Git in ZIP archives. Existing archives will be removed
from the current tree only after their external copies and links are verified;
no history rewrite is part of this task.

`EVIDENCE_RELEASE.json` lists the prepared assets for
`LFDT-Nightstream/Nightstream`, tag `stage1-evidence-48f06fab`.
**Upload and fresh-download verification are pending.** The commands below
require that release to exist. The local prepared copies are under
`/home/nicoarq/develop/nightstream-stage1-evidence`.

There are 55 historical archives (142,442,459 bytes). Their records can include
failed, incomplete and superseded attempts. The new child-witness asset adds
the previously missing sixteen matrices and `split.json` (162,917,001 bytes).
All seventeen archived members match the committed prior input manifests.
`TERMINAL_REPLAY_INPUTS.json` records those hashes and the required package
Git LFS OID. Copy verification is not a new terminal or protocol check.

## Fresh-checkout terminal inputs

Run from the repository root after setting up its documented Rust and Git LFS
tools. The selected package must be materialized from Git LFS; a pointer file
is not the package. The Lean caller fixture is already tracked. Downloaded
witnesses go into the ignored test-fixture directory below.

```bash
nightstream_repo="$PWD"
nightstream_replay="$PWD/crates/neo-fold-clean/tests/nifs/fixtures/stage1_terminal_replay"
mkdir -p "$nightstream_replay"
git lfs pull --include='formal/nightstream-fprime/artifacts/nightstream-fprime-stage1-poseidon2-hash-chain-v1.json'
gh release download stage1-evidence-48f06fab --repo LFDT-Nightstream/Nightstream --pattern 'NATIVE_ENVELOPE_EVIDENCE-b9ab74dc98c7d5875dcc5925d4f7be4ca42729f841f3708c24043f2096f679fb.zip' --pattern 'NATIVE_TERMINAL_CHILD_WITNESSES_EVIDENCE-933ef4b4e31310b59abf1ff07856efe308d39513a88c4af36966a68b83d3f10e.zip' --dir "$nightstream_replay"
(cd "$nightstream_replay" && sha256sum --check --ignore-missing "$nightstream_repo/docs/reviews/nightstream-fprime-requirements/EVIDENCE_SHA256SUMS")
unzip -q -j "$nightstream_replay/NATIVE_ENVELOPE_EVIDENCE-b9ab74dc98c7d5875dcc5925d4f7be4ca42729f841f3708c24043f2096f679fb.zip" 'output/envelope.json' 'output/fresh-claim.json' 'output/fresh-witness.json' -d "$nightstream_replay/envelope"
unzip -q "$nightstream_replay/NATIVE_TERMINAL_CHILD_WITNESSES_EVIDENCE-933ef4b4e31310b59abf1ff07856efe308d39513a88c4af36966a68b83d3f10e.zip" 'child-witnesses/*' -d "$nightstream_replay"
```

The expected state comes from the tracked independent Lean caller fixture,
not the carried envelope. These are the already-recorded iteration-2 inputs;
they do not constitute the pending later fold.

## Capped checks

Compile all crate test targets first. Each terminal case is a separate
invocation with a new output file and the existing 300-second cap. The timer
reports maximum resident memory; the owner did not adopt a new memory limit.
Do not report a timed-out invocation as passed or as an ignored success.

```bash
/usr/bin/time -v timeout --signal=TERM 300 python3 -B scripts/lean_graph/guard.py --kind rust --cwd . -- cargo test -p neo-fold-clean --release --no-run
nightstream_terminal_checks() {
for nightstream_case in accepted ce-evaluation ce-matrix-evaluation fresh-private; do
  /usr/bin/time -v timeout --signal=TERM 300 python3 -B scripts/lean_graph/guard.py --kind rust --cwd . -- cargo run -p neo-fold-clean --release --bin generate_pi_ccs_fixture -- check-owned-terminal "$nightstream_case" "$nightstream_replay/envelope" "$nightstream_replay/child-witnesses" "$nightstream_replay/$nightstream_case.json" || return "$?"
done
}
nightstream_terminal_checks
```

Check each exit and result separately. The original logs remain in
`NATIVE_TERMINAL_EVIDENCE` and are identified in the release manifest. Until
the new asset is uploaded and downloaded successfully, terminal replay from
a fresh checkout remains an open delivery condition.

## Explicit-prior fixture checks

The small state inputs are tracked under
`crates/neo-fold-clean/tests/nifs/fixtures/stage1_recursive_states/`.
The C input and child claims come from the tracked complete Lean result;
the following extraction reproduces the bytes used by the recorded checks.
Run from the repository root with a new output directory.

```bash
nightstream_repo="$PWD"
nightstream_case=valid
nightstream_cases="$PWD/crates/neo-fold-clean/tests/nifs/fixtures/stage1_terminal_replay/emitter"
mkdir -p "$nightstream_cases"
python3 -B - "$nightstream_cases" <<'PY'
import json, sys
from pathlib import Path
result = json.loads(Path('formal/nightstream-fprime/artifacts/nightstream-fprime-stage1-base-nifs-result-v1.json').read_text())
for name, value in [('input', result[1]), ('children', result[9][16][1])]:
    (Path(sys.argv[1]) / (name + '.json')).write_text(json.dumps(value, separators=(',', ':')) + '\n')
PY
/usr/bin/time -v timeout --signal=TERM 1500 python3 -B scripts/lean_graph/guard.py --kind lean --cwd formal/nightstream-fprime -- bash scripts/validate.sh recursive-step-fixture 18363630987318625048 9406776669274472459 1104198490699942438 1757792822492309855 "$nightstream_cases/input.json" "$nightstream_cases/children.json" "$nightstream_repo/crates/neo-fold-clean/tests/nifs/fixtures/stage1_recursive_states/$nightstream_case.json" "$nightstream_cases/$nightstream_case-output.json"
```

For `valid`, require exit 0 and exact byte equality with the tracked
`nightstream-fprime-stage1-actual-recursive-step-fixture-v1.json` artifact.
Set `nightstream_case` to each name below and repeat the capped command.
Each rejection must write no output file and contain the stated error.

| Case | Exit | Error text |
| --- | --- | --- |
| `zero-counter` | 1 | `prior counter must be positive with a canonical successor` |
| `counter-no-successor` | 1 | `prior counter must be positive with a canonical successor` |
| `wrong-state-width` | 2 | `expected four state or message words` |
| `noncanonical-word` | 2 | `noncanonical Goldilocks word` |

## Nonzero-running NIFS replay

`NONZERO_NIFS_GATES.json` records the completed staged C/R/D check on the
iteration-2 envelope. All sixteen original running witnesses enter the
producer, including six nonzero witnesses. Its new split has seven nonzero
digits. The complete result and 945,983 proof bytes agree with Lean; the
normal verifier and required mutations pass. The subsequent caller, full
assignment and terminal checks also pass; they are recorded separately in
`NONZERO_SUCCESSOR_GATES.json`. Uninterrupted public `extend` remains open.

The two release assets above are sufficient starting inputs. The 5.2 GB R
witness is regenerated by the measured R stage, and the new digit matrices
by the measured split stage. They need not be downloaded as extra assets.
The exact later message is tracked as `stage1_recursive_states/nonzero-running.json`.
Use a fresh `nonzero` directory after the earlier download and extraction.

```bash
nightstream_nonzero="$nightstream_replay/nonzero"
nightstream_sources="$nightstream_nonzero/sources"
nightstream_package="$nightstream_repo/formal/nightstream-fprime/artifacts/nightstream-fprime-stage1-poseidon2-hash-chain-v1.json"
mkdir -p "$nightstream_sources" "$nightstream_nonzero/openings"
for nightstream_file in envelope.json fresh-claim.json fresh-witness.json; do
  ln -s "$nightstream_replay/envelope/$nightstream_file" "$nightstream_sources/$nightstream_file"
done
for nightstream_child in {0..15}; do
  ln -s "$nightstream_replay/child-witnesses/digit-$nightstream_child.json" "$nightstream_sources/digit-$nightstream_child.json"
done
cp "$nightstream_repo/crates/neo-fold-clean/tests/nifs/fixtures/stage1_recursive_states/nonzero-running.json" "$nightstream_sources/next-message-input.json"
nightstream_native() {
  /usr/bin/time -v timeout --signal=TERM 300 python3 -B scripts/lean_graph/guard.py --kind rust --cwd . -- cargo run -p neo-fold-clean --release --features perf-timers --bin generate_pi_ccs_fixture -- "$@"
}
nightstream_nonzero_nifs() {
  nightstream_native check-owned-sources "$nightstream_package" "$nightstream_sources" || return "$?"
  nightstream_native prove-owned-ccs "$nightstream_package" "$nightstream_sources" "$nightstream_nonzero/ccs.json" || return "$?"
  nightstream_native prove-owned-rlc "$nightstream_package" "$nightstream_sources" "$nightstream_nonzero/ccs.json" "$nightstream_nonzero/parent" || return "$?"
  nightstream_native check-owned-parent "$nightstream_package" "$nightstream_sources" "$nightstream_nonzero/parent" "$nightstream_nonzero/material" || return "$?"
  for nightstream_child in {0..6}; do
    nightstream_native open-owned-child "$nightstream_package" "$nightstream_sources" "$nightstream_nonzero/parent" "$nightstream_nonzero/material" "$nightstream_child" "$nightstream_nonzero/openings/child-$nightstream_child.json" || return "$?"
  done
  nightstream_native assemble-owned-nifs "$nightstream_package" "$nightstream_sources" "$nightstream_nonzero/parent" "$nightstream_nonzero/material" "$nightstream_nonzero/openings" "$nightstream_nonzero/actual" || return "$?"
  /usr/bin/time -v timeout --signal=TERM 1500 python3 -B scripts/lean_graph/guard.py --kind lean --cwd formal/nightstream-fprime -- bash scripts/validate.sh pi-dec-input-check 9705822157724451396 520958727644325895 9285622073986934000 874020794279380938 "$nightstream_nonzero/actual/pi_ccs_input.json" "$nightstream_nonzero/actual/children.json" "$nightstream_nonzero/lean.json" || return "$?"
  nightstream_native check-owned-nifs "$nightstream_package" "$nightstream_nonzero/actual" "$nightstream_nonzero/lean.json"
}
nightstream_nonzero_nifs
```

Each command keeps the existing cap and stops this sequence on failure.
The seven opening files are specific to this deterministic input. Assembly
recomputes the full split and all commitments, checks every saved digit, and
derives the other nine zero openings from those exact values. It does not
trust a saved activity flag. Timing records are measurements on the checked
host, not runtime guarantees for another host. Release upload and a fresh
download remain pending; these commands do not claim that delivery has passed.

## Later successor and terminal replay

Continue from the regenerated NIFS result above. The explicit request binds
the previous counter and state. The preceding Lean caller fixture supplies
the independent iteration-2 output for the raw-row continuity check.

```bash
nightstream_artifacts="$nightstream_repo/formal/nightstream-fprime/artifacts"
git lfs pull --include='formal/nightstream-fprime/artifacts/nightstream-fprime-stage1-poseidon2-hash-chain-v1-expanded.json,formal/nightstream-fprime/artifacts/nightstream-fprime-stage1-poseidon2-hash-chain-v1-binding-v1.json'
nightstream_later_step() {
  /usr/bin/time -v timeout --signal=TERM 1500 python3 -B scripts/lean_graph/guard.py --kind lean --cwd formal/nightstream-fprime -- bash scripts/validate.sh recursive-step-fixture 18363630987318625048 9406776669274472459 1104198490699942438 1757792822492309855 "$nightstream_nonzero/actual/pi_ccs_input.json" "$nightstream_nonzero/actual/children.json" "$nightstream_sources/next-message-input.json" "$nightstream_nonzero/step.json" || return "$?"
  nightstream_native complete-later-envelope "$nightstream_sources" "$nightstream_nonzero/actual" "$nightstream_nonzero/step.json" "$nightstream_nonzero/material" "$nightstream_nonzero/envelope" || return "$?"
  /usr/bin/time -v timeout --signal=TERM 1500 python3 -B scripts/lean_graph/guard.py --kind lean --cwd formal/nightstream-fprime -- bash scripts/validate.sh pi-rlc-input-check 9705822157724451396 520958727644325895 9285622073986934000 874020794279380938 "$nightstream_nonzero/actual/pi_ccs_input.json" "$nightstream_nonzero/rlc-lean.json" || return "$?"
  /usr/bin/time -v timeout --signal=TERM 300 python3 -B scripts/lean_graph/guard.py --kind rust --cwd . -- cargo run -p nightstream-fprime --release --bin check_package_conformance -- recursive "$nightstream_package" "$nightstream_artifacts/nightstream-fprime-stage1-poseidon2-hash-chain-v1-binding-v1.json" "$nightstream_artifacts/nightstream-fprime-ajtai-setup-v1-parity.json" 12322613552781674794 11618660216342328080 8890015725622288919 10115040070495147315 "$nightstream_artifacts/nightstream-fprime-stage1-poseidon2-hash-chain-v1-expanded.json" "$nightstream_nonzero/step.json" "$nightstream_artifacts/nightstream-fprime-stage1-actual-recursive-step-fixture-v1.json" "$nightstream_nonzero/actual/pi_ccs_input.json" "$nightstream_nonzero/actual/children.json" "$nightstream_nonzero/rlc-lean.json" "$nightstream_nonzero/lean.json" || return "$?"
  nightstream_native check-later-terminal accepted "$nightstream_nonzero/envelope" "$nightstream_nonzero/material" "$nightstream_nonzero/step.json" "$nightstream_nonzero/accepted.json" || return "$?"
  mkdir -p "$nightstream_nonzero/mutations"
  for nightstream_case in ce-evaluation ce-matrix-evaluation fresh-private; do
    nightstream_native prepare-terminal-mutation "$nightstream_case" "$nightstream_nonzero/envelope" "$nightstream_nonzero/material" "$nightstream_nonzero/step.json" "$nightstream_nonzero/mutations/$nightstream_case" || return "$?"
    nightstream_native check-prepared-terminal-mutation "$nightstream_case" "$nightstream_nonzero/mutations/$nightstream_case" "$nightstream_nonzero/material" "$nightstream_nonzero/step.json" "$nightstream_nonzero/$nightstream_case.json" || return "$?"
  done
}
nightstream_later_step
```

The later terminal acceptance took 298.21 seconds. Preparation of a changed
commitment is therefore a separate capped command. Verification still checks
the complete retained witnesses with the original public verifier and requires
the exact Pad, matrix-evaluation or fresh-row error. It trusts no preparation
flag. Each verified mutation took less than 300 seconds on the checked host.

`NONZERO_EXECUTION_RELEASE.json` describes the prepared external archive with
the later envelope, all sixteen child matrices, the independent Lean caller,
logs and source snapshots. Those outputs permit direct terminal replay without
rebuilding C/R/D. Upload and download verification are still pending; no new
archive is committed to Git. The reviewed public-call proposal remains in
`PUBLIC_ACTIVE_VALIDATION.md` and has no approved longer allowance yet.
