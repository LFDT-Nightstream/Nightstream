# Parallel campaign extraction

This note preserves the small, reusable evidence from
`origin/claude/minimizer-campaign` at commit `1d69db1d69ae`. It does not copy
the generated Lean row lists from that branch.

## Assurance boundary

The entries below are model-level search evidence for the campaign profiles.
They are not permission to remove a constraint. The diagnostic digests identify
the campaign artifacts, but they are not authority.

Before reuse, regenerate each profile from the current Rust relation, replay
every retained Rust row, bind the artifact to the current final relation, and
produce a Lean-checked counterexample. The campaign did not complete those
checks for the current lifecycle relation.

## Campaign artifact identities

### Base arm

- Profile: `campaign-base-classification-v1`
- Source rows: `39,949`
- Source columns: `38,626`
- Public columns: `2,426`
- Source diagnostic digest:
  `sha256:54bec6fa7de4ec475e2fd43a1c015bfede809d2d1370b67677ea66dbda6839e7`
- Final rows: `1,415,271`
- Final columns: `6,559,326`
- Final public columns: `2,430`
- Final-plan diagnostic digest:
  `sha256:3024cf0eea6ac9093157e5dc1674187abc9fa3f17f8598d72ab41e45504e50fc`
- Projected-slice diagnostic digest:
  `sha256:f156e407a1da3a9d78cc7a558e30ab671ee7930e7c65c451bfdf5ac998491e94`

### Terminal relation

- Profile: `campaign-terminal-classification-v1`
- Source rows: `58,593`
- Source columns: `58,592`
- Public columns: `48,871`
- Source diagnostic digest:
  `sha256:85b400cebcfaa8fac702072aff342d67c6acca87e4470199d86a935c98264461`
- Padded Spartan rows: `65,536`
- Padded Spartan columns: `114,407`
- Native guards: `18`
- Terminal-binding diagnostic digest:
  `sha256:63664e95c3f91dcf35db99ad3e0dd235643d274e5ccfd9be6a18252eb8a12f98`

## Compact mutation ledger

Every generated necessity assignment is one coordinate change from one shared
accepted assignment for its scope.

| Scope | Family | Column | Background | Witness |
|---|---|---:|---:|---:|
| Base | `fprime.base.finalize.application` | 257 | 0 | 18446744069414584320 |
| Base | `fprime.base.step.output` | 3801 | 9368634332541568730 | 9368634332541568731 |
| Base | `fprime.base.step.initial` | 3811 | 1055183102398969389 | 1055183102398969390 |
| Base | `fprime.base.step.prelude` | 3819 | 6050346961767540117 | 6050346961767540118 |
| Base | `fprime.base.step.source` | 13708 | 1 | 2 |
| Base | `fprime.base.step.advance` | 13786 | 1 | 2 |
| Terminal | `terminal.running.commitment` | 1 | 0 | 1 |
| Terminal | `terminal.running.public_projection` | 973 | 0 | 1 |
| Terminal | `terminal.running.evaluations` | 1243 | 0 | 1 |
| Terminal | `terminal.fresh.commitment` | 47629 | 3019848447529899698 | 3019848447529899699 |
| Terminal | `terminal.fresh.public_projection` | 48601 | 1 | 2 |
| Terminal | `terminal.running.norm` | 49195 | 0 | 1 |
| Terminal | `terminal.fresh.norm` | 58267 | 1 | 2 |
| Terminal | `terminal.fresh.selected_relation` | 58591 | 0 | 1 |

The campaign stored `700,492` field entries across fourteen counterexample
modules. One shared background per scope plus these fourteen mutations needs
`97,232` field entries before further encoding. This removes `86.12%` of the
duplicated assignment data. The terminal background is also sparse: only 976
of its 58,592 entries are nonzero in the inspected witness.

## Reusable Lean shape

The useful proof shape is already supported by
`Nightstream.Assurance.ConstraintMinimization`:

1. Bind one complete, current Rust source artifact to its final relation.
2. Carry one accepted background assignment per lifecycle scope.
3. Define each candidate assignment as one checked coordinate mutation.
4. Check `RemovalCounterexample.Valid` against all retained families.
5. Apply `necessary_normalized_of_full_bound_valid` or
   `necessary_normalized_of_full_terminal_bound_valid`.

The next emitter should use a shared background and small mutation modules. It
must split generated data by the repository line limit, validate shard
coverage and locality, and add fail-closed axiom guards. It must not reuse the
campaign's byte-sized data splitting or its generated row lists unchanged.

## Rejected campaign output

- The generated Lean tree adds about 10.8 million lines.
- Individual data modules exceed 500,000 lines and the 1,500-line source-file
  limit.
- The campaign axiom file audits 34 declarations, while its handoff says 24.
- The handoff states that the terminal Lean build was not completed.
- The artifacts use the campaign profiles, not the current authoritative
  lifecycle profile.
