# Campaign cost report

Metrics follow the campaign bar: recurring rows, then committed, public, and
auxiliary columns, per relation. "Before" numbers come from the frozen
campaign profile v2 (PROFILE.md; k_rho = 12, the Definition-14 minimal
foldable shape). "After" columns fill in only when a Lean-authorized removal
batch lands and the relations regenerate (bar 6). No removal is applied yet,
so every "after" cell is `pending`. The final report also separates
committed-active columns from total width (Nebula Lemma 10: zeros commit for
free) and carries the per-family exclusive-column ledger from the
column-ownership census.

## Campaign profile v2 (k_rho = 12)

| Relation | Metric | Before | After |
|---|---|---|---|
| Base arm (source) | rows | 39,949 | pending |
| Base arm (source) | columns | 38,626 | pending |
| Base arm (source) | public columns | 2,426 | pending |
| Recursive arm (source) | rows | 11,187,825 | pending |
| Recursive arm (source) | columns | 11,078,210 | pending |
| Recursive arm (source) | public columns | 2,426 | pending |
| Selective fixed point | recurring rows | 3,666,055 | pending |
| Selective fixed point | total columns | 13,314,834 | pending |
| Selective fixed point | public columns | 2,430 | pending |
| Terminal (source) | rows | 58,593 | pending |
| Terminal (source) | columns | 58,592 | pending |
| Terminal (source) | public columns | 48,871 | pending |
| Terminal (padded Spartan) | rows | 65,536 | pending |
| Terminal (padded Spartan) | columns | 114,407 | pending |

Family counts per census: base 6, terminal 8, recursive 82. Auxiliary
columns per relation are total columns minus public columns.

## Column concentration (v1 measurement; re-measure at v2 for the report)

96.4% of recursive-arm columns are exclusively owned by one family; the top
four owners (`nifs.pi_ccs.padded_row.binding`,
`nifs.pi_rlc.verify.projection_binding.sis_digest`,
`nifs.pi_ccs.padded_row.output_digest.sis`,
`fprime.recursive.step.accumulator.output_authority.child_digests`) hold
~80% of all columns. The probe
(`probe_recursive_column_ownership_census`) re-runs at v2 for the final
ledger.

## Profile v1 record (superseded 2026-08-16)

Base 39,949 x 38,626; recursive 4,530,315 x 4,480,464; fixed point
1,415,271 x 6,559,326. Unfoldable (Definition 14); see PROFILE.md.
