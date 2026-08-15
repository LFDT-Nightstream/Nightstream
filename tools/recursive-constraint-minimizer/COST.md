# Campaign cost report

Metrics follow the campaign bar: recurring rows, then committed, public, and
auxiliary columns, per relation. "Before" numbers come from the committed
generated artifacts (grep `totalRows`, `columnCount`, `publicInputCount`,
`finalRows`, `finalColumns`, `finalPublicInputCount` under
`formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/MinimizerCampaign/Generated/`).
"After" columns fill in only when a Lean-authorized removal batch lands and the
relations regenerate (bar 6). No removal is accepted yet, so every "after"
cell is `pending`.

## Campaign profile v1 (minimal shape; pre-freeze)

| Relation | Metric | Before | After |
|---|---|---|---|
| Base arm (source) | rows | 39,949 | pending |
| Base arm (source) | columns | 38,626 | pending |
| Base arm (source) | public columns | 2,426 | pending |
| Base arm (selective fixed point) | recurring rows | 1,415,271 | pending |
| Base arm (selective fixed point) | total columns | 6,559,326 | pending |
| Base arm (selective fixed point) | public columns | 2,430 | pending |
| Terminal (source) | rows | 58,593 | pending |
| Terminal (source) | columns | 58,592 | pending |
| Terminal (source) | public columns | 48,871 | pending |
| Terminal (padded Spartan) | rows | 65,536 | pending |
| Terminal (padded Spartan) | columns | 114,407 | pending |
| Recursive arm | all metrics | pending capture (bar 4) | pending |

Family counts per census: base 6, terminal 8, recursive 82. Auxiliary columns
per relation are total columns minus public columns.
