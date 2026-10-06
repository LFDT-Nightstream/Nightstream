# Prepared fixed-source bound

The single current verifier has 12,715,193 fixed array/u64 nodes before
application metadata. The compiler's exact count is

`N(W,L,R) = 12,715,193 + W + 4*[W>0] + 4*[L>0]`.

`W` counts application witness fields and `L` counts generated locals. The
reference application has `W=4`, `L=5,480`, hence 12,715,205 nodes. Application
rows and recipe payloads are stored separately from this fixed envelope.

The selected manifest has logical width `44,915,688 + 41*(W+L)`. The approved
maximum key is unchanged at 4,708,530 columns, or 254,260,620 scalar coordinates.
Thus `W+L <= floor((254,260,620 - 44,915,688)/41) = 5,105,973`. Ring padding does
not change that inequality because the maximum capacity is divisible by 54.

The largest variable metadata uses `W=5,105,972`, `L=1`: both nonempty source
segments add four nodes, giving **17,821,173 nodes**. The compiler regression
counts the complete assembled envelope for this case and for `W=L=0`.
It also checks that one additional field exceeds the maximum key.

The source-shape argument is unchanged: witness ports contribute W numeric
atoms; the application witness and local assignment runs contribute four
nodes each when nonempty. Other geometry and row counts replace atoms without
changing their number. The application matrix connector has fixed shape.
Application records are external to this count. Shifted Phi81 runs retain
their nonoverlapping source ranges.

Compact numeric-array JSON needs at most 21 bytes per node, so the byte
preflight is **374,244,633 bytes**. The decoder also counts nodes before each
allocation and retains its normal nesting check. This is a format-derived
bound, not a process RSS bound or authority for saved package identities.

The older measurements under `prepared-package-20260922` describe the previous
sampler and smaller application headroom. They do not determine this limit.
