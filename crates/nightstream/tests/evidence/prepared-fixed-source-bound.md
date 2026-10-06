# Prepared fixed-source bound

The single current verifier has 13,991,754 fixed array/u64 nodes before
application metadata. The compiler's exact count is

`N(W,L,R) = 13,991,754 + W + 4*[W>0] + 4*[L>0]`.

`W` counts application witness fields and `L` counts generated locals. The
reference application has `W=4`, `L=5,480`, hence 13,991,766 nodes. Application
rows and recipe payloads are stored separately from this fixed envelope.

The selected manifest has logical width `59,507,383 + 41*(W+L)`. The approved
maximum key is unchanged at 4,708,530 columns, or 254,260,620 scalar coordinates.
The `2^27` domain is smaller than that key. Its complete 54-coordinate blocks
hold 134,217,702 scalar coordinates. Thus
`W+L <= floor((134,217,702 - 59,507,383)/41) = 1,822,202`.
The next field crosses the ring-padded domain bound.

The largest variable metadata uses `W=1,822,201`, `L=1`: both nonempty source
segments add four nodes, giving **15,813,963 nodes**. The compiler regression
counts the complete assembled envelope for this case and for `W=L=0`.
It also checks that one additional field exceeds the exported domain.

The source-shape argument is unchanged: witness ports contribute W numeric
atoms; the application witness and local assignment runs contribute four
nodes each when nonempty. Other geometry and row counts replace atoms without
changing their number. The application matrix connector has fixed shape.
Application records are external to this count. Shifted Phi81 runs retain
their nonoverlapping source ranges.

Compact numeric-array JSON needs at most 21 bytes per node, so the byte
preflight is **332,093,223 bytes**. The decoder also counts nodes before each
allocation and retains its normal nesting check. This is a format-derived
bound, not a process RSS bound or authority for saved package identities.

The older measurements under `prepared-package-20260922` describe the previous
sampler and smaller application headroom. They do not determine this limit.
