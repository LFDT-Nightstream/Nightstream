## 7. Nightstream fold refinement

### NS-PIRLC-PROFILE — Selected PiRLC inputs and challenges

PiRLC MUST take the 15 CE outputs of NS-PICCS-TERMINAL in the same source
order. It MUST sample exactly 15 ring challenges with NS-SAMPLER-CANDIDATES
after every PiCCS output is transcript-bound. Coefficient `j` of challenge
`i` MUST be the accepted signed digit for global source `i`, coefficient `j`,
encoded through `iota_q`.

Decision: NSD-SAMPLER-001 and NSD-TRANSCRIPT-001.

### NS-PIDEC-PROFILE — Selected PiDEC children

PiDEC MUST use NS-SPLIT-BINARY and output exactly 14 ordered CE children. It
MUST derive each 270-field child public input from the public parent and check
the commitment and all 14 ring-evaluation recomposition equations. In a
sequence, fold `j+1` MUST use those children as its 14 ordered running claims
without insertion, removal, reordering, or value change. A bad sequence link
MUST reject.

Decision: NSD-SPLIT-001 and NSD-AUTHORITY-001.

### NS-RED-PADDED-RELATIONS — Reduction relation refinement

The Nightstream PiCCS strong relation MUST be the paper relation under the
zero-row embedding. Its output and ambient relations MUST remain
`BatchCE_15(b,L)` and `BatchCE_15(B_amb,L)`. The commitment projection MUST
remain unchanged and no padding or cache field may enter it.

Decision: NSD-REDUCTION-FRAMEWORK-001 and NSD-NORM-BINDING-001.

### NS-RED-COMPOSITION — Padded fold proof obligation

The end-to-end fold proof MUST first establish that zero-row embedding
preserves the paper PiCCS identities and strong conditions. It MUST then use
the reviewed weak PiRLC and PiDEC composition without an extended carrier
relation or an extra batching lemma.

Decision: NSD-REDUCTION-FRAMEWORK-001, NSD-BATCH-COINS-001, and
NSD-COLUMN-001.
