import Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.Assignment
import Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.CcsRefinement
import Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.ColumnMap
import Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.Necessity
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.NumericRowMap
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.NumericRowPadding

/-!
Curated model-level surface for the typed five-ring F' public carrier.

This facade intentionally does not import `PiRLCAlgebra`: carrier ownership is
independent of the concurrent algebra instantiation and therefore cannot create
an import cycle or imply NIFS closure.

The joint protocol owns its padded-identity opening. This carrier facade does
not export that protocol-level value.

| Protocol | Phase | Family | Public module |
|---|---|---|---|
| F' / CCS | fresh assignment | dimensions, mapping, projection | `Assignment` |
| F' / CCS | matrix source | aligned logical / completed carrier columns | `ColumnMap` |
| F' / CCS | matrix source | numeric / Boolean row mapping | `RowMap` |
| F' / CCS | matrix source | finite-row zero padding | `RowPadding` |
| F' / CCS | relation semantics | matrix image / residual preservation | `CcsRefinement` |
| assurance | necessity | norm-valid nonzero padding witness | `Necessity` |
-/
