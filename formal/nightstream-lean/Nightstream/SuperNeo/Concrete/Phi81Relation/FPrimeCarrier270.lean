import Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.Assignment
import Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.CcsRefinement
import Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.ColumnMap
import Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.Necessity
import Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.RowMap

/-!
Curated model-level surface for the typed five-ring F' public carrier.

This facade intentionally does not import `PiRLCAlgebra`: carrier ownership is
independent of the concurrent algebra instantiation and therefore cannot create
an import cycle or imply NIFS closure.

| Protocol | Phase | Family | Public module |
|---|---|---|---|
| F' / CCS | fresh assignment | dimensions, mapping, projection | `Assignment` |
| F' / CCS | matrix source | aligned logical / completed carrier columns | `ColumnMap` |
| F' / CCS | matrix source | numeric / Boolean row mapping | `RowMap` |
| F' / CCS | relation semantics | matrix image / residual preservation | `CcsRefinement` |
| assurance | necessity | norm-valid nonzero padding witness | `Necessity` |
-/
