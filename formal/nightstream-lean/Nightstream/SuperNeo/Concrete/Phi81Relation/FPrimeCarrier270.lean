import Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.Assignment
import Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.CcsRefinement
import Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.ColumnMap
import Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.Necessity
import Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.PiCcsDomain
import Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.PiCcsSources
import Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.RowMap
import Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.RowPadding

/-!
Curated model-level surface for the typed five-ring F' public carrier.

This facade intentionally does not import `PiRLCAlgebra`: carrier ownership is
independent of the concurrent algebra instantiation and therefore cannot create
an import cycle or imply NIFS closure.

It also does not export `PaddedIdentityEvaluation`. That separate auxiliary CE
opening remains a direct import until a protocol theorem composes it with the
active relation; exposing it here would blur carrier ownership with authority.

| Protocol | Phase | Family | Public module |
|---|---|---|---|
| F' / CCS | fresh assignment | dimensions, mapping, projection | `Assignment` |
| F' / CCS | matrix source | aligned logical / completed carrier columns | `ColumnMap` |
| F' / CCS | matrix source | numeric / Boolean row mapping | `RowMap` |
| F' / CCS | matrix source | finite-row zero padding | `RowPadding` |
| F' / CCS | relation semantics | matrix image / residual preservation | `CcsRefinement` |
| `Pi_CCS` | plain NC domain | five-ring column / lane coverage | `PiCcsDomain` |
| `Pi_CCS` | split source adapter | fresh / running ownership and fresh truth | `PiCcsSources` |
| assurance | necessity | norm-valid nonzero padding witness | `Necessity` |
-/
