/-
Generated file: production combined-NC artifact; do not hand-edit.

Owns: at most 128 exact normalized source A/B/C rows.

Does not own: decoding, row satisfaction, transcript authority, commitment
binding, semantic acceptance, costs, or permission to remove rows.

Emits constraints: no.

| Stable stage path | Obligation | Authority class |
|---|---|---|
| `f_prime.pi_ccs_nc.delayed.combined.generated` | The generated payload named by `Owns` above | computed artifact |
-/

import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Schema

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.SourceRows.Chunk37

set_option maxRecDepth 100000 in
def values : List RawSourceRow := [
  { schemaVersion := 1
    rows := 11308137
    columns := 10997363
    sourceRow := 4292316
    a := [{ column := 4078031, coefficient := 1 }, { column := 4078041, coefficient := 18446744069414584320 }, { column := 4078043, coefficient := 1 }]
    b := [{ column := 0, coefficient := 1 }]
    c := [] }
,
  { schemaVersion := 1
    rows := 11308137
    columns := 10997363
    sourceRow := 4292317
    a := [{ column := 4078032, coefficient := 1 }, { column := 4078042, coefficient := 18446744069414584320 }, { column := 4078044, coefficient := 1 }]
    b := [{ column := 0, coefficient := 1 }]
    c := [] }
,
  { schemaVersion := 1
    rows := 11308137
    columns := 10997363
    sourceRow := 4292318
    a := [{ column := 77639, coefficient := 1 }]
    b := [{ column := 4075048, coefficient := 1 }]
    c := [{ column := 4078045, coefficient := 1 }] }
,
  { schemaVersion := 1
    rows := 11308137
    columns := 10997363
    sourceRow := 4292319
    a := [{ column := 77640, coefficient := 1 }]
    b := [{ column := 4075049, coefficient := 1 }]
    c := [{ column := 4078046, coefficient := 1 }] }
,
  { schemaVersion := 1
    rows := 11308137
    columns := 10997363
    sourceRow := 4292320
    a := [{ column := 77639, coefficient := 1 }, { column := 77640, coefficient := 1 }]
    b := [{ column := 4075048, coefficient := 1 }, { column := 4075049, coefficient := 1 }]
    c := [{ column := 4078047, coefficient := 1 }] }
,
  { schemaVersion := 1
    rows := 11308137
    columns := 10997363
    sourceRow := 4292321
    a := [{ column := 4078045, coefficient := 18446744069414584320 }, { column := 4078046, coefficient := 18446744069414584314 }, { column := 4078048, coefficient := 1 }]
    b := [{ column := 0, coefficient := 1 }]
    c := [] }
,
  { schemaVersion := 1
    rows := 11308137
    columns := 10997363
    sourceRow := 4292322
    a := [{ column := 4078045, coefficient := 1 }, { column := 4078046, coefficient := 1 }, { column := 4078047, coefficient := 18446744069414584320 }, { column := 4078049, coefficient := 1 }]
    b := [{ column := 0, coefficient := 1 }]
    c := [] }
,
  { schemaVersion := 1
    rows := 11308137
    columns := 10997363
    sourceRow := 4292323
    a := [{ column := 77641, coefficient := 1 }]
    b := [{ column := 4075053, coefficient := 1 }]
    c := [{ column := 4078050, coefficient := 1 }] }
,
  { schemaVersion := 1
    rows := 11308137
    columns := 10997363
    sourceRow := 4292324
    a := [{ column := 77642, coefficient := 1 }]
    b := [{ column := 4075054, coefficient := 1 }]
    c := [{ column := 4078051, coefficient := 1 }] }
,
  { schemaVersion := 1
    rows := 11308137
    columns := 10997363
    sourceRow := 4292325
    a := [{ column := 77641, coefficient := 1 }, { column := 77642, coefficient := 1 }]
    b := [{ column := 4075053, coefficient := 1 }, { column := 4075054, coefficient := 1 }]
    c := [{ column := 4078052, coefficient := 1 }] }
,
  { schemaVersion := 1
    rows := 11308137
    columns := 10997363
    sourceRow := 4292326
    a := [{ column := 4078050, coefficient := 18446744069414584320 }, { column := 4078051, coefficient := 18446744069414584314 }, { column := 4078053, coefficient := 1 }]
    b := [{ column := 0, coefficient := 1 }]
    c := [] }
,
  { schemaVersion := 1
    rows := 11308137
    columns := 10997363
    sourceRow := 4292327
    a := [{ column := 4078050, coefficient := 1 }, { column := 4078051, coefficient := 1 }, { column := 4078052, coefficient := 18446744069414584320 }, { column := 4078054, coefficient := 1 }]
    b := [{ column := 0, coefficient := 1 }]
    c := [] }
,
  { schemaVersion := 1
    rows := 11308137
    columns := 10997363
    sourceRow := 4292328
    a := [{ column := 77643, coefficient := 1 }]
    b := [{ column := 4075058, coefficient := 1 }]
    c := [{ column := 4078055, coefficient := 1 }] }
,
  { schemaVersion := 1
    rows := 11308137
    columns := 10997363
    sourceRow := 4292329
    a := [{ column := 77644, coefficient := 1 }]
    b := [{ column := 4075059, coefficient := 1 }]
    c := [{ column := 4078056, coefficient := 1 }] }
,
  { schemaVersion := 1
    rows := 11308137
    columns := 10997363
    sourceRow := 4292330
    a := [{ column := 77643, coefficient := 1 }, { column := 77644, coefficient := 1 }]
    b := [{ column := 4075058, coefficient := 1 }, { column := 4075059, coefficient := 1 }]
    c := [{ column := 4078057, coefficient := 1 }] }
,
  { schemaVersion := 1
    rows := 11308137
    columns := 10997363
    sourceRow := 4292331
    a := [{ column := 4078055, coefficient := 18446744069414584320 }, { column := 4078056, coefficient := 18446744069414584314 }, { column := 4078058, coefficient := 1 }]
    b := [{ column := 0, coefficient := 1 }]
    c := [] }
,
  { schemaVersion := 1
    rows := 11308137
    columns := 10997363
    sourceRow := 4292332
    a := [{ column := 4078055, coefficient := 1 }, { column := 4078056, coefficient := 1 }, { column := 4078057, coefficient := 18446744069414584320 }, { column := 4078059, coefficient := 1 }]
    b := [{ column := 0, coefficient := 1 }]
    c := [] }
,
  { schemaVersion := 1
    rows := 11308137
    columns := 10997363
    sourceRow := 4292333
    a := [{ column := 77645, coefficient := 1 }]
    b := [{ column := 4075063, coefficient := 1 }]
    c := [{ column := 4078060, coefficient := 1 }] }
,
  { schemaVersion := 1
    rows := 11308137
    columns := 10997363
    sourceRow := 4292334
    a := [{ column := 77646, coefficient := 1 }]
    b := [{ column := 4075064, coefficient := 1 }]
    c := [{ column := 4078061, coefficient := 1 }] }
,
  { schemaVersion := 1
    rows := 11308137
    columns := 10997363
    sourceRow := 4292335
    a := [{ column := 77645, coefficient := 1 }, { column := 77646, coefficient := 1 }]
    b := [{ column := 4075063, coefficient := 1 }, { column := 4075064, coefficient := 1 }]
    c := [{ column := 4078062, coefficient := 1 }] }
,
  { schemaVersion := 1
    rows := 11308137
    columns := 10997363
    sourceRow := 4292336
    a := [{ column := 4078060, coefficient := 18446744069414584320 }, { column := 4078061, coefficient := 18446744069414584314 }, { column := 4078063, coefficient := 1 }]
    b := [{ column := 0, coefficient := 1 }]
    c := [] }
,
  { schemaVersion := 1
    rows := 11308137
    columns := 10997363
    sourceRow := 4292337
    a := [{ column := 4078060, coefficient := 1 }, { column := 4078061, coefficient := 1 }, { column := 4078062, coefficient := 18446744069414584320 }, { column := 4078064, coefficient := 1 }]
    b := [{ column := 0, coefficient := 1 }]
    c := [] }
,
  { schemaVersion := 1
    rows := 11308137
    columns := 10997363
    sourceRow := 4292338
    a := [{ column := 77647, coefficient := 1 }]
    b := [{ column := 4075068, coefficient := 1 }]
    c := [{ column := 4078065, coefficient := 1 }] }
,
  { schemaVersion := 1
    rows := 11308137
    columns := 10997363
    sourceRow := 4292339
    a := [{ column := 77648, coefficient := 1 }]
    b := [{ column := 4075069, coefficient := 1 }]
    c := [{ column := 4078066, coefficient := 1 }] }
,
  { schemaVersion := 1
    rows := 11308137
    columns := 10997363
    sourceRow := 4292340
    a := [{ column := 77647, coefficient := 1 }, { column := 77648, coefficient := 1 }]
    b := [{ column := 4075068, coefficient := 1 }, { column := 4075069, coefficient := 1 }]
    c := [{ column := 4078067, coefficient := 1 }] }
,
  { schemaVersion := 1
    rows := 11308137
    columns := 10997363
    sourceRow := 4292341
    a := [{ column := 4078065, coefficient := 18446744069414584320 }, { column := 4078066, coefficient := 18446744069414584314 }, { column := 4078068, coefficient := 1 }]
    b := [{ column := 0, coefficient := 1 }]
    c := [] }
,
  { schemaVersion := 1
    rows := 11308137
    columns := 10997363
    sourceRow := 4292342
    a := [{ column := 4078065, coefficient := 1 }, { column := 4078066, coefficient := 1 }, { column := 4078067, coefficient := 18446744069414584320 }, { column := 4078069, coefficient := 1 }]
    b := [{ column := 0, coefficient := 1 }]
    c := [] }
,
  { schemaVersion := 1
    rows := 11308137
    columns := 10997363
    sourceRow := 4292343
    a := [{ column := 77649, coefficient := 1 }]
    b := [{ column := 4075073, coefficient := 1 }]
    c := [{ column := 4078070, coefficient := 1 }] }
,
  { schemaVersion := 1
    rows := 11308137
    columns := 10997363
    sourceRow := 4292344
    a := [{ column := 77650, coefficient := 1 }]
    b := [{ column := 4075074, coefficient := 1 }]
    c := [{ column := 4078071, coefficient := 1 }] }
,
  { schemaVersion := 1
    rows := 11308137
    columns := 10997363
    sourceRow := 4292345
    a := [{ column := 77649, coefficient := 1 }, { column := 77650, coefficient := 1 }]
    b := [{ column := 4075073, coefficient := 1 }, { column := 4075074, coefficient := 1 }]
    c := [{ column := 4078072, coefficient := 1 }] }
,
  { schemaVersion := 1
    rows := 11308137
    columns := 10997363
    sourceRow := 4292346
    a := [{ column := 4078070, coefficient := 18446744069414584320 }, { column := 4078071, coefficient := 18446744069414584314 }, { column := 4078073, coefficient := 1 }]
    b := [{ column := 0, coefficient := 1 }]
    c := [] }
,
  { schemaVersion := 1
    rows := 11308137
    columns := 10997363
    sourceRow := 4292347
    a := [{ column := 4078070, coefficient := 1 }, { column := 4078071, coefficient := 1 }, { column := 4078072, coefficient := 18446744069414584320 }, { column := 4078074, coefficient := 1 }]
    b := [{ column := 0, coefficient := 1 }]
    c := [] }
,
  { schemaVersion := 1
    rows := 11308137
    columns := 10997363
    sourceRow := 4292348
    a := [{ column := 77651, coefficient := 1 }]
    b := [{ column := 4075078, coefficient := 1 }]
    c := [{ column := 4078075, coefficient := 1 }] }
,
  { schemaVersion := 1
    rows := 11308137
    columns := 10997363
    sourceRow := 4292349
    a := [{ column := 77652, coefficient := 1 }]
    b := [{ column := 4075079, coefficient := 1 }]
    c := [{ column := 4078076, coefficient := 1 }] }
,
  { schemaVersion := 1
    rows := 11308137
    columns := 10997363
    sourceRow := 4292350
    a := [{ column := 77651, coefficient := 1 }, { column := 77652, coefficient := 1 }]
    b := [{ column := 4075078, coefficient := 1 }, { column := 4075079, coefficient := 1 }]
    c := [{ column := 4078077, coefficient := 1 }] }
,
  { schemaVersion := 1
    rows := 11308137
    columns := 10997363
    sourceRow := 4292351
    a := [{ column := 4078075, coefficient := 18446744069414584320 }, { column := 4078076, coefficient := 18446744069414584314 }, { column := 4078078, coefficient := 1 }]
    b := [{ column := 0, coefficient := 1 }]
    c := [] }
,
  { schemaVersion := 1
    rows := 11308137
    columns := 10997363
    sourceRow := 4292352
    a := [{ column := 4078075, coefficient := 1 }, { column := 4078076, coefficient := 1 }, { column := 4078077, coefficient := 18446744069414584320 }, { column := 4078079, coefficient := 1 }]
    b := [{ column := 0, coefficient := 1 }]
    c := [] }
,
  { schemaVersion := 1
    rows := 11308137
    columns := 10997363
    sourceRow := 4292353
    a := [{ column := 77653, coefficient := 1 }]
    b := [{ column := 4075083, coefficient := 1 }]
    c := [{ column := 4078080, coefficient := 1 }] }
,
  { schemaVersion := 1
    rows := 11308137
    columns := 10997363
    sourceRow := 4292354
    a := [{ column := 77654, coefficient := 1 }]
    b := [{ column := 4075084, coefficient := 1 }]
    c := [{ column := 4078081, coefficient := 1 }] }
,
  { schemaVersion := 1
    rows := 11308137
    columns := 10997363
    sourceRow := 4292355
    a := [{ column := 77653, coefficient := 1 }, { column := 77654, coefficient := 1 }]
    b := [{ column := 4075083, coefficient := 1 }, { column := 4075084, coefficient := 1 }]
    c := [{ column := 4078082, coefficient := 1 }] }
,
  { schemaVersion := 1
    rows := 11308137
    columns := 10997363
    sourceRow := 4292356
    a := [{ column := 4078080, coefficient := 18446744069414584320 }, { column := 4078081, coefficient := 18446744069414584314 }, { column := 4078083, coefficient := 1 }]
    b := [{ column := 0, coefficient := 1 }]
    c := [] }
,
  { schemaVersion := 1
    rows := 11308137
    columns := 10997363
    sourceRow := 4292357
    a := [{ column := 4078080, coefficient := 1 }, { column := 4078081, coefficient := 1 }, { column := 4078082, coefficient := 18446744069414584320 }, { column := 4078084, coefficient := 1 }]
    b := [{ column := 0, coefficient := 1 }]
    c := [] }
,
  { schemaVersion := 1
    rows := 11308137
    columns := 10997363
    sourceRow := 4292358
    a := [{ column := 77655, coefficient := 1 }]
    b := [{ column := 4075088, coefficient := 1 }]
    c := [{ column := 4078085, coefficient := 1 }] }
,
  { schemaVersion := 1
    rows := 11308137
    columns := 10997363
    sourceRow := 4292359
    a := [{ column := 77656, coefficient := 1 }]
    b := [{ column := 4075089, coefficient := 1 }]
    c := [{ column := 4078086, coefficient := 1 }] }
,
  { schemaVersion := 1
    rows := 11308137
    columns := 10997363
    sourceRow := 4292360
    a := [{ column := 77655, coefficient := 1 }, { column := 77656, coefficient := 1 }]
    b := [{ column := 4075088, coefficient := 1 }, { column := 4075089, coefficient := 1 }]
    c := [{ column := 4078087, coefficient := 1 }] }
,
  { schemaVersion := 1
    rows := 11308137
    columns := 10997363
    sourceRow := 4292361
    a := [{ column := 4078085, coefficient := 18446744069414584320 }, { column := 4078086, coefficient := 18446744069414584314 }, { column := 4078088, coefficient := 1 }]
    b := [{ column := 0, coefficient := 1 }]
    c := [] }
,
  { schemaVersion := 1
    rows := 11308137
    columns := 10997363
    sourceRow := 4292362
    a := [{ column := 4078085, coefficient := 1 }, { column := 4078086, coefficient := 1 }, { column := 4078087, coefficient := 18446744069414584320 }, { column := 4078089, coefficient := 1 }]
    b := [{ column := 0, coefficient := 1 }]
    c := [] }
,
  { schemaVersion := 1
    rows := 11308137
    columns := 10997363
    sourceRow := 4292363
    a := [{ column := 77657, coefficient := 1 }]
    b := [{ column := 4075093, coefficient := 1 }]
    c := [{ column := 4078090, coefficient := 1 }] }
,
  { schemaVersion := 1
    rows := 11308137
    columns := 10997363
    sourceRow := 4292364
    a := [{ column := 77658, coefficient := 1 }]
    b := [{ column := 4075094, coefficient := 1 }]
    c := [{ column := 4078091, coefficient := 1 }] }
,
  { schemaVersion := 1
    rows := 11308137
    columns := 10997363
    sourceRow := 4292365
    a := [{ column := 77657, coefficient := 1 }, { column := 77658, coefficient := 1 }]
    b := [{ column := 4075093, coefficient := 1 }, { column := 4075094, coefficient := 1 }]
    c := [{ column := 4078092, coefficient := 1 }] }
,
  { schemaVersion := 1
    rows := 11308137
    columns := 10997363
    sourceRow := 4292366
    a := [{ column := 4078090, coefficient := 18446744069414584320 }, { column := 4078091, coefficient := 18446744069414584314 }, { column := 4078093, coefficient := 1 }]
    b := [{ column := 0, coefficient := 1 }]
    c := [] }
,
  { schemaVersion := 1
    rows := 11308137
    columns := 10997363
    sourceRow := 4292367
    a := [{ column := 4078090, coefficient := 1 }, { column := 4078091, coefficient := 1 }, { column := 4078092, coefficient := 18446744069414584320 }, { column := 4078094, coefficient := 1 }]
    b := [{ column := 0, coefficient := 1 }]
    c := [] }
,
  { schemaVersion := 1
    rows := 11308137
    columns := 10997363
    sourceRow := 4292368
    a := [{ column := 77659, coefficient := 1 }]
    b := [{ column := 4075098, coefficient := 1 }]
    c := [{ column := 4078095, coefficient := 1 }] }
,
  { schemaVersion := 1
    rows := 11308137
    columns := 10997363
    sourceRow := 4292369
    a := [{ column := 77660, coefficient := 1 }]
    b := [{ column := 4075099, coefficient := 1 }]
    c := [{ column := 4078096, coefficient := 1 }] }
,
  { schemaVersion := 1
    rows := 11308137
    columns := 10997363
    sourceRow := 4292370
    a := [{ column := 77659, coefficient := 1 }, { column := 77660, coefficient := 1 }]
    b := [{ column := 4075098, coefficient := 1 }, { column := 4075099, coefficient := 1 }]
    c := [{ column := 4078097, coefficient := 1 }] }
,
  { schemaVersion := 1
    rows := 11308137
    columns := 10997363
    sourceRow := 4292371
    a := [{ column := 4078095, coefficient := 18446744069414584320 }, { column := 4078096, coefficient := 18446744069414584314 }, { column := 4078098, coefficient := 1 }]
    b := [{ column := 0, coefficient := 1 }]
    c := [] }
,
  { schemaVersion := 1
    rows := 11308137
    columns := 10997363
    sourceRow := 4292372
    a := [{ column := 4078095, coefficient := 1 }, { column := 4078096, coefficient := 1 }, { column := 4078097, coefficient := 18446744069414584320 }, { column := 4078099, coefficient := 1 }]
    b := [{ column := 0, coefficient := 1 }]
    c := [] }
,
  { schemaVersion := 1
    rows := 11308137
    columns := 10997363
    sourceRow := 4292373
    a := [{ column := 77661, coefficient := 1 }]
    b := [{ column := 4075103, coefficient := 1 }]
    c := [{ column := 4078100, coefficient := 1 }] }
,
  { schemaVersion := 1
    rows := 11308137
    columns := 10997363
    sourceRow := 4292374
    a := [{ column := 77662, coefficient := 1 }]
    b := [{ column := 4075104, coefficient := 1 }]
    c := [{ column := 4078101, coefficient := 1 }] }
,
  { schemaVersion := 1
    rows := 11308137
    columns := 10997363
    sourceRow := 4292375
    a := [{ column := 77661, coefficient := 1 }, { column := 77662, coefficient := 1 }]
    b := [{ column := 4075103, coefficient := 1 }, { column := 4075104, coefficient := 1 }]
    c := [{ column := 4078102, coefficient := 1 }] }
,
  { schemaVersion := 1
    rows := 11308137
    columns := 10997363
    sourceRow := 4292376
    a := [{ column := 4078100, coefficient := 18446744069414584320 }, { column := 4078101, coefficient := 18446744069414584314 }, { column := 4078103, coefficient := 1 }]
    b := [{ column := 0, coefficient := 1 }]
    c := [] }
,
  { schemaVersion := 1
    rows := 11308137
    columns := 10997363
    sourceRow := 4292377
    a := [{ column := 4078100, coefficient := 1 }, { column := 4078101, coefficient := 1 }, { column := 4078102, coefficient := 18446744069414584320 }, { column := 4078104, coefficient := 1 }]
    b := [{ column := 0, coefficient := 1 }]
    c := [] }
,
  { schemaVersion := 1
    rows := 11308137
    columns := 10997363
    sourceRow := 4292378
    a := [{ column := 77663, coefficient := 1 }]
    b := [{ column := 4075108, coefficient := 1 }]
    c := [{ column := 4078105, coefficient := 1 }] }
,
  { schemaVersion := 1
    rows := 11308137
    columns := 10997363
    sourceRow := 4292379
    a := [{ column := 77664, coefficient := 1 }]
    b := [{ column := 4075109, coefficient := 1 }]
    c := [{ column := 4078106, coefficient := 1 }] }
,
  { schemaVersion := 1
    rows := 11308137
    columns := 10997363
    sourceRow := 4292380
    a := [{ column := 77663, coefficient := 1 }, { column := 77664, coefficient := 1 }]
    b := [{ column := 4075108, coefficient := 1 }, { column := 4075109, coefficient := 1 }]
    c := [{ column := 4078107, coefficient := 1 }] }
,
  { schemaVersion := 1
    rows := 11308137
    columns := 10997363
    sourceRow := 4292381
    a := [{ column := 4078105, coefficient := 18446744069414584320 }, { column := 4078106, coefficient := 18446744069414584314 }, { column := 4078108, coefficient := 1 }]
    b := [{ column := 0, coefficient := 1 }]
    c := [] }
,
  { schemaVersion := 1
    rows := 11308137
    columns := 10997363
    sourceRow := 4292382
    a := [{ column := 4078105, coefficient := 1 }, { column := 4078106, coefficient := 1 }, { column := 4078107, coefficient := 18446744069414584320 }, { column := 4078109, coefficient := 1 }]
    b := [{ column := 0, coefficient := 1 }]
    c := [] }
,
  { schemaVersion := 1
    rows := 11308137
    columns := 10997363
    sourceRow := 4292383
    a := [{ column := 77665, coefficient := 1 }]
    b := [{ column := 4075113, coefficient := 1 }]
    c := [{ column := 4078110, coefficient := 1 }] }
,
  { schemaVersion := 1
    rows := 11308137
    columns := 10997363
    sourceRow := 4292384
    a := [{ column := 77666, coefficient := 1 }]
    b := [{ column := 4075114, coefficient := 1 }]
    c := [{ column := 4078111, coefficient := 1 }] }
,
  { schemaVersion := 1
    rows := 11308137
    columns := 10997363
    sourceRow := 4292385
    a := [{ column := 77665, coefficient := 1 }, { column := 77666, coefficient := 1 }]
    b := [{ column := 4075113, coefficient := 1 }, { column := 4075114, coefficient := 1 }]
    c := [{ column := 4078112, coefficient := 1 }] }
,
  { schemaVersion := 1
    rows := 11308137
    columns := 10997363
    sourceRow := 4292386
    a := [{ column := 4078110, coefficient := 18446744069414584320 }, { column := 4078111, coefficient := 18446744069414584314 }, { column := 4078113, coefficient := 1 }]
    b := [{ column := 0, coefficient := 1 }]
    c := [] }
,
  { schemaVersion := 1
    rows := 11308137
    columns := 10997363
    sourceRow := 4292387
    a := [{ column := 4078110, coefficient := 1 }, { column := 4078111, coefficient := 1 }, { column := 4078112, coefficient := 18446744069414584320 }, { column := 4078114, coefficient := 1 }]
    b := [{ column := 0, coefficient := 1 }]
    c := [] }
,
  { schemaVersion := 1
    rows := 11308137
    columns := 10997363
    sourceRow := 4292388
    a := [{ column := 77667, coefficient := 1 }]
    b := [{ column := 4075118, coefficient := 1 }]
    c := [{ column := 4078115, coefficient := 1 }] }
,
  { schemaVersion := 1
    rows := 11308137
    columns := 10997363
    sourceRow := 4292389
    a := [{ column := 77668, coefficient := 1 }]
    b := [{ column := 4075119, coefficient := 1 }]
    c := [{ column := 4078116, coefficient := 1 }] }
,
  { schemaVersion := 1
    rows := 11308137
    columns := 10997363
    sourceRow := 4292390
    a := [{ column := 77667, coefficient := 1 }, { column := 77668, coefficient := 1 }]
    b := [{ column := 4075118, coefficient := 1 }, { column := 4075119, coefficient := 1 }]
    c := [{ column := 4078117, coefficient := 1 }] }
,
  { schemaVersion := 1
    rows := 11308137
    columns := 10997363
    sourceRow := 4292391
    a := [{ column := 4078115, coefficient := 18446744069414584320 }, { column := 4078116, coefficient := 18446744069414584314 }, { column := 4078118, coefficient := 1 }]
    b := [{ column := 0, coefficient := 1 }]
    c := [] }
,
  { schemaVersion := 1
    rows := 11308137
    columns := 10997363
    sourceRow := 4292392
    a := [{ column := 4078115, coefficient := 1 }, { column := 4078116, coefficient := 1 }, { column := 4078117, coefficient := 18446744069414584320 }, { column := 4078119, coefficient := 1 }]
    b := [{ column := 0, coefficient := 1 }]
    c := [] }
,
  { schemaVersion := 1
    rows := 11308137
    columns := 10997363
    sourceRow := 4292393
    a := [{ column := 77669, coefficient := 1 }]
    b := [{ column := 4075123, coefficient := 1 }]
    c := [{ column := 4078120, coefficient := 1 }] }
,
  { schemaVersion := 1
    rows := 11308137
    columns := 10997363
    sourceRow := 4292394
    a := [{ column := 77670, coefficient := 1 }]
    b := [{ column := 4075124, coefficient := 1 }]
    c := [{ column := 4078121, coefficient := 1 }] }
,
  { schemaVersion := 1
    rows := 11308137
    columns := 10997363
    sourceRow := 4292395
    a := [{ column := 77669, coefficient := 1 }, { column := 77670, coefficient := 1 }]
    b := [{ column := 4075123, coefficient := 1 }, { column := 4075124, coefficient := 1 }]
    c := [{ column := 4078122, coefficient := 1 }] }
,
  { schemaVersion := 1
    rows := 11308137
    columns := 10997363
    sourceRow := 4292396
    a := [{ column := 4078120, coefficient := 18446744069414584320 }, { column := 4078121, coefficient := 18446744069414584314 }, { column := 4078123, coefficient := 1 }]
    b := [{ column := 0, coefficient := 1 }]
    c := [] }
,
  { schemaVersion := 1
    rows := 11308137
    columns := 10997363
    sourceRow := 4292397
    a := [{ column := 4078120, coefficient := 1 }, { column := 4078121, coefficient := 1 }, { column := 4078122, coefficient := 18446744069414584320 }, { column := 4078124, coefficient := 1 }]
    b := [{ column := 0, coefficient := 1 }]
    c := [] }
,
  { schemaVersion := 1
    rows := 11308137
    columns := 10997363
    sourceRow := 4292398
    a := [{ column := 77671, coefficient := 1 }]
    b := [{ column := 4075128, coefficient := 1 }]
    c := [{ column := 4078125, coefficient := 1 }] }
,
  { schemaVersion := 1
    rows := 11308137
    columns := 10997363
    sourceRow := 4292399
    a := [{ column := 77672, coefficient := 1 }]
    b := [{ column := 4075129, coefficient := 1 }]
    c := [{ column := 4078126, coefficient := 1 }] }
,
  { schemaVersion := 1
    rows := 11308137
    columns := 10997363
    sourceRow := 4292400
    a := [{ column := 77671, coefficient := 1 }, { column := 77672, coefficient := 1 }]
    b := [{ column := 4075128, coefficient := 1 }, { column := 4075129, coefficient := 1 }]
    c := [{ column := 4078127, coefficient := 1 }] }
,
  { schemaVersion := 1
    rows := 11308137
    columns := 10997363
    sourceRow := 4292401
    a := [{ column := 4078125, coefficient := 18446744069414584320 }, { column := 4078126, coefficient := 18446744069414584314 }, { column := 4078128, coefficient := 1 }]
    b := [{ column := 0, coefficient := 1 }]
    c := [] }
,
  { schemaVersion := 1
    rows := 11308137
    columns := 10997363
    sourceRow := 4292402
    a := [{ column := 4078125, coefficient := 1 }, { column := 4078126, coefficient := 1 }, { column := 4078127, coefficient := 18446744069414584320 }, { column := 4078129, coefficient := 1 }]
    b := [{ column := 0, coefficient := 1 }]
    c := [] }
,
  { schemaVersion := 1
    rows := 11308137
    columns := 10997363
    sourceRow := 4292403
    a := [{ column := 77673, coefficient := 1 }]
    b := [{ column := 4075133, coefficient := 1 }]
    c := [{ column := 4078130, coefficient := 1 }] }
,
  { schemaVersion := 1
    rows := 11308137
    columns := 10997363
    sourceRow := 4292404
    a := [{ column := 77674, coefficient := 1 }]
    b := [{ column := 4075134, coefficient := 1 }]
    c := [{ column := 4078131, coefficient := 1 }] }
,
  { schemaVersion := 1
    rows := 11308137
    columns := 10997363
    sourceRow := 4292405
    a := [{ column := 77673, coefficient := 1 }, { column := 77674, coefficient := 1 }]
    b := [{ column := 4075133, coefficient := 1 }, { column := 4075134, coefficient := 1 }]
    c := [{ column := 4078132, coefficient := 1 }] }
,
  { schemaVersion := 1
    rows := 11308137
    columns := 10997363
    sourceRow := 4292406
    a := [{ column := 4078130, coefficient := 18446744069414584320 }, { column := 4078131, coefficient := 18446744069414584314 }, { column := 4078133, coefficient := 1 }]
    b := [{ column := 0, coefficient := 1 }]
    c := [] }
,
  { schemaVersion := 1
    rows := 11308137
    columns := 10997363
    sourceRow := 4292407
    a := [{ column := 4078130, coefficient := 1 }, { column := 4078131, coefficient := 1 }, { column := 4078132, coefficient := 18446744069414584320 }, { column := 4078134, coefficient := 1 }]
    b := [{ column := 0, coefficient := 1 }]
    c := [] }
,
  { schemaVersion := 1
    rows := 11308137
    columns := 10997363
    sourceRow := 4292408
    a := [{ column := 77675, coefficient := 1 }]
    b := [{ column := 4075138, coefficient := 1 }]
    c := [{ column := 4078135, coefficient := 1 }] }
,
  { schemaVersion := 1
    rows := 11308137
    columns := 10997363
    sourceRow := 4292409
    a := [{ column := 77676, coefficient := 1 }]
    b := [{ column := 4075139, coefficient := 1 }]
    c := [{ column := 4078136, coefficient := 1 }] }
,
  { schemaVersion := 1
    rows := 11308137
    columns := 10997363
    sourceRow := 4292410
    a := [{ column := 77675, coefficient := 1 }, { column := 77676, coefficient := 1 }]
    b := [{ column := 4075138, coefficient := 1 }, { column := 4075139, coefficient := 1 }]
    c := [{ column := 4078137, coefficient := 1 }] }
,
  { schemaVersion := 1
    rows := 11308137
    columns := 10997363
    sourceRow := 4292411
    a := [{ column := 4078135, coefficient := 18446744069414584320 }, { column := 4078136, coefficient := 18446744069414584314 }, { column := 4078138, coefficient := 1 }]
    b := [{ column := 0, coefficient := 1 }]
    c := [] }
,
  { schemaVersion := 1
    rows := 11308137
    columns := 10997363
    sourceRow := 4292412
    a := [{ column := 4078135, coefficient := 1 }, { column := 4078136, coefficient := 1 }, { column := 4078137, coefficient := 18446744069414584320 }, { column := 4078139, coefficient := 1 }]
    b := [{ column := 0, coefficient := 1 }]
    c := [] }
,
  { schemaVersion := 1
    rows := 11308137
    columns := 10997363
    sourceRow := 4292413
    a := [{ column := 77677, coefficient := 1 }]
    b := [{ column := 4075143, coefficient := 1 }]
    c := [{ column := 4078140, coefficient := 1 }] }
,
  { schemaVersion := 1
    rows := 11308137
    columns := 10997363
    sourceRow := 4292414
    a := [{ column := 77678, coefficient := 1 }]
    b := [{ column := 4075144, coefficient := 1 }]
    c := [{ column := 4078141, coefficient := 1 }] }
,
  { schemaVersion := 1
    rows := 11308137
    columns := 10997363
    sourceRow := 4292415
    a := [{ column := 77677, coefficient := 1 }, { column := 77678, coefficient := 1 }]
    b := [{ column := 4075143, coefficient := 1 }, { column := 4075144, coefficient := 1 }]
    c := [{ column := 4078142, coefficient := 1 }] }
,
  { schemaVersion := 1
    rows := 11308137
    columns := 10997363
    sourceRow := 4292416
    a := [{ column := 4078140, coefficient := 18446744069414584320 }, { column := 4078141, coefficient := 18446744069414584314 }, { column := 4078143, coefficient := 1 }]
    b := [{ column := 0, coefficient := 1 }]
    c := [] }
,
  { schemaVersion := 1
    rows := 11308137
    columns := 10997363
    sourceRow := 4292417
    a := [{ column := 4078140, coefficient := 1 }, { column := 4078141, coefficient := 1 }, { column := 4078142, coefficient := 18446744069414584320 }, { column := 4078144, coefficient := 1 }]
    b := [{ column := 0, coefficient := 1 }]
    c := [] }
,
  { schemaVersion := 1
    rows := 11308137
    columns := 10997363
    sourceRow := 4292418
    a := [{ column := 77679, coefficient := 1 }]
    b := [{ column := 4075148, coefficient := 1 }]
    c := [{ column := 4078145, coefficient := 1 }] }
,
  { schemaVersion := 1
    rows := 11308137
    columns := 10997363
    sourceRow := 4292419
    a := [{ column := 77680, coefficient := 1 }]
    b := [{ column := 4075149, coefficient := 1 }]
    c := [{ column := 4078146, coefficient := 1 }] }
,
  { schemaVersion := 1
    rows := 11308137
    columns := 10997363
    sourceRow := 4292420
    a := [{ column := 77679, coefficient := 1 }, { column := 77680, coefficient := 1 }]
    b := [{ column := 4075148, coefficient := 1 }, { column := 4075149, coefficient := 1 }]
    c := [{ column := 4078147, coefficient := 1 }] }
,
  { schemaVersion := 1
    rows := 11308137
    columns := 10997363
    sourceRow := 4292421
    a := [{ column := 4078145, coefficient := 18446744069414584320 }, { column := 4078146, coefficient := 18446744069414584314 }, { column := 4078148, coefficient := 1 }]
    b := [{ column := 0, coefficient := 1 }]
    c := [] }
,
  { schemaVersion := 1
    rows := 11308137
    columns := 10997363
    sourceRow := 4292422
    a := [{ column := 4078145, coefficient := 1 }, { column := 4078146, coefficient := 1 }, { column := 4078147, coefficient := 18446744069414584320 }, { column := 4078149, coefficient := 1 }]
    b := [{ column := 0, coefficient := 1 }]
    c := [] }
,
  { schemaVersion := 1
    rows := 11308137
    columns := 10997363
    sourceRow := 4292423
    a := [{ column := 77681, coefficient := 1 }]
    b := [{ column := 4075153, coefficient := 1 }]
    c := [{ column := 4078150, coefficient := 1 }] }
,
  { schemaVersion := 1
    rows := 11308137
    columns := 10997363
    sourceRow := 4292424
    a := [{ column := 77682, coefficient := 1 }]
    b := [{ column := 4075154, coefficient := 1 }]
    c := [{ column := 4078151, coefficient := 1 }] }
,
  { schemaVersion := 1
    rows := 11308137
    columns := 10997363
    sourceRow := 4292425
    a := [{ column := 77681, coefficient := 1 }, { column := 77682, coefficient := 1 }]
    b := [{ column := 4075153, coefficient := 1 }, { column := 4075154, coefficient := 1 }]
    c := [{ column := 4078152, coefficient := 1 }] }
,
  { schemaVersion := 1
    rows := 11308137
    columns := 10997363
    sourceRow := 4292426
    a := [{ column := 4078150, coefficient := 18446744069414584320 }, { column := 4078151, coefficient := 18446744069414584314 }, { column := 4078153, coefficient := 1 }]
    b := [{ column := 0, coefficient := 1 }]
    c := [] }
,
  { schemaVersion := 1
    rows := 11308137
    columns := 10997363
    sourceRow := 4292427
    a := [{ column := 4078150, coefficient := 1 }, { column := 4078151, coefficient := 1 }, { column := 4078152, coefficient := 18446744069414584320 }, { column := 4078154, coefficient := 1 }]
    b := [{ column := 0, coefficient := 1 }]
    c := [] }
,
  { schemaVersion := 1
    rows := 11308137
    columns := 10997363
    sourceRow := 4292428
    a := [{ column := 77683, coefficient := 1 }]
    b := [{ column := 4075158, coefficient := 1 }]
    c := [{ column := 4078155, coefficient := 1 }] }
,
  { schemaVersion := 1
    rows := 11308137
    columns := 10997363
    sourceRow := 4292429
    a := [{ column := 77684, coefficient := 1 }]
    b := [{ column := 4075159, coefficient := 1 }]
    c := [{ column := 4078156, coefficient := 1 }] }
,
  { schemaVersion := 1
    rows := 11308137
    columns := 10997363
    sourceRow := 4292430
    a := [{ column := 77683, coefficient := 1 }, { column := 77684, coefficient := 1 }]
    b := [{ column := 4075158, coefficient := 1 }, { column := 4075159, coefficient := 1 }]
    c := [{ column := 4078157, coefficient := 1 }] }
,
  { schemaVersion := 1
    rows := 11308137
    columns := 10997363
    sourceRow := 4292431
    a := [{ column := 4078155, coefficient := 18446744069414584320 }, { column := 4078156, coefficient := 18446744069414584314 }, { column := 4078158, coefficient := 1 }]
    b := [{ column := 0, coefficient := 1 }]
    c := [] }
,
  { schemaVersion := 1
    rows := 11308137
    columns := 10997363
    sourceRow := 4292432
    a := [{ column := 4078155, coefficient := 1 }, { column := 4078156, coefficient := 1 }, { column := 4078157, coefficient := 18446744069414584320 }, { column := 4078159, coefficient := 1 }]
    b := [{ column := 0, coefficient := 1 }]
    c := [] }
,
  { schemaVersion := 1
    rows := 11308137
    columns := 10997363
    sourceRow := 4292433
    a := [{ column := 77685, coefficient := 1 }]
    b := [{ column := 4075163, coefficient := 1 }]
    c := [{ column := 4078160, coefficient := 1 }] }
,
  { schemaVersion := 1
    rows := 11308137
    columns := 10997363
    sourceRow := 4292434
    a := [{ column := 77686, coefficient := 1 }]
    b := [{ column := 4075164, coefficient := 1 }]
    c := [{ column := 4078161, coefficient := 1 }] }
,
  { schemaVersion := 1
    rows := 11308137
    columns := 10997363
    sourceRow := 4292435
    a := [{ column := 77685, coefficient := 1 }, { column := 77686, coefficient := 1 }]
    b := [{ column := 4075163, coefficient := 1 }, { column := 4075164, coefficient := 1 }]
    c := [{ column := 4078162, coefficient := 1 }] }
,
  { schemaVersion := 1
    rows := 11308137
    columns := 10997363
    sourceRow := 4292436
    a := [{ column := 4078160, coefficient := 18446744069414584320 }, { column := 4078161, coefficient := 18446744069414584314 }, { column := 4078163, coefficient := 1 }]
    b := [{ column := 0, coefficient := 1 }]
    c := [] }
,
  { schemaVersion := 1
    rows := 11308137
    columns := 10997363
    sourceRow := 4292437
    a := [{ column := 4078160, coefficient := 1 }, { column := 4078161, coefficient := 1 }, { column := 4078162, coefficient := 18446744069414584320 }, { column := 4078164, coefficient := 1 }]
    b := [{ column := 0, coefficient := 1 }]
    c := [] }
,
  { schemaVersion := 1
    rows := 11308137
    columns := 10997363
    sourceRow := 4292438
    a := [{ column := 77687, coefficient := 1 }]
    b := [{ column := 4075168, coefficient := 1 }]
    c := [{ column := 4078165, coefficient := 1 }] }
,
  { schemaVersion := 1
    rows := 11308137
    columns := 10997363
    sourceRow := 4292439
    a := [{ column := 77688, coefficient := 1 }]
    b := [{ column := 4075169, coefficient := 1 }]
    c := [{ column := 4078166, coefficient := 1 }] }
,
  { schemaVersion := 1
    rows := 11308137
    columns := 10997363
    sourceRow := 4292440
    a := [{ column := 77687, coefficient := 1 }, { column := 77688, coefficient := 1 }]
    b := [{ column := 4075168, coefficient := 1 }, { column := 4075169, coefficient := 1 }]
    c := [{ column := 4078167, coefficient := 1 }] }
,
  { schemaVersion := 1
    rows := 11308137
    columns := 10997363
    sourceRow := 4292441
    a := [{ column := 4078165, coefficient := 18446744069414584320 }, { column := 4078166, coefficient := 18446744069414584314 }, { column := 4078168, coefficient := 1 }]
    b := [{ column := 0, coefficient := 1 }]
    c := [] }
,
  { schemaVersion := 1
    rows := 11308137
    columns := 10997363
    sourceRow := 4292442
    a := [{ column := 4078165, coefficient := 1 }, { column := 4078166, coefficient := 1 }, { column := 4078167, coefficient := 18446744069414584320 }, { column := 4078169, coefficient := 1 }]
    b := [{ column := 0, coefficient := 1 }]
    c := [] }
,
  { schemaVersion := 1
    rows := 11308137
    columns := 10997363
    sourceRow := 4292443
    a := [{ column := 77689, coefficient := 1 }]
    b := [{ column := 4075173, coefficient := 1 }]
    c := [{ column := 4078170, coefficient := 1 }] }
]

end Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.SourceRows.Chunk37
