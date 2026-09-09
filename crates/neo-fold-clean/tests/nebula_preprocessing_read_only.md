
The lifecycle context cannot be replaced independently of its relation and plan.

```compile_fail,E0616
use neo_fold_clean::{frontends::nebula::NebulaFPrimePreprocessing, Preprocessing};

fn replace_context(mut wrapper: NebulaFPrimePreprocessing, replacement: Preprocessing) {
    wrapper.prep = replacement;
}
```

```no_run
use neo_fold_clean::{frontends::nebula::NebulaFPrimePreprocessing, Preprocessing};

fn context(wrapper: &NebulaFPrimePreprocessing) -> &Preprocessing {
    wrapper.preprocessing()
}
```
