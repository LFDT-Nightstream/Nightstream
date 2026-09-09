
The public API cannot replace one bound component while keeping the other
components and their cached identities.

```compile_fail,E0616
use neo_ajtai::AjtaiSModule;
use neo_fold_clean::Preprocessing;

fn replace_setup(mut context: Preprocessing, setup: AjtaiSModule) {
    context.log = setup;
}
```

```compile_fail,E0616
use neo_fold_clean::{Params, Preprocessing};

fn replace_parameters(mut context: Preprocessing, params: Params) {
    context.params = params;
}
```

```compile_fail,E0616
use neo_fold_clean::{Preprocessing, VerifierKey};

fn replace_key(mut context: Preprocessing, key: VerifierKey) {
    context.vk = key;
}
```

```compile_fail,E0616
use neo_fold_clean::Preprocessing;

fn replace_public_input_policy(mut context: Preprocessing) {
    context.public_input_len = Some(0);
}
```
