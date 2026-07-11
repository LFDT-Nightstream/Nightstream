#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WasmLookupArity {
    Unary,
    Binary,
    Tuple(u8),
}

impl WasmLookupArity {
    pub fn width(self) -> u8 {
        match self {
            Self::Unary => 1,
            Self::Binary => 2,
            Self::Tuple(width) => width,
        }
    }
}
