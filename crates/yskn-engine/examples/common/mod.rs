use yskn_engine::{Error, Result};

pub fn activations(k: usize, m: usize) -> (Vec<i8>, Vec<i32>) {
    let mut state = 0x87ab_1234_5678_9def_u64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let a = (0..m * k)
        .map(|_| ((next() % 255) as i16 - 127) as i8)
        .collect();
    let s = (0..m * (k / 512))
        .map(|_| (next() % 16_777_217) as i32)
        .collect();
    (a, s)
}

pub fn missing_path() -> Error {
    Error::InvalidInput(
        "usage: <example> <model.yskn> [--kernel portable|neon|dotprod] [--threads N] [--m 1,16]"
            .into(),
    )
}

pub fn argument_value(args: &mut impl Iterator<Item = String>) -> Result<String> {
    args.next().ok_or_else(missing_path)
}
