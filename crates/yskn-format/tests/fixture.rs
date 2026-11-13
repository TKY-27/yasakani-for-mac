use serde_json::Value;
use sha2::{Digest, Sha256};
use yskn_format::{Container, SignRule, Tensor, signs};

const FIXTURE: &[u8] = include_bytes!("../../../tools/testdata/tiny.yskn");
const EXPECTED: &str = include_str!("../../../tools/testdata/tiny_expected.json");

fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn float_digest(values: &[f32]) -> String {
    let bytes: Vec<_> = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    digest(&bytes)
}

fn hadamard(values: &mut [f32]) {
    assert_eq!(values.len(), 512);
    for stride in [1, 2, 4, 8, 16, 32, 64, 128, 256] {
        for base in (0..512).step_by(2 * stride) {
            for offset in 0..stride {
                let lower = values[base + offset];
                let upper = values[base + offset + stride];
                values[base + offset] = lower + upper;
                values[base + offset + stride] = lower - upper;
            }
        }
    }
    for value in values {
        *value *= f32::from_bits(0x3d35_04f3);
    }
}

fn original_coordinates(tensor: &Tensor, mut values: Vec<f32>) -> Vec<f32> {
    let cols = tensor.cols();
    for row in values.chunks_exact_mut(cols) {
        for block in row.chunks_exact_mut(512) {
            hadamard(block);
        }
        for (value, &sign) in row.iter_mut().zip(tensor.input_signs()) {
            *value *= f32::from(sign);
        }
    }
    if let Some(signs) = tensor.output_signs() {
        let mut column = [0.0_f32; 512];
        for start in (0..tensor.physical_rows()).step_by(512) {
            for col in 0..cols {
                for (offset, value) in column.iter_mut().enumerate() {
                    *value = values[(start + offset) * cols + col];
                }
                hadamard(&mut column);
                for (offset, &value) in column.iter().enumerate() {
                    values[(start + offset) * cols + col] =
                        value * f32::from(signs[start + offset]);
                }
            }
        }
    }
    values
}

#[test]
fn published_tiny_container_matches_every_expected_field() {
    assert_eq!(
        digest(FIXTURE),
        "3f4a48866f6a1751beabc3b9a8d06088672a3db0aa0a4c3f96e1877cb448d393"
    );
    let expected: Value = serde_json::from_str(EXPECTED).unwrap();
    let mut container = Container::from_bytes(FIXTURE.to_vec()).unwrap();
    let header = container.header();
    let numbers = [
        u64::from(header.version),
        header.header_bytes,
        header.data_offset,
        header.file_bytes,
    ];
    for (number, expected) in numbers.iter().zip(expected["header"].as_array().unwrap()) {
        assert_eq!(*number, expected.as_u64().unwrap());
    }
    for case in expected["signs"].as_array().unwrap() {
        let rule = match case[0].as_str().unwrap() {
            "splitmix64" => SignRule::SplitMix64,
            "mulberry32_7919_13" => SignRule::Mulberry32,
            _ => unreachable!(),
        };
        let values = signs(
            rule,
            case[1].as_u64().unwrap(),
            case[2].as_u64().unwrap(),
            case[3].as_u64().unwrap() as usize,
        )
        .unwrap();
        let bytes: Vec<_> = values.iter().map(|&value| value as u8).collect();
        assert_eq!(digest(&bytes), case[4].as_str().unwrap());
    }
    for (name, expected) in expected["tensors"].as_object().unwrap() {
        let tensor = container.load_tensor(name).unwrap();
        let mut state_bytes = Vec::new();
        let mut rotated = Vec::new();
        for row in 0..tensor.rows() {
            for state in tensor.row_states(row).unwrap() {
                state_bytes.extend_from_slice(&state.to_le_bytes());
            }
            rotated.extend(tensor.rotated_row(row).unwrap());
        }
        assert_eq!(
            digest(&state_bytes),
            expected["states_sha256"].as_str().unwrap(),
            "{name} states"
        );
        assert_eq!(
            float_digest(&rotated),
            expected["rotated_sha256"].as_str().unwrap(),
            "{name} rotated values"
        );

        let weights = original_coordinates(&tensor, rotated);
        for (position, expected_value) in expected["sample_at"]
            .as_array()
            .unwrap()
            .iter()
            .zip(expected["sample_values"].as_array().unwrap())
        {
            let row = position[0].as_u64().unwrap() as usize;
            let col = position[1].as_u64().unwrap() as usize;
            let expected_value = expected_value.as_f64().unwrap() as f32;
            // FORMAT fixes these coordinates mathematically, not the transform's
            // intermediate rounding. Its explicitly bit-defined hashes stay exact.
            assert!(
                (weights[row * tensor.cols() + col] - expected_value).abs() < 1e-7,
                "{name} sample ({row},{col})"
            );
        }
        let square_norm: f64 = weights
            .iter()
            .map(|&value| f64::from(value) * f64::from(value))
            .sum();
        let expected_norm = expected["frobenius"].as_f64().unwrap();
        assert!(
            (square_norm - expected_norm * expected_norm).abs() < 2e-7 * square_norm,
            "{name} Frobenius norm"
        );

        let table_max = tensor
            .table()
            .iter()
            .map(|value| value.abs())
            .fold(0.0_f32, f32::max);
        let table_scale = table_max / 127.0;
        let quantized_table: Vec<_> = tensor
            .table()
            .iter()
            .map(|&value| {
                (value / table_scale).round_ties_even().clamp(-127.0, 127.0) * table_scale
            })
            .collect();
        let mut quantized = Vec::new();
        for row in 0..tensor.rows() {
            for state in tensor.row_states(row).unwrap() {
                for component in 0..2 {
                    quantized.push(
                        tensor.scales()[row] * quantized_table[usize::from(state) * 2 + component],
                    );
                }
            }
        }
        let quantized = original_coordinates(&tensor, quantized);
        let maximum_difference = weights
            .iter()
            .zip(quantized)
            .map(|(&a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        let expected_bound = expected["engine_table_bound"].as_f64().unwrap() as f32;
        assert!(
            maximum_difference <= expected_bound,
            "{name} engine table bound: {maximum_difference} versus {expected_bound}"
        );

        for row in expected["naive_rows"].as_array().unwrap() {
            let row = row.as_u64().unwrap() as usize;
            let states = tensor.row_states(row).unwrap();
            for sequence in 0..tensor.cols() / 256 {
                for step in 0..128 {
                    assert_eq!(
                        tensor.state(row, sequence, step).unwrap(),
                        states[sequence * 128 + step]
                    );
                }
            }
        }
        assert!(tensor.state(tensor.physical_rows(), 0, 0).is_err());
        assert!(tensor.state(0, tensor.cols() / 256, 0).is_err());
        assert!(tensor.state(0, 0, 128).is_err());
    }
    for (name, expected) in expected["vectors"].as_object().unwrap() {
        let vector = container.load_vector(name).unwrap();
        assert_eq!(
            float_digest(&vector.values),
            expected["sha256"].as_str().unwrap(),
            "{name}"
        );
    }
}
