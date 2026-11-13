use serde_json::{Value, json};
use yskn_format::{Container, Error};

const FIXTURE: &[u8] = include_bytes!("../../../tools/testdata/tiny.yskn");

fn changed_header(change: impl FnOnce(&mut Value)) -> Vec<u8> {
    let mut bytes = FIXTURE.to_vec();
    let length = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
    let mut header: Value = serde_json::from_slice(&bytes[32..32 + length]).unwrap();
    change(&mut header);
    let header = serde_json::to_vec(&header).unwrap();
    assert!(header.len() < 4096 - 32);
    bytes[32..4096].fill(0);
    bytes[32..32 + header.len()].copy_from_slice(&header);
    bytes[8..16].copy_from_slice(&(header.len() as u64).to_le_bytes());
    bytes
}

#[test]
fn invalid_fixed_headers_and_truncated_data_return_errors() {
    for length in [0, 4, 31, 32, 4096, FIXTURE.len() - 1] {
        assert!(Container::from_bytes(FIXTURE[..length].to_vec()).is_err());
    }
    for (offset, value) in [(8, u64::MAX), (16, 0), (16, u64::MAX), (24, 32)] {
        let mut bytes = FIXTURE.to_vec();
        bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
        assert!(Container::from_bytes(bytes).is_err());
    }
    let mut bytes = FIXTURE.to_vec();
    bytes[0] = b'X';
    assert!(Container::from_bytes(bytes).is_err());
    let mut bytes = FIXTURE.to_vec();
    bytes[4] = 2;
    assert!(matches!(
        Container::from_bytes(bytes),
        Err(Error::Unsupported(_))
    ));
    let mut bytes = FIXTURE.to_vec();
    bytes[4095] = 1;
    assert!(Container::from_bytes(bytes).is_err());
}

#[test]
fn every_payload_range_is_validated_before_tensor_loading() {
    for key in ["stream_off", "lut_off", "scale_off"] {
        for offset in [0, FIXTURE.len() as u64, u64::MAX] {
            let bytes = changed_header(|header| header["tensors"][0][key] = json!(offset));
            assert!(Container::from_bytes(bytes).is_err(), "{key}={offset}");
        }
    }
    for key in ["stream_bytes", "cols", "physical_rows"] {
        let bytes = changed_header(|header| header["tensors"][0][key] = json!(u64::MAX));
        assert!(Container::from_bytes(bytes).is_err(), "{key}");
    }
    for key in ["off", "n"] {
        let bytes = changed_header(|header| header["vectors"][0][key] = json!(u64::MAX));
        assert!(Container::from_bytes(bytes).is_err(), "vector {key}");
    }
}

#[test]
fn unknown_or_undefined_encodings_return_errors() {
    for (key, value) in [
        ("l", json!(13)),
        ("d", json!(1)),
        ("t", json!(128)),
        ("bits", json!(3)),
        ("bits", json!(9)),
        ("start", json!("carry")),
        ("layout", json!("unknown")),
        ("rar", json!({})),
        ("bias_off", json!(4096)),
    ] {
        let bytes = changed_header(|header| header["tensors"][0][key] = value);
        assert!(
            matches!(Container::from_bytes(bytes), Err(Error::Unsupported(_))),
            "{key}"
        );
    }
    for side in ["rot_in", "rot_out"] {
        let bytes = changed_header(|header| header["tensors"][0][side]["kind"] = json!("AYA"));
        assert!(matches!(
            Container::from_bytes(bytes),
            Err(Error::Unsupported(_))
        ));
    }
    let bytes = changed_header(|header| header["vectors"][0]["dtype"] = json!("float32"));
    assert!(matches!(
        Container::from_bytes(bytes),
        Err(Error::Unsupported(_))
    ));
}

#[test]
fn invalid_shapes_rotation_dimensions_and_duplicate_names_are_rejected() {
    for (key, value) in [("rows", 0), ("cols", 513), ("physical_rows", 511)] {
        let bytes = changed_header(|header| header["tensors"][0][key] = json!(value));
        assert!(Container::from_bytes(bytes).is_err());
    }
    let bytes = changed_header(|header| header["tensors"][0]["rot_in"]["m"] = json!(2));
    assert!(Container::from_bytes(bytes).is_err());
    for kind in ["H512", "none"] {
        let bytes = changed_header(|header| {
            header["tensors"][0]["rot_out"]["kind"] = json!(kind);
            header["tensors"][0]["rot_out"]["qm_off"] = json!(u64::MAX);
        });
        assert!(Container::from_bytes(bytes).is_err(), "{kind} qm_off");
    }
    let bytes = changed_header(|header| header["vectors"][0]["name"] = json!("tiny.body6"));
    assert!(Container::from_bytes(bytes).is_err());
    let bytes = changed_header(|header| header["tensors"][1]["name"] = json!("tiny.body6"));
    assert!(Container::from_bytes(bytes).is_err());
}

#[test]
fn documented_defaults_preserve_the_fixture_values() {
    let mut original = Container::from_bytes(FIXTURE.to_vec()).unwrap();
    let body = original.load_tensor("tiny.body6").unwrap();
    let half = original.load_vector("tiny.half").unwrap();
    let bytes = changed_header(|header| {
        for key in [
            "physical_rows",
            "l",
            "d",
            "t",
            "start",
            "layout",
            "rot_in",
            "rot_out",
        ] {
            header["tensors"][0].as_object_mut().unwrap().remove(key);
        }
        header["vectors"][1]
            .as_object_mut()
            .unwrap()
            .remove("dtype");
        header["tensors"][1]["rot_in"]
            .as_object_mut()
            .unwrap()
            .remove("rule");
    });
    let mut defaults = Container::from_bytes(bytes).unwrap();
    let restored = defaults.load_tensor("tiny.body6").unwrap();
    assert_eq!(restored.input_signs(), body.input_signs());
    assert_eq!(restored.output_signs(), body.output_signs());
    assert_eq!(
        restored.rotated_row(42).unwrap(),
        body.rotated_row(42).unwrap()
    );
    assert_eq!(
        defaults.load_vector("tiny.half").unwrap().values,
        half.values
    );
    assert_eq!(
        defaults.load_tensor("tiny.body8").unwrap().input_signs(),
        original.load_tensor("tiny.body8").unwrap().input_signs()
    );
    assert!(defaults.load_tensor("absent").is_err());
    assert!(defaults.load_vector("absent").is_err());
}

fn known_codes(bits: usize, layout: &str) -> Vec<u8> {
    let rows = 16;
    let sequences = 2;
    let stream_bytes = rows * sequences * bits * 16;
    let stream_offset = 2048;
    let table_offset = stream_offset + stream_bytes;
    let scale_offset = table_offset + 16_384;
    let file_bytes = scale_offset + rows * 2;
    let header = json!({
        "model": {},
        "tensors": [{"name":"codes","role":"head","rows":rows,"cols":512,
            "bits":bits,"layout":layout,"stream_off":stream_offset,"stream_bytes":stream_bytes,
            "lut_off":table_offset,"scale_off":scale_offset}],
        "vectors": []
    });
    let header = serde_json::to_vec(&header).unwrap();
    let mut bytes = vec![0; file_bytes];
    bytes[0..4].copy_from_slice(b"YSKN");
    bytes[4..8].copy_from_slice(&1_u32.to_le_bytes());
    bytes[8..16].copy_from_slice(&(header.len() as u64).to_le_bytes());
    bytes[16..24].copy_from_slice(&(stream_offset as u64).to_le_bytes());
    bytes[24..32].copy_from_slice(&(file_bytes as u64).to_le_bytes());
    bytes[32..32 + header.len()].copy_from_slice(&header);
    for row in 0..rows {
        for sequence in 0..sequences {
            for step in 0..128 {
                let code = (row * 23 + sequence * 31 + step * 13) & ((1 << bits) - 1);
                for bit in 0..bits {
                    let position = step * bits + bit;
                    let word = position / 32;
                    let word_index = if layout == "row" {
                        (row * sequences + sequence) * (4 * bits) + word
                    } else {
                        let index = sequence * rows + row;
                        ((index / 32 * bits + word / 4) * 32 + index % 32) * 4 + word % 4
                    };
                    let byte = stream_offset + word_index * 4 + position % 32 / 8;
                    bytes[byte] |= (((code >> bit) & 1) << (position % 8)) as u8;
                }
            }
        }
    }
    bytes
}

#[test]
fn all_supported_bit_widths_follow_the_rolling_state_equation_in_both_layouts() {
    for bits in 4..=8 {
        for layout in ["row", "warp"] {
            let mut container = Container::from_bytes(known_codes(bits, layout)).unwrap();
            let tensor = container.load_tensor("codes").unwrap();
            for row in 0..tensor.rows() {
                for sequence in 0..2 {
                    let mut state = 0_u16;
                    for step in 0..128 {
                        let code =
                            ((row * 23 + sequence * 31 + step * 13) & ((1 << bits) - 1)) as u16;
                        state = (state >> bits) | (code << (12 - bits));
                        assert_eq!(tensor.state(row, sequence, step).unwrap(), state);
                    }
                }
            }
        }
    }
}
