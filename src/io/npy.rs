//! Minimal `.npy` v1/v2 reader for the checkpoint stats files.
//!
//! The stats directory holds six files per bundle
//! (`{motion,pre_quantization,post_quantization}/{mean,std}.npy`), all 1-D or
//! 2-D `<f4` arrays written by numpy. Rather than pull `ndarray` in for six
//! files, this is a hand-rolled reader in the spirit of ardy-mlx's
//! dependency-free `safetensors.py`.

use std::path::Path;

use crate::error::{Error, Result};

/// One numpy array: row-major shape plus flat f32 payload.
#[derive(Clone, Debug, PartialEq)]
pub struct NpyArray {
    /// Row-major shape; `shape[0]` is the outermost dimension.
    pub shape: Vec<usize>,
    /// Flat row-major payload, `shape` product long.
    pub data: Vec<f32>,
}

impl NpyArray {
    /// Byte length of one row (the last dimension).
    pub fn row_len(&self) -> usize {
        *self.shape.last().unwrap_or(&1)
    }

    /// Number of rows (all dimensions but the last folded together; a 1-D
    /// array is one row).
    pub fn rows(&self) -> usize {
        if self.shape.len() <= 1 {
            1
        } else {
            self.shape[..self.shape.len() - 1].iter().product()
        }
    }
}

/// Read an `.npy` file as an f32 array.
///
/// Accepts `|b1`, `|i1`, `|u1`, `<i2`, `<f2`/`<f4`/`<f8` style descriptors
/// that losslessly widen to f32; anything else (complex, strings, big-endian)
/// is an error. `fortran_order` must be false — the stats files are written
/// by `numpy.save` of contiguous arrays.
pub fn read_npy_f32(path: &Path) -> Result<NpyArray> {
    let bytes = std::fs::read(path).map_err(|source| Error::Io {
        path: path.to_path_buf(),
        source,
    })?;
    parse_npy_f32(path, &bytes)
}

/// Parse `.npy` bytes (see [`read_npy_f32`]).
pub fn parse_npy_f32(path: &Path, bytes: &[u8]) -> Result<NpyArray> {
    if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" {
        return Err(Error::Npy(format!(
            "{}: not a numpy .npy file",
            path.display()
        )));
    }
    let major = bytes[6];
    let header_len: usize = match major {
        1 => u16::from_le_bytes([bytes[8], bytes[9]]) as usize,
        2 | 3 => {
            if bytes.len() < 12 {
                return Err(Error::Npy(format!("{}: truncated header", path.display())));
            }
            u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize
        }
        v => {
            return Err(Error::Npy(format!(
                "{}: unsupported .npy version {v}",
                path.display()
            )));
        }
    };
    let header_start = if major == 1 { 10 } else { 12 };
    let header_end = header_start + header_len;
    if bytes.len() < header_end {
        return Err(Error::Npy(format!("{}: truncated header", path.display())));
    }
    let header = std::str::from_utf8(&bytes[header_start..header_end])
        .map_err(|e| Error::Npy(format!("{}: header is not UTF-8: {e}", path.display())))?;

    let (descr, fortran_order, shape) = parse_header_dict(path, header)?;
    if fortran_order {
        return Err(Error::Npy(format!(
            "{}: fortran_order arrays are not supported",
            path.display()
        )));
    }

    let (item_size, convert): (usize, fn(&[u8]) -> f32) = match descr.as_str() {
        "|b1" | "?" => (1, |b| if b[0] != 0 { 1.0 } else { 0.0 }),
        "|i1" => (1, |b| b[0] as i8 as f32),
        "|u1" => (1, |b| b[0] as f32),
        "<i2" => (2, |b| i16::from_le_bytes([b[0], b[1]]) as f32),
        "<u2" => (2, |b| u16::from_le_bytes([b[0], b[1]]) as f32),
        "<f2" => (2, |b| half_bits_to_f32(u16::from_le_bytes([b[0], b[1]]))),
        "<i4" => (4, |b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f32),
        "<f4" => (4, |b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])),
        "<f8" => (8, |b| {
            f64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]) as f32
        }),
        other => {
            return Err(Error::Npy(format!(
                "{}: dtype {other:?} cannot be read as f32",
                path.display()
            )));
        }
    };

    let count: usize = shape.iter().product();
    let payload = &bytes[header_end..];
    if payload.len() < count * item_size {
        return Err(Error::Npy(format!(
            "{}: payload is {} bytes, shape {shape:?} x {descr} needs {}",
            path.display(),
            payload.len(),
            count * item_size
        )));
    }
    let mut data = Vec::with_capacity(count);
    for i in 0..count {
        data.push(convert(&payload[i * item_size..(i + 1) * item_size]));
    }
    Ok(NpyArray { shape, data })
}

/// Parse the python-literal header dict: `{'descr': '<f4', 'fortran_order':
/// False, 'shape': (80, 330), }`.
fn parse_header_dict(path: &Path, header: &str) -> Result<(String, bool, Vec<usize>)> {
    let get_value = |key: &str| -> Option<&str> {
        let needle = format!("'{key}':");
        let start = header.find(&needle)? + needle.len();
        let rest = header[start..].trim_start();
        // Value ends at its own closing token: a quoted string at the matching
        // quote, a tuple at ')', anything else at the next ',' or dict '}'.
        let end = if let Some(q) = rest.chars().next().filter(|c| *c == '\'' || *c == '"') {
            rest[1..].find(q).map(|i| i + 2)?
        } else if rest.starts_with('(') {
            rest.find(')')? + 1
        } else {
            rest.find([',', '}']).unwrap_or(rest.len())
        };
        Some(rest[..end].trim())
    };

    let descr = get_value("descr")
        .ok_or_else(|| Error::Npy(format!("{}: header has no 'descr'", path.display())))?
        .trim_matches('\'')
        .trim_matches('"')
        .to_string();

    let fortran_order = get_value("fortran_order")
        .map(|v| v == "True")
        .unwrap_or(false);

    let shape_str = get_value("shape")
        .ok_or_else(|| Error::Npy(format!("{}: header has no 'shape'", path.display())))?;
    let inner = shape_str
        .trim_start_matches('(')
        .trim_end_matches(')')
        .trim();
    let shape: Vec<usize> = if inner.is_empty() {
        vec![]
    } else {
        inner
            .split(',')
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .map(|s| {
                s.parse::<usize>().map_err(|e| {
                    Error::Npy(format!("{}: bad shape entry {s:?}: {e}", path.display()))
                })
            })
            .collect::<Result<_>>()?
    };
    Ok((descr, fortran_order, shape))
}

/// Read every member of an `.npz` archive as f32 arrays.
///
/// An `.npz` is a zip whose members are `.npy` files. Only `ZIP_STORED`
/// (method 0) archives are supported: `numpy.savez` never deflates, and the
/// golden artifacts were written by it, so a bare local-header walk is enough
/// — no `flate2`. `numpy.savez` also streams its members: the local header
/// carries `0xffffffff` sizes and a data descriptor trails the payload, so a
/// member's true length is taken from its own `.npy` header
/// ([`npy_data_len`]). The central directory is never parsed.
///
/// Member names keep their `.npy` suffix (`"history.npy"`), in archive order.
pub fn read_npz_f32(path: &Path) -> Result<Vec<(String, NpyArray)>> {
    let bytes = std::fs::read(path).map_err(|source| Error::Io {
        path: path.to_path_buf(),
        source,
    })?;
    parse_npz_f32(path, &bytes)
}

/// Parse `.npz` bytes (see [`read_npz_f32`]).
pub fn parse_npz_f32(path: &Path, bytes: &[u8]) -> Result<Vec<(String, NpyArray)>> {
    const LOCAL_SIG: &[u8; 4] = b"PK\x03\x04";
    const DESCR_SIG: &[u8; 4] = b"PK\x07\x08";
    let mut out = Vec::new();
    let mut pos = 0usize;
    while pos + 30 <= bytes.len() && &bytes[pos..pos + 4] == LOCAL_SIG {
        let flags = u16::from_le_bytes([bytes[pos + 6], bytes[pos + 7]]);
        let method = u16::from_le_bytes([bytes[pos + 8], bytes[pos + 9]]);
        let comp_size = u32::from_le_bytes(bytes[pos + 18..pos + 22].try_into().unwrap()) as usize;
        let name_len = u16::from_le_bytes([bytes[pos + 26], bytes[pos + 27]]) as usize;
        let extra_len = u16::from_le_bytes([bytes[pos + 28], bytes[pos + 29]]) as usize;
        if method != 0 {
            let name = String::from_utf8_lossy(&bytes[pos + 30..pos + 30 + name_len]);
            return Err(Error::Npy(format!(
                "{}: npz member {name:?} is deflated; only numpy.savez \
                 (ZIP_STORED) archives are supported",
                path.display()
            )));
        }
        let name_start = pos + 30;
        let data_start = name_start + name_len + extra_len;
        // `numpy.savez` streams: header sizes are placeholders and a data
        // descriptor trails the payload, so the length comes from the .npy
        // header itself.
        let payload_len = if comp_size == 0xffff_ffff {
            if data_start >= bytes.len() {
                return Err(Error::Npy(format!(
                    "{}: truncated zip member",
                    path.display()
                )));
            }
            npy_data_len(&bytes[data_start..], path)?
        } else {
            comp_size
        };
        if data_start + payload_len > bytes.len() {
            return Err(Error::Npy(format!(
                "{}: truncated zip member",
                path.display()
            )));
        }
        let name = bytes[name_start..name_start + name_len].to_vec();
        let data = parse_npy_f32(path, &bytes[data_start..data_start + payload_len])?;
        out.push((
            String::from_utf8(name).map_err(|_| {
                Error::Npy(format!("{}: npz member name is not UTF-8", path.display()))
            })?,
            data,
        ));

        let mut next = data_start + payload_len;
        if flags & 0x08 != 0 {
            // Skip the data descriptor (crc32 + both sizes), with or without
            // its `PK\x07\x08` signature.
            next += if bytes.len() >= next + 4 && &bytes[next..next + 4] == DESCR_SIG {
                16
            } else {
                12
            };
        }
        pos = next;
    }
    if out.is_empty() {
        return Err(Error::Npy(format!(
            "{}: no ZIP_STORED npy members found",
            path.display()
        )));
    }
    Ok(out)
}

/// Byte length of the `.npy` payload starting at `bytes[0]`, derived from its
/// own header (descr + shape) — the zip-independent way to size a streamed
/// member.
fn npy_data_len(bytes: &[u8], path: &Path) -> Result<usize> {
    if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" {
        return Err(Error::Npy(format!(
            "{}: npz member is not a numpy .npy file",
            path.display()
        )));
    }
    let major = bytes[6];
    let (header_start, header_len): (usize, usize) = match major {
        1 => (10, u16::from_le_bytes([bytes[8], bytes[9]]) as usize),
        2 | 3 => (
            12,
            u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
        ),
        v => {
            return Err(Error::Npy(format!(
                "{}: unsupported .npy version {v}",
                path.display()
            )));
        }
    };
    let header_end = header_start + header_len;
    if bytes.len() < header_end {
        return Err(Error::Npy(format!("{}: truncated header", path.display())));
    }
    let header = std::str::from_utf8(&bytes[header_start..header_end])
        .map_err(|e| Error::Npy(format!("{}: header is not UTF-8: {e}", path.display())))?;
    let (descr, _, shape) = parse_header_dict(path, header)?;
    let item_size = match descr.as_str() {
        "|b1" | "?" | "|i1" | "|u1" => 1usize,
        "<i2" | "<u2" | "<f2" => 2,
        "<i4" | "<f4" => 4,
        "<f8" => 8,
        other => {
            return Err(Error::Npy(format!(
                "{}: dtype {other:?} cannot be read as f32",
                path.display()
            )));
        }
    };
    Ok(header_end + shape.iter().product::<usize>() * item_size)
}

/// Read one named member of an `.npz` archive (name without the `.npy`
/// suffix).
pub fn read_npz_member_f32(path: &Path, key: &str) -> Result<NpyArray> {
    parse_npz_member_f32(
        path,
        &std::fs::read(path).map_err(|source| Error::Io {
            path: path.to_path_buf(),
            source,
        })?,
        key,
    )
}

/// Parse one named member from `.npz` bytes (see [`read_npz_member_f32`]).
pub fn parse_npz_member_f32(path: &Path, bytes: &[u8], key: &str) -> Result<NpyArray> {
    parse_npz_f32(path, bytes)?
        .into_iter()
        .find(|(name, _)| name.strip_suffix(".npy") == Some(key))
        .map(|(_, arr)| arr)
        .ok_or_else(|| Error::Npy(format!("{}: no npz member {key:?}.npy", path.display())))
}

/// IEEE 754 binary16 -> f32, without the `half` crate.
fn half_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) as u32) << 31;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let frac = (bits & 0x3ff) as u32;
    let out = match exp {
        0 => {
            if frac == 0 {
                sign
            } else {
                // Subnormal half: normalize.
                let mut e = 113u32;
                let mut f = frac;
                while f & 0x400 == 0 {
                    f <<= 1;
                    e -= 1;
                }
                f &= 0x3ff;
                sign | (e << 23) | (f << 13)
            }
        }
        0x1f => sign | (0xff << 23) | (frac << 13),
        e => sign | ((e + 112) << 23) | (frac << 13),
    };
    f32::from_bits(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn npy_bytes(descr: &str, shape: &str, payload: &[u8]) -> Vec<u8> {
        let header = format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': {shape}, }}");
        let mut out = Vec::new();
        out.extend_from_slice(b"\x93NUMPY");
        out.push(1);
        out.push(0);
        out.extend_from_slice(&(header.len() as u16).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(payload);
        out
    }

    #[test]
    fn reads_1d_f32() {
        let mut payload = Vec::new();
        for v in [1.0f32, 2.0, 3.0] {
            payload.extend_from_slice(&v.to_le_bytes());
        }
        let bytes = npy_bytes("<f4", "(3,)", &payload);
        let arr = parse_npy_f32(Path::new("x.npy"), &bytes).unwrap();
        assert_eq!(arr.shape, vec![3]);
        assert_eq!(arr.data, vec![1.0, 2.0, 3.0]);
        assert_eq!(arr.rows(), 1);
        assert_eq!(arr.row_len(), 3);
    }

    #[test]
    fn reads_2d_f32() {
        let mut payload = Vec::new();
        payload.extend_from_slice(&1.0f32.to_le_bytes());
        payload.extend_from_slice(&2.0f32.to_le_bytes());
        payload.extend_from_slice(&3.0f32.to_le_bytes());
        payload.extend_from_slice(&4.0f32.to_le_bytes());
        let bytes = npy_bytes("<f4", "(2, 2)", &payload);
        let arr = parse_npy_f32(Path::new("x.npy"), &bytes).unwrap();
        assert_eq!(arr.shape, vec![2, 2]);
        assert_eq!(arr.rows(), 2);
        assert_eq!(arr.row_len(), 2);
    }

    #[test]
    fn reads_f64_and_f16() {
        let bytes = npy_bytes("<f8", "(1,)", &2.5f64.to_le_bytes());
        assert_eq!(
            parse_npy_f32(Path::new("x.npy"), &bytes).unwrap().data,
            vec![2.5]
        );

        // 2.5 in binary16 is 0x4100 (exp 16, frac 256).
        let bytes = npy_bytes("<f2", "(1,)", &[0x00, 0x41]);
        assert_eq!(
            parse_npy_f32(Path::new("x.npy"), &bytes).unwrap().data,
            vec![2.5]
        );
    }

    #[test]
    fn rejects_fortran_and_garbage() {
        let mut bytes = npy_bytes("<f4", "(1,)", &1.0f32.to_le_bytes());
        let header = b"{'descr': '<f4', 'fortran_order': True, 'shape': (1,), }";
        bytes.clear();
        bytes.extend_from_slice(b"\x93NUMPY\x01\x00");
        bytes.extend_from_slice(&(header.len() as u16).to_le_bytes());
        bytes.extend_from_slice(header);
        bytes.extend_from_slice(&1.0f32.to_le_bytes());
        assert!(parse_npy_f32(Path::new("x.npy"), &bytes).is_err());

        assert!(parse_npy_f32(Path::new("x.npy"), b"garbage").is_err());
    }

    /// One stored (method 0) zip local header + payload around `npy_bytes`.
    fn stored_member(name: &str, npy: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(b"PK\x03\x04");
        out.extend_from_slice(&20u16.to_le_bytes()); // version needed
        out.extend_from_slice(&0u16.to_le_bytes()); // flags
        out.extend_from_slice(&0u16.to_le_bytes()); // method: stored
        out.extend_from_slice(&0u16.to_le_bytes()); // time
        out.extend_from_slice(&0x2181u16.to_le_bytes()); // date
        out.extend_from_slice(&0u32.to_le_bytes()); // crc (unchecked here)
        out.extend_from_slice(&(npy.len() as u32).to_le_bytes());
        out.extend_from_slice(&(npy.len() as u32).to_le_bytes());
        out.extend_from_slice(&(name.len() as u16).to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes()); // extra len
        out.extend_from_slice(name.as_bytes());
        out.extend_from_slice(npy);
        out
    }

    #[test]
    fn reads_stored_npz_members() {
        let member = |name: &str, payload: &[u8]| {
            let mut b = b"\x93NUMPY\x01\x00".to_vec();
            let header = format!(
                "{{'descr': '<f4', 'fortran_order': False, 'shape': ({}), }}",
                payload.len() / 4
            );
            b.extend_from_slice(&(header.len() as u16).to_le_bytes());
            b.extend_from_slice(header.as_bytes());
            b.extend_from_slice(payload);
            stored_member(name, &b)
        };
        let mut bytes = member("history.npy", &1.0f32.to_le_bytes());
        bytes.extend(member("future.npy", &2.0f32.to_le_bytes()));

        let path = Path::new("masks.npz");
        let members = parse_npz_f32(path, &bytes).unwrap();
        assert_eq!(members.len(), 2);
        assert_eq!(members[0].0, "history.npy");
        assert_eq!(members[0].1.data, vec![1.0]);
        assert_eq!(members[1].1.data, vec![2.0]);

        let got = parse_npz_member_f32(path, &bytes, "future").unwrap();
        assert_eq!(got.data, vec![2.0]);
        assert!(parse_npz_member_f32(path, &bytes, "missing").is_err());
    }

    #[test]
    fn reads_streamed_npz_members_with_data_descriptors() {
        // numpy.savez streams: placeholder sizes, bit-3 flag, descriptor
        // (signature variant for the first member, bare for the second).
        let streamed_member = |name: &str, npy: &[u8], sig: bool| {
            let mut out = stored_member(name, npy);
            // overwrite sizes with placeholders and set the descriptor flag
            out[6..8].copy_from_slice(&0x0008u16.to_le_bytes());
            out[18..22].copy_from_slice(&0xffff_ffffu32.to_le_bytes());
            out[22..26].copy_from_slice(&0xffff_ffffu32.to_le_bytes());
            let desc_len = if sig { 16 } else { 12 };
            let mut desc = vec![0u8; desc_len];
            if sig {
                desc[..4].copy_from_slice(b"PK\x07\x08");
            }
            desc[desc_len - 12..desc_len - 8].copy_from_slice(&0x1234u32.to_le_bytes());
            desc[desc_len - 8..desc_len - 4].copy_from_slice(&(npy.len() as u32).to_le_bytes());
            desc[desc_len - 4..].copy_from_slice(&(npy.len() as u32).to_le_bytes());
            out.extend_from_slice(&desc);
            out
        };
        let npy = |v: f32| {
            let header = b"{'descr': '<f4', 'fortran_order': False, 'shape': (1,), }";
            let mut b = b"\x93NUMPY\x01\x00".to_vec();
            b.extend_from_slice(&(header.len() as u16).to_le_bytes());
            b.extend_from_slice(&header[..]);
            b.extend_from_slice(&v.to_le_bytes());
            b
        };
        let mut bytes = streamed_member("history.npy", &npy(1.0), true);
        bytes.extend(streamed_member("future.npy", &npy(2.0), false));

        let members = parse_npz_f32(Path::new("masks.npz"), &bytes).unwrap();
        assert_eq!(members.len(), 2);
        assert_eq!(members[0].1.data, vec![1.0]);
        assert_eq!(members[1].1.data, vec![2.0]);
    }
}
