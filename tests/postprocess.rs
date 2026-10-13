//! Differential tests for the MotionCorrection port against the *real*
//! upstream C++.
//!
//! A standalone probe binary (compiled from the unmodified upstream sources,
//! see `scripts/mc_probe/README.md`) replays the pybind `correct_motion`
//! marshaling over a flat `MCP1` binary record. Each test here generates a
//! synthetic (LCG — no NVIDIA data) record, feeds it to both the Rust port
//! and the probe, and diffs the corrected root/rotation arrays.
//!
//! Gated on the probe binary: set `ARDY_MC_PROBE=/path/to/probe` (default
//! `/tmp/mc-probe/probe`). Without it these tests pass vacuously — the rest
//! of the suite (`--lib postprocess`) pins the internal invariants.

use std::path::PathBuf;
use std::process::Command;

use ardy::model::skeleton::Skeleton;
use ardy::postprocess::{MotionConstraints, correct_motion};

/// Deterministic LCG so fixtures are synthetic and reproducible.
struct Lcg(u64);

impl Lcg {
    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 33) as u32
    }

    /// Uniform `[0, 1)`.
    fn unit(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / (1u32 << 24) as f32
    }

    /// Uniform `[a, b)`.
    fn range(&mut self, a: f32, b: f32) -> f32 {
        a + (b - a) * self.unit()
    }

    fn bernoulli(&mut self, p: f32) -> bool {
        self.unit() < p
    }
}

fn probe_path() -> Option<PathBuf> {
    let path = std::env::var("ARDY_MC_PROBE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/tmp/mc-probe/probe"));
    if path.is_file() { Some(path) } else { None }
}

struct Scenario {
    seed: u64,
    contact_p: f32,
    full_body_frames: Vec<usize>,
    root_frames: Vec<usize>,
    hand_foot_masks: bool,
}

struct Inputs {
    t: usize,
    j: usize,
    parents: Vec<i32>,
    ref_t: Vec<[f32; 3]>,
    ref_r: Vec<[f32; 4]>,
    root: Vec<f32>,
    rots: Vec<f32>,
    root_target: Vec<f32>,
    rots_target: Vec<f32>,
    full: Vec<f32>,
    lhand: Vec<f32>,
    rhand: Vec<f32>,
    lfoot: Vec<f32>,
    rfoot: Vec<f32>,
    root_mask: Vec<f32>,
    contacts: Vec<f32>,
    ee: [usize; 4],
}

/// Build the synthetic record inputs over the real cskel27 rig.
fn make_inputs(sc: &Scenario) -> Inputs {
    let skel = Skeleton::core();
    let j = skel.nbjoints;
    let t = 24;
    let mut rng = Lcg(sc.seed);

    let ref_t: Vec<[f32; 3]> = (0..j)
        .map(|i| {
            if skel.joint_parents[i] == -1 {
                skel.core_rig_root_translation()
            } else {
                skel.core_rig_offset(i)
            }
        })
        .collect();
    let ref_r = vec![[0.0f32, 0.0, 0.0, 1.0]; j]; // identity rig (xyzw)

    // random unit quats in wxyz layout (frame-major T*J*4)
    let mut quats = vec![0f32; 4 * j * t];
    for f in 0..t {
        for jj in 0..j {
            let mut q = [
                rng.range(-1.0, 1.0),
                rng.range(-1.0, 1.0),
                rng.range(-1.0, 1.0),
                rng.range(-1.0, 1.0),
            ];
            let n = (q.iter().map(|v| v * v).sum::<f32>()).sqrt();
            if n > 1e-3 {
                for v in q.iter_mut() {
                    *v /= n;
                }
            } else {
                q = [1.0, 0.0, 0.0, 0.0];
            }
            quats[4 * (j * f + jj)..4 * (j * f + jj) + 4].copy_from_slice(&q);
        }
    }
    let mut root = vec![0f32; 3 * t];
    for f in 0..t {
        root[3 * f] = rng.range(-0.5, 0.5);
        root[3 * f + 1] = rng.range(0.9, 1.1);
        root[3 * f + 2] = rng.range(-0.5, 0.5);
    }

    // targets: a second independent draw (constrained frames get these)
    let mut quats_target = vec![0f32; 4 * j * t];
    for f in 0..t {
        for jj in 0..j {
            let mut q = [
                rng.range(-1.0, 1.0),
                rng.range(-1.0, 1.0),
                rng.range(-1.0, 1.0),
                rng.range(-1.0, 1.0),
            ];
            let n = (q.iter().map(|v| v * v).sum::<f32>()).sqrt();
            if n > 1e-3 {
                for v in q.iter_mut() {
                    *v /= n;
                }
            } else {
                q = [1.0, 0.0, 0.0, 0.0];
            }
            quats_target[4 * (j * f + jj)..4 * (j * f + jj) + 4].copy_from_slice(&q);
        }
    }
    let mut root_target = vec![0f32; 3 * t];
    for f in 0..t {
        root_target[3 * f] = rng.range(-0.5, 0.5);
        root_target[3 * f + 1] = rng.range(0.9, 1.1);
        root_target[3 * f + 2] = rng.range(-0.5, 0.5);
    }

    let zeros = vec![0f32; t];
    let mut full = zeros.clone();
    for &f in &sc.full_body_frames {
        full[f] = 1.0;
    }
    let mut root_mask = zeros.clone();
    for &f in &sc.root_frames {
        root_mask[f] = 1.0;
    }
    let (lhand, rhand, lfoot, rfoot) = if sc.hand_foot_masks {
        (
            (0..t)
                .map(|_| if rng.bernoulli(0.2) { 1.0 } else { 0.0 })
                .collect::<Vec<_>>(),
            (0..t)
                .map(|_| if rng.bernoulli(0.2) { 1.0 } else { 0.0 })
                .collect::<Vec<_>>(),
            (0..t)
                .map(|_| if rng.bernoulli(0.2) { 1.0 } else { 0.0 })
                .collect::<Vec<_>>(),
            (0..t)
                .map(|_| if rng.bernoulli(0.2) { 1.0 } else { 0.0 })
                .collect::<Vec<_>>(),
        )
    } else {
        (zeros.clone(), zeros.clone(), zeros.clone(), zeros.clone())
    };

    // contacts [L foot, L toe, R foot, R toe] in contiguous runs so the
    // interval logic sees real phases
    let mut contacts = vec![0f32; 4 * t];
    let mut f = 0usize;
    while f < t {
        if rng.bernoulli(sc.contact_p) {
            let len = 2 + (rng.next_u32() % 4) as usize;
            for i in f..(f + len).min(t) {
                contacts[4 * i] = 1.0; // left foot
                contacts[4 * i + 1] = if rng.bernoulli(0.5) { 1.0 } else { 0.0 };
                contacts[4 * i + 2] = if rng.bernoulli(0.5) { 1.0 } else { 0.0 };
                contacts[4 * i + 3] = if rng.bernoulli(0.5) { 1.0 } else { 0.0 };
            }
            f += len + 2;
        } else {
            f += 1;
        }
    }

    let ee = [
        skel.joint_names
            .iter()
            .position(|&n| n == "LeftHand")
            .expect("LeftHand"),
        skel.joint_names
            .iter()
            .position(|&n| n == "RightHand")
            .expect("RightHand"),
        skel.joint_names
            .iter()
            .position(|&n| n == "LeftFoot")
            .expect("LeftFoot"),
        skel.joint_names
            .iter()
            .position(|&n| n == "RightFoot")
            .expect("RightFoot"),
    ];

    Inputs {
        t,
        j,
        parents: skel.joint_parents.clone(),
        ref_t,
        ref_r,
        root,
        rots: quats,
        root_target,
        rots_target: quats_target,
        full,
        lhand,
        rhand,
        lfoot,
        rfoot,
        root_mask,
        contacts,
        ee,
    }
}

fn write_record(path: &std::path::Path, inp: &Inputs, has_double: bool, ct: f32, rm: f32) {
    use std::io::Write;
    let mut w = Vec::new();
    let put = |w: &mut Vec<u8>, b: &[u8]| w.extend_from_slice(b);
    let put_i32 = |w: &mut Vec<u8>, v: i32| put(w, &v.to_le_bytes());
    let put_f32 = |w: &mut Vec<u8>, v: f32| put(w, &v.to_le_bytes());
    put(&mut w, b"MCP1");
    put_i32(&mut w, inp.t as i32);
    put_i32(&mut w, inp.j as i32);
    put_i32(&mut w, has_double as i32);
    put_f32(&mut w, ct);
    put_f32(&mut w, rm);
    for &p in &inp.parents {
        put_i32(&mut w, p);
    }
    for r in &inp.ref_t {
        for v in r {
            put_f32(&mut w, *v);
        }
    }
    for r in &inp.ref_r {
        // record layout stores the rig rotations wxyz (the probe's reading);
        // identity anyway
        for v in [r[3], r[0], r[1], r[2]] {
            put_f32(&mut w, v);
        }
    }
    for v in &inp.root {
        put_f32(&mut w, *v);
    }
    for v in &inp.rots {
        put_f32(&mut w, *v);
    }
    for v in &inp.root_target {
        put_f32(&mut w, *v);
    }
    for v in &inp.rots_target {
        put_f32(&mut w, *v);
    }
    for m in [
        &inp.full,
        &inp.lhand,
        &inp.rhand,
        &inp.lfoot,
        &inp.rfoot,
        &inp.root_mask,
    ] {
        for v in m.iter() {
            put_f32(&mut w, *v);
        }
    }
    for v in &inp.contacts {
        put_f32(&mut w, *v);
    }
    for &e in &inp.ee {
        put_i32(&mut w, e as i32);
    }
    let mut f = std::fs::File::create(path).expect("create record");
    f.write_all(&w).expect("write record");
}

fn read_outputs(path: &std::path::Path, inp: &Inputs) -> (Vec<f32>, Vec<f32>) {
    let bytes = std::fs::read(path).expect("read probe output");
    let mut off = 4 + 4 * 3 + 4 * 2 + 4 * inp.parents.len() + 4 * inp.j * 3 + 4 * inp.j * 4;
    let n = 3 * inp.t;
    let root = bytes[off..off + 4 * n]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    off += 4 * n;
    let rots = bytes[off..off + 4 * inp.t * inp.j * 4]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    (root, rots)
}

/// Run one scenario through both implementations and compare.
fn diff_against_probe(name: &str, sc: &Scenario, has_double: bool, ct: f32, rm: f32) {
    let Some(probe) = probe_path() else {
        eprintln!("skipping {name}: probe binary not found");
        return;
    };
    let inp = make_inputs(sc);
    let dir = std::env::temp_dir().join(format!("ardy-mc-test-{}-{}", name, sc.seed));
    std::fs::create_dir_all(&dir).expect("temp dir");
    let in_path = dir.join("in.mcp");
    let out_path = dir.join("out.mcp");
    write_record(&in_path, &inp, has_double, ct, rm);

    let status = Command::new(&probe)
        .arg(&in_path)
        .arg(&out_path)
        .status()
        .expect("run probe");
    assert!(status.success(), "probe failed for {name}");

    // Rust side on identical inputs
    let mut root = inp.root.clone();
    let mut rots = inp.rots.clone();
    correct_motion(
        &mut root,
        &mut rots,
        &inp.root_target,
        &inp.rots_target,
        &inp.full,
        &inp.lhand,
        &inp.rhand,
        &inp.lfoot,
        &inp.rfoot,
        &inp.root_mask,
        &inp.contacts,
        &inp.parents,
        &inp.ref_t,
        &inp.ref_r,
        inp.ee[0],
        inp.ee[1],
        inp.ee[2],
        inp.ee[3],
        ct,
        rm,
        has_double,
    )
    .expect("rust correct_motion");

    let (p_root, p_rots) = read_outputs(&out_path, &inp);
    let worst_root = root
        .iter()
        .zip(&p_root)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let worst_rot = rots
        .iter()
        .zip(&p_rots)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    // f32 engine math on both sides (SSE vs scalar): identical formulas, so
    // only rounding-level drift is expected; the ADMM interior runs in f64.
    let tol = 1e-5;
    assert!(
        worst_root <= tol,
        "{name}: root diff {worst_root} > {tol} (probe vs rust)"
    );
    assert!(
        worst_rot <= tol,
        "{name}: rotation diff {worst_rot} > {tol} (probe vs rust)"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn probe_no_constraints() {
    diff_against_probe(
        "no_constraints",
        &Scenario {
            seed: 1,
            contact_p: 0.0,
            full_body_frames: vec![],
            root_frames: vec![],
            hand_foot_masks: false,
        },
        false,
        0.5,
        0.04,
    );
}

#[test]
fn probe_feet_contacts() {
    diff_against_probe(
        "feet_contacts",
        &Scenario {
            seed: 2,
            contact_p: 0.6,
            full_body_frames: vec![],
            root_frames: vec![],
            hand_foot_masks: false,
        },
        false,
        0.5,
        0.04,
    );
}

#[test]
fn probe_full_body_and_root_pins() {
    diff_against_probe(
        "fullbody_root_pins",
        &Scenario {
            seed: 3,
            contact_p: 0.0,
            full_body_frames: vec![2, 3, 4, 15],
            root_frames: vec![8, 9, 20],
            hand_foot_masks: false,
        },
        false,
        0.5,
        0.04,
    );
}

#[test]
fn probe_fullbody_pins_only() {
    diff_against_probe(
        "fullbody_only",
        &Scenario {
            seed: 5,
            contact_p: 0.0,
            full_body_frames: vec![2, 3, 4, 15],
            root_frames: vec![],
            hand_foot_masks: false,
        },
        false,
        0.5,
        0.04,
    );
}

#[test]
fn probe_root_pins_only() {
    diff_against_probe(
        "root_only",
        &Scenario {
            seed: 6,
            contact_p: 0.0,
            full_body_frames: vec![],
            root_frames: vec![8, 9, 20],
            hand_foot_masks: false,
        },
        false,
        0.5,
        0.04,
    );
}

#[test]
fn probe_hand_foot_masks_with_contacts() {
    diff_against_probe(
        "hand_foot_masks",
        &Scenario {
            seed: 4,
            contact_p: 0.4,
            full_body_frames: vec![5, 6],
            root_frames: vec![],
            hand_foot_masks: true,
        },
        false,
        0.5,
        0.04,
    );
}

/// The masks dict / constraints plumbing (`post_process_motion`) on the
/// unconstrained path must not move an all-identity motion — a cheap
/// regression net for the Array-level glue that the probe does not cover.
#[test]
fn post_process_motion_glue_identity() {
    let skel = Skeleton::core();
    let (t, j) = (10, skel.nbjoints);
    let mut mats = vec![0f32; 9 * j * t];
    for o in (0..9 * j * t).step_by(9) {
        mats[o] = 1.0;
        mats[o + 4] = 1.0;
        mats[o + 8] = 1.0;
    }
    let mut root = vec![0.6f32; 3 * t];
    let before = (mats.clone(), root.clone());
    let cons = MotionConstraints::unconstrained(t, j);
    let out = ardy::postprocess::post_process_motion(
        &mut mats,
        &mut root,
        &vec![0.0f32; 4 * t],
        skel,
        &cons,
        0.5,
        0.04,
    )
    .expect("glue");
    for (a, b) in out.0.iter().zip(&before.0) {
        assert!((a - b).abs() < 1e-6);
    }
    for (a, b) in out.1.iter().zip(&before.1) {
        assert!((a - b).abs() < 1e-6);
    }
}
