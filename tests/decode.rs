//! Decode-stage replay against the golden torch-reference dumps.
//!
//! The golden run (`--no-postprocess`) ends at `model.motion_rep.inverse()`,
//! so the whole M5 stage is replayable:
//!
//! * `final_hybrid_translated` -> split -> per-horizon detokenise
//!   (`fsq_detokenize_{0,1}_*`) -> `final_motion_decoded_autoregressive`
//!   (≡ `motion_normalized`),
//! * that feature tensor -> `inverse` -> the seven tensors of
//!   `out/golden_core_h40_4s_seed0.npz`.
//!
//! The `.npz` also carries `fps` (int64) and `text` (<U23); the f32 npz
//! reader skips those two metadata members.

use ardy::io::bundle::Bundle;
use ardy::io::npy::{NpyArray, read_npz_f32};
use ardy::model::decode::{
    autoregressive_chunk_slices, explicit_from_hybrid_autoregressive, inverse, motion_from_hybrid,
};
use ardy::model::denoiser::split_hybrid_free;
use ardy::model::fsq::FsqVaeTransformer;
use ardy::model::motion_rep::{MotionRep, Stats};
use ardy::model::nn::WeightStore;
use ardy::model::skeleton::Skeleton;
use mlx_rs::ops::indexing::{Ellipsis, TryIndexOp};

mod common;
use common::*;

/// The golden FK output archive (`out/` sibling of the raw dump dir).
fn golden_out_npz() -> Option<Vec<(String, NpyArray)>> {
    let gdir = golden_dir()?;
    let out = gdir
        .parent()
        .expect("raw dir has a parent")
        .join("golden_core_h40_4s_seed0.npz");
    out.is_file().then(|| read_npz_f32(&out).unwrap())
}

/// `latent_utils.py`'s chunk plan: two full 10-token horizons for the 80-frame
/// golden clip, and history truncation once the crop bites.
#[test]
fn chunk_slices_match_latent_utils() {
    assert_eq!(
        autoregressive_chunk_slices(20, 10, None).unwrap(),
        vec![(0, 0, 10), (0, 10, 20)]
    );
    // crop_history_length 160 frames = 40 tokens >= 20 tokens: no truncation
    assert_eq!(
        autoregressive_chunk_slices(20, 10, Some(40)).unwrap(),
        vec![(0, 0, 10), (0, 10, 20)]
    );
    // a 1-token crop truncates chunk 1's history to its last token
    assert_eq!(
        autoregressive_chunk_slices(20, 10, Some(1)).unwrap(),
        vec![(0, 0, 10), (9, 10, 20)]
    );
    assert!(autoregressive_chunk_slices(20, 0, None).is_err());
}

/// FK from the golden `local_rot_mats` + `root_positions` reproduces
/// `posed_joints` (the ardy-mlx port measured 1.7e-07; float32 batching
/// differences put the Rust port in the same decade).
#[test]
fn fk_replays_golden_posed_joints() {
    let Some(members) = golden_out_npz() else {
        eprintln!("skip: no golden output npz");
        return;
    };
    let local = arr(npz_member(&members, "local_rot_mats"));
    let root_pos = arr(npz_member(&members, "root_positions"));
    let (glob, posed, _norm) = Skeleton::core().fk(&local, &root_pos).unwrap();

    assert_close(
        &flat(&glob),
        &flat(&arr(npz_member(&members, "global_rot_mats"))),
        // measured: 2.4e-7 (parent^T @ child product order)
        1e-6,
        "fk global_rot_mats",
    );
    assert_close(
        &flat(&posed),
        &flat(&arr(npz_member(&members, "posed_joints"))),
        // measured: 1.9e-6 over the 4 s clip
        1e-5,
        "fk posed_joints",
    );
    // joint 0 IS the root position, exactly
    assert_close(
        &flat(&posed.try_index((Ellipsis, 0i32, Ellipsis)).unwrap()),
        &flat(&root_pos),
        0.0,
        "posed root == root_positions",
    );
}

/// The full `inverse` chain (unnormalise -> cont6d -> global->local -> fk)
/// replays every tensor member of the golden output archive.
#[test]
fn inverse_replays_golden_output_npz() {
    let Some(bdir) = bundle_dir() else {
        eprintln!("skip: no bundle (set ARDY_BUNDLE)");
        return;
    };
    let Some(gdir) = golden_dir() else {
        eprintln!("skip: no golden run (set ARDY_GOLDEN)");
        return;
    };
    let Some(members) = golden_out_npz() else {
        eprintln!("skip: no golden output npz");
        return;
    };

    let bundle = Bundle::open(&bdir).unwrap();
    let mr = MotionRep::from_bundle(&bundle).unwrap();
    let skel = Skeleton::core();

    // motion_normalized ≡ final_motion_decoded_autoregressive (max diff 0.0)
    let features = arr(&npy(&gdir, "motion_normalized.npy"));
    let out = inverse(&mr, skel, &features, true).unwrap();

    assert_close(
        &flat(&out.local_rot_mats),
        &flat(&arr(npz_member(&members, "local_rot_mats"))),
        // measured: 5.1e-7 (cont6d + g2l trig)
        1e-5,
        "inverse local_rot_mats",
    );
    assert_close(
        &flat(&out.global_rot_mats),
        &flat(&arr(npz_member(&members, "global_rot_mats"))),
        // measured: 3.8e-7
        1e-5,
        "inverse global_rot_mats",
    );
    assert_close(
        &flat(&out.posed_joints),
        &flat(&arr(npz_member(&members, "posed_joints"))),
        // measured: 2.0e-6
        1e-5,
        "inverse posed_joints",
    );
    // pure feature slices: exact
    assert_close(
        &flat(&out.root_positions),
        &flat(&arr(npz_member(&members, "root_positions"))),
        0.0,
        "inverse root_positions",
    );
    assert_close(
        &flat(&out.root_positions),
        &flat(&arr(npz_member(&members, "smooth_root_pos"))),
        0.0,
        "smooth_root_pos is root_positions",
    );
    assert_close(
        &flat(&out.global_root_heading),
        &flat(&arr(npz_member(&members, "global_root_heading"))),
        0.0,
        "inverse global_root_heading",
    );
    // bool contact flags widen to 0/1 f32 in the golden archive
    assert_close(
        &flat(&out.foot_contacts.as_dtype(mlx_rs::Dtype::Float32).unwrap()),
        &flat(&arr(npz_member(&members, "foot_contacts"))),
        0.0,
        "inverse foot_contacts",
    );
}

/// The two per-horizon detokenise calls of the autoregressive decode, replayed
/// against the fsq hook dumps, then the stitched whole.
#[test]
fn autoregressive_decode_replays_golden() {
    let Some(bdir) = bundle_dir() else {
        eprintln!("skip: no bundle (set ARDY_BUNDLE)");
        return;
    };
    let Some(gdir) = golden_dir() else {
        eprintln!("skip: no golden run (set ARDY_GOLDEN)");
        return;
    };

    let bundle = Bundle::open(&bdir).unwrap();
    let mr = MotionRep::from_bundle(&bundle).unwrap();
    let mut tstore = WeightStore::new(bundle.tokenizer().clone());
    let mut fsq = FsqVaeTransformer::new(&mut tstore, &mr, "pose_net").unwrap();
    let post = bundle.stats("post_quantization").unwrap();
    let pre = bundle.stats("pre_quantization").unwrap();
    fsq.load_stats(
        Stats::from_npy(&post.mean, &post.std).unwrap(),
        Stats::from_npy(&pre.mean, &pre.std).unwrap(),
    );

    let hybrid = arr(&npy(&gdir, "final_hybrid_translated.npy"));
    let (root_motion, latent) = split_hybrid_free(&hybrid, 4, 20, mr.root_block_dim()).unwrap();
    // per-frame root block: root position (3) + heading (2)
    assert_eq!(root_motion.shape(), &[1, 80, 5]);
    assert_eq!(latent.shape(), &[1, 20, 128]);

    // chunk 0: tokens 0..10, frames 0..40
    let sub_root0 = root_motion
        .try_index(((0i32..), (0i32..40), (0i32..)))
        .unwrap();
    let local_root0 = mr
        .global_root_to_local_root(&sub_root0, true, None)
        .unwrap();
    let (_r0, body0) = fsq
        .detokenize(
            &latent.try_index(((0i32..), (0i32..10), (0i32..))).unwrap(),
            &local_root0,
            None,
        )
        .unwrap();
    assert_close(
        &flat(&body0),
        &flat(&arr(&npy(&gdir, "fsq_detokenize_0_out_body.npy"))),
        // measured: 6.2e-6 (decoder conv stacks)
        1e-4,
        "chunk0 detokenize body",
    );

    // chunk 1: full-sequence detokenise (tokens 0..20), sliced to 40..80 after
    let local_root1 = mr
        .global_root_to_local_root(&root_motion, true, None)
        .unwrap();
    let (_r1, body1) = fsq.detokenize(&latent, &local_root1, None).unwrap();
    assert_close(
        &flat(&body1),
        &flat(&arr(&npy(&gdir, "fsq_detokenize_1_out_body.npy"))),
        1e-4,
        "chunk1 detokenize body (full sequence)",
    );

    // the stitched autoregressive decode
    let motion =
        explicit_from_hybrid_autoregressive(&latent, &root_motion, &fsq, &mr, 4, 40, Some(160))
            .unwrap();
    assert_close(
        &flat(&motion),
        &flat(&arr(&npy(&gdir, "final_motion_decoded_autoregressive.npy"))),
        // measured: 1.1e-5
        1e-4,
        "explicit_from_hybrid_autoregressive",
    );
    assert_eq!(motion.shape(), &[1, 80, 330]);

    // the generate() dispatch: same tensor through motion_from_hybrid
    // (crop active -> autoregressive), trimmed to num_frames = 80
    let via_dispatch = motion_from_hybrid(&hybrid, &fsq, &mr, 4, 20, 40, Some(160), 80).unwrap();
    assert_close(
        &flat(&via_dispatch),
        &flat(&arr(&npy(&gdir, "motion_normalized.npy"))),
        1e-4,
        "motion_from_hybrid",
    );

    // stats-convention control: normalising both decodes and comparing —
    // a mean/std swap or slice slip would blow this up even where the raw
    // diff looks small. (Normalising normalized data is NOT an identity;
    // both sides sit around |max| ~ 43.)
    let golden_motion = arr(&npy(&gdir, "motion_normalized.npy"));
    let renorm_mine = mr.normalize(&motion).unwrap();
    let renorm_gold = mr.normalize(&golden_motion).unwrap();
    assert_close(
        &flat(&renorm_mine),
        &flat(&renorm_gold),
        1e-3,
        "renormalized decode vs renormalized golden",
    );
}
