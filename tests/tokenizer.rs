//! Stage A tokenizer replay against the golden torch-reference dumps
//! (`ardy-mlx/artifacts/golden-run-4s/raw`).
//!
//! The golden run hooked the **upstream reference** implementation, so these
//! assertions pin the Rust port to NVIDIA's behaviour at float32 roundoff
//! level, not merely to ardy-mlx's. Skipped with a message when either the
//! bundle or the golden run is absent.

use ardy::io::bundle::Bundle;
use ardy::model::fsq::FsqVaeTransformer;
use ardy::model::motion_rep::{MotionRep, Stats};
use ardy::model::nn::{Linear, WeightStore};
use mlx_rs::Array;

mod common;
use common::*;

#[test]
fn golden_replay_fsq() {
    let Some(bdir) = bundle_dir() else {
        eprintln!("skip: no bundle (set ARDY_BUNDLE)");
        return;
    };
    let Some(gdir) = golden_dir() else {
        eprintln!("skip: no golden run (set ARDY_GOLDEN)");
        return;
    };

    let bundle = Bundle::open(&bdir).unwrap();
    let mut store = WeightStore::new(bundle.tokenizer().clone());
    let mr = MotionRep::from_bundle(&bundle).unwrap();
    let mut fsq = FsqVaeTransformer::new(&mut store, &mr, "pose_net").unwrap();
    let post = bundle.stats("post_quantization").unwrap();
    let pre = bundle.stats("pre_quantization").unwrap();
    fsq.load_stats(
        Stats::from_npy(&post.mean, &post.std).unwrap(),
        Stats::from_npy(&pre.mean, &pre.std).unwrap(),
    );

    let motion = arr(&npy(&gdir, "motion_normalized.npy"));
    let mask = arr(&npy(&gdir, "probe_pad_mask.npy"));

    // --- tokenize: normalised features -> FSQ tokens --------------------------
    let tokens = fsq.tokenize(&motion, &mask).unwrap();
    let want_tokens = arr(&npy(&gdir, "fsq_tokenize_0_out.npy"));
    assert_close(&flat(&tokens), &flat(&want_tokens), 1e-5, "tokenize");

    // --- detokenize_2: tokens + reconstructed local root -> (root, body) ------
    // The golden dump captured no `external_cond`; the reference derives it
    // from the features via get_explicit_motion_from_hybrid, replayed here.
    let root_in = mr.extract_root(&motion).unwrap();
    let lengths = Array::from_slice(&[80.0f32], &[1]);
    let local_root = mr
        .global_root_to_local_root(&root_in, true, Some(&lengths))
        .unwrap();
    let (droot, dbody) = fsq
        .detokenize(&want_tokens, &local_root, Some(&mask))
        .unwrap();
    assert_close(
        &flat(&droot),
        &flat(&arr(&npy(&gdir, "fsq_detokenize_2_out_root.npy"))),
        2e-4,
        "detokenize root",
    );
    assert_close(
        &flat(&dbody),
        &flat(&arr(&npy(&gdir, "fsq_detokenize_2_out_body.npy"))),
        2e-4,
        "detokenize body",
    );

    // --- requantize: pure post_quantization-stats round -----------------------
    for k in 0..2 {
        let inp = arr(&npy(&gdir, &format!("fsq_requantize_{k}_in.npy")));
        let out = fsq.requantize(&inp).unwrap();
        let want = arr(&npy(&gdir, &format!("fsq_requantize_{k}_out.npy")));
        assert_close(&flat(&out), &flat(&want), 1e-5, &format!("requantize {k}"));
    }

    // --- module-level replay: encoder input_proj straight off the store -------
    let mdir = gdir.join("modules");
    let in0 = arr(&npy(
        &mdir,
        "autoencoder__encoder__input_proj__call0__in0.npy",
    ));
    let proj = Linear::new(&mut store, "pose_net.encoder.input_proj", true).unwrap();
    let out = proj.forward(&in0).unwrap();
    assert_close(
        &flat(&out),
        &flat(&arr(&npy(
            &mdir,
            "autoencoder__encoder__input_proj__call0__out.npy",
        ))),
        1e-5,
        "encoder input_proj replay",
    );

    // The Core checkpoint carries decoder target_cond_blocks weights that the
    // reference inference path never uses; loading must notice but not wire them.
    assert!(fsq.has_target_cond());
}
