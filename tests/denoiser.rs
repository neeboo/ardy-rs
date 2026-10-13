//! Stage B denoiser replay against the golden torch-reference dumps
//! (`ardy-mlx/artifacts/golden-run-4s/raw`).
//!
//! Window 0, diffusion step 0 (`t = 9`, the noisiest) is the sharpest probe:
//! it checks both the raw **tripled** forward (`denoiser_raw_0`, the
//! pre-CFG `[3, 20, 148]` batch the cfg wrapper runs) and the final CFG
//! composition against `denoise_step0_x_out`. All mask/length dumps are f32
//! (the golden hook casts every tensor), so masks are rebuilt with `> 0.5`
//! and lengths as exact integers.
//!
//! Window 0 is also the interesting edge: `history_len == 0`, so
//! `lengths = 0 + 40` against an 80-frame tensor — the case that makes
//! forwarding `lengths` through `global_root_to_local_root` mandatory.

use ardy::io::{DEFAULT_CFG_WEIGHT, bundle::Bundle};
use ardy::model::backbone::TransformerEncoderBlock;
use ardy::model::denoiser::{AutoLatentTwostageDenoiser, Window};
use ardy::model::diffusion::{Diffusion, denoising_step};
use ardy::model::motion_rep::MotionRep;
use ardy::model::nn::{Linear, WeightStore, scalar};
use mlx_rs::Array;
use mlx_rs::ops::{
    self,
    indexing::{Ellipsis, TryIndexOp},
};

mod common;
use common::*;

/// Rebuild the window-0 step-0 tensors shared by both tests in this file.
fn golden_window(gdir: &std::path::Path) -> (Window, MotionRep) {
    let bundle = Bundle::open(&bundle_dir().unwrap()).unwrap();
    let mr = MotionRep::from_bundle(&bundle).unwrap();

    let masks = npz(gdir, "masks_0.npz");
    let tmasks = npz(gdir, "token_masks_0.npz");
    let w = Window {
        x: arr(&npy(gdir, "denoise_step0_x_in.npy")),
        history_len: i32_arr(npz_member(&masks, "history_len")),
        generation_len: i32_arr(npz_member(&masks, "generation_len")),
        future_len: i32_arr(npz_member(&masks, "future_len")),
        history_mask: bool_arr(npz_member(&masks, "history")),
        generation_mask: bool_arr(npz_member(&masks, "generation")),
        future_mask: bool_arr(npz_member(&masks, "future")),
        history_token_mask: bool_arr(npz_member(&tmasks, "history")),
        generation_token_mask: bool_arr(npz_member(&tmasks, "generation")),
        future_token_mask: bool_arr(npz_member(&tmasks, "future")),
        text_feat: arr(&npy(gdir, "text_feat.npy")),
        // steps.jsonl line 1: {"phase": "window0", "step": 0, "t": [9.0]}
        timesteps: Array::from_slice(&[9i32], &[1]),
        first_heading_angle: Some(arr(&npy(gdir, "window0_first_heading_in.npy"))),
        // The golden run carried no constraints; the reference still evaluates
        // the constraint projections on zeros.
        motion_mask: None,
        observed_motion: None,
    };
    (w, mr)
}

#[test]
fn golden_replay_denoiser_window0_step0() {
    let Some(_bdir) = bundle_dir() else {
        eprintln!("skip: no bundle (set ARDY_BUNDLE)");
        return;
    };
    let Some(gdir) = golden_dir() else {
        eprintln!("skip: no golden run (set ARDY_GOLDEN)");
        return;
    };

    let (w, mr) = golden_window(&gdir);
    let bundle = Bundle::open(&bundle_dir().unwrap()).unwrap();
    let mut store = WeightStore::new(bundle.denoiser().clone());
    let denoiser = AutoLatentTwostageDenoiser::new(&mut store, &mr, "denoiser").unwrap();

    // (a) the raw tripled forward, pre-CFG: [3, 20, 148]
    let raw = denoiser.forward(&w.tripled().unwrap()).unwrap();
    assert_close(
        &flat(&raw),
        &flat(&arr(&npy(&gdir, "denoiser_raw_0.npy"))),
        5e-4,
        "raw tripled forward",
    );

    // (b) the full denoising step: CFG composition + DDIM update on the
    // generation tokens. `denoise_step0_x_out` is the *post-DDIM* tensor (the
    // golden hook wraps `Ardy.denoising_step`), not the raw CFG output.
    let diffusion = Diffusion::new(10).unwrap();
    let (w_text, w_cstr) = DEFAULT_CFG_WEIGHT;
    // window 0: no history, so the generation span is tokens 0..10
    let out = denoising_step(
        &denoiser,
        &diffusion,
        &w,
        0,
        10,
        9,
        w_text as f32,
        w_cstr as f32,
    )
    .unwrap();
    assert_close(
        &flat(&out),
        &flat(&arr(&npy(&gdir, "denoise_step0_x_out.npy"))),
        1e-3,
        "denoising step 0",
    );
}

/// Stage-boundary replay: every projection and both backbone transformers are
/// compared against the golden hook's per-module dumps
/// (`modules/denoiser__model__*__call0__*`), so a divergence localises to one
/// boundary instead of the whole 148-wide hybrid output.
#[test]
fn golden_replay_denoiser_module_anchors() {
    let Some(_) = bundle_dir() else {
        eprintln!("skip: no bundle (set ARDY_BUNDLE)");
        return;
    };
    let Some(gdir) = golden_dir() else {
        eprintln!("skip: no golden run (set ARDY_GOLDEN)");
        return;
    };

    let (w, mr) = golden_window(&gdir);
    let bundle = Bundle::open(&bundle_dir().unwrap()).unwrap();
    let mut store = WeightStore::new(bundle.denoiser().clone());
    let denoiser = AutoLatentTwostageDenoiser::new(&mut store, &mr, "denoiser").unwrap();

    let mdir = gdir.join("modules");
    let m = |module: &str, part: &str| -> Array {
        arr(&npy(
            &mdir,
            &format!("denoiser__model__{module}__call0__{part}.npy"),
        ))
    };

    let w3 = w.tripled().unwrap();
    assert_close(
        &flat(&w3.x),
        &flat(&m("global_root_hybrid_proj", "in0")),
        1e-6,
        "anchor: x tripled",
    );

    // --- stage 1: global root ------------------------------------------------
    let (global_root, latent) = denoiser.split_hybrid(&w3.x).unwrap();
    let bs = w3.x.shape()[0];
    let nt = w3.x.shape()[1];
    let nf = nt * 4;

    let token_index = {
        let ti =
            ops::broadcast_to(ops::arange::<i32, i32>(None, nt, None).unwrap(), &[bs, nt]).unwrap();
        let origin = ops::floor_divide(&w3.history_len, Array::from_slice(&[4i32], &[])).unwrap();
        ops::subtract(&ti, origin.expand_dims(1).unwrap()).unwrap()
    };

    // no constraints: the reference substitutes zeros and still projects them
    let z = Array::zeros::<f32>(&[bs, nf, mr.motion_rep_dim() as i32]).unwrap();
    let observed_root = mr.extract_root(&z).unwrap();
    let observed_body = mr.extract_body(&z).unwrap();
    let root_mask = mr.extract_root(&z).unwrap();
    let keep = ops::subtract(scalar(1.0), &root_mask).unwrap();
    let gr_infilled = ops::add(
        ops::multiply(&global_root, &keep).unwrap(),
        ops::multiply(&observed_root, &root_mask).unwrap(),
    )
    .unwrap();
    let root_flat = gr_infilled.reshape(&[bs, nt, 20]).unwrap();
    let x_ie = ops::concatenate(
        &[
            &latent,
            &root_flat,
            &observed_body.reshape(&[bs, nt, -1]).unwrap(),
            &z.reshape(&[bs, nt, -1]).unwrap(),
        ],
        -1,
    )
    .unwrap();
    assert_close(
        &flat(&x_ie),
        &flat(&m("global_root_hybrid_constraints_proj", "in0")),
        1e-5,
        "anchor: x_infilled_extended",
    );

    let future_gate = w3.future_token_mask.expand_dims(-1).unwrap();
    let future_token_proj = ops::multiply(
        Linear::new(&mut store, "denoiser.future_constraints_proj", true)
            .unwrap()
            .forward(&x_ie.try_index((Ellipsis, 128..)).unwrap())
            .unwrap(),
        &future_gate,
    )
    .unwrap();
    let gen_gate = w3.generation_token_mask.expand_dims(-1).unwrap();
    let hist_gate = w3.history_token_mask.expand_dims(-1).unwrap();

    let g_proj = Linear::new(&mut store, "denoiser.global_root_hybrid_proj", true).unwrap();
    let g_cproj = Linear::new(
        &mut store,
        "denoiser.global_root_hybrid_constraints_proj",
        true,
    )
    .unwrap();
    assert_close(
        &flat(&g_proj.forward(&w3.x).unwrap()),
        &flat(&m("global_root_hybrid_proj", "out")),
        1e-4,
        "anchor: global_root_hybrid_proj",
    );
    assert_close(
        &flat(&g_cproj.forward(&x_ie).unwrap()),
        &flat(&m("global_root_hybrid_constraints_proj", "out")),
        1e-4,
        "anchor: global_root_hybrid_constraints_proj",
    );
    let root_stage_input = ops::add(
        ops::add(
            ops::multiply(g_cproj.forward(&x_ie).unwrap(), &gen_gate).unwrap(),
            ops::multiply(g_proj.forward(&w3.x).unwrap(), &hist_gate).unwrap(),
        )
        .unwrap(),
        &future_token_proj,
    )
    .unwrap();
    assert_close(
        &flat(&root_stage_input),
        &flat(&m("root_model", "in0")),
        1e-4,
        "anchor: root stage input",
    );

    let root_model = TransformerEncoderBlock::new(
        &mut store,
        "denoiser.backbone.root_model",
        1024,
        8,
        8,
        20,
        1,
        true,
    )
    .unwrap();
    let pad = ops::logical_or(
        ops::logical_or(&w3.history_token_mask, &w3.generation_token_mask).unwrap(),
        &w3.future_token_mask,
    )
    .unwrap();
    let root_pred = root_model
        .forward(
            &root_stage_input,
            &pad,
            &w3.text_feat,
            &w3.timesteps,
            w3.first_heading_angle.as_ref(),
            Some(&token_index),
        )
        .unwrap();
    assert_close(
        &flat(&root_pred),
        &flat(&m("root_model", "out")),
        5e-4,
        "anchor: root_model",
    );

    // --- stage 2: body --------------------------------------------------------
    let root_pred_f = root_pred.reshape(&[bs, nf, 5]).unwrap();
    let gen_frame = w3.generation_mask.expand_dims(-1).unwrap();
    let hist_frame = w3.history_mask.expand_dims(-1).unwrap();
    let root_pred_f = ops::add(
        ops::multiply(&root_pred_f, &gen_frame).unwrap(),
        ops::multiply(&gr_infilled, &hist_frame).unwrap(),
    )
    .unwrap();
    let lengths = ops::add(&w3.history_len, &w3.generation_len).unwrap();
    let local_root = mr
        .global_root_to_local_root(&root_pred_f, true, Some(&lengths))
        .unwrap();
    let x_new =
        ops::concatenate(&[&local_root.reshape(&[bs, nt, 16]).unwrap(), &latent], -1).unwrap();
    // `global_root_to_local_root` is trig-based (heading rotation + recenter),
    // so its f32 roundoff sits ~1e-5 relative rather than the pure-linear
    // stages' ~1e-6; everything downstream inherits it.
    assert_close(
        &flat(&x_new),
        &flat(&m("local_root_hybrid_proj", "in0")),
        1e-4,
        "anchor: x_new",
    );

    let l_proj = Linear::new(&mut store, "denoiser.local_root_hybrid_proj", true).unwrap();
    let l_cproj = Linear::new(
        &mut store,
        "denoiser.local_root_hybrid_constraints_proj",
        true,
    )
    .unwrap();
    assert_close(
        &flat(&l_proj.forward(&x_new).unwrap()),
        &flat(&m("local_root_hybrid_proj", "out")),
        1e-4,
        "anchor: local_root_hybrid_proj",
    );
    let x_new_ext = ops::concatenate(
        &[
            &x_new,
            &observed_body.reshape(&[bs, nt, -1]).unwrap(),
            &z.reshape(&[bs, nt, -1]).unwrap(),
        ],
        -1,
    )
    .unwrap();
    // concat includes local_root, so it inherits the geometry roundoff above
    assert_close(
        &flat(&x_new_ext),
        &flat(&m("local_root_hybrid_constraints_proj", "in0")),
        1e-4,
        "anchor: x_new_extended",
    );
    assert_close(
        &flat(&l_cproj.forward(&x_new_ext).unwrap()),
        &flat(&m("local_root_hybrid_constraints_proj", "out")),
        1e-4,
        "anchor: local_root_hybrid_constraints_proj",
    );
    let body_stage_input = ops::add(
        ops::add(
            ops::multiply(l_cproj.forward(&x_new_ext).unwrap(), &gen_gate).unwrap(),
            ops::multiply(l_proj.forward(&x_new).unwrap(), &hist_gate).unwrap(),
        )
        .unwrap(),
        &future_token_proj,
    )
    .unwrap();
    assert_close(
        &flat(&body_stage_input),
        &flat(&m("body_model", "in0")),
        1e-4,
        "anchor: body stage input",
    );

    let body_model = TransformerEncoderBlock::new(
        &mut store,
        "denoiser.backbone.body_model",
        1024,
        8,
        8,
        128,
        1,
        true,
    )
    .unwrap();
    let body_pred = body_model
        .forward(
            &body_stage_input,
            &pad,
            &w3.text_feat,
            &w3.timesteps,
            w3.first_heading_angle.as_ref(),
            Some(&token_index),
        )
        .unwrap();
    assert_close(
        &flat(&body_pred),
        &flat(&m("body_model", "out")),
        5e-4,
        "anchor: body_model",
    );
}
