# MotionCorrection differential probe

`probe.cpp` is the bridge that lets `tests/postprocess.rs` diff the Rust
MotionCorrection port against the **real upstream C++** on identical inputs.

It is a standalone binary that replicates `MotionCorrection`'s pybind
`correct_motion` marshaling (`BindingsPython.cpp`) minus pybind: it reads one
flat little-endian `MCP1` record (layout documented in the `probe.cpp` header),
runs the *unmodified* upstream `TrajectoryCorrector::correctMotion` pipeline,
and writes the same record back with the corrected root translations and joint
rotations.

## Build

The probe compiles against upstream sources — it does not vendor them. Check
out `nv-tlabs/ardy` (Apache-2.0) and point the include path at the MotionCorrection
tree:

```sh
git clone https://github.com/nv-tlabs/ardy /tmp/ardy-ref   # any revision works;
                                                           # 693f74d was used for the M6 run
mkdir -p /tmp/mc-probe && cd /tmp/mc-probe
cp <this repo>/scripts/mc_probe/probe.cpp .
cp -r /tmp/ardy-ref/MotionCorrection/src/cpp src

clang++ -std=c++17 -O1 \
    -I/tmp/mc-probe/src/cpp -I/opt/homebrew/include/eigen3 \
    probe.cpp \
    src/cpp/AnimProcessing/TrajectoryCorrector.cpp \
    src/cpp/AnimProcessing/Utility.cpp \
    -o probe
```

(Eigen 3.4+ via Homebrew; adjust `-I` for your install.)

## Run the differential tests

```sh
ARDY_MC_PROBE=/tmp/mc-probe/probe cargo test --test postprocess
```

The tests generate synthetic LCG records over the real `cskel27` rig (no
NVIDIA data), feed each to the probe and to the Rust port, and assert
max-abs diff ≤ 1e-5 on the corrected root and rotations. Without the probe
binary the tests pass vacuously.

## Note on a verified upstream quirk

The probe is what pinned down that upstream's
`TrajectoryCorrector::computeDiffMats` velocity weighting is dead code: its
`multVelWeights` computes weighted triplets into a local vector but never
calls `setFromTriplets`, so `V` stays unweighted (verified by calling
`computeDiffMats` directly with distinctive weights and printing `V`). The
Rust port mirrors the *observed* behavior and documents this in
`DiffMats`'s doc comment.
