// Differential-test probe: replicates MotionCorrection's pybind
// `correct_motion` marshaling (BindingsPython.cpp) minus pybind, over a flat
// little-endian binary record so the Rust port can be diffed against the
// real C++ on identical inputs.
//
// Record layout (all little-endian):
//   magic  "MCP1" (4 bytes)
//   i32 T, i32 J, i32 has_double_ankle_joints
//   f32 contact_threshold, f32 root_margin
//   i32[J]                 joint_parents
//   f32[J*3]               joint_ref_translations
//   f32[J*4]               joint_ref_rotations (w x y z)
//   f32[T*3]               rootTranslations        (in/out)
//   f32[T*J*4]             jointRotations          (in/out, w x y z)
//   f32[T*3]               rootTranslationsTarget
//   f32[T*J*4]             jointRotationsTarget
//   f32[T]                 fullPoseMask
//   f32[T]                 leftHandMask
//   f32[T]                 rightHandMask
//   f32[T]                 leftFootMask
//   f32[T]                 rightFootMask
//   f32[T]                 rootMask
//   f32[T*4]               contacts
// Output: the same record with rootTranslations/jointRotations replaced by
// the corrected values.

#include "AnimProcessing/Utility.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {

float strip_nan_inf(float x)
{
    if (std::isnan(x)) return 0;
    if (std::isinf(x)) return 0;
    return x;
}

struct Reader {
    const uint8_t* p;
    size_t n;
    size_t off = 0;

    void read(void* dst, size_t bytes)
    {
        if (off + bytes > n) { std::cerr << "probe: truncated input\n"; exit(2); }
        std::memcpy(dst, p + off, bytes);
        off += bytes;
    }
    int32_t i32() { int32_t v; read(&v, 4); return v; }
    float f32() { float v; read(&v, 4); return v; }
    std::vector<float> f32v(size_t k)
    {
        std::vector<float> v(k);
        if (k) read(v.data(), k * 4);
        return v;
    }
    std::vector<int32_t> i32v(size_t k)
    {
        std::vector<int32_t> v(k);
        if (k) read(v.data(), k * 4);
        return v;
    }
};

void write_all(const std::string& path, const std::vector<uint8_t>& bytes)
{
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) { std::cerr << "probe: cannot write " << path << "\n"; exit(2); }
    fwrite(bytes.data(), 1, bytes.size(), f);
    fclose(f);
}

} // namespace

int main(int argc, char** argv)
{
    if (argc != 3)
    {
        std::cerr << "usage: probe <in.mcp> <out.mcp>\n";
        return 2;
    }

    FILE* f = fopen(argv[1], "rb");
    if (!f) { std::cerr << "probe: cannot open " << argv[1] << "\n"; return 2; }
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> bytes(size);
    fread(bytes.data(), 1, size, f);
    fclose(f);

    Reader r{bytes.data(), bytes.size()};
    char magic[5] = {0};
    r.read(magic, 4);
    if (std::string(magic) != "MCP1") { std::cerr << "probe: bad magic\n"; return 2; }

    int32_t T = r.i32();
    int32_t J = r.i32();
    bool has_double_ankle_joints = r.i32() != 0;
    float contact_threshold = r.f32();
    float root_margin = r.f32();

    auto joint_parents_i = r.i32v(J);
    auto ref_t = r.f32v(J * 3);
    auto ref_r = r.f32v(J * 4);
    auto rootT = r.f32v(T * 3);
    auto jointR = r.f32v((size_t)T * J * 4);
    auto rootT_target = r.f32v(T * 3);
    auto jointR_target = r.f32v((size_t)T * J * 4);
    auto fullPoseMask = r.f32v(T);
    auto leftHandMask = r.f32v(T);
    auto rightHandMask = r.f32v(T);
    auto leftFootMask = r.f32v(T);
    auto rightFootMask = r.f32v(T);
    auto rootMask = r.f32v(T);
    auto contacts = r.f32v((size_t)T * 4);

    std::vector<int> joint_parents_vec(joint_parents_i.begin(), joint_parents_i.end());

    // ---- begin: literal copy of BindingsPython::correct_motion marshaling ----
    std::vector<Math::Transform> defaultPose(J);
    for (int j = 0; j < J; ++j)
    {
        defaultPose[j].SetTranslation(Math::Vector(ref_t[3*j], ref_t[3*j+1], ref_t[3*j+2]));
        defaultPose[j].SetRotation(Math::Quaternion(ref_r[4*j+1], ref_r[4*j+2], ref_r[4*j+3], ref_r[4*j]));
    }

    size_t num_joints = J;
    size_t gen_length = T;

    std::vector<Animation::ContactInfo> endEffectorPins(4);
    endEffectorPins[0].jointIndex = 0; // patched below from masks; indices passed via parents? no:
    // (the probe has no separate ee index fields; the real skeleton's hand/foot
    // indices are what the Rust side passes too — encode them as the parents of
    // nothing? Simpler: probe reads them from the record tail appended by the
    // generator, see below.)

    // -- the four ee indices + toe discovery need real indices; the generator
    //    appends them after `contacts`: i32 lhand, rhand, lfoot, rfoot.
    int32_t left_hand_idx = r.i32();
    int32_t right_hand_idx = r.i32();
    int32_t left_foot_idx = r.i32();
    int32_t right_foot_idx = r.i32();

    endEffectorPins[0].jointIndex = left_hand_idx;
    endEffectorPins[0].hintOffset = Math::Vector(0.0f, 0.0f, -0.1f);
    endEffectorPins[1].jointIndex = right_hand_idx;
    endEffectorPins[1].hintOffset = Math::Vector(0.0f, 0.0f, -0.1f);
    endEffectorPins[2].jointIndex = left_foot_idx;
    endEffectorPins[2].hintOffset = Math::Vector(0.0f, 0.0f, 0.1f);
    endEffectorPins[3].jointIndex = right_foot_idx;
    endEffectorPins[3].hintOffset = Math::Vector(0.0f, 0.0f, 0.1f);

    for (size_t i = 0; i < gen_length; ++i)
    {
        endEffectorPins[0].contactMask.push_back((1.0f - fullPoseMask.at(i)) * leftHandMask.at(i));
        endEffectorPins[1].contactMask.push_back((1.0f - fullPoseMask.at(i)) * rightHandMask.at(i));
        endEffectorPins[2].contactMask.push_back((1.0f - fullPoseMask.at(i)) * leftFootMask.at(i));
        endEffectorPins[3].contactMask.push_back((1.0f - fullPoseMask.at(i)) * rightFootMask.at(i));
    }

    std::vector<Animation::ContactInfo> contactInfo(2);

    auto footTranslation = Animation::JointLocalToGlobal(
        joint_parents_vec, right_foot_idx, defaultPose).GetTranslation();
    contactInfo[0].jointIndex = right_foot_idx;
    contactInfo[0].hintOffset = Math::Vector(0.0f, 0.0f, 0.1f);
    contactInfo[0].minHeight = footTranslation.GetY();

    footTranslation = Animation::JointLocalToGlobal(
        joint_parents_vec, left_foot_idx, defaultPose).GetTranslation();
    contactInfo[1].jointIndex = left_foot_idx;
    contactInfo[1].hintOffset = Math::Vector(0.0f, 0.0f, 0.1f);
    contactInfo[1].minHeight = footTranslation.GetY();

    auto& rContacts = contactInfo[0].contactMask;
    auto& lContacts = contactInfo[1].contactMask;
    rContacts.resize(fullPoseMask.size());
    lContacts.resize(fullPoseMask.size());
    for (int i = 0; i < (int)fullPoseMask.size(); ++i)
    {
        rContacts[i] = rightFootMask.at(i) ? 0 : contacts.at(4 * i + 2);
        lContacts[i] = leftFootMask.at(i) ? 0 : contacts.at(4 * i + 0);
        rContacts[i] = std::min((rightFootMask.at(i) ? 0 : contacts.at(4 * i + 3)) + rContacts[i], 1.0f);
        lContacts[i] = std::min((leftFootMask.at(i) ? 0 : contacts.at(4 * i + 1)) + lContacts[i], 1.0f);
    }

    int left_toe_idx = -1;
    int right_toe_idx = -1;
    for (int i = 0; i < num_joints; ++i)
    {
        if (joint_parents_vec[i] == left_foot_idx) left_toe_idx = i;
        if (joint_parents_vec[i] == right_foot_idx) right_toe_idx = i;
    }

    if (left_toe_idx != -1 && right_toe_idx != -1)
    {
        auto toeTranslation = Animation::JointLocalToGlobal(
            joint_parents_vec, right_toe_idx, defaultPose).GetTranslation();
        contactInfo.resize(4);
        contactInfo[2].jointIndex = right_toe_idx;
        contactInfo[2].contactType = Animation::kOneBone;
        contactInfo[2].minHeight = toeTranslation.GetY();
        contactInfo[3].jointIndex = left_toe_idx;
        contactInfo[3].contactType = Animation::kOneBone;
        contactInfo[3].minHeight = toeTranslation.GetY();

        auto& rToeContacts = contactInfo[2].contactMask;
        auto& lToeContacts = contactInfo[3].contactMask;
        rToeContacts.resize(fullPoseMask.size());
        lToeContacts.resize(fullPoseMask.size());
        for (int i = 0; i < (int)fullPoseMask.size(); ++i)
        {
            rToeContacts[i] = rightFootMask.at(i) ? 0 : contacts.at(4 * i + 3);
            lToeContacts[i] = leftFootMask.at(i) ? 0 : contacts.at(4 * i + 1);
        }
    }

    auto setTransforms = [gen_length, num_joints](
        std::vector< std::vector<Math::Transform> >& poses,
        const std::vector<float>& rootTranslations,
        const std::vector<float>& jointRotations)
    {
        for (size_t fr = 0; fr < gen_length; ++fr)
        {
            poses[fr][0].SetTranslation({
                strip_nan_inf(rootTranslations.at(3*fr+0)),
                strip_nan_inf(rootTranslations.at(3*fr+1)),
                strip_nan_inf(rootTranslations.at(3*fr+2))});
        }
        for (size_t fr = 0; fr < gen_length; ++fr)
        {
            for (size_t j = 0; j < num_joints; ++j)
            {
                Math::Quaternion q(
                    strip_nan_inf(jointRotations.at(4 * (num_joints * fr + j) + 1)),
                    strip_nan_inf(jointRotations.at(4 * (num_joints * fr + j) + 2)),
                    strip_nan_inf(jointRotations.at(4 * (num_joints * fr + j) + 3)),
                    strip_nan_inf(jointRotations.at(4 * (num_joints * fr + j) + 0)));
                q.Normalize();
                poses[fr][j].SetRotation(q);
            }
        }
    };

    std::vector< std::vector<Math::Transform> > posesFixed(gen_length, defaultPose);
    setTransforms(posesFixed, rootT, jointR);
    std::vector< std::vector<Math::Transform> > posesTarget(gen_length, defaultPose);
    setTransforms(posesTarget, rootT_target, jointR_target);

    std::vector<float> fullPoseMask_vec(fullPoseMask);
    std::vector<float> rootMask_vec(rootMask);

    Animation::CorrectMotion(
        posesFixed, posesTarget, fullPoseMask_vec, rootMask_vec,
        contactInfo, endEffectorPins, joint_parents_vec, defaultPose,
        contact_threshold, root_margin, has_double_ankle_joints);

    for (size_t fr = 0; fr < gen_length; ++fr)
    {
        auto t = posesFixed[fr][0].GetTranslation();
        rootT[3*fr+0] = t.GetX();
        rootT[3*fr+1] = t.GetY();
        rootT[3*fr+2] = t.GetZ();
    }
    for (size_t fr = 0; fr < gen_length; ++fr)
    {
        for (size_t j = 0; j < num_joints; ++j)
        {
            auto q = posesFixed[fr][j].GetRotation();
            jointR[4 * (num_joints * fr + j) + 0] = ((float*)&q)[3];
            jointR[4 * (num_joints * fr + j) + 1] = ((float*)&q)[0];
            jointR[4 * (num_joints * fr + j) + 2] = ((float*)&q)[1];
            jointR[4 * (num_joints * fr + j) + 3] = ((float*)&q)[2];
        }
    }
    // ---- end marshaling copy ----

    // rewrite the record with updated in/out arrays
    std::vector<uint8_t> out;
    auto put = [&out](const void* p, size_t n)
    {
        const uint8_t* b = (const uint8_t*)p;
        out.insert(out.end(), b, b + n);
    };
    auto put_i32 = [&](int32_t v) { put(&v, 4); };
    auto put_f32v = [&](const std::vector<float>& v) { put(v.data(), v.size() * 4); };
    auto put_i32v = [&](const std::vector<int32_t>& v) { put(v.data(), v.size() * 4); };

    put("MCP1", 4);
    put_i32(T); put_i32(J); put_i32(has_double_ankle_joints ? 1 : 0);
    put_i32(0); put_i32(0); // (threshold, margin were f32 -- rewrite properly)
    out.resize(out.size() - 8);
    put(&contact_threshold, 4); put(&root_margin, 4);
    put_i32v(joint_parents_i);
    put_f32v(ref_t);
    put_f32v(ref_r);
    put_f32v(rootT);
    put_f32v(jointR);
    put_f32v(rootT_target);
    put_f32v(jointR_target);
    put_f32v(fullPoseMask);
    put_f32v(leftHandMask);
    put_f32v(rightHandMask);
    put_f32v(leftFootMask);
    put_f32v(rightFootMask);
    put_f32v(rootMask);
    put_f32v(contacts);
    put_i32(left_hand_idx); put_i32(right_hand_idx);
    put_i32(left_foot_idx); put_i32(right_foot_idx);

    write_all(argv[2], out);
    std::cerr << "probe: ok T=" << T << " J=" << J << "\n";
    return 0;
}
