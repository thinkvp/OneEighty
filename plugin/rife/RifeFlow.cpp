#ifndef NOMINMAX
    #define NOMINMAX
#endif
#include <windows.h>

#include "RifeFlow.h"

// ncnn (vcpkg ncnn[vulkan]; include dir ...\include\ncnn)
#include <command.h>
#include <gpu.h>
#include <layer.h>
#include <net.h>
#include <pipeline.h>

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <mutex>

namespace oe::rife
{
namespace
{

#include "RifeWarpShaders.inl"

// RIFE's custom "rife.Warp" layer (backward warp with bilinear sampling),
// adapted from rife-ncnn-vulkan's warp.cpp (MIT, Copyright (c) 2020 nihui).
class Warp : public ncnn::Layer
{
public:
    Warp() { support_vulkan = true; }

    int create_pipeline(const ncnn::Option& opt) override
    {
        if (!vkdev)
            return 0;
        const std::vector<ncnn::vk_specialization_type> spec;
        pipeline_ = Make(kWarpComp, spirv1_, mu1_, opt, spec);
        pipeline4_ = Make(kWarpPack4Comp, spirv4_, mu4_, opt, spec); // current ncnn packs GPU blobs by 1 or 4
        return 0;
    }

    int destroy_pipeline(const ncnn::Option&) override
    {
        delete pipeline_;
        delete pipeline4_;
        pipeline_ = pipeline4_ = nullptr;
        return 0;
    }

    int forward(const std::vector<ncnn::Mat>& bottom, std::vector<ncnn::Mat>& top, const ncnn::Option& opt) const override
    {
        const ncnn::Mat& image = bottom[0];
        const ncnn::Mat& flow = bottom[1];
        const int w = image.w, h = image.h, channels = image.c;
        ncnn::Mat& out = top[0];
        out.create(w, h, channels);
        if (out.empty())
            return -100;
        for (int q = 0; q < channels; q++)
        {
            float* o = out.channel(q);
            const ncnn::Mat img = image.channel(q);
            const float* fx = flow.channel(0);
            const float* fy = flow.channel(1);
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++, ++o, ++fx, ++fy)
                {
                    const float sx = x + *fx, sy = y + *fy;
                    int x0 = (int)std::floor(sx), y0 = (int)std::floor(sy);
                    int x1 = x0 + 1, y1 = y0 + 1;
                    x0 = std::clamp(x0, 0, w - 1);
                    y0 = std::clamp(y0, 0, h - 1);
                    x1 = std::clamp(x1, 0, w - 1);
                    y1 = std::clamp(y1, 0, h - 1);
                    const float al = sx - x0, be = sy - y0;
                    const float v4 = img.row(y0)[x0] * (1 - al) + img.row(y0)[x1] * al;
                    const float v5 = img.row(y1)[x0] * (1 - al) + img.row(y1)[x1] * al;
                    *o = v4 * (1 - be) + v5 * be;
                }
        }
        (void)opt;
        return 0;
    }

    int forward(const std::vector<ncnn::VkMat>& bottom, std::vector<ncnn::VkMat>& top, ncnn::VkCompute& cmd,
                const ncnn::Option& opt) const override
    {
        const ncnn::VkMat& image = bottom[0];
        const ncnn::VkMat& flow = bottom[1];
        ncnn::VkMat& out = top[0];
        out.create(image.w, image.h, image.c, image.elemsize, image.elempack, opt.blob_vkallocator);
        if (out.empty())
            return -100;
        std::vector<ncnn::VkMat> bindings = {image, flow, out};
        std::vector<ncnn::vk_constant_type> constants(4);
        constants[0].i = out.w;
        constants[1].i = out.h;
        constants[2].i = out.c;
        constants[3].i = (int)out.cstep;
        const ncnn::Pipeline* p = image.elempack == 4 ? pipeline4_ : pipeline_;
        cmd.record_pipeline(p, bindings, constants, out);
        return 0;
    }

private:
    ncnn::Pipeline* Make(const char* src, std::vector<uint32_t>& spirv, ncnn::Mutex& mu, const ncnn::Option& opt,
                         const std::vector<ncnn::vk_specialization_type>& spec)
    {
        {
            ncnn::MutexLockGuard guard(mu);
            if (spirv.empty())
                ncnn::compile_spirv_module(src, (int)std::strlen(src), opt, spirv);
        }
        auto* p = new ncnn::Pipeline(vkdev);
        p->set_optimal_local_size_xyz();
        p->create(spirv.data(), spirv.size() * 4, spec);
        return p;
    }

    ncnn::Pipeline* pipeline_ = nullptr;
    ncnn::Pipeline* pipeline4_ = nullptr;
    static std::vector<uint32_t> spirv1_, spirv4_;
    static ncnn::Mutex mu1_, mu4_;
};
std::vector<uint32_t> Warp::spirv1_, Warp::spirv4_;
ncnn::Mutex Warp::mu1_, Warp::mu4_;

ncnn::Layer* WarpCreator(void*) { return new Warp; }

bool Fail(std::string* err, const std::string& msg)
{
    if (err)
        *err = msg;
    return false;
}

// One frame as RIFE input: planar RGB, 0..1 (clamped), zero-padded to (wp, hp).
ncnn::Mat ToInput(const cv::Mat& small, const ChannelOrder& o, int wp, int hp)
{
    ncnn::Mat m(wp, hp, 3);
    m.fill(0.0f);
    const bool isFloat = small.depth() == CV_32F;
    const int ch[3] = {o.r, o.g, o.b};
    cv::parallel_for_(cv::Range(0, small.rows), [&](const cv::Range& rows) {
        for (int q = 0; q < 3; ++q)
        {
            float* dst = m.channel(q);
            for (int y = rows.start; y < rows.end; ++y)
            {
                float* d = dst + (size_t)y * wp;
                if (isFloat)
                {
                    const float* s = small.ptr<float>(y) + ch[q];
                    for (int x = 0; x < small.cols; ++x, s += 4)
                        d[x] = std::clamp(*s, 0.0f, 1.0f);
                }
                else
                {
                    const uchar* s = small.ptr<uchar>(y) + ch[q];
                    for (int x = 0; x < small.cols; ++x, s += 4)
                        d[x] = *s * (1.0f / 255.0f);
                }
            }
        }
    });
    return m;
}

// Downloaded blob -> fp32, one element per channel.
ncnn::Mat Unpack(const ncnn::Mat& m, const ncnn::Option& opt)
{
    ncnn::Mat a = m;
    if (a.elempack != 1)
    {
        ncnn::Mat b;
        ncnn::convert_packing(a, b, 1, opt);
        a = b;
    }
    if (a.elembits() == 16)
    {
        ncnn::Mat b;
        ncnn::cast_float16_to_float32(a, b, opt);
        a = b;
    }
    return a;
}

class EngineImpl final : public Engine
{
public:
    bool Init(const std::wstring& modelDir, std::string* err)
    {
        if (ncnn::create_gpu_instance() != 0 || ncnn::get_gpu_count() <= 0)
            return Fail(err, "no Vulkan GPU");
        vkdev_ = ncnn::get_gpu_device(ncnn::get_default_gpu_index());
        if (!vkdev_)
            return Fail(err, "no Vulkan device");
        ncnn::Option opt;
        opt.use_vulkan_compute = true;
        opt.use_fp16_packed = true;
        opt.use_fp16_storage = true;
        opt.use_fp16_arithmetic = false;
        opt.use_int8_storage = false;
        opt.num_threads = 1;
        net_.opt = opt;
        net_.set_vulkan_device(vkdev_);
        net_.register_custom_layer("rife.Warp", WarpCreator);
        for (const wchar_t* name : {L"flownet.param", L"flownet.bin"})
        {
            const std::wstring path = modelDir + L"\\" + name;
            FILE* fp = nullptr;
            if (_wfopen_s(&fp, path.c_str(), L"rb") != 0 || !fp)
                return Fail(err, "RIFE model missing (rife-v4.6\\flownet.param / .bin next to the plugin)");
            const int r = std::wcsstr(name, L".param") ? net_.load_param(fp) : net_.load_model(fp);
            std::fclose(fp);
            if (r != 0)
                return Fail(err, "RIFE model failed to load");
        }
        name_ = vkdev_->info.device_name();
        return true;
    }

    bool Estimate(const cv::Mat& a, const cv::Mat& b, const ChannelOrder& order, int maxPixels,
                  const std::vector<float>& ts, std::vector<Field>* out, std::string* err) override
    {
        if (a.empty() || b.empty() || a.size() != b.size() || a.type() != b.type())
            return Fail(err, "RIFE: frames differ in size or type");
        const int W = a.cols, H = a.rows;
        const double s = std::min(1.0, std::sqrt((double)maxPixels / ((double)W * H)));
        const int aw = std::max(32, (int)std::lround(W * s)), ah = std::max(32, (int)std::lround(H * s));
        cv::Mat sa = a, sb = b;
        if (aw != W || ah != H)
        {
            cv::resize(a, sa, cv::Size(aw, ah), 0, 0, cv::INTER_AREA);
            cv::resize(b, sb, cv::Size(aw, ah), 0, 0, cv::INTER_AREA);
        }
        const int wp = (aw + 31) / 32 * 32, hp = (ah + 31) / 32 * 32;
        const ncnn::Mat in0 = ToInput(sa, order, wp, hp), in1 = ToInput(sb, order, wp, hp);
        // only the GPU part is serialised: the CPU preparation above overlaps
        // with another render's inference (Premiere renders frames in parallel)
        std::lock_guard<std::mutex> lock(mu_);

        ncnn::VkAllocator* blob = vkdev_->acquire_blob_allocator();
        ncnn::VkAllocator* staging = vkdev_->acquire_staging_allocator();
        ncnn::Option opt = net_.opt;
        opt.blob_vkallocator = blob;
        opt.workspace_vkallocator = blob;
        opt.staging_vkallocator = staging;
        bool ok = true;
        out->assign(ts.size(), Field());
        {
            ncnn::VkCompute cmd(vkdev_);
            ncnn::VkMat in0g, in1g;
            cmd.record_upload(in0, in0g, opt); // casts to fp16 storage and packs as the network expects
            cmd.record_upload(in1, in1g, opt);
            for (size_t i = 0; i < ts.size() && ok; ++i)
            {
                ncnn::VkMat fg, mg;
                const ncnn::VkMat tg = Timestep(ts[i], wp, hp, cmd, opt);
                ncnn::Extractor ex = net_.create_extractor();
                ex.set_blob_vkallocator(blob);
                ex.set_workspace_vkallocator(blob);
                ex.set_staging_vkallocator(staging);
                ex.input("in0", in0g);
                ex.input("in1", in1g);
                ex.input("in2", tg);
                // final flows (4 channels) and the blend mask after the sigmoid: see
                // the end of flownet.param (out0 = warp(in0, flow[0:2]) * mask +
                // warp(in1, flow[2:4]) * (1 - mask))
                if (ex.extract("327", fg, cmd) != 0 || ex.extract("332", mg, cmd) != 0)
                {
                    ok = Fail(err, "RIFE: inference failed");
                    break;
                }
                ncnn::Mat fc, mc;
                cmd.record_download(fg, fc, opt); // back to fp32
                cmd.record_download(mg, mc, opt);
                if (cmd.submit_and_wait() != 0)
                {
                    ok = Fail(err, "RIFE: GPU submit failed");
                    break;
                }
                cmd.reset();
                Field& fld = (*out)[i];
                if (fc.c == 1 && fc.elempack == 4 && fc.elembits() == 32 && mc.c == 1 && mc.elempack == 1 &&
                    mc.elembits() == 32 && fc.w >= aw && fc.h >= ah && mc.w >= aw && mc.h >= ah)
                {
                    // the 4 flow channels arrive packed per pixel: exactly the float4 field the
                    // CUDA side wants, so hand the downloaded buffers over without converting
                    fld.flow = cv::Mat(ah, aw, CV_32FC4, fc.data, (size_t)fc.w * fc.elemsize);
                    fld.mask = cv::Mat(ah, aw, CV_32FC1, mc.data, (size_t)mc.w * mc.elemsize);
                    fld.keep = std::make_shared<std::pair<ncnn::Mat, ncnn::Mat>>(fc, mc);
                    continue;
                }
                const ncnn::Mat f = Unpack(fc, opt), m = Unpack(mc, opt); // other layouts: convert
                if (f.c < 4 || m.c < 1 || f.w < aw || f.h < ah)
                {
                    ok = Fail(err, "RIFE: unexpected output shape");
                    break;
                }
                fld.flow.create(ah, aw, CV_32FC4);
                fld.mask.create(ah, aw, CV_32FC1);
                for (int y = 0; y < ah; ++y)
                {
                    float* d = fld.flow.ptr<float>(y);
                    const float* c0 = f.channel(0).row(y);
                    const float* c1 = f.channel(1).row(y);
                    const float* c2 = f.channel(2).row(y);
                    const float* c3 = f.channel(3).row(y);
                    for (int x = 0; x < aw; ++x, d += 4)
                    {
                        d[0] = c0[x];
                        d[1] = c1[x];
                        d[2] = c2[x];
                        d[3] = c3[x];
                    }
                    std::memcpy(fld.mask.ptr<float>(y), m.channel(0).row(y), sizeof(float) * aw);
                }
            }
        }
        vkdev_->reclaim_blob_allocator(blob);
        vkdev_->reclaim_staging_allocator(staging);
        return ok;
    }

private:
    // RIFE's time input is a constant map the size of the frame. Sample times
    // repeat from frame to frame (one set for 2:1 cadences, a short cycle for
    // 59.94p -> 25p), so the uploaded maps are kept on the GPU and reused.
    ncnn::VkMat Timestep(float t, int wp, int hp, ncnn::VkCompute& cmd, const ncnn::Option& opt)
    {
        t = std::clamp(t, 0.0f, 1.0f);
        const long long key = ((long long)wp << 40) ^ ((long long)hp << 20) ^ (long long)std::lround(t * 100000.0f);
        for (auto& e : timesteps_)
            if (e.first == key)
                return e.second;
        if (!timestepAlloc_)
            timestepAlloc_ = std::make_unique<ncnn::VkBlobAllocator>(vkdev_);
        ncnn::Mat tm(wp, hp, 1);
        tm.fill(t);
        ncnn::Option o = opt;
        o.blob_vkallocator = timestepAlloc_.get();
        ncnn::VkMat g;
        cmd.record_upload(tm, g, o);
        if (timesteps_.size() >= kMaxTimesteps)
            timesteps_.erase(timesteps_.begin()); // oldest first
        timesteps_.emplace_back(key, g);
        return g;
    }

    static constexpr size_t kMaxTimesteps = 48; // ~4 MB each at 2 MP (fp16)
    std::unique_ptr<ncnn::VkBlobAllocator> timestepAlloc_;    // declared first: outlives the maps below
    std::vector<std::pair<long long, ncnn::VkMat>> timesteps_;
    std::mutex mu_;
    const ncnn::VulkanDevice* vkdev_ = nullptr;
    ncnn::Net net_;
    std::string name_;
};

} // namespace

std::unique_ptr<Engine> Engine::Create(const std::wstring& modelDir, std::string* err)
{
    auto e = std::make_unique<EngineImpl>();
    if (!e->Init(modelDir, err))
        return nullptr;
    return e;
}

std::wstring DefaultModelDir()
{
    HMODULE h = nullptr;
    wchar_t path[MAX_PATH] = {0};
    if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           reinterpret_cast<LPCWSTR>(&DefaultModelDir), &h) &&
        GetModuleFileNameW(h, path, MAX_PATH))
    {
        std::wstring dir(path);
        const size_t slash = dir.find_last_of(L"\\/");
        if (slash != std::wstring::npos)
            dir.resize(slash);
        return dir + L"\\rife-v4.6";
    }
    return L"rife-v4.6";
}

} // namespace oe::rife
