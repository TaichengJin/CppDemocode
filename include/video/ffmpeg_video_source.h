#pragma once
#include <string>
#include "video/video_source.h"

// 前置声明，避免在头文件里引入大量 FFmpeg 头（减少编译污染）
struct AVFormatContext;
struct AVCodecContext;
struct AVFrame;
struct AVPacket;
struct SwsContext;

namespace video {

    class FFmpegVideoSource final : public IVideoSource {
    public:
        FFmpegVideoSource();
        ~FFmpegVideoSource() override;

        void Open(const std::string& url) override;
        bool Read(Frame& out) override;
        void Close() override;

    private:
        void Cleanup();
        void InitScalerIfNeeded(int src_w, int src_h, int src_pix_fmt);

    private:
        AVFormatContext* fmt_ = nullptr;
        AVCodecContext* dec_ = nullptr;
        AVFrame* frame_ = nullptr;   // 解码后的原始帧（通常 YUV）
        AVPacket* pkt_ = nullptr;     // 压缩数据包（H264/H265）
        SwsContext* sws_ = nullptr;     // 像素格式转换（YUV -> BGR）

        int video_stream_index_ = -1;

        // 缓存输出 BGR 的缓冲区信息（避免反复分配）
        cv::Mat bgr_;
    };

} // namespace video
