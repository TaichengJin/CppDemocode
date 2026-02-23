#include <iostream>
#include <stdexcept>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

#include "infer/InferEngine.h"
#include "infer/postprocess_rtdetr.h"
#include "common/visualize.h"
#include "video/ffmpeg_video_source.h"

enum class ExitCode : int {
    Ok = 0,
    InputError = 1,
    OrtError = 2,
    RuntimeError = 3
};

struct InputError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

int main() {
    try {
        const std::wstring model_path = L"models\\rtdetr-l.onnx";
        //const std::string image_path = "assets\\test_frame.png";

        InferEngine::Options opt;
        opt.input_w = 640;
        opt.input_h = 640;
        opt.use_cuda = false;

        InferEngine engine(opt);
        engine.LoadModel(model_path);

        //cv::Mat bgr = cv::imread(image_path, cv::IMREAD_COLOR);
        //if (bgr.empty()) {
        //    throw InputError("Failed to read image: " + image_path);
        //}
        //auto result = engine.Run(bgr);
        /*PostprocessOptions pp;
        pp.score_thresh = 0.49f;

        auto dets = PostprocessRTDETR(
            result.outputs[0],
            engine.InputW(), engine.InputH(),
            result.lb,
            result.orig_w, result.orig_h,
            pp
        );*/

        // ��������ͷ����

        video::FFmpegVideoSource src;
        const std::string url = "rtsp://admin:%40%40admin7434@192.168.1.100:554/0/onvif/profile1/media.smp"; // TODO: �������
        src.Open(url);

        PostprocessOptions pp;
        pp.score_thresh = 0.6f;

        cv::namedWindow("RT-DETR Live", cv::WINDOW_NORMAL);

        video::Frame frame;
        while (true) {
            // ��һ֡��BGR Mat��
            if (!src.Read(frame)) {

                std::cerr << "[Video] Read failed / EOF / disconnected.\n";
                break; // Phase1: ��ֱ���˳�������������������
            }

            // ����ѡ����ʱ���� FPS
            auto t0 = std::chrono::high_resolution_clock::now();

            // ����
            auto result = engine.Run(frame.bgr);

            // ����
            auto dets = PostprocessRTDETR(
                result.outputs[0],
                engine.InputW(), engine.InputH(),
                result.lb,
                result.orig_w, result.orig_h,
                pp
            );

            // ���ӻ�������ֱ���ڵ�ǰ֡�ϻ������� clone �Ķ��⿪����
            cv::Mat vis = frame.bgr.clone();   // Phase1��ȫд��������ȾԭͼҲ����
            DrawDetections(vis, dets);

            // ����ѡ�����ӵ�����Ϣ
            auto t1 = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            cv::putText(vis, cv::format("Inference+PP: %.1f ms", ms),
                { 10, 30 }, cv::FONT_HERSHEY_SIMPLEX, 0.8, { 0,255,0 }, 2);

            cv::imshow("RT-DETR Live", vis);

            // waitKey(1) �ô���ˢ�� + �����������
            int key = cv::waitKey(1);
            if (key == 27 || key == 'q' || key == 'Q') {
                break;
            }
        }

        src.Close();
        cv::destroyAllWindows();
        return 0;
        



        /*if (!DrawAndSaveDetections(bgr, dets, "assets\\vis_result.jpg")) {
            throw std::runtime_error("Failed to write assets\\vis_result.jpg");
        }*/

        std::cout << "[INFO] Done.\n";
        return static_cast<int>(ExitCode::Ok);
    }
    catch (const Ort::Exception& e) {
        std::cerr << "[ORT ERROR] " << e.what() << "\n";
        return static_cast<int>(ExitCode::OrtError);
    }
    catch (const InputError& e) {
        std::cerr << "[INPUT ERROR] " << e.what() << "\n";
        return static_cast<int>(ExitCode::InputError);
    }
    catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << "\n";
        return static_cast<int>(ExitCode::RuntimeError);
    }
}
