#include "LaneDetector.hpp"
#include <algorithm>
#include <iostream>


static bool try_open_gst(cv::VideoCapture& cap, const std::string& pipeline) {
    std::cout << "[CAMERA] Try pipeline:\n" << pipeline << "\n";
    if (!cap.open(pipeline, cv::CAP_GSTREAMER)) {
        std::cerr << "[CAMERA] Open failed.\n";
        return false;
    }
    std::cout << "[CAMERA] Opened OK.\n";
    return true;
}

LaneDetector::LaneDetector(const std::string& videoPath, int width, int height)
    : width(width), height(height)
{
    const int CAP_W = 640, CAP_H = 480;
    const int OUT_W = width, OUT_H = height;
    const int FPS   = 30;

    if (videoPath.find("/dev/") != std::string::npos) {
	// Chỉ dùng libcamera + videoscale
    const std::string p0 =
        "libcamerasrc ! "
        "video/x-raw, width=" + std::to_string(CAP_W) +
        ", height=" + std::to_string(CAP_H) +
        ", framerate=" + std::to_string(FPS) + "/1 ! "
        "videoconvert ! videoscale ! "
        "video/x-raw, format=(string)BGR, width=" + std::to_string(OUT_W) +
        ", height=" + std::to_string(OUT_H) + " ! "
        "appsink max-buffers=1 drop=true sync=false";

    const std::string p1 =
        "libcamerasrc ! "
        "videoconvert ! videoscale ! "
        "video/x-raw, format=(string)BGR, width=" + std::to_string(OUT_W) +
        ", height=" + std::to_string(OUT_H) + " ! "
        "appsink max-buffers=1 drop=true sync=false";

        if (!try_open_gst(cap, p0) && !try_open_gst(cap, p1)) {
            throw std::runtime_error("Cannot open camera with libcamerasrc: " + videoPath);
        }
    } else {
        if (!cap.open(videoPath)) {
            throw std::runtime_error("Cannot open file: " + videoPath);
        }
    }
}

LaneDetector::~LaneDetector() {
    cap.release();
    cv::destroyAllWindows();
}

bool LaneDetector::getFrame(cv::Mat& frame) {
    cap >> frame;

    static int saved_raw = 0;
    if (saved_raw == 0) {
    cv::imwrite("pi_raw.jpg", frame);
    std::cout << "Saved pi_raw.jpg" << std::endl;
    saved_raw = 1;
    }
   cv::Mat undistorted;
    cv::Mat cameraMatrix = (cv::Mat_<double>(3,3) <<
        262.08953333143063, 0.0, 330.77574325128484,
        0.0, 263.57901348164575, 250.50298224489268,
        0.0, 0.0, 1.0);

    cv::Mat distCoeffs = (cv::Mat_<double>(1,5) <<
        -0.27166331922859776, 0.09924985737514846,
        -0.0002707688044880526, 0.0006724194580262318,
        -0.01935517123682299);

    cv::Rect validROI;
    cv::Mat newCameraMatrix = cv::getOptimalNewCameraMatrix(
        cameraMatrix,
        distCoeffs,
        frame.size(),
        0.0,              // 0 = crop mạnh để giảm viền đen, 1 = giữ full FoV
        frame.size(),
        &validROI
    );

    cv::undistort(frame, undistorted, cameraMatrix, distCoeffs, newCameraMatrix);

    // crop vùng hợp lệ
    if (validROI.width > 0 && validROI.height > 0) {
        undistorted = undistorted(validROI).clone();
    }

    frame = undistorted.clone();

    cv::resize(frame, frame_resize, cv::Size(width, height));
    return !frame.empty();
    
}

bool LaneDetector::isOpened() const {
    return cap.isOpened();
}

cv::Mat LaneDetector::getFrameResize() {
    return frame_resize;
}
cv::Mat LaneDetector::getMask() const{
    return mask;
}
// copyright by quan
#if 0
void LaneDetector::processFrame(cv::Mat& frame_resize) {
    bird_eye_view = applyIPM(frame_resize);
    mask = processMask(bird_eye_view);

    static int minpix = 30;

    std::vector<cv::Point> left_points, right_points;
    cv::Vec3f left_coeffs(0, 0, 0), right_coeffs(0, 0, 0);
    static cv::Vec3f prev_left(0, 0, 0), prev_right(0, 0, 0);

    bool left_ok = false, right_ok = false;

    // Không giữ trạng thái merge cũ giữa các frame nếu frame hiện tại không đủ dữ liệu
    bool merging_left_flag = false;
    bool merging_right_flag = false;

    auto evalX = [](cv::Vec3f c, float y) {
        return c[0] * y * y + c[1] * y + c[2];
    };

    // =====================================================
    // 1) DETECT LANE
    // =====================================================
    if (!initialized) {
        slidingWindow(mask, left_points, right_points, bird_eye_view, minpix);

        left_ok  = (left_points.size()  >= 30);
        right_ok = (right_points.size() >= 30);

        std::cout << "[INIT] left_points=" << left_points.size()
                  << " right_points=" << right_points.size() << std::endl;

        if (left_ok) {
            left_coeffs = fitPoly(left_points, bird_eye_view, true);
            left_type = classifyLaneMarking(mask, left_coeffs);
            switch (left_type) {
                case LaneLineType::SOLID:  std::cout << "LEFTLINE: SOLID\n"; break;
                case LaneLineType::DASHED: std::cout << "LEFTLINE: DASHED\n"; break;
                default:                   std::cout << "LEFTLINE: UNKNOWN\n"; break;
            }
        }

        if (right_ok) {
            right_coeffs = fitPoly(right_points, bird_eye_view, false);
            right_type = classifyLaneMarking(mask, right_coeffs);
            switch (right_type) {
                case LaneLineType::SOLID:  std::cout << "RIGHTLINE: SOLID\n"; break;
                case LaneLineType::DASHED: std::cout << "RIGHTLINE: DASHED\n"; break;
                default:                   std::cout << "RIGHTLINE: UNKNOWN\n"; break;
            }
        }

        if (left_ok && right_ok) {
            float x_left = evalX(left_coeffs, height - 1.0f);
            float x_right = evalX(right_coeffs, height - 1.0f);
            float lane_width = std::fabs(x_right - x_left);

            if (lane_width > 80.0f && lane_width < 500.0f) {
                initialized = true;
                prev_left = left_coeffs;
                prev_right = right_coeffs;
            }
        }
    }
    else {
        // Detect theo lane cũ
        slidingWindowAdaptive(mask, left_points, bird_eye_view, prev_left);
        slidingWindowAdaptive(mask, right_points, bird_eye_view, prev_right);

        left_ok  = (left_points.size()  >= 30);
        right_ok = (right_points.size() >= 30);

        std::cout << "[TRACK] left_points=" << left_points.size()
                  << " right_points=" << right_points.size() << std::endl;

        if (left_ok) {
            left_coeffs = fitPoly(left_points, bird_eye_view, true);
            left_type = classifyLaneMarking(mask, left_coeffs);
            switch (left_type) {
                case LaneLineType::SOLID:  std::cout << "LEFTLINE: SOLID\n"; break;
                case LaneLineType::DASHED: std::cout << "LEFTLINE: DASHED\n"; break;
                default:                   std::cout << "LEFTLINE: UNKNOWN\n"; break;
            }
        } else {
            left_coeffs = prev_left; // fallback chỉ để giữ quán tính hình học
        }

        if (right_ok) {
            right_coeffs = fitPoly(right_points, bird_eye_view, false);
            right_type = classifyLaneMarking(mask, right_coeffs);
            switch (right_type) {
                case LaneLineType::SOLID:  std::cout << "RIGHTLINE: SOLID\n"; break;
                case LaneLineType::DASHED: std::cout << "RIGHTLINE: DASHED\n"; break;
                default:                   std::cout << "RIGHTLINE: UNKNOWN\n"; break;
            }
        } else {
            right_coeffs = prev_right;
        }

        // =====================================================
        // 2) CHECK MERGING CHỈ KHI CẢ 2 LANE ĐỀU HỢP LỆ
        // =====================================================
        if (left_ok && right_ok) {
            float x_left = evalX(left_coeffs, height - 1.0f);
            float x_right = evalX(right_coeffs, height - 1.0f);
            float lane_width = std::fabs(x_right - x_left);

            if (lane_width < 100.0f) {
                if (x_left < bird_eye_view.cols / 2.0f) {
                    merging_right_flag = true;
                } else {
                    merging_left_flag = true;
                }
            }
        }

        if (merging_left_flag) {
            left_ok = false;
            left_coeffs = prev_left;
        }

        if (merging_right_flag) {
            right_ok = false;
            right_coeffs = prev_right;
        }

        // =====================================================
        // 3) ERROR CHECK - CHỈ CHECK LANE HỢP LỆ
        // =====================================================
        int mid_y = bird_eye_view.rows / 2;
        bool error_lane = false;

        if (left_ok) {
            float slope_left = computeLaneSlope(left_coeffs, static_cast<float>(mid_y));
            if (std::fabs(slope_left) > 0.35f) {
                error_lane = true;
                std::cout << "[ERR] slope_left=" << slope_left << std::endl;
            }
        }

        if (right_ok) {
            float slope_right = computeLaneSlope(right_coeffs, static_cast<float>(mid_y));
            if (std::fabs(slope_right) > 0.35f) {
                error_lane = true;
                std::cout << "[ERR] slope_right=" << slope_right << std::endl;
            }
        }

        std::cout << "Error lane flag: " << error_lane << std::endl;

        if (error_lane) {
            initialized = false;
        } else {
            // chỉ cập nhật prev khi lane hiện tại còn đáng tin
            if (left_ok)  prev_left = left_coeffs;
            if (right_ok) prev_right = right_coeffs;
        }
    }

    // =====================================================
    // 4) COMPUTE CENTERLINE LUÔN LUÔN
    // =====================================================
    centerline = computeCenterline(left_coeffs, right_coeffs, left_ok, right_ok, bird_eye_view);
    has_valid_lane_ = (centerline.size() >= 3);

    // =====================================================
    // 5) LƯU CHO PLANNER
    // =====================================================
    left_coeffs_ = left_coeffs;
    right_coeffs_ = right_coeffs;
    has_left_lane_ = left_ok;
    has_right_lane_ = right_ok;

    if (left_ok && right_ok) {
        float y_ref = static_cast<float>(height - 1);
        float x_left = evalX(left_coeffs, y_ref);
        float x_right = evalX(right_coeffs, y_ref);
        float lane_width = std::fabs(x_right - x_left);

        if (lane_width > 80.0f && lane_width < 500.0f) {
            lane_width_px_ = lane_width;
        }
    }

    std::cout << "[FINAL] left_ok=" << left_ok
              << " right_ok=" << right_ok
              << " centerline_pts=" << centerline.size()
              << std::endl;
}
#endif
#if 1
void LaneDetector::processFrame(cv::Mat& frame_resize) {
    bird_eye_view = applyIPM(frame_resize);
    mask = processMask(bird_eye_view);

    static int minpix = 30;

    std::vector<cv::Point> left_points, right_points;
    cv::Vec3f left_coeffs(0.0f, 0.0f, 0.0f), right_coeffs(0.0f, 0.0f, 0.0f);

    bool left_ok = false, right_ok = false;

    bool merging_left_flag = false;
    bool merging_right_flag = false;

    auto evalX = [](const cv::Vec3f& c, float y) {
        return c[0] * y * y + c[1] * y + c[2];
    };

    // =====================================================
    // 1) DETECT LANE
    // =====================================================
    // Chỉ tracking khi đã initialized và thực sự có prev hợp lệ cho cả 2 lane
    bool can_track = initialized && has_prev_left_ && has_prev_right_;

    if (!can_track) {
        slidingWindow(mask, left_points, right_points, bird_eye_view, minpix);

        left_ok  = (left_points.size()  >= 30);
        right_ok = (right_points.size() >= 30);

        std::cout << "[INIT] left_points=" << left_points.size()
                  << " right_points=" << right_points.size() << std::endl;

        if (left_ok) {
            left_coeffs = fitPoly(left_points, bird_eye_view, true);
            left_type = classifyLaneMarking(mask, left_coeffs);
            switch (left_type) {
                case LaneLineType::SOLID:  std::cout << "LEFTLINE: SOLID\n"; break;
                case LaneLineType::DASHED: std::cout << "LEFTLINE: DASHED\n"; break;
                default:                   std::cout << "LEFTLINE: UNKNOWN\n"; break;
            }

            // Lưu prev lane trái nếu lane hiện tại detect tốt
            prev_left_ = left_coeffs;
            has_prev_left_ = true;
        }

        if (right_ok) {
            right_coeffs = fitPoly(right_points, bird_eye_view, false);
            right_type = classifyLaneMarking(mask, right_coeffs);
            switch (right_type) {
                case LaneLineType::SOLID:  std::cout << "RIGHTLINE: SOLID\n"; break;
                case LaneLineType::DASHED: std::cout << "RIGHTLINE: DASHED\n"; break;
                default:                   std::cout << "RIGHTLINE: UNKNOWN\n"; break;
            }

            // Lưu prev lane phải nếu lane hiện tại detect tốt
            prev_right_ = right_coeffs;
            has_prev_right_ = true;
        }

        // Chỉ bật initialized khi cả 2 lane hợp lệ và lane width hợp lý
        if (left_ok && right_ok) {
            float x_left  = evalX(left_coeffs,  height - 1.0f);
            float x_right = evalX(right_coeffs, height - 1.0f);
            float lane_width = std::fabs(x_right - x_left);

            if (lane_width > 80.0f && lane_width < 500.0f) {
                initialized = true;
            } else {
                initialized = false;
            }
        } else {
            initialized = false;
        }
    }
    else {
        // Detect theo lane cũ
        slidingWindowAdaptive(mask, left_points,  bird_eye_view, prev_left_);
        slidingWindowAdaptive(mask, right_points, bird_eye_view, prev_right_);

        left_ok  = (left_points.size()  >= 30);
        right_ok = (right_points.size() >= 30);

        // std::cout << "[TRACK] left_points=" << left_points.size()
        //           << " right_points=" << right_points.size() << std::endl;

        if (left_ok) {
            left_coeffs = fitPoly(left_points, bird_eye_view, true);
            left_type = classifyLaneMarking(mask, left_coeffs);
            switch (left_type) {
                case LaneLineType::SOLID:  std::cout << "LEFTLINE: SOLID\n"; break;
                case LaneLineType::DASHED: std::cout << "LEFTLINE: DASHED\n"; break;
                default:                   std::cout << "LEFTLINE: UNKNOWN\n"; break;
            }
        } else if (has_prev_left_) {
            left_coeffs = prev_left_;   // fallback chỉ khi thực sự có prev hợp lệ
        }

        if (right_ok) {
            right_coeffs = fitPoly(right_points, bird_eye_view, false);
            right_type = classifyLaneMarking(mask, right_coeffs);
            switch (right_type) {
                case LaneLineType::SOLID:  std::cout << "RIGHTLINE: SOLID\n"; break;
                case LaneLineType::DASHED: std::cout << "RIGHTLINE: DASHED\n"; break;
                default:                   std::cout << "RIGHTLINE: UNKNOWN\n"; break;
            }
        } else if (has_prev_right_) {
            right_coeffs = prev_right_;
        }

        // =====================================================
        // 2) CHECK MERGING CHỈ KHI CẢ 2 LANE ĐỀU HỢP LỆ
        // =====================================================
        if (left_ok && right_ok) {
            float x_left  = evalX(left_coeffs,  height - 1.0f);
            float x_right = evalX(right_coeffs, height - 1.0f);
            float lane_width = std::fabs(x_right - x_left);

            if (lane_width < 100.0f) {
                if (x_left < bird_eye_view.cols / 2.0f) {
                    merging_right_flag = true;
                } else {
                    merging_left_flag = true;
                }
            }
        }

        if (merging_left_flag) {
            left_ok = false;
            if (has_prev_left_) {
                left_coeffs = prev_left_;
            }
        }

        if (merging_right_flag) {
            right_ok = false;
            if (has_prev_right_) {
                right_coeffs = prev_right_;
            }
        }

        // =====================================================
        // 3) ERROR CHECK - CHỈ CHECK LANE HỢP LỆ
        // =====================================================
        int mid_y = bird_eye_view.rows / 2;
        bool error_lane = false;

        if (left_ok) {
            float slope_left = computeLaneSlope(left_coeffs, static_cast<float>(mid_y));
            if (std::fabs(slope_left) > 0.35f) {
                error_lane = true;
                std::cout << "[ERR] slope_left=" << slope_left << std::endl;
            }
        }

        if (right_ok) {
            float slope_right = computeLaneSlope(right_coeffs, static_cast<float>(mid_y));
            if (std::fabs(slope_right) > 0.35f) {
                error_lane = true;
                std::cout << "[ERR] slope_right=" << slope_right << std::endl;
            }
        }

        // std::cout << "Error lane flag: " << error_lane << std::endl;

        if (error_lane) {
            // Reset hoàn toàn state tracking cũ
            initialized = false;

            has_prev_left_ = false;
            has_prev_right_ = false;
            prev_left_ = cv::Vec3f(0.0f, 0.0f, 0.0f);
            prev_right_ = cv::Vec3f(0.0f, 0.0f, 0.0f);
        } else {
            // Chỉ cập nhật prev khi lane hiện tại còn đáng tin
            if (left_ok) {
                prev_left_ = left_coeffs;
                has_prev_left_ = true;
            }
            if (right_ok) {
                prev_right_ = right_coeffs;
                has_prev_right_ = true;
            }

            // Nếu mất 1 lane thì vẫn giữ lane còn lại, nhưng không tracking cả 2 lane nữa
            if (!(has_prev_left_ && has_prev_right_)) {
                initialized = false;
            }
        }
    }

    // =====================================================
    // 4) COMPUTE CENTERLINE LUÔN LUÔN
    // =====================================================
    centerline = computeCenterline(left_coeffs, right_coeffs, left_ok, right_ok, bird_eye_view);
    has_valid_lane_ = (centerline.size() >= 3);

    // =====================================================
    // 5) LƯU CHO PLANNER
    // =====================================================
    left_coeffs_ = left_coeffs;
    right_coeffs_ = right_coeffs;
    has_left_lane_ = left_ok;
    has_right_lane_ = right_ok;

    if (left_ok && right_ok) {
        float y_ref = static_cast<float>(height - 1);
        float x_left  = evalX(left_coeffs,  y_ref);
        float x_right = evalX(right_coeffs, y_ref);
        float lane_width = std::fabs(x_right - x_left);

        if (lane_width > 80.0f && lane_width < 500.0f) {
            lane_width_px_ = lane_width;
        }
    }

    std::cout << "[FINAL] left_ok=" << left_ok
              << " right_ok=" << right_ok
              << " centerline_pts=" << centerline.size()
              << std::endl;
}
#endif
cv::Mat LaneDetector::applyIPM(cv::Mat& frame)
{
    if (frame.empty()) {
        bird_eye_view = cv::Mat::zeros(height, width, CV_8UC3);
        return bird_eye_view;
    }

    // =====================================================
    // SOURCE POINTS
    // =====================================================
    cv::Point2f tl(235.0f, 285.0f);
    cv::Point2f tr(405.0f, 285.0f);
    cv::Point2f br(560.0f, 470.0f);
    cv::Point2f bl( 95.0f, 470.0f);

    std::vector<cv::Point2f> src_points = { tl, tr, br, bl };

    // =====================================================
    // DESTINATION POINTS
    // =====================================================
    const float margin_x = 150.0f;

    cv::Point2f dst_tl(margin_x, 0.0f);
    cv::Point2f dst_tr(width - margin_x, 0.0f);
    cv::Point2f dst_br(width - margin_x, height - 1.0f);
    cv::Point2f dst_bl(margin_x, height - 1.0f);

    std::vector<cv::Point2f> dst_points = { dst_tl, dst_tr, dst_br, dst_bl };

    // Debug vẽ vùng nguồn
    cv::circle(frame, tl, 5, cv::Scalar(0, 255, 0), -1);
    cv::circle(frame, tr, 5, cv::Scalar(0, 255, 0), -1);
    cv::circle(frame, br, 5, cv::Scalar(0, 0, 255), -1);
    cv::circle(frame, bl, 5, cv::Scalar(0, 0, 255), -1);

    cv::line(frame, tl, tr, cv::Scalar(255, 0, 0), 2);
    cv::line(frame, tr, br, cv::Scalar(255, 0, 0), 2);
    cv::line(frame, br, bl, cv::Scalar(255, 0, 0), 2);
    cv::line(frame, bl, tl, cv::Scalar(255, 0, 0), 2);

    cv::Mat M = cv::getPerspectiveTransform(src_points, dst_points);

    cv::Mat warped;
    cv::warpPerspective(
        frame,
        warped,
        M,
        cv::Size(width, height),
        cv::INTER_LINEAR,
        cv::BORDER_CONSTANT,
        cv::Scalar(0, 0, 0)
    );

    // =====================================================
    // AUTO-CROP vùng không đen
    // =====================================================
    cv::Mat gray, validMask;
    cv::cvtColor(warped, gray, cv::COLOR_BGR2GRAY);

    // pixel > 5 coi là hợp lệ
    cv::threshold(gray, validMask, 5, 255, cv::THRESH_BINARY);

    // đóng/mở nhẹ để mask liền hơn
    cv::morphologyEx(
        validMask,
        validMask,
        cv::MORPH_CLOSE,
        cv::getStructuringElement(cv::MORPH_RECT, cv::Size(9, 9))
    );

    std::vector<cv::Point> nz;
    cv::findNonZero(validMask, nz);

    if (!nz.empty()) {
        cv::Rect roi = cv::boundingRect(nz);

        // chỉ crop ngang mạnh, giữ gần như full chiều cao
        int pad_x = 5;
        roi.x = std::max(0, roi.x + pad_x);
        roi.width = std::min(warped.cols - roi.x, roi.width - 2 * pad_x);

        if (roi.width > 50 && roi.height > 50) {
            cv::Mat cropped = warped(roi).clone();
            cv::resize(cropped, bird_eye_view, cv::Size(width, height));
            return bird_eye_view;
        }
    }

    bird_eye_view = warped.clone();
    return bird_eye_view;
}

void LaneDetector::slidingWindow(const cv::Mat& mask,
                       std::vector<cv::Point>& left_points,
                       std::vector<cv::Point>& right_points,
                       cv::Mat& outImg, int minpix) {
    int nwindows = 15;
    int margin   = 60;
    int height   = mask.rows;
    int width    = mask.cols;
    int window_height = height / nwindows;

    cv::Mat hist;
    cv::reduce(mask(cv::Rect(0, height/2, width, height/2)),
            hist, 0, cv::REDUCE_SUM, CV_32S);
    int midpoint = hist.cols / 2;
    int leftx_base  = std::max_element(hist.begin<int>(), hist.begin<int>() + midpoint) - hist.begin<int>();
    int rightx_base = std::max_element(hist.begin<int>() + midpoint, hist.end<int>()) - hist.begin<int>();

    int leftx_current  = leftx_base;
    int rightx_current = rightx_base;

    std::vector<cv::Point> nonzero;
    cv::findNonZero(mask, nonzero);

    for (int window = 0; window < nwindows; window++) {
        int win_y_low  = height - (window+1) * window_height;

        cv::Rect left_win(leftx_current - margin, win_y_low, margin*2, window_height);
        cv::Rect right_win(rightx_current - margin, win_y_low, margin*2, window_height);

        std::vector<cv::Point> good_left, good_right;
        for (auto &p : nonzero) {
            if (left_win.contains(p))  good_left.push_back(p);
            if (right_win.contains(p)) good_right.push_back(p);
        }

        if (!good_left.empty()) {
            cv::rectangle(outImg, left_win, cv::Scalar(0,255,0), 2);
        }
        if (!good_right.empty()) {
            cv::rectangle(outImg, right_win, cv::Scalar(0,255,0), 2);
        }
        left_points.insert(left_points.end(), good_left.begin(), good_left.end());
        right_points.insert(right_points.end(), good_right.begin(), good_right.end());

        if ((int)good_left.size() > minpix) {
            int sumx = 0;
            for (auto &p : good_left) sumx += p.x;
            leftx_current = sumx / (int)good_left.size();
        }
        if ((int)good_right.size() > minpix) {
            int sumx = 0;
            for (auto &p : good_right) sumx += p.x;
            rightx_current = sumx / (int)good_right.size();
        }
    }
}

void LaneDetector::slidingWindowAdaptive(const cv::Mat& mask,
                               std::vector<cv::Point>& lane_points,
                               cv::Mat& outImg,
                               cv::Vec3f prev_poly) {
    int nwindows = 15;
    int margin   = 60;
    int height   = mask.rows;
    int window_height = height / nwindows;

    std::vector<cv::Point> nonzero;
    cv::findNonZero(mask, nonzero);

    int x_center = mask.cols / 2;

    for (int window = 0; window < nwindows; window++) {
        int win_y_low  = height - (window+1) * window_height;
        int win_y_high = height - window * window_height;
        int y_mid      = (win_y_low + win_y_high) / 2;

        x_center = (int)(prev_poly[0]*y_mid*y_mid +
                            prev_poly[1]*y_mid +
                            prev_poly[2]);

        cv::Rect win(x_center - margin, win_y_low, margin*2, window_height);
        win &= cv::Rect(0, 0, mask.cols, mask.rows);

        cv::rectangle(outImg, win, cv::Scalar(0,255,0), 2);

        for (auto &p : nonzero) {
            if (win.contains(p)) {
                lane_points.push_back(p);
            }
        }
    }
}

cv::Mat LaneDetector::processMask(const cv::Mat& bird_eye_view) {
    cv::Mat hsv, mask;
    cv::cvtColor(bird_eye_view, hsv, cv::COLOR_BGR2HSV);
    cv::inRange(hsv, cv::Scalar(0, 0, 200), cv::Scalar(180, 40, 255), mask);
    return mask;
}

cv::Vec3f LaneDetector::fitPoly(const std::vector<cv::Point>& points, cv::Mat& outImg, bool isLeft) {
    if (points.size() < 2) return cv::Vec3f(0,0,0);
    std::vector<float> x_vals, y_vals;
    for (const auto& pt : points) {
        x_vals.push_back((float)pt.x);
        y_vals.push_back((float)pt.y);
    }
    cv::Mat Y(y_vals.size(), 1, CV_32F, y_vals.data());
    cv::Mat X(x_vals.size(), 1, CV_32F, x_vals.data());
    cv::Vec3f coeff_out(0,0,0);
    if (points.size() >= 3) {
        cv::Mat A2(Y.rows, 3, CV_32F);
        for (int i = 0; i < Y.rows; ++i) {
            float y = Y.at<float>(i, 0);
            A2.at<float>(i, 0) = y * y;
            A2.at<float>(i, 1) = y;
            A2.at<float>(i, 2) = 1.0f;
        }
        cv::Mat coeffs2;
        bool ok = cv::solve(A2, X, coeffs2, cv::DECOMP_SVD);
        if (ok) coeff_out = cv::Vec3f(coeffs2.at<float>(0), coeffs2.at<float>(1), coeffs2.at<float>(2));
    }
    if (coeff_out == cv::Vec3f(0,0,0) || std::fabs(coeff_out[0]) < 1e-6) {
        cv::Mat A1(Y.rows, 2, CV_32F);
        for (int i = 0; i < Y.rows; ++i) {
            float y = Y.at<float>(i, 0);
            A1.at<float>(i, 0) = y;
            A1.at<float>(i, 1) = 1.0f;
        }
        cv::Mat coeffs1;
        bool ok = cv::solve(A1, X, coeffs1, cv::DECOMP_SVD);
        if (ok) coeff_out = cv::Vec3f(0, coeffs1.at<float>(0), coeffs1.at<float>(1));
    }
    cv::Scalar lineColor = isLeft ? cv::Scalar(0,255,0) : cv::Scalar(0,0,255);
    if (coeff_out != cv::Vec3f(0,0,0)) {
        for (int y = 0; y < outImg.rows; ++y) {
            float x = coeff_out[0]*y*y + coeff_out[1]*y + coeff_out[2];
            int ix = (int)std::round(x);
            if (ix >= 0 && ix < outImg.cols) {
                cv::circle(outImg, cv::Point(ix, y), 2, lineColor, -1);
            }
        }
    }
    return coeff_out;
}

std::vector<cv::Point> LaneDetector::computeCenterline(cv::Vec3f coeff_left,
                                            cv::Vec3f coeff_right,
                                            bool has_left, bool has_right,
                                            cv::Mat& outImg) {
    std::vector<cv::Point> centerline;
    if (outImg.empty()) return centerline;

    const float LANE_WIDTH_PX = 800.0f;
    static float laneW_avg = LANE_WIDTH_PX;  

    auto evalX = [](cv::Vec3f c, float y) {
        return c[0] * y * y + c[1] * y + c[2];
    };

    if (has_left && has_right) {
        std::vector<float> widths;
        float d;
        for (int y = 0; y < outImg.rows; y += 20) {
            d = fabs(evalX(coeff_right, y) - evalX(coeff_left, y));  
            if (d > 50 && d < 600)  
                widths.push_back(d);
        }
        if (!widths.empty()) {
            float sum =0;
            for(auto b: widths) sum += b;
            laneW_avg = sum/widths.size();
        }
    }

    float current_laneW;
    if(laneW_avg < 395.0f || laneW_avg > 405.0f) 
    {
        current_laneW = LANE_WIDTH_PX;
    }
    else
    {
        current_laneW = laneW_avg;
    }

    for (int y = 0; y < outImg.rows; y += 10) {
        float xc = -1;
        if (has_left && has_right) {
            xc = 0.5f * (evalX(coeff_left, y) + evalX(coeff_right, y));  
        }

        else if (has_left) {
            xc = evalX(coeff_left, y) + 0.5f * current_laneW;  
        }

        else if (has_right) {
            xc = evalX(coeff_right, y) - 0.5f * current_laneW;
        } else {
            continue;  
        }
        int x = std::clamp((int)std::round(xc), 0, outImg.cols - 1);
	if (x > 0)
	{
            centerline.push_back({x, y});  
            cv::circle(outImg, {x, y}, 2, {255, 255, 0}, -1);
	}
    }
    // std::cout << "Lane Width Average: " << current_laneW << "px" << std::endl;

    // ==== Debug hiển thị ==== 
    std::string dbg_text = "W=" + std::to_string((int)laneW_avg) + "px";
    cv::putText(outImg, dbg_text, {30, 40}, cv::FONT_HERSHEY_SIMPLEX, 1.0, {0, 255, 255}, 2);

    // Hiển thị trạng thái lane
    if (has_left && !has_right)
        cv::putText(outImg, "LEFT ONLY", {30, 80}, cv::FONT_HERSHEY_SIMPLEX, 1.0, {0, 255, 0}, 2);
    else if (!has_left && has_right)
        cv::putText(outImg, "RIGHT ONLY", {30, 80}, cv::FONT_HERSHEY_SIMPLEX, 1.0, {0, 0, 255}, 2);
    else if (has_left && has_right)
        cv::putText(outImg, "BOTH LANES", {30, 80}, cv::FONT_HERSHEY_SIMPLEX, 1.0, {255, 255, 255}, 2);
    else if (!has_left && !has_right)
        cv::putText(outImg, "NO LANE", {30, 80}, cv::FONT_HERSHEY_SIMPLEX, 1.0, {0, 0, 255}, 2);

    return centerline; 
}

float LaneDetector::computeLaneSlope(const cv::Vec3f& coeffs, float y) {
    float a = coeffs[0];
    float b = coeffs[1];

    if (std::fabs(a) > 1e-6) {
        return 2*a*y + b;
    } else {
        return b;
    }
}


LaneLineType LaneDetector::classifyLaneMarking(const cv::Mat& mask,
                                               const cv::Vec3f& coeff,
                                               int band_half_width,
                                               int y_step,
                                               int min_segment_len,
                                               int min_gap_len)
{
    if (mask.empty() || mask.type() != CV_8UC1) {
        return LaneLineType::UNKNOWN;
    }

    auto evalLaneX = [](const cv::Vec3f& c, float y) {
        return c[0] * y * y + c[1] * y + c[2];
    };

    std::vector<int> occupancy;
    occupancy.reserve(mask.rows / y_step + 1);

    // Chỉ dùng vùng giữa -> dưới ảnh, bỏ phần quá xa vì rất nhiễu
    int y_start = mask.rows / 5;
    int y_end   = mask.rows - 1;

    for (int y = y_start; y < y_end; y += y_step) {
        int x_center = static_cast<int>(std::round(evalLaneX(coeff, static_cast<float>(y))));
        int x1 = std::max(0, x_center - band_half_width);
        int x2 = std::min(mask.cols - 1, x_center + band_half_width);

        if (x1 >= x2) {
            occupancy.push_back(0);
            continue;
        }

        cv::Rect roi(x1, y, x2 - x1 + 1, std::min(y_step, mask.rows - y));
        cv::Mat band = mask(roi);

        int white_pixels = cv::countNonZero(band);
        int total_pixels = roi.width * roi.height;

        // Hạ threshold một chút để lane liền đỡ bị đứt giả
        float white_ratio = static_cast<float>(white_pixels) / static_cast<float>(total_pixels);
        occupancy.push_back((white_ratio > 0.18f) ? 1 : 0);
    }

    if (occupancy.empty()) {
        return LaneLineType::UNKNOWN;
    }

    // =====================================================
    // Lấp các lỗ nhỏ: 1 0 1 -> 1 1 1
    //                 1 0 0 1 -> 1 1 1 1
    // =====================================================
    for (size_t i = 1; i + 1 < occupancy.size(); ++i) {
        if (occupancy[i - 1] == 1 && occupancy[i] == 0 && occupancy[i + 1] == 1) {
            occupancy[i] = 1;
        }
    }

    for (size_t i = 1; i + 2 < occupancy.size(); ++i) {
        if (occupancy[i - 1] == 1 &&
            occupancy[i] == 0 &&
            occupancy[i + 1] == 0 &&
            occupancy[i + 2] == 1) {
            occupancy[i] = 1;
            occupancy[i + 1] = 1;
        }
    }

    // =====================================================
    // Tách segment trắng và gap đen
    // =====================================================
    std::vector<int> white_segments;
    std::vector<int> black_gaps;

    int current_len = 1;
    for (size_t i = 1; i < occupancy.size(); ++i) {
        if (occupancy[i] == occupancy[i - 1]) {
            current_len++;
        } else {
            if (occupancy[i - 1] == 1) {
                white_segments.push_back(current_len);
            } else {
                black_gaps.push_back(current_len);
            }
            current_len = 1;
        }
    }

    if (occupancy.back() == 1) {
        white_segments.push_back(current_len);
    } else {
        black_gaps.push_back(current_len);
    }

    int valid_white_segments = 0;
    int valid_black_gaps = 0;
    int max_white_len = 0;
    int max_black_len = 0;

    for (int len : white_segments) {
        if (len >= min_segment_len) valid_white_segments++;
        if (len > max_white_len) max_white_len = len;
    }

    for (int len : black_gaps) {
        if (len >= min_gap_len) valid_black_gaps++;
        if (len > max_black_len) max_black_len = len;
    }

    int white_count = std::count(occupancy.begin(), occupancy.end(), 1);
    float white_ratio = static_cast<float>(white_count) / static_cast<float>(occupancy.size());

    // =====================================================
    // Heuristic mới
    // =====================================================

    // Lane liền: phủ trắng cao, có 1 dải trắng dài, không có gap dài thật
    if (white_ratio > 0.60f && max_white_len >= 6 && max_black_len <= 2) {
        return LaneLineType::SOLID;
    }

    // Lane đứt: nhiều đoạn trắng + có gap đủ dài
    if (valid_white_segments >= 2 && valid_black_gaps >= 1 && max_black_len >= min_gap_len) {
        return LaneLineType::DASHED;
    }

    // fallback
    if (white_ratio > 0.70f) {
        return LaneLineType::SOLID;
    }

    if (white_ratio > 0.20f && valid_black_gaps >= 1) {
        return LaneLineType::DASHED;
    }

    return LaneLineType::UNKNOWN;
}