#ifndef LOGGER_HPP
#define LOGGER_HPP

#include <fstream>
#include <mutex>
#include <string>
#include <chrono>
#include <iomanip>
#include <opencv2/opencv.hpp>

class Logger {
public:
    explicit Logger(const std::string& filename);
    ~Logger();

    void log(const std::string& tag, double duration_ms);

    static void drawPolyline(cv::Mat& img,
                             const std::vector<cv::Point>& line,
                             const cv::Scalar& color);

private:
    std::ofstream file_;
    std::mutex mutex_;
};

#endif