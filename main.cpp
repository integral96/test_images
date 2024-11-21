#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>
#include <fstream>
#include <iterator>
#include <future>
#include <thread>
#include <memory>
#include <numeric>
#include <algorithm>

template<typename T>
class slice_mtrx : public std::shared_ptr<T[]> {
    size_t N;
    size_t M;
    size_t K;
public:
    slice_mtrx(size_t N_, size_t M_, size_t K_) : std::shared_ptr<T[]>(new T[N_*M_*K_]), N(N_), M(M_), K(K_) {}
    const T & operator()(size_t i, size_t j, size_t k) const {
        assert(i < N);
        assert(j < M);
        assert(k < K);
        return (*this)[k + K*(j + M*i)];
    }
    T & operator()(size_t i, size_t j, size_t k) {
        assert(i < N);
        assert(j < M);
        assert(k < K);
        return (*this)[k + K*(j + M*i)];
    }
    size_t size(size_t i) const {
        assert(i < 3);
        if(i == 0) return N;
        else if(i == 1) return M;
        else return K;
    }
};
static size_t color_N = 17;
static size_t conform_color = color_N*color_N;

class fill_slicer
{
    const std::array<cv::Mat, 5>& sources;
public:
    fill_slicer(const std::array<cv::Mat, 5>& sources_) : sources(sources_) {}
    template<size_t index, std::enable_if_t<(index < 5), bool> = true>
    void fill(slice_mtrx<int>& frame_slices) const {
        size_t var = 0;
        int val_color = 0;
        size_t rows = sources[index].rows;
        size_t cols = sources[index].cols;
        assert(rows == frame_slices.size(1) && cols == frame_slices.size(2));
        while(var < color_N) {
            for (size_t j = 0; j < rows; ++j) {
                for (size_t k = 0; k < cols; ++k) {
                    int frame_10001_val = sources[index].col(k).template at<int>(j) & 0xFF;
                    if(abs(frame_10001_val - val_color) < 15) {
                        frame_slices(var, j, k) = 255;
                    }
                    else frame_slices(var, j, k) = 0;
                }
            }
            ++var;
            val_color += 15;
        }
    }
};

class solver_slice
{
    const slice_mtrx<int>& frame_1;
    const slice_mtrx<int>& frame_2;
public:
    solver_slice(const slice_mtrx<int>& frame_1, const slice_mtrx<int>& frame_2) : frame_1(frame_1), frame_2(frame_2) {}
    size_t solv(size_t i_i) {
        size_t i_j = 0;
        size_t count_slices = 0;
        while(i_j < frame_1.size(0)) {
            double count = 0;
            for (size_t j = 0; j < frame_1.size(1); ++j) {
                for (size_t k = 0; k < frame_1.size(2); ++k) {
                    if(frame_1(i_i, j, k) - frame_2(i_j, j, k) == 0) {
                        ++count;
                    }
                }
            }
            double value_error = count/(frame_1.size(1)*frame_1.size(2));
            if(value_error >= 0.99966) {
                ++count_slices;
            }
            ++i_j;
        }
        return count_slices;
    }
};


int main()
{
    std::string frame_10001_pth = cv::samples::findFile("frame_10001.png");
    cv::Mat frame_10001 = imread(frame_10001_pth, cv::IMREAD_COLOR);

    std::string frame_10002_pth = cv::samples::findFile("frame_10002.png");
    cv::Mat frame_10002 = imread(frame_10002_pth, cv::IMREAD_COLOR);

    std::string frame_10003_pth = cv::samples::findFile("frame_10003.png");
    cv::Mat frame_10003 = imread(frame_10003_pth, cv::IMREAD_COLOR);

    std::string frame_10004_pth = cv::samples::findFile("frame_10004.png");
    cv::Mat frame_10004 = imread(frame_10004_pth, cv::IMREAD_COLOR);

    std::string frame_10004_pth_new = cv::samples::findFile("frame_10004_new.png");
    cv::Mat frame_10004_new = imread(frame_10004_pth_new, cv::IMREAD_COLOR);


    size_t rows = frame_10001.rows;
    size_t cols = frame_10001.cols;

    assert(rows == frame_10002.rows && cols == frame_10002.cols);

    std::array<cv::Mat, 5> sources = {frame_10001, frame_10002, frame_10003, frame_10004, frame_10004_new};
    slice_mtrx<int> frame_10001_slices(color_N, rows, cols);
    slice_mtrx<int> frame_10002_slices(color_N, rows, cols);
    slice_mtrx<int> frame_10003_slices(color_N, rows, cols);
    slice_mtrx<int> frame_10004_slices(color_N, rows, cols);
    slice_mtrx<int> frame_10005_slices(color_N, rows, cols);

    fill_slicer* launch_fill = new fill_slicer(sources);

    std::thread frame_10001_th(std::bind(&fill_slicer::template fill<0>, launch_fill, frame_10001_slices));
    std::thread frame_10002_th(std::bind(&fill_slicer::template fill<1>, launch_fill, frame_10002_slices));
    std::thread frame_10003_th(std::bind(&fill_slicer::template fill<2>, launch_fill, frame_10003_slices));
    std::thread frame_10004_th(std::bind(&fill_slicer::template fill<3>, launch_fill, frame_10004_slices));
    std::thread frame_10005_th(std::bind(&fill_slicer::template fill<4>, launch_fill, frame_10005_slices));

    frame_10001_th.join();
    frame_10002_th.join();
    frame_10003_th.join();
    frame_10004_th.join();
    frame_10005_th.join();

    delete launch_fill;

    std::array<std::pair<std::string, slice_mtrx<int>>, 5> slicers_arr = {
                                                                          std::make_pair("frame_10001.png", frame_10001_slices),
                                                                          std::make_pair("frame_10002.png", frame_10002_slices),
                                                                          std::make_pair("frame_10003.png", frame_10003_slices),
                                                                          std::make_pair("frame_10004.png", frame_10004_slices),
        std::make_pair("frame_10004_new.png", frame_10005_slices)};

    std::vector<size_t> vec_index(color_N);
    size_t idx = 0;
    for(auto& x : vec_index) {
        x = idx;
        ++idx;
    }
    std::vector<std::future<size_t>> result_solver(color_N);

    for(auto& arr : slicers_arr) {
        solver_slice launch_solver(frame_10001_slices, arr.second);
        auto to_fill_slicer([=](size_t index){
            return std::async(std::launch::async, &solver_slice::solv, launch_solver, index);
        });
        std::string name_out = "Сравнение картинок: frame_10001.png и " + arr.first;
        std::cout << name_out  << std::endl;
        std::transform(vec_index.begin(), vec_index.end(), result_solver.begin(), to_fill_slicer);
        bool predicate = false;
        for(auto& x : result_solver) {
            if(x.get() > 0) predicate = true;
        }
        if(predicate)
            std::cout << "Есть совпадение!"  << std::endl;
        else std::cout << "Нет совпадений!"  << std::endl;
    }

    return 0;
}
