#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

void printMat(Mat mat, string name = "Matrix") {

    cout << name << " =" << endl << mat << endl << endl;

}

void sharpen(const Mat& image, Mat& result) {

    CV_Assert(image.depth() == CV_8U);

    result.create(image.size(), image.type());
    const int num_channels = image.channels();

    cout << "Number of channels are: " << num_channels << endl;

    for(int j = 1; j < image.rows - 1; ++j) {
        
        const uchar* previous = image.ptr<uchar>(j - 1);
        const uchar* current = image.ptr<uchar>(j);
        const uchar* next = image.ptr<uchar>(j + 1);

        uchar* output = result.ptr<uchar>(j);

        for(int i = num_channels; i < num_channels * (image.cols - 1); ++i) {
            *output++ = saturate_cast<uchar>(5 * current[i] - current[i - num_channels] - current[i + num_channels] - previous[i] - next[i]);
        }

    }

    result.row(0).setTo(Scalar(0));
    result.row(result.rows - 1).setTo(Scalar(0));
    result.col(0).setTo(Scalar(0));
    result.col(result.cols - 1).setTo(Scalar(0));
}

int main(int argc, char** argv) {

    Mat image = imread("messi.jpg", CV_LOAD_IMAGE_COLOR);
    //Mat filter_image = imread("messi.jpg", CV_LOAD_IMAGE_COLOR);
    Mat filter_image = image.clone();
    Mat filter_edge_image = filter_image.clone();

    sharpen(image, image);
    //printMat(image, "Original");
    imwrite("messi_sharpen.jpg", image);

    Mat kernel_sharpen = (Mat_<char>(3,3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);
    Mat kernel_edge = (Mat_<char>(3,3) << -1, -1, -1, -1, 8, -1, -1, -1, -1);

    filter2D(filter_image, filter_image, filter_image.depth(), kernel_sharpen);
    imwrite("messi_filter.jpg", filter_image);

    filter2D(filter_edge_image, filter_edge_image, filter_edge_image.depth(), kernel_edge);
    imwrite("messi_edge.jpg", filter_edge_image);

    return 0;
}