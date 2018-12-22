#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

void printMat(Mat mat, string name = "Matrix") {

    cout << name << " =" << endl << mat << endl << endl;

}

void matObject(string image) {

    Mat A, C;
    A = imread(image, CV_LOAD_IMAGE_COLOR);

    Mat B(A);

    C = A;

    Mat D (A, Rect(10, 10, 100, 100)); // using rectangle
    Mat E = A(Range::all(), Range(1, 3)); // using row and column boundaries

    // Print the width of image
    std::cout << "Width of image: " << A.size().width << std::endl;

    // Cloning and copy functions
    Mat F = A.clone();
    Mat G;

    A.copyTo(G);

}

void creating() {

    Mat M(2, 2, CV_8UC3, Scalar(0, 0, 255));
    std::cout << "M = " << endl << " " << M << endl << endl;

    // Create a header for an already existing Iplimage pointer
    IplImage *img = cvLoadImage("messi.jpg");
    Mat mtx(img); // convret IplImage* -> Mat

    M.create(4, 4, CV_8UC(2));
    std::cout << "M = " << endl << " " << M << endl << endl; 

    Mat Zeros = Mat::zeros(3,3, CV_8UC1);
    cout << "Z = " << endl << " " << Zeros << endl << endl;

   Mat C = (Mat_ <double>(3, 3) << 0, -1, 0, -1, 23, 34, 324, 45, 34);
   cout << "C = " << endl << endl << C << endl << endl;

   Mat RowClone = C.row(1).clone();
   printMat(RowClone, "Row Clone");

   // Fill matrix with random values
   Mat R = Mat(3, 2, CV_8UC3);
   randu(R, Scalar::all(0), Scalar::all(255));
   printMat(R, "Random Mat");
}

int main(int argc, char** argv) {

    matObject("messi.jpg");
    creating();

    return 0;
}