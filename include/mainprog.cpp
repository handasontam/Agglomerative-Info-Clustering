#include <iostream>
#include <fstream>
#include <sstream>
#include <matrix.hpp>
#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <fstream>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/core/cvstd.hpp>
#include <opencv2/core/eigen.hpp>

using namespace std;

int conv() {
    MatrixXd A = load_csv<MatrixXd>("GDS3893.soft"); //Input file path of matrix
    removeRow(A,0); //Remove first row of matrix
    removeZeroRows(A); //Remove rows at the end of matrix with zero values
    MatrixXd G;
    G.noalias() = A.transpose();
    writeToCSVfile("GDS3893_clean.csv", G); //Write MatrixXd to CSV first
    cv::Ptr<cv::ml::TrainData> raw_data = cv::ml::TrainData::loadFromCSV("GDS3893_clean.csv");
    return 0;
}
