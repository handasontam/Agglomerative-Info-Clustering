/*
Reads a CSV file and returns a MatrixXd type
Michael Lim
25 February 2019

Adapted from:
https://stackoverflow.com/questions/34247057/how-to-read-csv-file-and-assign-to-eigen-matrix
https://stackoverflow.com/questions/25114675/removing-zero-rows-from-a-matrix-elegant-way
*/

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
//#if defined(_GLIBCXX_HAS_GTHREADS) && defined(_GLIBCXX_USE_C99_STDINT_TR1)

using namespace Eigen;

template<typename M>
M load_csv (std::string path) {
    //Input file path of CSV and load matrix
    std::ifstream indata;
    indata.open(path);
    std::string line;

    std::vector<double> values;
    int rows = 0;
    int colno;

    //While-loop for each line in the CSV
    while (std::getline(indata, line)) {
        colno = 0; //Count column index
        std::stringstream lineStream(line); //Turn line into linestream
        std::string cell; //Define cell string

        while (std::getline(lineStream, cell, '\t')) {
            ++colno;
            //Skip column 1 and 2 which does not contain numberical value
            if (colno==1 || colno==2) {
                continue;
            }
            //Insert value of cell that is in contained in CSV
            values.push_back(atof(cell.c_str()));
        }
        rows += 1;
    }

    //Return MatrixXd that has 122 features
    return Map<const Matrix<typename M::Scalar, M::RowsAtCompileTime, M::ColsAtCompileTime, RowMajor>>(values.data(), rows, 122);
}

//Function to remove rows with zero as value
void removeZeroRows(Eigen::MatrixXd& mat) {
  Matrix<bool, Dynamic, 1> empty = (mat.array() == 0).rowwise().all();
  size_t last = mat.rows() - 1;
  for (size_t i = 0; i < last + 1;) {
    if (empty(i)) {
      empty.segment<1>(i).swap(empty.segment<1>(last));
      --last;
    } else ++i;
  }
  mat.conservativeResize(last + 1, mat.cols());
}

//Function to remove a selected row
void removeRow(Eigen::MatrixXd& mat, int rowToRemove) {
    unsigned int numRows = mat.rows() - 1;
    unsigned int numCols = mat.cols();
    unsigned int rowPos = numRows - rowToRemove;
    if( rowToRemove < numRows ) {
        mat.block(rowToRemove, 0, rowPos, numCols) = mat.block(rowToRemove + 1, 0, rowPos,numCols);
    }
    mat.conservativeResize(numRows, numCols);
}

//Write clean matrix to CSV
const static IOFormat CSVFormat(StreamPrecision, DontAlignCols, ", ", "\n");
int writeToCSVfile(std::string name, MatrixXd matrix) {
    std::ofstream file(name.c_str());
    file << matrix.format(CSVFormat);
    return 0;
}
