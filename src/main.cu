#include <iostream>
#include <cuda.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main(int argc, char** argv){
	if (argc != 2){
		cout<<"usage:\n";
		cout<<"  "<<argv[0]<<" [image file]\n";
		return -1;
	}
	
	Mat matSource;
	matSource = imread(argv[1], 0);
	
	namedWindow("Original", CV_WINDOW_AUTOSIZE);
	imshow("Original", matSource);
	cout<<"rows: "<<matSource.rows<<endl;
	cout<<"cols: "<<MatSource.cols<<endl;
	
	waitKey(0);
	return 0;
}
