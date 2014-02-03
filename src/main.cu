#include <iostream>
#include <cuda.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

void calcCornerBlockHist(const Mat src, int blockSizeX, int blockSizeY, unsigned char * outHist){
	unsigned char *input = (unsigned char*) src.data;
	int count = 0;
	int histIdx;
	for (int j = 0; j < blockSizeY; j++){
		for (int i = 0; i < blockSizeX; i++){
			histIdx = input[src.rows * j + i];
			cout<<"index = "<<src.rows * j + i<<endl;
			outHist[histIdx]++;
			count++;
		}
	}
	cout<<endl;
	cout<<"count = "<<count<<endl;
}

int main(int argc, char** argv){
	if (argc != 2){
		cout<<"usage:\n";
		cout<<"  "<<argv[0]<<" [image file]\n";
		return -1;
	}
	
	Mat matSource;
	matSource = imread(argv[1], 0);
	
	unsigned char hist[256] = {0};
	
	//namedWindow("Original", CV_WINDOW_AUTOSIZE);
	//imshow("Original", matSource);
	cout<<"rows: "<<matSource.rows<<endl;
	cout<<"cols: "<<matSource.cols<<endl;
	
	calcCornerBlockHist(matSource, 4, 4, hist);
	
	cout<<"histogram"<<endl;
	float mean = 0;
	for (int i = 0; i < 256; i++){
		cout<<(int)hist[i]<<" ";
		mean += (float)hist[i];
	}
	cout<<endl;
	
	cout<<"mean = "<<(float)mean/256.0;
	
	cout<<endl;
	//waitKey(0);
	
	cudaDeviceProp prop;
	int count;
	cudaGetDeviceCount(&count);
	cout<<"devices = "<<count<<endl;
	cudaGetDeviceProperties(&prop, 0);
	cout<<"name = "<<prop.name<<endl;
	cout<<"max threads per block = "<<prop.maxThreadsPerBlock<<endl;
	return 0;
}
