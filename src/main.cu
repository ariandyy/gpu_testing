#include <iostream>
#include <iomanip>
#include <cuda.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

__global__ void parCalcCornerBlockHist(){
	
}


void calcCornerBlockHist(const Mat src, int blockSizeX, int blockSizeY, int beginX, int beginY, unsigned char * outHist){
	unsigned char *input = (unsigned char*) src.data;
	int count = 0;
	int bin;
	for (int j = beginY; j < beginY + blockSizeY; j++){
		for (int i = beginX; i < beginX + blockSizeX; i++){
			bin = input[src.rows * j + i];
			//cout<<"index = "<<src.rows * j + i<<endl;
			outHist[bin]++;
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
	
	cudaEvent_t start, stop;
	
	Mat matSource;
	matSource = imread(argv[1], 0);
	
	unsigned char hist[256] = {0};
	
	//namedWindow("Original", CV_WINDOW_AUTOSIZE);
	//imshow("Original", matSource);
	cout<<"source rows: "<<matSource.rows<<endl;
	cout<<"source cols: "<<matSource.cols<<endl;
	
	// timer
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	
	// function call
	calcCornerBlockHist(matSource, 32, 32, 0, 0, hist);
	// for 128x128 block size, why total occurences is less than 128x128=16384?
	
	// timer
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cout<<"CPU histogram took "<<setprecision(5)<<elapsedTime<<" ms"<<endl;
	
	cout<<"histogram"<<endl;
	float mean = 0;
	float sum = 0;
	int occurences = 0;
	for (int i = 0; i < 256; i++){
		cout<<(int)hist[i]<<"+";
		sum += (float)i * hist[i];
		occurences += hist[i];
	}
	mean = sum/occurences;
	cout<<endl;
	cout<<"sum = "<<sum<<endl;
	cout<<"occurences = "<<occurences<<endl;
	cout<<"mean = "<<mean<<endl;
	//cout<<"mean = "<<mean/256.0;
	
	cout<<endl;
	//waitKey(0);
	
	// cuda
	
	
	return 0;
}
