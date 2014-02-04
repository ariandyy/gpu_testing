#include <iostream>
#include <iomanip>
#include <cuda.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

__global__ void cudaCalcCornerBlockHist(unsigned char * src, 
										int rows, 
										int cols, 
										int blockSizeX, 
										int blockSizeY, 
										int beginX, 
										int beginY, 
										unsigned int * outHist){
	int idx = rows * (threadIdx.y + beginY) + (threadIdx.x + beginX);
	atomicAdd(&(outHist[src[idx]]), 1);
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

void processHistogram(unsigned char * hist, int max){
	float mean = 0;
	float sum = 0;
	int occurences = 0;
	
	for (int i = 0; i < max; i++){
		cout<<(int)hist[i]<<" ";
		sum += (float)i * hist[i];
		occurences += hist[i];
	}
	mean = sum/occurences;
	cout<<endl;
	cout<<"sum = "<<sum<<endl;
	cout<<"occurences = "<<occurences<<endl;
	cout<<"mean = "<<setprecision(5)<<mean<<endl;
}

void processHistogram(unsigned int * hist, int max){
	float mean = 0;
	float sum = 0;
	int occurences = 0;
	
	for (int i = 0; i < max; i++){
		cout<<(int)hist[i]<<" ";
		sum += (float)i * hist[i];
		occurences += hist[i];
	}
	mean = sum/occurences;
	cout<<endl;
	cout<<"sum = "<<sum<<endl;
	cout<<"occurences = "<<occurences<<endl;
	cout<<"mean = "<<setprecision(5)<<mean<<endl;
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
	
	int blockSizeX = 32;
	int blockSizeY = 32;
	int startX = 0;
	int startY = 0;
	
	
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
	calcCornerBlockHist(matSource, blockSizeX, blockSizeY, startX, startY, hist);
	// for 128x128 block size, why total occurences is less than 128x128=16384?
	
	// timer
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cout<<"CPU histogram took "<<setprecision(5)<<elapsedTime*1000<<" ns"<<endl;
	
	cout<<"histogram"<<endl;
	processHistogram(hist, 256);
	cout<<endl;
	
	// cuda
	unsigned char* host_image = matSource.data;
	unsigned int host_hist[256];
	
	unsigned char* dev_image;
	size_t size = matSource.rows*matSource.cols*sizeof(unsigned char);
	cudaMalloc(&dev_image, size);
	cudaMemcpy(dev_image, host_image, size, cudaMemcpyHostToDevice);
	
	unsigned int * dev_hist;
	cudaMalloc(&dev_hist, 256*sizeof(unsigned int));
	cudaMemset(dev_hist, 0, 256*sizeof(unsigned int));
	
	cudaEventRecord(start, 0);
	cudaCalcCornerBlockHist<<<1, dim3(32,32, 1)>>>(dev_image, matSource.rows, matSource.cols, blockSizeX, blockSizeY, startX, startY, dev_hist);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cout<<"GPU histogram took "<<setprecision(5)<<elapsedTime*1000<<" ns"<<endl;
	
	cudaMemcpy(host_hist, dev_hist, 256*sizeof(unsigned int), cudaMemcpyDeviceToHost);
	cout<<"histogram from cuda"<<endl;
	processHistogram(host_hist, 256);
	
	// cleanup
	cudaFree(dev_image);
	cudaFree(dev_hist);
	return 0;
}
