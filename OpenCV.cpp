#include <opencv2/opencv.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/videoio.hpp"
#include <iostream>
#include <sys/time.h>

using namespace cv;
using namespace std;

// comment out the below line to run without timing printouts 
#define TIME

//colors
const Scalar RED = Scalar(0, 0, 255), BLUE = Scalar(255, 0, 0), GREEN = Scalar(
		0, 255, 0), ORANGE = Scalar(0, 128, 255), YELLOW = Scalar(0, 255, 255),
		PINK = Scalar(255, 0, 255), WHITE = Scalar(255, 255, 255);

//info about target
struct targetInfo {
	bool cameraConnected;
	bool targetDetected;

	bool aligned;

	double targetDistance;
	double targetAngle;

	Point center;

};
//threshold parameters for goal tape
int minRVal = 254;
int minGVal = 254;
int minBVal = 254;
int maxRVal = 255; //bad values....
int maxGVal = 255;
int maxBVal = 255;

Mat colorFilter(Mat original) {
	Mat colorFilter;
	cvtColor(original,colorFilter,cv::COLOR_BGR2GRAY);
//	inRange(original, Scalar( minRVal, minGVal, minBVal), Scalar(maxRVal, maxGVal,maxBVal),
//			colorFilter);

	return colorFilter;
}
Mat close(Mat original) {
	//dilate
	Mat dilated;
	dilate(original, dilated, MORPH_RECT);

	//erode
	Mat closed;
	erode(dilated, closed, MORPH_RECT);

	return closed;
}
Mat goalFilter(Mat original) {
	//should return a mat with only the goals left in it
	Mat colorfilter = colorFilter(original);

	Mat close1 = close(colorfilter);
	Mat close2 = close(close1);

	return close2;
}
bool targetDetected(Mat frame, Mat contoursMat) {
	//detect all contours
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	cv::findContours(frame, contours, hierarchy, RETR_EXTERNAL,
			CHAIN_APPROX_SIMPLE);
/*	
	int idx = 0;
	for (; idx >= 0; idx = hierarchy[idx][0]) {
		Scalar color(rand() & 255, rand() & 255, rand() & 255);
		drawContours(contoursMat, contours, idx, color, CV_FILLED, 8,
				hierarchy);
	}
*/
	return false;
}
Mat getImage(VideoCapture cap,int number){

	//flush buffer
	Mat frame;
	for(int i=0; i<1; i++){
		cap >> frame;
	}
	return frame;
	//printf("frame type %i",frame.type());
	Mat mean = Mat::zeros(frame.rows, frame.cols,CV_32FC3);
	for(int i = 0; i < number; i++){
		Mat tempFrame;
		cap >> tempFrame;
		cv::accumulate(tempFrame, mean);
	}
	mean = mean / number;
	mean.convertTo(mean,16);
	return mean;
}

void flushFrames(VideoCapture cap, int count) {

#ifdef TIME
	struct timeval start;
	gettimeofday(&start, NULL);
	long int start_ms = start.tv_sec * 1000 + start.tv_usec / 1000;
#endif

	for(int i=0; i<count; i++) {
		Mat temp;
		cap >> temp;
	}

#ifdef TIME
	struct timeval end;
	gettimeofday(&end, NULL);
	long int end_ms = end.tv_sec * 1000 + end.tv_usec / 1000;
	long total_ms = end_ms - start_ms;
	printf("\n flusing %i frames took %i ms", count, total_ms);
#endif

}

void readFrame(VideoCapture cap, Mat& mat) {

#ifdef TIME
	struct timeval start;
	gettimeofday(&start, NULL);
	long int start_ms = start.tv_sec * 1000 + start.tv_usec / 1000;
#endif

	cap >> mat;

#ifdef TIME
	struct timeval end;
	gettimeofday(&end, NULL);
	long int end_ms = end.tv_sec * 1000 + end.tv_usec / 1000;
	long total_ms = end_ms - start_ms;
	printf("\n reading a frame took %i ms", total_ms);
#endif
}


#define VIDEO_CAPTURE_INDEX 0 
VideoCapture setupCamera() {

#ifdef TIME
	struct timeval start;
	gettimeofday(&start, NULL);
	long int start_ms = start.tv_sec * 1000 + start.tv_usec / 1000;
#endif

	cv::VideoCapture cap(VIDEO_CAPTURE_INDEX); //selects webcam 0 as our main camera
	if (!cap.isOpened()) {
		//if no webcam exists, kill program
		printf("No webcam detected! Closing...");
		return -1;
	}
	//setup the camera with correct settings to detect goal
	cap.set(CV_CAP_PROP_BUFFERSIZE, 1);
	cap.set(CV_CAP_PROP_FRAME_WIDTH, 640);
	cap.set(CV_CAP_PROP_FRAME_HEIGHT, 480);
	return cap;

#ifdef TIME
	struct timeval end;
	gettimeofday(&end, NULL);
	long int end_ms = end.tv_sec * 1000 + end.tv_usec / 1000;
	long total_ms = end_ms - start_ms;
	printf("\n setting up camera took %i ms", total_ms);
#endif

}

void toggleLed() {

}

void detectGoalTargets(VideoCapture cap) {
	
	// Assumption: LED is on

	/*
		General processing idea:

		g = garbage frame
		o = frame with LED on
		f = frame with LED off

		[ g | g | g | g | g ]
		  read with LED on
		[ o | g | g | g | g ]
		  read with LED off
		[ f | o | g | g | g ]
		  flush 3 frames out
		[ g | g | g | f | o ]
		  read 2 useful images

	*/
	
	flushFrames(cap, 1); 	// put LED on frame in buffer
	toggleLed();		// LED should be off now
	flushFrames(cap, 1); 	// put LED off frame in buffer
	toggleLed();		// LED should be on now
	flushFrames(cap, 3);
	
	Mat ledOnFrame;
	Mat ledOffFrame;
	readFrame(cap, ledOnFrame);
	readFrame(cap, ledOffFrame);

	// perform subtraction

	// detect contours

}

void runEagleVision() {

	VideoCapture cap = setupCamera();

	while(true) {

#ifdef TIME
		struct timeval start;
		gettimeofday(&start, NULL);
		long int start_ms = start.tv_sec * 1000 + start.tv_usec / 1000;
#endif

		detectGoalTargets(cap);

#ifdef TIME
		struct timeval end;
		gettimeofday(&end, NULL);
		long int end_ms = end.tv_sec * 1000 + end.tv_usec / 1000;
		long total_ms = end_ms - start_ms;
		printf("\n full detect goal operation took %i ms\n", total_ms);
#endif
	}
}

/*
int videoCapture() {
	
	printf("STARTING VIDEO CAPTURE\n");
	time(&start_time);
	
	cv::VideoCapture cap(VIDEO_CAPTURE_INDEX); //selects webcam 0 as our main camera
	if (!cap.isOpened()) {
		//if no webcam exists, kill program
		printf("No webcam detected! Closing...");
		return -1;
	}
	//setup the camera with correct settings to detect goal
	cap.set(CV_CAP_PROP_BUFFERSIZE, 1);
	cap.set(CV_CAP_PROP_FRAME_WIDTH, 640);
	cap.set(CV_CAP_PROP_FRAME_HEIGHT, 480);

	for(int i=0; i< 10; i++) {
		printf("\nstarting iteration %i\n", i);
	
		Mat frame;
		Mat img1;
		Mat img2;
		Mat subtractedImage;
		Mat contourDrawing;


		readFrameAfterFlushing(cap, img1, 5);

		while(true) {
			struct timeval start_sti;
			gettimeofday(&start_sti, NULL);
			long int start_ms = start_sti.tv_sec * 1000 + start_sti.tv_usec / 1000;
			
			readFrameAfterFlushing(cap, img1, 0);
			imshow("img1", img1);
			cv::waitKey(0);
		
			struct timeval end_sti;
			gettimeofday(&end_sti, NULL);
			long int end_ms = end_sti.tv_sec * 1000 + end_sti.tv_usec / 1000;
			long total_ms = end_ms - start_ms;
			printf("\n reading the image took %i ms\n", total_ms);
		}
	
		//img2 = readFrameAfterFlushing(cap, 0);

		cv::subtract(img1, img2, subtractedImage);
		contourDrawing = Mat::zeros(subtractedImage.rows, subtractedImage.cols, CV_8UC3);
	
	#ifndef BENCHMARK_MODE
		imshow("filtered Image", goalFilter(subtractedImage));
	#endif

		//targetDetected(goalFilter(subtractedImage), contourDrawing);

	#ifndef BENCHMARK_MODE
		imshow("contours",contourDrawing);
		waitKey(0);
	#endif
	}
	return 0;
}
*/

void imageCapture(int argc, char** argv) {
	Mat image;
	Mat contourDrawing = Mat::zeros(image.rows, image.cols, CV_8UC3);

	image = imread(argv[1], 1);

	namedWindow("Display Image", WINDOW_AUTOSIZE);

	targetDetected(goalFilter(image), contourDrawing);

	imshow("Display Image", image);
	imshow("filtered Image", goalFilter(image));
	imshow("Dst", contourDrawing);

	waitKey(0);
}

int main(int argc, char** argv) {
	runEagleVision();
	return 0;
}
