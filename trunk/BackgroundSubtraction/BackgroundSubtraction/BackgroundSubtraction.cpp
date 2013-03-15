// BackgroundSubtraction.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <opencv2\core\core.hpp>
#include <opencv/cvaux.h>
#include <opencv/cxmisc.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/legacy/legacy.hpp>

using namespace std;
using namespace cv;
using namespace cv::gpu;


/********************************************/
/** Application Options						*/
/********************************************/
const bool TRACK_MOTION = true;			// whether or not to track motion
const bool DRAW_CONTOURS = false;		// display contour outlines in the output
const bool DEBUG = false;
const bool LIVE_DEMO = false;			// use input from a live traffic camera in real-time
const bool SAVE_OUTPUT = false;			// save the output to an avi file named output.avi

// motion tracking algorthm selection
const bool USE_LUCAS_KANADE = false;
const int  MAX_CORNERS = 500;
const bool USE_SIFT = false;
const bool BLOB_COMPARE = true;




// Variables for saving output as a video
VideoCapture camera;
CvVideoWriter* writer;
IplImage *writeFrame = 0;

//VARIABLES for CODEBOOK METHOD:
CvBGCodeBookModel* model = 0;
const int NCHANNELS = 3;
bool ch[NCHANNELS]={true,true,true}; // This sets what channels should be adjusted for background bounds


// VARIABLES for FindContours
int thresh = 200;
int max_thresh = 255;
cv::RNG rng(12345);


// input filename, if used
const char* filename = 0;


IplImage ipl_image;

IplImage *rawImage = 0;
IplImage *resized = 0;
IplImage *yuvImage = 0; // yuvImage is for codebook method, because a color-spaced correlated with brightness is considered to be preferential for this method

IplImage *justForeground = 0;
IplImage *justForegroundGray = 0;

IplImage *ImaskCodeBook = 0;
IplImage *ImaskCodeBookInv = 0;
IplImage *ImaskCodeBookClosed = 0;

IplImage *prevFrameRaw = 0;
IplImage *prevFrameGray = 0;
IplImage *prevFrameMotionBlobs = 0;


IplImage *output = 0;
CvCapture *capture = 0;
cv::Mat_<cv::Vec3b> image;						

int c, n, nframes = 0;


// number of frames used to learn the background using the codebook method
int nframesToLearnBG = 300;


// Find Contours Variables
IplImage* canny_output = 0;
IplImage* canny_output1 = 0;
IplImage* canny_output2 = 0;
CvMemStorage* 	mem_storage = NULL;
CvMemStorage* 	mem_storage1 = NULL;
CvMemStorage* 	mem_storage2 = NULL;
CvSeq* contours = 0;
CvSeq* contours1 = 0;
CvSeq* contours2 = 0;
CvSeq* contours_remember1 = 0;
CvSeq* contours_remember2 = 0;


// SIFT Vars
Mat img1;
Mat img2;
int minHessian = 400;
SiftFeatureDetector detector( minHessian );
std::vector<KeyPoint> keypoints_1, keypoints_2;
SiftDescriptorExtractor extractor;
Mat descriptors_1, descriptors_2;
BruteForceMatcher< L2<float> > matcher;
// alternative option for matcher - FlannBasedMatcher
//FlannBasedMatcher matcher;
std::vector< DMatch > matches;
std::vector< DMatch > good_matches;
int fuzziness_min = -1;
int fuzziness_max = 80;
Mat img_matches;
int keypoint_distance = 0;
int key1;
int key2;
int x_dist;
int y_dist;

// used to convert int to char* for printing number labels
char buffer [33];

// font used for printing text on the output image
CvFont font;
	



// custom blob matcher variables
double compare_result;
int count1 = 0;
int count2 = 0;
double bestMatchesVal[100];
int bestMatchesId[100];
CvScalar prevFrameColors[100];
CvScalar curFrameColors[100];
int curFrameLabels[100];
int prevFrameLabels[100];
int labelsInUse[100];
CvRect rect1;
CvRect rect2;
CvScalar color;

// custom blob matcher weights
float moment_weight = 1;
float size_weight = 2;
float proximity_weight = 20;
float min_blob_area = 20;
int blob_counter = 0;
float blob_compare_threshold = 15;

void processFrame();





int main(int argc, char** argv)
{

	if(SAVE_OUTPUT)
	{
		// write mjpeg stream to video
		CvSize size = cvSize(640,480);
		writer = cvCreateVideoWriter("output.avi",CV_FOURCC('D','I','V','X'),15,size);
	}


	if(LIVE_DEMO)
	{
		printf("opening remote webcam...\n");
		camera.open("http://itwebcamcp700.fullerton.edu/mjpg/video.mjpg");
		
		// alternative cameras used in testing
		//camera.open("http://cam1.brentwood-tn.org/mjpg/video.mjpg");
		//camera.open("http://216.8.159.21/mjpg/video.mjpg");

		//MAIN PROCESSING LOOP:
		for(;;)
		{
			camera.grab();
			camera.retrieve(image);
			
			// Setup a rectangle to define region of interest
			// bottom half of input from test cameras is the only relevant part for vehicles and pedestrians
			cv::Rect roi(0, cvRound(image.rows/2)-1, image.cols, cvRound(image.rows/2));

			// crop input image to just roi
			ipl_image = image(roi);


			// resize input video frame size to 1/4 of the original size to speed up performance to real-time*
			// *real-time performance is dependant on the tracking algorithm selected and the number of objects needing to be tracked
			resized = cvCreateImage( cvSize(ipl_image.width/2,ipl_image.height/2),8, 3 );
			cvResize(&ipl_image,resized);
			

			rawImage = resized;

			// update the frame counter
			++nframes;

			printf("frame: %d\n",nframes);

			if(!rawImage) 
				break;

			// process this frame of video
			processFrame();
		}
	}
	else
	{
		// use local test video rather than live traffic camera
		capture = cvCaptureFromAVI("output1.avi"); 
		for(;;)
		{
			image = cvQueryFrame( capture );

			if(image.rows == 0 || image.cols == 0)
				break;

			// Setup a rectangle to define region of interest
			cv::Rect roi(0, cvRound(image.rows/2)-1, image.cols, cvRound(image.rows/2));

			// crop input image to just roi
			ipl_image = image(roi);

			resized = cvCreateImage( cvSize(ipl_image.width/2,ipl_image.height/2),8, 3 );
			cvResize(&ipl_image,resized);

			rawImage = resized;

			++nframes;

			printf("frame: %d\n",nframes);

			if(!rawImage) 
				break;

			processFrame();
		}
	}

    // release capture and gui output windows
    cvReleaseCapture( &capture );
    cvDestroyWindow("Raw");
	cvDestroyWindow("Output");
    cvDestroyWindow("Foreground CodeBook");
	cvDestroyWindow("Foreground CodeBook Closed");

    return 0;
}


void processFrame()
{
	// during the first frame, allocate the necessary image containers for processing
    if( nframes == 1 && rawImage )
    {
        // CODEBOOK METHOD ALLOCATION
        yuvImage = cvCloneImage(rawImage);
		justForeground = cvCreateImage( cvGetSize(resized), IPL_DEPTH_8U, rawImage->nChannels );
		justForegroundGray = cvCreateImage( cvGetSize(resized), IPL_DEPTH_8U, 1 );
        ImaskCodeBook = cvCreateImage( cvGetSize(resized), IPL_DEPTH_8U, 1 );
		ImaskCodeBookInv = cvCreateImage( cvGetSize(resized), IPL_DEPTH_8U, 1 );
		ImaskCodeBookClosed = cvCreateImage( cvGetSize(resized), IPL_DEPTH_8U, 1 );
        canny_output = cvCreateImage( cvGetSize(resized), IPL_DEPTH_8U, 1 );
		canny_output1 = cvCreateImage( cvGetSize(resized), IPL_DEPTH_8U, 1 );
		canny_output2 = cvCreateImage( cvGetSize(resized), IPL_DEPTH_8U, 1 );
		output = cvCreateImage(cvGetSize(resized),IPL_DEPTH_8U,3);
		prevFrameGray = cvCreateImage( cvGetSize(resized), IPL_DEPTH_8U, 1 );
        cvSet(ImaskCodeBook,cvScalar(255));


		// initialize codebook model
		model = cvCreateBGCodeBookModel();
		//Set color thresholds to default values
		// oringal min = 3, max = 10 --> now 2 and 14
		model->modMin[0] = 1;
		model->modMin[1] = model->modMin[2] = 1;
		model->modMax[0] = 14;
		model->modMax[1] = model->modMax[2] = 14;
		model->cbBounds[0] = model->cbBounds[1] = model->cbBounds[2] = 14;

		// initialize font
		cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX, 0.5f, 0.5f, 0, 2);

		// initialize contour memory
		contours_remember1 = contours1;
		contours_remember2 = contours2;

		// initialize blob labels
		for(int f=0;f<100;f++)
		{
			prevFrameLabels[f] = f;
		}

        cvNamedWindow("Raw",1);
		cvNamedWindow("Output",1);
        cvNamedWindow("Foreground CodeBook",1);
		cvNamedWindow("Foreground CodeBook Closed",1);
    }
              
    if( rawImage )
    {
        cvCvtColor( rawImage, yuvImage, CV_BGR2YCrCb ); // convert to YUV

        // build background model
        if(nframes-1 < nframesToLearnBG  )
            cvBGCodeBookUpdate( model, yuvImage );

        if( nframes-1 == nframesToLearnBG  )
		{
			printf("codebook stale entries cleared\n");
            cvBGCodeBookClearStale( model, model->t/2 );
		}

        // Find the foreground if any
        if( nframes-1 >= nframesToLearnBG  )
        {
            // Find foreground by codebook method
            cvBGCodeBookDiff( model, yuvImage, ImaskCodeBook );

			cvCopy(ImaskCodeBook,ImaskCodeBookClosed);
			
			// dilate and erode to close blobs
			cvDilate(ImaskCodeBookClosed,ImaskCodeBookClosed);
			cvDilate(ImaskCodeBookClosed,ImaskCodeBookClosed);
			cvDilate(ImaskCodeBookClosed,ImaskCodeBookClosed);
			cvErode(ImaskCodeBookClosed,ImaskCodeBookClosed);
			cvErode(ImaskCodeBookClosed,ImaskCodeBookClosed);
			cvErode(ImaskCodeBookClosed,ImaskCodeBookClosed);
			cvErode(ImaskCodeBookClosed,ImaskCodeBookClosed);
			

			// get a list of blobs (contours)
			cvCanny( ImaskCodeBookClosed, canny_output, thresh, thresh*2, 3 );

			/// Find contours
			if( mem_storage == NULL )
			{
				mem_storage = cvCreateMemStorage(0);
			}
			else
			{
				cvClearMemStorage( mem_storage );
			}
			cvFindContours(canny_output, mem_storage, &contours,88,CV_RETR_EXTERNAL);


			if(contours->first->count > 50)
				cvBGCodeBookClearStale( model, 0 );

			// fill in holes within each blob
			cvDrawContours(ImaskCodeBookClosed,contours,cvScalarAll(255),cvScalarAll(255),1,CV_FILLED, 8, cvPoint(0,0));

			/*
			// invert the foreground mask to get the background mask -- used to show only the background (foreground removed)
			for(int i=0;i<ImaskCodeBookClosed->height;i++)
				for(int j=0;j<ImaskCodeBookClosed->width;j++)
					for(int k=0;k<ImaskCodeBookClosed->nChannels;k++)  //loop to read for each channel
						ImaskCodeBookInv->imageData[i*ImaskCodeBookClosed->widthStep+j*ImaskCodeBookClosed->nChannels+k]=255-ImaskCodeBookClosed->imageData[i*ImaskCodeBookClosed->widthStep+j*ImaskCodeBookClosed->nChannels+k];    //inverting the image
			*/

			// create an image of just moving objects
			cvCopy(rawImage,justForeground,ImaskCodeBookClosed);
            

			/************************************/
			/* Track Motion						*/
			/************************************/
			if(TRACK_MOTION && USE_LUCAS_KANADE)
			{
				if(prevFrameMotionBlobs)
				{

					cvCopy(rawImage,output);

					cvCvtColor(justForeground,justForegroundGray,CV_BGR2GRAY);
						

					CvSize      img_sz    = cvGetSize( prevFrameMotionBlobs );
					int         win_size = 10;
				
				
					// The first thing we need to do is get the features
					// we want to track.
					//
					IplImage* eig_image = cvCreateImage( img_sz, IPL_DEPTH_32F, 1 );
					IplImage* tmp_image = cvCreateImage( img_sz, IPL_DEPTH_32F, 1 );
					int              corner_count = MAX_CORNERS;
					CvPoint2D32f* cornersA        = new CvPoint2D32f[ MAX_CORNERS ];
					cvGoodFeaturesToTrack(
						prevFrameMotionBlobs,
						eig_image,
						tmp_image,
						cornersA,
						&corner_count,
						0.01,
						5.0,
						0,
						3,
						0,
						0.04
					);

					cvFindCornerSubPix(
						prevFrameMotionBlobs,
						cornersA,
						corner_count,
						cvSize(win_size,win_size),
						cvSize(-1,-1),
						cvTermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS,20,0.03)
					);

					// Call the Lucas Kanade algorithm
					//
					char features_found[ MAX_CORNERS ];
					float feature_errors[ MAX_CORNERS ];
					CvSize pyr_sz = cvSize( prevFrameMotionBlobs->width+8, justForegroundGray->height/3 );
					IplImage* pyrA = cvCreateImage( pyr_sz, IPL_DEPTH_32F, 1 );
					IplImage* pyrB = cvCreateImage( pyr_sz, IPL_DEPTH_32F, 1 );
					CvPoint2D32f* cornersB        = new CvPoint2D32f[ MAX_CORNERS ];
				
					cvCalcOpticalFlowPyrLK(
						prevFrameMotionBlobs,
						justForegroundGray,
						pyrA,
						pyrB,
						cornersA,
						cornersB,
						corner_count,
						cvSize( win_size,win_size ),
						5,
						features_found,
						feature_errors,
						cvTermCriteria( CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, .3 ),
						0
					);

					// Now make some image of what we are looking at:
					//
					for( int i=0; i<corner_count; i++ )
					{
						if( features_found[i]==0|| feature_errors[i]>550 )
						{
							//  printf("Error is %f/n",feature_errors[i]);
							continue;
						}
						//    printf("Got it/n");
						CvPoint p0 = cvPoint(
						cvRound( cornersA[i].x ),
						cvRound( cornersA[i].y )
						);
						CvPoint p1 = cvPoint(
						cvRound( cornersB[i].x ),
						cvRound( cornersB[i].y )
						);
						if(abs(p0.x*p0.x - p1.x*p1.x) + abs(p0.y*p0.y - p1.y*p1.y) > 10)
							cvLine( output, p0, p1, CV_RGB(255,0,0),2 );
					}

					// save current frame as previous frame for next round
					cvCopy(justForegroundGray,prevFrameMotionBlobs);
				}
				else
				{
					// create the first previous frame
					prevFrameMotionBlobs = cvCreateImage( cvGetSize(resized), IPL_DEPTH_8U, 1 );
					cvCvtColor(justForeground,prevFrameMotionBlobs,CV_BGR2GRAY);
				}
			}


			if(TRACK_MOTION && BLOB_COMPARE)
			{
				if(prevFrameMotionBlobs)
				{
					cvCopy(resized,output);

					// get a list of blobs (contours)
					cvCanny(prevFrameMotionBlobs, canny_output1, thresh, thresh*2, 3 );
					if( mem_storage1 == NULL )
					{
						mem_storage1 = cvCreateMemStorage(0);
					}
					else
					{
						cvClearMemStorage( mem_storage1 );
					}
					cvFindContours(canny_output1, mem_storage1, &contours1,88,CV_RETR_EXTERNAL);

					cvCanny(ImaskCodeBookClosed, canny_output2, thresh, thresh*2, 3 );
					if( mem_storage2 == NULL )
					{
						mem_storage2 = cvCreateMemStorage(0);
					}
					else
					{
						cvClearMemStorage( mem_storage2 );
					}
					cvFindContours(canny_output2, mem_storage2, &contours2,88,CV_RETR_EXTERNAL);


					contours_remember1 = contours1;
					contours_remember2 = contours2;

					fill_n(bestMatchesVal, 100, 999999999);
					fill_n(bestMatchesId, 100, -1);
					fill_n(curFrameLabels, 100, -1);
					
					if(DEBUG)
						printf("contour1 size: %d\n",contours1->first->count);
					
					for (; contours1 != 0; contours1 = contours1->h_next)
					{
							
						count1++;

						rect1 = cvBoundingRect( contours1);

						if(rect1.width * rect1.height < min_blob_area)
						{
							// ignoring small blob from prev frame
							continue;
						}

						cvPutText(prevFrameRaw,itoa(count1,buffer,10),Point(rect1.x,rect1.y), &font, CV_RGB(255, 255, 255));
						cvDrawContours(prevFrameRaw,contours1,cvScalarAll(255),cvScalarAll(255),-1);
				
						cvCopy(resized,output);

						for(; contours2 != 0; contours2 = contours2->h_next)
						{
							count2++;
							rect2 = cvBoundingRect( contours2);
							if(rect2.width * rect2.height < min_blob_area)
							{
								//printf("throwing out small blob from current frame....%d\n",count2);
								//cvClearSeq(contours2);
								continue;
							}

					
							// log((moment_weight)moment difference) + log((size_weight)size difference) + log((location_weight)location difference)
							compare_result =	log(moment_weight * (1+cvMatchShapes(contours1,contours2,CV_CONTOURS_MATCH_I2))) +
												log((size_weight * (1+abs(rect1.width*rect1.height - rect2.width*rect2.height)))) +
												log((proximity_weight*(1+(abs(rect1.x-rect2.x)*abs(rect1.y-rect2.y)))));

							if(compare_result < bestMatchesVal[count2])
							{
								bestMatchesVal[count2] = compare_result;
								bestMatchesId[count2] = count1;
							}
							if(DEBUG)
							{
								printf("[%d][%d] compare result: %f\n",count1,count2,compare_result);
								printf("moment diff: %f\n",log(moment_weight * (1+cvMatchShapes(contours1,contours2,CV_CONTOURS_MATCH_I2))));
								printf("area diff: %f\n",log((size_weight * (1+abs(rect1.width*rect1.height - rect2.width*rect2.height)))));
								printf("location diff: %f\n",log((proximity_weight*(1+(abs(rect1.x-rect2.x)*abs(rect1.y-rect2.y))))));
								cvWaitKey(1);
							}
						}
						count2 = 0;
						contours2 = contours_remember2;
						
					}
					count1 = 0;
					contours1 = contours_remember1;		
			
					// identify the best score of each blob in the current frame
					for(int m=0;m<100;m++)
					{
						if(bestMatchesId[m] != -1)
						{
							if(DEBUG)
								printf("best match for curr: %d is prev: %d with score of %f\n",m,bestMatchesId[m],bestMatchesVal[m]);

							if(bestMatchesVal[m] > blob_compare_threshold)
							{
								// assign blob a new id and color
								curFrameColors[m] = CV_RGB( rand()&255, rand()&255, rand()&255 );
								curFrameLabels[m] = ++blob_counter;
							}
							else
							{
								// assign matching id from previous frame
								//curFrameColors[m] = prevFrameColors[bestMatchesId[m]];
								curFrameLabels[m] = prevFrameLabels[bestMatchesId[m]];
							}
							cvWaitKey(1);
						}
					}
					
					if(DEBUG)
					{
						cvShowImage("prev frame raw",prevFrameRaw);
						cvWaitKey(1);
					}
					
					// print labels for best matching blob from previous frame if a best match exists
					count2=0;
					for(int f=0;f<100;f++)
					{
						prevFrameLabels[f] = f;
						labelsInUse[f] = -1;
					}
					for(; contours2 != 0; contours2 = contours2->h_next)
					{
						count2++;
						rect2 = cvBoundingRect(contours2);
						if(rect2.width * rect2.height < min_blob_area)
						{
							// throwing out small blob when displaying result
							continue;
						}

						if(curFrameLabels[count2] != -1)
						{
							if(DEBUG)
								printf("matching label from previous frame %d\n",curFrameLabels[count2]);

							// if label is in use in the current frame... two blobs match a single blob in the previous frame, then split blob labels by giving subsequent blobs a fresh label
							int label = -1;
							for(int l=0;l<100;l++)
							{
								if(curFrameLabels[count2] == labelsInUse[l])
								{
									label = ++blob_counter;
									break;
								}
							}
							if(label == -1)
							{
								label = curFrameLabels[count2];
							}
							labelsInUse[count2] = label;
							curFrameLabels[count2] = label;
							cvPutText(output,itoa(label,buffer,10),Point(rect2.x,rect2.y), &font, CV_RGB(0, 255, 0));
							curFrameColors[count2] = CV_RGB( rand()&255, rand()&255, rand()&255 );
						}
					}
					count2 = 0;	
						

					// save current frame as previous frame for next round
					cvCopy(ImaskCodeBookClosed,prevFrameMotionBlobs);
					cvCopy(resized,prevFrameRaw);
						
					// copy colors
					for(int c=0;c<100;c++)
					{
						prevFrameColors[c] = curFrameColors[c];
						curFrameColors[c] = CV_RGB(0,0,0);
						prevFrameLabels[c] = curFrameLabels[c];
						curFrameLabels[c] = -1;
					}
				}
				else
				{
					// create the first previous frame
					prevFrameMotionBlobs = cvCreateImage( cvGetSize(resized), IPL_DEPTH_8U, 1 );
					prevFrameRaw = cvCreateImage( cvGetSize(resized), IPL_DEPTH_8U, 3 );
					cvCopy(ImaskCodeBookClosed,prevFrameMotionBlobs);
					cvCopy(resized,prevFrameRaw);
				}
			}


			if(TRACK_MOTION && USE_SIFT)
			{
				if(prevFrameGray)
				{
					keypoints_1.clear();
					keypoints_2.clear();
					matches.clear();
					good_matches.clear();

					cvCvtColor(resized,justForegroundGray,CV_BGR2GRAY);

					try
					{
						img1 = Mat(justForegroundGray);
						img2 = Mat(prevFrameGray);
	
						//-- Step 1: Detect the keypoints using SURF Detector
						detector.detect( img1, keypoints_1 );
						detector.detect( img2, keypoints_2 );

						//-- Step 2: Calculate descriptors (feature vectors)
						extractor.compute( img1, keypoints_1, descriptors_1 );
						extractor.compute( img2, keypoints_2, descriptors_2 );

						//-- Step 3: Matching descriptor vectors using FLANN matcher
						
						matcher.match( descriptors_1, descriptors_2, matches );
					}
					catch(Exception ex)
					{
						printf("exception when extracting keypoints...%s\n",ex.msg);
					}

						
					for( int i = 0; i < matches.size(); i++ )
					{
						key1 = matches[i].queryIdx;
						key2 = matches[i].trainIdx;
		
						if(DEBUG)
							printf("%d) compare matches x[%f,%f] y[%f,%f]\n\n",i,keypoints_1[key1].pt.x,keypoints_2[key2].pt.x,keypoints_1[key1].pt.y,keypoints_2[key2].pt.y);

						try
						{
							// calculate keypoint_distance
							x_dist = abs(keypoints_1[key1].pt.x-keypoints_2[key2].pt.x);
							y_dist = abs(keypoints_1[key1].pt.y-keypoints_2[key2].pt.y);
							if(DEBUG)
								printf("x-dist: %d | y-dist:%d | key1: %d | key2: %d\n",x_dist,y_dist,key1,key2);
							keypoint_distance = sqrt((double)((x_dist*x_dist) + (y_dist*y_dist)));
							if(DEBUG)
								printf("keypoint_distance: %d\n",keypoint_distance);

							// only consider a match a good match if it hasn't moved unrealistically far
							if(keypoint_distance < fuzziness_max)
							{
								good_matches.push_back( matches[i]);
							}
						}
						catch(Exception ex)
						{
							printf("exception when finding good matches...\n");
						}
					}
						
						
					// get a list of blobs (contours)
					cvCanny( ImaskCodeBookClosed, canny_output, thresh, thresh*2, 3 );

					/// Find contours
					if( mem_storage == NULL )
					{
						mem_storage = cvCreateMemStorage(0);
					}
					else
					{
						cvClearMemStorage( mem_storage );
					}
					cvFindContours(canny_output, mem_storage, &contours,88,CV_RETR_EXTERNAL);
						
					vector<RotatedRect> minEllipse( contours->total );

					if( contours )
					{
						int contour_counter = 0;
						for (; contours != 0; contours = contours->h_next)
						{
							for( int i = 0; i < good_matches.size(); i++ )
							{
								if(cvPointPolygonTest(contours,keypoints_2[good_matches[i].trainIdx].pt,0) > 0)
								{
									cvDrawContours(justForeground, contours, CV_RGB(255,0,0), CV_RGB(255,0,0), -1, CV_FILLED, 8, cvPoint(0,0));
									break;
								}
							}
						}
					}


					//-- Draw only "good" matches
					try
					{
						drawMatches( img1, keypoints_1, img2, keypoints_2,
								good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
								vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
						imshow( "Matches", img_matches );
					}
					catch(Exception ex)
					{
						printf("exception when drawing matches...\n");
					}
						
					// save current frame as previous frame for next round
					cvCopy(justForegroundGray,prevFrameGray);
				}
				else
				{
					// create the first previous frame
					cvCvtColor(resized,prevFrameGray,CV_BGR2GRAY);
				}
			}

			if(DRAW_CONTOURS)
			{
				cvCanny( ImaskCodeBookClosed, canny_output, thresh, thresh*2, 3 );

				/// Find contours
				if( mem_storage == NULL )
				{
					mem_storage = cvCreateMemStorage(0);
				}
				else
				{
					cvClearMemStorage( mem_storage );
				}
				//cvFindContours(canny_output, mem_storage, &contours);
				cvFindContours(canny_output, mem_storage, &contours,88,CV_RETR_EXTERNAL);
				/// Draw contours
				if( contours )
				{
					cvDrawContours(output,contours,cvScalarAll(255),cvScalarAll(255),1);
				}
			}
        }



        // display graphical results
        cvShowImage("Raw", rawImage );
		cvShowImage("Output",output);
		cvShowImage("Foreground CodeBook",ImaskCodeBook);
		cvShowImage("Foreground CodeBook Closed",ImaskCodeBookClosed);
		

		if(SAVE_OUTPUT)
		{
			// create output image
			Rect top(0, 0, rawImage->width, rawImage->height);
			Rect bottom(0, rawImage->height-1, rawImage->width, rawImage->height);
			writeFrame = cvCreateImage( cvSize(640,480), IPL_DEPTH_8U, 3 );
			cvZero(writeFrame);
			cvSetImageROI( writeFrame, top ); 
			cvCopy(rawImage, writeFrame);
			cvResetImageROI(writeFrame); 
			cvSetImageROI( writeFrame, bottom);
			//cvMerge(justForeground, justForeground, justForeground, NULL, writeFrame);
			cvCopy(output, writeFrame);
			cvWriteFrame(writer,writeFrame);
			if(nframes >= 790)
			{
				cvReleaseVideoWriter( &writer );
			}
		}


		if(nframes > nframesToLearnBG || LIVE_DEMO)
			cvWaitKey(1);
		else
			cvWaitKey(40);
    }

    
}