// BackgroundSubtraction.cpp : Defines the entry point for the console application.
//
#include <iostream>
#include "stdafx.h"
#include <opencv2\core\core.hpp>
#include <opencv/cvaux.h>
#include <opencv/cxmisc.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/legacy/legacy.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>

using namespace std;
using namespace cv;
using namespace cv::gpu;

//VARIABLES for CODEBOOK METHOD:
CvBGCodeBookModel* model = 0;
const int NCHANNELS = 3;
bool ch[NCHANNELS]={true,true,true}; // This sets what channels should be adjusted for background bounds


const bool DRAW_CONTOURS = false;
const bool TRACK_MOTION = true;
const bool USE_LUCAS_KANADE = false;
const int MAX_CORNERS = 500;
const bool USE_SIFT = false;
const bool BLOB_COMPARE = true;
const bool DEBUG = true;
const bool LIVE_DEMO = false;

// VARIABLES for FindContours
int thresh = 200;
int max_thresh = 255;
cv::RNG rng(12345);






const char* filename = 0;

bool pause = false;
bool singlestep = false;

IplImage ipl_image;
IplImage *rawImage = 0, *yuvImage = 0; //yuvImage is for codebook method
IplImage *justForeground = 0;
IplImage *justForegroundGray = 0;
IplImage *ImaskCodeBook = 0,*ImaskCodeBookCC = 0;
IplImage *ImaskCodeBookInv = 0;
IplImage *ImaskCodeBookClosed = 0;
IplImage *prevFrameMotionBlobs = 0;
IplImage *prevFrameRaw = 0;
IplImage *output = 0;
CvCapture *capture = 0;
cv::Mat_<cv::Vec3b> image;						

int c, n, nframes = 0;
int nframesToLearnBG = 150; // originally 300


// Find Contours
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

char buffer [33];

// fitting ellipses
CvMemStorage* stor;
CvBox2D32f* box;
CvPoint* PointArray;
CvPoint2D32f* PointArray2D32f;

// font
CvFont font;
	



// flakus blob motion
double compare_result;
int count1 = 0;
int count2 = 0;
double bestMatchesVal[100];
int bestMatchesId[100];
CvScalar prevFrameColors[100];
CvScalar curFrameColors[100];
int curFrameLabels[100];
int prevFrameLabels[100];
CvRect rect1;
CvRect rect2;
CvScalar color;

// flakus weights
float moment_weight = 1;
float size_weight = 2;
float proximity_weight = 10;
float min_blob_area = 26;



void processFrame();

void help()
{
    printf("\nLearn background and find foreground using simple average and average difference learning method:\n"
        "\nUSAGE:\nbgfg_codebook [--nframes=300] [movie filename, else from camera]\n"
        "***Keep the focus on the video windows, NOT the consol***\n\n"
        "INTERACTIVE PARAMETERS:\n"
        "\tESC,q,Q  - quit the program\n"
        "\th	- print this help\n"
        "\tp	- pause toggle\n"
        "\ts	- single step\n"
        "\tr	- run mode (single step off)\n"
        "=== AVG PARAMS ===\n"
        "\t-    - bump high threshold UP by 0.25\n"
        "\t=    - bump high threshold DOWN by 0.25\n"
        "\t[    - bump low threshold UP by 0.25\n"
        "\t]    - bump low threshold DOWN by 0.25\n"
        "=== CODEBOOK PARAMS ===\n"
        "\ty,u,v- only adjust channel 0(y) or 1(u) or 2(v) respectively\n"
        "\ta	- adjust all 3 channels at once\n"
        "\tb	- adjust both 2 and 3 at once\n"
        "\ti,o	- bump upper threshold up,down by 1\n"
        "\tk,l	- bump lower threshold up,down by 1\n"
        "\tSPACE - reset the model\n"
        );
}

int main(int argc, char** argv)
{

	printf("opening remote webcam...\n");

    

	VideoCapture camera;

	if(LIVE_DEMO)
	{
	
		camera.open("http://itwebcamcp700.fullerton.edu/mjpg/video.mjpg");
		//camera.open("http://cam1.brentwood-tn.org/mjpg/video.mjpg");
		//camera.open("http://216.8.159.21/mjpg/video.mjpg");
		//camera.open("http://cyclops.american.edu/mjpg/video.mjpg");

		//MAIN PROCESSING LOOP:
		for(;;)
		{
			if( !pause )
			{
				camera.grab();
				camera.retrieve(image);
			
				// Setup a rectangle to define region of interest
				cv::Rect roi(0, cvRound(image.rows/2)-1, image.cols, cvRound(image.rows/2));
				//cv::Rect roi(0, 0, image.cols-1, image.rows-1);

				// crop input image to just roi
				ipl_image = image(roi);

				/*
				// equalize histogram to broaden color differences
				if(ipl_image_eq == 0)
				{
					ipl_image_eq = cvCreateImage( cvGetSize(&ipl_image), IPL_DEPTH_8U, 3 );
					r = cvCreateImage(cvGetSize(&ipl_image), IPL_DEPTH_8U, 1);
					g = cvCreateImage(cvGetSize(&ipl_image), IPL_DEPTH_8U, 1);
					b = cvCreateImage(cvGetSize(&ipl_image), IPL_DEPTH_8U, 1);
				}
				cvSplit(&ipl_image, b, g, r, NULL);
				cvEqualizeHist( r, r );    //equalise r
				cvEqualizeHist( g, g );    //equalise g
				cvEqualizeHist( b, b );    //equalise b
				cvMerge(b, g, r, NULL, ipl_image_eq);
				*/
			

				rawImage = &ipl_image;

				++nframes;

				printf("frame: %d\n",nframes);

				if(!rawImage) 
					break;


				processFrame();
			}
			if( singlestep )
				pause = true;
        
			// User input:
			c = cvWaitKey(10)&0xFF;
			c = tolower(c);
			// End processing on ESC, q or Q
			if(c == 27 || c == 'q')
				break;
			//Else check for user input
			switch( c )
			{
			case 'h':
				help();
				break;
			case 'p':
				pause = !pause;
				break;
			case 's':
				singlestep = !singlestep;
				pause = false;
				break;
			case 'r':
				pause = false;
				singlestep = false;
				break;
			case ' ':
				cvBGCodeBookClearStale( model, 0 );
				nframes = 0;
				break;
				//CODEBOOK PARAMS
			case 'y': case '0':
			case 'u': case '1':
			case 'v': case '2':
			case 'a': case '3':
			case 'b': 
				ch[0] = c == 'y' || c == '0' || c == 'a' || c == '3';
				ch[1] = c == 'u' || c == '1' || c == 'a' || c == '3' || c == 'b';
				ch[2] = c == 'v' || c == '2' || c == 'a' || c == '3' || c == 'b';
				printf("CodeBook YUV Channels active: %d, %d, %d\n", ch[0], ch[1], ch[2] );
				break;
			case 'i': //modify max classification bounds (max bound goes higher)
			case 'o': //modify max classification bounds (max bound goes lower)
			case 'k': //modify min classification bounds (min bound goes lower)
			case 'l': //modify min classification bounds (min bound goes higher)
				{
				uchar* ptr = c == 'i' || c == 'o' ? model->modMax : model->modMin;
				for(n=0; n<NCHANNELS; n++)
				{
					if( ch[n] )
					{
						int v = ptr[n] + (c == 'i' || c == 'l' ? 1 : -1);
						ptr[n] = CV_CAST_8U(v);
					}
					printf("%d,", ptr[n]);
				}
				printf(" CodeBook %s Side\n", c == 'i' || c == 'o' ? "High" : "Low" );
				}
				break;
			}
		}
	}
	else
	{
		capture = cvCaptureFromAVI("output1.avi"); 

		for(;;)
		{
			if( !pause )
			{
				  
				image = cvQueryFrame( capture );

				if(image.rows == 0 || image.cols == 0)
					break;

				// Setup a rectangle to define region of interest
				cv::Rect roi(0, cvRound(image.rows/2)-1, image.cols, cvRound(image.rows/2));

				// crop input image to just roi
				ipl_image = image(roi);

				rawImage = &ipl_image;

				++nframes;

				printf("frame: %d\n",nframes);

				if(!rawImage) 
					break;

				processFrame();
			}
		}
	}

	


    		
    
    cvReleaseCapture( &capture );
    cvDestroyWindow( "Raw" );
    cvDestroyWindow( "ForegroundCodeBook");
	// TODO: destroy other windows
    return 0;
}


void processFrame()
{
	//First time:
    if( nframes == 1 && rawImage )
    {
        // CODEBOOK METHOD ALLOCATION
        yuvImage = cvCloneImage(rawImage);
		justForeground = cvCreateImage( cvGetSize(rawImage), IPL_DEPTH_8U, rawImage->nChannels );
		justForegroundGray = cvCreateImage( cvGetSize(rawImage), IPL_DEPTH_8U, 1 );
        ImaskCodeBook = cvCreateImage( cvGetSize(rawImage), IPL_DEPTH_8U, 1 );
		ImaskCodeBookInv = cvCreateImage( cvGetSize(rawImage), IPL_DEPTH_8U, 1 );
		ImaskCodeBookClosed = cvCreateImage( cvGetSize(rawImage), IPL_DEPTH_8U, 1 );
        ImaskCodeBookCC = cvCreateImage( cvGetSize(rawImage), IPL_DEPTH_8U, 1 );
        canny_output = cvCreateImage( cvGetSize(rawImage), IPL_DEPTH_8U, 1 );
		canny_output1 = cvCreateImage( cvGetSize(rawImage), IPL_DEPTH_8U, 1 );
		canny_output2 = cvCreateImage( cvGetSize(rawImage), IPL_DEPTH_8U, 1 );
		output = cvCreateImage(cvGetSize(rawImage),IPL_DEPTH_8U,3);
			
        cvSet(ImaskCodeBook,cvScalar(255));


		// initialize font
		cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX, 0.5f, 0.5f, 0, 2);

		// initialize codebook model
		model = cvCreateBGCodeBookModel();
		//Set color thresholds to default values
		// oringal min = 3, max = 10 --> now 2 and 14
		model->modMin[0] = 2;
		model->modMin[1] = model->modMin[2] = 2;
		model->modMax[0] = 14;
		model->modMax[1] = model->modMax[2] = 14;
		model->cbBounds[0] = model->cbBounds[1] = model->cbBounds[2] = 14;

		// initialize contour memory (probably not necessary)
		contours_remember1 = contours1;
		contours_remember2 = contours2;

		for(int f=0;f<100;f++)
		{
			prevFrameLabels[f] = f;
		}

        cvNamedWindow("Raw",1);
		cvNamedWindow("JustForeground",1);
        cvNamedWindow("ForegroundCodeBook",1);
		cvNamedWindow("CodeBookClosed",1);
    }
              
    if( rawImage )
    {
		// meanshift
		//cvPyrMeanShiftFiltering(rawImage,rawImage,10,10,1);

		// blur input image to reduce noise
		cvSmooth(rawImage,rawImage);
			

        cvCvtColor( rawImage, yuvImage, CV_BGR2YCrCb ); // convert to YUV

        // build background model
        if( !pause && nframes-1 < nframesToLearnBG  )
            cvBGCodeBookUpdate( model, yuvImage );

        if( nframes-1 == nframesToLearnBG  )
            cvBGCodeBookClearStale( model, model->t/2 );
            
        //Find the foreground if any
        if( nframes-1 >= nframesToLearnBG  )
        {
            // Find foreground by codebook method
            cvBGCodeBookDiff( model, yuvImage, ImaskCodeBook );

			if(nframes % 100 == 0)
			{
				printf("codebook stale entries cleared\n");
				cvBGCodeBookClearStale(model,model->t/2);
			}
			cvCopy(ImaskCodeBook,ImaskCodeBookClosed);
				
			cvDilate(ImaskCodeBookClosed,ImaskCodeBookClosed);
			cvDilate(ImaskCodeBookClosed,ImaskCodeBookClosed);
			cvErode(ImaskCodeBookClosed,ImaskCodeBookClosed);
			cvErode(ImaskCodeBookClosed,ImaskCodeBookClosed);
			cvErode(ImaskCodeBookClosed,ImaskCodeBookClosed);
			cvDilate(ImaskCodeBookClosed,ImaskCodeBookClosed);
				
			/*
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
			cvFindContours(canny_output, mem_storage, &contours);

			// fill in holes within each blob
			cvDrawContours(ImaskCodeBookClosed,contours,cvScalarAll(255),cvScalarAll(255),1,CV_FILLED, 8, cvPoint(0,0));
			*/


			// invert the mask
			for(int i=0;i<ImaskCodeBookClosed->height;i++)
				for(int j=0;j<ImaskCodeBookClosed->width;j++)
					for(int k=0;k<ImaskCodeBookClosed->nChannels;k++)  //loop to read for each channel
						ImaskCodeBookInv->imageData[i*ImaskCodeBookClosed->widthStep+j*ImaskCodeBookClosed->nChannels+k]=255-ImaskCodeBookClosed->imageData[i*ImaskCodeBookClosed->widthStep+j*ImaskCodeBookClosed->nChannels+k];    //inverting the image

            // This part just to visualize bounding boxes and centers if desired
            cvCopy(ImaskCodeBook,ImaskCodeBookCC);
			cvSet(justForeground, cvScalar(0,0,0));
			cvCopy(rawImage,justForeground,ImaskCodeBookClosed);
            //cvSegmentFGMask( ImaskCodeBookCC );


				
			// save image as jpg
			if(nframes == nframesToLearnBG+51)
				cvSaveImage("jfg1.jpg",justForeground);
			else if(nframes == nframesToLearnBG+52)
				cvSaveImage("jfg2.jpg",justForeground);
			else if(nframes == nframesToLearnBG+53)
				cvSaveImage("jfg3.jpg",justForeground);
			else if(nframes == nframesToLearnBG+54)
				cvSaveImage("jfg4.jpg",justForeground);
			else if(nframes == nframesToLearnBG+55)
				cvSaveImage("jfg5.jpg",justForeground);
			else if(nframes == nframesToLearnBG+56)
				cvSaveImage("jfg6.jpg",justForeground);
			else if(nframes == nframesToLearnBG+57)
				cvSaveImage("jfg7.jpg",justForeground);
			else if(nframes == nframesToLearnBG+58)
				cvSaveImage("jfg8.jpg",justForeground);
			else if(nframes == nframesToLearnBG+59)
				cvSaveImage("jfg9.jpg",justForeground);
			else if(nframes == nframesToLearnBG+60)
				cvSaveImage("jfg10.jpg",justForeground);

			if(nframes == nframesToLearnBG+51)
				cvSaveImage("blobs1.jpg",ImaskCodeBookClosed);
			else if(nframes == nframesToLearnBG+52)
				cvSaveImage("blobs2.jpg",ImaskCodeBookClosed);
			else if(nframes == nframesToLearnBG+53)
				cvSaveImage("blobs3.jpg",ImaskCodeBookClosed);
			else if(nframes == nframesToLearnBG+54)
				cvSaveImage("blobs4.jpg",ImaskCodeBookClosed);
			else if(nframes == nframesToLearnBG+55)
				cvSaveImage("blobs5.jpg",ImaskCodeBookClosed);
			else if(nframes == nframesToLearnBG+56)
				cvSaveImage("blobs6.jpg",ImaskCodeBookClosed);
			else if(nframes == nframesToLearnBG+57)
				cvSaveImage("blobs7.jpg",ImaskCodeBookClosed);
			else if(nframes == nframesToLearnBG+58)
				cvSaveImage("blobs8.jpg",ImaskCodeBookClosed);
			else if(nframes == nframesToLearnBG+59)
				cvSaveImage("blobs9.jpg",ImaskCodeBookClosed);
			else if(nframes == nframesToLearnBG+60)
				cvSaveImage("blobs10.jpg",ImaskCodeBookClosed);

			if(nframes == nframesToLearnBG+51)
				cvSaveImage("raw1.jpg",rawImage);
			else if(nframes == nframesToLearnBG+52)
				cvSaveImage("raw2.jpg",rawImage);
			else if(nframes == nframesToLearnBG+53)
				cvSaveImage("raw3.jpg",rawImage);
			else if(nframes == nframesToLearnBG+54)
				cvSaveImage("raw4.jpg",rawImage);
			else if(nframes == nframesToLearnBG+55)
				cvSaveImage("raw5.jpg",rawImage);
			else if(nframes == nframesToLearnBG+56)
				cvSaveImage("raw6.jpg",rawImage);
			else if(nframes == nframesToLearnBG+57)
				cvSaveImage("raw7.jpg",rawImage);
			else if(nframes == nframesToLearnBG+58)
				cvSaveImage("raw8.jpg",rawImage);
			else if(nframes == nframesToLearnBG+59)
				cvSaveImage("raw9.jpg",rawImage);
			else if(nframes == nframesToLearnBG+60)
				cvSaveImage("raw10.jpg",rawImage);
				

				


			/************************************/
			/* Track Motion						*/
			/************************************/
			if(TRACK_MOTION && USE_LUCAS_KANADE)
			{
				if(prevFrameMotionBlobs)
				{
					cvCvtColor(justForeground,justForegroundGray,CV_BGR2GRAY);
						
					/*
					IplImage pf_gray = prev_frame_gray;
					IplImage* prevFrameMotionBlobs = &pf_gray;
					IplImage nf_gray = frame_gray;
					IplImage* justForegroundGray = &nf_gray;
					IplImage nf = roi_image;
					IplImage* imgC = &nf;
					*/

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
						cvLine( justForeground, p0, p1, CV_RGB(255,0,0),2 );
					}

					// save current frame as previous frame for next round
					cvCopy(justForegroundGray,prevFrameMotionBlobs);
				}
				else
				{
					// create the first previous frame
					prevFrameMotionBlobs = cvCreateImage( cvGetSize(rawImage), IPL_DEPTH_8U, 1 );
					cvCvtColor(justForeground,prevFrameMotionBlobs,CV_BGR2GRAY);
				}
			}


			if(TRACK_MOTION && BLOB_COMPARE)
			{
				if(prevFrameMotionBlobs)
				{
					printf("\n\nNEW FRAME\n");

					cvCopy(rawImage,output);

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


					//cvShowImage("canny 1",canny_output1);
					//cvShowImage("canny 2",canny_output2);
					//cvWaitKey(0);

					contours_remember1 = contours1;
					contours_remember2 = contours2;

					fill_n(bestMatchesVal, 100, 999999999);
					fill_n(bestMatchesId, 100, -1);
					fill_n(curFrameLabels, 100, -1);
					
			
					//printf("contour1 size: %d\n",contours1->first->count);

					for (; contours1 != 0; contours1 = contours1->h_next)
					{
							

						rect1 = cvBoundingRect( contours1);

						if(rect1.width * rect1.height < min_blob_area)
						{
							printf("ignoring small blob from prev frame...%d\n",count1);
							continue;
						}

						cvPutText(prevFrameRaw,itoa(count1,buffer,10),Point(rect1.x,rect1.y), &font, CV_RGB(255, 255, 255));
						cvDrawContours(prevFrameRaw,contours1,cvScalarAll(255),cvScalarAll(255),-1,CV_FILLED);
						cvShowImage("prev raw",prevFrameRaw);
				
						cvCopy(rawImage,output);

						for(; contours2 != 0; contours2 = contours2->h_next)
						{
							rect2 = cvBoundingRect( contours2);
							if(rect2.width * rect2.height < min_blob_area)
							{
								printf("throwing out small blob from current frame....%d\n",count2);
								continue;
							}

								
							cvPutText(output,itoa(count2,buffer,10),Point(rect2.x,rect2.y), &font, CV_RGB(255, 255, 255));
							cvDrawContours(output,contours2,cvScalarAll(255),cvScalarAll(255),-1,CV_FILLED);
							cvShowImage("current raw",output);
							
					
							// log((moment_weight)moment difference) + log((size_weight)size difference) + log((location_weight)location difference)
							compare_result =	log(moment_weight * (1+cvMatchShapes(contours1,contours2,CV_CONTOURS_MATCH_I2))) +
												log((size_weight * (1+abs(rect1.width*rect1.height - rect2.width*rect2.height)))) +
												log((proximity_weight*(1+(abs(rect1.x-rect2.x)*abs(rect1.y-rect2.y)))));

							if(compare_result < bestMatchesVal[count2])
							{
								bestMatchesVal[count2] = compare_result;
								bestMatchesId[count2] = count1;
								curFrameColors[count2] = prevFrameColors[count1];
								curFrameLabels[count2] = prevFrameLabels[count1];
							}

							printf("[%d][%d] compare result: %f\n",count1,count2,compare_result);
							printf("area diff: %d | %d | %d\n",rect1.width*rect1.height,rect2.width*rect2.height,abs(rect1.width*rect1.height - rect2.width*rect2.height));
							printf("location diff: %d | %d | %d\n",abs(rect1.x-rect2.x),abs(rect1.y-rect2.y),abs(rect1.x-rect2.x)*abs(rect1.y-rect2.y));
							cvWaitKey(0);
							count2++;
						}
						count2 = 0;
						contours2 = contours_remember2;
						count1++;
					}
					count1 = 0;
					contours1 = contours_remember1;			
			
					for(int m=0;m<100;m++)
					{
						if(bestMatchesId[m] != -1)
						{
							printf("best match for curr: %d is prev: %d with score of %f\n",m,bestMatchesId[m],bestMatchesVal[m]);
							cvWaitKey(0);
						}
					}

					// print contours with best match colors
					count2=0;
					for(; contours2 != 0; contours2 = contours2->h_next)
					{
						rect2 = cvBoundingRect( contours2);
						if(rect2.width * rect2.height < min_blob_area)
						{
							printf("throwing out small blob....%d\n",count2);
							continue;
						}

						if(curFrameLabels[count2] != -1)
						{
							printf("matching label from previous frame %d\n",curFrameLabels[count2]);
							cvPutText(output,itoa(curFrameLabels[count2],buffer,10),Point(rect2.x,rect2.y), &font, CV_RGB(0, 255, 0));
							curFrameColors[count2] = CV_RGB( rand()&255, rand()&255, rand()&255 );
						}

						
						cvDrawContours(output,contours2,curFrameColors[count2],curFrameColors[count2],-1,CV_FILLED);
						cvShowImage("current raw",output);
						cvWaitKey(0);

						count2++;
					}
					count2 = 0;	
						
					


					//cvDrawContours(r2,contours2,cvScalarAll(255),cvScalarAll(255),1,CV_FILLED);
					//cvShowImage("current raw",output);
					//cvDrawContours(r1,contours1,cvScalarAll(255),cvScalarAll(255),1,CV_FILLED);
					//cvShowImage("prev raw",prevFrameRaw);

					//cvWaitKey(0);

					// save current frame as previous frame for next round
					cvCopy(ImaskCodeBookClosed,prevFrameMotionBlobs);
					cvCopy(rawImage,prevFrameRaw);
						
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
					prevFrameMotionBlobs = cvCreateImage( cvGetSize(rawImage), IPL_DEPTH_8U, 1 );
					prevFrameRaw = cvCreateImage( cvGetSize(rawImage), IPL_DEPTH_8U, 3 );
					cvCopy(ImaskCodeBookClosed,prevFrameMotionBlobs);
					cvCopy(rawImage,prevFrameRaw);
				}
			}








			if(TRACK_MOTION && USE_SIFT)
			{
				if(prevFrameMotionBlobs)
				{
					keypoints_1.clear();
					keypoints_2.clear();
					matches.clear();
					good_matches.clear();

					cvCvtColor(justForeground,justForegroundGray,CV_BGR2GRAY);

					try
					{
						img1 = Mat(prevFrameMotionBlobs);
						img2 = Mat(justForegroundGray);
	
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

					/*
					for( int i = 0; i < keypoints_1.size(); i++ )
					{
						printf("keypoint1 %d (%f,%f)\n",i,keypoints_1[i].pt.x,keypoints_1[i].pt.y);
					}
					for( int i = 0; i < keypoints_2.size(); i++ )
					{
						printf("keypoint2 %d (%f,%f)\n",i,keypoints_2[i].pt.x,keypoints_2[i].pt.y);
					}
					*/

					//-- Draw only "good" matches (i.e. whose keypoint_distance is less than 2*min_dist )
					//-- PS.- radiusMatch can also be used here.

						
					for( int i = 0; i < matches.size(); i++ )
					{
						key1 = matches[i].queryIdx;
						key2 = matches[i].trainIdx;
		
						//printf("%d) compare matches x[%f,%f] y[%f,%f]\n\n",i,keypoints_1[key1].pt.x,keypoints_2[key2].pt.x,keypoints_1[key1].pt.y,keypoints_2[key2].pt.y);
						try
						{
							/*
							if(abs(keypoints_1[key1].pt.x-keypoints_2[key2].pt.x) < fuzziness_max)
							{
								printf("good x match x[%f,%f] y[%f,%f]\n\n",keypoints_1[key1].pt.x,keypoints_2[key2].pt.x,keypoints_1[key1].pt.y,keypoints_2[key2].pt.y);
							}

							if(abs(keypoints_1[key1].pt.y-keypoints_2[key2].pt.y) < fuzziness_max)
							{
								printf("good y match x[%f,%f] y[%f,%f]\n\n",keypoints_1[key1].pt.x,keypoints_2[key2].pt.x,keypoints_1[key1].pt.y,keypoints_2[key2].pt.y);
							}
							*/

							// calculate keypoint_distance
							x_dist = abs(keypoints_1[key1].pt.x-keypoints_2[key2].pt.x);
							y_dist = abs(keypoints_1[key1].pt.y-keypoints_2[key2].pt.y);
							printf("x-dist: %d | y-dist:%d | key1: %d | key2: %d\n",x_dist,y_dist,key1,key2);
							keypoint_distance = sqrt((double)((x_dist*x_dist) + (y_dist*y_dist)));
							printf("keypoint_distance: %d\n",keypoint_distance);
							if(keypoint_distance >= fuzziness_min && keypoint_distance < fuzziness_max)
							{
								printf("");
								good_matches.push_back( matches[i]);
								printf("");
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
					cvFindContours(canny_output, mem_storage, &contours);

						
					//cvDrawContours(justForeground,contours,cvScalarAll(255),cvScalarAll(255),1);
					vector<RotatedRect> minEllipse( contours->total );

					if( contours )
					{
						int contour_counter = 0;
						for (; contours != 0; contours = contours->h_next)
						{
								
							/*
							// try to fit ellipses to blobs to fill them in
							int i; // Indicator of cycle.
							int count = contours->total; // This is number point in contour
							CvPoint center;
							CvSize size;
        
							// Number point must be more than or equal to 6 (for cvFitEllipse_32f).        
							if( count < 6 )
								continue;
        
							// Alloc memory for contour point set.    
							PointArray = (CvPoint*)malloc( count*sizeof(CvPoint) );
							PointArray2D32f= (CvPoint2D32f*)malloc( count*sizeof(CvPoint2D32f) );
        
							// Alloc memory for ellipse data.
							box = (CvBox2D32f*)malloc(sizeof(CvBox2D32f));
        
							// Get contour point set.
							cvCvtSeqToArray(contours, PointArray, CV_WHOLE_SEQ);
        
							// Convert CvPoint set to CvBox2D32f set.
							for(i=0; i<count; i++)
							{
								PointArray2D32f[i].x = (float)PointArray[i].x;
								PointArray2D32f[i].y = (float)PointArray[i].y;
							}
        
							// Fits ellipse to current contour.
							cvFitEllipse(PointArray2D32f, count, box);
							//cvFitEllipse2();
        
							// Draw current contour.
							//cvDrawContours(rawImage,cont,CV_RGB(255,255,255),CV_RGB(255,255,255),0,1,8,cvPoint(0,0));
        
							// Convert ellipse data from float to integer representation.
							center.x = cvRound(box->center.x);
							center.y = cvRound(box->center.y);
							size.width = cvRound(box->size.width*0.5);
							size.height = cvRound(box->size.height*0.5);
							box->angle = -box->angle;
        
							// Draw ellipse.
							cvEllipse(ImaskCodeBookClosed, center, size,
										box->angle, 0, 360,
										CV_RGB(255,255,255), CV_FILLED, CV_AA, 0);

							*/





							for( int i = 0; i < good_matches.size(); i++ )
							{
								if(cvPointPolygonTest(contours,keypoints_2[good_matches[i].trainIdx].pt,0) > 0)
								{
									cvDrawContours(justForeground, contours, CV_RGB(255,0,0), CV_RGB(255,0,0), -1, CV_FILLED, 8, cvPoint(0,0));
									break;
								}
								else
								{
									//cvDrawContours(justForeground, contours, CV_RGB(0,255,0), CV_RGB(0,255,0), -1, CV_FILLED, 8, cvPoint(0,0));
								}
							}
						}
						/*
							
							for (; contours != 0; contours = contours->h_next)
							{
								// check to see if keypoint is in this contour
								printf("%d) poly test: %f\n",i,cvPointPolygonTest(contours,keypoints_2[good_matches[i].trainIdx].pt,0));
									
								if(cvPointPolygonTest(contours,keypoints_2[good_matches[i].trainIdx].pt,0) > 0)
								{
									cvDrawContours(justForeground, contours, CV_RGB(0,255,0), CV_RGB(255,0,0), -1, CV_FILLED, 8, cvPoint(0,0));
								}
								else
								{
									//CvScalar ext_color = CV_RGB( rand()&255, rand()&255, rand()&255 ); //randomly color different contours
									cvDrawContours(justForeground, contours, CV_RGB(255,0,0), CV_RGB(255,0,0), -1, CV_FILLED, 8, cvPoint(0,0));
								}
								break;
							}
						}
						*/
					}
						
					//cvDrawContours(justForeground, contours, CV_RGB(255,0,0), CV_RGB(255,0,0), -1, CV_FILLED, 8, cvPoint(0,0));
						
					// for each good match, identify the blob the match pertains to

						


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
						

					/*
					for( int i = 0; i < matches.size(); i++ )
					{ 
						printf( "-- Good Match [%d] Keypoint 1: %d  -- Keypoint 2: %d  \n", i, matches[i].queryIdx, matches[i].trainIdx ); 
					}
					*/
						
					// save current frame as previous frame for next round
					cvCopy(justForegroundGray,prevFrameMotionBlobs);
				}
				else
				{
					// create the first previous frame
					prevFrameMotionBlobs = cvCreateImage( cvGetSize(rawImage), IPL_DEPTH_8U, 1 );
					cvCvtColor(justForeground,prevFrameMotionBlobs,CV_BGR2GRAY);
				}
			}

			/// Detect edges using canny
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
				cvFindContours(canny_output, mem_storage, &contours);

				/// Draw contours
				if( contours )
				{
					cvDrawContours(justForeground,contours,cvScalarAll(255),cvScalarAll(255),1);
				}
			}
        }
        //Display
        cvShowImage( "Raw", rawImage );
        cvShowImage( "JustForeground", justForeground );
        cvShowImage( "ForegroundCodeBook",ImaskCodeBook);
		cvShowImage("CodeBookClosed",ImaskCodeBookClosed);
    }

    
}