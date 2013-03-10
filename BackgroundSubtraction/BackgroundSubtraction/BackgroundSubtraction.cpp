// BackgroundSubtraction.cpp : Defines the entry point for the console application.
//
#include <iostream>
#include "stdafx.h"
#include <opencv2\core\core.hpp>
#include <opencv/cvaux.h>
#include <opencv/cxmisc.h>
#include <opencv/highgui.h>
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
const bool USE_SIFT = true;
const bool USE_ORB = true;
const int MAX_CORNERS = 500;
const bool DEBUG = true;

// VARIABLES for FindContours
int thresh = 200;
int max_thresh = 255;
cv::RNG rng(12345);


void help(void)
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

	printf("main started...\n");



	const char* filename = 0;
    IplImage *rawImage = 0, *yuvImage = 0; //yuvImage is for codebook method
	IplImage *justForeground = 0;
	IplImage *justForegroundGray = 0;
    IplImage *ImaskCodeBook = 0,*ImaskCodeBookCC = 0;
	IplImage *ImaskCodeBookInv = 0;
	IplImage *ImaskCodeBookClosed = 0;
	IplImage *prevFrameMotionBlobs = 0;
	CvCapture *capture = 0;
						

    int c, n, nframes = 0;
    int nframesToLearnBG = 200; // originally 300


	// Find Contours
	IplImage* canny_output = 0;
	CvMemStorage* 	mem_storage = NULL;
	CvSeq* contours = 0;


	// SIFT Vars
	Mat img1;
	Mat img2;
	int minHessian = 400;
	SiftFeatureDetector detector( minHessian );
	std::vector<KeyPoint> keypoints_1, keypoints_2;
	SiftDescriptorExtractor extractor;
	Mat descriptors_1, descriptors_2;
	BruteForceMatcher< L2<float> > matcher;
	std::vector< DMatch > matches;
	std::vector< DMatch > good_matches;
	int fuzziness_min = 3;
	int fuzziness_max = 30;
	Mat img_matches;


    model = cvCreateBGCodeBookModel();
    
    //Set color thresholds to default values
	// oringal min = 3, max = 10 --> now 2 and 14
    model->modMin[0] = 2;
    model->modMin[1] = model->modMin[2] = 2;
    model->modMax[0] = 14;
    model->modMax[1] = model->modMax[2] = 14;
    model->cbBounds[0] = model->cbBounds[1] = model->cbBounds[2] = 14;

    bool pause = false;
    bool singlestep = false;

    for( n = 1; n < argc; n++ )
    {
        static const char* nframesOpt = "--nframes=";
        if( strncmp(argv[n], nframesOpt, strlen(nframesOpt))==0 )
        {
            if( sscanf(argv[n] + strlen(nframesOpt), "%d", &nframesToLearnBG) == 0 )
            {
                help();
                return -1;
            }
        }
        else
            filename = argv[n];
    }

	cv::VideoCapture camera;
	camera.open("http://itwebcamcp700.fullerton.edu/mjpg/video.mjpg");
	//camera.open("http://cam1.brentwood-tn.org/mjpg/video.mjpg");
	
	/*
    if( !filename )
    {
        printf("Capture from camera\n");
        capture = cvCaptureFromCAM( 0 );
    }
    else
    {
        printf("Capture from file %s\n",filename);
        capture = cvCreateFileCapture( filename );
    }

	
    if( !capture )
    {
        printf( "Can not initialize video capturing\n\n" );
        help();
        return -1;
    }
	*/

	IplImage ipl_image;
	//IplImage *ipl_image_eq = 0;
	//IplImage* r = 0;
	//IplImage* g = 0;
	//IplImage* b = 0;

    //MAIN PROCESSING LOOP:
    for(;;)
    {
        if( !pause )
        {
            //rawImage = cvQueryFrame( capture );
			cv::Mat_<cv::Vec3b> image;
			camera.grab();
	        camera.retrieve(image);
			
			// Setup a rectangle to define region of interest
			cv::Rect roi(0, cvRound(image.rows/2)-1, image.cols, cvRound(image.rows/2));
			
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
            if(!rawImage) 
                break;
        }
        if( singlestep )
            pause = true;
        
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
			
            cvSet(ImaskCodeBook,cvScalar(255));
            
            cvNamedWindow("Raw",1);
			cvNamedWindow("JustForeground",1);
            cvNamedWindow("ForegroundCodeBook",1);
            cvNamedWindow("CodeBook_ConnectComp",1);
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
					cvSaveImage("image1b.jpg",justForeground);
				else if(nframes == nframesToLearnBG+52)
					cvSaveImage("image2b.jpg",justForeground);
				else if(nframes == nframesToLearnBG+53)
					cvSaveImage("image3b.jpg",justForeground);
				else if(nframes == nframesToLearnBG+54)
					cvSaveImage("image4b.jpg",justForeground);
				else if(nframes == nframesToLearnBG+55)
					cvSaveImage("image5b.jpg",justForeground);
				


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
							printf("exception when extracting keypoints...\n");
						}


						for( int i = 0; i < keypoints_1.size(); i++ )
						{
							printf("keypoint1 %d (%f,%f)\n",i,keypoints_1[i].pt.x,keypoints_1[i].pt.y);
						}
						for( int i = 0; i < keypoints_2.size(); i++ )
						{
							printf("keypoint2 %d (%f,%f)\n",i,keypoints_2[i].pt.x,keypoints_2[i].pt.y);
						}


						//-- Draw only "good" matches (i.e. whose distance is less than 2*min_dist )
						//-- PS.- radiusMatch can also be used here.

						
						for( int i = 0; i < matches.size(); i++ )
						{
							int key1 = matches[i].queryIdx;
							int key2 = matches[i].trainIdx;
		
							printf("%d) compare matches x[%f,%f] y[%f,%f]\n\n",i,keypoints_1[key1].pt.x,keypoints_2[key2].pt.x,keypoints_1[key1].pt.y,keypoints_2[key2].pt.y);
							try
							{
								if(abs(keypoints_1[key1].pt.x-keypoints_2[key2].pt.x) < fuzziness_max)
								{
									printf("good x match x[%f,%f] y[%f,%f]\n\n",keypoints_1[key1].pt.x,keypoints_2[key2].pt.x,keypoints_1[key1].pt.y,keypoints_2[key2].pt.y);
								}

								if(abs(keypoints_1[key1].pt.y-keypoints_2[key2].pt.y) < fuzziness_max)
								{
									printf("good y match x[%f,%f] y[%f,%f]\n\n",keypoints_1[key1].pt.x,keypoints_2[key2].pt.x,keypoints_1[key1].pt.y,keypoints_2[key2].pt.y);
								}

								if(abs(keypoints_1[key1].pt.x-keypoints_2[key2].pt.x) < fuzziness_max
									&& abs(keypoints_1[key1].pt.y-keypoints_2[key2].pt.y) < fuzziness_max
									&& abs(keypoints_1[key1].pt.y-keypoints_2[key2].pt.y) > fuzziness_min
									&& abs(keypoints_1[key1].pt.y-keypoints_2[key2].pt.y) > fuzziness_min)
								{
									good_matches.push_back( matches[i]);
								}
							}
							catch(Exception ex)
							{
								printf("exception when finding good matches...\n");
							}
						}

						//-- Draw only "good" matches
						try
						{
							drawMatches( img1, keypoints_1, img2, keypoints_2,
									good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
									vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
						}
						catch(Exception ex)
						{
							printf("exception when drawing matches...\n");
						}
						//-- Show detected matches
						imshow( "Good Matches", img_matches );

						for( int i = 0; i < matches.size(); i++ )
						{ 
							printf( "-- Good Match [%d] Keypoint 1: %d  -- Keypoint 2: %d  \n", i, matches[i].queryIdx, matches[i].trainIdx ); 
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
            //cvShowImage( "CodeBook_ConnectComp",ImaskCodeBookCC);
        }

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
    
    cvReleaseCapture( &capture );
    cvDestroyWindow( "Raw" );
    cvDestroyWindow( "ForegroundCodeBook");
    cvDestroyWindow( "CodeBook_ConnectComp");
    return 0;
}

