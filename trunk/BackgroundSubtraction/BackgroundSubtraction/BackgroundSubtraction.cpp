// BackgroundSubtraction.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

#include <opencv/cvaux.h>
#include <opencv/cxmisc.h>
#include <opencv/highgui.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>


//VARIABLES for CODEBOOK METHOD:
CvBGCodeBookModel* model = 0;
const int NCHANNELS = 3;
bool ch[NCHANNELS]={true,true,true}; // This sets what channels should be adjusted for background bounds


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
    IplImage* rawImage = 0, *yuvImage = 0; //yuvImage is for codebook method
	IplImage* justForeground = 0;
    IplImage *ImaskCodeBook = 0,*ImaskCodeBookCC = 0;
	IplImage *ImaskCodeBookInv = 0;
	IplImage *ImaskCodeBookClosed = 0;
	CvCapture* capture = 0;

    int c, n, nframes = 0;
    int nframesToLearnBG = 100; // originally 300


	// Find Contours
	IplImage* canny_output = 0;
	CvMemStorage* 	mem_storage = NULL;
	CvSeq* contours = 0;


    model = cvCreateBGCodeBookModel();
    
    //Set color thresholds to default values
    model->modMin[0] = 3;
    model->modMin[1] = model->modMin[2] = 3;
    model->modMax[0] = 10;
    model->modMax[1] = model->modMax[2] = 10;
    model->cbBounds[0] = model->cbBounds[1] = model->cbBounds[2] = 10;

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

            cvCvtColor( rawImage, yuvImage, CV_BGR2YCrCb ); // converty to YUV

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
				cvCopy(rawImage,justForeground,ImaskCodeBook);
                //cvSegmentFGMask( ImaskCodeBookCC );


				

				/// Detect edges using canny
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

