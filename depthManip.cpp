#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <vector>
using namespace std;
using namespace cv;

Mat erosion_dst, dilation_dst;

int erosion_elem = 0;
int erosion_size = 0;
int dilation_elem = 0;
int dilation_size = 0;
int const max_elem = 2;
int const max_kernel_size = 21;

Mat Erosion( Mat src )
{
  int erosion_type;
  if( erosion_elem == 0 ){ erosion_type = MORPH_RECT; }
  else if( erosion_elem == 1 ){ erosion_type = MORPH_CROSS; }
  else if( erosion_elem == 2) { erosion_type = MORPH_ELLIPSE; }

  Mat element = getStructuringElement( erosion_type,
                                       Size( 2*erosion_size + 1, 2*erosion_size+1 ),
                                       Point( erosion_size, erosion_size ) );

  /// Apply the erosion operation
  erode( src, erosion_dst, element );
  return src;
}

/** @function Dilation */

Mat Dilation( Mat src )
{
  int dilation_type;
  if( dilation_elem == 0 ){ dilation_type = MORPH_RECT; }
  else if( dilation_elem == 1 ){ dilation_type = MORPH_CROSS; }
  else if( dilation_elem == 2) { dilation_type = MORPH_ELLIPSE; }

  Mat element = getStructuringElement( dilation_type,
                                       Size( 2*dilation_size + 1, 2*dilation_size+1 ),
                                       Point( dilation_size, dilation_size ) );
  /// Apply the dilation operation
  dilate( src, dilation_dst, element );
  return src;
}


int main(){
	
	namedWindow( "OriginalFrame", CV_WINDOW_AUTOSIZE );
	namedWindow( "Original Hist", CV_WINDOW_AUTOSIZE );
	namedWindow( "Normalized Hist", CV_WINDOW_AUTOSIZE );
	namedWindow( "BG Sub Frame", CV_WINDOW_AUTOSIZE );
	namedWindow( "Temp Frame", CV_WINDOW_AUTOSIZE );
	CvCapture* capture = cvCreateFileCapture( "10.avi" );
	//CvCapture* capture = cvCreateCameraCapture(0);
	
	CvSize sizeFrame = cvSize((int)cvGetCaptureProperty( capture, CV_CAP_PROP_FRAME_WIDTH),(int)cvGetCaptureProperty( capture, CV_CAP_PROP_FRAME_HEIGHT));
	//cout<<sizeFrame.height<<"\t"<<sizeFrame.width<<endl;
	//system("pause");
	int DELAY_CAPTION = 1500;
	int DELAY_BLUR = 100;
	int MAX_KERNEL_LENGTH = 31;
	IplImage* frame;
	IplImage* frameCopy;
	IplImage* prevImg;
	IplImage* diffImg;
	IplImage *frameTemp;
	

	vector< vector<int> >histPoints(30000,vector<int>(2,0));
	int x,y;
	int k=0;
	bool f1=false,f2=false;
	int count=0;
	int m1=400;int m2=15;
	int pixelCount=0;
	int sumX=0,sumY=0;

	/* dataset creation */

	ofstream dataOut("test10.txt");
	int frameCount=1;


	
	while(1) {
	
		
		/* motion profile - background subtraction */

		frame = cvQueryFrame(capture);
		if(!frame) break;
		dataOut<<"10"<<"\t";
		frameCopy = cvCreateImage( cvGetSize(frame),8,1);
		diffImg = cvCreateImage(cvGetSize(frame),8,1);
		cvCvtColor(frame,frameCopy,CV_BGR2GRAY);
		frameTemp=cvCloneImage(frameCopy);
	
		
		/*BG Sub*/
		
		frame = cvQueryFrame(capture);
		if(!frame) break;

		cvCvtColor(frame,frameCopy,CV_BGR2GRAY);
		cvShowImage("OriginalFrame",frameCopy);
		cvAbsDiff(frameCopy,frameTemp,diffImg);
		
		Mat diffImgMat(diffImg);
		

		for(int i=0;i<diffImg->width;i++){
			for(int j=0;j<diffImg->height;j++){
				CvScalar pixelVal = cvGet2D(diffImg,j,i);
				if(pixelVal.val[0] > 50){
					//diffImgMat.at<uchar>(j,i)=255;
					//pixelVal.val[0]= 220; 
					pixelCount++;
					sumX+=i;
					sumY+=j;
				}
			}
		}

		vector <int>binCount(10,0);
		int l=0;int m=0;
		for(int i=0;i<5;i++){
			for(int j=0;j<5;j++){
				if((l+64)<320 && (m+48)<240){ 
					Rect roi(l,m,64,48);
					Mat imgROI = diffImgMat(roi);
					for(int x=0;x<imgROI.rows;x++){
						for(int y=0;y<imgROI.cols;y++){
							if(imgROI.at<uchar>(x,y)<=5)
								binCount[0]++;
							else if(imgROI.at<uchar>(x,y)>5 && imgROI.at<uchar>(x,y)<=10)
								binCount[1]++;
							else if(imgROI.at<uchar>(x,y)>10 && imgROI.at<uchar>(x,y)<=15)
								binCount[2]++;
							else if(imgROI.at<uchar>(x,y)>15 && imgROI.at<uchar>(x,y)<=20)
								binCount[3]++;
							else if(imgROI.at<uchar>(x,y)>40 && imgROI.at<uchar>(x,y)<=50)
								binCount[4]++;
							else if(imgROI.at<uchar>(x,y)>50 && imgROI.at<uchar>(x,y)<=70)
								binCount[5]++;
							else if(imgROI.at<uchar>(x,y)>70 && imgROI.at<uchar>(x,y)<=90)
								binCount[6]++;
							else if(imgROI.at<uchar>(x,y)>90 && imgROI.at<uchar>(x,y)<=110)
								binCount[7]++;
							else if(imgROI.at<uchar>(x,y)>110 && imgROI.at<uchar>(x,y)<=155)
								binCount[8]++;
							else if(imgROI.at<uchar>(x,y)>155 && imgROI.at<uchar>(x,y)<=255)
								binCount[9]++;
						}
					}
				}
				l+=64;
				int binCountSum=0;
				for(int i=0;i<10;i++){
					binCountSum+=binCount[i];
				}
				binCountSum/=(24*32);

				for(int i=0;i<10;i++){
					if(binCountSum!=0)
						binCount[i]=binCount[i]/binCountSum;
						dataOut<<binCount[i]<<"\t";
						cout<<binCount[i]<<"\t";
				}
				cout<<endl;
				//m+=48;
			}
			m+=48;
			l=0;
		}

		
		double meanX=(double)sumX/pixelCount;
		double meanY=(double)sumY/pixelCount;

		//cout<<pixelCount<<"\t"<<meanX<<"\t"<<meanY<<"\n";


		////if(meanY<diffImg->width && meanX<diffImg->height){
		cvCircle(diffImg, cvPoint(meanX,meanY),10,cvScalar(255,0,0,0), 2,8,0);
		//}


		cvShowImage("BG Sub Frame",diffImg);
		imshow("Temp Frame",diffImgMat);

			
		
		/* depth profile - histogram */
		
		
		Mat imgMat(frame);
		//imshow( "Example1",imgMat );
		
		vector<Mat> bgr_planes;
		split( imgMat, bgr_planes );

		/// Establish the number of bins
		int histSize = 256;

		/// Set the ranges ( for B,G,R) )
		float range[] = { 0,256 } ;
		const float* histRange = { range };

		bool uniform = true; bool accumulate = false;

		Mat b_hist, g_hist, r_hist;

		/// Compute the histograms:
		calcHist( &bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate );
		//calcHist( &bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate );
		//calcHist( &bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate );

		// Draw the histograms for B, G and R
		int hist_w = 512; int hist_h = 400;
		int bin_w = cvRound( (double) hist_w/histSize );

		Mat histImage( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );

		/// Normalize the result to [ 0, histImage.rows ]
		normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
		//normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
		//normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );

		/// Draw for each channel
		vector<int>depthHistPoints(256,0);
		for(int i=0;i<256;i++){
			depthHistPoints[i] = b_hist.at<float>(i);
			dataOut<<depthHistPoints[i]<<"\t";
			cout<<depthHistPoints[i]<<"\t";
			i++;
		}


		for( int i = 1; i < 255; i++ ){
		
			if(i<=200){
					
				line( histImage, Point( bin_w*(i-1), hist_h - cvRound(b_hist.at<float>(i-1)) ) ,
								Point( bin_w*(i), hist_h - cvRound(b_hist.at<float>(i)) ),
								Scalar( 255, 0, 0), 2, 8, 0  );
				//line( histImage, Point( bin_w*(i-1), hist_h - cvRound(g_hist.at<float>(i-1)) ) ,
								//Point( bin_w*(i), hist_h - cvRound(g_hist.at<float>(i)) ),
								//Scalar( 0, 255, 0), 2, 8, 0  );
				//line( histImage, Point( bin_w*(i-1), hist_h - cvRound(r_hist.at<float>(i-1)) ) ,
								//Point( bin_w*(i), hist_h - cvRound(r_hist.at<float>(i)) ),
								//Scalar( 0, 0, 255), 2, 8, 0  );
				
				int x1 = Point(bin_w*(i-1), hist_h - cvRound(b_hist.at<float>(i-1))).x;
				int y1 = Point( bin_w*(i-1), hist_h - cvRound(b_hist.at<float>(i-1)) ).y;
				


				histPoints[k][0] = x1;
				histPoints[k++][1] = y1;
			
				//cout<<x1<<"\t"<<y1<<endl;
			}
		}

		
		imshow("Original Hist",histImage );
		
		/* gaussian blur on the histogram */
		Mat dest;		
		for ( int i = 1; i < MAX_KERNEL_LENGTH; i = i + 2 ){ 
			GaussianBlur( histImage, dest, Size( i, i ), 0, 0 );
		}

		imshow( "Normalized Hist",dest);

		/*eroding and dilating image */

		Mat manipHist;
		manipHist = Dilation(dest);
		manipHist = Erosion(manipHist);

		imshow("Normalized Hist", manipHist);
		
		dataOut<<endl;
		cout<<endl;
		frameCount++;

		char c = cvWaitKey(33);
		if( c == 33 ) break;
		
	}

	
	ofstream fout("histPoints.txt");
	int min=100;
	int max=400;
	for(int i=0;i<=k-1;i++){
		fout<<histPoints[i][0]<<"\t"<<histPoints[i][1]<<endl;
	}
	bool maxFlag=false;bool minFlag=false;
	int threshold=0;
	for(int i=0;i<=k-1;i++){
		if(histPoints[i][1] >= max)
			maxFlag=true;
		if(maxFlag==true && histPoints[i][1]<=min && histPoints[i][1]>0){
			minFlag=true;
			threshold = i;
			break;
		}
	}

	//cout<<"threshold point"<<threshold<<endl;
	




	cout<<"end"<<endl;
	cout<<"Frame Count - "<<frameCount<<endl;
	//system("pause");

}