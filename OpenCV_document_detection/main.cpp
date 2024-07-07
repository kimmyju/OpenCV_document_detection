#include <iostream>
#include <cmath>
#include "opencv2/opencv.hpp"
using namespace cv;
using namespace std;

int main()
{
	//Mat origin = imread("C:/Users/yuido/OneDrive/���� ȭ��/c++/opencv_pro/opencv_pro/development.png", IMREAD_REDUCED_COLOR_2);// , IMREAD_REDUCED_COLOR_2);

	//	bool tf = origin.empty();
	//	if (origin.empty())
	//	{
	//		cerr << "Original image load failed!" << endl;
	//		return -1;
	//	}
	//	imshow("origin", origin);

	//	Mat image;
	//	cvtColor(origin, image, COLOR_BGR2GRAY);

	//	// medianBlur(image, image, 63);
	//	// imshow("median", image);

	//	
	//	Mat image1;
	//	bilateralFilter(image, image1, -1, 10, 5);
	//	//image1.copyTo(image);
	//	imshow("bilateral", image1);


	//	waitKey();

		//adaptiveThreshold(image, image, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 11, 5);
		//imshow("adaptive threshold", image);

		//cornerHarris(image, image, 3, 3, 0.04);
		//normalize(image, image, 0, 255, NORM_MINMAX, CV_8U);
		//for (int j = 1; j < image.rows - 1; j++) {
		//	for (int i = 1; i < image.cols - 1; i++) {
		//		if (image.at<uchar>(j, i) > 150) {
		//			if (image.at<float>(j, i) > image.at<float>(j - 1, i) &&
		//				image.at<float>(j, i) > image.at<float>(j + 1, i) &&
		//				image.at<float>(j, i) > image.at<float>(j, i - 1) &&
		//				image.at<float>(j, i) > image.at<float>(j, i + 1)) {
		//				circle(origin, Point(i, j), 5, Scalar(0, 255, 255), 2);
		//			}
		//		}
		//	}
		//}

		//imshow("harris", origin);
		//waitKey();








	//���� 9,erode,open
	Mat src;
	src = imread("C:/Users/yuido/OneDrive/���� ȭ��/c++/opencv_pro/opencv_pro/qr.png", 0);

	if (src.empty())//��������� true, ��� ������ false��ȯ
	{
		cout << "the file is empty!!" << endl;
		return -1;
	}


	resize(src, src, Size(900, 800));
	imshow("src", src);

	Mat src1;
	bilateralFilter(src, src1, -1, 19, 5);
	//GaussianBlur(src, src1, Size(), 0.2);


	//medianBlur(src, src1, 9);
	imshow("gau", src1);
	//imshow("medianBlur", src1);

	//cvtColor(src1, src1, COLOR_RGB2GRAY);


	Mat img1;
	adaptiveThreshold(src1, img1, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 7, 2);//����:210
	//threshold(src1, img1, 165, 255, THRESH_BINARY);
	imshow("thresholding", img1);

	Mat img2;
	img1.copyTo(img2);
	morphologyEx(img2, img2, MORPH_OPEN, Mat(), Point(-1, -1), 1);//open ħ��-->��â
	imshow("CLOSE1", img2);

	Mat img3;
	img2.copyTo(img3);
	floodFill(img3, Point(0, 0), 0);
	imshow("floodFill1", img3);



	//�󺧸� �۾�
	Mat labels, stats, centroids, im_out2;
	img3.copyTo(im_out2);
	int cnt = connectedComponentsWithStats(im_out2, labels, stats, centroids);

	unsigned int* p; //label������
	unsigned int* q;

	int maxIn = 1;
	for (int i = 2; i < cnt; i++) //���� ū ���̺� ã��
	{
		q = stats.ptr<unsigned int>(maxIn);
		p = stats.ptr<unsigned int>(i);
		if (p[4] > q[4])
			maxIn = i;
	}

	for (int i = 0; i < labels.rows; i++)
	{
		p = labels.ptr<unsigned int>(i);
		for (int j = 0; j < labels.cols; j++)
		{
			if (p[j] != maxIn && p[j] != 0)
				//im_out2.at<uchar>(i,j) = 0;
				im_out2.ptr<uchar>(i)[j] = 0;
		}
	}

	imshow("connectedComponents", im_out2);


	Mat inv_img2;
	bitwise_not(im_out2, inv_img2);//��Ʈ ����(����) //���� img_out

	floodFill(inv_img2, Point(0, 0), 0);
	imshow("bitewise_not2", inv_img2);

	Mat img5 = im_out2 | inv_img2;//��Ʈ ������ OR ����(thresholding,��Ʈ���� �̹���)
	imshow("bitewise_OR", img5);



	////
	////Mat harris;
	////cornerHarris(img4, harris, 3, 3, 0.04);//0.04 ���ڰ� �����ϱ�

	////Mat harris_norm;
	////normalize(harris, harris_norm, 0, 255, NORM_MINMAX, CV_8U);
	////imshow("harris_norm", harris_norm);

	Mat img6;
	img5.copyTo(img6);
	vector<Point2i> corners;
	goodFeaturesToTrack(img6, corners, 4, 0.01, 380);


	cvtColor(img6, img6, COLOR_GRAY2BGR);

	for (int i = 0; i < size(corners); i++)
		circle(img6, corners[i], 5, Scalar(0, 0, 255), 2);

	imshow("goodFeaturesToTrack", img6);
	imwrite("./corner.jpg", img6);

	/*Point2f im_out2_Quad[4];
	Point2f dst_Quad[4];
	int cntVertex = 0;
	for (int j = 1; j < harris.rows - 1; j++)
	{
		for (int i = 1; i < harris.cols - 1; i++)
		{
			if (harris_norm.at<uchar>(j, i) > 120)
			{
				if (harris.at<float>(j, i) > harris.at<float>(j - 1, i) &&
					harris.at<float>(j, i) > harris.at<float>(j + 1, i) &&
					harris.at<float>(j, i) > harris.at<float>(j, i - 1) &&
					harris.at<float>(j, i) > harris.at<float>(j, i + 1))
				{
						im_out2_Quad[cntVertex++] = Point2f(i, j);
						circle(dst, Point(i, j), 5, Scalar(0, 0, 255), 2);


				}
			}
		}

	}


	imshow("harrisCorner", dst);*/

	//�������
	//int w = 900, h = 700;
	//dst_Quad[0] = Point2f(0, 0); // ��� ���� ��ǥ
	//dst_Quad[1] = Point2f(w-1, 0);
	//dst_Quad[2] = Point2f(0, h-1);
	//dst_Quad[3] = Point2f(w-1, h-1);
	//
	//Mat pers = getPerspectiveTransform(im_out2_Quad, dst_Quad);
	// 

	//Mat output_img;
	//warpPerspective(src, output_img, pers, Size(w, h));

	//namedWindow("perspectiveTransform", WINDOW_NORMAL);
	//resizeWindow("perspectiveTransform", 900, 700);
	//imshow("perspectiveTransform", output_img);

	//Mat out_img2;
	//flip(output_img, out_img2, 1);

	//namedWindow("flip", WINDOW_NORMAL);
	//resizeWindow("flip", 900, 700);
	//imshow("flip", out_img2);

	//


	//vector<vector<Point>> contours;
	//findContours(im_out2, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

	//Mat dst;
	//cvtColor(im_out2, dst, COLOR_GRAY2BGR);
	//drawContours(dst, contours,0, Scalar(0, 0, 255), 2);
	//imshow("dst", dst);

	//vector<Point> vertex;
	//approxPolyDP(contours[0], vertex, arcLength(contours[0], true) * 0.02, true);
	//

	Point2f in_Quad[4];

	if (size(corners) == 4)
	{
		//�� �������� ����
		for (int i = size(corners) - 1; i > 0; i--)//i = indexSize~1����
			for (int j = 0; j < i; j++)
			{
				if ((corners[j].y > corners[j + 1].y) || ((corners[j].y == corners[j + 1].y) && (corners[j].x > corners[j + 1].x)))//���� ���� ��� �ٽ� �����ϱ�
				{
					Point2i temp;
					temp = corners[j];
					corners[j] = corners[j + 1];
					corners[j + 1] = temp;
				}


			}



		////1. �������� ������ ������ ���������� ������ ������ �������� ã�� �۾�
		int minRow = 0;
		int maxRow = 3;
		int cnt = 0;
		for (int i = 1; i < size(corners); i++)
			if (corners[i].x <= corners[minRow].x)
				cnt++;



		if (cnt == 1)//���������� ������ �����϶�
		{
			in_Quad[0] = corners[minRow];
			in_Quad[2] = corners[maxRow];

			if (corners[1].x >= corners[minRow].x)
			{
				in_Quad[1] = corners[1];
				in_Quad[3] = corners[2];
			}

			else
			{
				in_Quad[1] = corners[2];
				in_Quad[3] = corners[1];
			}

		}

		else//������ �������� ������ �ִ� ��� ���� ���ʿ� �ִ� ���� ã�´�
		{
			in_Quad[3] = corners[maxRow];
			in_Quad[1] = corners[minRow];
			if (corners[1].x <= corners[maxRow].x)
			{
				in_Quad[0] = corners[1];
				in_Quad[2] = corners[2];
			}
			else
			{
				in_Quad[0] = corners[2];
				in_Quad[2] = corners[1];
			}
		}



		Point2f dst_Quad[4];
		int w = 900, h = 800;
		dst_Quad[0] = Point2f(0, 0); // ��� ���� ��ǥ
		dst_Quad[1] = Point2f(w - 1, 0);// ��, ��
		dst_Quad[2] = Point2f(w - 1, h - 1);
		dst_Quad[3] = Point2f(0, h - 1);

		Mat pers = getPerspectiveTransform(in_Quad, dst_Quad);


		Mat output_img;
		warpPerspective(src, output_img, pers, Size(w, h));
		imshow("perspectiveTransform", output_img);
		//imwrite("./qr.jpg", output_img);
	}



	waitKey();//����Ʈ�� = 0





}

