#include <cassert>
#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include "ex.hpp"

using namespace cv;
using namespace std;
using namespace caffe;


BBox::BBox(int x, int y, int w, int h) {
	this->x = x; this->y = y;
	this->width = w; this->height = h;
	this->rect = Rect(x, y, w, h);
}

BBox::BBox(const Rect &rect) {
	this->x = rect.x; this->y = rect.y;
	this->width = rect.width; this->height = rect.height;
	this->rect = rect;
}

void BBox::Project(const vector<Point2f> &absLandmark, vector<Point2f> &relLandmark) const {
	assert(absLandmark.size() == relLandmark.size());
	for (int i = 0; i < absLandmark.size(); i++) {
		const Point2f &point1 = absLandmark[i];
		Point2f &point2 = relLandmark[i];
		point2.x = (point1.x - this->x) / this->width;
		point2.y = (point1.y - this->y) / this->height;
	}
}

void BBox::ReProject(const vector<Point2f> &relLandmark, vector<Point2f> &absLandmark) const {
	assert(relLandmark.size() == absLandmark.size());
	for (int i = 0; i < relLandmark.size(); i++) {
		const Point2f &point1 = relLandmark[i];
		Point2f &point2 = absLandmark[i];
		point2.x = point1.x*this->width + this->x;
		point2.y = point1.y*this->height + this->y;
	}
}

BBox BBox::subBBox(float left, float right, float top, float bottom) const {
	assert(right>left && bottom>top);
	float x, y, w, h;
	x = this->x + left*this->width;
	y = this->y + top*this->height;
	w = this->width*(right - left);
	h = this->height*(bottom - top);
	return BBox(x, y, w, h);
}


CNN::CNN(const string &network, const string &model) {
	cnn = new Net<float>(network, caffe::TEST);
	assert(cnn);
	cnn->CopyTrainedLayersFrom(model);
}

vector<Point2f> CNN::forward(const Mat &data, const string &layer) {
	float loss = 0.0;
	boost::shared_ptr<caffe::MemoryDataLayer<float> > md_layer;
	md_layer = boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float> >(cnn->layers()[0]);
	if (!md_layer) {
		LOG(FATAL) << "The first layer is not a MemoryDataLayer!";
	}
	std::vector<int> fake(1, 0);
	md_layer->set_batch_size(1);
	md_layer->AddMatVector(vector<Mat>(1, data), fake);
	cnn->ForwardPrefilled(&loss);
	boost::shared_ptr<caffe::Blob<float> > landmarks = cnn->blob_by_name(layer);
	vector<Point2f> points(landmarks->count() / 2);
	for (int i = 0; i < points.size(); i++) {
		Point2f &point = points[i];
		point.x = landmarks->data_at(0, 2 * i, 0, 0);
		point.y = landmarks->data_at(0, 2 * i + 1, 0, 0);
	}
	return points;
}


void FaceDetector::LoadXML(const string &path) {
    bool res = cc.load(path);
	assert(res);
}

int FaceDetector::DetectFace(const Mat &img, vector<Rect> &rects) {
	assert(img.type() == CV_8UC1);
	Mat gray(img.rows, img.cols, CV_8UC1);
	img.copyTo(gray);

	cc.detectMultiScale(gray, rects, 1.2, 2, 0 | CV_HAAR_SCALE_IMAGE, \
		Size(30, 30), Size(gray.cols, gray.rows));

	return rects.size();
}

void Landmarker::LoadModel(const string &path) {
	string network = path + "/1_F.prototxt";
	string model = path + "/1_F.caffemodel";
	F_1 = new CNN(network, model);
}

vector<Point2f> Landmarker::DetectLandmark(const Mat &img, const BBox &bbox){
	assert(img.type() == CV_8UC1);
	Mat data;
	resize(img(bbox.rect), data, Size(39, 39));
	data.convertTo(data, CV_32FC1);

	Scalar meanScalar, stdScalar;
	cv::meanStdDev(data, meanScalar, stdScalar);
	float mean = meanScalar.val[0];
	float std = stdScalar.val[0];
	data = (data - mean) / std;

	vector<Point2f> landmarks = F_1->forward(data, string("fc2"));
	bbox.ReProject(landmarks, landmarks);
	return landmarks;
}
