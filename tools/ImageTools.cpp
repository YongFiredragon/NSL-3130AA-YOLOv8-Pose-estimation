/*
 * Copyright (c) 2013, NANOSYSTEMS CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include "ImageTools.h"

typedef enum pose_position_{
	nose_position,
	left_shoulder_position,
	right_shoulder_position,
	left_elbow_position,
	right_elbow_position,
	left_wrist_position,
	right_wrist_position,
	left_hip_position,
	right_hip_position,
	left_knee_position,
	right_knee_position,
	left_ankle_position,
	right_ankle_position,
}pose_position;

#ifdef DEEP_LEARNING

void ImageTools::drawSkeleton(cv::Mat &image, YoloPose::Person &item)
{
    auto poseColor = cv::Scalar(0,0,255);
	int thickness = 2;
	
	/*
		0 : nose
		1 : left_shoulder
		2 : right_shoulder
		3 : left_elbow
		4 : right_elbow
		5 : left_wrist
		6 : right_wrist
		7 : left_hip
		8 : right_hip
		9 : left_knee
		10 : right_knee
		11 : left_ankle
		12 : right_ankle
	*/
	for (YoloPose::Keypoint kp:item.kp) {
		cv::circle(image, kp.position, 3, poseColor, cv::FILLED);
	}
	
	cv::Point2d center_shoulder;
	center_shoulder.x = item.kp[left_shoulder_position].position.x + (item.kp[right_shoulder_position].position.x - item.kp[left_shoulder_position].position.x)/2 ;
	center_shoulder.y = item.kp[left_shoulder_position].position.y;
	
	cv::Point2d center_hip;
	center_hip.x = item.kp[left_hip_position].position.x + (item.kp[right_hip_position].position.x - item.kp[left_hip_position].position.x)/2 ;
	center_hip.y = item.kp[left_hip_position].position.y;
	
	cv::circle(image, center_shoulder, 3, poseColor, cv::FILLED);
	cv::circle(image, center_hip, 3, poseColor, cv::FILLED);

	// nose - shoulder
	cv::line(image, item.kp[nose_position].position, center_shoulder, poseColor, thickness);
	// center shoulder -> center hip
	cv::line(image, center_shoulder, center_hip, poseColor, thickness);
	
	// shoulder
	cv::line(image, item.kp[left_shoulder_position].position, item.kp[right_shoulder_position].position, poseColor, thickness);
	// left arm
	cv::line(image, item.kp[left_shoulder_position].position, item.kp[left_elbow_position].position, poseColor, thickness);
	cv::line(image, item.kp[left_elbow_position].position, item.kp[left_wrist_position].position, poseColor, thickness);
	// right arm
	cv::line(image, item.kp[right_shoulder_position].position, item.kp[right_elbow_position].position, poseColor, thickness);
	cv::line(image, item.kp[right_elbow_position].position, item.kp[right_wrist_position].position, poseColor, thickness);
	
	// hip
	cv::line(image, item.kp[left_hip_position].position, item.kp[right_hip_position].position, poseColor, thickness);
	// left leg
	cv::line(image, item.kp[left_hip_position].position, item.kp[left_knee_position].position, poseColor, thickness);
	cv::line(image, item.kp[left_knee_position].position, item.kp[left_ankle_position].position, poseColor, thickness);
	// right leg
	cv::line(image, item.kp[right_hip_position].position, item.kp[right_knee_position].position, poseColor, thickness);
	cv::line(image, item.kp[right_knee_position].position, item.kp[right_ankle_position].position, poseColor, thickness);
}


void ImageTools::drawPose(std::vector<YoloPose::Person> &detections, cv::Mat &image) {
    auto textColor = cv::Scalar(255, 255, 255);
    auto boxColor = cv::Scalar(0, 255, 0);
	int thickness = 2;

    for (YoloPose::Person &item: detections) {
        cv::rectangle(image, item.box, boxColor, thickness);

        std::string infoString = std::to_string(item.score);
        cv::Size textSize = cv::getTextSize(infoString, cv::QT_FONT_NORMAL, 1, 1, nullptr);
        cv::Rect textBox(item.box.x, item.box.y - 40, textSize.width + 10, textSize.height + 20);

        cv::rectangle(image, textBox, boxColor, cv::FILLED);
        cv::putText(image, infoString, cv::Point(item.box.x + 5, item.box.y - 10), cv::FONT_HERSHEY_DUPLEX, 1, textColor, 1, 0);

		drawSkeleton(image, item);
    }
}

void ImageTools::draw(std::vector<YoloPose::Person> &detections, cv::Mat &image) {
    auto textColor = cv::Scalar(255, 255, 255);
    auto boxColor = cv::Scalar(0, 255, 0);
	int thickness = 2;

    for (YoloPose::Person &item: detections) {
        cv::rectangle(image, item.box, boxColor, thickness);

        std::string infoString = std::to_string(item.score);
        cv::Size textSize = cv::getTextSize(infoString, cv::QT_FONT_NORMAL, 1, 1, nullptr);
        cv::Rect textBox(item.box.x, item.box.y - 40, textSize.width + 10, textSize.height + 20);

        cv::rectangle(image, textBox, boxColor, cv::FILLED);
        cv::putText(image, infoString, cv::Point(item.box.x + 5, item.box.y - 10), cv::FONT_HERSHEY_DUPLEX, 1, textColor, 1, 0);
    }
}


void ImageTools::draw(std::vector<YoloDet::Detection> &detections, cv::Mat &image, cv::Scalar boxColor) {
    auto textColor = cv::Scalar(255, 255, 255);
	int thickness = 2;

    for (YoloDet::Detection &item: detections) {
        cv::Rect box = item.box;

        cv::rectangle(image, box, boxColor, thickness);

        std::string infoString = std::to_string(item.confidence).substr(0, 4);
        cv::Size textSize = cv::getTextSize(infoString, cv::FONT_HERSHEY_DUPLEX, 1, 1, nullptr);
        cv::Rect textBox(box.x, box.y - 40, textSize.width + 10, textSize.height + 20);

        cv::rectangle(image, textBox, boxColor, cv::FILLED);
        cv::putText(image, infoString, cv::Point(box.x + 5, box.y - 10), cv::FONT_HERSHEY_DUPLEX, 1, textColor, 1, 0);
    }
}

#endif

