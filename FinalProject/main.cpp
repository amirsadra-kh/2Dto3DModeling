//
//  main.cpp
//  FinalProject
//
//  Created by Amir Sadra on 4/5/17.
//  Copyright © 2017 Amir Sadra. All rights reserved.
//

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <Eigen/Sparse>
#include <Eigen/Eigen>

//Defining namespaces
using namespace cv;
using namespace std;

//Streaming txt file for grabbing the image set.
void readInputText (string txt, vector<Mat> &result, Mat &mask){
    ifstream infile(txt);
    string line;
    Mat imR;
    
    int i = 0;
    while(infile >> line){
        string img = line;
        imR = imread(img,1);
        imR.convertTo(imR, CV_32F);
        imR /= 255.0;
        if (i==0)
            mask = imR;
        else{
            imR = imR.mul(mask);
            result.push_back(imR);
        }
        i++;
    }
}

int main(int argc, char * argv[]) {
    
    //Initiating a Matrix of size 13 for storing images.
    vector<Mat> chrome;
    vector<Mat> src;
    Mat chrome_mask;
    Mat src_mask;
    
    //Reading image sets
    readInputText("chrome.txt", chrome, chrome_mask);
    readInputText("buddha.txt", src, src_mask);
    
    int size = (int)chrome.size();
    
    //Initializing vectors of matrices for chrome
    std::vector<std::vector<Mat>> chrome_channels;
    chrome_channels.resize(size);
    for (int i =0; i<size; i++) {
        chrome_channels[i].resize(3);
    }
    
    //Initializing vectors of matrices for source pic
    std::vector<std::vector<Mat>> src_channels;
    src_channels.resize(size);
    for (int i =0; i<size; i++) {
        src_channels[i].resize(3);
    }
    
    //Splitting source and chrome images channels
    for (int i=0; i<chrome.size(); i++){
        split(chrome[i], chrome_channels[i]);
        split(src[i], src_channels[i]);
        
    }
    
    //Finding center of the sphere and it's radius
    vector<Vec3f> circles;
    Mat chrome_mask_gray;
    cvtColor(chrome_mask, chrome_mask_gray, CV_BGR2GRAY);
    chrome_mask_gray *= 255.0;
    chrome_mask_gray.convertTo(chrome_mask_gray, CV_8U);
    HoughCircles(chrome_mask_gray, circles, CV_HOUGH_GRADIENT, 1, chrome_mask.rows/8,200,100);
    float radius = circles[0][2];
    float xC = circles[0][0];
    float yC = circles[0][1];
    
    //Finding hot spot of sphere | NOTE THAT i=1 because we skip the mask which is at 0.
    vector <Point> hotspot;
    Point min, max;
    double minV, maxV;
    for (int i=0; i<chrome_channels.size(); i++){
        minMaxLoc(chrome_channels[i][0], &minV, &maxV, &min, &max);
        hotspot.push_back(max);
        circle(chrome_channels[i][0], hotspot[i], 4, Scalar(255,0,0));
        imshow("ttt", chrome_channels[i][0]);
        //waitKey(0);
    }
    
    //Finding 'z' of sphere's hotspots and calculating their normal vectors
    vector<Mat> normal;
    normal.resize(12);
    for (int i=0; i<chrome.size();i++)
        normal[i] = Mat (1,3, CV_32FC1);
    
    float x = 0;
    float y = 0;
    float z = 0;
    
    for (int i=0; i<chrome.size(); i++){
        x = hotspot[i].x - xC;
        y = yC - hotspot[i].y;
        z = sqrt(pow(radius,2) - pow(x,2) - pow(y,2));
        
        normal[i].at<float>(0,0) = x;
        normal[i].at<float>(0,1) = y;
        normal[i].at<float>(0,2) = z;
        
        normalize(normal[i],normal[i]);
    }
    
    //Calculating the light vectors.
    Mat reflection (1,3,CV_32FC1);
    reflection.at<float>(0,0) = 0;
    reflection.at<float>(0,1) = 0;
    reflection.at<float>(0,2) = 1;
    
    normalize(reflection,reflection);
    
    vector<Mat> light;
    light.resize(12);
    for (int i=0; i<light.size(); i++){
        light[i] = Mat(1, 3, CV_32FC1);
    }
    
    // L = 2(N.R)N - R
    for (int i=0; i<chrome.size();i++){
        
        normal[i] = normal[i] * 2*normal[i].dot(reflection);
        
        light[i] = normal[i] - reflection;
        
        normalize(light[i],light[i]);
    }
    
    vector<Mat> light_transpose;
    light_transpose.resize(12);
    for (int i=0; i<light_transpose.size(); i++){
        transpose(light[i], light_transpose[i]);
    }
    
    //Calculating Albedo and Normal map for source image
    Mat normal_map (src_mask.rows, src_mask.cols, CV_32FC3, Scalar(0.0, 0.0, 0.0));
    Mat albido (src_mask.rows, src_mask.cols, CV_32FC3, Scalar(0.0, 0.0, 0.0));
    vector<Mat> G1;
    vector<Mat> G2;
    vector<Mat> G3;
    
    G1.resize(12);
    G2.resize(12);
    G3.resize(12);
    
    Mat I (1,12,CV_32FC1);
    Mat L (3,12,CV_32FC1);
    Mat G (1,3,CV_32FC1);
    Vec3f Kd;
    Mat Nd;
    
    for(int i=0; i<3; i++){
        for(int j=0; j<12; j++){
            L.at<float>(i,j) = light[j].at<float>(0,i);
        }
    }
    
    for(int i=0; i<src_mask.rows; i++){
        for(int j=0; j<src_mask.cols; j++){
            if (src_mask.at<Vec3f>(i,j)[0] != 0){
                for (int channel = 0; channel < 3; channel++){
                    for (int k=0; k<12; k++){
                        I.at<float>(0,k) = src_channels[k][channel].at<float>(i,j);
                    }
                    G = I*L.t()*(L*L.t()).inv();
                    Kd[channel] = sqrt(pow(G.at<float>(0,0),2) + pow(G.at<float>(0,1),2) + pow(G.at<float>(0,2),2));
                    Nd = G/norm(G);
                }
                albido.at<Vec3f>(i,j) = Kd;
                normal_map.at<Vec3f>(i,j) = Vec3f(Nd.at<float>(0,0) , Nd.at<float>(0,1) , Nd.at<float>(0,2));
            }
        }
    }
    
    
    //Calculating Depth Map
    
    int r = normal_map.rows;
    int c = normal_map.cols;
    
    int number_of_pixels = normal_map.rows * normal_map.cols;
    Eigen::SparseMatrix<double> M(2*number_of_pixels, 2*number_of_pixels);
    Eigen::VectorXd v(2*number_of_pixels);
    
    int id = 0;
    for (int i=1; i<r-90; i++) {
        for (int j=1; j<normal_map.cols-1; j++) {
            if(src_mask.at<Vec3f>(i,j)[0] != 0){
                id = i * normal_map.cols + j;
                M.insert(id, id) = normal_map.at<Vec3f>(i,j)[2];
                v[id] = -normal_map.at<Vec3f>(i,j)[1];
            }
        }
    }
    
    for (int i=1; i<r-90; i++) {
        for (int j=1; j<normal_map.cols-1; j++) {
            if(src_mask.at<Vec3f>(i,j)[0] != 0){
                id = i * normal_map.cols + j + number_of_pixels;
                M.insert(id, id) = normal_map.at<Vec3f>(i,j)[2];
                v[id] = -normal_map.at<Vec3f>(i,j)[0];
            }
        }
    }
    
    Eigen::ConjugateGradient<Eigen::SparseMatrix<double>> cg;
    cg.compute(M);
    Eigen::VectorXd z2(2*number_of_pixels);
    z2 = cg.solve(v);
    std::cout << "Error: " << cg.error() << std::endl;
    std::cout << "# iterations: " << cg.iterations() << std::endl;
    
    Mat depth_X(src_mask.size(), CV_32FC1, Scalar(0.0));
    Mat depth_Y(src_mask.size(), CV_32FC1, Scalar(0.0));
    
    
    Mat change_X(src_mask.size(), CV_32FC1, Scalar(0.0));
    Mat change_Y(src_mask.size(), CV_32FC1, Scalar(0.0));
    
    for (int i=1; i<normal_map.rows-1; i++) {
        for (int j=1; j<normal_map.cols-1; j++) {
            id = i * normal_map.cols + j;
            change_X.at<float>(i,j) = z2[id];
        }
    }
    
    for (int i=1; i<normal_map.rows-1; i++) {
        for (int j=1; j<normal_map.cols-1; j++) {
            id = i * normal_map.cols + j + number_of_pixels;
            change_Y.at<float>(i,j) = z2[id];
        }
    }
    
    for (int i=1; i<normal_map.rows-1; i++) {
        for (int j=1; j<normal_map.cols-1; j++) {
            if (src_mask.at<Vec3f>(i,j)[0] != 0){
                id = i * normal_map.cols + j;
                depth_X.at<float>(i,j) = depth_X.at<float>(i-1,j) - change_X.at<float>(i,j);
            }
        }
    }
    
    for (int i=1; i<normal_map.rows-1; i++) {
        for (int j=1; j<normal_map.cols-1; j++) {
            if (src_mask.at<Vec3f>(i,j)[0] != 0){
                id = i * normal_map.cols + j + number_of_pixels;
                depth_Y.at<float>(i,j) = depth_Y.at<float>(i,j-1) + change_Y.at<float>(i,j);
            }
        }
    }
    
    //Showing the results
    imshow("ALBEDO", albido);
    imshow("change_Y", change_Y);
    imshow("change_X", change_X);
    imshow("depth_X", depth_X/50.0);
    imshow("depth_Y", depth_Y/100.0);
    imshow("depth", ((depth_Y+depth_X)/200.0) + 0.5);
    imshow("NORMAL", normal_map);
    
    waitKey(0);
    
}
