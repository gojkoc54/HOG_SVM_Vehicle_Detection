# Vehicle Detection HOG + SVM

This repository contains the implementation and a short overview of my Bachelor Thesis, 'Vehicle Detection Using Front Camera Image Classification'. The original thesis paper is written in Serbian, so this readme file will provide a recap of important points from the thesis. 

## Table of Content
* [Pipeline](#pipeline)
* [HOG features](#hog) 
* [SVM classifiers](#svm)
* [Sliding window](#sliding_window)
* [Results](#results)

## Pipeline

The pipeline is given in the following flowchart:

<p align="center">
<img src="images/flowcharts/pipeline.png" width="600"/>
</p>

 The main idea of the first data collection process was to gather enough labeled data for classifier training, meaning that we looked for a lot of labeled images of cars, as well as non-car items (parts of the road, road signs, pedestrians, ...). Following datasets were used:
* Car class: https://arxiv.org/abs/1506.08959
* Non-car class: https://btsd.ethz.ch/shareddata/

On the other hand, the second data collection process had the goal of gathering realistic and applicable training data, so we decided to go on a little data-hunt :camera:, driving around Serbia and collecting images from everyday traffic. 

## HOG features

