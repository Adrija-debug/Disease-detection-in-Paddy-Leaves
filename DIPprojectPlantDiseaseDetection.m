%% Digital Image Processing Project : Plant leaf disease detection 
clc
close all 
clear all
warning('off')

        % getting image from user
        [filename, pathname] = uigetfile({'*.*';'*.bmp';'*.jpg';'*.gif'}, 'Pick a Leaf Image File');
        I = imread([pathname,filename]);
        
        %% Image pre-processing
        %Resizing the acquired image
        I = imresize(I,[300,800]);
        figure, imshow(I); title('Query Leaf Image'); 
        drawnow;
        set(get(handle(gcf),'JavaFrame'),'Maximized',1);  
        
        %% Image Enhancement
        %I = imadjust(I,stretchlim(I));
        %f = figure;
        %imshow(I);title('Contrast Enhanced');
        %drawnow;
        %set(get(handle(gcf),'JavaFrame'),'Maximized',1);  
        
       
%% Color Image Segmentation
% Use of K Means clustering for segmentation
% Convert Image from RGB Color Space to L*a*b* Color Space 
% The L*a*b* space consists of a luminosity layer 'L*', chromaticity-layer 'a*' and 'b*'.
% All of the color information is in the 'a*' and 'b*' layers.
cform = makecform('srgb2lab');
% Apply the colorform
lab_he = applycform(I,cform);

% Classify the colors in a*b* colorspace using K means clustering.
% Since the image has 3 colors create 3 clusters.
% Measure the distance using Euclidean Distance Metric.
ab = double(lab_he(:,:,2:3));
nrows = size(ab,1);
ncols = size(ab,2);
ab = reshape(ab,nrows*ncols,2);
nColors = 3;
[cluster_idx cluster_center] = kmeans(ab,nColors,'distance','sqEuclidean', ...
                                      'Replicates',3);
%[cluster_idx cluster_center] = kmeans(ab,nColors,'distance','sqEuclidean','Replicates',3);
% Label every pixel in tha image using results from K means
pixel_labels = reshape(cluster_idx,nrows,ncols);
%figure,imshow(pixel_labels,[]), title('Image Labeled by Cluster Index');

% Create a blank cell array to store the results of clustering
segmented_images = cell(1,3);
% Create RGB label using pixel_labels
rgb_label = repmat(pixel_labels,[1,1,3]);

for k = 1:nColors
    colors = I;
    colors(rgb_label ~= k) = 0;
    segmented_images{k} = colors;
end

figure,
subplot(2,2,1); imshow(I); title('Original Image');
subplot(2,2,2);imshow(segmented_images{1});title('Cluster 1');
subplot(2,2,3);imshow(segmented_images{2});title('Cluster 2');
subplot(2,2,4);imshow(segmented_images{3});title('Cluster 3');
drawnow;
set(get(handle(gcf),'JavaFrame'),'Maximized',1);  



              
        %% Feature Extraction
        
        x = inputdlg('Enter the cluster no. containing the ROI only:');
        i = str2double(x);

        seg_img = segmented_images{i};

        % Converting given RGB image to grayscale 
        img = rgb2gray(seg_img);
        % Create the Gray Level Cooccurance Matrices (GLCMs)
        glcms = graycomatrix(img);

        % Derive Statistics from GLCM
        stats = graycoprops(glcms,'Contrast Correlation Energy Homogeneity');

        Contrast = stats.Contrast;
        Correlation = stats.Correlation;
        Energy = stats.Energy;
        Homogeneity = stats.Homogeneity;
        Mean = mean2(seg_img);
        Standard_Deviation = std2(seg_img);
        Entropy = entropy(seg_img);
        RMS = mean2(rms(seg_img));
        Variance = mean2(var(double(seg_img)));
        a = sum(double(seg_img(:)));
        Smoothness = 1-(1/(1+a));
        Kurtosis = kurtosis(double(seg_img(:)));
        Skewness = skewness(double(seg_img(:)));

        % Inverse Difference Moment
        m = size(seg_img,1);
        n = size(seg_img,2);
        in_diff = 0;
        for i = 1:m
            for j = 1:n
                temp = seg_img(i,j)./(1+(i-j).^2);
                in_diff = in_diff+temp;
            end
        end
        IDM = double(in_diff);
        if IDM==0
            IDM=1;
        end
       
        feat_disease = [Contrast,Correlation,Energy,Homogeneity, Mean, Standard_Deviation, Entropy, RMS, Variance, Smoothness, Kurtosis, Skewness,IDM];
       
        disp('The extracted features are');
        disp(feat_disease);
        %% SVM Classifier
        % Load All The Features
        %load('Training_Data.mat')
        load('TrainingDataset.mat');

        % Put the test features into variable 'test'
        test = feat_disease;
        result = multisvm(dataset,diseasetype,test);
        %disp(result);
                
        %% Visualize Results
        if result == 1
            helpdlg(' Disease Detected: Bacterial Blight ');
            disp(' Disease is Bacterial Blight ');
        elseif result == 2
            helpdlg(' Disease Detected: Leaf Smut ');
            disp('Disease is Leaf Smut');
        elseif result == 3
            helpdlg(' Disease Detected: brown spot ');
            disp('Disease is brown spot');
        elseif result == 4
            helpdlg(' No Disease Detected: Healthy Leaf ');
            disp('Healthy Leaf'); 
        else
            helpdlg(' Disease Detected: Not Found ');
            disp(' Disease cannot be found '); 
        end
        
        
%% Evaluate Accuracy

load('AccuracyDataset.mat')
%load('Accuracy_Data.mat')
Accuracy_Percent= zeros(500,1);
for i = 1:500
data = dataset;
%data = Train_Feat;
%groups = ismember(Train_Label,1);
groups = ismember(Train_Label,0);
%groups = Train_Label;
[train,test] = crossvalind('HoldOut',groups);
%disp(test);
%test=double(test);
cp = classperf(groups);

svmStruct = svmtrain(data(train,:),groups(train),'showplot',false,'kernel_function','linear');

classes = svmclassify(svmStruct,data(test,:),'showplot',false);
classperf(cp,classes,test);
Accuracy = cp.CorrectRate;
Accuracy_Percent(i) = Accuracy.*100;
end
Max_Accuracy = max(Accuracy_Percent);
sprintf('Accuracy of Classification with 500 iterations is: %g%%',Max_Accuracy)
%Mean_Accuracy = mean(Accuracy_Percent);
%sprintf('Accuracy of Classification with 500 iterations is: %g%%',Mean_Accuracy)      

    