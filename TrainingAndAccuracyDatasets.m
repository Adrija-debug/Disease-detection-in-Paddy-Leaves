clc
clear all
close all 
xx = [];
%Train_Feat = [];
%% dataset for diseased leaves
        folder_name = uigetdir(pwd, 'Select the directory of images');
        %loading folder containing all data samples
        jpgImagesDir = fullfile(folder_name, '*.jpg');
        num_of_jpg_images = numel( dir(jpgImagesDir) );
        jpg_files = dir(jpgImagesDir);
        jpg_counter = 0;
% calculate total number of images

totalImages = num_of_jpg_images;

        for k=1:totalImages
            %reading each image
            I = imread( fullfile(folder_name, jpg_files(jpg_counter+1).name ) );
            jpg_counter = jpg_counter + 1;
            
            I = imresize(I,[300,800]);
            %I = imadjust(I,stretchlim(I));
            [I3,RGB] = createMask(I);
            
            seg_img = RGB;
            img = rgb2gray(seg_img);
            glcms = graycomatrix(img);
            
            stats = graycoprops(glcms,'Contrast Correlation Energy Homogeneity');

            Contrast = stats.Contrast;
            Correlation = stats.Correlation;
            Energy = stats.Energy;
            Homogeneity = stats.Homogeneity;
            Mean = mean2(seg_img);
            Standard_Deviation = std2(seg_img);
            Entropy = entropy(seg_img);
            RMS = mean2(rms(seg_img));
            %Skewness = skewness(img)
            Variance = mean2(var(double(seg_img)));
            a = sum(double(seg_img(:)));
            Smoothness = 1-(1/(1+a));
            Kurtosis = kurtosis(double(seg_img(:)));
            Skewness = skewness(double(seg_img(:)));

            % Inverse Difference Movement
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

             
            ff = [Contrast,Correlation,Energy,Homogeneity, Mean, Standard_Deviation, Entropy, RMS, Variance, Smoothness,Kurtosis,Skewness,IDM];
            %loading extracted features to dataset
            dataset(k, :) = ff;
            
            if k<=305 && k>=1
                xx = [xx;1]; % for bacterial blight
            elseif k>=306 && k<=608
                xx = [xx;2]; % for leaf smut  
            elseif k>608
                xx = [xx;3]; % for brown spot
            end
            
            diseasetype = transpose(xx);
        end
        
        %% loading dataset for healthy leaves
        folder_name = uigetdir(pwd, 'Select the directory of images');
        %loading folder containing all data samples
        jpgImagesDir = fullfile(folder_name, '*.jpg');
        num_of_jpg_images1 = numel( dir(jpgImagesDir) );
        jpg_files = dir(jpgImagesDir);
        jpg_counter = 0;
% calculate total number of images

totalImages1 = num_of_jpg_images1;

        for k=1:totalImages1
            %reading each image
            I = imread( fullfile(folder_name, jpg_files(jpg_counter+1).name ) );
            jpg_counter = jpg_counter + 1;
            
            I = imresize(I,[300,800]);
            %I = imadjust(I,stretchlim(I));
            
            [I3,RGB] = LABMASKhealthyLeaf(I);
            
            seg_img = RGB;
            img = rgb2gray(seg_img);
            glcms = graycomatrix(img);
            
            stats = graycoprops(glcms,'Contrast Correlation Energy Homogeneity');

            Contrast = stats.Contrast;
            Correlation = stats.Correlation;
            Energy = stats.Energy;
            Homogeneity = stats.Homogeneity;
            Mean = mean2(seg_img);
            Standard_Deviation = std2(seg_img);
            Entropy = entropy(seg_img);
            RMS = mean2(rms(seg_img));
            %Skewness = skewness(img)
            Variance = mean2(var(double(seg_img)));
            a = sum(double(seg_img(:)));
            Smoothness = 1-(1/(1+a));
            Kurtosis = kurtosis(double(seg_img(:)));
            Skewness = skewness(double(seg_img(:)));

            % Inverse Difference Movement
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
           
          
            ff = [Contrast,Correlation,Energy,Homogeneity, Mean, Standard_Deviation, Entropy, RMS, Variance, Smoothness,Kurtosis,Skewness,IDM];
            %loading extracted features to dataset
            dataset((k+num_of_jpg_images),:) = ff;
            
           
            xx = [xx;4]; % for healthy leaf
            
            
            diseasetype = transpose(xx);
        end
        
       uisave({'dataset','diseasetype'},'TrainingDataset')
        
       Train_Label = [ zeros(911,1); ones(381,1) ];
       uisave({'Train_Label','dataset'},'AccuracyDataset')
       %save Accuracy_Data
       
        disp('Train Completed');
    