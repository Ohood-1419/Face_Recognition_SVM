%% Svm classification
%% load images
imdsTrain = imageDatastore('Train',...
  'IncludeSubfolders',true,...
  'LabelSource','foldernames');
imdsTest = imageDatastore('Test');
%% count label
Train_disp = countEachLabel(imdsTrain);
disp(Train_disp);
Train_disp
imageSize = [256,256];
imagel = readimage(imdsTrain, 1);
scaleImage = imresize(imagel, imageSize);
[features, visualization] = extractHOGFeatures(scaleImage);
imshow(scaleImage); hold on; plot(visualization); title('HOG feature');

numImages = length(imdsTrain.Files);
featuresTrain = zeros(numImages, size(features,2),'single'); %featuresTrain
for i = 1:numImages
    imageTrain = readimage(imdsTrain, i);
    imageTrain = imresize(imageTrain, imageSize);
    featuresTrain (i, :) = extractHOGFeatures(imageTrain);
end
% train classifier
trainLabels = imdsTrain.Labels;
classifer = fitcecoc (featuresTrain,trainLabels);
%% classification
numTest = length(imdsTest.Files);
for i = 1:numTest
   testImage = readimage(imdsTest, i);
   scaleTestImage = imresize(testImage, imageSize);
   featureTest = extractHOGFeatures(scaleTestImage);
   [predictIndex, score] = predict (classifer, featureTest);
   figure;imshow(testImage);
   title(['predictImage: ', char(predictIndex)]);
end
