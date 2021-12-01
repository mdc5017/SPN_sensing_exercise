%% SPN Sensing Coursework
clc
clear

lead = 2;
%% Building data
build_data =  true;
features = zeros(2016,6); % data matrix [features labels] modify to add more features
labels = zeros(2016,1);
n = 1;

if build_data
    % label + pre-process data
    folder = ["ECG normal","ECG AF"];
    health_state = [0, 1]; % 0 healthy , 1 AF patient
    for i = 1:2        
        myFolder = folder(i);
        matFiles = dir(myFolder);        
        
        label = health_state(i);        
        
        for k = 3:length(matFiles)
          matFilename = fullfile(myFolder, matFiles(k).name); % get mat file name
          [path, name, ext] = fileparts(matFilename);
          matName = strcat(name, ext);
          
          load(matFilename); % load mat file
          
          [features_ECG] = pre_processing2(ECG,lead); % filtering + feature extraction
          
          features(n,:) = features_ECG; % save features into array
          labels(n,:) = label; % save labels into array
          n=n+1;
          
        end       
         
    end
    
    
    % merge all mat files healthy and AF 
    mkdir ECG_normal_AF;
    save('ECG_normal_AF\features_label.mat', 'features', 'labels');
    copyfile 'ECG normal' 'ECG_normal_AF'
    copyfile 'ECG AF' 'ECG_normal_AF'    
   
    % make a table with features and labels
    data_feat_label = array2table(features,'VariableNames',...
        {'covRR', 'bpm', 'QRSwidth', 'Pwave', 'age', 'sex'});
        %{'meanRR', 'covRR', 'meanSS', 'covSS', 'meanQQ', 'covQQ', 'bpm', 'widthRS','QRS', ...
        %'age', 'sex'});
    data_feat_label.Label = labels;
    save('ECG_normal_AF\features_label.mat', 'data_feat_label', '-append');
end 


%% Multiclass naive bayes model
clc 

load('ECG_normal_AF\features_label.mat') 

% clean data
labels(any(ismissing(features),2), :) = [];
features( any(ismissing(features),2), :) = [];

Mdl = fitcnb(features,labels, 'DistributionNames','kernel','Width',10.502);
isLabels1 = resubPredict(Mdl);
ConfusionMat = confusionmat(labels,isLabels1); % confusionchart
accuracy = sum(labels == isLabels1,'all')/numel(isLabels1);
sensitivity  = ConfusionMat(1,1)/(ConfusionMat(1,1)+ConfusionMat(1,2));
specificity = ConfusionMat(2,2)/(ConfusionMat(2,1)+ConfusionMat(2,2));


%% Optimisation
% Mdl = fitcnb(features,labels,'OptimizeHyperparameters','auto',...
%     'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName',...
%     'expected-improvement-plus'));

%% Cross-validation split train/test

n = 2016; % number of observations
c = cvpartition(n,'KFold',10);
accuracies = zeros(10,1);

for k = 1:10          
       
    idx_train = training(c,k);
    idx_test = test(c,k);
        
    % train dataset
    x_train = data_feat_label(idx_train,1:6); 
    y_train = data_feat_label(idx_train,7);

    % test dataset
    x_test = data_feat_label(idx_test,1:6);  
    y_test = data_feat_label(idx_test,7);    

    % classification
    Mdl = fitcnb(x_train,y_train,'DistributionNames','kernel','Width',10.502);
    y = predict(Mdl,x_test);
    accuracies(k) = sum(table2array(y_test) == y,'all')/numel(y); 

end

mean_acc_bn = mean(accuracies);
std_acc_bn = std(accuracies);

%% 2D visualisation - example age and bpm

x = min(data_feat_label.bpm):1:max(data_feat_label.bpm);
y =  min(data_feat_label.age):1:max(data_feat_label.age);

[x1 x2] = meshgrid(x,y);
x = [x1(:) x2(:)];
Mdl = fitcnb(features(:,[2,5]),labels);
ms = predict(Mdl, x);
figure;
gscatter(x1(:),x2(:),ms,'cym');
hold on;
gscatter(data_feat_label.bpm,data_feat_label.age, data_feat_label.Label,'gr','.',15);
xlabel("bpm")
ylabel("age")


%% GMM Model 
clc
% estimate best k
AIC = zeros(1,4);
GMModels = cell(1,4);
options = statset('MaxIter',500);
for k = 1:4
    GMModels{k} = fitgmdist(features,k,'Options',options,'CovarianceType','diagonal');
    AIC(k)= GMModels{k}.AIC;
end

[minAIC,numComponents] = min(AIC);
numComponents;
%%
n = 2016; % number of observations
c = cvpartition(n,'KFold',10);
accuracies = zeros(10,1);

for k = 1:10          
       
    idx_train = training(c,k);
    idx_test = test(c,k);    
    
    % train dataset
    x_train = data_feat_label(idx_train,1:6); 
    y_train = data_feat_label(idx_train,7);

    % test dataset
    x_test = data_feat_label(idx_test,1:6);  
    y_test = data_feat_label(idx_test,7);    

    % classification
    GMModel = fitgmdist(table2array(x_train),2);
    P = posterior(GMModel, table2array(x_test)); 
    [~,y] = max(P,[],2); 
    y(y==1)=0;
    y(y==2)=1;
    accuracies(k) = sum(table2array(y_test) == y,'all')/numel(y);

end

mean_acc_gmm = mean(accuracies);
std_acc_gmm = std(accuracies);


%% plot GMM
gm = fitgmdist(features(:,[2,5]),2);
P = posterior(gm,features(:,[2,5]));
[~,y] = max(P,[],2); 
y(y==1)=0;
y(y==2)=1;
ConfusionMat = confusionmat(labels,y); % confusionchart
accuracy = sum(labels == y,'all')/numel(y);
sensitivity  = ConfusionMat(1,1)/(ConfusionMat(1,1)+ConfusionMat(1,2));
specificity = ConfusionMat(2,2)/(ConfusionMat(2,1)+ConfusionMat(2,2));

%%
figure
scatter(features(:,2),features(:,5),10,P(:,1))
c2 = colorbar;
ylabel(c2,'Posterior Probability of Component 1')
grid on

%% Compare CNB and GMM
clc

cv = cvpartition(n,'HoldOut',0.3);
idx = cv.test;

x_train = data_feat_label(~idx,1:6); 
y_train = data_feat_label(~idx,7);


x_test = data_feat_label(idx,1:6);  
y_test = data_feat_label(idx,7);
    
Mdl = fitcnb(x_train,y_train,'DistributionNames','kernel','Width',10.502);
y = predict(Mdl,x_test);
accuracy_bn = sum(table2array(y_test) == y,'all')/numel(y);
[X,Y,~,~,~] = perfcurve(table2array(y_test),y,1);

figure;
plot(X,Y)
hold on

GMModel = fitgmdist(table2array(x_train),2);
P = posterior(GMModel, table2array(x_test));
[X,Y,T,AUC_trainGMM,optL] = perfcurve(table2array(y_test),P(:,2),1); 
[~,y] = max(P,[],2); 
y(y==1)=0;
y(y==2)=1;
accuracy_gmm = sum(table2array(y_test) == y,'all')/numel(Y);
plot(X,Y)

xlabel('False positive rate') 
ylabel('True positive rate')
ylim([0 1.05])
legend('Naïve Bayes', 'GMM')
title('ROC for Classification by Naive Bayes')

