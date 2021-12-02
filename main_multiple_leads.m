leads = [1,2, 4,10];

for l = 1:4
    lead = leads(l);
    build_data = true;
    features = zeros(2016,6,4); % data matrix [features labels] modify to add more features
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

              features(n,:,l) = features_ECG; % save features into array
              labels(n,:) = label; % save labels into array
              n=n+1;

            end       

        end
        
    end 
end


%%
% merge all mat files healthy and AF 
mkdir ECG_leads;
features = mean(features,3);

%%
% make a table with features and labels
data_feat_label = array2table(features,'VariableNames',...
    {'covRR', 'bpm', 'QRSwidth', 'Pwave', 'age', 'sex'});
    %{'meanRR', 'covRR', 'meanSS', 'covSS', 'meanQQ', 'covQQ', 'bpm', 'widthRS','QRS', ...
    %'age', 'sex'});
data_feat_label.Label = labels;
save('ECG_leads\features_label.mat', 'data_feat_label');


%%
% Mdl = fitcnb(features,labels,'OptimizeHyperparameters','auto',...
%     'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName',...
%     'expected-improvement-plus'));

%% Naive-Bayes classifier

labels(any(ismissing(features),2), :) = [];
features( any(ismissing(features),2), :) = [];

Mdl = fitcnb(features,labels, 'DistributionNames','kernel','Width',10.502);
isLabels1 = resubPredict(Mdl);
ConfusionMat = confusionmat(labels,isLabels1); % confusionchart
accuracy = sum(labels == isLabels1,'all')/numel(isLabels1)
sensitivity  = ConfusionMat(1,1)/(ConfusionMat(1,1)+ConfusionMat(1,2))
specificity = ConfusionMat(2,2)/(ConfusionMat(2,1)+ConfusionMat(2,2))

%% Cross-validation

n = 2016; % number of observations
c = cvpartition(n,'KFold',10);
accuracies = zeros(10,1);
accuracy_gmm = zeros(10,1);

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
    Mdl = fitcnb(x_train,y_train,'DistributionNames','kernel','Width',3.5068);
    y = predict(Mdl,x_test);
    accuracies(k) = sum(table2array(y_test) == y,'all')/numel(y); 
    
    GMModel = fitgmdist(table2array(x_train),2);
    P = posterior(GMModel, table2array(x_test));
    [~,y] = max(P,[],2); 
    y(y==1)=0;
    y(y==2)=1;
    accuracy_gmm(k) = sum(table2array(y_test) == y,'all')/numel(Y);

end

mean_acc_bn = mean(accuracies)
std_acc_bn = std(accuracies)

mean_acc_gmm = mean(accuracy_gmm)
std_acc_gmm = std(accuracy_gmm)


%%
% mdl = rica(features,6);
% GMModel = fitgmdist(mdl.TransformWeights,2);
% P = posterior(GMModel, mdl.TransformWeights);
% [~,y] = max(P,[],2); 
% y(y==1)=0;
% y(y==2)=1;
% accuracy_gmm = sum(table2array(y_test) == y,'all')/numel(Y)

