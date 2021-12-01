    
function features = pre_processing2(ECG,lead)
      whichdata=ECG.data;%plot the 'data' field in the matab data file
      whichdata = whichdata(lead,:);
    %% filtering
      cd('QRS Detection')
      filtered = filtering_ecg(whichdata);
      ecg_filtered_isoline = filtered';    
      Fs = 500;
      
  %% feature extraction
    % produce FPT Table
    % usage: [FPT_MultiChannel,FPT_Cell]=Process_ECG_Multi(signal,samplerate,varargin)
    [FPT_MultiChannel,FPT_Cell]=Annotate_ECG_Multi(ecg_filtered_isoline,Fs);

    % extract FPTs for Channel 1 (Lead I):    
    FPT_Lead = FPT_Cell{1,1};
    FPT_Lead(isempty(FPT_Lead))=0;
    
    if size(FPT_Lead,2)<12
        FPT_Lead = zeros(12);
    end
        
    Pwave_samples = reshape(FPT_Lead(:,1:3), [1,size(FPT_Lead(:,1:3),1)*size(FPT_Lead(:,1:3),2)]);
    QRS_samples = reshape([FPT_Lead(:,4),FPT_Lead(:,6), FPT_Lead(:,8)] , [1,size(FPT_Lead(:,1:3),1)*size(FPT_Lead(:,1:3),2)]);
    Twave_samples = reshape(FPT_Lead(:,10:12), [1,size(FPT_Lead(:,10:12),1)*size(FPT_Lead(:,10:12),2)]);
    cd ..
    
    sortedQRS = sort(QRS_samples);
    Rwave = sortedQRS(2:3:end);
    
    % R-R interval
    RRinterval=(1:length(Rwave)-1);
    for j=1:length(Rwave)-1
     RRinterval(j)=Rwave(j+1)-Rwave(j);
    end
    
    meanRR=mean(RRinterval);% find the average R-R interval
    secperbeat=mean(RRinterval)*1/(500);%second per beat (sampling at 500Hz)
    bpm=60/secperbeat;%beat per minute
    covRR = cov(RRinterval); % covariance to measure irregularity
    
    
    % QRS complex    
    Qwave = sortedQRS(1:3:end);
    Swave = sortedQRS(3:3:end);
    QRSwidth = mean(Swave-Qwave);
    
    % P wave
    sortedP = sort(Pwave_samples);
    Pwave = sortedP(2:3:end);
    Pwave = nonzeros(Pwave)';
    Pwave = mean(ecg_filtered_isoline(Pwave));
    
    % age and sex    
    age = ECG.age;
    if ECG.sex == "Female"
        sex = 0;
    else
        sex = 1;
    end
    
    features = [covRR, bpm, QRSwidth, Pwave, age, sex];
end