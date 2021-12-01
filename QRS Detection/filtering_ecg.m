function ecg_filtered_isoline = filtering_ecg(signal)

    ecg = signal';
    Fs = 500; % specify sample rate


    %% filtering of the 12-lead ECG

    % Remove baseline wander
    % usage: [filtered_signal,baseline]=ECG_Baseline_Removal(signal,samplerate,window_length,overlap)
    [ecg_filtered_baseline,~] = ECG_Baseline_Removal(ecg,Fs,1,0.5);

    % filter noise frequencies
    % frequencies are already optimized for ECG signals (literature values):
    % Lowpass: 120 Hz, Highpass: 0.3 Hz, Bandstop (49-51 Hz)
    [ecg_filtered_frq] = ECG_High_Low_Filter(ecg_filtered_baseline,Fs,1,150);
    ecg_filtered_frq=Notch_Filter(ecg_filtered_frq,Fs,50,1);

    % isoline correction
    % usage: [filteredsignal,offset,frequency_matrix,bins_matrix]=Isoline_Correction(signal,varargin)

    [ecg_filtered_isoline,~,~,~]=Isoline_Correction(ecg_filtered_frq);

end