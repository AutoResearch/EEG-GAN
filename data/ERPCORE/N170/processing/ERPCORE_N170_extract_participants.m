%Clean environment
clear all; clc;

%Participant IDs
participants = 1:40;

%Iterate through participants
EEG_data = zeros(1,132);
for participant_number = participants
    
    %Clear data
    data = [];
    header_info = [];

    %Determine participant filenames
    gen_fn = '_N170_shifted_ds_reref_ucbip_hpfilt_ica_corr_cbip_elist_bins_epoch_interp_ar';
    fn = strcat(num2str(participant_number),'/',num2str(participant_number),gen_fn);    
    
    %Load header and data
    header_info = load(strcat(fn,'.set'), '-mat');
    fid = fopen(strcat(fn,'.fdt'), 'r', 'ieee-le');
    for trialIdx = 1:header_info.EEG.trials % In case the saved data are epoched, loop the process for each epoch. Thanks Ramesh Srinivasan!
        currentTrialData = fread(fid, [header_info.EEG.nbchan header_info.EEG.pnts], 'float32');
        data(:,:,trialIdx) = currentTrialData; % Data dimentions are: electrodes, time points, and trials (the last one is for epoched data)                  
    end
    fclose(fid);
    
    %Extract labels
    labels = [];
    for i = 1:size(data,3)
        event_label = header_info.EEG.epoch(i).eventbinlabel;
        for c =1:length(event_label) %Search for which cell contains label
            if contains(event_label{c},'(')
                event_label = event_label{c};
                break
            end
        end
        lab = split(event_label,'(');
        lab = split(lab{end},')');
        labels(end+1) = str2num(lab{1});
    end

    %Relabel events
    labels(labels > 80) = 99; %Scrambled (ignored)
    labels(labels < 41) = 1; %Faces
    labels(labels > 40 & labels < 81) = 0; %Cars

    data = data(:,:,labels < 99); %Remove ignored labels
    label_binary = labels(labels < 99); %Remove ignored labels

    %Extract specific electrode
    electrode = 26; %PO8
    electrode_data = squeeze(data(electrode,:,:));

    %Process data
    downsampled_data = downsample(electrode_data,2); %Downsample data from 256 to 128Hz (it will speed up GAN training dramatically)

    %Create metadata and add to dataframe
    participant_numbers = repmat(participant_number,size(downsampled_data,2),1);
    trial_numbers = 1:size(downsampled_data,2);
    electrode_numbers = repmat(1,size(downsampled_data,2),1);
    metadata = [...
        participant_numbers'; ...
        label_binary; ...
        trial_numbers; ...
        electrode_numbers'];
    participant_data = [metadata', downsampled_data'];

    EEG_data = [EEG_data; participant_data];
end

%Remove placeholder row
EEG_data = EEG_data(2:end,:);

%Create table for saving as CSV
EEG_table = array2table(EEG_data);
header = {'Participant_ID', 'Condition', 'Trial', 'Electrode'};
for i = 1:128
    header{end+1} = strcat('Time',num2str(i));
end
EEG_table.Properties.VariableNames(1:132) = header;

%Save file
save_filename = '../Full Datasets/erpcore_N170_full.csv';
writetable(EEG_table,save_filename) %Save to csv