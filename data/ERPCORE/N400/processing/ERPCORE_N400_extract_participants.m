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
    gen_fn = '_N400_shifted_ds_reref_ucbip_hpfilt_ica_corr_cbip_elist_bins_epoch_interp_ar';
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
        labels(end+1) = str2num(header_info.EEG.epoch(i).eventbinlabel{1}(end-3:end-1));
    end
    
    %Merge labels into two conditions
    related = [111, 112, 211, 212]; %Related labels
    unrelated = [121, 122, 221, 222]; %Unrelated labels

    label_binary = [];
    for li = 1:length(labels) %Iterate through trial labels
        current_label = []; %Defaults to unrelated
        for index = 1:length(related) %Iterate through label IDs
            if labels(li) == related(index) %If related label
                current_label = 1; %If related is found, change it
                break
            elseif labels(li) == unrelated(index) %If unrelated label
                current_label = 0; %If unrelated is found, change it
                break
            end
        end
        label_binary(end+1) = current_label; %Add label to array
    end
    
    %Extract specific electrode
    electrode = 14; %CPz
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
save_filename = '../Full Datasets/erpcore_N400_full.csv'
writetable(EEG_table,save_filename) %Save to csv