%% SCRIPT 2 (averaging electrodes and downsampling)

clc;
clear;

base_path = '\\psyger-stor02.d.uzh.ch\methlab\Neurometric\Anti_newest\data\right\';
subjects = dir(base_path);
subjects = subjects([subjects.isdir] & ~ismember({subjects.name}, {'.', '..'}));

%electrodes of interest:
%Cz==105  E7==6 E106==89 E13==11 E6==6 E112==94
electrodes = [105, 6, 89, 11, 6, 94];  

for i = 1:length(subjects)
    subject_id = subjects(i).name;
    subject_folder = fullfile(base_path, subject_id);
    mat_file_path = fullfile(subject_folder, [subject_id '_stimlockedEEG.mat']);
    
    if exist(mat_file_path, 'file')
        load(mat_file_path, 'stimEEG');  
        
        % Extract data for specified electrodes+ compute the mean across these electrodes
        electrode_data = mean(stimEEG.data(electrodes, :, :), 1);  % Mean across the  electrodes
        
        electrode_data = squeeze(electrode_data);  
        electrode_data = reshape(electrode_data, 1, size(electrode_data, 1), size(electrode_data, 2));
        
        % Downsample the mean electrode data from 500 Hz to 125 Hz
        downsampled_data = downsample(electrode_data, 4, 2);  
   
        prep2stimEEG = stimEEG;  
        prep2stimEEG.data = downsampled_data;  
        save_path = fullfile(subject_folder, [subject_id '_prep2stimEEG.mat']);
        save(save_path, 'prep2stimEEG');  
    else
        fprintf('File not found: %s\n', mat_file_path);
    end
end

