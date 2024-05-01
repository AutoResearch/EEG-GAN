%% SCRIPT 4 (preparation for EEG GAN format)

clc;
clear;

base_path = '\\psyger-stor02.d.uzh.ch\methlab\Neurometric\Anti_newest\data\right\';
subjects = dir(base_path);
subjects = subjects([subjects.isdir] & ~ismember({subjects.name}, {'.', '..'}));
save_path = '\\psyger-stor02.d.uzh.ch\methlab\Neurometric\Anti_newest\data';

all_subjects_data = [];

% electrode of interest is '1' (this is the mean of 6 electrodes as
% defined in the script 2)
electrode = 1;

subject_id_map = containers.Map('KeyType', 'char', 'ValueType', 'int32');
unique_subject_id = 0;

for i = 1:length(subjects) 
    subject_identifier = subjects(i).name;  
    if ~isKey(subject_id_map, subject_identifier)
        unique_subject_id = unique_subject_id + 1;
        subject_id_map(subject_identifier) = unique_subject_id;
    end
    subject_id = subject_id_map(subject_identifier);  % Get numeric ID from map
    
    subject_folder = fullfile(base_path, subject_identifier);
    mat_file_path = fullfile(subject_folder, [subject_identifier '_prep2stimEEG.mat']);
    
    if exist(mat_file_path, 'file')
        load(mat_file_path, 'prep2stimEEG');
        
        n_epochs = size(prep2stimEEG.data, 3);
        conditions = zeros(n_epochs, 1);  % Initialize condition vector
        
        % Iterate over epochs
        for j = 1:2:n_epochs*2
            type1 = str2double(prep2stimEEG.event(j).type);
            type2 = str2double(prep2stimEEG.event(j+1).type);
            
            if type1 == 12 && type2 == 22
                conditions((j+1)/2) = 0;  % Condition 'pro' right
            elseif type1 == 13 && type2 == 23
                conditions((j+1)/2) = 1;  % Condition 'anti' left
            end
        end
        
        for k = 1:n_epochs
            time_points = double(squeeze(prep2stimEEG.data(1, :, k)));  
            trial_data = [double(subject_id), double(conditions(k)), double(electrode), double(k), time_points];  % Convert all elements to double
            all_subjects_data = [all_subjects_data; trial_data];  % Append to the main table
        end
        
    else
        fprintf('File not found: %s\n', mat_file_path);
    end
end



time_var_names = arrayfun(@(x) ['Time', num2str(x)], 1:125, 'UniformOutput', false);
var_names = [{'Participant_ID', 'Condition', 'Electrode', 'Trial'}, time_var_names];
all_subjects_table = array2table(all_subjects_data, 'VariableNames', var_names);

csv_path = fullfile(save_path, 'right_data_full.csv');
writetable(all_subjects_table, csv_path);

