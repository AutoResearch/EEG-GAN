%% SCRIPT 3 (plots)

clc;
clear;
close all;

base_path = '\\psyger-stor02.d.uzh.ch\methlab\Neurometric\Anti_newest\data\right\';
subjects = dir(base_path);
subjects = subjects([subjects.isdir] & ~ismember({subjects.name}, {'.', '..'}));

all_pro_ERPs = [];
all_anti_ERPs = [];

for i = 1:length(subjects)
    subject_id = subjects(i).name;
    subject_folder = fullfile(base_path, subject_id);
    mat_file_path = fullfile(subject_folder, [subject_id '_prep2stimEEG.mat']);
    
    if exist(mat_file_path, 'file')
        load(mat_file_path, 'prep2stimEEG');  %  the preprocessed EEG data, downsampled data
        
        pro_indices = [];
        anti_indices = [];

        % Loop through the event structure to find indices for '12' and
        % '13' (left right and leftanti)
        for j = 1:length(prep2stimEEG.event)
            if strcmp(prep2stimEEG.event(j).type, '12')
                pro_indices(end + 1) = ceil(j / 2);  % Calculate epoch indexes for pro right
            elseif strcmp(prep2stimEEG.event(j).type, '13')
                anti_indices(end + 1) = ceil(j / 2);  % Calculate epoch index for anti left
            end
        end
        
        % Extract EEG data for pro and anti stimuli using the calculated indices
        pro_EEG_data = prep2stimEEG.data(:, :, pro_indices);
        anti_EEG_data = prep2stimEEG.data(:, :, anti_indices);

        % Average the EEG data across trials for pro and anti stimuli
        mean_pro_EEG = mean(pro_EEG_data, 3);
        mean_anti_EEG = mean(anti_EEG_data, 3);
        
        % Concatenate the subject's average ERP to the overall ERP matrices
        if isempty(all_pro_ERPs)
            all_pro_ERPs = mean_pro_EEG;
            all_anti_ERPs = mean_anti_EEG;
        else
            all_pro_ERPs = cat(3, all_pro_ERPs, mean_pro_EEG);
            all_anti_ERPs = cat(3, all_anti_ERPs, mean_anti_EEG);
        end
    else
        fprintf('File not found: %s\n', mat_file_path);
    end
end

% Average across subjects
mean_pro_ERP = mean(all_pro_ERPs, 3);
mean_anti_ERP = mean(all_anti_ERPs, 3);

% Time vector for the downsampled data
time_vector = linspace(-0.2, 0.8, 125);

%% Plot ERP
figure;
hold on;
plot(time_vector, mean_pro_ERP, 'b', 'LineWidth', 2);  % Pro in blue
plot(time_vector, mean_anti_ERP, 'r', 'LineWidth', 2);  % Anti in red
xline(0, '--', 'LineWidth', 2);  % Vertical dashed line at timepoint 0 where the stim is
title('Comparison of ERPs for Pro [right] and Anti [left], Stimulus locked');
xlabel('Time (s)');
ylabel('Amplitude (uV)');
legend('Pro Stimuli', 'Anti Stimuli');
grid on;
hold off;