%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Written by Chad C. Williams                                             %
% www.chadcwilliams.com                                                   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%User inputs
downsampledLength = 100; %Datapoints to downsample data to {120}
numberParticipants = 500; %Number of participants used to reduce file size for testing
training = false; 
%electrodes = {'F7','T7','P7','F3','C3','P3','Fz','FCz','Cz','CPz','Pz','POz','P4','C4','F4','P8','T8','F8'};
%electrodes = {'F3', 'FCz', 'F4', 'C3', 'C4', 'P3', 'POz', 'P5'};
%electrodes ={'FCz', 'POz'};
electrodes = {'FCz'}
validation_participants = [];
testing_participants = [];

if training
    numberParticipants = 100;
    training_participants = [1, 2, 11, 12, 20, 23, 38, 43, 46, 49, 52, 53, 58, 60, 65, 74, 80, 81, 82, 93, 97, 106, 108, 109, 120, 127, 130, 131, 143, 158, 164, 166, 172, 173, 177, 194, 196, 197, 209, 212, 218, 222, 226, 230, 235, 238, 242, 244, 245, 251, 258, 260, 266, 273, 274, 277, 278, 281, 283, 285, 287, 291, 293, 302, 303, 309, 312, 315, 332, 339, 343, 351, 355, 356, 360, 372, 374, 383, 395, 398, 402, 403, 412, 422, 423, 427, 434, 435, 437, 439, 445, 459, 462, 466, 470, 476, 477, 482, 490, 493];
end

filenames = dir('*.mat');
if strcmp(numberParticipants,'full')
    numberParticipants = length(filenames);
end

for participant = 1:numberParticipants
    
    if training
        participant_fn = append('RewardProcessing_S2Final_', sprintf('%03d',training_participants(participant)), '.mat');
    else
        participant_fn = append('RewardProcessing_S2Final_', sprintf('%03d',participant), '.mat');
    end
    disp(participant_fn)
    EEGdata = load(participant_fn);
    
    for trialIdx = 1:size(EEGdata.EEG.data,3)
        if strmatch(EEGdata.EEG.epoch(trialIdx).eventtype,'S111')
            break
        end
    end

    trialCounter = 1;
    firstCondition = 1;
    thisData = zeros(1,downsampledLength+4);
    dataIndex = 1;
    for trial = 1:size(EEGdata.EEG.data,3)
        for electrode = 1:length(electrodes)

            %Find electrode index
            for electrodeIndex = 1:length(EEGdata.EEG.chanlocs)
                found = 0;
                if strcmp(EEGdata.EEG.chanlocs(electrodeIndex).labels,electrodes(electrode))
                    found = 1;
                    break
                end
            end

            %Record data
            thisData(dataIndex,1) = participant;
            thisData(dataIndex,2) = trial>=trialIdx;
            thisData(dataIndex,3) = trialCounter;
            thisData(dataIndex,4) = electrode;
            downsampledEEG = downsample(EEGdata.EEG.data(electrodeIndex,:,trial),length(EEGdata.EEG.data(electrodeIndex,:,trial))/downsampledLength);
            thisData(dataIndex,5:end) = downsampledEEG;
            dataIndex = dataIndex + 1;
        end
        trialCounter = trialCounter + 1;
    
        if trialCounter >= trialIdx & firstCondition
            trialCounter = 1;
            firstCondition = 0;
        end
    end
    thisDataTable = array2table(thisData);
    tableNames = ["ParticipantID", "Condition", "Trial", "Electrode"];
    for timeIndex = 1:downsampledLength
        tableNames(end+1) = strcat("Time", num2str(timeIndex));
    end
    thisDataTable.Properties.VariableNames = tableNames;
    if participant < 10
        pNum = ['000',num2str(participant)];
    elseif participant < 100
        pNum = ['00',num2str(participant)];
    else
        pNum = ['0',num2str(participant)];
    end
    
    writetable(thisDataTable,['ganTrialElectrodeERP_', pNum,'.csv'],'Delimiter',',');
end

%Combine data
filenames = dir('ganTrialElectrodeERP_0*');
allData = zeros(1,downsampledLength+4);
for filenameIndex = 1:numberParticipants
    disp(filenameIndex)
    participantEEG = readmatrix(filenames(filenameIndex).name);
    if filenameIndex == 1
        allData = participantEEG;
    else
        allData(end+1:end+size(participantEEG,1),:) = participantEEG;
    end
    if filenameIndex < 10
        pNum = ['000',num2str(filenameIndex)];
    elseif filenameIndex < 100
        pNum = ['00',num2str(filenameIndex)];
    else
        pNum = ['0',num2str(filenameIndex)];
    end
    delete(['ganTrialElectrodeERP_', pNum,'.csv']);
end

allDataTable = array2table(allData);
tableNames = ["ParticipantID", "Condition", "Trial", "Electrode"];
for timeIndex = 1:size(allData,2)-4
    tableNames(end+1) = strcat("Time", num2str(timeIndex));
end
allDataTable.Properties.VariableNames = tableNames;
writetable(allDataTable,strcat('ganTrialElectrodeERP_p', num2str(numberParticipants),'_e',num2str(length(electrodes)),'_len',num2str(downsampledLength),'.csv'),'Delimiter',',');
%zip(strcat('ganTrialElectrodeERP_len',num2str(downsampledLength)),strcat('ganTrialElectrodeERP_len',num2str(downsampledLength),'.csv'))