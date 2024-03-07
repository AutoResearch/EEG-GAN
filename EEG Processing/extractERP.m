%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Written by Chad C. Williams                                             %
% www.chadcwilliams.com                                                   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%User inputs
downsampledLength = 100; %Datapoints to downsample data to
electrode = 'FCz';

%% This first loop will extract the data for each participant and save data as a participant specific csv file

%Determine all files
filenames = dir('*.mat');

%Iterate through all participants
for participant = 1:length(filenames)
    %Display participant number
    participant

    %Load EEG Data
    EEGdata = load(filenames(participant).name);

    %Determine the index where the conditions switch (S110 = Win, S111 = Lose)
    %The data are pre-organized so all wins come first and all losses come second
    for trialIdx = 1:size(EEGdata.EEG.data,3)
        if strmatch(EEGdata.EEG.epoch(trialIdx).eventtype,'S111')
            break
        end
    end

    %Find electrode index
    for electrodeIndex = 1:length(EEGdata.EEG.chanlocs)
        if strcmp(EEGdata.EEG.chanlocs(electrodeIndex).labels,electrode)
            break
        end
    end

    %Setup variables
    trialCounter = 1;
    firstCondition = 1;
    thisData = zeros(1,downsampledLength+3);
    dataIndex = 1;

    %Iterate through trials
    for trial = 1:size(EEGdata.EEG.data,3)

        %Record data
        thisData(dataIndex,1) = participant;
        thisData(dataIndex,2) = trial>=trialIdx;
        thisData(dataIndex,3) = trialCounter;

        %Downsample and Record EEG
        downsampledEEG = downsample(EEGdata.EEG.data(electrodeIndex,:,trial),length(EEGdata.EEG.data(electrodeIndex,:,trial))/downsampledLength);
        thisData(dataIndex,4:end) = downsampledEEG;

        %Increase counts
        dataIndex = dataIndex + 1;
        trialCounter = trialCounter + 1;

        %Reset trial counter 
        if trialCounter >= trialIdx & firstCondition
            trialCounter = 1;
            firstCondition = 0;
        end
    end

    %Convert to a table for saving
    thisDataTable = array2table(thisData);

    %Create header and apply to table names
    tableNames = ["ParticipantID", "Condition", "Trial"];
    for timeIndex = 1:downsampledLength
        tableNames(end+1) = strcat("Time", num2str(timeIndex));
    end
    thisDataTable.Properties.VariableNames = tableNames;

    %Determine participant number
    if participant < 10
        pNum = ['000',num2str(participant)];
    elseif participant < 100
        pNum = ['00',num2str(participant)];
    else
        pNum = ['0',num2str(participant)];
    end

    %Save each participant data as a csv file (these will later be deleted)
    writetable(thisDataTable,['ganTrialElectrodeERP_', pNum,'.csv'],'Delimiter',',');
end

%% We will next combine all participant level data into a single file

%Determine all files
filenames = dir('ganTrialElectrode*');

%Iterate though all participants
for filenameIndex = 1:length(filenames)

    %Displace file number
    filenameIndex

    %Load data
    participantEEG = readmatrix(filenames(filenameIndex).name);

    %Record data - if it is the first participant, create the dataframe, otherwise extend it
    if filenameIndex == 1
        allData = participantEEG;
    else
        allData(end+1:end+size(participantEEG,1),:) = participantEEG;
    end

    %Determine participant number
    if filenameIndex < 10
        pNum = ['000',num2str(filenameIndex)];
    elseif filenameIndex < 100
        pNum = ['00',num2str(filenameIndex)];
    else
        pNum = ['0',num2str(filenameIndex)];
    end

    %Delete the participant level csv file
    delete(['ganTrialElectrodeERP_', pNum,'.csv']);
end

%Convert to a table for saving
allDataTable = array2table(allData);

%Create header and apply to table names
tableNames = ["ParticipantID", "Condition", "Trial"];
for timeIndex = 1:size(allData,2)-3
    tableNames(end+1) = strcat("Time", num2str(timeIndex));
end
allDataTable.Properties.VariableNames = tableNames;

%Save data as a csv file
writetable(allDataTable,strcat('../Data/Full Datasets/ganTrialERP_len',num2str(downsampledLength),'.csv'),'Delimiter',',');
%zip(strcat('ganTrialElectrodeERP_len',num2str(downsampledLength)),strcat('ganTrialElectrodeERP_len',num2str(downsampledLength),'.csv'))