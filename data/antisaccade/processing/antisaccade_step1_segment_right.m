%% SCRIPT 1 - > whole pipeline on preprocessed dataset

clc
clear
cd('\\psyger-stor02.d.uzh.ch\methlab\Neurometric\Antisaccades\code\eeglab14_1_2b')
eeglab;
close all

x = dir('\\psyger-stor02.d.uzh.ch\methlab\Neurometric\Anti_new\newprepdata_withopticat\');
subjects = {x.name};
clear x

study_path = '\\psyger-stor02.d.uzh.ch\methlab\Neurometric\Anti_new\';
analysis_path = [study_path 'scipts\data_processing_after_automagic\'];
cd(analysis_path)

subjects_exclude = [1,2];
subjects = subjects(~ismember(1:numel(subjects),subjects_exclude));
nsubjects = numel(subjects);

%these tables are for age info only:
addpath \\psyger-stor02.d.uzh.ch\methlab\Neurometric\Anti_new\behav_tables_from_tzvetan
OLD = readtable('old.xlsx');%,'Range','A1:A100');
YNG = readtable('yng.xlsx');%,'Range', 'A1:A104');

%%
for subj = 1:nsubjects


    datapath = strcat('\\psyger-stor02.d.uzh.ch\methlab\Neurometric\Anti_new\newprepdata_withopticat\',subjects{subj});

    cd (datapath)

    if exist(strcat('gip_',subjects{subj},'_AS_EEG.mat')) > 0
        datafile= strcat('gip_',subjects{subj},'_AS_EEG.mat');
        load (datafile)
    elseif exist(strcat('gp_',subjects{subj},'_AS_EEG.mat')) > 0
        datafile= strcat('gp_',subjects{subj},'_AS_EEG.mat');
        load (datafile)
    elseif exist(strcat('oip_',subjects{subj},'_AS_EEG.mat')) > 0
        datafile= strcat('oip_',subjects{subj},'_AS_EEG.mat');
        load (datafile)
    elseif exist(strcat('op_',subjects{subj},'_AS_EEG.mat')) > 0
        datafile= strcat('op_',subjects{subj},'_AS_EEG.mat');
        load (datafile)
    else
        continue
    end


    %% REDUCE TO 105 electrodes

    tbl_channels = struct2table(EEG.chanlocs);
    el_excl = {'E48' 'E49' 'E56' 'E63' 'E68' 'E73' 'E81' 'E88' 'E94' 'E99' 'E107' 'E113' 'E119' 'E1' 'E8' 'E14' 'E17' 'E21' 'E25' 'E32' 'E125' 'E126' 'E127' 'E128'};


    tbl_excl = table(el_excl');
    tbl_excl.Properties.VariableNames = {'labels'};


    ind_excl = [];
    for i = 1:size(tbl_excl,1)
        ind_excl(end+1,1) = find(strcmp(tbl_excl.labels(i),[tbl_channels.labels]));
    end
    ind_excl';


    EEG = pop_select(EEG,'nochannel',ind_excl );

    %% Re-reference to average reference
    EEG = pop_reref(EEG,[]); %thats fine, its done AFTER AUTOMAGIC


    %% triggers renaming

    countblocks = 1;
    for e = 1:length(EEG.event)
        if strcmp(EEG.event(e).type, 'boundary')
            countblocks = countblocks + 1;
            continue;
        end
        if countblocks == 2 || countblocks == 3 || countblocks == 4 % antisaccade blocks
            if strcmp(EEG.event(e).type,'10  ') % change 10 to 12 for AS
                EEG.event(e).type = '13';
            elseif strcmp(EEG.event(e).type,'11  ')
                EEG.event(e).type = '14'; % change 11 to 13 for AS
            end

            if strcmp(EEG.event(e).type,'40  ')
                EEG.event(e).type = '41  ';
            end

        end

        if countblocks == 1 || countblocks == 5 %prosaccade blocks
            if strcmp(EEG.event(e).type,'10  ') % change numbers to words
                EEG.event(e).type = '11';
            elseif strcmp(EEG.event(e).type,'11  ')
                EEG.event(e).type = '12'; % change numbers to words
            end
        end


    end

    EEG.event(strcmp('boundary',{EEG.event.type})) = [];
    rmEventsIx = strcmp('L_fixation',{EEG.event.type});
    rmEv =  EEG.event(rmEventsIx);
    EEG.event(rmEventsIx) = [];
    EEG.event(1).dir = []; %left or right
    EEG.event(1).cond = [];%pro or anti

    %% rename EEG.event.type

    previous = '';
    for e = 1:length(EEG.event)
        if strcmp(EEG.event(e).type, 'L_saccade')
            if strcmp(previous, '11')
                EEG.event(e).type = '21';
                EEG.event(e).cond = 'pro';
                EEG.event(e).dir = 'left';

                %pro left
            elseif strcmp(previous, '12')
                EEG.event(e).type = '22';
                EEG.event(e).cond = 'pro';
                EEG.event(e).dir = 'right';

            elseif strcmp(previous, '13')
                EEG.event(e).type = '23';
                EEG.event(e).cond = 'anti';
                EEG.event(e).dir = 'left';

            elseif strcmp(previous, '14')
                EEG.event(e).type = '24';
                EEG.event(e).cond = 'anti';
                EEG.event(e).dir = 'right';

            else
                EEG.event(e).type = 'invalid';
            end

        end
        if ~strcmp(EEG.event(e).type, 'L_fixation') ...
                && ~strcmp(EEG.event(e).type, 'L_blink')
            previous = EEG.event(e).type;
        end
    end

    %% remove everything from EEG.event which is not saccade or trigger sent by me from the stimulus pc

    tmpinv=strcmp({EEG.event.type}, 'invalid') | strcmp({EEG.event.type}, 'L_blink') ;
    EEG.event(tmpinv)=[];

    %% remove trigger sent to start the fixation periods
    tmpinv=strcmp({EEG.event.type}, '40  ') | strcmp({EEG.event.type}, '41  ') ;
    EEG.event(tmpinv)=[];

    %% remove trigger start blocks
    tmpinv=strcmp({EEG.event.type}, '20  ') | strcmp({EEG.event.type}, '30  ') ;
    EEG.event(tmpinv)=[];

    %% trigger start exp + end exp
    tmpinv=strcmp({EEG.event.type}, '94  ') | strcmp({EEG.event.type}, '50  ') ;
    EEG.event(tmpinv)=[];

    %% delete cues where there was no saccade afterwards
    tmperrcue10 = find(strcmp({EEG.event.type}, '11'));
    for i = 1:length(tmperrcue10)
        pos = tmperrcue10(i);
        if pos == length(EEG.event) || ~(strcmp(EEG.event(pos+1).type, '21') || strcmp(EEG.event(pos+1).type, '22'))
            EEG.event(pos).type = 'missingsacc'; %cue
        end
    end

    tmperrcue11 = find(strcmp({EEG.event.type}, '12'));
    for i = 1:length(tmperrcue11)
        pos = tmperrcue11(i);
        if pos == length(EEG.event) || ~(strcmp(EEG.event(pos+1).type, '21') || strcmp(EEG.event(pos+1).type, '22'))
            EEG.event(pos).type = 'missingsacc'; %cue
        end
    end

    tmperrcue12 = find(strcmp({EEG.event.type}, '13'));
    for i = 1:length(tmperrcue12)
        pos = tmperrcue12(i);
        if pos == length(EEG.event) || ~(strcmp(EEG.event(pos+1).type, '23') || strcmp(EEG.event(pos+1).type, '24'))
            EEG.event(pos).type = 'missingsacc'; %cue
        end
    end

    tmperrcue13 = find(strcmp({EEG.event.type}, '14'));
    for i = 1:length(tmperrcue13)
        pos = tmperrcue13(i);
        if pos == length(EEG.event) || ~(strcmp(EEG.event(pos+1).type, '23') || strcmp(EEG.event(pos+1).type, '24'))
            EEG.event(pos).type = 'missingsacc'; %cue
        end
    end

    % Remove events marked as 'missingsacc'
    tmpinv = strcmp({EEG.event.type}, 'missingsacc');
    EEG.event(tmpinv) = [];



    %% delete sacc where there was no cue before

    tmperrcue10=  find(strcmp({EEG.event.type}, '21')) ;
    for i=1:length(tmperrcue10)
        pos = tmperrcue10(i);
        if ~ (strcmp(EEG.event(pos-1).type , '11'))

            EEG.event(pos).type='missingcue'; %cue
        end
    end

    %%11
    tmperrcue11 =   find(strcmp({EEG.event.type}, '22'))    ;
    for i=1:length(tmperrcue11)
        pos = tmperrcue11(i);
        if ~ (strcmp(EEG.event(pos-1).type , '12'))

            EEG.event(pos).type='missingcue'; %cue
        end
    end


    tmperrcue12=  find(strcmp({EEG.event.type}, '23')) ;
    for i=1:length(tmperrcue12)
        pos = tmperrcue12(i);
        if ~ (strcmp(EEG.event(pos-1).type , '13'))

            EEG.event(pos).type='missingcue'; %cue
        end
    end

    %%11
    tmperrcue13 =   find(strcmp({EEG.event.type}, '24'))    ;
    for i=1:length(tmperrcue13)
        pos = tmperrcue13(i);
        if ~ (strcmp(EEG.event(pos-1).type , '14'))

            EEG.event(pos).type='missingcue'; %cue
        end
    end

    tmpinv=strcmp({EEG.event.type}, 'missingcue') ;
    EEG.event(tmpinv)=[];


    %% renaming errors


    for e = 1:length(EEG.event)

        if strcmp(EEG.event(e).type, '23') && ( EEG.event(e).sac_startpos_x > EEG.event(e).sac_endpos_x)
            EEG.event(e).accuracy = 'error_anti_sacc';
            EEG.event(e-1).accuracy = 'error_anti_cue';

        elseif strcmp(EEG.event(e).type, '23') && (EEG.event(e).sac_startpos_x < EEG.event(e).sac_endpos_x)
            EEG.event(e).accuracy = 'correct_anti_sacc';
            EEG.event(e-1).accuracy = 'correct_anti_cue';

        elseif strcmp(EEG.event(e).type, '24') && (EEG.event(e).sac_startpos_x <EEG.event(e).sac_endpos_x)
            EEG.event(e).accuracy = 'error_anti_sacc';
            EEG.event(e-1).accuracy = 'error_anti_cue';

        elseif strcmp(EEG.event(e).type, '24') && (EEG.event(e).sac_startpos_x >EEG.event(e).sac_endpos_x)
            EEG.event(e).accuracy = 'correct_anti_sacc';
            EEG.event(e-1).accuracy = 'correct_anti_cue';

        elseif strcmp(EEG.event(e).type, '21') && ( EEG.event(e).sac_startpos_x < EEG.event(e).sac_endpos_x)
            EEG.event(e).accuracy = 'error_pro_sacc';
            EEG.event(e-1).accuracy = 'error_pro_cue';

        elseif strcmp(EEG.event(e).type, '21') && (EEG.event(e).sac_startpos_x > EEG.event(e).sac_endpos_x)
            EEG.event(e).accuracy = 'correct_pro_sacc';
            EEG.event(e-1).accuracy = 'correct_pro_cue';

        elseif strcmp(EEG.event(e).type, '22') && (EEG.event(e).sac_startpos_x >EEG.event(e).sac_endpos_x)
            EEG.event(e).accuracy = 'error_pro_sacc';
            EEG.event(e-1).accuracy = 'error_pro_cue';

        elseif strcmp(EEG.event(e).type, '22') && (EEG.event(e).sac_startpos_x <EEG.event(e).sac_endpos_x)
            EEG.event(e).accuracy = 'correct_pro_sacc';
            EEG.event(e-1).accuracy = 'correct_pro_cue';


        else
            EEG.event(e).accuracy = 'NA';
        end
    end


    %% CALCULATE RT FOR EACH TRIALS


    for e = 1:length(EEG.event)

        if strcmp(EEG.event(e).type, '23')
            EEG.event(e).rt =   (EEG.event(e).latency - EEG.event(e-1).latency)*2;%for sacc
            EEG.event(e-1).rt = (EEG.event(e).latency - EEG.event(e-1).latency)*2; %for the "pair"cue


        elseif strcmp(EEG.event(e).type, '24')
            EEG.event(e).rt =   (EEG.event(e).latency - EEG.event(e-1).latency)*2;
            EEG.event(e-1).rt = (EEG.event(e).latency - EEG.event(e-1).latency)*2;


        elseif strcmp(EEG.event(e).type, '21')
            EEG.event(e).rt =   (EEG.event(e).latency - EEG.event(e-1).latency)*2;
            EEG.event(e-1).rt = (EEG.event(e).latency - EEG.event(e-1).latency)*2;


        elseif strcmp(EEG.event(e).type, '22')
            EEG.event(e).rt =   (EEG.event(e).latency - EEG.event(e-1).latency)*2;
            EEG.event(e-1).rt = (EEG.event(e).latency - EEG.event(e-1).latency)*2;


        else
            EEG.event(e).rt = 'NA';
        end
    end

    %% amplitude too small

    tmperrsacc6=find(strcmp({EEG.event.type}, '22') ...
        & [EEG.event.sac_amplitude]<1.5);
    tmperrsacc7=find(strcmp({EEG.event.type}, '21') ...
        & [EEG.event.sac_amplitude]<1.5);
    tmperrsacc8=find(strcmp({EEG.event.type}, '23') ...
        & [EEG.event.sac_amplitude]<1.5);
    tmperrsacc9=find(strcmp({EEG.event.type}, '24') ...
        & [EEG.event.sac_amplitude]<1.5);
    tmperr69=[tmperrsacc6 (tmperrsacc6-1) tmperrsacc7 (tmperrsacc7-1) tmperrsacc8 (tmperrsacc8-1) tmperrsacc9 (tmperrsacc9-1)];
    EEG.event(tmperr69)=[];

    clear tmperrsacc1 tmperrsacc2 tmperrsacc3 tmperrsacc4 tmperrsacc6 tmperrsacc7 tmperrsacc8 tmperrsacc9


    %% delete saccades and cues when the saccade comes faster than 100ms after cue

    tmpevent=length(EEG.event);
    saccpro=find(strcmp({EEG.event.type},'22')==1 | strcmp({EEG.event.type},'21')==1); % find rows where there is a saccade
    saccanti=find(strcmp({EEG.event.type},'24')==1 | strcmp({EEG.event.type},'23')==1);%find rows where there is a saccade

    for b=1:size(saccpro,2)

        if (EEG.event(saccpro(1,b)).latency-EEG.event(saccpro(1,b)-1).latency)<100 %50 because 100ms
            EEG.event(saccpro(b)).type='too_fast'; %saccade
            EEG.event(saccpro(b)-1).type = 'too_fast'; %cue
        end
    end

    for b=1:size(saccanti,2)

        if (EEG.event(saccanti(b)).latency-EEG.event(saccanti(1,b)-1).latency)<100
            EEG.event(saccanti(b)-1).type ='too_fast';
            EEG.event(saccanti(b)).type ='too_fast';
        end

    end

    tmpinv=strcmp({EEG.event.type}, 'too_fast') ;
    EEG.event(tmpinv)=[];
    clear tmpinv

    %% delete saccades and cues when the saccade comes slower than 800ms after cue

    tmpevent=length(EEG.event);
    saccpro=find(strcmp({EEG.event.type},'22')==1 | strcmp({EEG.event.type},'21')==1); % find rows where there is a saccade
    saccanti=find(strcmp({EEG.event.type},'24')==1 | strcmp({EEG.event.type},'23')==1);%find rows where there is a saccade

    for b=1:size(saccpro,2)

        if (EEG.event(saccpro(1,b)).latency-EEG.event(saccpro(1,b)-1).latency)>400 %400 because 800ms
            EEG.event(saccpro(b)).type='too_slow'; %saccade
            EEG.event(saccpro(b)-1).type = 'too_slow'; %cue
        end
    end

    for b=1:size(saccanti,2)

        if (EEG.event(saccanti(b)).latency-EEG.event(saccanti(1,b)-1).latency)>400
            EEG.event(saccanti(b)-1).type ='too_slow';
            EEG.event(saccanti(b)).type ='too_slow';
        end

    end

    tmpinv=strcmp({EEG.event.type}, 'too_slow') ;
    EEG.event(tmpinv)=[];
    clear tmpinv
    %% remove everything except stim-response
    tmpinv=strcmp({EEG.event.rt}, 'NA') ;
    EEG.event(tmpinv)=[];
    clear tmpinv


    %% change for keepimg only right movement
    for e = 1:length(EEG.event)

        if strcmp(EEG.event(e).dir, 'right') &&  strcmp(EEG.event(e).cond, 'pro')
            EEG.event(e).keep = 'yes';
            EEG.event(e-1).keep = 'yes';

        elseif strcmp(EEG.event(e).dir, 'left') &&  strcmp(EEG.event(e).cond, 'anti')
            EEG.event(e).keep = 'yes';
            EEG.event(e-1).keep = 'yes';

        elseif strcmp(EEG.event(e).dir, 'right') &&  strcmp(EEG.event(e).cond, 'anti')
            EEG.event(e).keep = 'no';
            EEG.event(e-1).keep = 'no';

        elseif strcmp(EEG.event(e).dir, 'left') &&  strcmp(EEG.event(e).cond, 'pro')
            EEG.event(e).keep = 'no';
            EEG.event(e-1).keep = 'no';

        else
            EEG.event(e).keep = 'problem';
        end
    end




    tmpinv=strcmp({EEG.event.keep}, 'no') ;
    EEG.event(tmpinv)=[];
    clear tmpinv

    %%

    for e = 1:length(EEG.event)

        if strcmp(EEG.event(e).type, '23')
            EEG.event(e).rt =   (EEG.event(e).latency - EEG.event(e-1).latency)*2;%for sacc
            EEG.event(e-1).rt = (EEG.event(e).latency - EEG.event(e-1).latency)*2; %for the "pair"cue


        elseif strcmp(EEG.event(e).type, '24')
            EEG.event(e).rt =   (EEG.event(e).latency - EEG.event(e-1).latency)*2;
            EEG.event(e-1).rt = (EEG.event(e).latency - EEG.event(e-1).latency)*2;


        elseif strcmp(EEG.event(e).type, '21')
            EEG.event(e).rt =   (EEG.event(e).latency - EEG.event(e-1).latency)*2;
            EEG.event(e-1).rt = (EEG.event(e).latency - EEG.event(e-1).latency)*2;


        elseif strcmp(EEG.event(e).type, '22')
            EEG.event(e).rt =   (EEG.event(e).latency - EEG.event(e-1).latency)*2;
            EEG.event(e-1).rt = (EEG.event(e).latency - EEG.event(e-1).latency)*2;


        else
            EEG.event(e).rt = 'NA';
        end
    end


    %% add trial number

    n =length(EEG.event)/2;
    a = reshape( repmat( 1:n, 2,1 ), 1, [] );
    a = a';

    for i =1:length(EEG.event)
        EEG.event(i).trial = a(i);
    end

    %% add age info
    id = regexp(EEG.comments(1,:), '.*All_Subjects[\/\\](?<ID>.*)[\/\\].*', 'names').ID;
    is_old = any(ismember(OLD.Subjects, id));
    for e=1:size(EEG.event,2)
        EEG.event(e).age = is_old;
    end


    %% remove trials that are erroneous
    tmpinv=strcmp({EEG.event.accuracy}, 'error_pro_cue');
    EEG.event(tmpinv)=[];

    tmpinv=strcmp({EEG.event.accuracy}, 'error_pro_sacc');
    EEG.event(tmpinv)=[];

    tmpinv=strcmp({EEG.event.accuracy}, 'error_anti_cue');
    EEG.event(tmpinv)=[];

    tmpinv=strcmp({EEG.event.accuracy}, 'error_anti_sacc');
    EEG.event(tmpinv)=[];

    %% new version of baseline
    baseline_samples = 100; % 200 ms

    % Epochs range from -0.2 to 0.8 seconds - total 1 sec
    epoch_range = [-0.2 0.8];

    stimEEG = pop_epoch(EEG, {'12', '11', '13', '14'}, epoch_range, 'epochinfo', 'yes');

    % Calculate baseline from stimulus epochs
    % The baseline starts from the beginning of the epoch (-0.2s), which is the start of the data array.
    baseline_start_idx = 1;  % Baseline starts from the very first sample
    baseline_end_idx = baseline_start_idx + baseline_samples - 1; % End index after 100 samples
    baseline_values = mean(stimEEG.data(:, baseline_start_idx:baseline_end_idx, :), 2);

    % Apply the calculated baseline to stimEEG
    for i = 1:size(stimEEG.data, 3)
        for j = 1:size(stimEEG.data, 1)
            stimEEG.data(j, :, i) = stimEEG.data(j, :, i) - baseline_values(j, 1, i);
        end
    end


    %% important:

    %check ifcorrect mapping between the number of stimulus events and the EEG data slices.
    if size(stimEEG.data, 3) ~= (numel(stimEEG.event))/2
        error('Mismatch in number of stimulus events and available EEG data slices.');
    end


    %% save
    mkdir('\\psyger-stor02.d.uzh.ch\methlab\Neurometric\Anti_newest\data\right\', id)
    save(['\\psyger-stor02.d.uzh.ch\methlab\Neurometric\Anti_newest\data\right\' id '\' id '_stimlockedEEG.mat'], 'stimEEG');

end





