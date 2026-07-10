%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%% STUDIE WS19 Mu_Ult2  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 16.11.19 Study to social influence 
% 15.3.20 Script adjusted according to Jojo Rodrigues Preprocessing Chain

% Marker: 
% Task 1: Choose fruit
%  58-59 Picture of the person who want's fruit: 58=male 59=female 
%  67-79: Marker M1: Decision+Feedback: 67=correct smile, 68=correct neutral, 69=correct sad, 77=wrong smile,78=wrong neutral, 79=wrong sad

% Task 2: Ultimatum as proposer: Only 10 trials, no EEG

% Task 3: Ultimatum as Responder 
%  200-205: Marker M2: Offer; 200: 0 Cent; 201: 1 Cents; 202: 2 Cents; 203: 3 Cents; 204: 4 Cents; 205: 5 Cents 
%  210-222: Marker M3: decision+facial_expression 21_: accept; 22_: reject; 2_0: happy; 2_1: neutral; 2_2: sad


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%% I HEADER %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Specify header-informationen 
close all; clear all;

%   cd(PATH_raw); % Get file-names directly from the files
%   file=dir('*.eeg');
%   for s_vp=1:length(file)
%       SUBJECT{1,s_vp}=file(s_vp,1).name(1:end-4);
%   end
SUBJECT = {'test1' 'vp_02' 'vp_03' 'vp_04' 'vp_05' 'vp_06' 'vp_07' 'vp_08' 'vp_09' 'vp_10' 'vp_11' 'vp_12' 'vp_13' 'vp_14' 'vp_15' 'vp_16' 'vp_17' 'vp_18' 'vp_19' 'vp_20' 'vp_21' 'vp_22' 'vp_23' 'vp_24' 'vp_25' 'vp_26' 'vp_27' 'vp_28' 'vp_29' 'vp_30' 'vp_31' 'vp_32' 'vp_33' 'vp_34' 'vp_35' 'vp_36' 'vp_37' 'vp_38' 'vp_39' 'vp_40' 'vp_41' 'vp_42' 'vp_43' 'vp_44' 'vp_45' 'vp_46' 'vp_47' 'vp_48' 'vp_49' 'vp_50' 'vp_51' 'vp_52' 'vp_53' 'vp_54' 'vp_56' 'vp_58' 'vp_59' 'vp_60'};

epoch_length=[-0.8 1.2];

baseline_period=[-200 0];

fil_ter1=[1,39];
fil_ter2=[20];

ALL.PATH_event{1}=['T1_face\']; % T1 facial feedback
ALL.CONDITION{1} = {'S 67' 'S 68' 'S 69' 'S 77' 'S 78' 'S 79'}; 
ALL.PATH_event{2}=['T3_offer\']; % or T3 offer
ALL.CONDITION{2} = {'S200' 'S201' 'S202' 'S203' 'S204' 'S205'}; 
ALL.PATH_event{3}=['T3_face\']; % or T3 facial feedback
ALL.CONDITION{3} = {'S210' 'S211' 'S221' 'S222'}; 

% General folders
PATH_gen=['C:\EEG\WS19_ult2\'];
PATH_raw = [PATH_gen, 'rawfiles\'];
PATH_study = [PATH_gen, 'study\'];

[ALLEEG EEG CURRENTSET ALLCOM] = eeglab;

%% Loop Marker
for s_marker=1:3
clear ERP

CONDITION=ALL.CONDITION{s_marker};

PATH_event=ALL.PATH_event{s_marker};
PATH_epoch = [PATH_gen, PATH_event, 'epoch\'];
PATH_ica = [PATH_gen, PATH_event, 'ica\']; 
PATH_rem = [PATH_gen, PATH_event, 'rem\'];

%% Make directory
mkdir([PATH_gen, 'study\']);
mkdir([PATH_gen, PATH_event, 'epoch\']);
mkdir([PATH_gen, PATH_event, 'ica\']);
mkdir([PATH_gen, PATH_event, 'rem\']);


%% I Open 

for s_vp = 1:length(SUBJECT)
%    if ~isfile([PATH_epoch,'VP', sprintf('%04d',s_vp), '_', SUBJECT{s_vp}, '.set']) % if the data have not yet been filtered
        EEG = pop_loadbv(PATH_raw, [SUBJECT{s_vp}, '.vhdr'], [], [1:32]);
        EEG.setname = SUBJECT{s_vp};
        EEG = eeg_checkset( EEG );
        EEG=pop_chanedit(EEG, 'append',32,'changefield',{33 'labels' 'FCz'},'changefield',{33 'theta' '0'},'changefield',{33 'radius' '0.12662'},'changefield',{33 'X' '0.37527'},'changefield',{33 'Y' '0'},'changefield',{33 'Z' '0.89311'},'changefield',{33 'datachan' 0},'setref',{'1:32' 'FCz'});
        EEG = eeg_checkset( EEG );

        EEG = pop_select( EEG, 'nochannel',{'ECG'}); % get rid of the ECG

%    end

%% II Re-reference to average; Epoch & Baseline

    % Re-Reference to average
    EEG = pop_reref( EEG, [],'refloc',struct('labels',{'FCz'},'sph_radius',{0.96875},'sph_theta',{0},'sph_phi',{67.2087},'theta',{0},'radius',{0.12662},'X',{0.37527},'Y',{0},'Z',{0.89311},'type',{''},'ref',{''},'urchan',{[]},'datachan',{0}));

    EEG = pop_epoch(EEG, CONDITION, epoch_length, 'epochinfo', 'yes'); % Epoch
    EEG = pop_rmbase( EEG, baseline_period); % Baseline correction

    % For later: find trial t of the rspective event for Nr. s_event
    %[mindist, t] = min(abs(ERP.marker_nr(s_vp,:) - EEG.event(1,s_event).urevent));
    

%% III Exclude bad channels
    % Check for bad channels, except FP1/FP2 (which we need for blinks)
    [~, indelec1]  = pop_rejchan(EEG, 'elec', [3:EEG.nbchan],'threshold',3.29,'norm','on','measure','prob'); % 3.29: Tabachnik & Fiedell, 2007 p. 73: Outlier detection criteria
    [~, indelec2]  = pop_rejchan(EEG, 'elec', [3:EEG.nbchan],'threshold',3.29,'norm','on','measure','kurt'); % 3.29: Tabachnik & Fiedell, 2007 p. 73: Outlier detection criteria
    [~, indelec3]  = pop_rejchan(EEG, 'elec', [3:EEG.nbchan],'threshold',3.29,'norm','on','measure','spec','freqrange',[1 125] ); % 3.29: Tabachnik & Fiedell, 2007 p. 73: Outlier detection criteria
    ERP.interpol(s_vp).value=sort(unique([indelec1,indelec2,indelec3])); % save the chanels to be removed for later
    
    if ~isfield(ERP,'chanlocs') % save chanlocs and times (only once)
        ERP.chanlocs=EEG.chanlocs; 
        ERP.times=EEG.times;
    end

    EEG = pop_select( EEG, 'nochannel',ERP.interpol(s_vp).value); % only exclude -> interpolate after ICA
%    EEG = eeg_interp(EEG, ERP.interpol(s_vp).value,'sherical'); % interpolate now

    EEG.setname = [SUBJECT{s_vp}]; % save before ICA to PATH_epoch
    EEG = eeg_checkset( EEG );
    EEG = pop_saveset(EEG, 'filename', ['VP', sprintf('%04d',s_vp), '_', SUBJECT{s_vp}, '.set'], 'filepath', PATH_epoch);
    [ALLEEG EEG CURRENTSET] = eeg_store(ALLEEG, EEG);
    clear indelec1 indelec2 indelec3


%% IV Filter, ICA, Trial Rejection, second ICA 
    % filtern;
    EEG = pop_eegfiltnew (EEG, fil_ter1(1), [], [], 0, [], 0); % high pass
%    EEG = pop_eegfiltnew (EEG, [], fil_ter1(2), [], 0, [], 0); % low pass: no more

    % IVa ICA I for trial rejection;
    EEG = pop_runica(EEG, 'extended',1,'icatype','runica');

    % IVb Trial rejection;
    % Channels not to exclude
    chan_excl={'Fp1' 'Fp2'};
    [~,chan]=setdiff({EEG.chanlocs.labels},chan_excl);
    chan=sort(chan.');

    ERP.anz_trials(s_vp,1) = size(EEG.data,3); % Documentation: Count trials before trial rejection
%    EEG = pop_eegthresh(EEG,1,chan,-300,300,-0.2,1.496,2,0); % w/o FP1 FP2 IO; absolute  (no more)
%    EEG = pop_jointprob(EEG,1,chan,200,3.29,0,0); % in STD; based on ICA (no more) 
    EEG = pop_jointprob(EEG,0,chan,20,3.29,0,0,0,[],0); % in STD; based on ICA 
    EEG = pop_rejkurt(EEG,0,chan,20,3.29,2,0,1,[],0); % in STD; based on ICA 
    EEG = eeg_rejsuperpose( EEG, 0, 1, 1, 1, 1, 1, 1, 1);
    ERP.reject_trial(s_vp) = EEG.reject; % save this also for later to write the ICA solution to the unfiltered data
    EEG = pop_rejepoch( EEG, EEG.reject.rejglobal ,0);
    ERP.anz_trials(s_vp,2) = size(EEG.data,3); % Documentation: Count trials after trial rejection

    % IVb ICA II for artefact rejection;
    EEG = pop_runica(EEG, 'extended',1,'icatype','runica');

    EEG.setname = [SUBJECT{s_vp}];     % save the data with ICA... just in case, because the ICA takes sooo long
    EEG = eeg_checkset( EEG )
    EEG = pop_saveset(EEG, 'filename', ['VP', sprintf('%04d',s_vp), '_', SUBJECT{s_vp}, '.set'], 'filepath', PATH_ica);

    
%% V Sasica, interpolate, filter
    % Needed Plug-ins: Sasica, Mara, Faster, Adjust
    % Create Structure with SASICA options
    SAS_cfg = struct;
    % MARA (enabled)
    SAS_cfg.MARA.enable=1;
    % FASTER (disabled)
    SAS_cfg.FASTER.enable=0;
    SAS_cfg.FASTER.blinkchans=[];
    % ADJUST (enabled)
    SAS_cfg.ADJUST.enable=0;
    % Channel correlations (disabled)
    SAS_cfg.chancorr.enable=0;
    SAS_cfg.chancorr.channames=[];
    SAS_cfg.chancorr.corthresh = 'auto 4';
    % EOG correlations (disabled)
    SAS_cfg.EOGcorr.enable=0;
    SAS_cfg.EOGcorr.Heogchannames=[];
    SAS_cfg.EOGcorr.corthreshH='auto 4';
    SAS_cfg.EOGcorr.Veogchannames=[];
    SAS_cfg.EOGcorr.corthreshV='auto 4';
    % Dipole fit residual variance (disabled)
    SAS_cfg.resvar.enable=0;
    SAS_cfg.resvar.thresh=15;
    % Signal to noise ratio (disabled)
    SAS_cfg.SNR.enable=0;
    SAS_cfg.SNR.snrcut=1;
    SAS_cfg.SNR.snrBL=[-Inf 0];
    SAS_cfg.SNR.snrPOI=[0 Inf]; 
    % Focal Trial Activity (disabled)
    SAS_cfg.trialfoc.enable=0;
    SAS_cfg.trialfoc.focaltrialout='auto';
    % Focal Components (disabled)
    SAS_cfg.focalcomp.enable=0;
    SAS_cfg.focalcomp.focalICAout='auto';
    % Autocorrelation (disabled)
    SAS_cfg.autocorr.enable=0;
    SAS_cfg.autocorr.autocorrint=20;
    SAS_cfg.autocorr.dropautocorr='auto';
    % Options (disabled)
    SAS_cfg.opts.noplot=1; % if = 1, no review and just store results in EEG. / if 0= review, but still it is automatic and will carry on with the script (it does not stop here, really, believe me... )
    SAS_cfg.opts.FontSize=12;
    % Run with these options
    EEG=eeg_SASICA(EEG,SAS_cfg);

    % Save ICA-Information
    ERP.ICA(s_vp).gcompreject = EEG.reject.gcompreject;
    ERP.ICA(s_vp).icawinv = EEG.icawinv;
    ERP.ICA(s_vp).icasphere = EEG.icasphere;
    ERP.ICA(s_vp).icaweights = EEG.icaweights;
    ERP.ICA(s_vp).icachansind = EEG.icachansind;

    % reload the unfiltered data
    EEG = pop_loadset('filename',['VP', sprintf('%04d',s_vp), '_', SUBJECT{s_vp}, '.set'], 'filepath',PATH_epoch);
    EEG = eeg_checkset( EEG )

    % Reject trials again, based on former ICA-detection
    EEG.reject = ERP.reject_trial(s_vp); % write former info to EEG
    EEG = pop_rejepoch( EEG, EEG.reject.rejglobal ,0);

    % Apply the ICA solution to the unfiltered EEG data        
    EEG.reject.gcompreject = ERP.ICA(s_vp).gcompreject;
    EEG.icawinv = ERP.ICA(s_vp).icawinv;
    EEG.icasphere = ERP.ICA(s_vp).icasphere;
    EEG.icaweights = ERP.ICA(s_vp).icaweights;
    EEG.icachansind = ERP.ICA(s_vp).icachansind;

    EEG = eeg_checkset( EEG );

    % Automatically reject all marked components
    EEG = pop_subcomp(EEG,[],0);

    %recompute EEG.icaact:
    EEG = eeg_checkset( EEG );

    % interpolate missing channels
    if length(EEG.chanlocs)<length(ERP.chanlocs)
        EEG = interpol( EEG, ERP.chanlocs);
    end

    % filter
    EEG = pop_eegfiltnew(EEG, [], fil_ter2(1), [], 0, [], 0); % low pass

    % save;
    EEG.setname = [SUBJECT{s_vp}];
    EEG = pop_saveset(EEG, 'filename', ['VP', sprintf('%04d',s_vp), '_', SUBJECT{s_vp}, '.set'], 'filepath', PATH_rem);
    
end % s_vp
    cd([PATH_gen, PATH_event]);
    save ERP.mat ERP 
    
    xlswrite([PATH_gen, PATH_event, '_trials1_frn.xls'],ERP.anz_trials(:,:)); % Documentation: Trials removed
    
%% VI Create Study
    [ALLEEG EEG CURRENTSET ALLCOM] = eeglab;
    [STUDY, ALLEEG] = create_study_fv(PATH_rem, PATH_study, ['study_',PATH_event(1:end-1)], 1); 
    
end % s_marker

% log
% VP9 & VP11-61: ADJUST had to be disabled to get SASICA run

% 'cycles', [3 0.5], 'nfreqs', 8, 'freqs', [3.5 13], 'ntimesout', 200, 'baseline', [-300 -200]

%% Export
    % use the function 'local_minmax' once to save the data in data{marker}
    % Header from the beginning must have been executed once (for folders)
for s_marker=1:3
    PATH_event=ALL.PATH_event{s_marker};
    PATH_rem = [PATH_gen, PATH_event, 'rem\'];
    cd([PATH_rem]);
    
    data{s_marker}=local_minmax({'TP9','TP10','Fz','Pz'},[200 350]); 
end

% open an EEG-file, ERP and BEHAV
s_marker=3;

    PATH_event=ALL.PATH_event{s_marker};
    PATH_rem = [PATH_gen, PATH_event, 'rem\'];
    cd([PATH_rem]);
    EEG = pop_loadset('filename',['VP', sprintf('%04d',1), '_', SUBJECT{1}, '.set'], 'filepath', PATH_rem); 

    cd([PATH_gen, PATH_event]);
    load ERP.mat 

    cd(PATH_gen);
    load BEHAV.mat 


%% Get Peaks
% Marker und (für Difference-Wave) Pfad auswählen
s_marker=3;

    PATH_event=ALL.PATH_event{s_marker};
    PATH_rem = [PATH_gen, PATH_event, 'rem\'];
    cd([PATH_rem]);

local_minmax({'TP9','TP10'},[130 210],'data',data{s_marker},'avrg_chan','on'); % N170
local_minmax({'Fz','FCz','Cz'},[130 200],'data',data{s_marker}); % P2
local_minmax({'FCz'},[200 350],'data',data{s_marker}); %N2
local_minmax({'Pz'},[300 500],'data',data{s_marker}); % P3a
local_minmax({'Pz'},[400 800],'data',data{s_marker}); % P3b

local_minmax({'Fz','FCz','Cz'},[300 350],'data',data{s_marker}); % P2 

% Difference-Wave from the files - choose correct path first
local_minmax({'FCz'},[130 200],'condition',{'S 67' 'S 77'},'diff_condition',{'S 68' 'S 78'}); % P2 
local_minmax({'Fz','FCz','Cz'},[200 350],'condition',{'S 67' 'S 77'},'diff_condition',{'S 68' 'S 69' 'S 78' 'S 79'}); % N2 

% Peaks
%                   N170(TP9/10)   P2(FCz)     N2(FCz)      P3a(Pz)       P3b(Pz)
% 1: T1 face        152            148         248          316           500
% 2: T3 offer       NaN            184         328          NaN           480
% 3: T3 face        148            152         308          324           420


% Peak +/- 20ms     N170    P2      N2      P3a     P3b
%                   TP9/10  FCz     FCz     Pz      Pz
BEHAV.times(1,:,:)=[132 177;128 168;228 268;296 236;480 520]; % 3: T1 face
BEHAV.times(2,:,:)=[NaN NaN;184 224;308 348;300 340;460 500]; % 2: T2 offer
BEHAV.times(3,:,:)=[128 168;132 172;288 328;304 344;400 440]; % 3: T3 face
BEHAV.points=find_samplingpoint(EEG.times,BEHAV.times);

% Visual inspection N170    P2      N2      P3a     P3b
%                   TP9/10  FCz     FCz     Pz      Pz
BEHAV.times(1,:,:)=[140 176;136 192;216 292;284 364;456 588]; % 3: T1 face
BEHAV.times(2,:,:)=[NaN NaN;164 220;288 372;336 384;432 540]; % 2: T2 offer
BEHAV.times(3,:,:)=[140 176;132 188;224 344;284 356;456 588]; % 3: T3 face
BEHAV.points=find_samplingpoint(EEG.times,BEHAV.times);

BEHAV.channel={{'TP9' 'TP10'};'FCz';'FCz';'Pz';'Pz'};
kk=find_channel_nr({ERP.chanlocs.labels},BEHAV.channel);

BEHAV.channel_nr=find_channel_nr({ERP.chanlocs.labels},BEHAV.channel);


%% Get the behavioral data from the markers
% s_marker=2: Offer; 200: 0 Cent; 201: 1 Cents; 202: 2 Cents; 203: 3 Cents; 204: 4 Cents; 205: 5 Cents 
% s_marker=3: decision+facial_expression 21_: accept; 22_: reject; 2_0: happy; 2_1: neutral; 2_2: sad

for s_vp = 1:length(SUBJECT) % get the raw data
    EEG = pop_loadbv(PATH_raw, [SUBJECT{s_vp}, '.vhdr'], [], [1:32]);
    EEG.setname = SUBJECT{s_vp};
    EEG = eeg_checkset( EEG );
    s_fruit=1;
    s_off=1;
    s_dec=1;
    for s_event=1:length(EEG.event)
        if sum(strcmp(EEG.event(s_event).type,ALL.CONDITION{1})) % fruit
            BEHAV.fruit(s_vp,s_fruit)=str2num(EEG.event(s_event).type(3:4)); %  67-79: Decision+Feedback: 67=correct smile, 68=correct neutral, 69=correct sad, 77=wrong smile,78=wrong neutral, 79=wrong sad
            BEHAV.fruit_urevent(s_vp,s_fruit)=s_event;
            s_fruit=s_fruit+1;
        elseif sum(strcmp(EEG.event(s_event).type,ALL.CONDITION{2})) % offer
            BEHAV.off(s_vp,s_off)=str2num(EEG.event(s_event).type(4));
            BEHAV.off_urevent(s_vp,s_off)=s_event;
            s_off=s_off+1;
        elseif sum(strcmp(EEG.event(s_event).type,ALL.CONDITION{3})) % facial feedback
            BEHAV.dec(s_vp,s_dec)=str2num(EEG.event(s_event).type(3)); % decision 1 acept 2 reject
            BEHAV.fac(s_vp,s_dec)=str2num(EEG.event(s_event).type(3:4)); % facial feedback: 10: A-happy; 11: A-neutral; 21: R-neutral; 22: R-sad 
            BEHAV.dec_urevent(s_vp,s_dec)=s_event;
            s_dec=s_dec+1;
        end
    end
end

%% Compute the time-frequency

points=find_samplingpoint(EEG.times,[100 500]);
points_base=find_samplingpoint(EEG.times,[-300 -100]);
channel_nr=find_channel_nr({EEG.chanlocs.labels},{'FCz'});

for s_marker=1:3
    PATH_event=ALL.PATH_event{s_marker};
    PATH_rem = [PATH_gen, PATH_event, 'rem\'];
    
            
    for s_vp=1:length(SUBJECT)
        EEG = pop_loadset('filename',['VP', sprintf('%04d',s_vp), '_', SUBJECT{s_vp}, '.set'], 'filepath',PATH_rem);
        EEG = eeg_checkset( EEG );
         
		k= wavelet_power_2(EEG,'lowfreq', 4, 'highfreq', 8, 'log_spacing', 1, 'fixed_cycles', 3.5); % 3 dim array = Channel x Time x Trials
        allTF{s_marker,s_vp}(:,:,:)=10*log10(k);
%        BEHAV.TF{s_marker,s_vp}(:)=10*log10(mean(k(channel_nr{:},points(1):points(2),:),2));
    end
end

% Channel, Time range & Baseline correction
for s_marker=1:3
    for s_vp=1:length(SUBJECT)
        BEHAV.TF{s_marker,s_vp}(:)=mean(allTF{s_marker,s_vp}(channel_nr{:},points(1):points(2),:),2) -...
            mean(allTF{s_marker,s_vp}(channel_nr{:},points_base(1):points_base(2),:),2); 
    end
end


%% Export 
% M1 & M3, EKPs. Just change 's_marker' to '1' or '3' accordingly
s_marker=1;
    PATH_event=ALL.PATH_event{s_marker};
    PATH_rem = [PATH_gen, PATH_event, 'rem\'];
    cd([PATH_gen, PATH_event]);
    CONDITION=ALL.CONDITION{s_marker};
% Open file
myfile = fopen(['M',num2str(s_marker),'_export.csv'], 'w');
fprintf(myfile, '%s\n', ['VP_nr,VP_name,condition,N170,P2,N2,P3a,P3b,Theta']);

                
for s_vp=1:length(SUBJECT)
    EEG = pop_loadset('filename',['VP', sprintf('%04d',s_vp), '_', SUBJECT{s_vp}, '.set'], 'filepath',PATH_rem);
    EEG = eeg_checkset( EEG );

    % create a logical for which trials belong to which condition
    s_trial=1;
    clear log_trial
    for s_event=1:length(EEG.event) 
        if sum(strcmp(EEG.event(s_event).type,CONDITION)) % is it a marker
            log_trial(s_trial,:)=strcmp(EEG.event(s_event).type,CONDITION); % k[trial,marker] logical
            s_trial=s_trial+1;
        end
    end

    for s_cond=1:length(CONDITION)
        fprintf(myfile, '%d,', s_vp);
        fprintf(myfile, '%s,', EEG.setname);
        fprintf(myfile, '%s,', CONDITION{s_cond});

        for s_comp=1:length(BEHAV.channel_nr) % EEG-data
            value=mean(mean(mean(EEG.data(BEHAV.channel_nr{s_comp},BEHAV.points(s_marker,s_comp,1):BEHAV.points(s_marker,s_comp,2),log_trial(:,s_cond)),3),2),1);
            fprintf(myfile, '%.4f,', value);
        end

    fprintf(myfile, '%.4f', mean(BEHAV.TF{s_marker,s_vp}(log_trial(:,s_cond)))); %TF
    fprintf(myfile, '%d\n', []); % new line
    end
end
fclose(myfile);




% M2, EKPs
s_marker=2;
    PATH_event=ALL.PATH_event{s_marker};
    PATH_rem = [PATH_gen, PATH_event, 'rem\'];
    cd([PATH_gen, PATH_event]);
    CONDITION=ALL.CONDITION{s_marker};
% Open file
myfile = fopen(['M',num2str(s_marker),'_export.csv'], 'w');
fprintf(myfile, '%s\n', ['VP_nr,VP_name,condition,P2,N2,P3a,P3b,Theta,decision']);

for s_vp=1:length(SUBJECT)
    EEG = pop_loadset('filename',['VP', sprintf('%04d',s_vp), '_', SUBJECT{s_vp}, '.set'], 'filepath',PATH_rem);
    EEG = eeg_checkset( EEG );

    % create a logical for which trials belong to which condition
    s_trial=1;
    clear log_trial
    for s_event=1:length(EEG.event) 
        if sum(strcmp(EEG.event(s_event).type,CONDITION)) % is it a marker
            log_trial(s_trial,:)=strcmp(EEG.event(s_event).type,CONDITION); % k[trial,marker] logical
            s_trial=s_trial+1;
        end
    end

    for s_cond=1:length(CONDITION)
        fprintf(myfile, '%d,', s_vp);
        fprintf(myfile, '%s,', EEG.setname);
        fprintf(myfile, '%s,', CONDITION{s_cond});

        for s_comp=2:length(BEHAV.channel_nr) % EEG-data
            value=mean(mean(mean(EEG.data(BEHAV.channel_nr{s_comp},BEHAV.points(s_marker,s_comp,1):BEHAV.points(s_marker,s_comp,2),log_trial(:,s_cond)),3),2),1);
            fprintf(myfile, '%.4f,', value);
        end

    fprintf(myfile, '%.4f,', mean(BEHAV.TF{s_marker,s_vp}(log_trial(:,s_cond)))); %TF
    
    behav=[];
    for s_event=1:length(EEG.event) 
        if sum(strcmp(EEG.event(s_event).type,CONDITION(s_cond))) % is it a marker
            s_urtrial=find_samplingpoint(BEHAV.off_urevent(s_vp,:),EEG.event(s_event).urevent);  % trial number before trial rejection
            behav=[behav,BEHAV.dec(s_vp,s_urtrial)];
        end
    end

    fprintf(myfile, '%d', mean(behav)); % Decision

    fprintf(myfile, '%d\n', []); % new line
    end
end
fclose(myfile);

% M1, single trial data
s_marker=1;
    PATH_event=ALL.PATH_event{s_marker};
    PATH_rem = [PATH_gen, PATH_event, 'rem\'];
    cd([PATH_gen, PATH_event]);
% Open file
myfile = fopen(['M1_single.csv'], 'w');
fprintf(myfile, '%s\n', ['VP_nr,VP_name,urtrial,fac_feed,N170,P2,N2,P3a,P3b,Theta']);

for s_vp=1:length(SUBJECT)
EEG = pop_loadset('filename',['VP', sprintf('%04d',s_vp), '_', SUBJECT{s_vp}, '.set'], 'filepath',PATH_rem);
EEG = eeg_checkset( EEG );
for s_event=1:length(EEG.event)
    if sum(strcmp(EEG.event(s_event).type,ALL.CONDITION{s_marker})) % is it a marker
        s_trial=EEG.event(s_event).epoch;
        s_urtrial=find(BEHAV.fruit_urevent(s_vp,:)==EEG.event(s_event).urevent);  % trial number before trial rejection
        
        fprintf(myfile, '%d,', s_vp);
        fprintf(myfile, '%s,', EEG.setname);
        fprintf(myfile, '%d,', s_urtrial);
        fprintf(myfile, '%d,', BEHAV.fruit(s_vp,s_urtrial)); % Facial Feedback

        for s_comp=1:length(BEHAV.channel_nr) % EEG-data
            value=mean(mean(EEG.data(BEHAV.channel_nr{s_comp},BEHAV.points(s_marker,s_comp,1):BEHAV.points(s_marker,s_comp,2),s_trial),2),1);
            fprintf(myfile, '%.4f,', value);
        end
        fprintf(myfile, '%.4f', BEHAV.TF{s_marker,s_vp}(s_trial)); %TF
        fprintf(myfile, '%d\n', []); % new line
    end
end
end
fclose(myfile);

% M2, single trial data
s_marker=2;
    PATH_event=ALL.PATH_event{s_marker};
    PATH_rem = [PATH_gen, PATH_event, 'rem\'];
    cd([PATH_gen, PATH_event]);
% Open file
myfile = fopen(['M2_single.csv'], 'w');
fprintf(myfile, '%s\n', ['VP_nr,VP_name,urtrial,offer,P2,N2,P3a,P3b,Theta,decision,offer_n1,decision_n1']);

for s_vp=1:length(SUBJECT)
EEG = pop_loadset('filename',['VP', sprintf('%04d',s_vp), '_', SUBJECT{s_vp}, '.set'], 'filepath',PATH_rem);
EEG = eeg_checkset( EEG );
for s_event=1:length(EEG.event)
    if sum(strcmp(EEG.event(s_event).type,ALL.CONDITION{s_marker})) % is it a marker
        s_trial=EEG.event(s_event).epoch;
        s_urtrial=find(BEHAV.off_urevent(s_vp,:)==EEG.event(s_event).urevent);  % trial number before trial rejection
        
        fprintf(myfile, '%d,', s_vp);
        fprintf(myfile, '%s,', EEG.setname);
        fprintf(myfile, '%d,', s_urtrial);
        fprintf(myfile, '%d,', BEHAV.off(s_vp,s_urtrial)); % Offer Marker

        for s_comp=2:length(BEHAV.channel_nr) % EEG-data; start from s_comp ==2 (no N170)
            value=mean(mean(EEG.data(BEHAV.channel_nr{s_comp},BEHAV.points(s_marker,s_comp,1):BEHAV.points(s_marker,s_comp,2),s_trial),2),1);
            fprintf(myfile, '%.4f,', value);
        end
        fprintf(myfile, '%.4f,', BEHAV.TF{s_marker,s_vp}(s_trial)); %TF
        
        fprintf(myfile, '%d,', BEHAV.dec(s_vp,s_urtrial)); % Decision
        if s_urtrial<size(BEHAV.off,2)
            fprintf(myfile, '%d,', BEHAV.off(s_vp,s_urtrial+1)); % Offer in n+1
            fprintf(myfile, '%d', BEHAV.dec(s_vp,s_urtrial+1)); % Decision in n+1 
        end
        fprintf(myfile, '%d\n', []); % new line
    end
end
end
fclose(myfile);


% M3, single trial data
s_marker=3;
    PATH_event=ALL.PATH_event{s_marker};
    PATH_rem = [PATH_gen, PATH_event, 'rem\'];
    cd([PATH_gen, PATH_event]);
% Open file
myfile = fopen(['M3_single.csv'], 'w');
fprintf(myfile, '%s\n', ['VP_nr,VP_name,urtrial,fac_feed,N170,P2,N2,P3a,P3b,Theta,offer,decision,offer_n1,decision_n1']);

for s_vp=1:length(SUBJECT)
EEG = pop_loadset('filename',['VP', sprintf('%04d',s_vp), '_', SUBJECT{s_vp}, '.set'], 'filepath',PATH_rem);
EEG = eeg_checkset( EEG );
for s_event=1:length(EEG.event)
    if sum(strcmp(EEG.event(s_event).type,ALL.CONDITION{s_marker})) % is it a marker
        s_trial=EEG.event(s_event).epoch;
        s_urtrial=find(BEHAV.dec_urevent(s_vp,:)==EEG.event(s_event).urevent);  % trial number before trial rejection
        
        fprintf(myfile, '%d,', s_vp);
        fprintf(myfile, '%s,', EEG.setname);
        fprintf(myfile, '%d,', s_urtrial);
        fprintf(myfile, '%d,', BEHAV.fac(s_vp,s_urtrial)); % Facial Feedback

        for s_comp=1:length(BEHAV.channel_nr) % EEG-data
            value=mean(mean(EEG.data(BEHAV.channel_nr{s_comp},BEHAV.points(s_marker,s_comp,1):BEHAV.points(s_marker,s_comp,2),s_trial),2),1);
            fprintf(myfile, '%.4f,', value);
        end
        fprintf(myfile, '%.4f,', BEHAV.TF{s_marker,s_vp}(s_trial)); %TF
        

        fprintf(myfile, '%d,', BEHAV.off(s_vp,s_urtrial)); % Offer
        fprintf(myfile, '%d,', BEHAV.dec(s_vp,s_urtrial)); % Decision
        if s_urtrial<size(BEHAV.off,2)
            fprintf(myfile, '%d,', BEHAV.off(s_vp,s_urtrial+1)); % Offer in n+1
            fprintf(myfile, '%d', BEHAV.dec(s_vp,s_urtrial+1)); % Decision in n+1 
        end
        fprintf(myfile, '%d\n', []); % new line
    end
end
end
fclose(myfile);

    cd(PATH_gen);
    save BEHAV.mat BEHAV 

%% Plot the data
s_marker=1;
    PATH_event=ALL.PATH_event{s_marker};
    PATH_rem = [PATH_gen, PATH_event, 'rem\'];
    cd([PATH_rem]);
    
erp=grand_average({'TP9','TP10'},[140 176],'condition',{{'S 67' 'S 77'} {'S 68' 'S 78'} {'S 69' 'S 79'}},'time_plot',[-200 700],'colorbar','on'); % for the topoplot
erp=grand_average({'FCz'},[224 344],'condition',{{'S 67' 'S 77'} {'S 68' 'S 78'} {'S 69' 'S 79'}},'time_plot',[-200 700]); % for the topoplot
erp=grand_average({'Pz'},[456 588],'condition',{{'S 67' 'S 77'} {'S 68' 'S 78'} {'S 69' 'S 79'}},'time_plot',[-200 700]); % for the topoplot

s_marker=2;
    PATH_event=ALL.PATH_event{s_marker};
    PATH_rem = [PATH_gen, PATH_event, 'rem\'];
    cd([PATH_rem]);

erp=grand_average({'FCz'},[288 372],'condition',{'S200' 'S201' 'S202' 'S203' 'S204' 'S205'},'time_plot',[-200 700]); % for the topoplot
erp=grand_average({'Pz'},[432 540],'condition',{'S200' 'S201' 'S202' 'S203' 'S204' 'S205'},'time_plot',[-200 700]); % for the topoplot

s_marker=3;
    PATH_event=ALL.PATH_event{s_marker};
    PATH_rem = [PATH_gen, PATH_event, 'rem\'];
    cd([PATH_rem]);

erp=grand_average({'TP9','TP10'},[140 176],'condition',{'S210' 'S211' 'S221' 'S222'},'time_plot',[-200 700]); % for the topoplot
erp=grand_average({'FCz'},[224 344],'condition',{'S210' 'S211' 'S221' 'S222'},'time_plot',[-200 700]); % for the topoplot
erp=grand_average({'Pz'},[456 588],'condition',{'S210' 'S211' 'S221' 'S222'},'time_plot',[-200 900]); % for the topoplot



ALL.CONDITION{1} = {'S 67' 'S 68' 'S 69' 'S 77' 'S 78' 'S 79'};

% Peaks
%                   N170(TP9/10)   P2(FCz)     N2(FCz)      P3a(Pz)       P3b(Pz)
% 1: T1 face        140-176        136-192     216-292      284-364       456-588
% 2: T3 offer                      164-220     288-372      336-384       432-540
% 3: T3 face        140-176        132-188     224-344      284-356       456-588


% TF-Plot (run wavelet_power_2 first, as allTF is too big to be written in
% BEHAV
s_marker=2;
for s_vp=1:length(SUBJECT)
    dat_help(s_vp,:)=mean(allTF{s_marker,s_vp}(channel_nr{:},:,:),3); % average across trials BEHAV.allTF{s_marker,s_vp}(channel,time,trial),3)
    dat(1,:)=mean(dat_help,1);
    dat(2,:)=mean(dat_help,1);
end

figure;
surf(EEG.times,1:2,dat,'EdgeColor','none','FaceColor','interp');
view(2)
colorbar
