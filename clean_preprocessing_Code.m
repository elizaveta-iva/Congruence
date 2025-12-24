%Adjust depending on the type of file 
raw = nirs.io.loadDirectory('C:\Users\ei00191\OneDrive - University of Surrey\fnirs\.nirs_congr\data\', {'subject'});

%Renaming the stimuli
job = nirs.modules.RenameStims();
job.listOfChanges = { ...
    '11',              'CongruentSmallNum';
    '22',              'CongruentSmallNum';
    '220',             'CongruentSmallNum';
    '33',              'CongruentSmallNum';
    '210',              'IncongruentSmallNum';
    '12',               'IncongruentSmallNum';
    '31',               'IncongruentSmallNum';
    '23',               'IncongruentSmallNum';
    '130',              'IncongruentSmallNum';
    '32',               'IncongruentSmallNum';
    '770',             'CongruentLargeNum';
    '55',              'CongruentLargeNum';
    '77',              'CongruentLargeNum';
    '660',             'CongruentLargeNum';
    '66',              'CongruentLargeNum';
    '76',              'IncongruentLargeNum';
    '56',              'IncongruentLargeNum';
    '670',             'IncongruentLargeNum';
    '57',              'IncongruentLargeNum';
    '75',              'IncongruentLargeNum';
    '65',              'IncongruentLargeNum';
    '10100',             'CongruentSem';
    '12120',             'CongruentSem';
    '1212',              'CongruentSem';
    '1010',              'CongruentSem';
    '1111',              'CongruentSem';
    '1012',              'IncongruentSem';
    '12100',             'IncongruentSem';
    '1011',              'IncongruentSem';
    '1112',              'IncongruentSem';
    '1110',              'IncongruentSem';
    '1211',              'IncongruentSem';
         };
raw = job.run(raw);

%alternative step where we combine the stimuli per block (cong+incong)
job = nirs.modules.RenameStims();
job.listOfChanges = { ...
    'CongruentSmallNum',                'SmallNum';
    'IncongruentSmallNum',              'SmallNum';
    'CongruentLargeNum',                'LargeNum';
    'IncongruentLargeNum',              'LargeNum';
    'CongruentSem',                     'Sem';
    'IncongruentSem',                   'Sem';
    };
raw = job.run(raw);

%Delete the stimuli from the first experiment
job = nirs.modules.DiscardStims;
job.listOfStims = {'0','20','6','2','10','1','4','8','9','200','stim_aux1', 'stim_aux2', 'stim_aux3', 'stim_aux4', 'stim_aux5', 'stim_aux6', 'stim_aux7', 'stim_aux8', 'stim_aux9', 'stim_aux10', 'stim_aux11', 'stim_aux12', 'stim_aux13', 'Brite'}; 
raw = job.run(raw);

% change stimulus duration - depending on whether you want to run congr and incongr trials separately
raw=nirs.design.change_stimulus_duration(raw,'CongruentSmallNum',6);
raw=nirs.design.change_stimulus_duration(raw,'IncongruentSmallNum',6);
raw=nirs.design.change_stimulus_duration(raw,'CongruentLargeNum',6);
raw=nirs.design.change_stimulus_duration(raw,'IncongruentLargeNum',6);
raw=nirs.design.change_stimulus_duration(raw,'CongruentSem',6);
raw=nirs.design.change_stimulus_duration(raw,'IncongruentSem',6);
% option 2
raw=nirs.design.change_stimulus_duration(raw,'SmallNum',6);
raw=nirs.design.change_stimulus_duration(raw,'LargeNum',6);
raw=nirs.design.change_stimulus_duration(raw,'Sem',6);

%Check all stim transformations were done correctly
%Hb_pruned = nirs.viz.StimUtil(Hb_pruned);
%raw = nirs.viz.StimUtil(raw);

for i = 1:49
    stim = raw(i).stimulus;  % Get the stimulus struct for this recording
    
    % Create the "smallnum_block" from 'smallnum' stimuli
    smallnum = stim('SmallNum');
    smallnum_block = nirs.design.StimulusEvents();
    smallnum_block.name = 'smallnum_block';
    smallnum_block.onset = [smallnum.onset(1), smallnum.onset(5), smallnum.onset(9)];
    smallnum_block.dur = [24, 24, 24];
    smallnum_block.amp = [1, 1, 1];
    raw(i).stimulus('smallnum_block') = smallnum_block;

    % Create the "largenum_block" from 'largenum' stimuli
    largenum = stim('LargeNum');
    largenum_block = nirs.design.StimulusEvents();
    largenum_block.name = 'largenum_block';
    largenum_block.onset = [largenum.onset(1), largenum.onset(5), largenum.onset(9)];
    largenum_block.dur = [24, 24, 24];
    largenum_block.amp = [1, 1, 1];
    raw(i).stimulus('largenum_block') = largenum_block;

    % Create the "sem_block" from 'sem' stimuli
    sem = stim('Sem');
    sem_block = nirs.design.StimulusEvents();
    sem_block.name = 'sem_block';
    sem_block.onset = [sem.onset(1), sem.onset(5), sem.onset(9)];
    sem_block.dur = [24, 24, 24];
    sem_block.amp = [1, 1, 1];
    raw(i).stimulus('sem_block') = sem_block;
end

for i = 50
    stim = raw(i).stimulus;  % Get the stimulus struct for this recording
    
    % Create the "smallnum_block" from 'smallnum' stimuli
    smallnum = stim('SmallNum');
    smallnum_block = nirs.design.StimulusEvents();
    smallnum_block.name = 'smallnum_block';
    smallnum_block.onset = [smallnum.onset(1), smallnum.onset(5), smallnum.onset(9)];
    smallnum_block.dur = [24, 24, 24];
    smallnum_block.amp = [1, 1, 1];
    raw(i).stimulus('smallnum_block') = smallnum_block;

    % Create the "largenum_block" from 'largenum' stimuli
    largenum = stim('LargeNum');
    largenum_block = nirs.design.StimulusEvents();
    largenum_block.name = 'largenum_block';
    largenum_block.onset = [largenum.onset(1), largenum.onset(5), largenum.onset(9)];
    largenum_block.dur = [24, 24, 24];
    largenum_block.amp = [1, 1, 1];
    raw(i).stimulus('largenum_block') = largenum_block;

    % Create the "sem_block" from 'sem' stimuli
    sem = stim('Sem');
    sem_block = nirs.design.StimulusEvents();
    sem_block.name = 'sem_block';
    sem_block.onset = [sem.onset(1), sem.onset(5)];
    sem_block.dur = [24, 24];
    sem_block.amp = [1, 1];
    raw(i).stimulus('sem_block') = sem_block;
end

job = nirs.modules.DiscardStims;
job.listOfStims = {'SmallNum','Sem','LargeNum'};
raw = job.run(raw);


% Trim baseline
job = nirs.modules.TrimBaseline();
job.preBaseline   = 5;
job.postBaseline  = 5;
raw = job.run(raw);

%save .nirs files at any stage before BeerLam, can be helpful for
%migrating data to another software
nirs.io.saveDotNirs(raw, '.nirs')

raw1 = raw;
%---------------------------------------------------------------------------------------------------------------
%QT-NIRS 
job = nirs.modules.QT;
job.qThreshold = 0.40; 
job.sciThreshold = 0.6;
job.pspThreshold = 0.06;
job.windowSec = 3;
%job.windowOverlap = true;
job.condMask = 'all';
job.guiFlag = 0; % do not run when working with several files, shows graphic results for each recording
job.fCut = [0.5 2.5];
ScansQuality = job.run(raw); 
%----------------------------------------------------------------------------------------------------------------
job = nirs.modules.OpticalDensity;
raw_OD = job.run(raw); 

job = nirs.modules.WaveletFilter;
job.sthresh = 3;
OD_Wav = job.run(raw_OD); 

job = nirs.modules.TDDR;
job.usePCA = true;
OD_WavTDDR = job.run(OD_Wav);

job = nirs.modules.BeerLambertLaw;
OD_WavTDDRBLL = job.run(OD_WavTDDR);

Hb_iir_filtered = OD_WavTDDRBLL;
hpf = 0.01;
lpf = 0.2;
sf  = OD_WavTDDRBLL.Fs;
order = 4;
[b, a] = butter(order, lpf*2/sf);
[d, c] = butter(order, hpf*2/sf, 'high');

for s = 1:size(Hb_iir_filtered, 1)
    lpf_dat = [];
    hpf_dat = [];

    % To apply the zero-phase/acausal filter, run the following function.
    lpf_dat = filtfilt(b, a, Hb_iir_filtered(s).data);
    % Alternatively, you can use the bandpass filter directly.
    hpf_dat = filtfilt(d, c, lpf_dat); 
    Hb_iir_filtered(s).data = hpf_dat;
end 

job = nirs.modules.Resample;
job.Fs=2; %depending on the computer
data = job.run(Hb_iir_filtered);

%CHANNEL PRUNING based on QTNIRS results
Hb_pruned = data;
for i=1:length(data)
   data(i) = data(i).sorted({'source', 'detector', 'type'});
   idxBadCh = find(ScansQuality(i).qMats.MeasListAct==0);
   fprintf('Scan:%i #BadChannels:%i\n',i,length(idxBadCh)/2);
   Hb_pruned(i).data(:,idxBadCh) = nan;
end


Hb_pruned = data;

for i = 1:length(data)
    data(i) = data(i).sorted({'source', 'detector', 'type'});
    
    % Identify bad channels
    idxBadCh = find(ScansQuality(i).qMats.MeasListAct == 0);
    fprintf('Scan: %i #BadChannels: %i\n', i, length(idxBadCh)/2);

    % Remove bad channels from data and link
    Hb_pruned(i).data(:, idxBadCh) = [];
    Hb_pruned(i).probe.link(idxBadCh, :) = [];
end


% Define the source-detector pairs of interest
targetPairs = [1 2; 3 1; 2 1];  % [S1-D2; S3-D1; S2-D1]

% Initialise counter
allThreePruned = 0;

for i = 1:length(Hb_pruned)
    link = Hb_pruned(i).probe.link;
    dataMatrix = Hb_pruned(i).data;
    
    prunedFlags = false(1, size(targetPairs,1));  % [S1D2, S3D1, S2D1]

    for p = 1:size(targetPairs,1)
        s = targetPairs(p,1);
        d = targetPairs(p,2);
        
        % Find channel indices for this source-detector pair (all types)
        ch_idx = find(link.source == s & link.detector == d);
        
        if all(all(isnan(dataMatrix(:, ch_idx))))
            prunedFlags(p) = true;
        end
    end

    if all(prunedFlags)
        allThreePruned = allThreePruned + 1;
        fprintf('Participant %d: ALL three channels pruned (S1-D2, S3-D1, S2-D1)\n', i);
    end
end

fprintf('\nTotal participants with ALL three channels pruned: %d\n', allThreePruned);



job=nirs.modules.GLM;
job.verbose = 1;
basis = nirs.design.basis.FIR;
basis.binwidth=1;
basis.isIRF = true;
basis.nbins=24; %depends on conditions (e.g., congr+incongr combined = 24sec) * sampling rate
job.basis('default')=basis;
SubjStats = job.run(Hb_pruned);

% Define the directory where you want to save the CSV files
directory = 'C:\fnirs\.nirs_congruency_task\glm\';

% Define the directory where you want to save the CSV files

% Loop through each dataset in SubjStats
for i = 1:numel(SubjStats)
    % Display some debugging information
    disp(['Processing dataset ', num2str(i)]);
    
    % Construct the filename
    filename = strcat(directory, num2str(i), '.csv');
    
    % Display the filename
    disp(['Writing to file: ', filename]);
    
    % Write the table to a CSV file
    writetable(SubjStats(i).table, filename);
    
    % Display a message indicating completion
    disp(['Dataset ', num2str(i), ' processed and written to file.']);
end

%Connectivity analysis - basic construction - to be expanded
job = nirs.modules.Connectivity;
Connectivity = job.run(Hb_ROI);


%GLM
job = nirs.modules.GLM;
job.type='AR-IRLS';
SubjStats=job.run(Hb_pruned);

job=nirs.modules.MixedEffects;
job.formula='beta~-1+cond';
GroupStats=job.run(SubjStats);

%%%%%%%%Generate channel Maps%%%%%%
GroupStats.draw('tstat',[-1, 1],'q<0.05')
%%%%%%%%%%%%%%Generate HRF Maps%%%%%
HRF=GroupStats.HRF('tstat'); 
HRF.gui.draw();
lines=nirs.viz.plot2D(HRF);


Contrast=GroupStats.ttest('LargeNum-SmallNum');
Contrast.draw('tstat',[-1 1],'q<0.05')

Contrast2=GroupStats.ttest('LargeNum-Sem');
Contrast2.draw('tstat',[-1 1],'q<0.05')

Contrast3=GroupStats.ttest('SmallNum-Sem');
Contrast3.draw('tstat',[-1 1],'q<0.05')


fixnan_data = nirs.util.fixnan(Hb_pruned);
fixnan_data=job.run(Hb_pruned)

data = Hb_pruned;

% Assuming 'data' is your preprocessed nirs.core.Data object (after Beer-Lambert)
for i = 1:length(data)
    % Identify channels with all NaNs
    badChannels = all(isnan(data(i).data), 1);
    
    % Keep only the good channels
    data(i).data = data(i).data(:, ~badChannels);
    data(i).probe.link = data(i).probe.link(~badChannels, :);
end


job = nirs.modules.Connectivity;
job.corrfcn=@(data)nirs.sFC.ar_corr(data,'4x',true); % Whitened correlation (using Pmax 4 x FS)
job.divide_events = 1;  % if true will parse into multiple conditions
job.min_event_duration=5;  % minimum duration of events
job.ignore = 0;  % time at transitions (on/off) to ignore (only valid if dividing events)
ConnStats = job.run(Hb_ROI);


job = nirs.modules.MixedEffectsConnectivity();
%  MixedEffectsConnectivity with properties:
%         formula: 'R ~ -1 + cond'
%     dummyCoding: 'full'
%      centerVars: 1
%            name: ''
%         prevJob: []
GroupConnStats = job.run(ConnStats);

GroupConnStats.draw('R','q<0.05')




t = GroupConnStats.table;
conditions = unique(t.condition);

for i = 1:length(conditions)
    conditionName = conditions{i};
    filtered = t(t.qvalue < 0.05 & strcmp(t.condition, conditionName), :);

    % Ensure columns are string
    filtered.SourceOrigin = string(filtered.SourceOrigin); 
    filtered.DetectorOrigin = string(filtered.DetectorOrigin);
    filtered.SourceDest = string(filtered.SourceDest); 
    filtered.DetectorDest = string(filtered.DetectorDest);

    for j = 1:height(filtered)
        if ismember(filtered.SourceOrigin(j), ["9", "10", "6", "8", "7"])
            filtered.SourceOrigin(j) = 'left';
        else
            filtered.SourceOrigin(j) = 'right';
        end
        
        if ismember(filtered.DetectorOrigin(j), ["2", "1", "5", "6"])
            filtered.DetectorOrigin(j) = 'parietal';
        else
            filtered.DetectorOrigin(j) = 'frontal';
        end

        if ismember(filtered.SourceDest(j), ["9", "10", "6", "8", "7"])
            filtered.SourceDest(j) = 'left';
        else
            filtered.SourceDest(j) = 'right';
        end
        
        if ismember(filtered.DetectorDest(j), ["2", "1", "5", "6"])
            filtered.DetectorDest(j) = 'parietal';
        else
            filtered.DetectorDest(j) = 'frontal';
        end
    end

    % Create channel labels
    filtered.OriginChannel = strcat(filtered.SourceOrigin, "-", filtered.DetectorOrigin, "_", filtered.TypeOrigin);
    filtered.DestChannel = strcat(filtered.SourceDest, "-", filtered.DetectorDest, "_", filtered.TypeDest);

    % Unique channels
    channels = unique([filtered.OriginChannel; filtered.DestChannel]);
    n = length(channels);
    connMatrix = NaN(n, n);

    for j = 1:height(filtered)
        src = find(strcmp(channels, filtered.OriginChannel(j)));
        dest = find(strcmp(channels, filtered.DestChannel(j)));
        connMatrix(src, dest) = filtered.R(j);
        connMatrix(dest, src) = filtered.R(j);
    end

    % Plot
    figure;
    h = heatmap(channels, channels, connMatrix, ...
        'Colormap', parula, ...
        'ColorLimits', [-1 1], ...
        'MissingDataColor', [0.8 0.8 0.8], ...
        'MissingDataLabel', 'No connection');
    title(['Functional Connectivity - ', conditionName, ' (q < 0.05)']);
    xlabel('Source-Detector Channels');
    ylabel('Source-Detector Channels');
end













for idx=1:60
    bas = dem.bas;  % make up an "age" for this 
    SubjStats(idx).demographics('bas')=bas(idx);   % making this numeric allows us to use it as a regressor
end
demographics = nirs.createDemographicsTable(SubjStats);

% You can view the demographics information for your data by typing:
disp(demographics)


% Plot the raw or filtered data to inspect cardiac oscillations
figure;
plot(Hb_iir_filtered(1).time, Hb_iir_filtered(1).data(:, 10)); % Plot the first channel
xlabel('Time (s)');
ylabel('Amplitude');
title('Cardiac Oscillations in fNIRS Data');

numChannels = size(Hb_iir_filtered(4).data, 2); % Number of channels
heartRates = zeros(numChannels, 1); % Array to store heart rate estimates
for i = 1:numChannels
    % Compute the power spectral density (PSD) for the current channel
    [pxx, f] = pwelch(Hb_iir_filtered(4).data(:, i), [], [], [], Hb_iir_filtered(1).Fs);

    % Find the peak frequency in the PSD within the heart rate range (0.5–2 Hz)
    [~, idx] = max(pxx(f >= 0.5 & f <= 2));
    heartRateFreq = f(f >= 0.5 & f <= 2);
    heartRates(i) = heartRateFreq(idx) * 60; % Convert Hz to BPM
end
highestHeartRate = max(heartRates);
disp(['Highest Heart Rate Estimate: ', num2str(highestHeartRate), ' BPM']);

figure;
plot(heartRates, 'o-');
xlabel('Channel Number');
ylabel('Heart Rate (BPM)');
title('Heart Rate Estimates Across Channels');

demographics = nirs.createDemographicsTable(raw);

job = nirs.modules.AddDemographics();
job.demoTable = demographics;
job.varToMatch = 'subject';  
Hb_pruned = job.run(Hb_pruned);



job = nirs.modules.RenameStims();
job.listOfChanges = { ...
    'SmallNum',                'easy';
    'Sem',              'easy'; };
Hb_pruned = job.run(Hb_pruned);















j=nirs.modules.Run_HOMER2();
    j.fcn = "hmrBlockAvg";
        tpre=-5;
        tpost=20;
    j.vars.trange = [tpre tpost];

blk = j.run(data);

% Get the size of blk.data (assuming blk contains multiple participants)
num_participants = length(blk);  % Number of participants
data_size = size(blk(1).data);    % Get the size of data for the first participant
time_points = data_size(1);    % Number of time points
num_channels = data_size(2);  % Number of channels
num_conditions = data_size(3);  % Number of conditions

% Initialize arrays to store the summed data for each condition (averaged across participants)
hbo_avg_all_participants = zeros(time_points, num_channels / 2, num_conditions); % HbO
hbr_avg_all_participants = zeros(time_points, num_channels / 2, num_conditions); % HbR

% Loop through each participant
for p = 1:num_participants
    % Extract data for the current participant
    cond_data = blk(p).data; % Data for participant p
    
    % Loop through each condition
    for cond = 1:num_conditions
        % Extract data for the current condition
        cond_data_current = cond_data(:,:,cond); 
        
        % Separate HbO and HbR data (odd and even columns)
        hbo_data = cond_data_current(:, 1:2:num_channels);  % Odd columns for HbO
        hbr_data = cond_data_current(:, 2:2:num_channels);  % Even columns for HbR
        
        % Accumulate the data for averaging later across participants
        hbo_avg_all_participants(:,:,cond) = hbo_avg_all_participants(:,:,cond) + hbo_data;
        hbr_avg_all_participants(:,:,cond) = hbr_avg_all_participants(:,:,cond) + hbr_data;
    end
end

% After accumulating data, average across participants
hbo_avg_all_participants = hbo_avg_all_participants / num_participants;
hbr_avg_all_participants = hbr_avg_all_participants / num_participants;

% Plot HbO and HbR averages across all participants for each channel, on one figure, for each condition
for cond = 1:num_conditions
    % Create a new figure
    figure;
    
    % Adjust the figure size to make it larger
    set(gcf, 'Position', [100, 100, 1200, 800]); % [x, y, width, height] to make the figure wider
    
    % Number of channels per condition (assuming each condition has num_channels/2 channels for HbO and HbR)
    num_channels_condition = num_channels / 2;  
    
    % Calculate number of rows and columns for the subplots
    num_rows = ceil(num_channels_condition / 6);  % Round up to ensure we fit all channels
    num_cols = 6;  % 6 subplots per row
    
    % Loop through each channel (for both HbO and HbR)
    for chan = 1:num_channels_condition
        % Create a subplot for each channel, adjusting rows and columns
        subplot(num_rows, num_cols, chan);  % Create a subplot for each channel
        
        % Plot HbO and HbR averages for the current channel
        plot(blk(1).time, mean(hbo_avg_all_participants(:,chan,cond), 2), 'b', 'LineWidth', 2); % Plot HbO (blue)
        hold on;
        plot(blk(1).time, mean(hbr_avg_all_participants(:,chan,cond), 2), 'r', 'LineWidth', 2); % Plot HbR (red)
        hold off;
        
        % Set plot properties
        title(['Channel ', num2str(chan), ' - Condition ', num2str(cond)]);
        xlabel('Time (s)');
        ylabel('\DeltaHbO and \DeltaHbR (µM)');
        legend({'HbO', 'HbR'}, 'Location', 'best');
        grid on;
    end
    
    % Add a common title for the whole figure using sgtitle
    sgtitle(['HbO and HbR Block Averages - Condition ', num2str(cond), ' (Across All Participants)']);
    
    % Add the annotation for channel locations
    annotation('textbox', [0.01, 0.5, 0.1, 0.2], 'String', ...
        ' 1-5 = Right Parietal  6-9 = Right Frontal 10-13 = Left Frontal 14-19 = Left Parietal', ...
        'EdgeColor', 'none', 'FontSize', 12, 'BackgroundColor', 'white');
end



%%%%%%%%%%%%%%%%%%%%%%%%
% Assuming 'data' is your table
data=SubjStats.table;

selected_cond = "SmallNum";
selected_source = 3;
selected_detector = 1;

% Filter for HBO
filtered_hbo = data( strcmp(data.cond, selected_cond) & ...
                      data.source == selected_source & ...
                      data.detector == selected_detector & ...
                      strcmp(data.type, 'hbo'), ...
                      {'subject', 'beta', 'source', 'detector'});

% Filter for HBR
filtered_hbr = data( strcmp(data.cond, selected_cond) & ...
                      data.source == selected_source & ...
                      data.detector == selected_detector & ...
                      strcmp(data.type, 'hbr'), ...
                      {'subject', 'beta', 'source', 'detector'});

% Display the new tables
disp('HBO Data:');
disp(filtered_hbo);

disp('HBR Data:');
disp(filtered_hbr);


