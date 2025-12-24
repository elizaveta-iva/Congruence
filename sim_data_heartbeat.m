%noise= nirs.testing.simARNoise(); unless you want some SPECIFC noise
raw = nirs.testing.simDataSet();
data_physio = nirs.testing.simPhysioNoise(raw);
for i = 1:length(data_physio)
    data_full(i) = nirs.testing.simMotionArtifact(data_physio(i));
end


dataObj = OD_PCA(2);  % assuming 'data' is your NIRS recording
Fs = dataObj.Fs;    % sampling frequency
signal = mean(dataObj.data, 2);  % average across channels
[pxx, f] = pwelch(signal, [], [], [], Fs);
plot(f, 10*log10(pxx));
xlabel('Frequency (Hz)');
ylabel('Power (dB)');
title('Power Spectral Density of fNIRS Signal');
xlim([0 3]);  % focus on 0–3 Hz range to see heartbeat


job = nirs.modules.OpticalDensity;
raw_OD = job.run(data_full); 

job = nirs.modules.WaveletFilter;
job.sthresh = 3;
OD_Wav = job.run(raw_OD); 

job = nirs.modules.TDDR;
job.usePCA = true;
OD_TDDR_WAW = job.run(OD_Wav);

job = nirs.modules.TDDR;
job.usePCA = true;
OD_TDDR = job.run(raw_OD);

job = nirs.modules.PCAFilter;
job.ncomp = 0.80;
OD_PCA = job.run(raw_OD);

job = nirs.modules.BaselinePCAFilter;
job.nSV = 0.80;
OD_BPCA = job.run(raw_OD);



all_data = struct( ...
    'Raw', raw_OD, ...
    'Wav', OD_Wav, ...
    'TDDR_Wav', OD_TDDR_WAW, ...
    'TDDR', OD_TDDR, ...
    'PCA', OD_PCA, ...
    'BPCA', OD_BPCA ...
);
method_names = fieldnames(all_data);


Fs = raw_OD(1).Fs;
window = 256; noverlap = 128; nfft = 1024;
cardiac_range = [0.8 1.5];

results = struct();

for m = 1:length(method_names)
    method = method_names{m};
    data_set = all_data.(method);
    f_card_peaks = [];
    cardiac_power = [];

    for subj = 1:length(data_set)
        data = data_set(subj).data;
        for ch = 1:size(data,2)
            signal = data(:,ch);
            [Pxx, f] = pwelch(signal, window, noverlap, nfft, Fs);
            band = f >= cardiac_range(1) & f <= cardiac_range(2);
            [pmax, idx] = max(Pxx(band));
            fpeak = f(band);
            f_card_peaks(end+1) = fpeak(idx);
            cardiac_power(end+1) = mean(Pxx(band));
        end
    end

    results.(method).peak_freqs = f_card_peaks;
    results.(method).mean_power = cardiac_power;
    results.(method).percent_detected = sum(~isnan(f_card_peaks)) / length(f_card_peaks) * 100;
    results.(method).mean_freq = mean(f_card_peaks);
    results.(method).std_freq = std(f_card_peaks);
end


% Preallocate arrays for summary
summary = struct('Method', {}, 'PercentDetected', {}, 'MeanFreq', {}, 'StdFreq', {});

for m = 1:length(method_names)
    method = method_names{m};
    r = results.(method);
    
    summary(m).Method = method;
    summary(m).PercentDetected = r.percent_detected;
    summary(m).MeanFreq = r.mean_freq;
    summary(m).StdFreq = r.std_freq;
end

% Convert to MATLAB table
summary_table = struct2table(summary);
