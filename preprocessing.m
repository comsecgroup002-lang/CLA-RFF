
base_dir = 'D:\path\to\data';
sample_count = 2e7;
sample_rate = 1e6;
window_size = 8192;
stride = 8192;
wavelet_type = 'db4';


stft_win_size = 256;
stft_overlap = 200;
stft_nfft = 512;
stft_fs = sample_rate;

if isempty(gcp('nocreate'))
    parpool('local', 4);
end

for device_idx = 1:11
    device_name = sprintf('Device%d', device_idx);
    device_dir = fullfile(base_dir, device_name);
    save_base_prefix = fullfile(device_dir, 'xb_v2');
    cD1_path = fullfile(save_base_prefix, 'cD1'); if ~exist(cD1_path, 'dir'), mkdir(cD1_path); end
    cD2_path = fullfile(save_base_prefix, 'cD2'); if ~exist(cD2_path, 'dir'), mkdir(cD2_path); end
    cA2_path = fullfile(save_base_prefix, 'cA2'); if ~exist(cA2_path, 'dir'), mkdir(cA2_path); end

    for i = 1:10
        dat_file = fullfile(device_dir, sprintf('IQ_%d.dat', i));
        if ~exist(dat_file, 'file')
            warning('%s', sprintf('The file does not exist.: %s', dat_file));
            continue;
        end

        fprintf('📂 Processing: %s\n', dat_file);
        try
            fid = fopen(dat_file, 'rb');
            raw = fread(fid, sample_count, 'float32');
            fclose(fid);
        catch read_err
            if fid > 0, fclose(fid); end
            warning('%s', ['❌ Data read failed: ', read_err.message]);
            continue;
        end

        I = raw(1:2:end);
        Q = raw(2:2:end);
        IQ_data = complex(I, Q);
        total_points = length(IQ_data);
        num_windows = floor((total_points - window_size) / stride) + 1;

        parfor w = 1:num_windows
            try
                idx_start = (w - 1) * stride + 1;
                idx_end = idx_start + window_size - 1;
                segment = IQ_data(idx_start:idx_end);

                
               Ts = 1 / sample_rate;
               retain_ratio = 0.5;
               if rand() < retain_ratio
                   segment_aug = segment;  
               else
                   try
                       segment_aug = apply_pipeline_augmentation(segment, Ts);
                   catch pipeline_err
                       warning('⚠️ CLA failed（win %d）: %s', w, pipeline_err.message);
                       segment_aug = segment; 
                   end
               end


                L = 2^7;
                if length(segment_aug) < 2*L
                    continue;
                end

                phase = unwrap(angle(segment_aug(1:L)));
                inst_freq = diff(phase) / (2*pi*Ts);
                cfo_coarse = mean(inst_freq);
                t = (0:length(segment_aug)-1) * Ts;
                segment_c = segment_aug .* exp(-1j*2*pi*cfo_coarse*t);

                phi = angle(sum(segment_c(1:L) .* conj(segment_c(1+L:2*L))));
                cfo_fine = -phi / (2*pi * L * Ts);
                segment_cf = segment_c .* exp(-1j*2*pi*cfo_fine*t);
                segment_cfo_corrected = segment_cf;

               
                rms_val = sqrt(mean(abs(segment_cfo_corrected).^2));
                if rms_val < 1e-8
                    continue;
                end
                segment_norm = segment_cfo_corrected / rms_val;

                
                segment_real = real(segment_norm);
                [cA1, cD1] = dwt(segment_real, wavelet_type);
                [cA2, cD2] = dwt(cA1, wavelet_type);

                title_base = sprintf('Device%d IQ%d Window%d -', device_idx, i, w);

                save_stft_spectrogram(cD1, stft_win_size, stft_overlap, stft_nfft, stft_fs, ...
                    fullfile(cD1_path, sprintf('IQ%02d_win%03d.png', i, w)), ...
                    [title_base ' cD1']);

                save_stft_spectrogram(cD2, stft_win_size, stft_overlap, stft_nfft, stft_fs, ...
                    fullfile(cD2_path, sprintf('IQ%02d_win%03d.png', i, w)), ...
                    [title_base ' cD2']);

                save_stft_spectrogram(cA2, stft_win_size, stft_overlap, stft_nfft, stft_fs, ...
                    fullfile(cA2_path, sprintf('IQ%02d_win%03d.png', i, w)), ...
                    [title_base ' cA2']);
            catch seg_err
                warning('%s', sprintf('⚠️ STFT failed (Device %d, IQ_%d, win %d): %s', ...
                    device_idx, i, w, seg_err.message));
                continue;
            end
        end
        fprintf('✅ Complete the analysis of% d windows for IQ_% d.dat in% s\n', device_name, i, num_windows);
    end
end

disp('🎉 All data processing completed！');


function [sig_out, myPathGain] = augmentation(sig_in, Ts)
    sig_len = length(sig_in);

  
    t_rms = (150 - 20) * rand() + 20;  
    t_rms = t_rms * 1e-9;              

    fD = 0.1 + (5 - 0.1) * rand();     
    k_factor = 3 + (10 - 3) * rand();  


    [avgPathGains, pathDelays] = exp_PDP(t_rms, Ts);
    wirelessChan = comm.RicianChannel( ...
        'SampleRate', 1/Ts, ...
        'KFactor', k_factor, ...
        'MaximumDopplerShift', fD, ...
        'PathDelays', pathDelays, ...
        'AveragePathGains', avgPathGains, ...
        'DopplerSpectrum', doppler('Jakes'), ...
        'PathGainsOutputPort', true);

    chanInfo = info(wirelessChan);
    delay = chanInfo.ChannelFilterDelay;
    chInput = [sig_in; zeros(50,1)];
    [chOut, myPathGain] = wirelessChan(chInput);
    sig_out = chOut(delay+1:sig_len+delay);
end


function [avgPathGains, pathDelays] = exp_PDP(tau_d, Ts)
    A_dB = -30;  
    A = 10^(A_dB/10);
    lmax = ceil(-tau_d * log(A) / Ts);  
    p = 0:lmax;
    pathDelays = p * Ts;

    sigma_tau = tau_d;
    p_vals = (1/sigma_tau) * exp(-p*Ts/sigma_tau);
    p_norm = p_vals / sum(p_vals);  
    avgPathGains = 10 * log10(p_norm);  
end


function sig_out = augmentation_alt(sig_in)
   
    snr = 30 + rand() * 10;  
    sig_noise = awgn(sig_in, snr, 'measured');

    
    gain = 1 + 0.05 * randn();                 
    sig_nl = gain * sig_noise + 0.003 * (sig_noise.^3);  

    
    alpha = 1 + 0.02 * randn();                 
    theta = deg2rad(1 * randn());               
    I = real(sig_nl);
    Q = imag(sig_nl);
    I_new = alpha * I;
    Q_new = Q * cos(theta) + I * sin(theta);
    sig_out = complex(I_new, Q_new);
end


function sig_out = augmentation_shift(sig_in)
    max_shift = 2;
    shift = randi([-max_shift, max_shift]);
    sig_out = circshift(sig_in, shift);
end

function sig_out = augmentation_drift(sig_in)
    N = length(sig_in);

    
    ppm_range = 300:1000;  
    drift_ppm = ppm_range(randi(length(ppm_range))) * (-1)^randi([0 1]);
    drift_ratio = 1 + drift_ppm / 1e6;

    
    [P, Q] = rat(drift_ratio, 1e-6);  
    max_val = 2^15;
    if P > max_val || Q > max_val
        P = round(drift_ratio * 1e4);
        Q = 1e4;
    end

    resampled = resample(sig_in, P, Q);

    
    if length(resampled) >= N
        sig_out = resampled(1:N);
    else
        sig_out = [resampled; zeros(N - length(resampled), 1)];
    end
end


function sig_out = apply_pipeline_augmentation(sig_in, Ts)
    
    sig_out = augmentation(sig_in, Ts);  

    
    if rand() < 0.7
        sig_out = augmentation_alt(sig_out);  
    end

    
    r = rand();
    if r < 0.05
        sig_out = augmentation_drift(sig_out);
    elseif r < 0.15
        sig_out = augmentation_shift(sig_out);
    end
end




function save_stft_spectrogram(signal, win_size, overlap, nfft, fs, save_path, title_text)
    [S, F, T] = spectrogram(signal, win_size, overlap, nfft, fs, 'yaxis');
    P = 10 * log10(abs(S));
    fig = figure('Visible', 'off', 'Position', [100, 100, 560, 420]);
    imagesc(T, F, P);
    axis xy;
    xlabel('Time (s)');
    ylabel('Frequency (Hz)');
    colorbar;
    colormap('parula');
    set(gca, 'FontSize', 10);
    exportgraphics(fig, save_path, 'Resolution', 150);
    close(fig);
end
