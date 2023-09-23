function y = generate_training_signal(symbols_per_second, sample_offset, phase_offset, frequency_offset_Hz, snr_dB)

    total_samples = 1024;
    symbols = 136;

    tx_samples_per_symbol = 8;
    tx_sample_rate = tx_samples_per_symbol * symbols_per_second;

    sync_pattern = generate_sync_pattern(tx_samples_per_symbol, symbols / 4);

    sample_instants = (0:(numel(sync_pattern) - 1)).' / tx_sample_rate;

    rx_sample_rate = 25600;
    sync_pattern = resample(sync_pattern, sample_instants, rx_sample_rate);

    sync_pattern = sync_pattern * exp(1j * phase_offset);

    n = (0:(numel(sync_pattern) - 1)).';
    frequency_offset_digital = 2 * pi * frequency_offset_Hz / rx_sample_rate;
    frequency_rotator = exp(1j * n * frequency_offset_digital);

    sync_pattern = sync_pattern .* frequency_rotator;

    signal_samples = numel(sync_pattern);
    pad_samples = total_samples - signal_samples;

    trailing_samples = pad_samples - sample_offset;

    y = [zeros([sample_offset 1]); sync_pattern; zeros([trailing_samples 1])];
    y = y(1:total_samples);

    y = awgn(y, snr_dB);
end