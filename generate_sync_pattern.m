function y = generate_sync_pattern(samples_per_symbol, cycles)

    base_pattern = [1 1 -1 -1].';

    pattern = repmat(base_pattern, [cycles 1]);

    pattern_samples = repelem(pattern, samples_per_symbol);

    h = 0.5;

    phase_advance_per_symbol = h * pi;
    phase_advance_per_sample = phase_advance_per_symbol / samples_per_symbol;

    pattern_samples = pattern_samples * phase_advance_per_sample;

    phases = cumsum(pattern_samples);

    y = exp(1j * phases);
end