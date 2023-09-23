function example = generate_example()
    is_noise = rand < 0.2;

    max_snr = 25;
    min_snr = -1;
    snr = rand * (max_snr - min_snr) + min_snr;

    if is_noise
        samples = single(generate_noise(snr));

        example.I = real(samples);
        example.Q = imag(samples);
        example.signal_present = int8(0);
        example.symbol_rate = int32(0);
        example.phase_offset = 0;
        example.time_offset = int32(0);
        example.snr = snr;
    else
        symbol_rates = [4800; 9600; 14400; 16000; 19200; 24000; 28000];
        symbols_per_second = symbol_rates(randi(7));
        time_offset = round(rand * 600);
        phase_offset = rand * 2 * pi;

        min_frequency_offset = -1500;
        max_frequency_offset = 1500;

        frequency_offset = rand * (max_frequency_offset - min_frequency_offset) + min_frequency_offset;

        samples = single(generate_training_signal(symbols_per_second, time_offset, phase_offset, frequency_offset, snr));

        example.I = real(samples);
        example.Q = imag(samples);
        example.signal_present = int8(1);
        example.symbol_rate = int32(symbols_per_second);
        example.phase_offset = phase_offset;
        example.time_offset = int32(time_offset);
        example.snr = snr;
    end
end
