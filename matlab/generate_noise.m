function y = generate_noise(snr)
    total_samples = 1024;

    y = complex(zeros([1024 1]));
    y = awgn(y, snr);
end