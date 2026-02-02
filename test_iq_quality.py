#!/usr/bin/env python3
"""
I/Q Quality Diagnostic for IC-R8600

Analyzes the raw I/Q data quality and FM demodulation to understand
why SNR might be low despite strong signals.
"""

import sys
import time
import numpy as np

sys.path.insert(0, '/home/philj/dev/pjfm')
import icom_r8600 as r8600

def main():
    print("=" * 60)
    print("I/Q Quality Diagnostic")
    print("=" * 60)

    radio = r8600.IcomR8600()

    try:
        radio.open()

        # Get frequency from command line or use default
        freq = 88.1e6 if len(sys.argv) < 2 else float(sys.argv[1]) * 1e6
        print(f"\nFrequency: {freq/1e6:.1f} MHz")

        # Configure streaming with unity gain to see raw levels
        radio._iq_gain = 1.0  # Unity gain for analysis
        radio.configure_iq_streaming(freq=freq, sample_rate=480000)

        # Let it stabilize
        time.sleep(0.5)
        radio.flush_iq()
        time.sleep(0.3)

        # Collect several blocks of I/Q data
        print("\nCollecting I/Q samples...")
        all_iq = []
        for _ in range(20):
            iq = radio.fetch_iq(8192)
            all_iq.append(iq)
            time.sleep(0.02)

        iq = np.concatenate(all_iq)
        print(f"Collected {len(iq)} samples")

        # ========== RAW I/Q ANALYSIS ==========
        print("\n" + "=" * 40)
        print("RAW I/Q ANALYSIS (unity gain)")
        print("=" * 40)

        # Amplitude statistics
        amplitude = np.abs(iq)
        print(f"\nAmplitude:")
        print(f"  Mean:  {amplitude.mean():.6f}")
        print(f"  Std:   {amplitude.std():.6f}")
        print(f"  Min:   {amplitude.min():.6f}")
        print(f"  Max:   {amplitude.max():.6f}")

        # Power
        power = np.mean(amplitude**2)
        power_db = 10 * np.log10(power + 1e-10)
        print(f"\nPower: {power:.6f} ({power_db:.1f} dB)")

        # Effective bits (assuming sine wave, 6.02 dB per bit)
        # But we're normalized, so calculate from std relative to full scale
        # Full scale normalized = 1.0, our std is what we have
        effective_bits = np.log2(1.0 / (amplitude.std() + 1e-10))
        print(f"Effective dynamic range: ~{effective_bits:.1f} bits")

        # I/Q balance
        i_power = np.var(iq.real)
        q_power = np.var(iq.imag)
        iq_ratio = i_power / (q_power + 1e-10)
        print(f"\nI/Q Balance:")
        print(f"  I power: {i_power:.6f}")
        print(f"  Q power: {q_power:.6f}")
        print(f"  Ratio:   {iq_ratio:.4f} ({10*np.log10(iq_ratio):.2f} dB)")

        # DC offset
        dc_i = np.mean(iq.real)
        dc_q = np.mean(iq.imag)
        print(f"\nDC Offset:")
        print(f"  I: {dc_i:.6f}")
        print(f"  Q: {dc_q:.6f}")

        # ========== FM DEMODULATION ==========
        print("\n" + "=" * 40)
        print("FM DEMODULATION ANALYSIS")
        print("=" * 40)

        # Quadrature discriminator
        product = iq[1:] * np.conj(iq[:-1])
        phase_diff = np.angle(product)

        # Convert to frequency (Hz)
        sample_rate = radio.iq_sample_rate
        inst_freq = phase_diff * sample_rate / (2 * np.pi)

        print(f"\nInstantaneous Frequency:")
        print(f"  Mean:  {inst_freq.mean():.1f} Hz (DC offset)")
        print(f"  Std:   {inst_freq.std():.1f} Hz (FM deviation + noise)")
        print(f"  Min:   {inst_freq.min():.1f} Hz")
        print(f"  Max:   {inst_freq.max():.1f} Hz")

        # FM deviation (should be ~75 kHz for broadcast FM)
        fm_deviation = inst_freq.std()
        if 50000 < fm_deviation < 100000:
            print(f"  --> FM deviation looks correct for broadcast FM")
        elif fm_deviation < 10000:
            print(f"  --> LOW deviation - signal may be weak or noise")
        else:
            print(f"  --> Unusual deviation")

        # ========== SNR ESTIMATION ==========
        print("\n" + "=" * 40)
        print("SNR ESTIMATION")
        print("=" * 40)

        # Method 1: From I/Q amplitude variance
        # For FM, amplitude should be constant; variance indicates noise
        amplitude_mean = amplitude.mean()
        amplitude_std = amplitude.std()
        amplitude_snr = amplitude_mean / (amplitude_std + 1e-10)
        amplitude_snr_db = 20 * np.log10(amplitude_snr)
        print(f"\nAmplitude-based SNR:")
        print(f"  {amplitude_snr_db:.1f} dB")

        # Method 2: From phase noise
        # For a clean FM signal, consecutive phase differences should be smooth
        phase_diff_diff = np.diff(phase_diff)  # Second derivative of phase
        phase_noise_std = phase_diff_diff.std()
        # Convert to equivalent FM noise
        phase_noise_hz = phase_noise_std * sample_rate / (2 * np.pi)
        print(f"\nPhase noise (jitter in inst. freq):")
        print(f"  {phase_noise_hz:.1f} Hz RMS")

        # Method 3: Spectral analysis
        # Look at baseband spectrum - signal should be in 0-53kHz, noise above
        print("\n" + "=" * 40)
        print("SPECTRAL ANALYSIS")
        print("=" * 40)

        # Compute baseband (FM demodulated)
        deviation = 75000  # 75 kHz for broadcast FM
        baseband = phase_diff * (sample_rate / (2 * np.pi * deviation))

        # FFT of baseband
        n = len(baseband)
        fft = np.fft.rfft(baseband)
        freqs = np.fft.rfftfreq(n, 1/sample_rate)
        power_spectrum = np.abs(fft)**2 / n

        # Signal power (0-53 kHz)
        signal_band = (freqs >= 30) & (freqs <= 53000)
        signal_power = power_spectrum[signal_band].sum() / signal_band.sum()

        # Noise power (60-80 kHz - above all FM subcarriers)
        noise_band = (freqs >= 60000) & (freqs <= 80000)
        noise_power = power_spectrum[noise_band].sum() / noise_band.sum()

        # Scale noise to signal bandwidth
        signal_bw = 53000  # Hz
        noise_bw = 20000   # Hz
        noise_power_scaled = noise_power * (signal_bw / noise_bw)

        if noise_power_scaled > 0:
            spectral_snr = signal_power / noise_power_scaled
            spectral_snr_db = 10 * np.log10(spectral_snr)
            print(f"\nSpectral SNR (0-53kHz vs 60-80kHz):")
            print(f"  Signal power: {10*np.log10(signal_power+1e-10):.1f} dB")
            print(f"  Noise power:  {10*np.log10(noise_power_scaled+1e-10):.1f} dB")
            print(f"  SNR: {spectral_snr_db:.1f} dB")
        else:
            print("\nSpectral SNR: Could not calculate (no noise detected)")

        # ========== SUMMARY ==========
        print("\n" + "=" * 40)
        print("SUMMARY")
        print("=" * 40)
        print(f"\nI/Q Level: {power_db:.1f} dB ({amplitude.mean()*100:.1f}% of full scale)")
        print(f"Effective bits: ~{effective_bits:.0f}")
        print(f"FM Deviation: {fm_deviation/1000:.1f} kHz")
        print(f"Amplitude SNR: {amplitude_snr_db:.1f} dB")
        if noise_power_scaled > 0:
            print(f"Spectral SNR: {spectral_snr_db:.1f} dB")

        if amplitude.mean() < 0.05:
            print("\nWARNING: I/Q level is very low (<5% of full scale)")
            print("This may limit demodulation quality due to quantization noise.")
            print("Consider increasing iq_gain for better demodulation.")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            radio.close()
        except:
            pass

    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
