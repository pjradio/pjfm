# pyfm

A real-time FM broadcast receiver for the SignalHound BB60D spectrum analyzer, featuring software-defined stereo demodulation and RDS decoding.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-proprietary-red.svg)

## Overview

pyfm is a command-line FM radio application that receives broadcast FM signals (88-108 MHz) via the SignalHound BB60D, performs all demodulation in software, and plays audio through the default Linux audio device. It features a rich terminal UI with real-time signal metrics, a 16-band spectrum analyzer, and full RDS (Radio Data System) decoding.

## Features

- **Real-time FM stereo reception** with automatic mono/stereo switching
- **RDS decoding** with station identification, program type, radio text, and clock time
- **16-band audio spectrum analyzer** with peak hold
- **Signal quality metrics** including S-meter, SNR, and pilot detection
- **Frequency presets** (5 programmable, saved to config)
- **Tone controls** with bass and treble boost
- **Squelch** for muting weak signals
- **Responsive terminal UI** built with Rich

## Technical Architecture

### IQ Streaming and FM Demodulation

pyfm uses the BB60D's IQ streaming mode to capture raw RF samples at 312.5 kHz (40 MHz / 128 decimation). The FM demodulation chain is implemented entirely in software:

```
IQ Samples (312.5 kHz) -> Quadrature Discriminator -> Baseband (0-100 kHz)
```

**Quadrature Discriminator**: FM demodulation is performed using the classic quadrature method:

```python
product = samples[n] * conj(samples[n-1])
baseband = angle(product) * (sample_rate / (2 * pi * deviation))
```

This extracts the instantaneous frequency deviation from the phase difference between consecutive samples, normalized by the FM deviation (75 kHz for broadcast FM).

### FM Stereo Decoding

Broadcast FM stereo uses a pilot-tone multiplexing system defined by the FCC:

| Frequency Range | Content |
|----------------|---------|
| 0-15 kHz | L+R (mono-compatible sum) |
| 19 kHz | Pilot tone |
| 23-53 kHz | L-R on 38 kHz DSB-SC carrier |
| 57 kHz | RDS subcarrier (optional) |

**Pilot Detection**: A 4th-order Butterworth bandpass filter (18.5-19.5 kHz) extracts the pilot tone. A Phase-Locked Loop (PLL) tracks the pilot with 100 Hz loop bandwidth and 0.707 damping factor (critically damped), providing:
- Coherent lock detection (prevents false stereo on noise)
- Phase-accurate 38 kHz carrier regeneration for L-R demodulation

**Stereo Matrix Decoding**:
```
L-R carrier = 2 * cos(2 * pilot_phase)  // PLL-derived
L-R = BPF(baseband, 23-53 kHz) * carrier * 2
Left  = (L+R) + (L-R)
Right = (L+R) - (L-R)
```

**De-emphasis**: A 75 μs de-emphasis filter compensates for the pre-emphasis applied at the transmitter, rolling off high frequencies to reduce noise.

### RDS (Radio Data System) Decoding

RDS transmits digital data at 1187.5 bps on a 57 kHz subcarrier (3× pilot frequency), using BPSK modulation with differential encoding.

#### Signal Processing Chain

1. **Bandpass Filtering**: 4th-order Butterworth filter extracts 57 kHz ± 2.4 kHz
2. **Coherent Demodulation**: Pilot tone is tripled using the identity:
   ```
   cos(3θ) = 4cos³(θ) - 3cos(θ)
   ```
   This derives a phase-locked 57 kHz carrier from the 19 kHz pilot.
3. **Differential Decoding**: RDS uses differential encoding where data is represented by phase *changes*. Decoded by multiplying with a one-symbol-delayed version.
4. **Symbol Timing Recovery**: Gardner Timing Error Detector (TED) with PI loop filter tracks symbol boundaries at 1187.5 Hz (210.526 samples/symbol at 250 kHz).

#### Block Synchronization and Error Correction

RDS data is organized into 26-bit blocks (16 data + 10 checkword), grouped into 4-block groups:

| Block | Offset Word | Content |
|-------|-------------|---------|
| A | 0x0FC | PI code (station ID) |
| B | 0x198 | Group type, PTY, flags |
| C/C' | 0x168/0x350 | Varies by group type |
| D | 0x1B4 | Varies by group type |

**CRC Polynomial**: x¹⁰ + x⁸ + x⁷ + x⁵ + x⁴ + x³ + 1 (0x1B9)

**Error Correction**: The (26,16) shortened cyclic code can correct burst errors up to 5 bits. A precomputed syndrome lookup table enables real-time correction:

```python
syndrome = compute_crc(received_block)
error_syndrome = syndrome ^ expected_offset
if error_syndrome in correction_table:
    corrected = received_block ^ error_mask
```

#### Decoded RDS Data

| Group Type | Content |
|------------|---------|
| 0A/0B | Program Service name (8 chars, station branding) |
| 2A/2B | RadioText (64 chars, now playing info) |
| 4A | Clock Time and Date (UTC with offset) |

**PI Code Decoding**: North American RBDS encodes station call letters in the PI code:
- 0x1000-0x54A7: K stations (KAAA-KZZZ)
- 0x54A8-0x994F: W stations (WAAA-WZZZ)

### Audio Processing

**Sample Rate Conversion**: Polyphase resampling from 312.5 kHz to 48 kHz audio output.

**Tone Controls**: Biquad shelf filters designed using the Audio EQ Cookbook:
- Bass boost: +3 dB low shelf at 250 Hz
- Treble boost: +3 dB high shelf at 3.5 kHz

**Soft Limiting**: Prevents harsh clipping on over-modulated stations using tanh saturation:
```python
output = tanh(input * 1.5) / tanh(1.5)
```

### Signal Quality Metrics

**S-Meter**: Calibrated to VHF/UHF standard (S9 = -93 dBm, 6 dB/S-unit)

**SNR Estimation**: Measured by comparing signal power (0-53 kHz) to noise power in an out-of-band region (75-95 kHz), scaled by bandwidth ratio.

## Usage

```bash
./pyfm.py [frequency_mhz]
```

### Controls

| Key | Function |
|-----|----------|
| ←/→ | Tune down/up 100 kHz |
| ↑/↓ | Volume up/down |
| 1-5 | Recall preset |
| !@#$% | Set preset (Shift+1-5) |
| r | Toggle RDS decoding |
| a | Toggle spectrum analyzer |
| b | Toggle bass boost |
| t | Toggle treble boost |
| Q | Toggle squelch |
| q | Quit |

### Configuration

Settings are saved to `pyfm.cfg`:
- Last tuned frequency (restored on startup)
- Frequency presets
- Tone control settings

## Requirements

- SignalHound BB60D spectrum analyzer
- SignalHound BB API and Python bindings
- Python 3.8+
- Linux with ALSA/PulseAudio

### Python Dependencies

```bash
pip install numpy scipy sounddevice rich
```

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           SignalHound BB60D                              │
│                     IQ Streaming @ 312.5 kHz                            │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        Quadrature Discriminator                          │
│                   baseband = angle(s[n] * conj(s[n-1]))                 │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
            ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
            │   L+R LPF    │ │  Pilot BPF   │ │  RDS BPF     │
            │   0-15 kHz   │ │   19 kHz     │ │   57 kHz     │
            └──────────────┘ └──────────────┘ └──────────────┘
                    │               │               │
                    │               ▼               ▼
                    │        ┌──────────────┐ ┌──────────────┐
                    │        │     PLL      │ │  Coherent    │
                    │        │   Tracker    │ │   Demod      │
                    │        └──────────────┘ └──────────────┘
                    │               │               │
                    │               ▼               ▼
                    │        ┌──────────────┐ ┌──────────────┐
                    │        │  L-R Demod   │ │  Symbol      │
                    │        │  @ 38 kHz    │ │  Recovery    │
                    │        └──────────────┘ └──────────────┘
                    │               │               │
                    ▼               ▼               ▼
            ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
            │   Stereo     │ │              │ │  Block Sync  │
            │   Matrix     │◄┤              │ │  & CRC       │
            │   L = S+D    │ │              │ └──────────────┘
            │   R = S-D    │ │              │        │
            └──────────────┘ └──────────────┘        ▼
                    │                         ┌──────────────┐
                    ▼                         │  RDS Data    │
            ┌──────────────┐                  │  PS, RT, CT  │
            │  Resample    │                  └──────────────┘
            │  → 48 kHz    │
            └──────────────┘
                    │
                    ▼
            ┌──────────────┐
            │ De-emphasis  │
            │ Tone Control │
            │ Soft Limiter │
            └──────────────┘
                    │
                    ▼
            ┌──────────────┐
            │    Audio     │
            │   Output     │
            └──────────────┘
```

## License

Copyright (c) 2026 Phil Jensen. All rights reserved.

## Acknowledgments

- SignalHound for the BB60D API
- The Rich library for terminal UI
- IEC 62106 (RDS specification)
