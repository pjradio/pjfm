#!/usr/bin/env python3
"""
RDS (Radio Data System) Decoder for FM Broadcasts

Supports two demodulation methods:
1. Coherent demodulation (default): Uses 19 kHz pilot to derive 57 kHz carrier
   via tripling (cos(3θ) = 4cos³(θ) - 3cos(θ)), then differential decoding.
2. Delay-and-multiply (fallback): Non-coherent detection when pilot unavailable.

Signal processing chain:
- Bandpass filtering at 57 kHz to extract RDS subcarrier
- BPSK demodulation (coherent or delay-and-multiply)
- Differential decoding (RDS uses differential encoding)
- Symbol timing recovery via Gardner TED
- Syndrome-based block synchronization
- Standard group decoding (PS, PTY, PI, RT)
"""

import numpy as np
from scipy import signal
from collections import deque


# RDS Constants
RDS_CARRIER_FREQ = 57000.0      # 57 kHz subcarrier (3 x 19 kHz pilot)
RDS_SYMBOL_RATE = 1187.5        # Symbols per second
RDS_BITS_PER_BLOCK = 26         # 16 data + 10 checkword
RDS_BLOCKS_PER_GROUP = 4        # A, B, C/C', D

# CRC generator polynomial: x^10 + x^8 + x^7 + x^5 + x^4 + x^3 + 1 = 0x1B9
CRC_POLY = 0x1B9

# Block offset words (added to CRC to identify block position)
OFFSET_WORDS = {
    'A':  0x0FC,
    'B':  0x198,
    'C':  0x168,
    'Cp': 0x350,  # C' used in type B groups
    'D':  0x1B4,
}

# Error correction lookup table: maps error syndromes to (error_mask, burst_length)
# Built by computing syndromes for all burst errors up to 5 bits at all positions
# Per IEC 62106, the RDS code can correct burst errors up to 5 bits
ERROR_CORRECTION_TABLE = None  # Lazily initialized


def _build_error_correction_table():
    """
    Build syndrome lookup table for burst error correction.

    The RDS (26,16) shortened cyclic code can correct:
    - All single-bit errors (burst length 1)
    - All 2-bit burst errors (burst length 2)
    - All burst errors up to 5 bits in length

    For each burst error pattern at each position, we compute what syndrome
    it would produce. When decoding, if the error syndrome matches one of
    these, we can correct the error by XORing the error pattern.
    """
    table = {}

    # For burst errors of length 1 to 5 bits
    for burst_len in range(1, 6):
        # Error pattern is burst_len consecutive 1s
        error_pattern = (1 << burst_len) - 1  # e.g., 0b11111 for 5-bit burst

        # Try all positions where this burst can fit in a 26-bit block
        # Position 0 means MSB, position 25 means LSB
        for pos in range(26 - burst_len + 1):
            # Shift error pattern to position (from MSB)
            # pos=0 means error at bits 25-21 (MSB end)
            error_mask = error_pattern << (26 - burst_len - pos)

            # Compute syndrome for this error pattern
            error_syndrome = compute_syndrome(error_mask)

            # Only store if this syndrome isn't already in table
            # (prefer shorter bursts - they're more likely)
            if error_syndrome not in table:
                table[error_syndrome] = (error_mask, burst_len)

    return table


def get_error_correction_table():
    """Get or build the error correction lookup table."""
    global ERROR_CORRECTION_TABLE
    if ERROR_CORRECTION_TABLE is None:
        ERROR_CORRECTION_TABLE = _build_error_correction_table()
    return ERROR_CORRECTION_TABLE


def correct_block(block_26bit, expected_offset):
    """
    Attempt to correct burst errors in a received RDS block.

    Args:
        block_26bit: The 26-bit received block
        expected_offset: The expected offset word for this block position

    Returns:
        (corrected_block, burst_length) if correction successful, else (None, 0)
    """
    syndrome = compute_syndrome(block_26bit)

    # If already valid, no correction needed
    if syndrome == expected_offset:
        return (block_26bit, 0)

    # Compute error syndrome by XORing with expected offset
    error_syndrome = syndrome ^ expected_offset

    # Look up in correction table
    table = get_error_correction_table()
    if error_syndrome in table:
        error_mask, burst_len = table[error_syndrome]
        # Apply correction by XORing error pattern
        corrected = block_26bit ^ error_mask

        # Verify correction worked
        if compute_syndrome(corrected) == expected_offset:
            return (corrected, burst_len)

    # Could not correct
    return (None, 0)

# PTY (Program Type) codes for RBDS (North America)
def pi_to_callsign(pi_hex):
    """
    Decode RBDS PI code to North American call letters.

    PI codes encode US/Canadian station call signs:
    - 0x1000-0x54A7: K stations (KAAA-KZZZ)
    - 0x54A8-0x994F: W stations (WAAA-WZZZ)
    - 0x9950-0x9EFF: 3-letter calls (KEX, KOA, etc.)
    - 0xA100-0xAFFF: Canadian stations

    Returns call letters or None if not decodable.
    """
    if not pi_hex:
        return None

    try:
        pi = int(pi_hex, 16)
    except (ValueError, TypeError):
        return None

    # 4-letter K calls: KAAA (0x1000) to KZZZ (0x54A7)
    if 0x1000 <= pi <= 0x54A7:
        offset = pi - 0x1000
        l4 = offset % 26
        offset //= 26
        l3 = offset % 26
        offset //= 26
        l2 = offset
        if l2 <= 25:
            return 'K' + chr(ord('A') + l2) + chr(ord('A') + l3) + chr(ord('A') + l4)

    # 4-letter W calls: WAAA (0x54A8) to WZZZ (0x994F)
    elif 0x54A8 <= pi <= 0x994F:
        offset = pi - 0x54A8
        l4 = offset % 26
        offset //= 26
        l3 = offset % 26
        offset //= 26
        l2 = offset
        if l2 <= 25:
            return 'W' + chr(ord('A') + l2) + chr(ord('A') + l3) + chr(ord('A') + l4)

    # Could add 3-letter calls and Canadian stations here if needed

    return None


PTY_NAMES = {
    0: "None", 1: "News", 2: "Information", 3: "Sports",
    4: "Talk", 5: "Rock", 6: "Classic Rock", 7: "Adult Hits",
    8: "Soft Rock", 9: "Top 40", 10: "Country", 11: "Oldies",
    12: "Soft", 13: "Nostalgia", 14: "Jazz", 15: "Classical",
    16: "R&B", 17: "Soft R&B", 18: "Language", 19: "Religious Music",
    20: "Religious Talk", 21: "Personality", 22: "Public", 23: "College",
    24: "Spanish Talk", 25: "Spanish Music", 26: "Hip Hop",
    27: "Unassigned", 28: "Unassigned", 29: "Weather",
    30: "Emergency Test", 31: "Emergency",
}


def compute_syndrome(block_26bit):
    """
    Compute syndrome for a 26-bit RDS block.

    The syndrome is the remainder when dividing the received block
    by the CRC polynomial. For a valid block, the syndrome equals
    the offset word for that block position.
    """
    reg = 0
    for i in range(25, -1, -1):
        bit = (block_26bit >> i) & 1
        feedback = (reg >> 9) & 1
        reg = ((reg << 1) | bit) & 0x3FF
        if feedback:
            reg ^= CRC_POLY
    return reg


class RDSDecoder:
    """
    RDS decoder with coherent demodulation using pilot-derived carrier.

    Signal processing chain:
    1. Bandpass filter to extract 57 kHz RDS subcarrier
    2. Extract 19 kHz pilot, triple to 57 kHz carrier
    3. Coherent demodulation + differential decoding
    4. Low-pass filter to extract baseband
    5. AGC normalization
    6. Symbol timing recovery (Gardner TED)
    7. Block synchronization via syndrome matching
    8. Group assembly and decoding
    """

    def __init__(self, sample_rate=250000):
        """
        Initialize RDS decoder.

        Args:
            sample_rate: Sample rate of input baseband signal in Hz
        """
        self.sample_rate = sample_rate
        self.samples_per_symbol = sample_rate / RDS_SYMBOL_RATE

        # Delay in samples for delay-and-multiply (one symbol period)
        self.delay_samples = int(round(self.samples_per_symbol))

        nyq = sample_rate / 2

        # Design bandpass filter for 57 kHz RDS band (+/- 2.4 kHz)
        rds_bw = 2400
        low = (RDS_CARRIER_FREQ - rds_bw) / nyq
        high = (RDS_CARRIER_FREQ + rds_bw) / nyq
        low = max(0.01, low)
        high = min(0.99, high)
        self.bpf_b, self.bpf_a = signal.butter(4, [low, high], btype='band')
        self.bpf_zi = signal.lfilter_zi(self.bpf_b, self.bpf_a) * 0

        # Design low-pass filter for baseband after multiply
        lpf_cutoff = 1200 / nyq
        self.lpf_b, self.lpf_a = signal.butter(2, lpf_cutoff, btype='low')
        self.lpf_zi = signal.lfilter_zi(self.lpf_b, self.lpf_a) * 0

        # Delay line for delay-and-multiply
        self.delay_buffer = np.zeros(self.delay_samples)

        # For coherent demod: use IIR filter for pilot to match RDS IIR filter
        # Both filters are 4th order Butterworth, so group delay characteristics match
        pilot_low = 18500 / nyq
        pilot_high = 19500 / nyq
        self.pilot_iir_b, self.pilot_iir_a = signal.butter(4, [pilot_low, pilot_high], btype='band')
        self.pilot_iir_zi = signal.lfilter_zi(self.pilot_iir_b, self.pilot_iir_a) * 0

        # Delay buffer for differential decoding in coherent mode
        self._coherent_delay_buf = np.zeros(self.delay_samples)

        # Symbol timing recovery state (Gardner TED with PI loop)
        self.timing_mu = 0.5              # Fractional sample offset within symbol (phase)
        self.timing_freq = 0.0            # Fractional accumulator for exact sample rate
        self.timing_freq_offset = 0.0     # Symbol rate offset (frequency error tracking)
        self.timing_gain_p = 0.02         # Proportional gain for phase tracking (lower = less noise)
        self.timing_gain_i = 0.003        # Integral gain for frequency tracking
        self.sample_buffer = []

        # Bit buffer and block sync state
        self.bit_buffer = []
        self.synced = False
        self.inverted = False  # Whether signal is inverted
        self.expected_block = 0
        self.group_blocks = [0, 0, 0, 0]
        self.group_valid = [False, False, False, False]  # Track validity of each block in current group
        self.sync_confidence = 0
        self.consecutive_good = 0
        self.consecutive_bad = 0

        # Decoded RDS data
        self._pi_code = 0
        self._ps_chars = [' '] * 8
        self._ps_segment_seen = [False] * 4
        self._pty = 0
        self._rt_chars = [' '] * 64
        self._rt_flag = 0
        self._ct_time = None  # (hour, minute, utc_offset_half_hours) or None

        # Statistics
        self.groups_received = 0
        self.blocks_received = 0    # Valid blocks (including corrected)
        self.blocks_expected = 0    # Total blocks attempted
        self.blocks_corrected = 0   # Blocks recovered via error correction
        self.crc_errors = 0         # Uncorrectable errors
        self.rds_signal_level = 0.0

        # Error correction statistics by burst length
        self.corrections_by_burst = [0, 0, 0, 0, 0, 0]  # Index 0 unused, 1-5 for burst lengths

        # Detailed diagnostics
        self.block_errors = [0, 0, 0, 0]  # Errors per block position (A, B, C, D)
        self.group_type_counts = {}        # Count of each group type received
        self.last_syndromes = []           # Last few syndromes for debugging
        self.bit_distribution = [0, 0]     # Count of 0s and 1s
        self.timing_mu_min = 0.5
        self.timing_mu_max = 0.5
        self.timing_mu_count = 0

        # Debug diagnostics buffer (ring buffer, no I/O during processing)
        self._diag_enabled = False
        self._diag_buffer = deque(maxlen=12000)  # ~10 seconds of symbols
        self._wrap_count_up = 0    # Times timing_mu wrapped 1.0 -> 0.0
        self._wrap_count_down = 0  # Times timing_mu wrapped 0.0 -> 1.0

    def enable_diagnostics(self, enabled=True):
        """Enable/disable diagnostic data collection."""
        self._diag_enabled = enabled
        if enabled:
            self._diag_buffer.clear()

    def get_diagnostics(self):
        """
        Get diagnostic data and statistics.

        Returns dict with:
        - 'samples': list of (timing_mu, timing_error, timing_freq_offset) tuples
        - 'histogram': timing_mu distribution in 10 bins
        - 'stats': min, max, mean, std of timing_mu
        """
        if not self._diag_buffer:
            return None

        samples = list(self._diag_buffer)
        mu_values = [s[0] for s in samples]

        # Histogram of timing_mu (10 bins from 0 to 1)
        hist, bin_edges = np.histogram(mu_values, bins=10, range=(0, 1))

        return {
            'samples': samples,
            'count': len(samples),
            'histogram': hist.tolist(),
            'bin_edges': bin_edges.tolist(),
            'stats': {
                'min': min(mu_values),
                'max': max(mu_values),
                'mean': np.mean(mu_values),
                'std': np.std(mu_values),
            },
            'wraps_up': self._wrap_count_up,
            'wraps_down': self._wrap_count_down,
        }

    def dump_diagnostics(self, filepath='/tmp/rds_timing_diag.txt'):
        """Dump diagnostic data to file (call when NOT processing)."""
        diag = self.get_diagnostics()
        if not diag:
            return False

        with open(filepath, 'w') as f:
            f.write("RDS Timing Diagnostics\n")
            f.write("=" * 50 + "\n\n")

            stats = diag['stats']
            f.write(f"Samples: {diag['count']}\n")
            f.write(f"timing_mu: min={stats['min']:.4f} max={stats['max']:.4f} "
                    f"mean={stats['mean']:.4f} std={stats['std']:.4f}\n")
            f.write(f"Wraps: up={diag['wraps_up']} (late) down={diag['wraps_down']} (early)\n\n")

            f.write("Histogram (timing_mu distribution):\n")
            for i, count in enumerate(diag['histogram']):
                lo = i * 0.1
                hi = (i + 1) * 0.1
                bar = '#' * (count // 20)
                f.write(f"  {lo:.1f}-{hi:.1f}: {count:5d} {bar}\n")

            f.write(f"\nLast 100 samples (mu, error, freq_offset):\n")
            for mu, err, foff in diag['samples'][-100:]:
                f.write(f"  {mu:.4f}  {err:+.6f}  {foff:+.6f}\n")

        return True

    def process(self, baseband, use_coherent=None):
        """
        Process baseband samples and extract RDS data.

        Args:
            baseband: FM-demodulated baseband signal at sample_rate
            use_coherent: If not None, use coherent demodulation (extracts
                         pilot from baseband). If None, use delay-and-multiply.

        Returns:
            Dictionary containing decoded RDS data and status
        """
        if len(baseband) == 0:
            return self._get_result()

        baseband = np.asarray(baseband, dtype=np.float64)

        # Bandpass filter to extract 57 kHz RDS signal
        rds_signal, self.bpf_zi = signal.lfilter(
            self.bpf_b, self.bpf_a, baseband, zi=self.bpf_zi
        )

        # Measure RDS signal level (RMS of filtered signal)
        rds_power = np.sqrt(np.mean(rds_signal ** 2))
        self.rds_signal_level = 0.9 * self.rds_signal_level + 0.1 * rds_power

        # Demodulate BPSK
        if use_coherent is not None:
            bpsk_baseband = self._coherent_demod(rds_signal, baseband)
        else:
            bpsk_baseband = self._delay_multiply_demod(rds_signal)

        # AGC - normalize signal amplitude for consistent symbol decisions
        rms = np.sqrt(np.mean(bpsk_baseband ** 2))
        if rms > 1e-10:
            bpsk_baseband = bpsk_baseband / rms

        # Symbol timing recovery and bit extraction
        self._extract_symbols(bpsk_baseband)

        return self._get_result()

    def _coherent_demod(self, rds_signal, baseband):
        """
        Coherent BPSK demodulation using pilot-derived 57 kHz carrier.

        Uses the 19 kHz pilot to generate a phase-locked 57 kHz carrier via
        the tripling identity: cos(3θ) = 4cos³(θ) - 3cos(θ)

        RDS uses differential encoding, so after carrier mixing we multiply
        by a one-symbol-delayed version to perform differential decoding.
        """
        baseband = np.asarray(baseband, dtype=np.float64)

        # Extract 19 kHz pilot using IIR BPF (matched to RDS BPF characteristics)
        pilot, self.pilot_iir_zi = signal.lfilter(
            self.pilot_iir_b, self.pilot_iir_a, baseband, zi=self.pilot_iir_zi
        )

        # Normalize pilot to unit amplitude using RMS (more stable than max)
        # For a sinusoid, RMS = amplitude / sqrt(2), so amplitude = RMS * sqrt(2)
        pilot_rms = np.sqrt(np.mean(pilot ** 2))
        if pilot_rms < 1e-10:
            return self._delay_multiply_demod(rds_signal)
        pilot_norm = pilot / (pilot_rms * np.sqrt(2))

        # Generate 57 kHz carrier: cos(3θ) = 4cos³(θ) - 3cos(θ)
        carrier_57k = 4 * pilot_norm**3 - 3 * pilot_norm

        # Mix RDS with carrier (carrier removal)
        coherent_product = rds_signal * carrier_57k

        # Differential decoding: multiply by one-symbol-delayed version
        n = len(coherent_product)
        delay = self.delay_samples

        delayed = np.zeros(n)
        if n >= delay:
            delayed[:delay] = self._coherent_delay_buf
            delayed[delay:] = coherent_product[:n - delay]
        else:
            delayed[:n] = self._coherent_delay_buf[:n]

        if n >= delay:
            self._coherent_delay_buf = coherent_product[-delay:].copy()
        else:
            self._coherent_delay_buf[:-n] = self._coherent_delay_buf[n:]
            self._coherent_delay_buf[-n:] = coherent_product

        diff_decoded = coherent_product * delayed

        # Low-pass filter to extract baseband
        bpsk_baseband, self.lpf_zi = signal.lfilter(
            self.lpf_b, self.lpf_a, diff_decoded, zi=self.lpf_zi
        )

        return bpsk_baseband

    def _delay_multiply_demod(self, rds_signal):
        """
        Delay-and-multiply demodulation (fallback when no pilot).
        """
        n = len(rds_signal)

        # Create delayed version using delay buffer for continuity
        delayed = np.zeros(n)

        if n >= self.delay_samples:
            delayed[:self.delay_samples] = self.delay_buffer
            delayed[self.delay_samples:] = rds_signal[:n - self.delay_samples]
        else:
            delayed[:n] = self.delay_buffer[:n]

        # Update delay buffer
        if n >= self.delay_samples:
            self.delay_buffer = rds_signal[-self.delay_samples:].copy()
        else:
            self.delay_buffer[:-n] = self.delay_buffer[n:]
            self.delay_buffer[-n:] = rds_signal

        # Multiply signal by delayed version
        multiplied = rds_signal * delayed

        # Low-pass filter to remove 114 kHz and extract baseband
        bpsk_baseband, self.lpf_zi = signal.lfilter(
            self.lpf_b, self.lpf_a, multiplied, zi=self.lpf_zi
        )

        return bpsk_baseband

    def _extract_symbols(self, samples):
        """
        Extract symbols with precise fractional sample tracking.

        Uses two-part timing:
        1. Fractional accumulator to track exact sample rate (263.158 samples/symbol)
        2. Gardner TED for fine phase adjustment
        """
        self.sample_buffer.extend(samples.tolist())

        sps = self.samples_per_symbol  # 263.157894...
        sps_int = int(sps)
        sps_frac = sps - sps_int       # 0.157894...
        half_sps = sps / 2.0

        # Need at least 2 symbols worth of samples plus margin
        while len(self.sample_buffer) >= sps_int * 2 + 2:
            # Use timing_mu for interpolation (phase within symbol)
            mu = self.timing_mu
            if mu < 0:
                mu = 0
            elif mu > 1:
                mu = 1

            # Sample indices with interpolation
            prev_idx = 0
            prev_sample = self.sample_buffer[prev_idx]

            # Midpoint
            mid_pos = half_sps
            mid_idx = int(mid_pos)
            mid_frac = mid_pos - mid_idx
            if mid_idx + 1 >= len(self.sample_buffer):
                break
            mid_sample = (1 - mid_frac) * self.sample_buffer[mid_idx] + mid_frac * self.sample_buffer[mid_idx + 1]

            # Current symbol
            curr_idx = sps_int
            if curr_idx + 1 >= len(self.sample_buffer):
                break
            curr_sample = (1 - mu) * self.sample_buffer[curr_idx] + mu * self.sample_buffer[curr_idx + 1]

            # Hard decision
            data_bit = 0 if curr_sample > 0 else 1
            self.bit_distribution[data_bit] += 1

            # Gardner TED for phase and frequency adjustment (PI loop)
            timing_error = -(curr_sample - prev_sample) * mid_sample
            norm = abs(curr_sample) + abs(prev_sample) + abs(mid_sample) + 1e-10
            timing_error = timing_error / norm
            # Clamp error to prevent runaway near timing_mu boundaries
            timing_error = max(-0.3, min(0.3, timing_error))

            # Proportional term: immediate phase correction
            self.timing_mu += self.timing_gain_p * timing_error

            # Integral term: frequency offset adaptation
            # This allows tracking stations that are slightly off the nominal 1187.5 Hz
            self.timing_freq_offset += self.timing_gain_i * timing_error
            # Clamp frequency offset to reasonable range (+/- 1.0 samples/symbol max)
            self.timing_freq_offset = max(-1.0, min(1.0, self.timing_freq_offset))

            # Advance buffer - use fractional accumulator to track exact rate
            # Apply frequency offset to adjust effective samples per symbol
            effective_sps_frac = sps_frac + self.timing_freq_offset
            self.timing_freq += effective_sps_frac
            advance = sps_int
            while self.timing_freq >= 1.0:
                advance += 1
                self.timing_freq -= 1.0
            while self.timing_freq < 0.0:
                advance -= 1
                self.timing_freq += 1.0

            # Allow timing_mu to wrap and adjust advance accordingly
            # This is critical - without wrapping, timing gets stuck at boundaries
            while self.timing_mu >= 1.0:
                self.timing_mu -= 1.0
                advance += 1  # We're late, consume extra sample
                self._wrap_count_up += 1
            while self.timing_mu < 0.0:
                self.timing_mu += 1.0
                advance -= 1  # We're early, consume fewer samples
                self._wrap_count_down += 1

            # Safety: ensure we always advance at least 1 sample
            if advance < 1:
                advance = 1

            # Diagnostic logging (cheap append to deque, no I/O)
            if self._diag_enabled:
                self._diag_buffer.append((self.timing_mu, timing_error, self.timing_freq_offset))

            # Track statistics (after wrapping, so we see actual operating range)
            self.timing_mu_min = min(self.timing_mu_min, self.timing_mu)
            self.timing_mu_max = max(self.timing_mu_max, self.timing_mu)
            self.timing_mu_count += 1
            if self.timing_mu_count >= 1187:
                self.timing_mu_min = self.timing_mu
                self.timing_mu_max = self.timing_mu
                self.timing_mu_count = 1

            self.sample_buffer = self.sample_buffer[advance:]
            self.bit_buffer.append(data_bit)

            if len(self.bit_buffer) >= RDS_BITS_PER_BLOCK:
                self._process_bits()

    def _process_bits(self):
        """Process accumulated bits to find and decode blocks."""
        if len(self.bit_buffer) < RDS_BITS_PER_BLOCK:
            return

        if not self.synced:
            self._search_sync()
        else:
            self._decode_block()

    def _search_sync(self):
        """Search for block sync by checking syndromes."""
        if len(self.bit_buffer) < RDS_BITS_PER_BLOCK:
            return

        block = self._bits_to_int(self.bit_buffer[:RDS_BITS_PER_BLOCK])

        # Try normal and inverted
        for invert in [False, True]:
            check_block = block if not invert else (block ^ 0x3FFFFFF)
            check_syndrome = compute_syndrome(check_block)

            for block_name, offset in OFFSET_WORDS.items():
                if check_syndrome == offset:
                    block_idx = {'A': 0, 'B': 1, 'C': 2, 'Cp': 2, 'D': 3}[block_name]
                    self.expected_block = (block_idx + 1) % 4
                    self.group_blocks[block_idx] = (check_block >> 10) & 0xFFFF
                    self.synced = True
                    self.inverted = invert  # Remember if we need to invert
                    self.sync_confidence = 1
                    self.consecutive_good = 1
                    self.consecutive_bad = 0
                    self.bit_buffer = self.bit_buffer[RDS_BITS_PER_BLOCK:]
                    return

        # No sync, slide by one bit
        self.bit_buffer = self.bit_buffer[1:]

    def _decode_block(self):
        """Decode a block when synchronized."""
        if len(self.bit_buffer) < RDS_BITS_PER_BLOCK:
            return

        block = self._bits_to_int(self.bit_buffer[:RDS_BITS_PER_BLOCK])
        self.bit_buffer = self.bit_buffer[RDS_BITS_PER_BLOCK:]

        # Apply inversion if we synced with inverted polarity
        if self.inverted:
            block = block ^ 0x3FFFFFF

        syndrome = compute_syndrome(block)

        block_names = ['A', 'B', 'C', 'D']
        expected_name = block_names[self.expected_block]
        expected_offset = OFFSET_WORDS[expected_name]

        valid = (syndrome == expected_offset)
        corrected = False
        burst_len = 0

        # For block C, also try C' offset
        if not valid and self.expected_block == 2:
            valid = (syndrome == OFFSET_WORDS['Cp'])

        # If still not valid, attempt error correction
        if not valid:
            # Try primary offset first
            corrected_block, burst_len = correct_block(block, expected_offset)

            # For block C, also try C' offset if primary failed
            if corrected_block is None and self.expected_block == 2:
                corrected_block, burst_len = correct_block(block, OFFSET_WORDS['Cp'])

            if corrected_block is not None:
                block = corrected_block
                valid = True
                corrected = True

        # Reset validity tracking at start of new group
        if self.expected_block == 0:
            self.group_valid = [False, False, False, False]

        self.blocks_expected += 1

        if valid:
            data = (block >> 10) & 0xFFFF
            self.group_blocks[self.expected_block] = data
            self.group_valid[self.expected_block] = True
            self.blocks_received += 1
            self.consecutive_good += 1
            self.consecutive_bad = 0
            self.sync_confidence = min(10, self.sync_confidence + 1)

            # Track correction statistics
            if corrected:
                self.blocks_corrected += 1
                if 1 <= burst_len <= 5:
                    self.corrections_by_burst[burst_len] += 1

            # Only decode if we have all 4 valid blocks
            if self.expected_block == 3 and all(self.group_valid):
                self._decode_group()
        else:
            self.crc_errors += 1
            self.block_errors[self.expected_block] += 1
            self.group_valid[self.expected_block] = False
            self.consecutive_bad += 1
            self.consecutive_good = 0
            self.sync_confidence = max(0, self.sync_confidence - 1)
            # Track recent syndromes for debugging
            self.last_syndromes.append((self.expected_block, syndrome))
            if len(self.last_syndromes) > 20:
                self.last_syndromes.pop(0)

            if self.consecutive_bad >= 25:
                # Try to re-sync at current position before giving up
                self.synced = False
                self.sync_confidence = 0
                return

        self.expected_block = (self.expected_block + 1) % 4

    def _bits_to_int(self, bits):
        """Convert list of bits to integer, MSB first."""
        val = 0
        for b in bits:
            val = (val << 1) | (b & 1)
        return val

    def _decode_group(self):
        """Decode a complete RDS group."""
        self.groups_received += 1

        block_a = self.group_blocks[0]
        block_b = self.group_blocks[1]
        block_c = self.group_blocks[2]
        block_d = self.group_blocks[3]

        self._pi_code = block_a

        group_type = (block_b >> 12) & 0x0F
        version = (block_b >> 11) & 0x01
        pty = (block_b >> 5) & 0x1F
        self._pty = pty

        # Track group type distribution
        gt_key = f"{group_type}{'A' if version == 0 else 'B'}"
        self.group_type_counts[gt_key] = self.group_type_counts.get(gt_key, 0) + 1

        if group_type == 0:
            self._decode_group_0(block_b, block_c, block_d, version)
        elif group_type == 2:
            self._decode_group_2(block_b, block_c, block_d, version)
        elif group_type == 4 and version == 0:
            self._decode_group_4a(block_b, block_c, block_d)

    def _decode_group_0(self, block_b, block_c, block_d, version):
        """Decode Group 0A/0B: Program Service name."""
        segment = block_b & 0x03
        char1 = (block_d >> 8) & 0xFF
        char2 = block_d & 0xFF

        pos = segment * 2
        if 0x20 <= char1 <= 0x7E:
            self._ps_chars[pos] = chr(char1)
        if 0x20 <= char2 <= 0x7E:
            self._ps_chars[pos + 1] = chr(char2)

        self._ps_segment_seen[segment] = True

    def _decode_group_2(self, block_b, block_c, block_d, version):
        """Decode Group 2A/2B: RadioText."""
        text_flag = (block_b >> 4) & 0x01
        segment = block_b & 0x0F

        if text_flag != self._rt_flag:
            self._rt_flag = text_flag
            self._rt_chars = [' '] * 64

        if version == 0:
            pos = segment * 4
            chars = [
                (block_c >> 8) & 0xFF,
                block_c & 0xFF,
                (block_d >> 8) & 0xFF,
                block_d & 0xFF,
            ]
            for i, c in enumerate(chars):
                if pos + i < 64:
                    if 0x20 <= c <= 0x7E:
                        self._rt_chars[pos + i] = chr(c)
                    elif c == 0x0D:
                        self._rt_chars[pos + i] = ' '
        else:
            pos = segment * 2
            char1 = (block_d >> 8) & 0xFF
            char2 = block_d & 0xFF
            if pos < 64 and 0x20 <= char1 <= 0x7E:
                self._rt_chars[pos] = chr(char1)
            if pos + 1 < 64 and 0x20 <= char2 <= 0x7E:
                self._rt_chars[pos + 1] = chr(char2)

    def _decode_group_4a(self, block_b, block_c, block_d):
        """
        Decode Group 4A: Clock Time and Date (CT).

        Format:
        - MJD (Modified Julian Day): 17 bits spanning block_b[1:0] and block_c[15:1]
        - Hour: 5 bits spanning block_c[0] and block_d[15:12]
        - Minute: 6 bits in block_d[11:6]
        - UTC offset: 6 bits in block_d[5:0], signed (half-hours from UTC)
        """
        # Extract MJD (17 bits)
        mjd = ((block_b & 0x03) << 15) | ((block_c >> 1) & 0x7FFF)

        # Extract time
        hour = ((block_c & 0x01) << 4) | ((block_d >> 12) & 0x0F)
        minute = (block_d >> 6) & 0x3F
        offset = block_d & 0x3F

        # Sign-extend offset (6-bit signed value, half-hours from UTC)
        if offset & 0x20:
            offset = offset - 64  # Convert to signed

        # Validate ranges
        if hour < 24 and minute < 60 and -24 <= offset <= 24:
            self._ct_time = (hour, minute, offset)

    def _get_result(self):
        """Build result dictionary."""
        # Calculate block success rate
        if self.blocks_expected > 0:
            block_rate = self.blocks_received / self.blocks_expected
        else:
            block_rate = 0.0

        # Calculate bit balance (should be ~50/50 for random data)
        total_bits = self.bit_distribution[0] + self.bit_distribution[1]
        if total_bits > 0:
            bit_balance = self.bit_distribution[0] / total_bits
        else:
            bit_balance = 0.5

        return {
            'pi_code': self._pi_code,
            'pi_hex': f'{self._pi_code:04X}' if self._pi_code else '',
            'station_name': self.station_name,
            'program_type': self.program_type,
            'pty_code': self._pty,
            'radio_text': self.radio_text,
            'clock_time': self.clock_time,
            'synced': self.synced and self.sync_confidence >= 3,
            'groups_received': self.groups_received,
            'blocks_received': self.blocks_received,
            'blocks_expected': self.blocks_expected,
            'blocks_corrected': self.blocks_corrected,
            'block_rate': block_rate,
            'crc_errors': self.crc_errors,
            'signal_level': self.rds_signal_level,
            'lock_level': self.sync_confidence / 10.0,
            # Diagnostics
            'block_errors': self.block_errors.copy(),
            'group_types': dict(self.group_type_counts),
            'bit_balance': bit_balance,
            'samples_per_symbol': self.samples_per_symbol,
            'sample_rate': self.sample_rate,
            'timing_range': (self.timing_mu_min, self.timing_mu_max),
            'timing_freq': self.timing_freq,
            'timing_freq_offset': self.timing_freq_offset,
            # Error correction statistics
            'corrections_by_burst': self.corrections_by_burst[1:].copy(),  # Burst lengths 1-5
        }

    @property
    def station_name(self):
        """Get decoded station name (PS) only when complete."""
        # Only return PS name when all 4 segments have been received
        if all(self._ps_segment_seen):
            name = ''.join(self._ps_chars).strip()
            return name if name else None
        return None

    @property
    def program_type(self):
        """Get program type as string."""
        return PTY_NAMES.get(self._pty, f"Unknown ({self._pty})")

    @property
    def radio_text(self):
        """Get decoded RadioText if available."""
        text = ''.join(self._rt_chars).strip()
        return text if text else None

    @property
    def clock_time(self):
        """Get decoded clock time as formatted string, or None if not received."""
        if self._ct_time is None:
            return None
        hour, minute, offset = self._ct_time
        # Format offset as +/-HH:MM
        offset_hours = offset // 2
        offset_half = (offset % 2) * 30
        if offset >= 0:
            offset_str = f"+{offset_hours}:{offset_half:02d}"
        else:
            offset_str = f"{offset_hours}:{abs(offset_half):02d}"
        return f"{hour:02d}:{minute:02d} UTC{offset_str}"

    def reset(self):
        """Reset decoder state."""
        # Filter states
        self.bpf_zi = signal.lfilter_zi(self.bpf_b, self.bpf_a) * 0
        self.lpf_zi = signal.lfilter_zi(self.lpf_b, self.lpf_a) * 0
        self.pilot_iir_zi = signal.lfilter_zi(self.pilot_iir_b, self.pilot_iir_a) * 0

        # Delay buffers
        self.delay_buffer = np.zeros(self.delay_samples)
        self._coherent_delay_buf = np.zeros(self.delay_samples)

        # Timing recovery
        self.timing_mu = 0.5
        self.timing_freq = 0.0
        self.timing_freq_offset = 0.0
        self.sample_buffer = []
        self.timing_mu_min = 0.5
        self.timing_mu_max = 0.5
        self.timing_mu_count = 0

        # Block sync
        self.bit_buffer = []
        self.synced = False
        self.inverted = False
        self.expected_block = 0
        self.group_blocks = [0, 0, 0, 0]
        self.group_valid = [False, False, False, False]
        self.sync_confidence = 0
        self.consecutive_good = 0
        self.consecutive_bad = 0

        # Decoded data
        self._pi_code = 0
        self._ps_chars = [' '] * 8
        self._ps_segment_seen = [False] * 4
        self._pty = 0
        self._rt_chars = [' '] * 64
        self._rt_flag = 0
        self._ct_time = None

        # Statistics
        self.groups_received = 0
        self.blocks_received = 0
        self.blocks_expected = 0
        self.blocks_corrected = 0
        self.crc_errors = 0
        self.rds_signal_level = 0.0
        self.block_errors = [0, 0, 0, 0]
        self.corrections_by_burst = [0, 0, 0, 0, 0, 0]
        self.group_type_counts = {}
        self.last_syndromes = []
        self.bit_distribution = [0, 0]

        # Diagnostics
        self._diag_buffer.clear()
        self._wrap_count_up = 0
        self._wrap_count_down = 0
