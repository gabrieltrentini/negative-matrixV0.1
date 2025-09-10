import numpy as np
import sounddevice as sd
import tkinter as tk
from tkinter import ttk
from scipy.signal import butter, lfilter, freqz
import random
import threading
import time
import math

# Try to import matplotlib, but make it optional
try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Oscilloscope and Spectrogram will be disabled.")

# -----------------------------
# ADSR Envelope Class
# -----------------------------
class ADSREnvelope:
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
        self.attack = 0.01
        self.decay = 0.1
        self.sustain = 0.7
        self.release = 0.3
        self.phase = "off"  # "off", "attack", "decay", "sustain", "release"
        self.value = 0.0
        self.time_in_phase = 0.0
        self.gate = False
        
    def set_adsr(self, attack, decay, sustain, release):
        self.attack = max(0.001, attack)
        self.decay = max(0.001, decay)
        self.sustain = max(0.0, min(1.0, sustain))
        self.release = max(0.001, release)
    
    def trigger(self):
        self.gate = True
        self.phase = "attack"
        self.time_in_phase = 0.0
    
    def release_trigger(self):
        self.gate = False
        if self.phase in ["attack", "decay", "sustain"]:
            self.phase = "release"
            self.time_in_phase = 0.0
    
    def process(self, frames):
        output = np.zeros(frames)
        
        for i in range(frames):
            dt = 1.0 / self.sample_rate
            
            if self.phase == "attack" and self.gate:
                self.value += dt / self.attack
                if self.value >= 1.0:
                    self.value = 1.0
                    self.phase = "decay"
                    self.time_in_phase = 0.0
                    
            elif self.phase == "decay" and self.gate:
                target = self.sustain
                decay_rate = (1.0 - target) / self.decay
                self.value -= decay_rate * dt
                if self.value <= target:
                    self.value = target
                    self.phase = "sustain"
                    
            elif self.phase == "sustain" and self.gate:
                self.value = self.sustain
                
            elif self.phase == "release" or (not self.gate and self.phase != "off"):
                if self.phase != "release":
                    self.phase = "release"
                    self.time_in_phase = 0.0
                self.value -= self.value * dt / self.release
                if self.value <= 0.001:
                    self.value = 0.0
                    self.phase = "off"
                    
            output[i] = self.value
            self.time_in_phase += dt
            
        return output

# -----------------------------
# Psy Effects Classes
# -----------------------------
class PsyDistortion:
    def __init__(self):
        self.drive = 1.0
        self.type = "soft"
        
    def process(self, signal, drive=None, distortion_type=None):
        if drive is None:
            drive = self.drive
        if distortion_type is None:
            distortion_type = self.type
            
        if drive <= 1.0:
            return signal
            
        if distortion_type == "soft":
            return np.tanh(signal * drive) / np.tanh(drive)
        elif distortion_type == "hard":
            return np.clip(signal * drive, -1.0, 1.0)
        elif distortion_type == "tube":
            return np.sign(signal) * (1 - np.exp(-np.abs(signal * drive)))
        return signal

class PsyChorus:
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
        self.delay_buffers = [np.zeros(int(sample_rate * 0.05)) for _ in range(3)]
        self.delay_indices = [0, 0, 0]
        self.lfo_phase = 0.0
        self.rate = 0.5
        self.depth = 0.3
        self.mix = 0.5
        
    def process(self, signal):
        if self.mix <= 0:
            return signal
            
        output = signal.copy()
        lfo_inc = 2 * np.pi * self.rate / self.sample_rate
        
        for i, sample in enumerate(signal):
            self.lfo_phase += lfo_inc
            
            # Three chorus voices with different delays and LFO phases
            chorus_sum = 0
            for voice in range(3):
                lfo_offset = voice * 2 * np.pi / 3  # 120 degrees apart
                lfo_val = np.sin(self.lfo_phase + lfo_offset)
                
                # Variable delay based on LFO
                base_delay = 0.01 + voice * 0.003  # 10ms, 13ms, 16ms base delays
                mod_delay = base_delay + (lfo_val * self.depth * 0.005)
                delay_samples = int(mod_delay * self.sample_rate)
                
                # Read from delay buffer
                buf = self.delay_buffers[voice]
                buf_len = len(buf)
                read_idx = (self.delay_indices[voice] - delay_samples) % buf_len
                
                chorus_sum += buf[read_idx]
                
                # Write to delay buffer
                buf[self.delay_indices[voice]] = sample
                self.delay_indices[voice] = (self.delay_indices[voice] + 1) % buf_len
            
            output[i] = sample + (chorus_sum / 3.0) * self.mix
            
        return output

class PsyGate:
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
        self.phase = 0.0
        self.rate = 16  # 16th notes
        self.depth = 0.8
        self.pattern = [1,0,1,0, 1,0,1,1, 1,0,1,0, 1,1,0,1]  # Full-on pattern
        self.pattern_idx = 0
        
    def process(self, signal, bpm):
        if self.depth <= 0:
            return signal
            
        output = np.zeros_like(signal)
        gate_freq = (bpm / 60.0) * (self.rate / 4.0)
        
        for i, sample in enumerate(signal):
            # Calculate gate pattern position
            pattern_freq = gate_freq / len(self.pattern)
            self.phase += pattern_freq / self.sample_rate
            
            if self.phase >= 1.0:
                self.phase -= 1.0
                self.pattern_idx = (self.pattern_idx + 1) % len(self.pattern)
            
            gate_value = self.pattern[self.pattern_idx]
            gate_amplitude = gate_value * self.depth + (1.0 - self.depth)
            output[i] = sample * gate_amplitude
            
        return output

class NoiseGenerator:
    def __init__(self):
        self.burst_prob = 0.1
        self.burst_length = 100
        self.burst_counter = 0
        self.burst_active = False
        
    def generate_burst(self, frames):
        output = np.zeros(frames)
        
        for i in range(frames):
            if not self.burst_active and random.random() < self.burst_prob:
                self.burst_active = True
                self.burst_counter = self.burst_length
                
            if self.burst_active:
                output[i] = random.uniform(-0.3, 0.3)
                self.burst_counter -= 1
                if self.burst_counter <= 0:
                    self.burst_active = False
                    
        return output

# -----------------------------
# Enhanced Oscillator + Filter Engine
# -----------------------------
class Osc1Synth:
    def __init__(self, sample_rate=44100, osc_id="OSC1"):
        self.sample_rate = sample_rate
        self.osc_id = osc_id
        self.phase = 0.0
        self.freq = 220.0
        self.wave_type = "Sine"
        self.filter_type = "Lowpass"
        self.cutoff = 1000.0
        self.resonance = 0.7
        self.octave_shift = 0
        self.master_pitch_shift = 0
        self.enabled = True

        # ADSR Envelope
        self.envelope = ADSREnvelope(sample_rate)
        
        # Psy Effects
        self.distortion = PsyDistortion()
        self.chorus = PsyChorus(sample_rate)
        self.gate = PsyGate(sample_rate)
        self.noise_gen = NoiseGenerator()
        
        # Psy effect parameters
        self.psy_distortion_amount = 0.0
        self.psy_distortion_type = "soft"
        self.psy_chorus_mix = 0.0
        self.psy_chorus_rate = 0.5
        self.psy_chorus_depth = 0.3
        self.psy_gate_depth = 0.0
        self.psy_gate_rate = 16
        self.psy_noise_amount = 0.0
        self.fm_amount = 0.0
        self.fm_ratio = 2.0

        # Delay parameters (existing)
        self.delay_on = False
        self.delay_amount = 0.5
        self.delay_time_sec = 0.3
        self.delay_buffer = np.zeros(int(self.sample_rate*2), dtype=np.float32)
        self.delay_idx = 0

        # Sample & Hold (existing)
        self.snh_value = 0.8
        self.snh_phase = 0.0

        # LFO parameters (existing)
        self.lfo_on = False
        self.lfo_rate = 5.0
        self.lfo_depth = 0.0
        self.lfo_phase = 0.0
        self.lfo_mode = "Hz"
        self.bpm = 120
        self.lfo_sync_div = "1/4"

        # Modulation targets (existing)
        self.mod_freq = 0.0
        self.mod_cutoff = 0.0
        self.mod_resonance = 0.0
        self.mod_delay_time = 0.0
        self.mod_delay_amount = 0.0

        # Arpeggiator parameters (existing)
        self.arp_on = False
        self.arp_rate = 4.0
        self.arp_mode = "Hz"
        self.arp_sync_div = "1/8"
        self.arp_notes = [0]*8
        self.arp_octaves = [0]*8
        self.arp_idx = 0
        self.arp_phase_samples = 0.0

    def set_enabled(self, enabled):
        self.enabled = enabled

    def set_wave(self, wave_type):
        self.wave_type = wave_type

    def set_frequency(self, freq):
        self.freq = freq

    def set_filter(self, filter_type, cutoff, resonance):
        self.filter_type = filter_type
        self.cutoff = cutoff
        self.resonance = resonance

    def set_delay(self, on, amount, time_sec=None):
        self.delay_on = on
        self.delay_amount = amount
        if time_sec is not None:
            self.delay_time_sec = time_sec

    def set_octave(self, shift):
        try:
            self.octave_shift = int(shift)
        except:
            self.octave_shift = 0

    def set_master_pitch(self, shift):
        try:
            self.master_pitch_shift = int(shift)
        except:
            self.master_pitch_shift = 0

    def set_lfo(self, on=None, rate=None, depth=None, mode=None, sync_div=None):
        if on is not None:
            self.lfo_on = on
        if rate is not None:
            self.lfo_rate = rate
        if depth is not None:
            self.lfo_depth = depth
        if mode is not None:
            self.lfo_mode = mode
        if sync_div is not None:
            self.lfo_sync_div = sync_div

    # New methods for psy effects
    def set_psy_distortion(self, amount, distortion_type="soft"):
        self.psy_distortion_amount = amount
        self.psy_distortion_type = distortion_type
        
    def set_psy_chorus(self, mix, rate=None, depth=None):
        self.psy_chorus_mix = mix
        if rate is not None:
            self.psy_chorus_rate = rate
            self.chorus.rate = rate
        if depth is not None:
            self.psy_chorus_depth = depth
            self.chorus.depth = depth
        self.chorus.mix = mix
        
    def set_psy_gate(self, depth, rate=None):
        self.psy_gate_depth = depth
        if rate is not None:
            self.psy_gate_rate = rate
            self.gate.rate = rate
        self.gate.depth = depth
        
    def set_psy_noise(self, amount):
        self.psy_noise_amount = amount
        
    def set_fm(self, amount, ratio=None):
        self.fm_amount = amount
        if ratio is not None:
            self.fm_ratio = ratio

    def trigger_envelope(self):
        self.envelope.trigger()
        
    def release_envelope(self):
        self.envelope.release_trigger()

    def get_lfo_freq(self):
        if self.lfo_mode == "Hz":
            return self.lfo_rate
        else:
            note_lengths = {
                "1/1": 4.0, "1/2": 2.0, "1/4": 1.0, "1/8": 0.5, "1/16": 0.25
            }
            beats = note_lengths.get(self.lfo_sync_div, 1.0)
            seconds_per_beat = 60.0 / self.bpm
            period = beats * seconds_per_beat
            if period <= 0:
                return 1.0
            return 1.0 / period

    def get_lfo_value(self, frames):
        if not self.lfo_on:
            return np.zeros(frames)
        
        lfo_freq = self.get_lfo_freq()
        lfo_t = (np.arange(frames) + self.lfo_phase) / self.sample_rate
        return np.sin(2 * np.pi * lfo_freq * lfo_t)

    def get_arp_rate_hz(self):
        if self.arp_mode == "Hz":
            return max(0.01, self.arp_rate)
        else:
            note_lengths = {
                "1/1": 4.0, "1/2": 2.0, "1/4": 1.0, "1/8": 0.5, "1/16": 0.25
            }
            beats = note_lengths.get(self.arp_sync_div, 0.5)
            seconds_per_beat = 60.0 / self.bpm
            period = beats * seconds_per_beat
            if period <= 0:
                return 1.0
            return 1.0 / period

    def generate_wave(self, frames):
        if not self.enabled:
            return np.zeros(frames, dtype=np.float32)

        lfo_signal = self.get_lfo_value(frames)
        freq = self.freq * (2 ** (self.octave_shift + self.master_pitch_shift))
        
        if self.arp_on:
            step_rate = self.get_arp_rate_hz()
            samples_per_step = max(1.0, self.sample_rate / step_rate)
            self.arp_phase_samples += frames
            step_adv = int(self.arp_phase_samples // samples_per_step)
            if step_adv > 0:
                self.arp_phase_samples -= step_adv * samples_per_step
                self.arp_idx = (self.arp_idx + step_adv) % max(1, len(self.arp_notes))
            semitone = self.arp_notes[self.arp_idx] + 12 * self.arp_octaves[self.arp_idx]
            freq = freq * (2.0 ** (semitone / 12.0))

        if self.lfo_on and self.lfo_depth > 0:
            mod_amount = freq * self.lfo_depth
            freq_array = freq + lfo_signal * mod_amount
        else:
            freq_array = np.full(frames, freq)
        
        freq_array += self.mod_freq
        freq_array = np.maximum(1.0, freq_array)

        t = (np.arange(frames) + self.phase) / self.sample_rate

        # Basic waveform generation
        if self.wave_type == "Sine":
            # Add FM synthesis
            if self.fm_amount > 0:
                fm_signal = np.sin(2 * np.pi * freq_array * self.fm_ratio * t) * self.fm_amount
                wave = np.sin(2 * np.pi * freq_array * t + fm_signal)
            else:
                wave = np.sin(2 * np.pi * freq_array * t)
        elif self.wave_type == "Saw":
            frac = (freq_array * t) - np.floor(freq_array * t + 0.5)
            wave = 2.0 * frac
        elif self.wave_type == "Square":
            wave = np.sign(np.sin(2 * np.pi * freq_array * t))
        elif self.wave_type == "Triangle":
            frac = (freq_array * t) - np.floor(freq_array * t + 0.5)
            wave = 2.0 * np.abs(2.0 * frac) - 1.0
        elif self.wave_type == "Sample&Hold":
            wave = np.zeros(frames, dtype=np.float32)
            for i in range(frames):
                self.snh_phase += freq_array[i] / self.sample_rate
                if self.snh_phase >= 1.0:
                    self.snh_value = random.uniform(-1, 1)
                    self.snh_phase -= 1.0
                wave[i] = self.snh_value
        else:
            wave = np.zeros(frames, dtype=np.float32)

        # Add noise bursts for Forest Psy
        if self.psy_noise_amount > 0:
            noise = self.noise_gen.generate_burst(frames)
            wave = wave + noise * self.psy_noise_amount

        self.phase += frames
        if self.lfo_on:
            self.lfo_phase += frames

        # Apply ADSR envelope
        envelope_values = self.envelope.process(frames)
        wave = wave * envelope_values

        # Apply delay (existing)
        if self.delay_on:
            delayed_wave = np.zeros_like(wave)
            modulated_delay_time = max(0.001, self.delay_time_sec + self.mod_delay_time)
            modulated_delay_amount = max(0.0, min(1.0, self.delay_amount + self.mod_delay_amount))
            
            delay_samples = int(modulated_delay_time * self.sample_rate)
            buf_len = len(self.delay_buffer)
            for i in range(frames):
                read_idx = (self.delay_idx - delay_samples) % buf_len
                delayed_wave[i] = wave[i] + self.delay_buffer[read_idx] * modulated_delay_amount
                self.delay_buffer[self.delay_idx] = delayed_wave[i]
                self.delay_idx = (self.delay_idx + 1) % buf_len
            wave = delayed_wave

        return wave.astype(np.float32)

    def apply_filter(self, signal):
        nyq = 0.5 * self.sample_rate
        modulated_cutoff = max(50.0, min(8000.0, self.cutoff + self.mod_cutoff))
        modulated_resonance = max(0.1, min(2.0, self.resonance + self.mod_resonance))
        
        normal_cutoff = min(max(modulated_cutoff / nyq, 0.0001), 0.99)
        if self.filter_type == "Lowpass":
            b, a = butter(2, normal_cutoff, btype="low", analog=False)
        elif self.filter_type == "Highpass":
            b, a = butter(2, normal_cutoff, btype="high", analog=False)
        elif self.filter_type == "Bandpass":
            low = max(0.0001, (modulated_cutoff * max(0.0001, (1 - modulated_resonance))) / nyq)
            high = min(0.9999, (modulated_cutoff * (1 + max(0.0001, modulated_resonance))) / nyq)
            b, a = butter(2, [low, high], btype="band")
        elif self.filter_type == "Notch":
            # New filter type for Forest Psy
            low = max(0.0001, (modulated_cutoff * 0.9) / nyq)
            high = min(0.9999, (modulated_cutoff * 1.1) / nyq)
            b, a = butter(2, [low, high], btype="bandstop")
        elif self.filter_type == "Comb":
            # Simple comb filter implementation
            delay_samples = int(self.sample_rate / modulated_cutoff)
            if not hasattr(self, 'comb_buffer'):
                self.comb_buffer = np.zeros(delay_samples)
                self.comb_idx = 0
            
            output = np.zeros_like(signal)
            for i, sample in enumerate(signal):
                delayed = self.comb_buffer[self.comb_idx]
                output[i] = sample + delayed * modulated_resonance
                self.comb_buffer[self.comb_idx] = output[i]
                self.comb_idx = (self.comb_idx + 1) % len(self.comb_buffer)
            return output
        else:
            return signal
            
        try:
            return lfilter(b, a, signal)
        except Exception:
            return signal

    def apply_psy_effects(self, signal):
        # Apply distortion
        if self.psy_distortion_amount > 1.0:
            signal = self.distortion.process(signal, self.psy_distortion_amount, self.psy_distortion_type)
        
        # Apply chorus
        if self.psy_chorus_mix > 0:
            signal = self.chorus.process(signal)
            
        # Apply gate
        if self.psy_gate_depth > 0:
            signal = self.gate.process(signal, self.bpm)
            
        return signal

    def get_filter_response(self, frequencies):
        """Calculate filter frequency response for visualization"""
        nyq = 0.5 * self.sample_rate
        modulated_cutoff = max(50.0, min(8000.0, self.cutoff + self.mod_cutoff))
        modulated_resonance = max(0.1, min(2.0, self.resonance + self.mod_resonance))
        
        normal_cutoff = min(max(modulated_cutoff / nyq, 0.0001), 0.99)
        
        try:
            if self.filter_type == "Lowpass":
                b, a = butter(2, normal_cutoff, btype="low", analog=False)
            elif self.filter_type == "Highpass":
                b, a = butter(2, normal_cutoff, btype="high", analog=False)
            elif self.filter_type == "Bandpass":
                low = max(0.0001, (modulated_cutoff * max(0.0001, (1 - modulated_resonance))) / nyq)
                high = min(0.9999, (modulated_cutoff * (1 + max(0.0001, modulated_resonance))) / nyq)
                b, a = butter(2, [low, high], btype="band")
            elif self.filter_type == "Notch":
                low = max(0.0001, (modulated_cutoff * 0.9) / nyq)
                high = min(0.9999, (modulated_cutoff * 1.1) / nyq)
                b, a = butter(2, [low, high], btype="bandstop")
            else:
                return np.ones(len(frequencies))
            
            w, h = freqz(b, a, worN=frequencies, fs=self.sample_rate)
            return np.abs(h)
        except Exception:
            return np.ones(len(frequencies))

# -----------------------------
# Enhanced Modulation Matrix
# -----------------------------
class ModulationMatrix:
    def __init__(self):
        self.connections = {}
        
    def set_connection(self, source, target, amount):
        if source not in self.connections:
            self.connections[source] = {}
        self.connections[source][target] = amount
    
    def get_connection(self, source, target):
        if source in self.connections and target in self.connections[source]:
            return self.connections[source][target]
        return 0.0
    
    def apply_modulation(self, synths, frames):
        # Reset all modulation targets
        for synth in synths:
            synth.mod_freq = synth.mod_cutoff = synth.mod_resonance = 0.0
            synth.mod_delay_time = synth.mod_delay_amount = 0.0
        
        # Calculate LFO values
        lfo_values = {}
        for i, synth in enumerate(synths):
            if synth.lfo_on:
                lfo_values[f"LFO{i+1}"] = np.mean(synth.get_lfo_value(frames))
        
        # Apply modulations
        for source, targets in self.connections.items():
            if source in lfo_values:
                source_value = lfo_values[source]
                
                for target, amount in targets.items():
                    mod_value = source_value * amount
                    
                    # Parse target (e.g., "OSC1_FREQ", "OSC2_CUTOFF")
                    parts = target.split('_')
                    if len(parts) >= 2:
                        osc_num = int(parts[0].replace('OSC', '')) - 1
                        param = '_'.join(parts[1:])
                        
                        if 0 <= osc_num < len(synths):
                            synth = synths[osc_num]
                            
                            if param == "FREQ":
                                synth.mod_freq = mod_value * 100
                            elif param == "CUTOFF":
                                synth.mod_cutoff = mod_value * 2000
                            elif param == "RESONANCE":
                                synth.mod_resonance = mod_value * 0.5
                            elif param == "DELAY_TIME":
                                synth.mod_delay_time = mod_value * 0.2
                            elif param == "DELAY_AMOUNT":
                                synth.mod_delay_amount = mod_value * 0.3

# -----------------------------
# Oscilloscope Class (existing)
# -----------------------------
class Oscilloscope:
    def __init__(self):
        self.enabled = False
        self.buffer_size = 1024
        self.signal_buffer = np.zeros(self.buffer_size)
        self.time_axis = np.linspace(0, self.buffer_size/44100, self.buffer_size)
        
    def update_signal(self, signal):
        if self.enabled and len(signal) > 0:
            if len(signal) >= self.buffer_size:
                self.signal_buffer = signal[-self.buffer_size:]
            else:
                shift_amount = len(signal)
                self.signal_buffer[:-shift_amount] = self.signal_buffer[shift_amount:]
                self.signal_buffer[-shift_amount:] = signal

# -----------------------------
# Spectrogram Class (existing)
# -----------------------------
class FilterSpectrogram:
    def __init__(self):
        self.enabled = False
        self.frequencies = np.logspace(1, np.log10(22000), 512)  # 10 Hz to 22 kHz
        
    def get_combined_response(self, synths):
        """Get combined filter response for visualization"""
        if not self.enabled:
            return np.ones(len(self.frequencies))
            
        responses = []
        for synth in synths:
            if synth.enabled:
                response = synth.get_filter_response(self.frequencies)
                responses.append(response)
        
        if responses:
            return np.mean(responses, axis=0)
        else:
            return np.ones(len(self.frequencies))

# -----------------------------
# Global Variables (updated for 4 oscillators)
# -----------------------------
synth1 = Osc1Synth(osc_id="OSC1")
synth2 = Osc1Synth(osc_id="OSC2")
synth3 = Osc1Synth(osc_id="OSC3")
synth4 = Osc1Synth(osc_id="OSC4")

synth2.freq = 440.0
synth3.freq = 330.0
synth4.freq = 660.0

synths = [synth1, synth2, synth3, synth4]
sync_oscs = False
mod_matrix = ModulationMatrix()
oscilloscope = Oscilloscope()
filter_spectrogram = FilterSpectrogram()
stream = None

# GUI Variables (will be initialized later)
root = None
fig = None
ax = None
canvas_mpl = None
fig_spec = None
ax_spec = None
canvas_spec = None

def audio_callback(outdata, frames, time, status):
    if status:
        print(status)
    
    mod_matrix.apply_modulation(synths, frames)
    
    # Generate waves for all oscillators
    waves = []
    for synth in synths:
        wave = synth.generate_wave(frames)
        filtered = synth.apply_filter(wave)
        processed = synth.apply_psy_effects(filtered)
        waves.append(processed)
    
    # Mix oscillators
    if sync_oscs:
        # Mix all enabled oscillators
        active_waves = [wave for i, wave in enumerate(waves) if synths[i].enabled]
        if active_waves:
            output = np.sum(active_waves, axis=0) / len(active_waves)
        else:
            output = np.zeros(frames)
    else:
        # Only use first enabled oscillator
        output = np.zeros(frames)
        for i, wave in enumerate(waves):
            if synths[i].enabled:
                output = wave
                break
    
    oscilloscope.update_signal(output)
    outdata[:] = output.reshape(-1, 1)

def start_audio():
    global stream
    if stream is None:
        stream = sd.OutputStream(
            channels=1, callback=audio_callback,
            samplerate=synth1.sample_rate, blocksize=512
        )
        stream.start()

def stop_audio():
    global stream
    if stream is not None:
        stream.stop()
        stream.close()
        stream = None

# -----------------------------
# Button Color Update Function
# -----------------------------
def update_button_color(button, is_on):
    if is_on:
        button.configure(style="Green.TButton")
    else:
        button.configure(style="TButton")

# -----------------------------
# Oscilloscope Functions
# -----------------------------
def toggle_oscilloscope():
    if not MATPLOTLIB_AVAILABLE:
        print("Oscilloscope not available - matplotlib not installed")
        return
        
    oscilloscope.enabled = not oscilloscope.enabled
    update_button_color(osc_button, oscilloscope.enabled)
    if oscilloscope.enabled:
        start_oscilloscope_thread()

def start_oscilloscope_thread():
    if not MATPLOTLIB_AVAILABLE:
        return
        
    def update_scope():
        while oscilloscope.enabled:
            if oscilloscope.enabled and ax is not None:
                try:
                    ax.clear()
                    ax.plot(oscilloscope.time_axis * 1000, oscilloscope.signal_buffer, color='#00FF44', linewidth=1)
                    ax.set_ylim(-1.1, 1.1)
                    ax.set_xlim(0, oscilloscope.buffer_size/44100 * 1000)
                    ax.set_xlabel('Time (ms)', color='#00FF44', fontsize=8)
                    ax.set_ylabel('Amplitude', color='#00FF44', fontsize=8)
                    ax.set_title('Real-Time Signal Oscilloscope', color='#00FF44', fontsize=9)
                    ax.grid(True, alpha=0.3, color='#00FF44')
                    ax.set_facecolor('#000000')
                    ax.tick_params(colors='#00FF44', labelsize=7)
                    if canvas_mpl is not None:
                        canvas_mpl.draw()
                except Exception as e:
                    pass
            time.sleep(0.05)
    
    if oscilloscope.enabled:
        thread = threading.Thread(target=update_scope, daemon=True)
        thread.start()

# -----------------------------
# Spectrogram Functions
# -----------------------------
def toggle_spectrogram():
    if not MATPLOTLIB_AVAILABLE:
        print("Spectrogram not available - matplotlib not installed")
        return
        
    filter_spectrogram.enabled = not filter_spectrogram.enabled
    update_button_color(spec_button, filter_spectrogram.enabled)
    if filter_spectrogram.enabled:
        start_spectrogram_thread()

def start_spectrogram_thread():
    if not MATPLOTLIB_AVAILABLE:
        return
        
    def update_spectrogram():
        while filter_spectrogram.enabled:
            if filter_spectrogram.enabled and ax_spec is not None:
                try:
                    response = filter_spectrogram.get_combined_response(synths)
                    response_db = 20 * np.log10(np.maximum(response, 1e-6))
                    
                    ax_spec.clear()
                    ax_spec.semilogx(filter_spectrogram.frequencies, response_db, color='#00FF44', linewidth=2)
                    
                    # Mark cutoff frequencies for all oscillators
                    colors = ['#FF4400', '#4400FF', '#FFAA00', '#AA00FF']
                    for i, synth in enumerate(synths):
                        if synth.enabled:
                            ax_spec.axvline(synth.cutoff, color=colors[i], linestyle='--', alpha=0.7, 
                                          label=f'OSC{i+1} Cutoff ({synth.cutoff:.0f}Hz)')
                    
                    ax_spec.set_xlim(10, 22000)
                    ax_spec.set_ylim(-60, 6)
                    ax_spec.set_xlabel('Frequency (Hz)', color='#00FF44', fontsize=8)
                    ax_spec.set_ylabel('Magnitude (dB)', color='#00FF44', fontsize=8)
                    ax_spec.set_title('Filter Frequency Response', color='#00FF44', fontsize=9)
                    ax_spec.grid(True, alpha=0.3, color='#00FF44')
                    ax_spec.set_facecolor('#000000')
                    ax_spec.tick_params(colors='#00FF44', labelsize=7)
                    
                    if any(synth.enabled for synth in synths):
                        ax_spec.legend(fontsize=7, facecolor='#000000', edgecolor='#00FF44', labelcolor='#00FF44')
                    
                    if canvas_spec is not None:
                        canvas_spec.draw()
                except Exception as e:
                    pass
            time.sleep(0.1)
    
    if filter_spectrogram.enabled:
        thread = threading.Thread(target=update_spectrogram, daemon=True)
        thread.start()

# -----------------------------
# Master Controls
# -----------------------------
def set_master_pitch(val):
    pitch_shift = int(float(val))
    for synth in synths:
        synth.set_master_pitch(pitch_shift)

def toggle_osc(osc_num):
    synths[osc_num].set_enabled(not synths[osc_num].enabled)
    update_button_color(osc_buttons[osc_num], synths[osc_num].enabled)

def toggle_sync():
    global sync_oscs
    sync_oscs = not sync_oscs
    update_button_color(sync_button, sync_oscs)

def trigger_all_envelopes():
    for synth in synths:
        if synth.enabled:
            synth.trigger_envelope()

def release_all_envelopes():
    for synth in synths:
        synth.release_envelope()

# -----------------------------
# Modulation Functions
# -----------------------------
def update_modulation(*args):
    source = mod_source_var.get()
    target = mod_target_var.get()
    amount = mod_amount_var.get() / 100.0
    
    if source != "None" and target != "None":
        mod_matrix.set_connection(source, target, amount)

def reset_modulation():
    mod_matrix.connections.clear()
    mod_amount_var.set(0)

def set_bpm(val):
    bpm = int(float(val))
    for synth in synths:
        synth.bpm = bpm
    bpm_label.config(text=f"BPM: {bpm}")

# -----------------------------
# Generic Oscillator Controls
# -----------------------------
def create_osc_callbacks(osc_num):
    """Create callback functions for a specific oscillator"""
    
    def update_wave(event=None):
        synths[osc_num].set_wave(wave_vars[osc_num].get())
    
    def update_freq(val):
        synths[osc_num].set_frequency(float(val))
    
    def update_cutoff(val):
        synths[osc_num].set_filter(filter_vars[osc_num].get(), float(val), synths[osc_num].resonance)
    
    def update_resonance(val):
        synths[osc_num].set_filter(filter_vars[osc_num].get(), synths[osc_num].cutoff, float(val))
    
    def update_filter(event=None):
        synths[osc_num].set_filter(filter_vars[osc_num].get(), synths[osc_num].cutoff, synths[osc_num].resonance)
    
    def toggle_delay():
        synths[osc_num].set_delay(not synths[osc_num].delay_on, synths[osc_num].delay_amount, synths[osc_num].delay_time_sec)
        update_button_color(delay_buttons[osc_num], synths[osc_num].delay_on)
    
    def set_delay_amount(val):
        synths[osc_num].set_delay(synths[osc_num].delay_on, float(val)/100.0, synths[osc_num].delay_time_sec)
    
    def set_delay_time(val):
        synths[osc_num].set_delay(synths[osc_num].delay_on, synths[osc_num].delay_amount, float(val)/1000.0)
    
    def set_octave(val):
        synths[osc_num].set_octave(int(float(val)))
    
    def toggle_lfo():
        synths[osc_num].set_lfo(on=not synths[osc_num].lfo_on)
        update_button_color(lfo_buttons[osc_num], synths[osc_num].lfo_on)
    
    def set_lfo_rate(val):
        synths[osc_num].set_lfo(rate=float(val))
    
    def set_lfo_depth(val):
        synths[osc_num].set_lfo(depth=float(val)/100.0)
    
    def set_lfo_mode(event=None):
        synths[osc_num].set_lfo(mode=lfo_mode_vars[osc_num].get())
    
    def set_lfo_sync(event=None):
        synths[osc_num].set_lfo(sync_div=lfo_sync_vars[osc_num].get())
    
    # ADSR callbacks
    def set_attack(val):
        synths[osc_num].envelope.set_adsr(float(val)/1000, synths[osc_num].envelope.decay, 
                                         synths[osc_num].envelope.sustain, synths[osc_num].envelope.release)
    
    def set_decay(val):
        synths[osc_num].envelope.set_adsr(synths[osc_num].envelope.attack, float(val)/1000,
                                         synths[osc_num].envelope.sustain, synths[osc_num].envelope.release)
    
    def set_sustain(val):
        synths[osc_num].envelope.set_adsr(synths[osc_num].envelope.attack, synths[osc_num].envelope.decay,
                                         float(val)/100, synths[osc_num].envelope.release)
    
    def set_release(val):
        synths[osc_num].envelope.set_adsr(synths[osc_num].envelope.attack, synths[osc_num].envelope.decay,
                                         synths[osc_num].envelope.sustain, float(val)/1000)
    
    def trigger_envelope():
        synths[osc_num].trigger_envelope()
    
    def release_envelope():
        synths[osc_num].release_envelope()
    
    # Psy effects callbacks
    def set_distortion_amount(val):
        synths[osc_num].set_psy_distortion(float(val))
    
    def set_distortion_type(event=None):
        synths[osc_num].set_psy_distortion(synths[osc_num].psy_distortion_amount, distortion_type_vars[osc_num].get())
    
    def set_chorus_mix(val):
        synths[osc_num].set_psy_chorus(float(val)/100)
    
    def set_chorus_rate(val):
        synths[osc_num].set_psy_chorus(synths[osc_num].psy_chorus_mix, float(val))
    
    def set_chorus_depth(val):
        synths[osc_num].set_psy_chorus(synths[osc_num].psy_chorus_mix, synths[osc_num].psy_chorus_rate, float(val)/100)
    
    def set_gate_depth(val):
        synths[osc_num].set_psy_gate(float(val)/100)
    
    def set_gate_rate(val):
        synths[osc_num].set_psy_gate(synths[osc_num].psy_gate_depth, int(float(val)))
    
    def set_noise_amount(val):
        synths[osc_num].set_psy_noise(float(val)/100)
    
    def set_fm_amount(val):
        synths[osc_num].set_fm(float(val)/100)
    
    def set_fm_ratio(val):
        synths[osc_num].set_fm(synths[osc_num].fm_amount, float(val))
    
    # Arpeggiator callbacks
    def toggle_arp():
        synths[osc_num].arp_on = not synths[osc_num].arp_on
        if osc_num < len(arp_buttons) and arp_buttons[osc_num] is not None:
            update_button_color(arp_buttons[osc_num], synths[osc_num].arp_on)
        if synths[osc_num].arp_on:
            synths[osc_num].arp_idx = 0
            synths[osc_num].arp_phase_samples = 0.0
    
    def set_arp_mode(event=None):
        if osc_num < len(arp_mode_vars) and arp_mode_vars[osc_num] is not None:
            synths[osc_num].arp_mode = arp_mode_vars[osc_num].get()
    
    def set_arp_rate(val):
        try:
            synths[osc_num].arp_rate = float(val)
        except:
            pass
    
    def set_arp_sync(event=None):
        if osc_num < len(arp_sync_vars) and arp_sync_vars[osc_num] is not None:
            synths[osc_num].arp_sync_div = arp_sync_vars[osc_num].get()
    
    def set_arp_note(idx, val):
        try:
            synths[osc_num].arp_notes[int(idx)] = int(float(val))
        except:
            pass
    
    def set_arp_octave(idx, val):
        try:
            synths[osc_num].arp_octaves[int(idx)] = int(float(val))
        except:
            pass
    
    return {
        'update_wave': update_wave,
        'update_freq': update_freq,
        'update_cutoff': update_cutoff,
        'update_resonance': update_resonance,
        'update_filter': update_filter,
        'toggle_delay': toggle_delay,
        'set_delay_amount': set_delay_amount,
        'set_delay_time': set_delay_time,
        'set_octave': set_octave,
        'toggle_lfo': toggle_lfo,
        'set_lfo_rate': set_lfo_rate,
        'set_lfo_depth': set_lfo_depth,
        'set_lfo_mode': set_lfo_mode,
        'set_lfo_sync': set_lfo_sync,
        'set_attack': set_attack,
        'set_decay': set_decay,
        'set_sustain': set_sustain,
        'set_release': set_release,
        'trigger_envelope': trigger_envelope,
        'release_envelope': release_envelope,
        'set_distortion_amount': set_distortion_amount,
        'set_distortion_type': set_distortion_type,
        'set_chorus_mix': set_chorus_mix,
        'set_chorus_rate': set_chorus_rate,
        'set_chorus_depth': set_chorus_depth,
        'set_gate_depth': set_gate_depth,
        'set_gate_rate': set_gate_rate,
        'set_noise_amount': set_noise_amount,
        'set_fm_amount': set_fm_amount,
        'set_fm_ratio': set_fm_ratio,
        'toggle_arp': toggle_arp,
        'set_arp_mode': set_arp_mode,
        'set_arp_rate': set_arp_rate,
        'set_arp_sync': set_arp_sync,
        'set_arp_note': set_arp_note,
        'set_arp_octave': set_arp_octave
    }

# Create callbacks for all oscillators
osc_callbacks = [create_osc_callbacks(i) for i in range(4)]

# Initialize GUI variable arrays
wave_vars = []
filter_vars = []
lfo_mode_vars = []
lfo_sync_vars = []
distortion_type_vars = []
arp_mode_vars = []
arp_sync_vars = []
delay_buttons = []
lfo_buttons = []
arp_buttons = []

# -----------------------------
# Build GUI (Enhanced for 4 oscillators)
# -----------------------------
def build_gui():
    global root, fig, ax, canvas_mpl, fig_spec, ax_spec, canvas_spec
    global osc_buttons, sync_button, osc_button, spec_button
    global bpm_label
    global mod_source_var, mod_target_var, mod_amount_var
    
    root = tk.Tk()
    root.title("Enhanced 4-Oscillator Psy Trance Synth")
    root.configure(bg="#000000")
    root.geometry("1600x1000")

    style = ttk.Style()
    try:
        style.theme_use("clam")
    except Exception:
        pass

    green = "#00FF44"
    dark = "#001100"

    style.configure("TLabel", background="#000000", foreground=green, font=("Consolas", 8))
    style.configure("TButton", background="#000000", foreground=green, font=("Consolas", 8))
    style.configure("TFrame", background="#000000")
    style.configure("TCombobox", fieldbackground="#000000", background="#000000", foreground=green)
    style.map("TButton", background=[("active", dark)], foreground=[("active", green)])
    style.configure("Green.TButton", background="#00AA22", foreground="#FFFFFF", font=("Consolas", 8))
    style.map("Green.TButton", background=[("active", "#00DD33")], foreground=[("active", "#FFFFFF")])

    main_container = ttk.Frame(root, padding=4, style="TFrame")
    main_container.pack(fill="both", expand=True)

    # Master controls
    master_frame = ttk.Frame(main_container, padding=3, style="TFrame")
    master_frame.pack(fill="x", pady=(0,6))

    # Oscillator enable buttons
    osc_buttons = []
    for i in range(4):
        btn = ttk.Button(master_frame, text=f"OSC{i+1}", command=lambda i=i: toggle_osc(i))
        btn.pack(side="left", padx=(0,4))
        osc_buttons.append(btn)

    sync_button = ttk.Button(master_frame, text="Sync OSCs", command=toggle_sync)
    sync_button.pack(side="left", padx=(0,10))

    ttk.Label(master_frame, text="Master Pitch:").pack(side="left", padx=(0,3))
    master_pitch_slider = tk.Scale(master_frame, from_=-8, to=8, orient="horizontal", 
                                  command=set_master_pitch, length=80,
                                  bg="#000000", fg=green, troughcolor="#003300", highlightbackground="#000000")
    master_pitch_slider.set(0)
    master_pitch_slider.pack(side="left", padx=(0,10))

    ttk.Label(master_frame, text="BPM:").pack(side="left", padx=(0,3))
    bpm_slider = tk.Scale(master_frame, from_=60, to=200, orient="horizontal",
                         command=set_bpm, length=80,
                         bg="#000000", fg=green, troughcolor="#003300", highlightbackground="#000000")
    bpm_slider.set(120)
    bpm_slider.pack(side="left", padx=(0,10))
    bpm_label = ttk.Label(master_frame, text="BPM: 120")
    bpm_label.pack(side="left", padx=(0,10))

    # Envelope triggers
    trigger_btn = ttk.Button(master_frame, text="Trigger All", command=trigger_all_envelopes)
    trigger_btn.pack(side="left", padx=3)
    release_btn = ttk.Button(master_frame, text="Release All", command=release_all_envelopes)
    release_btn.pack(side="left", padx=3)

    start_btn = ttk.Button(master_frame, text="Start", command=start_audio)
    start_btn.pack(side="right", padx=2)
    stop_btn = ttk.Button(master_frame, text="Stop", command=stop_audio)
    stop_btn.pack(side="right", padx=2)

    if MATPLOTLIB_AVAILABLE:
        osc_button = ttk.Button(master_frame, text="Oscilloscope", command=toggle_oscilloscope)
        osc_button.pack(side="right", padx=4)
        spec_button = ttk.Button(master_frame, text="Filter Spectrum", command=toggle_spectrogram)
        spec_button.pack(side="right", padx=4)

    # Modulation Matrix Section
    mod_frame = ttk.Frame(main_container, padding=3, style="TFrame")
    mod_frame.pack(fill="x", pady=(0,6))

    title_frame = ttk.Frame(mod_frame, style="TFrame")
    title_frame.pack(fill="x", pady=(0,3))
    ttk.Label(title_frame, text="NEGATIVE MATRIX", font=("Consolas", 11, "bold")).pack(side="left")
    ttk.Label(title_frame, text="by Gabriel Trentini", font=("Consolas", 6)).pack(side="right")

    mod_controls_frame = ttk.Frame(mod_frame, style="TFrame")
    mod_controls_frame.pack(fill="x")

    ttk.Label(mod_controls_frame, text="Source:").pack(side="left", padx=(0,3))
    mod_source_var = tk.StringVar(value="None")
    mod_source_menu = ttk.Combobox(mod_controls_frame, textvariable=mod_source_var,
        values=["None", "LFO1", "LFO2", "LFO3", "LFO4"], state="readonly", width=6)
    mod_source_menu.pack(side="left", padx=(0,6))

    ttk.Label(mod_controls_frame, text="Target:").pack(side="left", padx=(0,3))
    mod_target_var = tk.StringVar(value="None")
    targets = ["None"]
    for i in range(1, 5):
        targets.extend([f"OSC{i}_FREQ", f"OSC{i}_CUTOFF", f"OSC{i}_RESONANCE", 
                       f"OSC{i}_DELAY_TIME", f"OSC{i}_DELAY_AMOUNT"])
    mod_target_menu = ttk.Combobox(mod_controls_frame, textvariable=mod_target_var,
        values=targets, state="readonly", width=12)
    mod_target_menu.pack(side="left", padx=(0,6))

    ttk.Label(mod_controls_frame, text="Amount:").pack(side="left", padx=(0,3))
    mod_amount_var = tk.IntVar(value=0)
    mod_amount_slider = tk.Scale(mod_controls_frame, from_=-100, to=100, orient="horizontal", 
                                variable=mod_amount_var, length=100,
                                bg="#000000", fg=green, troughcolor="#003300", highlightbackground="#000000")
    mod_amount_slider.pack(side="left", padx=(0,6))

    reset_btn = ttk.Button(mod_controls_frame, text="Reset All", command=reset_modulation)
    reset_btn.pack(side="right")

    mod_source_var.trace('w', update_modulation)
    mod_target_var.trace('w', update_modulation)
    mod_amount_var.trace('w', update_modulation)

    # Visualization Frame
    if MATPLOTLIB_AVAILABLE:
        viz_frame = ttk.Frame(main_container, padding=3, style="TFrame")
        viz_frame.pack(fill="x", pady=(0,6))

        # Oscilloscope
        fig, ax = plt.subplots(figsize=(8, 2), facecolor='#000000')
        fig.patch.set_facecolor('#000000')
        ax.set_facecolor('#000000')
        ax.set_xlim(0, oscilloscope.buffer_size/44100 * 1000)
        ax.set_ylim(-1.1, 1.1)
        ax.set_xlabel('Time (ms)', color='#00FF44', fontsize=7)
        ax.set_ylabel('Amplitude', color='#00FF44', fontsize=7)
        ax.set_title('Real-Time Signal Oscilloscope', color='#00FF44', fontsize=8)
        ax.grid(True, alpha=0.3, color='#00FF44')
        ax.tick_params(colors='#00FF44', labelsize=6)

        canvas_mpl = FigureCanvasTkAgg(fig, viz_frame)
        canvas_mpl.get_tk_widget().pack(side="left", fill="both", expand=True, padx=(0,4))

        # Filter Spectrogram
        fig_spec, ax_spec = plt.subplots(figsize=(8, 2), facecolor='#000000')
        fig_spec.patch.set_facecolor('#000000')
        ax_spec.set_facecolor('#000000')
        ax_spec.set_xlim(10, 22000)
        ax_spec.set_ylim(-60, 6)
        ax_spec.set_xlabel('Frequency (Hz)', color='#00FF44', fontsize=7)
        ax_spec.set_ylabel('Magnitude (dB)', color='#00FF44', fontsize=7)
        ax_spec.set_title('Filter Frequency Response', color='#00FF44', fontsize=8)
        ax_spec.grid(True, alpha=0.3, color='#00FF44')
        ax_spec.tick_params(colors='#00FF44', labelsize=6)

        canvas_spec = FigureCanvasTkAgg(fig_spec, viz_frame)
        canvas_spec.get_tk_widget().pack(side="right", fill="both", expand=True, padx=(4,0))

    # Create notebook for oscillator tabs
    notebook = ttk.Notebook(main_container)
    notebook.pack(fill="both", expand=True, pady=(0,4))

    # Create oscillator tabs
    for osc_num in range(4):
        tab_frame = ttk.Frame(notebook, style="TFrame")
        notebook.add(tab_frame, text=f"OSC {osc_num+1}")
        
        # Create scrollable frame for each oscillator
        canvas_scroll = tk.Canvas(tab_frame, bg="#000000", highlightthickness=0)
        scrollbar = ttk.Scrollbar(tab_frame, orient="vertical", command=canvas_scroll.yview)
        scrollable_frame = ttk.Frame(canvas_scroll, style="TFrame")

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas_scroll.configure(scrollregion=canvas_scroll.bbox("all"))
        )

        canvas_scroll.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas_scroll.configure(yscrollcommand=scrollbar.set)

        # Build oscillator controls
        build_oscillator_controls(scrollable_frame, osc_num)

        canvas_scroll.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Mouse wheel scrolling
        def _on_mousewheel(event, canvas=canvas_scroll):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas_scroll.bind_all("<MouseWheel>", _on_mousewheel)

    # Initialize button colors
    def initialize_button_colors():
        for i, synth in enumerate(synths):
            update_button_color(osc_buttons[i], synth.enabled)
        update_button_color(sync_button, sync_oscs)
        if MATPLOTLIB_AVAILABLE:
            update_button_color(osc_button, oscilloscope.enabled)
            update_button_color(spec_button, filter_spectrogram.enabled)
        for i, btn in enumerate(delay_buttons):
            if btn is not None:
                update_button_color(btn, synths[i].delay_on)
        for i, btn in enumerate(lfo_buttons):
            if btn is not None:
                update_button_color(btn, synths[i].lfo_on)
        for i, btn in enumerate(arp_buttons):
            if btn is not None:
                update_button_color(btn, synths[i].arp_on)

    initialize_button_colors()

    root.resizable(True, True)
    return root

def build_oscillator_controls(parent, osc_num):
    """Build controls for a specific oscillator"""
    global wave_vars, filter_vars, lfo_mode_vars, lfo_sync_vars
    global distortion_type_vars, arp_mode_vars, arp_sync_vars
    global delay_buttons, lfo_buttons, arp_buttons
    
    green = "#00FF44"
    callbacks = osc_callbacks[osc_num]
    
    # Main container with padding
    main_frame = ttk.Frame(parent, padding=8, style="TFrame")
    main_frame.pack(fill="both", expand=True)

    # Title
    title = ttk.Label(main_frame, text=f"OSCILLATOR {osc_num+1}", font=("Consolas", 12, "bold"))
    title.pack(pady=(0,12))

    # Create two columns
    columns_frame = ttk.Frame(main_frame, style="TFrame")
    columns_frame.pack(fill="both", expand=True)
    
    left_column = ttk.Frame(columns_frame, style="TFrame")
    left_column.pack(side="left", fill="both", expand=True, padx=(0,8))
    
    right_column = ttk.Frame(columns_frame, style="TFrame")
    right_column.pack(side="right", fill="both", expand=True, padx=(8,0))

    # LEFT COLUMN - Basic Synthesis
    
    # Waveform
    ttk.Label(left_column, text="A. Waveform", font=("Consolas", 9, "bold")).pack(anchor="w", pady=(2,0))
    wave_var = tk.StringVar(value="Sine")
    wave_vars.append(wave_var)
    wave_menu = ttk.Combobox(left_column, textvariable=wave_var,
        values=["Sine", "Saw", "Square", "Triangle", "Sample&Hold"], state="readonly", width=15)
    wave_menu.bind("<<ComboboxSelected>>", callbacks['update_wave'])
    wave_menu.pack(fill="x", pady=(0,6))

    # Frequency
    ttk.Label(left_column, text="B. Frequency", font=("Consolas", 9, "bold")).pack(anchor="w")
    freq_slider = tk.Scale(left_column, from_=50, to=5000, orient="horizontal", command=callbacks['update_freq'],
                           bg="#000000", fg=green, troughcolor="#003300", highlightbackground="#000000")
    freq_slider.set(synths[osc_num].freq)
    freq_slider.pack(fill="x", pady=(0,6))

    # Filter Type
    ttk.Label(left_column, text="C. Filter Type", font=("Consolas", 9, "bold")).pack(anchor="w")
    filter_var = tk.StringVar(value="Lowpass")
    filter_vars.append(filter_var)
    filter_menu = ttk.Combobox(left_column, textvariable=filter_var,
        values=["Lowpass", "Highpass", "Bandpass", "Notch", "Comb"], state="readonly", width=15)
    filter_menu.bind("<<ComboboxSelected>>", callbacks['update_filter'])
    filter_menu.pack(fill="x", pady=(0,6))

    # Cutoff Frequency
    ttk.Label(left_column, text="D. Cutoff Frequency", font=("Consolas", 9, "bold")).pack(anchor="w")
    cutoff_slider = tk.Scale(left_column, from_=100, to=8000, orient="horizontal", command=callbacks['update_cutoff'],
                             bg="#000000", fg=green, troughcolor="#003300", highlightbackground="#000000")
    cutoff_slider.set(1000)
    cutoff_slider.pack(fill="x", pady=(0,6))

    # Resonance
    ttk.Label(left_column, text="E. Resonance", font=("Consolas", 9, "bold")).pack(anchor="w")
    res_slider = tk.Scale(left_column, from_=0.1, to=2.0, resolution=0.1, orient="horizontal",
                          command=callbacks['update_resonance'], bg="#000000", fg=green, troughcolor="#003300", highlightbackground="#000000")
    res_slider.set(0.7)
    res_slider.pack(fill="x", pady=(0,8))

    # ADSR Envelope Section
    ttk.Label(left_column, text="F. ADSR ENVELOPE", font=("Consolas", 10, "bold")).pack(anchor="w", pady=(4,4))
    
    # Envelope triggers
    env_buttons_frame = ttk.Frame(left_column, style="TFrame")
    env_buttons_frame.pack(fill="x", pady=(0,4))
    
    trigger_btn = ttk.Button(env_buttons_frame, text="Trigger", command=callbacks['trigger_envelope'])
    trigger_btn.pack(side="left", padx=(0,4))
    release_btn = ttk.Button(env_buttons_frame, text="Release", command=callbacks['release_envelope'])
    release_btn.pack(side="left")

    # Attack
    ttk.Label(left_column, text="F1. Attack (ms)").pack(anchor="w")
    attack_slider = tk.Scale(left_column, from_=1, to=5000, orient="horizontal", command=callbacks['set_attack'],
                            bg="#000000", fg=green, troughcolor="#003300", highlightbackground="#000000")
    attack_slider.set(10)
    attack_slider.pack(fill="x", pady=(0,3))

    # Decay
    ttk.Label(left_column, text="F2. Decay (ms)").pack(anchor="w")
    decay_slider = tk.Scale(left_column, from_=1, to=5000, orient="horizontal", command=callbacks['set_decay'],
                           bg="#000000", fg=green, troughcolor="#003300", highlightbackground="#000000")
    decay_slider.set(100)
    decay_slider.pack(fill="x", pady=(0,3))

    # Sustain
    ttk.Label(left_column, text="F3. Sustain (%)").pack(anchor="w")
    sustain_slider = tk.Scale(left_column, from_=0, to=100, orient="horizontal", command=callbacks['set_sustain'],
                             bg="#000000", fg=green, troughcolor="#003300", highlightbackground="#000000")
    sustain_slider.set(70)
    sustain_slider.pack(fill="x", pady=(0,3))

    # Release
    ttk.Label(left_column, text="F4. Release (ms)").pack(anchor="w")
    release_slider = tk.Scale(left_column, from_=1, to=5000, orient="horizontal", command=callbacks['set_release'],
                             bg="#000000", fg=green, troughcolor="#003300", highlightbackground="#000000")
    release_slider.set(300)
    release_slider.pack(fill="x", pady=(0,8))

    # RIGHT COLUMN - Advanced Features

    # Delay Section
    ttk.Label(right_column, text="G. DELAY", font=("Consolas", 10, "bold")).pack(anchor="w", pady=(2,4))
    
    delay_button = ttk.Button(right_column, text="Delay On/Off", command=callbacks['toggle_delay'])
    delay_button.pack(anchor="w", pady=(0,3))
    delay_buttons.append(delay_button)

    ttk.Label(right_column, text="G1. Delay Amount (%)").pack(anchor="w")
    delay_slider = tk.Scale(right_column, from_=0, to=100, orient="horizontal", command=callbacks['set_delay_amount'],
                            bg="#000000", fg=green, troughcolor="#003300", highlightbackground="#000000")
    delay_slider.set(50)
    delay_slider.pack(fill="x", pady=(0,3))

    ttk.Label(right_column, text="G2. Delay Time (ms)").pack(anchor="w")
    delay_time_slider = tk.Scale(right_column, from_=10, to=1000, orient="horizontal", command=callbacks['set_delay_time'],
                                 bg="#000000", fg=green, troughcolor="#003300", highlightbackground="#000000")
    delay_time_slider.set(300)
    delay_time_slider.pack(fill="x", pady=(0,8))

    # LFO Section
    ttk.Label(right_column, text="H. LFO", font=("Consolas", 10, "bold")).pack(anchor="w", pady=(4,4))

    lfo_button = ttk.Button(right_column, text="LFO On/Off", command=callbacks['toggle_lfo'])
    lfo_button.pack(anchor="w", pady=(0,3))
    lfo_buttons.append(lfo_button)

    ttk.Label(right_column, text="H1. LFO Mode").pack(anchor="w")
    lfo_mode_var = tk.StringVar(value="Hz")
    lfo_mode_vars.append(lfo_mode_var)
    lfo_mode_menu = ttk.Combobox(right_column, textvariable=lfo_mode_var,
        values=["Hz", "Sync"], state="readonly", width=15)
    lfo_mode_menu.bind("<<ComboboxSelected>>", callbacks['set_lfo_mode'])
    lfo_mode_menu.pack(fill="x", pady=(0,3))

    ttk.Label(right_column, text="H2. LFO Rate").pack(anchor="w")
    lfo_rate_slider = tk.Scale(right_column, from_=0.1, to=20.0, resolution=0.1, orient="horizontal", command=callbacks['set_lfo_rate'],
                               bg="#000000", fg=green, troughcolor="#003300", highlightbackground="#000000")
    lfo_rate_slider.set(5.0)
    lfo_rate_slider.pack(fill="x", pady=(0,3))

    ttk.Label(right_column, text="H3. LFO Sync Division").pack(anchor="w")
    lfo_sync_var = tk.StringVar(value="1/4")
    lfo_sync_vars.append(lfo_sync_var)
    lfo_sync_menu = ttk.Combobox(right_column, textvariable=lfo_sync_var,
        values=["1/1", "1/2", "1/4", "1/8", "1/16"], state="readonly", width=15)
    lfo_sync_menu.bind("<<ComboboxSelected>>", callbacks['set_lfo_sync'])
    lfo_sync_menu.pack(fill="x", pady=(0,3))

    ttk.Label(right_column, text="H4. LFO Depth (%)").pack(anchor="w")
    lfo_depth_slider = tk.Scale(right_column, from_=0, to=100, orient="horizontal", command=callbacks['set_lfo_depth'],
                                bg="#000000", fg=green, troughcolor="#003300", highlightbackground="#000000")
    lfo_depth_slider.set(0)
    lfo_depth_slider.pack(fill="x", pady=(0,8))

    # Octave Shift
    ttk.Label(right_column, text="I. Octave Shift", font=("Consolas", 9, "bold")).pack(anchor="w")
    octave_slider = tk.Scale(right_column, from_=-4, to=4, orient="horizontal", command=callbacks['set_octave'],
                             bg="#000000", fg=green, troughcolor="#003300", highlightbackground="#000000")
    octave_slider.set(0)
    octave_slider.pack(fill="x", pady=(0,8))

    # Create second row for psy effects
    psy_frame = ttk.Frame(main_frame, style="TFrame")
    psy_frame.pack(fill="both", expand=True, pady=(12,0))

    psy_left = ttk.Frame(psy_frame, style="TFrame")
    psy_left.pack(side="left", fill="both", expand=True, padx=(0,8))
    
    psy_right = ttk.Frame(psy_frame, style="TFrame")
    psy_right.pack(side="right", fill="both", expand=True, padx=(8,0))

    # PSY EFFECTS - Left Column
    ttk.Label(psy_left, text="J. PSY DISTORTION", font=("Consolas", 10, "bold")).pack(anchor="w", pady=(4,4))

    ttk.Label(psy_left, text="J1. Distortion Type").pack(anchor="w")
    distortion_type_var = tk.StringVar(value="soft")
    distortion_type_vars.append(distortion_type_var)
    dist_type_menu = ttk.Combobox(psy_left, textvariable=distortion_type_var,
        values=["soft", "hard", "tube"], state="readonly", width=15)
    dist_type_menu.bind("<<ComboboxSelected>>", callbacks['set_distortion_type'])
    dist_type_menu.pack(fill="x", pady=(0,3))

    ttk.Label(psy_left, text="J2. Distortion Drive").pack(anchor="w")
    distortion_slider = tk.Scale(psy_left, from_=1.0, to=10.0, resolution=0.1, orient="horizontal", command=callbacks['set_distortion_amount'],
                                bg="#000000", fg=green, troughcolor="#003300", highlightbackground="#000000")
    distortion_slider.set(1.0)
    distortion_slider.pack(fill="x", pady=(0,8))

    # Chorus
    ttk.Label(psy_left, text="K. PSY CHORUS", font=("Consolas", 10, "bold")).pack(anchor="w", pady=(4,4))

    ttk.Label(psy_left, text="K1. Chorus Mix (%)").pack(anchor="w")
    chorus_mix_slider = tk.Scale(psy_left, from_=0, to=100, orient="horizontal", command=callbacks['set_chorus_mix'],
                                bg="#000000", fg=green, troughcolor="#003300", highlightbackground="#000000")
    chorus_mix_slider.set(0)
    chorus_mix_slider.pack(fill="x", pady=(0,3))

    ttk.Label(psy_left, text="K2. Chorus Rate").pack(anchor="w")
    chorus_rate_slider = tk.Scale(psy_left, from_=0.1, to=5.0, resolution=0.1, orient="horizontal", command=callbacks['set_chorus_rate'],
                                 bg="#000000", fg=green, troughcolor="#003300", highlightbackground="#000000")
    chorus_rate_slider.set(0.5)
    chorus_rate_slider.pack(fill="x", pady=(0,3))

    ttk.Label(psy_left, text="K3. Chorus Depth (%)").pack(anchor="w")
    chorus_depth_slider = tk.Scale(psy_left, from_=0, to=100, orient="horizontal", command=callbacks['set_chorus_depth'],
                                  bg="#000000", fg=green, troughcolor="#003300", highlightbackground="#000000")
    chorus_depth_slider.set(30)
    chorus_depth_slider.pack(fill="x", pady=(0,8))

    # PSY EFFECTS - Right Column
    
    # Gate
    ttk.Label(psy_right, text="L. PSY GATE", font=("Consolas", 10, "bold")).pack(anchor="w", pady=(4,4))

    ttk.Label(psy_right, text="L1. Gate Depth (%)").pack(anchor="w")
    gate_depth_slider = tk.Scale(psy_right, from_=0, to=100, orient="horizontal", command=callbacks['set_gate_depth'],
                                bg="#000000", fg=green, troughcolor="#003300", highlightbackground="#000000")
    gate_depth_slider.set(0)
    gate_depth_slider.pack(fill="x", pady=(0,3))

    ttk.Label(psy_right, text="L2. Gate Rate").pack(anchor="w")
    gate_rate_slider = tk.Scale(psy_right, from_=1, to=32, orient="horizontal", command=callbacks['set_gate_rate'],
                               bg="#000000", fg=green, troughcolor="#003300", highlightbackground="#000000")
    gate_rate_slider.set(16)
    gate_rate_slider.pack(fill="x", pady=(0,8))

    # FM Synthesis
    ttk.Label(psy_right, text="M. FM SYNTHESIS", font=("Consolas", 10, "bold")).pack(anchor="w", pady=(4,4))

    ttk.Label(psy_right, text="M1. FM Amount (%)").pack(anchor="w")
    fm_amount_slider = tk.Scale(psy_right, from_=0, to=100, orient="horizontal", command=callbacks['set_fm_amount'],
                               bg="#000000", fg=green, troughcolor="#003300", highlightbackground="#000000")
    fm_amount_slider.set(0)
    fm_amount_slider.pack(fill="x", pady=(0,3))

    ttk.Label(psy_right, text="M2. FM Ratio").pack(anchor="w")
    fm_ratio_slider = tk.Scale(psy_right, from_=0.1, to=8.0, resolution=0.1, orient="horizontal", command=callbacks['set_fm_ratio'],
                              bg="#000000", fg=green, troughcolor="#003300", highlightbackground="#000000")
    fm_ratio_slider.set(2.0)
    fm_ratio_slider.pack(fill="x", pady=(0,3))

    # Forest Noise
    ttk.Label(psy_right, text="N. Forest Noise (%)").pack(anchor="w")
    noise_slider = tk.Scale(psy_right, from_=0, to=100, orient="horizontal", command=callbacks['set_noise_amount'],
                           bg="#000000", fg=green, troughcolor="#003300", highlightbackground="#000000")
    noise_slider.set(0)
    noise_slider.pack(fill="x", pady=(0,8))

    # Arpeggiator (for all oscillators)
    arp_frame = ttk.Frame(main_frame, style="TFrame")
    arp_frame.pack(fill="both", expand=True, pady=(12,0))

    ttk.Label(arp_frame, text="O. ARPEGGIATOR", font=("Consolas", 10, "bold")).pack(anchor="w", pady=(4,4))

    arp_controls_frame = ttk.Frame(arp_frame, style="TFrame")
    arp_controls_frame.pack(fill="x", pady=(0,6))

    arp_button = ttk.Button(arp_controls_frame, text="Arp On/Off", command=callbacks['toggle_arp'])
    arp_button.pack(side="left", padx=(0,8))
    arp_buttons.append(arp_button)

    ttk.Label(arp_controls_frame, text="Mode:").pack(side="left", padx=(0,3))
    arp_mode_var = tk.StringVar(value="Hz")
    arp_mode_vars.append(arp_mode_var)
    arp_mode_menu = ttk.Combobox(arp_controls_frame, textvariable=arp_mode_var,
        values=["Hz", "Sync"], state="readonly", width=8)
    arp_mode_menu.bind("<<ComboboxSelected>>", callbacks['set_arp_mode'])
    arp_mode_menu.pack(side="left", padx=(0,8))

    ttk.Label(arp_controls_frame, text="Rate:").pack(side="left", padx=(0,3))
    arp_rate_slider = tk.Scale(arp_controls_frame, from_=0.5, to=20.0, resolution=0.1, orient="horizontal", 
                              command=callbacks['set_arp_rate'], length=100,
                              bg="#000000", fg=green, troughcolor="#003300", highlightbackground="#000000")
    arp_rate_slider.set(4.0)
    arp_rate_slider.pack(side="left", padx=(0,8))

    ttk.Label(arp_controls_frame, text="Sync:").pack(side="left", padx=(0,3))
    arp_sync_var = tk.StringVar(value="1/8")
    arp_sync_vars.append(arp_sync_var)
    arp_sync_menu = ttk.Combobox(arp_controls_frame, textvariable=arp_sync_var,
        values=["1/1", "1/2", "1/4", "1/8", "1/16"], state="readonly", width=8)
    arp_sync_menu.bind("<<ComboboxSelected>>", callbacks['set_arp_sync'])
    arp_sync_menu.pack(side="left")

    # Arp Pattern
    pattern_frame = ttk.Frame(arp_frame, style="TFrame")
    pattern_frame.pack(fill="x", pady=(6,0))

    ttk.Label(pattern_frame, text="Arp Pattern (Semitones & Octaves):", font=("Consolas", 9, "bold")).pack(anchor="w", pady=(0,4))

    for i in range(8):
        step_frame = ttk.Frame(pattern_frame, style="TFrame")
        step_frame.pack(fill="x", pady=1)
        
        ttk.Label(step_frame, text=f"Step {i+1}:", font=("Consolas", 8)).pack(side="left", padx=(0,4))
        
        ttk.Label(step_frame, text="Semi:").pack(side="left", padx=(0,2))
        note_slider = tk.Scale(step_frame, from_=-24, to=24, orient="horizontal", 
                             command=lambda val, idx=i: callbacks['set_arp_note'](idx, val),
                             bg="#000000", fg=green, troughcolor="#003300", highlightbackground="#000000", length=80)
        note_slider.set(0)
        note_slider.pack(side="left", padx=(0,8))
        
        ttk.Label(step_frame, text="Oct:").pack(side="left", padx=(0,2))
        oct_slider = tk.Scale(step_frame, from_=-4, to=4, orient="horizontal", 
                            command=lambda val, idx=i: callbacks['set_arp_octave'](idx, val),
                            bg="#000000", fg=green, troughcolor="#003300", highlightbackground="#000000", length=60)
        oct_slider.set(0)
        oct_slider.pack(side="left")

# -----------------------------
# Main Execution
# -----------------------------
if __name__ == "__main__":
    root = build_gui()
    root.mainloop()