import numpy as np
import sounddevice as sd
import tkinter as tk
from tkinter import ttk
from scipy.signal import butter, lfilter, freqz
import random
import threading
import time

# Try to import matplotlib, but make it optional
try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Oscilloscope and Spectrogram will be disabled.")

# -----------------------------
# Oscillator + Filter Engine
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
        self.enabled = True  # NEW: Oscillator on/off state

        # Delay parameters
        self.delay_on = False
        self.delay_amount = 0.5
        self.delay_time_sec = 0.3
        self.delay_buffer = np.zeros(int(self.sample_rate*2), dtype=np.float32)
        self.delay_idx = 0

        # Sample & Hold
        self.snh_value = 0.8
        self.snh_phase = 0.0

        # LFO parameters
        self.lfo_on = False
        self.lfo_rate = 5.0
        self.lfo_depth = 0.0
        self.lfo_phase = 0.0
        self.lfo_mode = "Hz"
        self.bpm = 120
        self.lfo_sync_div = "1/4"

        # Modulation targets
        self.mod_freq = 0.0
        self.mod_cutoff = 0.0
        self.mod_resonance = 0.0
        self.mod_delay_time = 0.0
        self.mod_delay_amount = 0.0

        # Arpeggiator parameters
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

        if self.wave_type == "Sine":
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

        self.phase += frames
        if self.lfo_on:
            self.lfo_phase += frames

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
        else:
            return signal
        try:
            return lfilter(b, a, signal)
        except Exception:
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
            else:
                return np.ones(len(frequencies))
            
            w, h = freqz(b, a, worN=frequencies, fs=self.sample_rate)
            return np.abs(h)
        except Exception:
            return np.ones(len(frequencies))

# -----------------------------
# Modulation Matrix
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
    
    def apply_modulation(self, synth1, synth2, frames):
        synth1.mod_freq = synth1.mod_cutoff = synth1.mod_resonance = 0.0
        synth1.mod_delay_time = synth1.mod_delay_amount = 0.0
        synth2.mod_freq = synth2.mod_cutoff = synth2.mod_resonance = 0.0
        synth2.mod_delay_time = synth2.mod_delay_amount = 0.0
        
        lfo1_value = np.mean(synth1.get_lfo_value(frames)) if synth1.lfo_on else 0.0
        lfo2_value = np.mean(synth2.get_lfo_value(frames)) if synth2.lfo_on else 0.0
        
        for source, targets in self.connections.items():
            if source == "LFO1" and synth1.lfo_on:
                source_value = lfo1_value
            elif source == "LFO2" and synth2.lfo_on:
                source_value = lfo2_value
            else:
                continue
                
            for target, amount in targets.items():
                mod_value = source_value * amount
                
                if target == "OSC1_FREQ":
                    synth1.mod_freq = mod_value * 100
                elif target == "OSC1_CUTOFF":
                    synth1.mod_cutoff = mod_value * 2000
                elif target == "OSC1_RESONANCE":
                    synth1.mod_resonance = mod_value * 0.5
                elif target == "OSC1_DELAY_TIME":
                    synth1.mod_delay_time = mod_value * 0.2
                elif target == "OSC1_DELAY_AMOUNT":
                    synth1.mod_delay_amount = mod_value * 0.3
                elif target == "OSC2_FREQ":
                    synth2.mod_freq = mod_value * 100
                elif target == "OSC2_CUTOFF":
                    synth2.mod_cutoff = mod_value * 2000
                elif target == "OSC2_RESONANCE":
                    synth2.mod_resonance = mod_value * 0.5
                elif target == "OSC2_DELAY_TIME":
                    synth2.mod_delay_time = mod_value * 0.2
                elif target == "OSC2_DELAY_AMOUNT":
                    synth2.mod_delay_amount = mod_value * 0.3

# -----------------------------
# Oscilloscope Class
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
# Spectrogram Class
# -----------------------------
class FilterSpectrogram:
    def __init__(self):
        self.enabled = False
        self.frequencies = np.logspace(1, np.log10(22000), 512)  # 10 Hz to 22 kHz
        
    def get_combined_response(self, synth1, synth2):
        """Get combined filter response for visualization"""
        if not self.enabled:
            return np.ones(len(self.frequencies))
            
        response1 = synth1.get_filter_response(self.frequencies) if synth1.enabled else np.ones(len(self.frequencies))
        response2 = synth2.get_filter_response(self.frequencies) if synth2.enabled else np.ones(len(self.frequencies))
        
        if synth1.enabled and synth2.enabled:
            return (response1 + response2) * 0.5
        elif synth1.enabled:
            return response1
        elif synth2.enabled:
            return response2
        else:
            return np.ones(len(self.frequencies))

# -----------------------------
# Global Variables
# -----------------------------
synth1 = Osc1Synth(osc_id="OSC1")
synth2 = Osc1Synth(osc_id="OSC2")
synth2.freq = 440.0
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
    
    mod_matrix.apply_modulation(synth1, synth2, frames)
    
    wave1 = synth1.generate_wave(frames)
    filtered1 = synth1.apply_filter(wave1)
    
    wave2 = synth2.generate_wave(frames)
    filtered2 = synth2.apply_filter(wave2)
    
    if sync_oscs:
        output = (filtered1 + filtered2) * 0.5
    else:
        output = filtered1
    
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
# Button Color Update Function (Changed from Red to Green)
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
                    response = filter_spectrogram.get_combined_response(synth1, synth2)
                    response_db = 20 * np.log10(np.maximum(response, 1e-6))
                    
                    ax_spec.clear()
                    ax_spec.semilogx(filter_spectrogram.frequencies, response_db, color='#00FF44', linewidth=2)
                    
                    # Mark cutoff frequencies
                    if synth1.enabled:
                        ax_spec.axvline(synth1.cutoff, color='#FF4400', linestyle='--', alpha=0.7, label=f'OSC1 Cutoff ({synth1.cutoff:.0f}Hz)')
                    if synth2.enabled:
                        ax_spec.axvline(synth2.cutoff, color='#4400FF', linestyle='--', alpha=0.7, label=f'OSC2 Cutoff ({synth2.cutoff:.0f}Hz)')
                    
                    ax_spec.set_xlim(10, 22000)
                    ax_spec.set_ylim(-60, 6)
                    ax_spec.set_xlabel('Frequency (Hz)', color='#00FF44', fontsize=8)
                    ax_spec.set_ylabel('Magnitude (dB)', color='#00FF44', fontsize=8)
                    ax_spec.set_title('Filter Frequency Response', color='#00FF44', fontsize=9)
                    ax_spec.grid(True, alpha=0.3, color='#00FF44')
                    ax_spec.set_facecolor('#000000')
                    ax_spec.tick_params(colors='#00FF44', labelsize=7)
                    
                    if synth1.enabled or synth2.enabled:
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
    synth1.set_master_pitch(pitch_shift)
    synth2.set_master_pitch(pitch_shift)

def toggle_osc1():
    synth1.set_enabled(not synth1.enabled)
    update_button_color(osc1_button, synth1.enabled)

def toggle_osc2():
    synth2.set_enabled(not synth2.enabled)
    update_button_color(osc2_button, synth2.enabled)

def toggle_sync():
    global sync_oscs
    sync_oscs = not sync_oscs
    update_button_color(sync_button, sync_oscs)

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

# -----------------------------
# OSC1 Callbacks
# -----------------------------
def update_wave(event=None):
    synth1.set_wave(wave_var.get())

def update_freq(val):
    synth1.set_frequency(float(val))

def update_cutoff(val):
    synth1.set_filter(filter_var.get(), float(val), synth1.resonance)

def update_resonance(val):
    synth1.set_filter(filter_var.get(), synth1.cutoff, float(val))

def update_filter(event=None):
    synth1.set_filter(filter_var.get(), synth1.cutoff, synth1.resonance)

def toggle_delay():
    synth1.set_delay(not synth1.delay_on, synth1.delay_amount, synth1.delay_time_sec)
    update_button_color(delay_button, synth1.delay_on)

def set_delay_amount(val):
    synth1.set_delay(synth1.delay_on, float(val)/100.0, synth1.delay_time_sec)

def set_delay_time(val):
    synth1.set_delay(synth1.delay_on, synth1.delay_amount, float(val)/1000.0)

def set_octave(val):
    synth1.set_octave(int(float(val)))

def toggle_lfo():
    synth1.set_lfo(on=not synth1.lfo_on)
    update_button_color(lfo_button, synth1.lfo_on)

def set_lfo_rate(val):
    synth1.set_lfo(rate=float(val))

def set_lfo_depth(val):
    synth1.set_lfo(depth=float(val)/100.0)

def set_lfo_mode(event=None):
    synth1.set_lfo(mode=lfo_mode_var.get())

def set_lfo_sync(event=None):
    synth1.set_lfo(sync_div=lfo_sync_var.get())

def bmp_up():
    if synth1.bpm < 300:
        synth1.bmp += 1
        synth2.bpm = synth1.bpm
        bpm_label.config(text=f"BPM: {synth1.bpm}")

def bmp_down():
    if synth1.bpm > 30:
        synth1.bmp -= 1
        synth2.bpm = synth1.bpm
        bpm_label.config(text=f"BPM: {synth1.bpm}")

# -----------------------------
# OSC2 Callbacks
# -----------------------------
def update_wave2(event=None):
    synth2.set_wave(wave_var2.get())

def update_freq2(val):
    synth2.set_frequency(float(val))

def update_cutoff2(val):
    synth2.set_filter(filter_var2.get(), float(val), synth2.resonance)

def update_resonance2(val):
    synth2.set_filter(filter_var2.get(), synth2.cutoff, float(val))

def update_filter2(event=None):
    synth2.set_filter(filter_var2.get(), synth2.cutoff, synth2.resonance)

def toggle_delay2():
    synth2.set_delay(not synth2.delay_on, synth2.delay_amount, synth2.delay_time_sec)
    update_button_color(delay_button2, synth2.delay_on)

def set_delay_amount2(val):
    synth2.set_delay(synth2.delay_on, float(val)/100.0, synth2.delay_time_sec)

def set_delay_time2(val):
    synth2.set_delay(synth2.delay_on, synth2.delay_amount, float(val)/1000.0)

def set_octave2(val):
    synth2.set_octave(int(float(val)))

def toggle_lfo2():
    synth2.set_lfo(on=not synth2.lfo_on)
    update_button_color(lfo_button2, synth2.lfo_on)

def set_lfo_rate2(val):
    synth2.set_lfo(rate=float(val))

def set_lfo_depth2(val):
    synth2.set_lfo(depth=float(val)/100.0)

def set_lfo_mode2(event=None):
    synth2.set_lfo(mode=lfo_mode_var2.get())

def set_lfo_sync2(event=None):
    synth2.set_lfo(sync_div=lfo_sync_var2.get())

# -----------------------------
# Arpeggiator Callbacks
# -----------------------------
def toggle_arp2():
    synth2.arp_on = not synth2.arp_on
    update_button_color(arp_button2, synth2.arp_on)
    if synth2.arp_on:
        synth2.arp_idx = 0
        synth2.arp_phase_samples = 0.0

def set_arp_mode2(event=None):
    synth2.arp_mode = arp_mode_var2.get()

def set_arp_rate2(val):
    try:
        synth2.arp_rate = float(val)
    except:
        pass

def set_arp_sync2(event=None):
    synth2.arp_sync_div = arp_sync_var2.get()

def set_arp_note2(idx, val):
    try:
        synth2.arp_notes[int(idx)] = int(float(val))
    except:
        pass

def set_arp_octave2(idx, val):
    try:
        synth2.arp_octaves[int(idx)] = int(float(val))
    except:
        pass

# -----------------------------
# Build GUI
# -----------------------------
def build_gui():
    global root, fig, ax, canvas_mpl, fig_spec, ax_spec, canvas_spec
    global osc1_button, osc2_button, sync_button, osc_button, spec_button
    global delay_button, delay_button2, lfo_button, lfo_button2, arp_button2
    global bpm_label
    global mod_source_var, mod_target_var, mod_amount_var
    global wave_var, filter_var, lfo_mode_var, lfo_sync_var
    global wave_var2, filter_var2, lfo_mode_var2, lfo_sync_var2
    global arp_mode_var2, arp_sync_var2
    
    root = tk.Tk()
    root.title("Dual Oscillator Synth with Oscilloscope & Filter Spectrogram")
    root.configure(bg="#000000")
    root.geometry("1400x900")

    style = ttk.Style()
    try:
        style.theme_use("clam")
    except Exception:
        pass

    green = "#00FF44"
    dark = "#001100"

    style.configure("TLabel", background="#000000", foreground=green, font=("Consolas", 9))
    style.configure("TButton", background="#000000", foreground=green, font=("Consolas", 9))
    style.configure("TFrame", background="#000000")
    style.configure("TCombobox", fieldbackground="#000000", background="#000000", foreground=green)
    style.map("TButton", background=[("active", dark)], foreground=[("active", green)])
    # Changed from Red to Green for active buttons
    style.configure("Green.TButton", background="#00AA22", foreground="#FFFFFF", font=("Consolas", 9))
    style.map("Green.TButton", background=[("active", "#00DD33")], foreground=[("active", "#FFFFFF")])

    main_container = ttk.Frame(root, padding=6, style="TFrame")
    main_container.pack(fill="both", expand=True)

    # Master controls
    master_frame = ttk.Frame(main_container, padding=4, style="TFrame")
    master_frame.pack(fill="x", pady=(0,8))

    osc1_button = ttk.Button(master_frame, text="OSC1", command=toggle_osc1)
    osc1_button.pack(side="left", padx=(0,5))

    osc2_button = ttk.Button(master_frame, text="OSC2", command=toggle_osc2)
    osc2_button.pack(side="left", padx=(0,15))

    sync_button = ttk.Button(master_frame, text="Sync OSCs", command=toggle_sync)
    sync_button.pack(side="left", padx=(0,15))

    ttk.Label(master_frame, text="Master Pitch:").pack(side="left", padx=(0,4))
    master_pitch_slider = tk.Scale(master_frame, from_=-8, to=8, orient="horizontal", 
                                  command=set_master_pitch, length=96,
                                  bg="#000000", fg=green, troughcolor="#003300", highlightbackground="#000000")
    master_pitch_slider.set(0)
    master_pitch_slider.pack(side="left", padx=(0,15))

    bmp_down_btn = ttk.Button(master_frame, text="BPM-", command=bmp_down)
    bmp_down_btn.pack(side="left", padx=1)
    bpm_label = ttk.Label(master_frame, text=f"BPM: {synth1.bpm}")
    bpm_label.pack(side="left", padx=(4,4))
    bmp_up_btn = ttk.Button(master_frame, text="BPM+", command=bmp_up)
    bmp_up_btn.pack(side="left", padx=1)

    start_btn = ttk.Button(master_frame, text="Start", command=start_audio)
    start_btn.pack(side="right", padx=3)
    stop_btn = ttk.Button(master_frame, text="Stop", command=stop_audio)
    stop_btn.pack(side="right", padx=3)

    if MATPLOTLIB_AVAILABLE:
        osc_button = ttk.Button(master_frame, text="Oscilloscope", command=toggle_oscilloscope)
        osc_button.pack(side="right", padx=5)
        spec_button = ttk.Button(master_frame, text="Filter Spectrum", command=toggle_spectrogram)
        spec_button.pack(side="right", padx=5)

    # Negative Matrix Section
    mod_frame = ttk.Frame(main_container, padding=4, style="TFrame")
    mod_frame.pack(fill="x", pady=(0,8))

    title_frame = ttk.Frame(mod_frame, style="TFrame")
    title_frame.pack(fill="x", pady=(0,4))
    ttk.Label(title_frame, text="NEGATIVE MATRIX", font=("Consolas", 11, "bold")).pack(side="left")
    ttk.Label(title_frame, text="by Gabriel Trentini", font=("Consolas", 6)).pack(side="right")

    mod_controls_frame = ttk.Frame(mod_frame, style="TFrame")
    mod_controls_frame.pack(fill="x")

    ttk.Label(mod_controls_frame, text="Source:").pack(side="left", padx=(0,4))
    mod_source_var = tk.StringVar(value="None")
    mod_source_menu = ttk.Combobox(mod_controls_frame, textvariable=mod_source_var,
        values=["None", "LFO1", "LFO2"], state="readonly", width=8)
    mod_source_menu.pack(side="left", padx=(0,8))

    ttk.Label(mod_controls_frame, text="Target:").pack(side="left", padx=(0,4))
    mod_target_var = tk.StringVar(value="None")
    mod_target_menu = ttk.Combobox(mod_controls_frame, textvariable=mod_target_var,
        values=["None", "OSC1_FREQ", "OSC1_CUTOFF", "OSC1_RESONANCE", "OSC1_DELAY_TIME", "OSC1_DELAY_AMOUNT",
                "OSC2_FREQ", "OSC2_CUTOFF", "OSC2_RESONANCE", "OSC2_DELAY_TIME", "OSC2_DELAY_AMOUNT"], 
        state="readonly", width=15)
    mod_target_menu.pack(side="left", padx=(0,8))

    ttk.Label(mod_controls_frame, text="Amount:").pack(side="left", padx=(0,4))
    mod_amount_var = tk.IntVar(value=0)
    mod_amount_slider = tk.Scale(mod_controls_frame, from_=-100, to=100, orient="horizontal", 
                                variable=mod_amount_var, length=120,
                                bg="#000000", fg=green, troughcolor="#003300", highlightbackground="#000000")
    mod_amount_slider.pack(side="left", padx=(0,8))

    reset_btn = ttk.Button(mod_controls_frame, text="Reset All", command=reset_modulation)
    reset_btn.pack(side="right")

    mod_source_var.trace('w', update_modulation)
    mod_target_var.trace('w', update_modulation)
    mod_amount_var.trace('w', update_modulation)

    # Visualization Frame (both oscilloscope and spectrogram)
    if MATPLOTLIB_AVAILABLE:
        viz_frame = ttk.Frame(main_container, padding=4, style="TFrame")
        viz_frame.pack(fill="x", pady=(0,8))

        # Oscilloscope
        fig, ax = plt.subplots(figsize=(7, 2.5), facecolor='#000000')
        fig.patch.set_facecolor('#000000')
        ax.set_facecolor('#000000')
        ax.set_xlim(0, oscilloscope.buffer_size/44100 * 1000)
        ax.set_ylim(-1.1, 1.1)
        ax.set_xlabel('Time (ms)', color='#00FF44', fontsize=8)
        ax.set_ylabel('Amplitude', color='#00FF44', fontsize=8)
        ax.set_title('Real-Time Signal Oscilloscope', color='#00FF44', fontsize=9)
        ax.grid(True, alpha=0.3, color='#00FF44')
        ax.tick_params(colors='#00FF44', labelsize=7)

        canvas_mpl = FigureCanvasTkAgg(fig, viz_frame)
        canvas_mpl.get_tk_widget().pack(side="left", fill="both", expand=True, padx=(0,5))

        # Filter Spectrogram
        fig_spec, ax_spec = plt.subplots(figsize=(7, 2.5), facecolor='#000000')
        fig_spec.patch.set_facecolor('#000000')
        ax_spec.set_facecolor('#000000')
        ax_spec.set_xlim(10, 22000)
        ax_spec.set_ylim(-60, 6)
        ax_spec.set_xlabel('Frequency (Hz)', color='#00FF44', fontsize=8)
        ax_spec.set_ylabel('Magnitude (dB)', color='#00FF44', fontsize=8)
        ax_spec.set_title('Filter Frequency Response', color='#00FF44', fontsize=9)
        ax_spec.grid(True, alpha=0.3, color='#00FF44')
        ax_spec.tick_params(colors='#00FF44', labelsize=7)

        canvas_spec = FigureCanvasTkAgg(fig_spec, viz_frame)
        canvas_spec.get_tk_widget().pack(side="right", fill="both", expand=True, padx=(5,0))

    # Main scrollable frame
    main_scroll_frame = ttk.Frame(main_container, style="TFrame")
    main_scroll_frame.pack(fill="both", expand=True)

    canvas_scroll = tk.Canvas(main_scroll_frame, bg="#000000", highlightthickness=0)
    scrollbar = ttk.Scrollbar(main_scroll_frame, orient="vertical", command=canvas_scroll.yview)
    scrollable_frame = ttk.Frame(canvas_scroll, style="TFrame")

    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas_scroll.configure(scrollregion=canvas_scroll.bbox("all"))
    )

    canvas_scroll.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas_scroll.configure(yscrollcommand=scrollbar.set)

    columns_frame = ttk.Frame(scrollable_frame, style="TFrame")
    columns_frame.pack(fill="both", expand=True, padx=5, pady=5)

    # OSC1 Column
    osc1_frame = ttk.Frame(columns_frame, padding=6, style="TFrame")
    osc1_frame.pack(side="left", fill="both", expand=True, padx=(0,4))

    osc1_title = ttk.Label(osc1_frame, text="OSCILLATOR 1", font=("Consolas", 11, "bold"))
    osc1_title.pack(pady=(0,8))

    ttk.Label(osc1_frame, text="A. Waveform").pack(anchor="w", pady=(1,0))
    wave_var = tk.StringVar(value="Sine")
    wave_menu = ttk.Combobox(osc1_frame, textvariable=wave_var,
        values=["Sine", "Saw", "Square", "Triangle", "Sample&Hold"], state="readonly")
    wave_menu.bind("<<ComboboxSelected>>", update_wave)
    wave_menu.pack(fill="x", pady=(0,4))

    ttk.Label(osc1_frame, text="B. Frequency").pack(anchor="w")
    freq_slider = tk.Scale(osc1_frame, from_=50, to=5000, orient="horizontal", command=update_freq,
                           bg="#000000", fg=green, troughcolor="#003300", highlightbackground="#000000", length=160)
    freq_slider.set(220)
    freq_slider.pack(fill="x", pady=(0,4))

    ttk.Label(osc1_frame, text="C. Filter Type").pack(anchor="w")
    filter_var = tk.StringVar(value="Lowpass")
    filter_menu = ttk.Combobox(osc1_frame, textvariable=filter_var,
        values=["Lowpass", "Highpass", "Bandpass"], state="readonly")
    filter_menu.bind("<<ComboboxSelected>>", update_filter)
    filter_menu.pack(fill="x", pady=(0,4))

    ttk.Label(osc1_frame, text="D. Cutoff Frequency").pack(anchor="w")
    cutoff_slider = tk.Scale(osc1_frame, from_=100, to=8000, orient="horizontal", command=update_cutoff,
                             bg="#000000", fg=green, troughcolor="#003300", highlightbackground="#000000", length=160)
    cutoff_slider.set(1000)
    cutoff_slider.pack(fill="x", pady=(0,4))

    ttk.Label(osc1_frame, text="E. Resonance").pack(anchor="w")
    res_slider = tk.Scale(osc1_frame, from_=0.1, to=2.0, resolution=0.1, orient="horizontal",
                          command=update_resonance, bg="#000000", fg=green, troughcolor="#003300", highlightbackground="#000000", length=160)
    res_slider.set(0.7)
    res_slider.pack(fill="x", pady=(0,6))

    ttk.Label(osc1_frame, text="F. Delay On/Off").pack(anchor="w")
    delay_button = ttk.Button(osc1_frame, text="Delay", command=toggle_delay)
    delay_button.pack(anchor="w", pady=(0,3))

    ttk.Label(osc1_frame, text="G. Delay Amount (%)").pack(anchor="w")
    delay_slider = tk.Scale(osc1_frame, from_=0, to=100, orient="horizontal", command=set_delay_amount,
                            bg="#000000", fg=green, troughcolor="#003300", highlightbackground="#000000", length=160)
    delay_slider.set(int(synth1.delay_amount*100))
    delay_slider.pack(fill="x", pady=(0,4))

    ttk.Label(osc1_frame, text="H. Delay Time (ms)").pack(anchor="w")
    delay_time_slider = tk.Scale(osc1_frame, from_=10, to=1000, orient="horizontal", command=set_delay_time,
                                 bg="#000000", fg=green, troughcolor="#003300", highlightbackground="#000000", length=160)
    delay_time_slider.set(int(synth1.delay_time_sec*1000))
    delay_time_slider.pack(fill="x", pady=(0,6))

    ttk.Label(osc1_frame, text="I. Octave Shift").pack(anchor="w")
    octave_slider = tk.Scale(osc1_frame, from_=-3, to=3, orient="horizontal", command=set_octave,
                             bg="#000000", fg=green, troughcolor="#003300", highlightbackground="#000000", length=160)
    octave_slider.set(0)
    octave_slider.pack(fill="x", pady=(0,6))

    ttk.Label(osc1_frame, text="J. LFO On/Off").pack(anchor="w")
    lfo_button = ttk.Button(osc1_frame, text="LFO", command=toggle_lfo)
    lfo_button.pack(anchor="w", pady=(0,3))

    ttk.Label(osc1_frame, text="K. LFO Mode").pack(anchor="w")
    lfo_mode_var = tk.StringVar(value="Hz")
    lfo_mode_menu = ttk.Combobox(osc1_frame, textvariable=lfo_mode_var,
        values=["Hz", "Sync"], state="readonly")
    lfo_mode_menu.bind("<<ComboboxSelected>>", set_lfo_mode)
    lfo_mode_menu.pack(fill="x", pady=(0,3))

    ttk.Label(osc1_frame, text="L. LFO Rate (Hz)").pack(anchor="w")
    lfo_rate_slider = tk.Scale(osc1_frame, from_=0.1, to=20.0, resolution=0.1, orient="horizontal", command=set_lfo_rate,
                               bg="#000000", fg=green, troughcolor="#003300", highlightbackground="#000000", length=160)
    lfo_rate_slider.set(5.0)
    lfo_rate_slider.pack(fill="x", pady=(0,4))

    ttk.Label(osc1_frame, text="M. LFO Sync Division").pack(anchor="w")
    lfo_sync_var = tk.StringVar(value="1/4")
    lfo_sync_menu = ttk.Combobox(osc1_frame, textvariable=lfo_sync_var,
        values=["1/1", "1/2", "1/4", "1/8", "1/16"], state="readonly")
    lfo_sync_menu.bind("<<ComboboxSelected>>", set_lfo_sync)
    lfo_sync_menu.pack(fill="x", pady=(0,4))

    ttk.Label(osc1_frame, text="N. LFO Depth (%)").pack(anchor="w")
    lfo_depth_slider = tk.Scale(osc1_frame, from_=0, to=100, orient="horizontal", command=set_lfo_depth,
                                bg="#000000", fg=green, troughcolor="#003300", highlightbackground="#000000", length=160)
    lfo_depth_slider.set(int(synth1.lfo_depth*100))
    lfo_depth_slider.pack(fill="x", pady=(0,8))

    # OSC2 Column
    osc2_frame = ttk.Frame(columns_frame, padding=6, style="TFrame")
    osc2_frame.pack(side="right", fill="both", expand=True, padx=(4,0))

    osc2_title = ttk.Label(osc2_frame, text="OSCILLATOR 2", font=("Consolas", 11, "bold"))
    osc2_title.pack(pady=(0,8))

    ttk.Label(osc2_frame, text="O. Waveform").pack(anchor="w", pady=(1,0))
    wave_var2 = tk.StringVar(value="Sine")
    wave_menu2 = ttk.Combobox(osc2_frame, textvariable=wave_var2,
        values=["Sine", "Saw", "Square", "Triangle", "Sample&Hold"], state="readonly")
    wave_menu2.bind("<<ComboboxSelected>>", update_wave2)
    wave_menu2.pack(fill="x", pady=(0,4))

    ttk.Label(osc2_frame, text="P. Frequency").pack(anchor="w")
    freq_slider2 = tk.Scale(osc2_frame, from_=50, to=5000, orient="horizontal", command=update_freq2,
                            bg="#000000", fg=green, troughcolor="#003300", highlightbackground="#000000", length=160)
    freq_slider2.set(440)
    freq_slider2.pack(fill="x", pady=(0,4))

    ttk.Label(osc2_frame, text="Q. Filter Type").pack(anchor="w")
    filter_var2 = tk.StringVar(value="Lowpass")
    filter_menu2 = ttk.Combobox(osc2_frame, textvariable=filter_var2,
        values=["Lowpass", "Highpass", "Bandpass"], state="readonly")
    filter_menu2.bind("<<ComboboxSelected>>", update_filter2)
    filter_menu2.pack(fill="x", pady=(0,4))

    ttk.Label(osc2_frame, text="R. Cutoff Frequency").pack(anchor="w")
    cutoff_slider2 = tk.Scale(osc2_frame, from_=100, to=8000, orient="horizontal", command=update_cutoff2,
                              bg="#000000", fg=green, troughcolor="#003300", highlightbackground="#000000", length=160)
    cutoff_slider2.set(1000)
    cutoff_slider2.pack(fill="x", pady=(0,4))

    ttk.Label(osc2_frame, text="S. Resonance").pack(anchor="w")
    res_slider2 = tk.Scale(osc2_frame, from_=0.1, to=2.0, resolution=0.1, orient="horizontal",
                           command=update_resonance2, bg="#000000", fg=green, troughcolor="#003300", highlightbackground="#000000", length=160)
    res_slider2.set(0.7)
    res_slider2.pack(fill="x", pady=(0,6))

    ttk.Label(osc2_frame, text="T. Delay On/Off").pack(anchor="w")
    delay_button2 = ttk.Button(osc2_frame, text="Delay", command=toggle_delay2)
    delay_button2.pack(anchor="w", pady=(0,3))

    ttk.Label(osc2_frame, text="U. Delay Amount (%)").pack(anchor="w")
    delay_slider2 = tk.Scale(osc2_frame, from_=0, to=100, orient="horizontal", command=set_delay_amount2,
                             bg="#000000", fg=green, troughcolor="#003300", highlightbackground="#000000", length=160)
    delay_slider2.set(int(synth2.delay_amount*100))
    delay_slider2.pack(fill="x", pady=(0,4))

    ttk.Label(osc2_frame, text="V. Delay Time (ms)").pack(anchor="w")
    delay_time_slider2 = tk.Scale(osc2_frame, from_=10, to=1000, orient="horizontal", command=set_delay_time2,
                                  bg="#000000", fg=green, troughcolor="#003300", highlightbackground="#000000", length=160)
    delay_time_slider2.set(int(synth2.delay_time_sec*1000))
    delay_time_slider2.pack(fill="x", pady=(0,6))

    ttk.Label(osc2_frame, text="W. Octave Shift").pack(anchor="w")
    octave_slider2 = tk.Scale(osc2_frame, from_=-3, to=3, orient="horizontal", command=set_octave2,
                              bg="#000000", fg=green, troughcolor="#003300", highlightbackground="#000000", length=160)
    octave_slider2.set(0)
    octave_slider2.pack(fill="x", pady=(0,6))

    ttk.Label(osc2_frame, text="X. LFO On/Off").pack(anchor="w")
    lfo_button2 = ttk.Button(osc2_frame, text="LFO", command=toggle_lfo2)
    lfo_button2.pack(anchor="w", pady=(0,3))

    ttk.Label(osc2_frame, text="Y. LFO Mode").pack(anchor="w")
    lfo_mode_var2 = tk.StringVar(value="Hz")
    lfo_mode_menu2 = ttk.Combobox(osc2_frame, textvariable=lfo_mode_var2,
        values=["Hz", "Sync"], state="readonly")
    lfo_mode_menu2.bind("<<ComboboxSelected>>", set_lfo_mode2)
    lfo_mode_menu2.pack(fill="x", pady=(0,3))

    ttk.Label(osc2_frame, text="Z. LFO Rate (Hz)").pack(anchor="w")
    lfo_rate_slider2 = tk.Scale(osc2_frame, from_=0.1, to=20.0, resolution=0.1, orient="horizontal", command=set_lfo_rate2,
                                bg="#000000", fg=green, troughcolor="#003300", highlightbackground="#000000", length=160)
    lfo_rate_slider2.set(5.0)
    lfo_rate_slider2.pack(fill="x", pady=(0,4))

    ttk.Label(osc2_frame, text="AA. LFO Sync Division").pack(anchor="w")
    lfo_sync_var2 = tk.StringVar(value="1/4")
    lfo_sync_menu2 = ttk.Combobox(osc2_frame, textvariable=lfo_sync_var2,
        values=["1/1", "1/2", "1/4", "1/8", "1/16"], state="readonly")
    lfo_sync_menu2.bind("<<ComboboxSelected>>", set_lfo_sync2)
    lfo_sync_menu2.pack(fill="x", pady=(0,4))

    ttk.Label(osc2_frame, text="BB. LFO Depth (%)").pack(anchor="w")
    lfo_depth_slider2 = tk.Scale(osc2_frame, from_=0, to=100, orient="horizontal", command=set_lfo_depth2,
                                 bg="#000000", fg=green, troughcolor="#003300", highlightbackground="#000000", length=160)
    lfo_depth_slider2.set(int(synth2.lfo_depth*100))
    lfo_depth_slider2.pack(fill="x", pady=(0,8))

    # Arpeggiator Section
    ttk.Label(osc2_frame, text="II. Arp On/Off").pack(anchor="w")
    arp_button2 = ttk.Button(osc2_frame, text="Arp", command=toggle_arp2)
    arp_button2.pack(anchor="w", pady=(0,3))

    ttk.Label(osc2_frame, text="JJ. Arp Mode").pack(anchor="w")
    arp_mode_var2 = tk.StringVar(value="Hz")
    arp_mode_menu2 = ttk.Combobox(osc2_frame, textvariable=arp_mode_var2,
        values=["Hz", "Sync"], state="readonly")
    arp_mode_menu2.bind("<<ComboboxSelected>>", set_arp_mode2)
    arp_mode_menu2.pack(fill="x", pady=(0,3))

    ttk.Label(osc2_frame, text="KK. Arp Rate (Hz)").pack(anchor="w")
    arp_rate_slider2 = tk.Scale(osc2_frame, from_=0.5, to=20.0, resolution=0.1, orient="horizontal", command=set_arp_rate2,
                               bg="#000000", fg=green, troughcolor="#003300", highlightbackground="#000000", length=160)
    arp_rate_slider2.set(4.0)
    arp_rate_slider2.pack(fill="x", pady=(0,4))

    ttk.Label(osc2_frame, text="LL. Arp Sync Division").pack(anchor="w")
    arp_sync_var2 = tk.StringVar(value="1/8")
    arp_sync_menu2 = ttk.Combobox(osc2_frame, textvariable=arp_sync_var2,
        values=["1/1", "1/2", "1/4", "1/8", "1/16"], state="readonly")
    arp_sync_menu2.bind("<<ComboboxSelected>>", set_arp_sync2)
    arp_sync_menu2.pack(fill="x", pady=(0,4))

    # Arp Notes
    ttk.Label(osc2_frame, text="MM. Arp Notes (Semitones)", font=("Consolas", 9, "bold")).pack(anchor="w", pady=(4,2))
    arp_notes_frame2 = ttk.Frame(osc2_frame, style="TFrame")
    arp_notes_frame2.pack(fill="x", pady=(0,4))

    for i in range(8):
        note_frame = ttk.Frame(arp_notes_frame2, style="TFrame")
        note_frame.pack(fill="x", pady=1)
        ttk.Label(note_frame, text=f"{i+1}:", font=("Consolas", 8)).pack(side="left", padx=(0,2))
        slider = tk.Scale(note_frame, from_=-24, to=24, orient="horizontal", 
                         command=lambda val, idx=i: set_arp_note2(idx, val),
                         bg="#000000", fg=green, troughcolor="#003300", highlightbackground="#000000", length=120)
        slider.set(0)
        slider.pack(side="left", padx=(0,4))

    # Arp Octaves
    ttk.Label(osc2_frame, text="NN. Arp Octaves", font=("Consolas", 9, "bold")).pack(anchor="w", pady=(4,2))
    arp_octaves_frame2 = ttk.Frame(osc2_frame, style="TFrame")
    arp_octaves_frame2.pack(fill="x", pady=(0,8))

    for i in range(8):
        oct_frame = ttk.Frame(arp_octaves_frame2, style="TFrame")
        oct_frame.pack(fill="x", pady=1)
        ttk.Label(oct_frame, text=f"{i+1}:", font=("Consolas", 8)).pack(side="left", padx=(0,2))
        slider = tk.Scale(oct_frame, from_=-4, to=4, orient="horizontal", 
                         command=lambda val, idx=i: set_arp_octave2(idx, val),
                         bg="#000000", fg=green, troughcolor="#003300", highlightbackground="#000000", length=120)
        slider.set(0)
        slider.pack(side="left", padx=(0,4))

    canvas_scroll.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    def _on_mousewheel(event):
        canvas_scroll.yview_scroll(int(-1*(event.delta/120)), "units")

    canvas_scroll.bind_all("<MouseWheel>", _on_mousewheel)

    # Initialize button colors
    def initialize_button_colors():
        update_button_color(osc1_button, synth1.enabled)
        update_button_color(osc2_button, synth2.enabled)
        update_button_color(sync_button, sync_oscs)
        if MATPLOTLIB_AVAILABLE:
            update_button_color(osc_button, oscilloscope.enabled)
            update_button_color(spec_button, filter_spectrogram.enabled)
        update_button_color(delay_button, synth1.delay_on)
        update_button_color(delay_button2, synth2.delay_on)
        update_button_color(lfo_button, synth1.lfo_on)
        update_button_color(lfo_button2, synth2.lfo_on)
        update_button_color(arp_button2, synth2.arp_on)

    initialize_button_colors()

    root.resizable(True, True)
    return root

# -----------------------------
# Main Execution
# -----------------------------
if __name__ == "__main__":
    root = build_gui()
    root.mainloop()