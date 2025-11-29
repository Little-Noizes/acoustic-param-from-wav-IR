"""
Streamlit Web App for Room Acoustic Analysis
Upload WAV files and get T20, T30, EDT, C50, C80, D50, D80 results
Based on v3 implementation (best match to AARAE)
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import zipfile
from pathlib import Path
import tempfile

import pyfar as pf
import pyrato.rap as ra_params

# ============================================================
# CONFIGURATION
# ============================================================

OCTAVE_BANDS = np.array([31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000])
START_THRESH_DB = -20
NOISE_COMP = True

st.set_page_config(
    page_title="Room Acoustics Analyzer",
    page_icon="🔊",
    layout="wide"
)

# ============================================================
# HELPER FUNCTIONS (from v3)
# ============================================================

def load_mono_signal(uploaded_file) -> pf.Signal:
    """Load WAV via pyfar and convert to mono if multi-channel."""
    # Save uploaded file to temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name
    
    sig = pf.io.read_audio(tmp_path)
    
    # Clean up temp file
    Path(tmp_path).unlink()
    
    if sig.cshape[0] > 1:
        mono = np.mean(sig.time, axis=0, keepdims=True)
        sig = pf.Signal(mono, sig.sampling_rate)
    return sig


def filter_to_octave_bands(sig: pf.Signal, bands: np.ndarray):
    """Filter signal into octave bands using pyfar."""
    fmin = bands[0] / np.sqrt(2)
    fmax = bands[-1] * np.sqrt(2)

    sig_bands = pf.dsp.filter.fractional_octave_bands(
        sig, num_fractions=1, frequency_range=(fmin, fmax)
    )

    freq_result = pf.dsp.filter.fractional_octave_frequencies(
        num_fractions=1, frequency_range=(fmin, fmax)
    )
    
    if isinstance(freq_result, tuple):
        centre_freqs = freq_result[1]
    else:
        centre_freqs = freq_result

    if sig_bands.cshape[0] != len(centre_freqs):
        n = min(sig_bands.cshape[0], len(centre_freqs))
        sig_bands = pf.Signal(sig_bands.time[:n, :], sig_bands.sampling_rate)
        centre_freqs = centre_freqs[:n]

    return sig_bands, centre_freqs


def detect_ir_start(ir_data, thresh_db=-20):
    """Detect IR start point using threshold method."""
    ir_squared = ir_data ** 2
    peak_value = np.max(ir_squared)
    threshold = peak_value / (10 ** (abs(thresh_db) / 10))
    
    above_threshold = np.where(ir_squared >= threshold)[0]
    if len(above_threshold) > 0:
        startpoint = above_threshold[0]
    else:
        startpoint = 0
    
    return startpoint


def circular_shift_to_start(ir_data, startpoint):
    """Circularly shift IR so start is at index 0."""
    if startpoint > 0:
        ir_zeroed = ir_data.copy()
        ir_zeroed[:startpoint] = 0
        ir_shifted = np.roll(ir_zeroed, -startpoint)
    else:
        ir_shifted = ir_data.copy()
    
    return ir_shifted


def subtract_noise_chu(ir_data, fs):
    """Subtract noise using Chu method - subtracts energy of final 10%."""
    n_samples = len(ir_data)
    start_10pct = int(0.9 * n_samples)
    
    noise_segment = ir_data[start_10pct:]
    noise_power = np.mean(noise_segment ** 2)
    
    ir_squared = ir_data ** 2
    ir_squared_corrected = np.maximum(ir_squared - noise_power, 0)
    ir_corrected = np.sqrt(ir_squared_corrected) * np.sign(ir_data)
    
    return ir_corrected


def compute_schroeder_edc(ir_data):
    """Compute Energy Decay Curve using Schroeder backward integration."""
    energy = ir_data ** 2
    edc = np.flip(np.cumsum(np.flip(energy)))
    
    if edc[0] > 0:
        edc_normalized = edc / edc[0]
    else:
        edc_normalized = edc
    
    return edc_normalized


def compute_clarity_matlab_style(ir_data, fs):
    """Compute C50, C80, D50, D80 using MATLAB's approach."""
    results = {'C50': np.nan, 'C80': np.nan, 'D50': np.nan, 'D80': np.nan}
    
    # C50 calculation
    early_end_50 = int(np.floor(fs * 0.05)) + 1
    late_start_50 = int(np.ceil(fs * 0.05))
    
    if early_end_50 < len(ir_data) and late_start_50 < len(ir_data):
        early_50 = np.sum(ir_data[:early_end_50] ** 2)
        late_50 = np.sum(ir_data[late_start_50:] ** 2)
        
        if late_50 > 0 and early_50 > 0:
            results['C50'] = 10 * np.log10(early_50 / late_50)
            results['D50'] = 100 * early_50 / (early_50 + late_50)
    
    # C80 calculation
    early_end_80 = int(np.floor(fs * 0.08)) + 1
    late_start_80 = int(np.ceil(fs * 0.08))
    
    if early_end_80 < len(ir_data) and late_start_80 < len(ir_data):
        early_80 = np.sum(ir_data[:early_end_80] ** 2)
        late_80 = np.sum(ir_data[late_start_80:] ** 2)
        
        if late_80 > 0 and early_80 > 0:
            results['C80'] = 10 * np.log10(early_80 / late_80)
            results['D80'] = 100 * early_80 / (early_80 + late_80)
    
    return results


def analyze_impulse_response(sig: pf.Signal, filename: str, progress_callback=None):
    """
    Main analysis function - returns DataFrame with results.
    """
    sig_bands, centre_freqs = filter_to_octave_bands(sig, OCTAVE_BANDS)
    
    n_bands = sig_bands.cshape[0]
    fs = sig_bands.sampling_rate
    
    results = []
    
    for i in range(n_bands):
        try:
            if progress_callback:
                progress_callback(i / n_bands)
            
            ir_band = sig_bands.time[i, :, :].flatten()
            
            if np.sum(ir_band ** 2) == 0:
                continue
            
            # Step 1: Detect start point
            startpoint = detect_ir_start(ir_band, START_THRESH_DB)
            
            # Step 2: Circular shift
            ir_shifted = circular_shift_to_start(ir_band, startpoint)
            
            # Step 3: Optional noise compensation
            if NOISE_COMP:
                ir_corrected = subtract_noise_chu(ir_shifted, fs)
            else:
                ir_corrected = ir_shifted
            
            # Step 4: Compute EDC
            edc_data = compute_schroeder_edc(ir_corrected)
            
            time_vec = np.arange(len(edc_data)) / fs
            edc = pf.TimeData(edc_data.reshape(1, -1), times=time_vec)
            
            # Step 5: Calculate reverberation times
            T20 = float(np.squeeze(ra_params.reverberation_time_linear_regression(edc, T="T20")))
            T30 = float(np.squeeze(ra_params.reverberation_time_linear_regression(edc, T="T30")))
            EDT = float(np.squeeze(ra_params.reverberation_time_linear_regression(edc, T="EDT")))
            
            # Step 6: Clarity parameters
            clarity = compute_clarity_matlab_style(ir_shifted, fs)
            
            freq_hz = float(np.squeeze(centre_freqs[i]))
            
            results.append({
                'file': filename,
                'freq_Hz': freq_hz,
                'T20': T20,
                'T30': T30,
                'EDT': EDT,
                'C50': clarity['C50'],
                'C80': clarity['C80'],
                'D50': clarity['D50'],
                'D80': clarity['D80'],
            })
            
        except Exception as e:
            st.warning(f"Failed to process band {i}: {str(e)}")
            continue
    
    if progress_callback:
        progress_callback(1.0)
    
    return pd.DataFrame(results)


def create_summary_plots(df: pd.DataFrame):
    """Create summary plots for all parameters."""
    params = [
        ('T20', 'T20 [s]', None),
        ('T30', 'T30 [s]', None),
        ('EDT', 'EDT [s]', None),
        ('C50', 'C50 [dB]', None),
        ('C80', 'C80 [dB]', None),
        ('D50', 'D50 [%]', (0, 100)),
        ('D80', 'D80 [%]', (0, 100)),
    ]
    
    figs = []
    
    for param_name, ylabel, ylim in params:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=100)
        
        # Plot each file
        for filename, group in df.groupby('file'):
            ax.plot(group['freq_Hz'], group[param_name], 
                   marker='o', linewidth=2, markersize=6, 
                   label=filename, alpha=0.7)
        
        ax.set_xscale('log')
        ax.set_xlabel('Frequency [Hz]', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f'{param_name} vs Frequency', fontsize=14, fontweight='bold')
        ax.grid(True, which='both', linestyle='--', alpha=0.3)
        ax.legend(fontsize=9)
        
        if ylim is not None:
            ax.set_ylim(ylim)
        
        fig.tight_layout()
        figs.append((param_name, fig))
    
    return figs


# ============================================================
# STREAMLIT APP
# ============================================================

def main():
    st.title("🔊 Room Acoustics Analyzer")
    st.markdown("""
    Upload impulse response WAV files to analyze room acoustic parameters.
    
    **Calculated Parameters:**
    - **T20, T30, EDT**: Reverberation times
    - **C50, C80**: Clarity indices
    - **D50, D80**: Definition percentages
    
    **Frequency Bands:** 31.5, 63, 125, 250, 500, 1k, 2k, 4k, 8k, 16k Hz
    """)
    
    # Sidebar for settings
    with st.sidebar:
        st.header("⚙️ Settings")
        
        exclude_63hz = st.checkbox("Exclude 63 Hz band", value=True, 
                                   help="Low frequency band can be unreliable")
        
        show_plots = st.checkbox("Generate plots", value=True)
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This tool implements the **v3 algorithm** which best matches 
        MATLAB AARAE ReverberationTimeIR2 results.
        
        **Algorithm features:**
        - Threshold-based IR start detection (-20 dB)
        - Chu noise compensation
        - Schroeder backward integration for EDC
        - MATLAB-matched clarity calculations
        """)
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Upload impulse response WAV files",
        type=['wav'],
        accept_multiple_files=True,
        help="You can upload multiple files at once"
    )
    
    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} file(s) uploaded")
        
        if st.button("🚀 Analyze", type="primary"):
            all_results = []
            
            # Progress tracking
            overall_progress = st.progress(0)
            status_text = st.empty()
            
            for file_idx, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"Processing {uploaded_file.name}... ({file_idx+1}/{len(uploaded_files)})")
                
                try:
                    # Load and analyze
                    sig = load_mono_signal(uploaded_file)
                    
                    # Create progress callback for this file
                    def progress_callback(pct):
                        total_progress = (file_idx + pct) / len(uploaded_files)
                        overall_progress.progress(total_progress)
                    
                    df = analyze_impulse_response(sig, uploaded_file.name, progress_callback)
                    all_results.append(df)
                    
                except Exception as e:
                    st.error(f"❌ Failed to process {uploaded_file.name}: {str(e)}")
                    continue
            
            overall_progress.progress(1.0)
            status_text.text("✅ Analysis complete!")
            
            if all_results:
                # Combine all results
                combined_df = pd.concat(all_results, ignore_index=True)
                
                # Optionally exclude 63 Hz
                if exclude_63hz:
                    combined_df = combined_df[combined_df['freq_Hz'] >= 100].copy()
                
                # Display results
                st.header("📊 Results")
                
                # Summary statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Files Analyzed", len(uploaded_files))
                with col2:
                    st.metric("Frequency Bands", len(combined_df['freq_Hz'].unique()))
                with col3:
                    avg_t30 = combined_df['T30'].mean()
                    st.metric("Average T30", f"{avg_t30:.2f} s")
                
                # Results table
                st.subheader("📋 Detailed Results")
                
                # Format the dataframe for display
                display_df = combined_df.copy()
                for col in ['T20', 'T30', 'EDT']:
                    display_df[col] = display_df[col].apply(lambda x: f"{x:.3f}")
                for col in ['C50', 'C80']:
                    display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}")
                for col in ['D50', 'D80']:
                    display_df[col] = display_df[col].apply(lambda x: f"{x:.1f}")
                
                st.dataframe(display_df, use_container_width=True, height=400)
                
                # Download CSV
                csv_buffer = io.StringIO()
                combined_df.to_csv(csv_buffer, index=False)
                csv_data = csv_buffer.getvalue()
                
                st.download_button(
                    label="📥 Download Results (CSV)",
                    data=csv_data,
                    file_name="acoustic_analysis_results.csv",
                    mime="text/csv"
                )
                
                # Generate plots
                if show_plots:
                    st.header("📈 Plots")
                    
                    with st.spinner("Generating plots..."):
                        figs = create_summary_plots(combined_df)
                    
                    # Display plots in tabs
                    tabs = st.tabs([name for name, _ in figs])
                    
                    for tab, (param_name, fig) in zip(tabs, figs):
                        with tab:
                            st.pyplot(fig)
                            
                            # Save plot to buffer for download
                            buf = io.BytesIO()
                            fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
                            buf.seek(0)
                            
                            st.download_button(
                                label=f"📥 Download {param_name} plot",
                                data=buf,
                                file_name=f"{param_name}_plot.png",
                                mime="image/png"
                            )
                    
                    # Create ZIP of all plots
                    zip_buffer = io.BytesIO()
                    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                        for param_name, fig in figs:
                            img_buf = io.BytesIO()
                            fig.savefig(img_buf, format='png', dpi=300, bbox_inches='tight')
                            img_buf.seek(0)
                            zip_file.writestr(f"{param_name}_plot.png", img_buf.read())
                    
                    zip_buffer.seek(0)
                    
                    st.download_button(
                        label="📦 Download All Plots (ZIP)",
                        data=zip_buffer,
                        file_name="acoustic_plots.zip",
                        mime="application/zip"
                    )
    
    else:
        st.info("👆 Upload one or more WAV files to begin analysis")
        
        # Example/demo section
        with st.expander("ℹ️ How to use this tool"):
            st.markdown("""
            ### Step-by-step guide:
            
            1. **Prepare your files**: Ensure you have impulse response recordings in WAV format
            2. **Upload files**: Click the upload button and select one or more WAV files
            3. **Configure settings**: Use the sidebar to adjust options (e.g., exclude 63 Hz)
            4. **Analyze**: Click the "Analyze" button to process your files
            5. **Review results**: Examine the table and plots
            6. **Download**: Save CSV results and plots for your records
            
            ### Expected file format:
            - **Format**: WAV (PCM)
            - **Content**: Room impulse response recordings
            - **Channels**: Mono or stereo (will be converted to mono)
            - **Sample rate**: Any (typically 44.1 kHz or 48 kHz)
            - **Duration**: Typically 1-10 seconds
            
            ### Parameter definitions:
            - **T20**: Reverberation time extrapolated from -5 to -25 dB decay
            - **T30**: Reverberation time extrapolated from -5 to -35 dB decay
            - **EDT**: Early Decay Time (0 to -10 dB)
            - **C50**: Clarity (ratio of energy in first 50ms to later energy)
            - **C80**: Clarity (ratio of energy in first 80ms to later energy)
            - **D50**: Definition (percentage of energy in first 50ms)
            - **D80**: Definition (percentage of energy in first 80ms)
            """)


if __name__ == "__main__":
    main()
