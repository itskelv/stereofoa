import os
import pandas as pd
import librosa
import soundfile as sf
import numpy as np

def segment_dataset(src_audio_root, src_meta_dir, out_audio_dir, out_meta_dir, 
                    target_sr=24000, segment_len_s=25, fps=10):
    
     # Create output directories if they don't exist
    if not os.path.exists(out_audio_dir): os.makedirs(out_audio_dir)
    if not os.path.exists(out_meta_dir): os.makedirs(out_meta_dir)

    frames_per_seg = int(segment_len_s * fps) 
    samples_per_seg = int(segment_len_s * target_sr) 

    # os.walk travels through all subfolders automatically
    print(f"Searching for audio files in: {src_audio_root}")
    
    count = 0
    for root, dirs, files in os.walk(src_audio_root):
        for filename in files:
            if filename.endswith('.wav'):
                audio_path = os.path.join(root, filename)
                
                # Match metadata (Assuming .csv has same name as .wav)
                meta_filename = filename.replace('.wav', '.csv')
                meta_path = os.path.join(src_meta_dir, meta_filename)
                
                if not os.path.exists(meta_path):
                    # print(f"Skipping {filename}: Metadata not found at {meta_path}")
                    continue

                print(f"Processing: {filename}")
                count += 1
                
                # 1. Load Audio
                audio, _ = librosa.load(audio_path, sr=target_sr, mono=False)
                
                # 2. Load Metadata
                df = pd.read_csv(meta_path, header=None)
                # SELD metadata format: frame, class, source, azimuth, elevation
                df.columns = ['frame', 'class', 'source', 'azimuth', 'elevation']

                total_samples = audio.shape[-1]
                num_segments = int(np.ceil(total_samples / samples_per_seg))

                for i in range(num_segments):
                    start_sample = i * samples_per_seg
                    end_sample = start_sample + samples_per_seg
                    
                    start_frame = i * frames_per_seg
                    end_frame = start_frame + frames_per_seg

                    # --- Process Audio Chunk ---
                    chunk = audio[:, start_sample:end_sample]
                    if chunk.shape[-1] < samples_per_seg:
                        pad_width = samples_per_seg - chunk.shape[-1]
                        if chunk.ndim == 1: # Mono
                            chunk = np.pad(chunk, (0, pad_width), mode='constant')
                        else: # Multi-channel
                            chunk = np.pad(chunk, ((0, 0), (0, pad_width)), mode='constant')

                    # --- Process Metadata Chunk ---
                    mask = (df['frame'] >= start_frame) & (df['frame'] < end_frame)
                    chunk_df = df[mask].copy()
                    chunk_df['frame'] = chunk_df['frame'] - start_frame

                    # --- Save Files ---
                    # Include the original subfolder name in the segment name to avoid overwriting
                    parent_folder = os.path.basename(root)
                    seg_name = f"{parent_folder}_{os.path.splitext(filename)[0]}_seg{i}"
                    
                    # Save Audio (Transpose for soundfile)
                    sf.write(os.path.join(out_audio_dir, f"{seg_name}.wav"), chunk.T, target_sr)
                    
                    # Save CSV
                    chunk_df.to_csv(os.path.join(out_meta_dir, f"{seg_name}.csv"), index=False, header=False)

    print(f"Finished! Processed {count} source files.")

if __name__ == '__main__':
    segment_dataset(
        src_audio_root='../stereodataset/stereo_dev',
        src_meta_dir='../stereodataset/metadata_dev', 
        out_audio_dir='../stereodataset/segmented_audio', 
        out_meta_dir='../stereodataset/segmented_metadata'
    )