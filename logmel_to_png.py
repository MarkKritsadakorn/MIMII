import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import time


# Specify your dataset path and output path for saving log-Mel spectrograms
dataset_path = "D:\MIMII\dataset"
output_paths_1024_512 =  "D:\MIMII\logmel_1024_512"
output_paths_2048_512 =  "D:\MIMII\logmel_2048_512"
output_paths_4096_512 =  "D:\MIMII\logmel_4096_512"
output_paths_1024_256 =  "D:\MIMII\logmel_1024_256"
output_paths_2048_256 =  "D:\MIMII\logmel_2048_256"
output_paths_4096_256 =  "D:\MIMII\logmel_4096_256"


# Ensure each output directory exists
os.makedirs(output_paths_1024_512, exist_ok=True)
os.makedirs(output_paths_2048_512, exist_ok=True)
os.makedirs(output_paths_4096_512, exist_ok=True)
os.makedirs(output_paths_1024_256, exist_ok=True)
os.makedirs(output_paths_2048_256, exist_ok=True)
os.makedirs(output_paths_4096_256, exist_ok=True)


def logmel_1024_512():
    n_fft = 1024
    hop_length = 512
    n_mels = 128
    return n_fft, hop_length, n_mels

def logmel_2048_512():
    n_fft = 2048
    hop_length = 512
    n_mels = 128
    return n_fft, hop_length, n_mels

def logmel_4096_512():
    n_fft = 4096
    hop_length = 512
    n_mels = 128
    return n_fft, hop_length, n_mels

def logmel_1024_256():
    n_fft = 1024
    hop_length = 256
    n_mels = 128
    return n_fft, hop_length, n_mels

def logmel_2048_256():
    n_fft = 2048
    hop_length = 256
    n_mels = 128
    return n_fft, hop_length, n_mels

def logmel_4096_256():
    n_fft = 4096
    hop_length = 256
    n_mels = 128
    return n_fft, hop_length, n_mels



def process_audio_files(dataset_path, output_path, n_fft, hop_length, n_mels):

    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(".wav"):
                # Load the audio file
                file_path = os.path.join(root, file)
                y, sr = librosa.load(file_path, sr=None)

                # Generate the log-Mel spectrogram
                spect = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
                log_mel_spect = librosa.power_to_db(spect, ref=np.max)

                # Create a corresponding output folder structure
                relative_path = os.path.relpath(root, dataset_path)
                save_folder = os.path.join(output_path, relative_path)
                os.makedirs(save_folder, exist_ok=True)

                # Save the log-Mel spectrogram as a .png file
                save_path = os.path.join(save_folder, file.replace(".wav", ".png"))
                plt.imsave(save_path, log_mel_spect, cmap="viridis")



n_fft_1024, hop_length_512, n_mels = logmel_1024_512()
n_fft_2048, hop_length_512, n_mels = logmel_2048_512()
n_fft_4096, hop_length_512, n_mels = logmel_4096_512()

n_fft_1024, hop_length_256, n_mels = logmel_1024_256()
n_fft_2048, hop_length_256, n_mels = logmel_2048_256()
n_fft_4096, hop_length_256, n_mels = logmel_4096_256()



start = time.time()


start_1024_512 = time.time()
process_audio_files(dataset_path, output_paths_1024_512, n_fft_1024, hop_length_512, n_mels)
end_1024_512 = time.time()
print(f"Time taken for 1024 512 : {end_1024_512 - start_1024_512} seconds")

start_2048_512 = time.time()
process_audio_files(dataset_path, output_paths_2048_512, n_fft_2048, hop_length_512, n_mels)
end_2048_512 = time.time()
print(f"Time taken for 2048 512 : {end_2048_512 - start_2048_512} seconds")

start_4096_512 = time.time()
process_audio_files(dataset_path, output_paths_4096_512, n_fft_4096, hop_length_512, n_mels)
end_4096_512 = time.time()
print(f"Time taken for 4096 512 : {end_4096_512 - start_4096_512} seconds")

start_1024_256 = time.time()
process_audio_files(dataset_path, output_paths_1024_256, n_fft_1024, hop_length_256, n_mels)
end_1024_256 = time.time()
print(f"Time taken for 1024 256 : {end_1024_256 - start_1024_256} seconds")

start_2048_256 = time.time()
process_audio_files(dataset_path, output_paths_2048_256, n_fft_2048, hop_length_256, n_mels)
end_2048_256 = time.time()
print(f"Time taken for 2048 256 : {end_2048_256 - start_2048_256} seconds")

start_4096_256 = time.time()
process_audio_files(dataset_path, output_paths_4096_256, n_fft_4096, hop_length_256, n_mels)
end_4096_256 = time.time()
print(f"Time taken for 4096 256 : {end_4096_256 - start_4096_256} seconds")


end = time.time()

print(f"Total time taken : {end - start} seconds")
print("All done!")

# save the time taken to a file to .txt file

with open("D:\MIMII\logmel_time.txt", "w") as f:
    f.write(f"Time taken for 1024 512 : {end_1024_512 - start_1024_512} seconds\n")
    f.write(f"Time taken for 2048 512 : {end_2048_512 - start_2048_512} seconds\n")
    f.write(f"Time taken for 4096 512 : {end_4096_512 - start_4096_512} seconds\n")
    f.write(f"Time taken for 1024 256 : {end_1024_256 - start_1024_256} seconds\n")
    f.write(f"Time taken for 2048 256 : {end_2048_256 - start_2048_256} seconds\n")
    f.write(f"Time taken for 4096 256 : {end_4096_256 - start_4096_256} seconds\n")
    f.write(f"Total time taken : {end - start} seconds\n")
