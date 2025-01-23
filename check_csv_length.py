import csv
import wave
import contextlib

def average_wav_length(csv_file_path, audio_path_column_index=1):
    total_length = 0.0
    file_count = 0
    min_length = float("inf")
    max_length = float("-inf")
    max_length_path = None

    with open(csv_file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)  # skip header row if there is one
        for row in reader:
            if not row:
                continue
            audio_file = row[audio_path_column_index]
            try:
                with contextlib.closing(wave.open(audio_file, 'rb')) as wav:
                    frames = wav.getnframes()
                    rate = wav.getframerate()
                    duration = frames / float(rate)
                    total_length += duration
                    file_count += 1
                    if duration < min_length:
                        min_length = duration
                    if duration > max_length:
                        max_length = duration
                        max_length_path = audio_file
            except:
                pass  # skip files that can't be opened as wav

    if file_count > 0:
        avg_length = total_length / file_count
        return avg_length, min_length, max_length, max_length_path
    else:
        return 0.0, 0.0, 0.0, None

if __name__ == "__main__":
    import sys
    csv_path = sys.argv[1]
    avg_length, min_length, max_length, max_path = average_wav_length(csv_path, audio_path_column_index=1)
    print(f"Average WAV length: {avg_length:.2f} seconds")
    print(f"Min WAV length: {min_length:.2f} seconds")
    print(f"Max WAV length: {max_length:.2f} seconds")
    print(f"Max WAV length path: {max_path}")