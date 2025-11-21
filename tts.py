file_name = "input.txt"

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.rnn")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.nn.utils.weight_norm")

# Create output directory
output_dir = "output"
import os
os.makedirs(output_dir, exist_ok=True)

# Delete old outputs
for file in os.listdir(output_dir):
    os.remove(os.path.join(output_dir, file))

with open(file_name, "r") as f:
    text = f.read()

from kokoro import KPipeline
pipeline = KPipeline(lang_code='a')
generator = pipeline(text, voice='af_heart', speed=1, split_pattern=r'\n+')

for i, result in enumerate(generator, start=1):
    # Zero-padded filename starting from 001 (e.g., 001.wav, 002.wav, 003.wav, etc.)
    audio_path = f"{output_dir}/{i:03d}.wav"
    token_path = f"{output_dir}/{i:03d}.json"
    
    audio = result.audio
    tokens = result.tokens
    
    # Save as WAV file using soundfile (assuming 24kHz sample rate)
    import soundfile as sf
    sf.write(audio_path, audio, 24000)
    
    # Write tokens to JSON file
    token_data = [
        {
            "text": t.text,
            "whitespace": t.whitespace,
            "start_ts": t.start_ts,
            "end_ts": t.end_ts
        }
        for t in tokens
    ]
    with open(token_path, "w") as f:
        import json
        json.dump(token_data, f, indent=2)

print(f"\nOutput files saved to: {output_dir}")