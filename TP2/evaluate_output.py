import subprocess
import numpy as np
import os

# Settings
device = "cuda:1"
reference_path = "./generated_images/CGAN/"
sample_template = "npz_dataset/sample_{}.npz"

# Store FID results
fid_scores = []
learning_rate = "001"
# Run FID for sample_1.npz to sample_5.npz

for random_sample in os.listdir(reference_path):

    generated_path = os.path.join(reference_path, random_sample, learning_rate)
    print(generated_path)
    for i in range(1, 6):

        sample_path = sample_template.format(i)
        command = [
            "python", "-m", "pytorch_fid",
            "--device", device,
            generated_path,
            sample_path
        ]

        print(f"Running FID for: {sample_path}")
        result = subprocess.run(command, capture_output=True, text=True)

        # Parse FID score from stdout
        output = result.stdout.strip()
        print("Output:", output)
        try:
            fid = float(output.split()[-1])
            fid_scores.append(fid)
        except (IndexError, ValueError):
            print(f"Failed to parse FID score from: {output}")

    # Compute mean and standard deviation
    if fid_scores:
        mean_fid = np.mean(fid_scores)
        std_fid = np.std(fid_scores)
        print("\nFID Results:")
        print("Scores:", fid_scores)
        print(f"Mean: {mean_fid:.2f}")
        print(f"Standard Deviation: {std_fid:.2f}")
    else:
        print("No FID scores collected.")
