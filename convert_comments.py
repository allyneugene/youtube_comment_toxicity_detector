import _curses_panel as pd
import os

input_txt = os.path.join("venv", "comments.txt")
output_csv = os.path.join("data", "comments.csv")

# Ensure output directory exists
os.makedirs("data", exist_ok=True)

# Check if input file exists
if not os.path.exists(input_txt):
    raise FileNotFoundError(f"Input file not found: {input_txt}")

try:
    # Read the txt file (comma-separated, header included)
    df = pd.read_csv(input_txt, encoding="utf-8")
except Exception as e:
    print(f"Error reading {input_txt}: {e}")
    exit(1)

# Save as CSV
df.to_csv(output_csv, index=False)
print(f"Converted {input_txt} to {output_csv}")