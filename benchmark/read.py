from datetime import datetime
import argparse
from pathlib import Path
import json

parser = argparse.ArgumentParser(
    prog="read.py",
    description="Le programme de lecture des benchmarks Idefix"
)

parser.add_argument('--run-directory', type=str, default='./')

args = parser.parse_args()

current_dir = Path(args.run_directory)

performances_data = {}
date_dirs = [d for d in current_dir.iterdir() if d.is_dir()]
for date_dir in date_dirs:
    performances_data[date_dir.name] = {}
    problem_size_dirs = [d for d in date_dir.iterdir() if d.is_dir()]
    for problem_size_dir in problem_size_dirs:
        performances_data[date_dir.name][problem_size_dir.name] = {}
        ncores_dirs = [d for d in problem_size_dir.iterdir() if d.is_dir()]
        for ncores_dir in sorted(ncores_dirs, key=lambda x: int(x.name)):
            output_file = ncores_dir / "astra.0.log"
            if output_file.exists():
                print(f"Reading performance from {output_file}")
                with open(output_file, 'r') as f:
                    tag="Main: Perfs"
                    count = 0
                    perfs = 0.0
                    try:
                        while True:
                            count=count+1
                            line=f.readline()
                            if tag in line:
                                perfs=float(line.split()[3])
                                f.close()
                                break
                            if count > 500000:
                                raise ValueError("cannot locate end of header in " + file)
                    except:
                        print("Exception in read")
                        f.close()
                performances_data[date_dir.name][problem_size_dir.name][ncores_dir.name] = perfs
                
            else:
                print(f"Output file not found in {ncores_dir}")
# Save the performance data to a JSON file
with open(current_dir / 'performances_data.json', 'w') as json_file:
    json.dump(performances_data, json_file, ensure_ascii=False, indent=4)

print("Performance data has been written to " + str(current_dir / 'performances_data.json'))
