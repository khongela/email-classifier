import subprocess
import sys

# Run the three steps in order.
scripts = [
    'Ingestion/ingestion.py',
    'Modelling/modelling.py',
    'Reporting/reporting.py',
]

for script in scripts:
    print(f'Running {script}', flush=True)
    command = [sys.executable, script]
    subprocess.run(command, check=True)
