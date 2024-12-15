import subprocess
import time
import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))


while True:
    try:
        # Run the script using subprocess with the same command-line arguments
        result = subprocess.run(
            ['python', 'scripts/predict/download_and_predict.py'] + sys.argv[1:],
            text=True, check=True
        )

        # If download_and_predict ran successfully, break out of the loop
        break
    except subprocess.CalledProcessError as e:
        # Handle exceptions
        print(f'download_and_predict.py raised an exception: {e}')

        time.sleep(3)

print('download_and_predict.py ran successfully.')

