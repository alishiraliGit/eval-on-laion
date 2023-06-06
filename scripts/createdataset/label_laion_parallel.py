import sys
import os
import subprocess
import shlex

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))

import configs

if __name__ == '__main__':
    n_process = 3

    laion_parts = range(configs.LAIONConfig.NUM_PARTS)

    logs = []
    for laion_part in laion_parts:
        command = \
            'python scripts/createdataset/label_laion.py' + '\n' + \
            '--laion_part %d' % laion_part + '\n' + \
            '--n_process %d' % n_process + '\n' + \
            '--self_destruct'

        print(command)

        args = shlex.split(command)
        log = subprocess.Popen(args)
        logs.append(log)

    for log in logs:
        log.wait()
