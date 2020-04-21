import sys
import subprocess
import pkg_resources

required = {line.rstrip().split('=')[0] for line in open('requirements.txt')}
installed = {pkg.key for pkg in pkg_resources.working_set}
missing = required - installed
if 'pillow' in required:
    missing = missing - {'pillow'}
missing.add('pillow==6')


if missing:
    python = sys.executable
    subprocess.check_call([python, '-m', 'pip', 'install', *missing], stdout=subprocess.DEVNULL)
