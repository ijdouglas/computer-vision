import sys
import time
import datetime
print(f"Starting script at: {datetime.datetime.now()}")
time.sleep(1)
print("Slept one second")
print(f'arg length: {len(sys.argv)}')
if len(sys.argv) == 3:
    time.sleep(3)
    print({f"arg {i}": val for i, val in enumerate(sys.argv)})
else:
    raise ValueError('Script requires 2 arguments')
