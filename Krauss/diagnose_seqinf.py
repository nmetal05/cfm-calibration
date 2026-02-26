
from seqinf.flow import BayesFlow
import inspect

# Show _log_prob source around the error
src = inspect.getsource(BayesFlow._log_prob)
lines = src.split('\n')
for i, line in enumerate(lines):
    if '_context' in line or i < 5:
        print(f'  {i}: {line}')

print()

# Show __init__ to see what attributes are set
try:
    src2 = inspect.getsource(BayesFlow.__init__)
    print('BayesFlow.__init__:')
    for line in src2.split('\n')[:30]:
        print(f'  {line}')
except:
    pass

print()

# Check what attributes exist
print('BayesFlow class attributes with \"context\":')
for attr in dir(BayesFlow):
    if 'context' in attr.lower() or 'base' in attr.lower():
        print(f'  {attr}')