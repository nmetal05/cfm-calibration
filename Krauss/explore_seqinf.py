import inspect
from seqinf.methods.posterior import ASNPE, ASNPECollector, SNPE, BayesianSNPE

print('=== ASNPE ===')
print(f'type: {type(ASNPE)}')
print(f'MRO: {[c.__name__ for c in ASNPE.__mro__]}')
try:
    print(f'signature: {inspect.signature(ASNPE)}')
except: pass
print(f"methods: {[m for m in dir(ASNPE) if not m.startswith('_')]}")
print()

print('=== ASNPECollector ===')
print(f'type: {type(ASNPECollector)}')
print(f'MRO: {[c.__name__ for c in ASNPECollector.__mro__]}')
try:
    sig = inspect.signature(ASNPECollector.__init__)
    print(f'__init__{sig}')
except: pass
try:
    sig = inspect.signature(ASNPECollector.collect)
    print(f'collect{sig}')
except: pass
print(f"methods: {[m for m in dir(ASNPECollector) if not m.startswith('_')]}")
print()

print('=== SNPE (seqinf version) ===')
print(f'type: {type(SNPE)}')
try:
    print(f'signature: {inspect.signature(SNPE)}')
except: pass
print()

print('=== SequentialInference.collector_cls ===')
from seqinf import SequentialInference
try:
    src = inspect.getsource(SequentialInference.collector_cls.fget)
    print(src)
except Exception as e:
    print(f'Error: {e}')
    # Try accessing it differently
    try:
        src = inspect.getsource(SequentialInference)
        for line in src.split('\n'):
            if 'collector' in line.lower():
                print(line)
    except: pass

print()
print('=== How ASNPE creates collector ===')
try:
    src = inspect.getsource(ASNPE)
    for line in src.split('\n'):
        if 'collector' in line.lower() or 'active' in line.lower() or 'acqui' in line.lower():
            print(line.rstrip())
except Exception as e:
    print(f'Error: {e}')
    # Try the full source
    try:
        print(inspect.getsource(ASNPE)[:2000])
    except: pass
