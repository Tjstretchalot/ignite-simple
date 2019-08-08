"""This can be used to test when it is ok to use __name__ even when it might
be __main__. This will find the path that can be used to load the __main__
file in other processes with possibly different __main__'s"""

import sys
from ignite_simple.utils import fix_imports

print(sys.argv[0])

print(f'__name__ = {__name__}')

imp = (__name__, 'joe', tuple(), dict())
imp = fix_imports(imp)

print(f'fixed: {imp[0]}')
