import numpy

from ltron.name_span import NameSpan

ns = NameSpan()

ns.add_names(help=(2,3,4))
ns.add_names(me=NameSpan(jon=(2,2,3), keto=(4,5,6)))

print(ns.unravel(124))
print(ns.ravel('help', 1, 2, 3))
print(ns.ravel('me', 'jon', 0,0,0))
print(ns.ravel('me', 'jon', 0,0,1))
print(ns.ravel('me', 'jon', 0,0,2))
print(ns.ravel('me', 'jon', 0,1,0))
print(ns.ravel('me', 'jon', 0,1,1))

print(ns.ravel('me', 'jon', 1,1,2))
print(ns.ravel('me', 'keto', 0,0,0))
print(ns.ravel('me', 'keto', 0,0,1))

r = list(range(ns.total))

print(ns.unravel_all(r))
