import math

from splendor.obj_mesh import load_mesh

raise LtronDeprecatedError(
    'Was an experiment trying to implement this paper:'
    'https://hal.inria.fr/inria-00379201/document, '
    'but I think this is overkill at the moment, may come back to it someday.')

mesh = load_mesh(
    '/home/awalsman/.cache/splendor/ltron_assets_low/meshes/3003.obj')

def M(direction, p):
    result = 0
    for l in range(p+1):
        for m in range(-2 * l, 2 * l + 1):
            result += c2p2lm(p,l,m) * ym2l(direction)

def C(p,l,m):
    surface_integral = 0
    for surface_point in surface_points:
        smag = sum(s**2 for s in surface_point)**p

def S(l,p):
    a = ((4*l+1) * math.pi)**0.5 / (2**(2*l))
    b = 0
    for k in range(l, 2*l+1):
        sign = (-1)**k
        n1 = 2**(2*p+1)
        n2 = p
        n3 = 2*k
        n4 = p+k-l
        
        d1 = 2*(p+k-l)+1
        d2 = k-l
        d3 = k
        d4 = 2*l-k
        
        element = sign
        element *= n1
        element *= factorial_fraction(n3,d3)
        element *= factorial_fraction(n4,d1)
        element *= factorial_fraction(n2,d2)
        element /= math.factorial(d4)
        
        b += element
    
    return a * b

def N(l, m, mp):
    sign = (-1)**(m-mp)
    a = (factorial_fraction(l+m,l-m)
    b = math.factorial(l+mp)
    c = math.factorial(l-mp)
    return sign * (a / (b*c))**0.5

def factorial_fraction(a,b):
    if a > b:
        return product(range(b+1, a+1))
    else:
        return 1. / product(range(a+1, b+1))

def product(elements):
    result = 1
    for element in elements:
        result *= element
    
    return result
