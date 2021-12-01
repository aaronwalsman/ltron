pi_theta = [0.1, 0.9]
q = [1, 5]

alpha = min(q) + 1

while True:
    pi_bar = []
    for pt, qq in zip(pi_theta, q):
        pi_bar.append(pt / (alpha - qq))
    
    sum_pi_bar = sum(pi_bar)
    
    if sum_pi_bar < 1.0:
        alpha -= 0.00001
    else:
        alpha += 0.00001
    
    print('Alpha: %f (%f)'%(alpha, sum_pi_bar))

'''
pi0 / (a-q0) = b0
pi1 / (a-q1) = b1
pi2 / (a-q2) = b2
...
b0 + b1 + b2 + ... = 1

pi0 / (a-q0) + pi1 / (a-q1) = 1
'''

