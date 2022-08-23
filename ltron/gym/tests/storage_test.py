import numpy

from ltron.gym.rollout_storage import RolloutStorage

s = RolloutStorage(4)
s.start_new_seqs([True, True, True, True])

s.append_batch(blah = numpy.array([1,2,3,4]))
s.start_new_seqs([False, False, False, False])

s.append_batch(blah = numpy.array([2,4,6,8]))
s.start_new_seqs([False, False, True, True])

s.append_batch(blah = numpy.array([3,6,9,12]))
s.start_new_seqs([False, False, False, False])

import pdb
pdb.set_trace()
