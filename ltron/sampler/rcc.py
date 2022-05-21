AntennaSampler = RotatorSubAssemblySampler(
        '4592.dat', '4593.dat', (1,0,0), (-math.pi/2, math.pi/2))

SpinnerPlateSampler = RotatorSubAssemblySampler(
        '3680.dat', '3679.dat', (0,1,0), (0, math.pi*2))

FolderSampler = RotatorSubAssemblySampler(
        '3937.dat', '3938.dat', (1,0,0), (0, math.pi/2), pivot=(0,10,0))

WheelASampler = MultiSubAssemblySampler(
        ['30027b.dat', '30028.dat'],
        [numpy.array([
            [ 1, 0, 0, 0],
            [ 0, 1, 0, 0],
            [ 0, 0, 1,-2],
            [ 0, 0, 0, 1]]),
         numpy.array([
            [ 0, 0, 1, 0],
            [ 0, 1, 0, 0],
            [-1, 0, 0, 3],
            [ 0, 0, 0, 1]])],
        global_transform = numpy.eye(4))

WheelBSampler = MultiSubAssemblySampler(
        ['4624.dat', '3641.dat'],
        [numpy.eye(4), numpy.eye(4)],
        global_transform = numpy.eye(4))

WheelCSampler = MultiSubAssemblySampler(
        ['6014.dat', '6015.dat'],
        [numpy.eye(4), numpy.array([
            [ 1, 0, 0, 0],
            [ 0, 1, 0, 0],
            [ 0, 0, 1,-6],
            [ 0, 0, 0, 1]])],
        global_transform = numpy.eye(4))

RegularAxleWheelSampler = AxleWheelSubAssemblySampler(
        '4600.dat',
        [numpy.array([
            [0, 0,-1, 30],
            [0, 1, 0,  5],
            [1, 0, 0,  0],
            [0, 0, 0,  1]]),
         numpy.array([
            [0, 0, 1,-30],
            [0, 1, 0,  5],
            [1, 0, 0,  0],
            [0, 0, 0,  1]])],
        [WheelASampler, WheelBSampler, WheelCSampler])

WideAxleWheelSampler = AxleWheelSubAssemblySampler(
        '6157.dat',
        [numpy.array([
            [0, 0,-1, 40],
            [0, 1, 0,  5],
            [1, 0, 0,  0],
            [0, 0, 0,  1]]),
         numpy.array([
            [0, 0, 1,-40],
            [0, 1, 0,  5],
            [1, 0, 0,  0],
            [0, 0, 0,  1]])],
        [WheelASampler, WheelBSampler, WheelCSampler])

