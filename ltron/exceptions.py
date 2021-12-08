class LtronException(Exception):
    pass

class LtronDeprecatedException(LtronException):
    pass

class ThisShouldNeverHappen(LtronException):
    pass
