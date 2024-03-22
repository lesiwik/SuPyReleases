calls: dict[str, int] = {}
callType = None


def registerCall():
    global calls
    global callType
    val = calls.get(callType, 0)
    calls[callType] = val + 1


def beginCounting(cType):
    global callType
    callType = cType


def endCounting():
    beginCounting(None)
