def isSorted(s, i = 0):
    if len(s[i:]) < 2:
        return True
    elif s[i] <= s[i+1]:
        return isSorted(s[(i+1):])
    return False
