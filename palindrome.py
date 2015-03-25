def isPalindrome(word):
    return matchPalindrome(word, 0, len(word) - 1)
def matchPalindrome(word, start, end):
    if start >= end:
        return True
    else:
        return (word[start] == word[end] and matchPalindrome(word, start + 1, end - 1))
    
isPalindrome('noon')
