def same_cheon(word1, word2, max_josa_length=3):
    if word1 == word2:
        return True
    for i in range(max_josa_length):
        if word1[:-i] == word2:
            return True
    for i in range(max_josa_length):
        if word2[:-i] == word1:
            return True

    return False