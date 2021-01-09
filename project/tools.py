import string


def clean_string(s):
    s = s.strip().lower()

    s = s.translate(str.maketrans('', '',string.punctuation))

    return s
