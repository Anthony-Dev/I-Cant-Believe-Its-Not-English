import sys
import math
import string
import time
import unicodedata

def timeSince(since):
    ''' Return time since current moment and argument '''
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def printProgress(i, i_max=False):
    ''' Prints a progress bar '''
    if (not i_max):
        j = int(i / 100)
        j = j % 10
        printOW('Loading: ' + '.' * (j-1) + 'o' + '.' * (10-j))
    else:
        progress =  i/float(i_max)
        progress = int(progress * 100)
        j = int(i / 100)
        j = j % 10
        phrase = 'Loading: ' + str(progress) + '% '
        printOW(phrase + '[' + ' ' * (j) + '=' + ' ' * (9-j) + ']')

def printOW(string):
    ''' Print to stdout, overwriting the current line. '''
    sys.stdout.write('\r' + ' ' * 35)
    sys.stdout.write('\r' + str(string))
    sys.stdout.flush()

all_letters = string.printable + string.whitespace + ' ' + "\x03" # The last one is EOF
n_letters = len(all_letters)

def unicodeToAscii(s):
    ''' Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427 '''
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

def readLines(filename,sentenceDelimiter='@'):
    '''Split contents of filename into an array of phrases split by the string sentenceDelimiter.'''
    with open(filename, "r") as file:
        contents = file.read().strip('s').split(sentenceDelimiter)
        contents = [unicodeToAscii(sentence) for sentence in contents if sentence]
    return contents

def readBook(filename):
    ''' Return a string from filename, whose text is converted to ASCII. '''
    with open(filename, "r") as file:
        contents = unicodeToAscii(file.read())
        return contents
