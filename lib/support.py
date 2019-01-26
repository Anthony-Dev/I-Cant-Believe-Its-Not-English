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
