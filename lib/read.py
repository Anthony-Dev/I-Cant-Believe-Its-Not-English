import unicodedata
import string

all_letters = string.printable # + string.whitespace + ' ' + "\x03" # The last one is EOF
n_letters = len(all_letters)

def unicodeToAscii(s):
    ''' Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427 '''
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

def toGenerator(filename,string_length=100):
    ''' Yields a tuple of strings, one is the input string to the network, the second is the validation string '''
    with open(filename,"r") as file:
        contents = unicodeToAscii(file.read())
        for i in range(len(contents)-1):
            input_string = contents[i: i + string_length]
            validation_string = contents[i+1: i+1+string_length]
            yield input_string,validation_string

def toString(filename):
    ''' Return a string from filename, whose text is converted to ASCII. '''
    with open(filename, "r") as file:
        contents = unicodeToAscii(file.read())
        return contents

def toArray(filename,mode='sequence',sentenceDelimiter='@'):
    '''Split contents of filename into an array of phrases split by the string sentenceDelimiter. mode="sequence" splits the file into batches of 100 chars. mode="sentence" splits the file into batches after every sentenceDelimiter.'''

    with open(filename, "r") as file:
        if mode == 'sentence':
            contents = file.read().strip('s').split(sentenceDelimiter)
        elif mode == 'sequence':

            BATCH_SIZE = 200

            read_string = file.read().strip('s')
            contents = [read_string[i:i + BATCH_SIZE] for i in range(0,len(read_string),BATCH_SIZE)]
        contents = [unicodeToAscii(sentence) for sentence in contents if sentence]
    return contents
