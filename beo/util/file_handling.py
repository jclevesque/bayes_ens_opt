# MIT License

# Copyright (c) 2017 Julien-Charles Lévesque

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

'''
Utility functions for testing script generation/backupping.
'''

import bz2
import shutil
import pickle
import uuid


def load_pickle(pickle_file):
    ft = file_type(pickle_file)
    if ft == "bz2":
        f = bz2.open(pickle_file, 'rb')
    elif ft == "text":
        f = open(pickle_file, 'rb')
    else:
        raise Exception("Unhandled filetype: %s" % (ft))

    content = []
    while True:
        try:
            res = pickle.load(f)
            content.append(res)
        except EOFError as exc:
            #if we loaded at least one thing it's fine.
            if len(content) < 1:
                raise
            break

    f.close()

    if len(content) == 1:
        content = content[0]

    return content


def safe_save_pickle(pickle_content, pickle_file, bz2_file=True):
    # This function is meant to reduce the risk of program being killed 
    # (e.g. by supercomputer) while writing a file the disk
    # It does not handle race conditions so you need to make sure
    # this is handled on the outside
    tmpfn = pickle_file + str(uuid.uuid4())
    if bz2_file:
        f = bz2.BZ2File(tmpfn, 'wb')
    else:
        f = open(tmpfn, 'wb')
    pickle.dump(pickle_content, f)
    f.close()

    shutil.move(tmpfn, pickle_file)


def file_type(filename):
    magic_dict = {
        "\x1f\x8b\x08": "gz",
        "\x42\x5a\x68": "bz2",
        "\x50\x4b\x03\x04": "zip"
    }
    max_len = max(len(x) for x in magic_dict)
    try:
        with open(filename, 'rb') as f:
            file_start = f.read(max_len).decode()
        for magic, filetype in magic_dict.items():
            if file_start.startswith(magic):
                return filetype
    except:
        # if we can't decode assume it is text
        pass

    return "text"
