# Copyright (C) 2015 Julien-Charles Levesque
# Based on pylockfile by openstack: https://github.com/openstack/pylockfile
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import os
import time
import socket
import functools


class LockError(Exception):
    #Base class for error arising from attempts to acquire the lock.
    pass

class LockTimeout(LockError):
    #Raised when lock creation fails within a user-defined period of time.
    pass

class AlreadyLocked(LockError):
    #Some other thread/process is locking the file.
    pass

class LockFailed(LockError):
    #Lock file creation failed for some other reason.
    pass

class UnlockError(Exception):
    #Base class for errors arising from attempts to release the lock.
    pass

class NotLocked(UnlockError):
    #Raised when an attempt is made to unlock an unlocked file.
    pass

class NotMyLock(UnlockError):
    #Raised when an attempt is made to unlock a file someone else locked.
    pass

def locked(path, timeout=None):
    """Decorator which enables locks for decorated function.
    Arguments:
     - path: path for lockfile.
     - timeout (optional): Timeout for acquiring lock.
     Usage:
         @locked('/var/run/myname', timeout=0)
         def myname(...):
             ...
    """
    def decor(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            lock = Locker(path, timeout=timeout)
            lock.acquire()
            try:
                return func(*args, **kwargs)
            finally:
                lock.release()
        return wrapper
    return decor

class Locker:
    """Lock access to a file using atomic property of link(2).
    >>> lock = LinkLockFile('somefile')
    """
    def __enter__(self):
        """
        Context manager support.
        """
        self.acquire()
        return self

    def __exit__(self, *_exc):
        """
        Context manager support.
        """
        self.release()

    def __repr__(self):
        return "<%s: %r>" % (self.__class__.__name__, self.path)

    def __init__(self, path, timeout=None):
        """
        >>> lock = LockBase('somefile')
        """
        self.path = path
        self.lock_file = os.path.abspath(path) + ".lock"
        self.hostname = socket.gethostname()
        self.pid = os.getpid()
        dirname = os.path.dirname(self.lock_file)

        # unique name is mostly about the current process, but must
        # also contain the path -- otherwise, two adjacent locked
        # files conflict (one file gets locked, creating lock-file and
        # unique file, the other one gets locked, creating lock-file
        # and overwriting the already existing lock-file, then one
        # gets unlocked, deleting both lock-file and unique file,
        # finally the last lock errors out upon releasing.
        self.unique_name = os.path.join(dirname,
                                        "%s.%s%s" % (self.hostname,
                                                       self.pid,
                                                       hash(self.path)))
        self.timeout = timeout

    def acquire(self, timeout=None):
        #print("Getting lock %s" % self.lock_file)
        try:
            open(self.unique_name, "wb").close()
        except IOError:
            raise LockFailed("failed to create %s" % self.unique_name)

        timeout = timeout is not None and timeout or self.timeout
        end_time = time.time()
        if timeout is not None and timeout > 0:
            end_time += timeout

        while True:
            # Try and create a hard link to it.
            try:
                os.link(self.unique_name, self.lock_file)
            except OSError:
                # Link creation failed.  Maybe we've double-locked?
                nlinks = os.stat(self.unique_name).st_nlink
                if nlinks == 2:
                    # The original link plus the one I created == 2.  We're
                    # good to go.
                    return
                else:
                    # Otherwise the lock creation failed.
                    if timeout is not None and time.time() > end_time:
                        os.unlink(self.unique_name)
                        if timeout > 0:
                            raise LockTimeout("Timeout waiting to acquire"
                                              " lock for %s" %
                                              self.path)
                        else:
                            raise AlreadyLocked("%s is already locked" %
                                                self.path)
                    time.sleep(timeout is not None and timeout/10 or 0.1)
            else:
                # Link creation succeeded.  We're good to go.
                return

    def release(self):
        #print("Releasing lock %s" % self.lock_file)
        if not self.is_locked():
            #raise NotLocked("%s is not locked" % self.path)
            print("Warning: %s i not locked." % self.path)
            return
        elif not os.path.exists(self.unique_name):
            raise NotMyLock("%s is locked, but not by me" % self.path)
        os.unlink(self.unique_name)
        os.unlink(self.lock_file)
        #print("Released lock %s" % self.lock_file)

    def is_locked(self):
        return os.path.exists(self.lock_file)

    def i_am_locking(self):
        return (self.is_locked() and
                os.path.exists(self.unique_name) and
                os.stat(self.unique_name).st_nlink == 2)

    def break_lock(self):
        if os.path.exists(self.lock_file):
            os.unlink(self.lock_file)

    def __del__(self):
        if self.i_am_locking():
            self.release()

