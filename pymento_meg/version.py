import sys
from os.path import lexists, dirname, join as opj, curdir

__version__ = "0.0.12"
__hardcoded_version__ = __version__
__full_version__ = __version__


moddir = dirname(__file__)
projdir = curdir if moddir == "datalad" else dirname(moddir)
if lexists(opj(projdir, ".git")):
    # If under git -- attempt to deduce a better "dynamic" version following git
    try:
        from subprocess import Popen, PIPE

        # Note: Popen does not support `with` way correctly in 2.7
        #
        git = Popen(
            ["git", "describe", "--abbrev=4", "--dirty", "--match", r"[0-9]*\.*"],
            stdout=PIPE,
            stderr=PIPE,
            cwd=projdir,
        )
        if git.wait() != 0:
            raise OSError("Could not run git describe")
        line = git.stdout.readlines()[0]
        _ = git.stderr.readlines()
        # Just take describe and replace initial '-' with .dev to be more "pythonish"
        # Encoding simply because distutils' LooseVersion compares only StringType
        # and thus misses in __cmp__ necessary wrapping for unicode strings
        __full_version__ = line.strip().decode("ascii").replace("-", ".dev", 1)
        # To follow PEP440 we can't have all the git fanciness
        __version__ = __full_version__.split("-")[0]
    except (SyntaxError, AttributeError, IndexError):
        raise
    except:
        # just stick to the hard-coded
        pass
