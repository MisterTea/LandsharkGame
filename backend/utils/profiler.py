import cProfile
import io
import pstats


class Profiler:
    def __init__(self, doProfile: bool):
        self.doProfile = doProfile

    def __enter__(self):
        if not self.doProfile:
            return
        self.pr = cProfile.Profile()
        self.pr.enable()

    def __exit__(self, type, value, traceback):
        if not self.doProfile:
            return False
        self.pr.disable()
        s = io.StringIO()
        sortby = pstats.SortKey.TIME
        ps = pstats.Stats(self.pr, stream=s).sort_stats(sortby)
        ps.print_stats(100)
        print(s.getvalue())
        return False
