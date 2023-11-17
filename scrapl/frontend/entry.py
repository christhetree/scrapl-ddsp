from kymatio.frontend.entry import ScatteringEntry

class TimeFrequencyScraplEntry(ScatteringEntry):
    def __init__(self, *args, **kwargs):
        super().__init__(
            name='ScRaPL', class_name='scattering1d', *args, **kwargs)
        
__all__ = ['TimeFrequencyScraplEntry']