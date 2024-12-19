
from dataclasses import dataclass, field
import numpy as np 
import h5py 

@dataclass
class Model:
    """Theoretical CO power spectrum model"""

    path: str = field(default_factory=str)
    
    def read_model(self):
        """Method reading in theoretical model from file and
        loads them into class attributes,
        """
        with np.load(self.path) as infile:
            for key, value in infile.items():
                setattr(self, key, value)
    
    def interpolate_model(self):
        """Method generating interpolation of theoretical model spectrum
        """
        import scipy.interpolate as interpolate
        power_spectrum_noatt = np.trapezoid(self.Pkmu_noatt, self.mu)
        
        self.interpolation = interpolate.CubicSpline(
            self.k, 
            power_spectrum_noatt,
        ) 
