from malab.instruments.voltsource import *
from malab.instruments.instrumenttypes import *

########## instrument setup ##########

### YOKO setup
YOKO_2 = 192.168.1.73

dcflux2 = YokogawaGS200(address=YOKO_3) # dc coil
dcflux2.recv_length = 1024
print(dcflux2.recv_length)
print(dcflux2.get_id())
print(dcflux2.get_id())
dcflux2.set_mode('current')
dcflux2.set_range(0.01)  # 10mA range
dcflux2.set_output(True)

def drive_YOKO(pt):

    print("Driving YOKO at (%.3f) mA" % (pt[0]))
    dcflux2.ramp_current(pt[0]*1e-3, 0.0005) # second argument is rate?
    time.sleep(0.2)