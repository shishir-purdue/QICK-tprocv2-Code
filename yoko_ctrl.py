from malab.instruments.voltsource import *
from malab.dsfit import fithanger_new_withQc
# YOKO1 IP = "xxx.xxx.xxx.xxx"
# YOKO2 IP = "xxx.xxx.xxx.xxx"
dcflux2 = YokogawaGS200(address="xxx.xxx.xxx.xxx")
dcflux2.recv_length = 1024
print(dcflux2.recv_length)
print(dcflux2.get_id())
print(dcflux2.get_id())
dcflux2.set_mode('current')
dcflux2.set_range(0.010)  # 100mA range
dcflux2.ramp_current(current=0.00, sweeprate=-0.001) # second argument is rate? time.sleep(0.2)
dcflux2.set_output(True)