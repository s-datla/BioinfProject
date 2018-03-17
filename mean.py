from Bio.SeqUtils.ProtParamData import kd, Flex
import numpy as np

kd_s = []
flex_s = []
for key,values in kd.iteritems():
    kd_s += [values]
for key,values in Flex.iteritems():
    flex_s += [values]
kd_mean = np.mean(kd_s)
flex_mean = np.mean(flex_s)

print(kd_mean)
print(flex_mean)
