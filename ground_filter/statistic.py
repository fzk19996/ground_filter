import numpy as np

for i in range(10):
    file_name = str(i+1).zfill(2)+'.npy'
    data = np.load('result/'+file_name)
    avr_ = np.average(data)
    max_ = np.max(data)
    min_ = np.min(data)
    print('min:' + str(min_) + ' max:' + str(max_) + ' avr:' + str(avr_))
