# NSSADNN_IQA
Pytorch version of IEEE Transactions on Multimedia 2019:"[B. Yan, B. Bare and W. Tan, "Naturalness-Aware Deep No-Reference Image Quality Assessment," in IEEE Transactions on Multimedia, vol. 21, no. 10, pp. 2603-2615, Oct. 2019, doi: 10.1109/TMM.2019.2904879.]"(https://ieeexplore.ieee.org/document/8666733)

# Note
*I did not use the learning rate that used in the paper: 0.01, because the ideal result cannot be obtained when the initial learning rate is 0.01, so the initial learning rate set here is 0.001 instead of 0.01.
*This training progress only support on LIVE II database now, the training progress on TID2013, CSIQ, LIVEMD, CLIVE will be released soon.

# Train
`python train.py`

# TODO
* Cross dataset test code will be published
* Train on different distortion types on LIVE, TID2013, CSIQ will be published
* Code of evaluations on Waterloo Exploration Database (D-test, L-test, P-test and gMAd competition) will be published
