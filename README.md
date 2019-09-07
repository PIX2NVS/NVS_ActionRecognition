# Neuromorphic Vision Sensing for CNN-based Action Recognition

## Summary
This is the implemtation code for the following paper. Please cite following paper if you use this code or dataset in your own work. 

MLA:
    
   [1] Chadha A, Bi Y, Abbas A, et al. Neuromorphic Vision Sensing for CNN-based Action Recognition[C]//ICASSP 2019-2019 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2019: 7968-7972.
    
BibTex:
    
    @inproceedings{chadha2019neuromorphic,
    title={Neuromorphic Vision Sensing for CNN-based Action Recognition},
    author={Chadha, A and Bi, Y and Abbas, A and Andreopoulos, Y},
    booktitle={ICASSP 2019-2019 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
    pages={7968--7972},
    year={2019},
    organization={IEEE}
 }


## Teacher-student Framework for Action Recognition

Contrary to recent work that considers homogeneous transfer between flow domains (optical flow to motion vectors), we propose to embed an NVS emulator into a multi-modal transfer learning framework that carries out heterogeneous transfer from optical flow to NVS. The potential of our framework is showcased by the fact that, for the first time, our NVS-based results achieve comparable action recognition performance to motion-vector or opticalflow based methods.

<img height="360" width='800' src="https://github.com/PIX2NVS/NVS_ActionRecognition/blob/master/images/Framework.JPG">

## Code Implementation
### Requirements:
     Python 2.7 
     Tensorflow 1.4 
     
    
### Running examples:
    cd code
    python main.py



## Contact 
For any questions or bug reports, please contact Yin Bi at yin.bi.16@ucl.ac.uk .
