# SRN: Stacked Regression Network for Real-time 3D Hand Pose Estimation[paper](https://bmvc2019.org/wp-content/uploads/papers/0918-paper.pdf)

# Realtime demo
## normal hand  
![demo1](realtime/gif/normal.gif)
## small hand
![demo2](realtime/gif/small_hand.gif)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;

<sup>\* Demos above are realtime results from Kinect V2 using models trained on [Hands17](http://icvl.ee.ic.ac.uk/hands17/challenge/) dataset (Intel Realsense SR300). </sup></br>
# Usage 

## Prepare pre-trained models
Download models from [here](https://drive.google.com/drive/folders/1QG6F9aD4t-LLupoguWVpBm-fUyGPNRl0?usp=sharing).

Put checkpoint/ in realtime/

## Prepare test data
We provide some data collected by Kinect V2 in realtime/data/kinect2 for testing.

(Currently only Kinect V2 and realsense R300 are supported. If you want to use the depth map obtained by other sensors, you can configure the corresponding sensor parameters in run.py)

## Testing
```bash
cd realtime
python run.py --data_dir ./data/your_folder_name --test_dataset kinect2 or realsense  
```

# Comparison with state-of-the-art methods
## NYU
![NYU](https://github.com/RenFeiTemp/SRN/blob/master/fig/nyu.png)
## ICVL
![ICVL](https://github.com/RenFeiTemp/SRN/blob/master/fig/icvl.png)
## MSRA
![MSRA](https://github.com/RenFeiTemp/SRN/blob/master/fig/msra.png)
## HANDS17
![HANDS17](https://github.com/RenFeiTemp/SRN/blob/master/fig/hands17.jpg)
