# Official Code of A Serial Perspective on Photometric Stereo of Filtering and Serializing Spatial Information

The codebase for both MAV and DNS-S+B has been fully organized and is now stored in the MAV directory. The network's code is coming soon and will not take much time.

## MAV and DNS dataset
### MAV
The fuction of calculating MAV is in ./mav/mav.py, the params are the following:
```python
def cal_mav(img, msk, mode=3, method='mean', threshold=None):
    """
    :param img: Target Surface Normal, should be numpy
    :param msk: Mask, you can generate it by using generate_mask()
    :param mode: MAV's windowsize, String '4' means 4-neighbor, Int 4 means 4*4 window.
    :param method: How to calculate MAV in a window, 'mean' means average, 'min' means minimum, 'max' means maximum.
    :param threshold: If you want to calculate the percentage of MAV greater than a certain value, set the threshold.
    :return: Result and MAV map
    """
```
Feel free to modify or use it. Please note that if you do not have a mask, you can generate one using the generate_mask() function in ./mav/organize.py. However, be aware of the background pixel values in the normal maps: for PS_Sculpture and PS_Blooby, the background value is (127, 127, 127), while for the DiLiGenT series, it is (0, 0, 0).
### DNS dataset
To organize the DNS dataset, you first need to preprocess the data into the following format. Alternatively, you can directly download the PS_Blooby and PS_Sculpture datasets that we use, as these datasets require no additional preprocessing.

```bash
PS_Blooby
├── Images
│   ├── blob10_s-0.90_x-350_y-290
│   │   ├── blue-metallic-paint2
│   │   │   ├── blob10_s-0.90_x-350_y-290_blue-metallic-paint2.txt
│   │   │   ├── l_000,0.62,-0.34,0.71.png
│   │   │   ├── l_001,-0.89,0.28,0.35.png
│   │   │   └── ...
│   │   ├── gray-plastic
│   │   │   ├── blob10_s-0.90_x-350_y-290_gray-plastic.txt
│   │   │   ├── l_000,0.62,-0.34,0.71.png
│   │   │   ├── l_001,-0.89,0.28,0.35.png
│   │   │   └── ...
│   │   └── blob10_s-0.90_x-350_y-290_normal.png
│   ├── blob10_s-0.90_x-350_y-290
│   │   └── ...
│   └── ...
├── mtrl.txt
├── train_mtrl.txt
└── val_mtrl.txt
```

Next, you can run ./mav/organize.py to generate the DNS dataset. Please note that the default generation method creates soft links for all selected datasets in the output directory. You can modify the code to tailor it to your specific requirements.
```bash
python ./mav/organize.py
```

If you find our work helps, please cite:
```bibtex
@ARTICLE{10907979,
  author={Xu, Minzhe and Ding, Xin and Yang, You and Zheng, Yinqiang and Liu, Qiong},
  journal={IEEE Transactions on Visualization and Computer Graphics}, 
  title={A Serial Perspective on Photometric Stereo of Filtering and Serializing Spatial Information}, 
  year={2025},
  volume={},
  number={},
  pages={1-15},
  keywords={Feature extraction;Accuracy;Training;Lighting;Surface reconstruction;Measurement;Image reconstruction;Complexity theory;Calibration;Learning systems;3D reconstruction;deep neural networks;photometric stereo;sequence models;surface normal estimate},
  doi={10.1109/TVCG.2025.3546657}}
```
