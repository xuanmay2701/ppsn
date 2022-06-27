# Code
Perceptual Position-aware Shapelet Network (Accepted to ECML PKDD 2022)

![alt text](https://github.com/xuanmay2701/ppsn/tree/main/img/ppsn.png?raw=true)

## Dependencies
- pytorch 1.11.0 and above

## Usage
We provide the demo of ECGFiveDays in the UCR dataset.  The ECGFiveDays dataset is located in `dataset/UCRArchive_2018/ECGFiveDays/`. You can run the command
```
python ppsn_demo.py
```
to test the model.

# Classification Result
You can see the full results on 112 UCR datasets in `results/`, in that `results/ppsn_vs_sbc.csv` contains the results of PPSN and other Shapelet-based Classifiers, while `results/ppsn_vs_sota.csv` contains the results of SOTA methods.

![alt text]([https://github.com/xuanmay2701/ppsn/tree/main/img)/vs_sota.png?raw=true)
Critical different diagram shows the average ranks of PPSN and 7 SOTA methods on 109 UCR datasets.

![alt text](https://github.com/xuanmay2701/ppsn/tree/main/img/vs_sbc.png?raw=true)
Critical different diagram shows the average ranks of PPSN and 6 shapelet-based methods on 85 UCR Datasets.
