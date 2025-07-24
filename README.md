# MFAASR

## Requirements
- Python 3.6 (Anaconda is recommended)
- skimage
- imageio
- Pytorch (Pytorch version >=1.2.0 is recommended)
- tqdm 
- pandas
- cv2 (pip install opencv-python)



## Test

#### Quick start
1. Download the testset
The testset can be downloaded from [[BaiduYun]](https://pan.baidu.com/s/18NsZHMbhSu14GxAw9jMgIw)(code:hl0v) and unzip it to ./results

2. cd to `EMASRN` and run **one of following commands** for evaluation:

   ```shell
   # EMASRN
   python test.py -opt options/test/test_example_x3.json
   python test.py -opt options/test/test_example_x4.json
   
3. Edit `./options/test/test_example_x3.json` or `./options/test/test_example_x4.json` for your needs

## Train

1. Download training set DIV2K [[Official Link]](https://data.vision.ee.ethz.ch/cvl/DIV2K/) or DF2K [[GoogleDrive]](https://drive.google.com/drive/folders/1B-uaxvV9qeuQ-t7MFiN1oEdA6dKnj2vW?usp=sharing) [[BaiduYun]](https://pan.baidu.com/s/1CFIML6KfQVYGZSNFrhMXmA#list/path=%2F) (provided by [BasicSR](https://github.com/xinntao/BasicSR)).

2. Run `./scripts/Prepare_TrainData_HR_LR.m` in Matlab to generate HR/LR training pairs with corresponding degradation model and scale factor. (**Note**: Please place generated training data to **SSD (Solid-State Drive)** for fast training)

3. Run `./results/Prepare_TestData_HR_LR.m` in Matlab to generate HR/LR test images with corresponding degradation model and scale factor, and choose one of SR benchmark for evaluation during training.

4. Edit `./options/train/train_SRFBN_example.json` for your needs according to [`./options/train/README.md`.](./options/train/README.md)

5. Then, run command:
   ```shell
   cd SRFBN_CVPR19
   python train.py -opt options/train/train_SRFBN_example.json
   ```

6. You can monitor the training process in `./experiments`.

7. Finally, you can follow the **test pipeline** to evaluate your model.

4. ## Acknowledgements

- Thank [Paper99](https://github.com/Paper99/SRFBN_CVPR19), Our code structure is derived from his repository 
