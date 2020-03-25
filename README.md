# EfficientDet: Scalable and Efficient Object Detection, in PyTorch
A [PyTorch](http://pytorch.org/) implementation of [EfficientDet](https://arxiv.org/abs/1911.09070) from the 2019 paper by Mingxing Tan Ruoming Pang Quoc V. Le
Google Research, Brain Team.  The official and original: comming soon.

### prerequisites

* Python 3.6+
* PyTorch 1.3+
* Torchvision 0.4.0+ (**We need high version because Torchvision support nms now.**)
* requirements.txt 


##### Download VOC2007 + VOC2012 trainval & test
```Shell
# specify a directory for dataset to be downloaded into, else default is ~/data/
sh datasets/scripts/VOC2007.sh
sh datasets/scripts/VOC2012.sh
```

## Training EfficientDet

- To train EfficientDet using the train script simply specify the parameters listed in `train.py` as a flag or manually change them.

```Shell
python train.py --network effcientdet-d0  # Example
```

  - With VOC Dataset:
  ```Shell
  # DataParallel
  python train.py --dataset VOC --dataset_root /root/data/VOCdevkit/ --network effcientdet-d0 --batch_size 32 
  # DistributedDataParallel with backend nccl
  python train.py --dataset VOC --dataset_root /root/data/VOCdevkit/ --network effcientdet-d0 --batch_size 32 --multiprocessing-distributed
  ```
  - With COCO Dataset:
  ```Shell
  # DataParallel
  python train.py --dataset COCO --dataset_root ~/data/coco/ --network effcientdet-d0 --batch_size 32
  # DistributedDataParallel with backend nccl
  python train.py --dataset COCO --dataset_root ~/data/coco/ --network effcientdet-d0 --batch_size 32 --multiprocessing-distributed
  ```

## Evaluation
To evaluate a trained network:
 - With VOC Dataset:
    ```Shell
    python eval_voc.py --dataset_root ~/data/VOCdevkit --weight ./checkpoint_VOC_efficientdet-d0_261.pth
    ```
- With COCO Dataset
comming soon.
## Demo

```Shell
python demo.py --threshold 0.5 --iou_threshold 0.5 --score --weight checkpoint_VOC_efficientdet-d1_34.pth --file_name demo.png
```

Output: 

<p align="center">
<img src= "./docs/demo.png">
</p>


## TODO
We have accumulated the following to-do list, which we hope to complete in the near future
- Still to come:
  * [x] EfficientDet-[D0-7]
  * [x] GPU-Parallel
  * [x] NMS
  * [ ] Soft-NMS
  * [x] Pretrained model
  * [x] Demo
  * [ ] Model zoo
  * [ ] TorchScript
  * [ ] Mobile
  * [ ] C++ Onnx
  

## Authors

* [**Toan Dao Minh**](https://github.com/toandaominh1997)

***Note:*** Unfortunately, this is just a hobby of ours and not a full-time job, so we'll do our best to keep things up to date, but no guarantees.  That being said, thanks to everyone for your continued help and feedback as it is really appreciated. We will try to address everything as soon as possible.

## References
- tanmingxing, rpang, qvl, et al. "EfficientDet: Scalable and Efficient Object Detection." [EfficientDet](https://arxiv.org/abs/1911.09070).
- A list of other great EfficientDet ports that were sources of inspiration:
  * [EfficientNet](https://github.com/lukemelas/EfficientNet-PyTorch)
  * [SSD.Pytorch](https://github.com/amdegroot/ssd.pytorch)
  * [mmdetection](https://github.com/open-mmlab/mmdetection)
  * [RetinaNet.Pytorch](https://github.com/yhenon/pytorch-retinanet)
  * [NMS.Torchvision](https://pytorch.org/docs/stable/torchvision/ops.html)
  

## Citation

    @article{efficientdetpytoan,
        Author = {Toan Dao Minh},
        Title = {A Pytorch Implementation of EfficientDet Object Detection},
        Journal = {github.com/toandaominh1997/EfficientDet.Pytorch},
        Year = {2019}
    }
