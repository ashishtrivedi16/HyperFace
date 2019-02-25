# HyperFace
Contains the implementation of [HyperFace: A deep multi task learning framework for facial recognition, landmark detection, pose and gender detection](https://arxiv.org/pdf/1603.01249.pdf)

TODO :
- Implement HyperFace-AlexNet
    - ~~Selective search~~
    - Iterative region proposal (IRP)
    - Landmark based - Non Maximum Suppression (L-NMS)
    - skip training RCNN on imagenet and directly train on region proposals from AFLW dataset
    - Copy weights to the hyperface model and retrain it on AFLW dataset

- Implement HyperFace-ResNet101
    - ~~Uses keras.application to load model weights and retrain R-CNN~~
    - Re-write the model architecture for resnet and carry out training

- Experimental stuff
    - ~~Find better loss functions/ Implement custom loss functions~~
    - ~~Find better optimizer
    - ~~Decide on using ReduceLROnPlateau or not, look for useful callbacks~~

- Future Plans
    - Implement using ResNet
    - ~~Divide the code into seperate files
    - support command line arguments

- Known issues
    - ~~openCV's imread function reads in BGR instead of RGB, skimage ioread is comparatively slower than openCV, matplotlib shows wrong image plots because of BGR mode~~
    - Some images are bw by default so proper dimension conversion can not be done at all images (at the moment those images are skipped to save me from headache)
    - ~~Processing all images uses up 14gb+ of RAM, so try and implement some other way because I plan on implementing data augementation in future to increase face detection accuracy~~


#### Current model archietecture (AlexNet) is shown below -
![HyperFace AlexNet](https://github.com/ashishtrivedi16/HyperFace/blob/master/src/model_plots/model_HyperFace_Alexnet.png)
