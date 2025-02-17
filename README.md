# RMNet
RMNet: a RNA m6A cross-species methylation detection method for nanopore sequencing

N6-methyladenosine (m6A) is the most common and abundant RNA modification in eukaryotic cells. The m6A modification participates in almost every stage of the RNA life cycle. Traditional wet-lab methods for detecting RNA m6A are time-consuming, labor-intensive, and prone to human error. The statistical approaches based methods for m6A detection require generating control samples. Previous proposed machine learning based methods exist limitation on cross-species methylation detection. Here, we introduce RMNet, a RNA m6A detection method based on Conformer and RNN, enhanced by contrastive learning. The study results indicate RMNet has robust cross-species detection ability.

![RMNet_f1](https://github.com/user-attachments/assets/6d7db916-f1fe-41e1-a219-24eb11e47dbb)

# How to install
Before running RMNet, please install GCRTcall
```
git clone https://github.com/liqingwen98/GCRTcall.git
cd GCRTcall
pip install -e .
```
Then
```
git clone https://github.com/liqingwen98/RMNet.git
cd RMNet
pip install .
```
# How to use
```
RMNet --input_file path_to_input_file --save_dir path_to_output --ckpt path_to_checkpoint
```

Because of the github store limitation, the check point file can not be added to my repository. 

If you need check point file or have any question, welcome to contact me: li.qing.wen@foxmail.com
