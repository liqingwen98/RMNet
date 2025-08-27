# RMNet
RMNet: an RNA m6A cross-species methylation detection method for nanopore sequencing
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

Check point file can be found here: https://huggingface.co/liqingwen/RMNet/tree/main

# Citations
``` bibtex
@article{:/content/journals/cdt/10.2174/0113894501405283250627072052,
   author = "Li, Qingwen and Sun, Chen and Wang, Daqian and Lou, Jizhong",
   title = "RMNet: An RNA m6A Cross-species Methylation Detection Method for Nanopore Sequencing",
   journal = "Current Drug Targets",
   issn = "1389-4501",
   year = "2025",
   publisher = "Bentham Science Publishers",
   doi = "https://doi.org/10.2174/0113894501405283250627072052",
}
``` 

