#  LAVD: Learning with Annotation of Various Degrees

Source codes for **Joey Tianyi Zhou, Meng Fang, Hao Zhang, Chen Gong, Xi Peng, Zhiguo Cao, Rick Siow Mong Goh, "Learning 
with Annotation of Various Degrees", IEEE Transactions on Neural Network and Learning Systems (TNNLS).** 

If you feel this project helpful to your research, please cite the following paper
```bibtex
@article{zhou2019LAVD,
  author    = {Joey Tianyi Zhou and Meng Fang and Hao Zhang and Chen Gong and Xi Peng and Zhiguo Cao and Rick Siow Mong Goh},
  title     = {Learning with Annotation of Various Degrees},
  journal   = {{IEEE} Trans. Neural Netw. Learning Syst.},
  doi       = {10.1109/TNNLS.2018.2885854}
}
```
or
```bibtex
@article{zhou2019learning, 
  author = {J. T. {Zhou} and M. {Fang} and H. {Zhang} and C. {Gong} and X. {Peng} and Z. {Cao} and R. S. M. {Goh}},
  title = {Learning With Annotation of Various Degrees},
  journal = {IEEE Transactions on Neural Networks and Learning Systems},
  year = {2019},
  doi = {10.1109/TNNLS.2018.2885854},
  ISSN = {2162-237X}
}
```

This project includes all the variants of Deep CRF (i.e., Deep CRF with in/complete annotation, Deep Auto-encoder for Sequence Labeling task such as NER, POS Tagging) implementations reported in the paper. 

## Requirement
* numpy 1.10+
* matplotlib 1.8+
* tensorflow 1.6+
* python 3.6
* scikit-learn 1.16+
 
## Usage
To train a DeepCRF model, run:
```bash
$ python3 train_dcrf.py --task conll2003_ner \  # indicate the dataset for training
                        --use_gpu true \  # use GPU
                        --gpu_idx 0 \  # specify which GPU is utilized
                        --partial_rate 0.5 \  # drop rate for partially labeled data
                        --raw_path <dataset_path> \  # dataset path
                        --wordvec_path <pre-trained word vectors path> \  # pretrained word vector path
                        --save_path  <save path> \  # save path
                        --epochs 30 \  # training epochs
                        ...
```
For more details, please ref. `train_dcrf.py`.
