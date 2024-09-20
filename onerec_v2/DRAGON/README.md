# DRAGON

Pytorch implementation for "Enhancing Dyadic Relations with Homogeneous Graphs for Multimodal Recommendation" -ECAI'23 [arxiv](https://arxiv.org/pdf/2301.12097.pdf)

## Data
Data could be download from DropBox: [Baby/Sports/Clothing](https://www.dropbox.com/sh/yti9m3pprzprukv/AAA9LhKKUDZiPUp3kVv1hZALa?dl=0)  
## Run the code
1. Download the data from the data link we provided above, then put the download data to the ./data folder
2. Use ```conda env create -f dragon.yml``` to create the enviroment with correct dependencies
2. Run generate-u-u-matrix.py on the dataset you want to generate the user co-occurrence graph
3. Enter the src folder and run with
`python main.py -m DRAGON -d dataset_name`  
## The parameters to reproduce the result in our paper
| Datasets | learning rate | reg weight |
|----------|--------|---------|
| Baby     | 0.0001      | 0.001     |
| Sports   | 0.0001      | 0.001     |
| Clothing     | 0.0001      | 0.1     |

#### Please consider to cite our paper if this model helps you, thanks:
```
@article{zhou2023enhancing,
  title={Enhancing Dyadic Relations with Homogeneous Graphs for Multimodal Recommendation},
  author={Zhou, Hongyu and Zhou, Xin and Shen, Zhiqi},
  journal={arXiv preprint arXiv:2301.12097},
  year={2023}
}
```
