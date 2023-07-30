# HELIOS (Hyper-relational Schema Model)

HELIOS is a hyper-relational schema model, which directly learns from hyper-relational schema tuples in a KG. HELIOS captures not only the correlation between multiple types of a single entity, but also the correlation between types of different entities and relations in a hyper-relational schema tuple. Please see the details in our paper below:
- Yuhuan Lu, Bangchao Deng, Weijian Yu, and Dingqi Yang. 2023. HELIOS: Hyper-Relational Schema Modeling from Knowledge Graphs. In Proceedings of the 31st ACM International Conference on Multimedia (MM ’23), October 29–November 3, 2023, Ottawa, ON, Canada.

## How to run the code
###### Train and evaluate model (suggested parameters for JF17k, WikiPeople and WD50K dataset)
```
python run.py --dataset jf17k --gpu 0

python run.py --dataset wikipeople --gpu 0

python run.py --dataset wd50k --gpu 0
```
The datasets are available here: https://www.dropbox.com/s/iz5wxp0uldx5i05/data.zip?dl=0 , and put them into the data folder.

###### Parameter setting:
In `run.py`, you can set:

`--dataset`: input dataset

`--epochs`: number of training epochs

`--batch_size`: batch size of training set

`--dim`: embedding size

`--learning_rate`: learning rate

`--self_attention_layers`: number of self-attention layers

`--gat_layers`: number of GAT layers

`--gpu`: gpu to be used for train and test the model

`--num_attention_heads`: number of attention heads

# Python lib versions
Python: 3.7.13

torch: 1.11.0

# Reference
If you use our code or datasets, please cite:
```
@inproceedings{lu2023helios,
  title={HELIOS: Hyper-Relational Schema Modeling from Knowledge Graphs},
  author={Lu, Yuhuan and Deng, Bangchao and Yu, Weijian and Yang, Dingqi},
  booktitle={Proceedings of the 31st ACM International Conference on Multimedia},
  pages={xxxx--xxxx},
  year={2023}
}
```
