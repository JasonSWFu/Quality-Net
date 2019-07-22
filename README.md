# Quality-Net: An End-to-End Non-intrusive Speech Quality Assessment Model based on BLSTM (Interspeech 2018)


### Introduction
Herein, we propose a novel, end-to-end, and non-intrusive speech quality evaluation model, termed Quality-Net, based on bidirectional long short-term memory (BLSTM). In addition, to prevent Quality-Net from becoming an incomprehensible black box, its structure is designed to automatically learn (infer) a reasonable frame-level quality. This gives Quality-Net the ability to locate the degraded regions in an utterance. Although our ultimate goal is to learn the mapping function of the human listening perception, an off-the-shelf data set with labels that meets our requirements does not exist (here, we focus on predicting the quality of noisy speech and enhanced speech given by a deep-learning-based speech enhancement model). Therefore, we apply Quality-Net to predict the PESQ scores without a clean reference.


### Major Contribution
Quality-Net is the first "end-to-end", and "non-intrusive" quality
assessment model (as shown in Fig. 1) to yield "frame-level" quality.


![teaser](https://github.com/JasonSWFu/Quality-Net/blob/master/images/Quality_Net.png)



For more details and evaluation results, please check out our  [paper](https://arxiv.org/ftp/arxiv/papers/1808/1808.05344.pdf).



### Citation

If you find the code useful in your research, please cite:
    
    @inproceedings{fu2018quality,
      title={Quality-Net: An end-to-end non-intrusive speech quality assessment model based on blstm},
      author={Fu, Szu-Wei and Tsao, Yu and Hwang, Hsin-Te and Wang, Hsin-Min},
      booktitle={Interspeech},
      year={2018}}
    
### Contact

e-mail: jasonfu@iis.sinica.edu.tw or d04922007@ntu.edu.tw

