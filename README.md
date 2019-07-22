# Quality-Net: An End-to-End Non-intrusive Speech Quality Assessment Model based on BLSTM (Interspeech 2018)


### Introduction
This paper tries to solve the mismatch (as in Fig.1) between training objective function and evaluation metrics which are usually highly correlated to human perception. Due to the inconsistency, there is no guarantee that the trained model can provide optimal performance in applications. In this study, we propose an end-to-end utterance-based speech enhancement framework using fully convolutional neural networks (FCN) to reduce the gap between the model optimization and the evaluation criterion. Because of the utterance-based optimization, temporal correlation information of long speech segments, or even at the entire utterance level, can be considered to directly optimize perception-based objective functions.

### Major Contribution
1) Utterance-based waveform enhancement
2) Direct short-time objective intelligibility (STOI) score optimization (without any approximation)


For more details and evaluation results, please check out our  [paper](https://arxiv.org/ftp/arxiv/papers/1808/1808.05344.pdf).

![teaser](https://github.com/JasonSWFu/End-to-end-waveform-utterance-enhancement/blob/master/images/Fig1_3.png)


### Dependencies:
* Python 2.7


### Citation

If you find the code useful in your research, please cite:
    
    @inproceedings{fu2018quality,
      title={Quality-Net: An end-to-end non-intrusive speech quality assessment model based on blstm},
      author={Fu, Szu-Wei and Tsao, Yu and Hwang, Hsin-Te and Wang, Hsin-Min},
      booktitle={Interspeech},
      year={2018}}
    
### Contact

e-mail: jasonfu@iis.sinica.edu.tw or d04922007@ntu.edu.tw

