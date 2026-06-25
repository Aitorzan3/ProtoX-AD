# ProtoX-AD: Self-Explainable Time Series Anomaly Detection and Characterization

This is the official code for a PyTorch implementation of the paper **ProtoX-AD: Self-Explainable Time Series Anomaly Detection and Characterization**, which can be found here: [https://arxiv.org/abs/2606.13277](https://arxiv.org/abs/2606.13277).

ProtoX-AD is a self-supervised framework for time series anomaly detection that combines transformation-based representation learning with prototype-based characterization. The method is designed not only to detect anomalous time series, but also to provide insight into the types of anomaly patterns captured by the model through learned prototypes.

The code allows users to reproduce our experiments and easily apply ProtoX-AD to their own time series datasets. A step-by-step tutorial is provided in [`ProtoX-AD_USAGE.ipynb`](ProtoX-AD_USAGE.ipynb), where we show how to load data, define transformation modules, train ProtoX-AD, evaluate anomaly detection performance, and inspect the learned prototypes for anomaly characterization.

ProtoX-AD is designed to be easily adapted to different datasets. Users can define their own transformation module to generate dataset-specific augmented views and plug it directly into the ProtoX-AD pipeline. This plug-and-play design makes it simple to incorporate prior knowledge about the target problem while keeping the rest of the workflow unchanged.

## Citation

Please consider citing our paper if you use **ProtoX-AD** in your work:

```bibtex
@article{sanchez2026protox,
  title={ProtoX-AD: Self-Explainable Time Series Anomaly Detection and Characterization},
  author={S{\'a}nchez-Ferrera, Aitor and Wetzer, Elisabeth and Wickstr{\o}m, Kristoffer and Kampffmeyer, Michael and Jenssen, Robert},
  journal={arXiv preprint arXiv:2606.13277},
  year={2026}
}
