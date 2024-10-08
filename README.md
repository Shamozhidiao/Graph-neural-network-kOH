# Graph-neural-network-kOH

Set up conda environment
```
pytorch=1.12.1
torch-geometric==2.4.0
torch-cluster==1.6.0+pt112cu113
torch-scatter==2.1.0+pt112cu113
torch-sparse==0.6.16+pt112cu113
torch-spline-conv==1.2.1+pt112cu113
torchmetrics==1.2.0
rdkit==2023.9.1
matplotlib==3.8.2
numpy=1.24.3
pandas==2.1.3
```
The data for pre-training can be found in pubchem-10m-clean.txt. The link is available here:
https://doi.org/10.48550/arXiv.2010.09885
https://github.com/seyonechithrananda/bert-loves-chemistry

### Repo

https://github.com/yuyangw/MolCLR

https://github.com/junxia97/Mole-BERT

https://github.com/snap-stanford/pretrain-gnns

https://github.com/microsoft/DVMP

### Paper

[1] Wu, Z., Ramsundar, B., Feinberg, E. N., Gomes, J., Geniesse, C., Pappu, A. S., ... & Pande, V. (2018). MoleculeNet: a benchmark for molecular machine learning. Chemical science, 9(2), 513-530.

[2] Yang, K., Swanson, K., Jin, W., Coley, C., Eiden, P., Gao, H., ... & Barzilay, R. (2019). Analyzing learned molecular representations for property prediction. Journal of chemical information and modeling, 59(8), 3370-3388.

[3] Wang, Y., Wang, J., Cao, Z., & Barati Farimani, A. (2022). Molecular contrastive learning of representations via graph neural networks. Nature Machine Intelligence, 4(3), 279-287.

[4] Liu, S., Demirel, M. F., & Liang, Y. (2019). N-gram graph: Simple unsupervised representation for graphs, with applications to molecules. Advances in neural information processing systems, 32.

[5] Zhu, J., Xia, Y., Wu, L., Xie, S., Zhou, W., Qin, T., ... & Liu, T. Y. (2023, August). Dual-view molecular pre-training. In Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (pp. 3615-3627).
