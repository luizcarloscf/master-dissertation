# Datasets

The public datasets used in this work are:

- **NTU RGB+D**  
  [Amir Shahroudy, Jun Liu, Tian-Tsong Ng, Gang Wang, "NTU RGB+D: A Large Scale Dataset for 3D Human Activity Analysis", CVPR 2016](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Shahroudy_NTU_RGBD_A_CVPR_2016_paper.pdf). Also known as **"NTU RGB+D"**. This dataset contains **60 action classes** and **56,880 video samples**.

- **NTU RGB+D 120**  
  [Jun Liu, Amir Shahroudy, Mauricio Perez, Gang Wang, Ling-Yu Duan, Alex C. Kot, "NTU RGB+D 120: A Large-Scale Benchmark for 3D Human Activity Understanding", TPAMI 2019](https://arxiv.org/pdf/1905.04757). Also known as **"NTU RGB+D 120"**. It extends the original NTU RGB+D by adding **60 additional action classes** and **57,600 more videos**, totaling **120 classes** and **114,480 samples**.

Instructions for downloading, preprocessing, and normalizing these datasets are provided in [`ntu/README.md`](ntu/README.md).

In addition to these public benchmarks, this work also includes a custom gesture recognition dataset developed specifically for the project. Instructions for this dataset can be found in [`hri-is/README.md`](hri-is/README.md).