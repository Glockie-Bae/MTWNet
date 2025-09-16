

Global Trend and Local Fluctuation: A Modeling Strategy for Non-Periodic Multi-Stage Data

In this paper, we present MTWNet as a powerful framework for industrial and benchmark time series analysis, which can

🏆 Achieve state-of-the-art performance on both the industrial spot welding dataset and the UEA2018 benchmark datasets.

🌟 Overcome the limitations of non-periodic time series analysis by decomposing sequences into global trends and local fluctuations, thereby compensating for the lack of explicit periodic information.



## Multi-perspective Temporal 2D-Variation Wavelet Net for Non-Periodic Time Series Classification

Most time series methods rely on trend–seasonality decomposition, but industrial production data are multi-stage and usually non-periodic. MTWNet tackles this by using adaptive windows to capture potential temporal dependencies, overcoming the limitations of existing methods on non-periodic data.

<p align="center">
<img src=".\pic\MTWNet.png" alt="" align=center />
</p>

## Get Started

1. Install requirements. ```pip install -r requirements.txt```
2. Download data. You can download UEA2018 datasets from [Time Series Machine Learning Websitr](https://www.timeseriesclassification.com/index.php)
 > The  industrial spot welding dataset is available upon request from the corresponding author (augepang@scnu.edu.cn).
3. Train the model. We provide the experiment scripts of all benchmarks under the folder `./scripts`.

## Main Results
We conduct extensive experiments to evaluate the performance of MTWNet, covering classification and short-term forecasting, including 28 UEA real-world and 6 real-world forecasting benchmarks.
**🏆 MTWNet achieves consistent state-of-the-art performance in all benchmarks**, covering a large variety of series with different frequencies, variate numbers and real-world scenarios.

### Industrial datasets


<p align="center">
<img src=".\pic\Industrial.png" alt="" align=center />
</p>

### UEA datasets


<p align="center">
<img src=".\pic\UEA.png" alt="" align=center />
</p>

### Model Ablations

To verify the effectiveness of each component of MTWNet, we provide a detailed ablation study of the Multi-scale Wavelet Convolution and Frequency-domain Detail Enhancement modules on all 29 experimental benchmarks.

<p align="center">
<img src=".\pic\ablation.png" alt="" align=center />
</p>



## Acknowledgement

We appreciate the following GitHub repos a lot for their valuable code and efforts.
- Time-Series-Library (https://github.com/thuml/Time-Series-Library)
- Autoformer (https://github.com/thuml/Autoformer)

## Contact

If you have any questions or want to use the code, feel free to contact:
* Rongjie Zhang (zrj22127@gmail.com)
* Xiongwen Pang (augepang@scnu.edu.cn)




