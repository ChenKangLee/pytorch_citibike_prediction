# pytorch_citibike_prediction

Pytorch Implementation of Bike Flow Prediction with Multi-Graph Convolutional Networks [1].


## Outline
1. Preprocessing
2. The Model
3. Training Process
4. Evaluation

---


## 1. Data & Preprocessing

### 1.1 Station Graphs

`Proprocess.ipynb` performs the data preprocessing

We are using the NYC Citibike records from June 2013 to March 2019. 

First, a python `dictionary` is created to hold key attributes of each stations

```python
stations_dict = {
    stationid: {
        index: (int)               // the index in tensors that each stationid maps to
        is_alive: (bool)           // whether the station is still alive

        earliest: (datetime)       // the time of the earliest entry involving stationid
        latest: (datetime)         // the time of the latest entry involving stationid
        alive_time: (int)          // the duration this station is alive (in hours)

        lat: (float)               // latitude
        long: (float)              // longitude
    }
}
```

`is_alive` can be obtained from the citibike stations feeds. In the dataset, there exists stations that have `lat = 0` and `long = 0`. A search on the station list shows that these are test stations and bike depots. The entries that involves these stations are removed from the station list.

With this object, we can calculate the graphs necessary to perform graph fusion, namely **Station Distance**, **Average Ride Count**, **Inbound Correlation** and **Outbound Correlation**. All four graphs are stored in a python dictionary of dictionaries structure. 

Pair-wise distances between stations are calculated using `geopy.distance`.

### 1.2 Weather Data

For weather data we are using the NOAA Local Climatological Data (LCD). It provides daily summary of weather data observed from the weather station in Manhattan Central Park, NY.

The dataset can be requested at the [NOAA website](https://www.ncdc.noaa.gov/cdo-web/datatools/lcd). Note that we have to manually select the starting and ending dates of the weather data, so that it matches the Citibike dataset.

There are multiple missing values in the LCD dataset. This implementation ended up only keeping **18** types of weather data. The missing values in these data are either **zeroed** or **generated via interpolation** .

All the weather data are Z-normailized.

The day of week is also included as weather data. The value is one-hot encoded.

### 1.3 Training/Testing Data

This implementation takes in data in the form of inflow and outflow numbers of each stations in each time interval. The prediction granularity is set to **one hour**. The last 40 days of the dataset were used as test data, the rest are training data. 

    Number of training data = 49414.
    Number of testing data = 960

### 1.4 Graph Fusion

Graph fusion is implemented in `Graph_fusion.ipynb`.

The graphs are first covert from python `dictionary` objects to Numpy arrays, in the form of dense adjacency matrices.

For the most part, the graph fusion process follows the paper. Before the weighted sum, each graph is first normalized. Here we are using scikitlearn's `preprocessing.normailize()` function, which uses L2-norm normalization across the whole matrix (instead of row-wise normalization).

## 2. The Model

The model basically follows the architecture specified in the paper. Aside from the following (potential) differences:

1. The paper did not specify if the inflow and outflow are modeled together. This implementation assumes so.

2. The paper did not explain the inconsistencies in the dimensions of the decoder outputs. We took the liberty and added a `torch.nn.Linear` layer to convert the output dimension from `64` to `2`


## 3. Training Process

### 3.1 PyTorch Geometric

For the Graph Convolution, we are using the `Pytorch-Geometric` package. It creates `Data` objects that represents a graph.

The package also provides a `Dataloader` that can efficently pack multiple graphs into one `Batch` object, which is a large disconnected graph.

### 3.2 Sliding Window

We first use `DataLoader` to load multiple bike flow data from consecutive time stamps. These flow data are combined into a `Batch` object for convolution. 

After graph convolution is performed on each `Batch` object, we generate the training data for the encoder-decoder in a sliding window manner. 

To better utilize all the training data, we padd the first `BATCH_SIZE - 1` with all zeros.

### 3.3 Optimization

During the training process, we encounter an issue where the convolution of the fully connected `fusion_graph` requires too much computing power (it takes around 2 hours to advance 1 epoch).

The solution is to trim the edges of the `fusion_graph`. Notice that the majority of the edge weights have relatively small values. These very weak links represents weak correlation between stations, or long physical distances. Turns out we can obtain much of the important information with only a fraction of the original edges. 

![Edge weight histogram](./edge_weight_hist.png)

 *Fig. 1: The edge weight histogram of the fusion_graph*



We remove the edges with weight below a certain threshold. In the current version, we are using `THRESHOLD = 0.011`, which leaves us with about 1/10 of the total edges (94100 edge as appose to 921600).

This gives a massive speed up to the training process. One epoch now takes about 3 minutes.

The reduction of the graph size also enables a bigger batch size.

### 3.4 Plotting the Loss

We trained both the encoder-decoder model and the encoder-fc model for 50 epochs, and the loss is presented in the following figures.

<img src="encoder_decoder.png" alt="encoder_decoder" style="zoom:50%;" />

*fig 2. the training loss of the encoder-decoder network*

<img src="pred_net.png" alt="pred_net" style="zoom:50%;" />

*fig 3. the training loss of the encoder-fc network*



## 4. Evaluation

We saved the model weight after training 60 epochs (30 encoder-decoder, 30 pred-network) and 100 epochs (50 encoder-decoder, 50 pred-network).

|       | Inflow Loss (RMSE) | Outflow Loss (RMSE) |
|  ---  |          ---:      |            ---:     |
| 30-30 | 4.346              | 4.322               |
| 50-50 | 4.471              | 4.499               |

The results are slightly worse than the ones presented in the paper.
And training for extended period of time seemed to cause the model to overfit and yield worse results.



## 5. Improvements

1. Change the `THRESHOLD` value to keep more edges?
2. Include or exclude some weather data. Since the paper did not give a detailed list of the weather data they are using.


## References
[1] [Di Chai, Leye Wang, Qiang Yang, Bike Flow Prediction with Multi-Graph Convolutional Networks, 2018](https://arxiv.org/abs/1807.10934)

