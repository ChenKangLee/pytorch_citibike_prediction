# pytorch_citibike_prediction

Pytorch Implementation of Bike Flow Prediction with Multi-Graph Convolutional Networks [1].

## Usage

The project is broken down into four files:
Execution produces intermediate files, which are stored in the path specified by the `data_path` and `model_path` variables.
*Note: Data dependencies exists between the files, and they should be run in the given order.*

- Preprocess.ipynb
- Graph_fusion.ipynb
- Citibike.ipynb
- Evaluate.ipynb
    
`Preprocess.ipynb`  Read in and cleans the ride entries from the NYC citibike dataset. It is also responsible for producing the weather data. For more detailed requirements for data preparation, please refer to the **Data** section

`Graph_fusion.ipynb` implements the graph fusion in the paper.

`Citibike.ipynb` hosts the training process of the models.

`Evaluate.ipynb` test the trained model on the test data. 

## Requirements & Prerequisites

python 3.6
[pytorch 1.3.1](https://pytorch.org)
[pytorch-geometric](https://github.com/rusty1s/pytorch_geometric)
[geopy](https://pypi.org/project/geopy/)
[Numpy](https://numpy.org)
[Pandas](https://pandas.pydata.org)
[bokeh](https://docs.bokeh.org/en/1.4.0/)

This implemention is meant to be run on powerful machines, and takes large amount memory (20 GB+).

## Data

Several data are required to run this implementation.

- [NYC citibike dataset](https://www.citibikenyc.com/system-data)
- [NOAA Local Climatological Data (LCD)](https://www.ncdc.noaa.gov/cdo-web/datatools/lcd)

*NYC citibike dataset*: The website provides summary of all the rides, grouped into years. Before running this implementation, the data should first be concatenated into one large `.csv` file.

*NOAA Local Climatological Data (LCD)*: Request the weather data of the NYC Central park station. The range of the weather data should match the time span of the Citibike data.


## References
[1] Di Chai, Leye Wang, Qiang Yang, Bike Flow Prediction with Multi-Graph Convolutional Networks, 2018

