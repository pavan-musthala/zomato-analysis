# Zomato Restaurant Analysis Dashboard

An interactive dashboard analyzing Zomato restaurant data using Streamlit.

## Features
- Interactive restaurant data exploration
- Geographical analysis of restaurant distribution
- Price range and rating analysis
- Cuisine type distribution
- Online ordering and table booking insights
- Customizable filters and visualizations

## Data Analysis
The dashboard provides insights into:
- Restaurant ratings and reviews
- Price distributions
- Popular cuisines
- Location-based analysis
- Online ordering trends

## Setup and Installation

1. Clone the repository:
```bash
git clone https://github.com/pavan-musthala/zomato-analysis.git
cd zomato-analysis
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the app locally:
```bash
streamlit run zomato_dashboard.py
```

## Data Source
The dataset used in this analysis is from Zomato's restaurant data, containing information about restaurants, their locations, cuisines, and ratings.

### Getting the Data
Due to file size limitations, the dataset is not included in this repository. To run this dashboard:

1. Download the Zomato dataset from [Google Drive](https://drive.google.com/file/d/1AI0hdk5m0povk_RnlyxC5LCF1IbLcy2A/view?usp=sharing)
2. Place the downloaded `zomato.csv` file in the `data/` directory of this project

## License
MIT License
