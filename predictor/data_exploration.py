import pandas as pd
import json
import plotly.express as px
import plotly.io as pio

def get_rwanda_map(df):
    district_counts = df['district'].value_counts().reset_index()
    district_counts.columns = ['district', 'client_count']
    
    with open('dummy-data/rwanda_districts.geojson', 'r', encoding='utf-8') as f:
        rwanda_geojson = json.load(f)

    fig = px.choropleth(
        district_counts,
        geojson=rwanda_geojson,
        locations='district',
        featureidkey='properties.NAME_2',
        color='client_count',
        color_continuous_scale="Viridis",
        title="Vehicle Clients per District in Rwanda",
        labels={'client_count': 'Number of Clients'}
    )
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(margin={"r":0,"t":40,"l":0,"b":0})
    
    return pio.to_html(fig, full_html=False)


# Data Exploration
def dataset_exploration(df):
  table_html = df.head().to_html(
  classes="table table-bordered table-striped table-sm",
  float_format="%.2f",
  justify="center",
  index=False,
  )
  return table_html
# Data description
def data_exploration(df):
  table_html = df.head().to_html(
  classes="table table-bordered table-striped table-sm",
  float_format="%.2f",
  justify="center",
  )
  return table_html