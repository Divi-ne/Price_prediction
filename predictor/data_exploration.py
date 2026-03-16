import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

def get_rwanda_map(df):
    district_counts = df['district'].value_counts().reset_index()
    district_counts.columns = ['district', 'client_count']

    with open('dummy-data/rwanda_districts.geojson', 'r', encoding='utf-8') as f:
        rwanda_geojson = json.load(f)

    centroids_lon = []
    centroids_lat = []
    texts = []

    for _, row in district_counts.iterrows():
        dist_name = row['district']
        count = row['client_count']

        lon_list = []
        lat_list = []
        for feature in rwanda_geojson['features']:
            if feature['properties'].get('NAME_2') == dist_name:
                geom = feature['geometry']
                if geom['type'] == 'Polygon':
                    coords = geom['coordinates'][0]
                elif geom['type'] == 'MultiPolygon':
                    coords = geom['coordinates'][0][0]
                else:
                    coords = []

                for pt in coords:
                    lon_list.append(pt[0])
                    lat_list.append(pt[1])
                break

        if lon_list and lat_list:
            centroids_lon.append(sum(lon_list)/len(lon_list))
            centroids_lat.append(sum(lat_list)/len(lat_list))
            texts.append(f"{dist_name}<br>{count}")

    fig = px.choropleth_mapbox(
        district_counts,
        geojson=rwanda_geojson,
        locations='district',
        featureidkey='properties.NAME_2',
        color='client_count',
        color_continuous_scale="Blues",
        mapbox_style="open-street-map",
        center={"lat": -1.9403, "lon": 29.8739},
        zoom=7.3,
        title="Vehicle Clients per District in Rwanda",
        labels={'client_count': 'Number of Clients'},
        opacity=0.6
    )

    # Update district borders
    fig.update_traces(marker_line_width=1.5, marker_line_color="#10368c")

    # Add static text numbers and names
    fig.add_trace(go.Scattermapbox(
        lon=centroids_lon,
        lat=centroids_lat,
        text=texts,
        mode="text",
        textfont=dict(color="#1a1a1a", size=11),
        showlegend=False,
        hoverinfo="skip"
    ))

    fig.update_layout(
        margin={"r":0,"t":40,"l":0,"b":0},
        mapbox=dict(
            center=dict(lat=-1.9403, lon=29.8739),
            zoom=7.3,
        ),
        dragmode="zoom",
    )

    return pio.to_html(fig, full_html=False, config={'scrollZoom': True})


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