
# Plot saptial-temporal map
def st_map(lat, lng, build_order, meta_info, file_name,
           zoom=11, style='light'):

    import numpy as np
    import plotly
    from plotly.graph_objs import Scattermapbox, Layout

    mapboxAccessToken = "pk.eyJ1Ijoicm1ldGZjIiwiYSI6ImNqN2JjN3l3NjBxc3MycXAzNnh6M2oxbGoifQ.WFNVzFwNJ9ILp0Jxa03mCQ"

    bikeStations = [Scattermapbox(
        lon=lng,
        lat=lat,
        text=meta_info,
        mode='markers',
        marker=dict(
            size=6,
            color=['rgb(%s, %s, %s)' % (255,
                                        195 - e * 195 / max(build_order),
                                        195 - e * 195 / max(build_order)) for e in build_order],
            opacity=1,
        ))]
    
    layout = Layout(
        title='Bike Station Location & The latest built stations with deeper color',
        autosize=True,
        hovermode='closest',
        showlegend=False,
        mapbox=dict(
            accesstoken=mapboxAccessToken,
            bearing=0,
            center=dict(
                lat=np.median(lat),
                lon=np.median(lng)
            ),
            pitch=0,
            zoom=zoom,
            style=style
        ),
    )

    fig = dict(data=bikeStations, layout=layout)
    plotly.offline.plot(fig, filename=file_name)