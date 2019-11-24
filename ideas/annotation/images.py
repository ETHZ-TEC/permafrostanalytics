import stuett
from stuett.global_config import get_setting, setting_exists, set_setting

import argparse
from pathlib import Path 

import xarray as xr
import numpy as np
import json
import pandas as pd
import os 

import dash_canvas
import dash
from dash.dependencies import Input, Output, State
import dash_html_components as html
import dash_core_components as dcc
from dash.exceptions import PreventUpdate
from dash_canvas.utils import (parse_jsonstring,
                              superpixel_color_segmentation,
                              image_with_contour, image_string_to_PILImage,
                              array_to_data_url)

from dash_canvas.utils import parse_jsonstring_rectangle
from flask_caching import Cache
import dash_table
from textwrap import dedent
import json 
import uuid
from skimage import io

parser = argparse.ArgumentParser(description="Image Annotation Tool")
parser.add_argument(
    "-p",
    "--path",
    type=str,
    default=str(Path(__file__).absolute().parent.joinpath("..", "..", "data/")),
    help="The path to the folder containing the permafrost hackathon data",
)
parser.add_argument('-a', '--azure', action='store_true', help='Load data from Azure')
args = parser.parse_args()

data_path = Path(args.path)

prefix = "timelapse_images"
if args.azure:
    from stuett.global_config import get_setting, setting_exists

    account_name = (
        get_setting("azure")["account_name"]
        if setting_exists("azure")
        else "storageaccountperma8980"
        )
    account_key = get_setting("azure")["account_key"] if setting_exists("azure") else None
    store = stuett.ABSStore(
        container="hackathon-on-permafrost",
        prefix=prefix,
        account_name=account_name,
        account_key=account_key,
        blob_service_kwargs={},
    )
else:
    folder = Path(data_path).joinpath(prefix)
    store = stuett.DirectoryStore(folder)
    if not folder.exists():
        raise RuntimeError('Please provide a valid path to the permafrost data or see README how to download it')


# Setting a user directory to speed up image lookup
set_setting('user_dir', str(Path(__file__).absolute().parent.joinpath("..", "..", "data", "user_dir")))
os.makedirs(get_setting('user_dir'),exist_ok=True)

# node = stuett.data.MHDSLRImages(base_directory=folder, output_format='base64', start_time = pd.to_datetime('2017-04-05'), end_time = pd.to_datetime('2017-04-06'))
node = stuett.data.MHDSLRFilenames(base_directory=folder, start_time = pd.to_datetime('2017-01-01'), end_time = pd.to_datetime('2017-12-31'))


data = node()


filename = str(folder.joinpath(data.iloc[0]['filename']).resolve())
# print(filename)
# filename = '/home/matthmey/repos/stuett/frontends/permafrostanalytics/data/timelapse_images/2017-01-05/20170105_080011.JPG'
# filename = "/app/apps/remove-background/assets/dress.jpg"
filename = 'https://bestpostarchive.com/wp-content/uploads/2019/02/driving-in-the-streets-of-san-fr-800x445.jpg'
# filename = 'https://iclothproducts.com/products/icloth-lens-and-screen-cleaner-pro-grade-individually-wrapped-wet-wipes-wipes-for-cleaning-small-electronic-devices-like-smartphones-and-tablets'





# img_app3 = img_app3[::32,::32,:]

# print(type(img_app3),img_app3.dtype,img_app3.shape)
# img_app3 = np.zeros((16,16,3),dtype='uint8')


app = dash.Dash(__name__)
server = app.server
# app.config.suppress_callback_exceptions = True


cache = Cache(app.server, config={
    'CACHE_TYPE': 'redis',
    # Note that filesystem cache doesn't work on systems with ephemeral
    # filesystems like Heroku.
    'CACHE_TYPE': 'filesystem',
    'CACHE_DIR': 'cache-directory',

    # should be equal to maximum number of users on the app at a single time
    # higher numbers will store more data in the filesystem / redis cache
    'CACHE_THRESHOLD': 200
})


def store_dataframe(session_id, data):
    @cache.memoize()
    def query_and_serialize_data(session_id, data):

        #TODO: replace with saving the data

        #TODO: load the old data from the store

        print(data)
        df = pd.DataFrame(data)
        print(df)

        # images = []
        # times = []
        # for timestamp, element in filenames.iterrows():
        #     filename = Path(self.config["base_directory"]).joinpath(element.filename)
        #     img = Image.open(filename)
        #     images.append(np.array(img))
        #     times.append(timestamp)

        # images = np.array(images)
        # data = xr.DataArray(
        #     images, coords={"time": times}, dims=["time", "x", "y", "c"], name="Image"
        # )
        # data.attrs["format"] = "jpg"


                

        return 

    return query_and_serialize_data(session_id, data)


# TODO: delete
list_columns = ['width', 'height', 'left', 'top', 'label']
columns = [{"name": i, "id": i} for i in list_columns]
columns[-1]['presentation'] = 'dropdown'

def serve_layout():
    session_id = str(uuid.uuid4())
    
    layout = html.Div([
        html.Div([
            html.Div(session_id, id='session-id', style={'display': 'none'}),
            html.Div([], id='storage', style={'display': 'none'}),
            html.H3('Permafrost Image Annotation'),
            dcc.DatePickerSingle(
                id='my-date-picker-single',
                min_date_allowed=stuett.to_datetime("2017-01-01"),
                max_date_allowed=stuett.to_datetime("2017-12-31"),
                initial_visible_month=stuett.to_datetime("2017-01-01"),
                date="2017-01-01",
                display_format='Y-MM-DD',
            ),html.Div(id='date_indicator'),
            dash_canvas.DashCanvas(
                id='canvas',
                width=500,
                tool="select",
                lineWidth=2,
                # json_data_in=json_template,
                # filename=filename,
                hide_buttons=['pencil', 'line'],
                goButtonTitle='Get coordinates',
                updateButtonTitle='MuUpdat'
                ),
                ], className="six columns"),
        html.Div([
            dcc.Dropdown(
                id='bb_label_dropdown',
                options=[
                    {'label': 'Mountaineer', 'value': 'red'},
                    {'label': 'Lens Flare', 'value': 'green'},
                ],
                value='red'
            ),
            dcc.Dropdown(
                id='static_label_dropdown',
                options=[
                    {'label': 'Snow', 'value': 'snow'},
                    {'label': 'Sunshine', 'value': 'sunshine'},
                ],
                value=[],
                multi=True
            ),
            dcc.Store(id='cache',data=0),
            dcc.Store(id='index',data=0),
            dcc.Store(id='sync',data=True),
            html.Div(id='loop_breaker_container', children=[]),
            # dash_table.DataTable(
            #     id='table',
            #     columns=columns,
            #     editable=True,
            #     row_deletable=True,
            #     dropdown={
            #             'label':{
            #             'options': [
            #                 {'label': i, 'value': i}
            #                 for i in ['car', 'truck', 'bike', 'pedestrian']
            #             ]
            #         },
            #     }
            #     ),
            ], className="six columns"),
            dcc.Markdown("Annotate by selecting per picture labels or draw bounding boxes with the rectangle tool"
                        "Note: Rotating bounding boxes will result in incorrect labels."
            )],# Div
        className="row")
    
    return layout





app.layout = serve_layout


@app.callback(
    Output('canvas', 'lineColor'),
    [Input('bb_label_dropdown', 'value')])
def update_output(value):
    return value

@app.callback(
    Output('static_label_dropdown', 'style'),
    [Input('static_label_dropdown', 'value')])
def update_output(value):
    print(value)
    raise PreventUpdate

# @app.callback(Output('cities', 'values'),
#               [Input('all', 'values')])
# def update_cities(inputs):
#     if len(inputs) == 0:
#         return []
#     else:
#         return ['NYC', 'MTL', 'SF']

# # Thanks to HadoopMarc 
# @app.callback(Output('loop_breaker_container', 'children'),
#               [Input('cities', 'values')],
#               [State('all', 'values')])
# def update_all(inputs, _):
#     states = dash.callback_context.states
#     if len(inputs) == 3 and states['all.values'] == []:
#         return [html.Div(id='loop_breaker', children=True)]
#     elif len(inputs) == 0 and states['all.values'] == ['all']:
#         return [html.Div(id='loop_breaker', children=False)]
#     else:
#         return []

# @app.callback(Output('all', 'values'),
#               [Input('loop_breaker', 'children')])
# def update_loop(all_true):
#     if all_true:
#         return ['all']
#     else:
#         return []

@app.callback(Output('my-date-picker-single','date'),
              [Input('canvas', 'prev_trigger'), Input('canvas', 'next_trigger')],[State('index','data')])
def reduce_help(prev_trigger,next_trigger,index):
    global data, folder

    ctx = dash.callback_context

    if not ctx.triggered:
        button_id = 'No clicks yet'
        raise PreventUpdate
    else:
        button_id = ctx.triggered[0]['prop_id']
    print(button_id)

    if index is None:
        index = 0

    if button_id == 'canvas.prev_trigger':
        index = index - 1
        if index < 0:
            index = 0
    elif button_id == 'canvas.next_trigger':
        index = index + 1
        if index >= len(data):
            index = len(data)-1
    else:
        raise PreventUpdate

    datetime = data.index[index]

    return datetime

@app.callback(
    [Output('canvas', 'id')],
    [Input('canvas', 'json_data_out'),Input('session-id', 'children')])
def get_json(data,session_id):
    ctx = dash.callback_context

    if not ctx.triggered:
        raise PreventUpdate
    else:
        prop_id = ctx.triggered[0]['prop_id']

    if prop_id != 'canvas.json_data_out':
        raise PreventUpdate

    print(session_id)
    props = parse_jsonstring_rectangle(data)
    
    df = store_dataframe(session_id, props)
    raise PreventUpdate 

@app.callback(
    [Output('canvas', 'json_data_in'), Output('index', 'data'), Output('date_indicator','children')],
    [Input('my-date-picker-single', 'date')])
def update_output(date):
    global data, folder
    if date is not None:
        print(date)
        index = data.index.get_loc(date, method="nearest")
        print('filedate',index)

        if isinstance(index,slice):
            index =  index.start

        print('clicked')

        info_box = str(data.index[index])
    
        filename = str(folder.joinpath(data.iloc[index]['filename']).resolve())

        try:
            img = io.imread(filename)
            print(img.shape)
        except:
            info_box = "Error loading the image"
            img = np.zeros((4288, 2848, 3),dtype='uint8')
            
        img = img[::8,::8,:]
        image_content = array_to_data_url(img)

        # load data from index
        start_and_image = '{"version":"2.4.3","objects":[{"type":"image","version":"2.4.3","originX":"left","originY":"top","left":0,"top":0,"width":%d,"height":%d,"src":"%s"},'%(img.shape[1],img.shape[0],image_content)
        rect = '{"type":"rect","version":"2.4.3","originX":"left","originY":"top","left":34,"top":41.57,"width":44,"height":47,"fill":"transparent","stroke":"%s","strokeWidth":2}'%("red")
        end_objects = ']}'

        json_data = ''.join([start_and_image, rect, end_objects])

        return json_data, index, info_box

    raise PreventUpdate


# @app.callback(Output('canvas', 'lineWidth'),
#               [Input('canvas', 'json_data_in')])
# def reduce_help(json_data_in):
#     # print(json_data_in)
#     raise PreventUpdate

# @app.callback(Output('table', 'data'),
# @app.callback(Output('cache', 'data'),
#               [Input('canvas', 'json_data_in')],[State('cache','data')])
# def show_string(string,cache):
#     print(cache)
#     props = parse_jsonstring_rectangle(string)
#     df = pd.DataFrame(props, columns=list_columns[:-1])
#     df['type'] = cache['type']
#     return df.to_dict("records")


# @app.callback([Output('cache', 'data'),Output('sync', 'data')],
#               [Input('table', 'data'),Input('canvas', 'json_data_in')],[State('sync','data')])
# def show_string(tabledata,string,sync):
#     if sync:
#         raise PreventUpdate
#     props = parse_jsonstring_rectangle(string)
#     print(tabledata,props,cache)
#     df = pd.DataFrame(props, columns=list_columns[:-1])
#     df['type'] = 'car'

#     return df.to_dict("records"), True


# @app.callback([Output('table', 'data'),Output('canvas', 'json_data_in')],
#               [Input('cache', 'data')],[State('sync','data')])
# def sync(cache,sync):
#     props = parse_jsonstring_rectangle(string)
#     print(tabledata,props,cache)
#     df = pd.DataFrame(props, columns=list_columns[:-1])
#     df['type'] = 'car'

#     return df.to_dict("records")


# @app.callback(Output('img-help', 'width'),
#               [Input('canvas', 'json_data_in')])
# def reduce_help(json_data_in):
#     if json_data_in:
#         return '0%'
#     else:
#         raise PreventUpdate

# @app.callback([Output('table', 'data'),Output('json_data_in', 'data')],
#               [Input('table', 'data')],[State('cache','data')])
# def show_string(string,cache):
#     print(cache)
#     props = parse_jsonstring_rectangle(string)
#     df = pd.DataFrame(props, columns=list_columns[:-1])
#     df['type'] = 'car'
#     return df.to_dict("records")





# 
# @app.callback(Output('canvas', 'json_data_in'),
#             [Input('table', 'derived_virtual_indices'),
#              Input('table', 'active_cell'),
#              Input('table', 'data')]
#             )
# def highlight_filter(indices, cell_index, data):
#     print(indices, cell_index, data)
#     return data


# @app.callback(Output('table-line', 'style_data_conditional'),
#              [Input('graph', 'hoverData')])
# def higlight_row(string):
#     """
#     When hovering hover label, highlight corresponding row in table,
#     using label column.
#     """
#     index = string['points'][0]['z']
#     return  [{
#         "if": {
#                 'filter': 'label eq num(%d)'%index
#             },
#         "backgroundColor": "#3D9970",
#         'color': 'white'
#         }]


if __name__ == '__main__':
    app.run_server(debug=True)

