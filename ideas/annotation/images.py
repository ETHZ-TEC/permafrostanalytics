import stuett
from stuett.global_config import get_setting, setting_exists, set_setting

import argparse
from pathlib import Path 

import xarray as xr
import numpy as np
import json
import pandas as pd
import os 
import zarr 

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
from skimage import io as imio
import io, codecs

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

def get_store(data_path, prefix):
    folder = Path(data_path).joinpath(prefix)
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
        store = stuett.DirectoryStore(folder)
        if not folder.exists():
            raise RuntimeError('Please provide a valid path to the permafrost data or see README how to download it')
    
    return store, folder

prefix = "timelapse_images"
store, folder = get_store(data_path,prefix)

# Setting a user directory to speed up image lookup
set_setting('user_dir', str(Path(__file__).absolute().parent.joinpath("..", "..", "data", "user_dir")))
local_annotation_path = Path(get_setting('user_dir')).joinpath('annotations')
os.makedirs(local_annotation_path,exist_ok=True)
local_store = stuett.DirectoryStore(local_annotation_path)
account_name = 'https://storageaccountperma8980.blob.core.windows.net/hackathon-public-rw'
account_key = 'st=2019-11-25T15%3A03%3A51Z&se=2019-12-04T15%3A03%3A00Z&sp=rw&sv=2018-03-28&sr=c&sig=1q9R%2Fp9y%2B%2Fwyam3qpfBf%2BJyG0cKlZ%2B4Ta9uNnAR7hJE%3D'

remote_store = None
remote_store = stuett.ABSStore(
            container="hackathon-public-rw",
            prefix='',
            account_name=account_name,
            account_key=account_key,
            blob_service_kwargs={},
        )

# node = stuett.data.MHDSLRFilenames(base_directory=folder, start_time = pd.to_datetime('2017-01-01'), end_time = pd.to_datetime('2017-12-31'))
node = stuett.data.MHDSLRFilenames(base_directory=folder, store=store, force_write_to_remote=True, start_time = pd.to_datetime('2017-01-01'), end_time = pd.to_datetime('2017-12-31'))
data = node()

static_label_mapping = {'snow':'Snow', 'sunshine':'Sunshine'}
mapping = {'red':'Mountaineer', 'green':'Lens Flare'}
reverse_mapping = {v: k for k, v in mapping.items()}
img_shape = (4288, 2848, 3)
img_downsampling = 8

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
            html.Div([
                html.Div([
                    dcc.DatePickerSingle(
                    id='my-date-picker-single',
                    min_date_allowed=stuett.to_datetime("2017-01-01"),
                    max_date_allowed=stuett.to_datetime("2017-12-31"),
                    initial_visible_month=None,
                    date="2017-01-01",
                    display_format='Y-MM-DD',
                )],style={'width':'50%', 'display': 'inline-block'}),
                html.Div(id='date_indicator',style={'width':'50%', 'display': 'inline-block'})
            ]),
            html.Div([
                dash_canvas.DashCanvas(
                    id='canvas',
                    width=500,
                    tool="select",
                    lineWidth=2,
                    # json_data_in=json_template,
                    # filename=filename,
                    hide_buttons=['pencil', 'line']),
                    ],style={'text-align':'center'}),
        ]),
        html.Div([
            dcc.Dropdown(
                id='bb_label_dropdown',
                options=[ {'label': mapping[m], 'value': m} for m in mapping.keys()],
                value='red'
            ),
            dcc.Dropdown(
                id='static_label_dropdown',
                options=[ {'label': static_label_mapping[m], 'value': m} for m in static_label_mapping.keys()],
                value=[],
                multi=True
            ),
            dcc.Store(id='cache',data=0),
            dcc.Store(id='index',data=0),
            dcc.Store(id='sync',data=True),
 
            ], className="six columns"),
            dcc.Markdown("Annotate by selecting per picture labels or draw bounding boxes with the rectangle tool"
                        "Note: Rotating bounding boxes will result in incorrect labels."
            )],style={'width':'50%'},# Div
        className="row")
    
    return layout





app.layout = serve_layout


@app.callback(
    [Output('canvas', 'lineColor'),Output('bb_label_dropdown', 'style')],
    [Input('bb_label_dropdown', 'value')])
def update_output(value):
    return value, {'color':value}

@app.callback(
    Output('static_label_dropdown', 'style'),
    [Input('static_label_dropdown', 'value')])
def update_output(value):
    print(value)
    raise PreventUpdate

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

def parse_labels(string,mapping, static_label):
    """Returns array of rectangles geometry and their labels
    
    Arguments:
        string {str} -- JSON string
        mapping {dict} -- Mapping from color to label
        static_label {list} -- List of labels valid for the whole image
    
    Returns:
        ndarray -- array containing rectangle information
    """
    try:
        data = json.loads(string)
    except:
        return None
    scale = 1
    img_width = 1
    img_height = 1
    props = []

    for obj in data['objects']:
        if obj['type'] == 'image':
            # print('img')
            # print(obj['scaleX'],obj['scaleY'],obj['width'],obj['height'])
            # scale = obj['scaleX']
            img_width  = obj['width']
            img_height = obj['height']

    for obj in data['objects']:     
        if obj['type'] == 'rect':
            print(obj)
            # scale_factor = obj['scaleX'] / scale
            try:
                label = mapping[obj['stroke']]
            except:
                raise RuntimeError(f'Could not find mapping for {obj["stroke"]}')
            item = [obj['right'],
                    obj['bottom'],
                    obj['left'],
                    obj['top']]

            print(item)

            item = np.array(item)
            # convert ltwh to corner points (ltrb)
            # item[0] = item[0] + item[2]
            # item[1] = item[1] + item[3]

            # item = scale_factor * item

            # item[0], item[2] = item[0] / img_width, item[2] / img_width
            # item[1], item[3] = item[1] / img_height, item[3] / img_height

            item = item.tolist()
            item += [label]

            props.append(item)

    if static_label is not None:
        for item in static_label:
            props.append([1,1,0,0,item])

    print('props', props)
    return (np.array(props))


@app.callback(
    [Output('canvas', 'id')],
    [Input('canvas', 'json_data_out'),Input('session-id', 'children'),Input('static_label_dropdown','value')],
    [State('index','data')])
def get_json(json_data_out,session_id,static_label,index):
    global mapping, data
    ctx = dash.callback_context

    if not ctx.triggered:
        raise PreventUpdate
    else:
        prop_id = ctx.triggered[0]['prop_id']

    if prop_id != 'canvas.json_data_out':
        raise PreventUpdate
    
    print('static',static_label)
    props = parse_labels(json_data_out, mapping, static_label)

    datetime = data.index[index]
    
    df = pd.DataFrame(props,columns= ['end_x','end_y','start_x','start_y','__target'])
    df['start_time'] = datetime
    df['end_time']  = datetime

    file_id = datetime.strftime("%Y%m%d_%H%M%S")
    to_csv(df, session_id,file_id)

    raise PreventUpdate 

def to_csv(df, session_id, file_id=None):
    global local_store
    if file_id is None:
        file_id = uuid.uuid4()

    filename = session_id+f'/data/{file_id}.csv'

    stuett.to_csv_with_store(local_store,filename,df)
    if remote_store is not None:
        stuett.to_csv_with_store(remote_store,filename,df)

def read_csv(session_id, file_id):
    global local_store
    filename = session_id+f'/data/{file_id}.csv'
    return stuett.read_csv_with_store(local_store,filename)

@app.callback(
    [Output('canvas', 'json_data_in'), Output('index', 'data'), Output('date_indicator','children'), Output('static_label_dropdown','value')],
    [Input('my-date-picker-single', 'date'),Input('session-id', 'children')])
def update_output(date,session_id):
    global data, folder

    static_label = [] 
    info_box = 'No info'

    if date is not None:
        print(date)
        index = data.index.get_loc(date, method="nearest")
        print('filedate',index)

        if isinstance(index,slice):
            index =  index.start
    
        filename = str(folder.joinpath(data.iloc[index]['filename']).resolve())
        
        try:
            key = data.iloc[index]['filename']
            img = imio.imread(io.BytesIO(store[key]))
            print(img.shape)
        except Exception as e:
            print(e)
            info_box = "Error loading the image"
            img = np.zeros(img_shape,dtype='uint8')
        

        img = img[::img_downsampling,::img_downsampling,:]
        image_content = array_to_data_url(img)

        # load data from index
        start_and_image = '{"version":"2.4.3","objects":[{"type":"image","version":"2.4.3", "originX":"left","originY":"top","left":0,"top":0,"width":%d,"height":%d,"src":"%s"}'%(img.shape[1],img.shape[0],image_content)

        datetime = data.index[index]
        file_id = datetime.strftime("%Y%m%d_%H%M%S")
        rectangles = ['']
        try:
            df = read_csv(session_id, file_id)
        except:
            df = None
            pass

        if df is not None:
            print(df)
            rects = []
            for i,row in df.iterrows():
                if row['__target'] in static_label_mapping:
                    static_label.append(row['__target'])
                    continue

                left = row['start_x']
                top = row['start_y']
                right = row['end_x']
                bottom = row['end_y']

                stroke = reverse_mapping[str(row['__target'])]
                rect = ',{"type":"rect","version":"2.4.3","originX":"left","originY":"top","left":%f,"top":%f,"right":%f,"bottom":%f,"width":0,"height":0,"fill":"transparent","stroke":"%s","strokeWidth":2}'%(left,top,right,bottom,stroke)

                rects.append(rect)
            
            if rects:
                rectangles = rects

            print(rects)

        end_objects = ']}'

        string = [start_and_image]
        string += rectangles
        string += [end_objects]

        json_data = ''.join(string)

        return json_data, index, str(datetime), static_label

    raise PreventUpdate


if __name__ == '__main__':
    app.run_server(debug=True)

