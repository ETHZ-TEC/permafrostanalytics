"""MIT License

Copyright (c) 2019, Swiss Federal Institute of Technology (ETH Zurich), Matthias Meyer

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE."""

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
from dash_canvas.utils import (
    parse_jsonstring,
    superpixel_color_segmentation,
    image_with_contour,
    image_string_to_PILImage,
    array_to_data_url,
)

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
parser.add_argument("-l", "--local", action="store_true", help="Only use local files and not data from Azure")
parser.add_argument("-hq", "--high_quality", action="store_true", help="Use the high resolution images (timelapse_images)")
args = parser.parse_args()

data_path = Path(args.path)

if args.high_quality:
    prefix = "timelapse_images"
else:
    prefix = "timelapse_images_fast"

if not args.local:
    from stuett.global_config import get_setting, setting_exists

    account_name = (
        get_setting("azure")["account_name"]
        if setting_exists("azure")
        else "storageaccountperma8980"
    )
    account_key = (
        get_setting("azure")["account_key"] if setting_exists("azure") else None
    )
    store = stuett.ABSStore(
        container="hackathon-on-permafrost",
        prefix=prefix,
        account_name=account_name,
        account_key=account_key, 
    )
    annotation_store = stuett.ABSStore(
        container="hackathon-on-permafrost",
        prefix="annotations",
        account_name=account_name,
        account_key=account_key, 
    )

else:
    store = stuett.DirectoryStore(Path(data_path).joinpath(prefix))
    if "2017-01-01/20170101_080018.JPG" not in store:
        raise RuntimeError(
            "Please provide a valid path to the permafrost timelapse_images data or see README how to download it"
        )
    annotation_store = stuett.DirectoryStore(Path(data_path).joinpath("annotations"))
    if "annotations.csv" not in annotation_store:
        print("WARNING: Please provide a valid path to the permafrost annotation data or see README how to download it"
        )

# Setting a user directory to speed up image lookup
set_setting(
    "user_dir",
    str(Path(__file__).absolute().parent.joinpath("..", "..", "data", "user_dir")),
)
local_annotation_path = Path(get_setting("user_dir")).joinpath("annotations")
os.makedirs(local_annotation_path, exist_ok=True)
local_store = stuett.DirectoryStore(local_annotation_path)
account_name = "storageaccountperma8980"
account_key = None


# connection to remote server
token = (
    "?sv=2018-03-28&si=mypolicy&sr=c&sig=Nv7eQYw91LTI1yIkBbcXEDuvg5ldk6cYVHkJCC4WDB8%3D"
)
account_name = "storageaccountperma8980"
remote_store = None
remote_store = stuett.ABSStore(
    container="hackathon-public-rw",
    prefix="annotations",
    account_name=account_name,
    account_key=None,
    blob_service_kwargs={"sas_token": token},
)


data = stuett.data.MHDSLRFilenames(
    store=store,
    start_time=pd.to_datetime("2017-01-01"),
    end_time=pd.to_datetime("2017-12-31"),
)()

# These are all the labels that are available to the tool
static_label_mapping = {
    "mountaineer": "Mountaineer",
    "headlamp": "Headlamp",
    "lens_flare": "Lens Flare",
    "ice_on_lens": "Ice on lens",
    "moon": "Moon (visible)",
    "fog": "Fog",
    "surface_water": "Surface Water",
    "bad_image_quality": "Bad Image Quality",

    "snow": "Snow",
    "snowfall": "Snowfall",
    "dark": "Dark",
    "moonlight": "Moonlight",
    "overcast": "Overcast",
    "cloudy": "Cloudy",
    "clear_day": "Clear day",
    "sunrise": "Sunrise",
    "sunset": "Sunset",
    "precipitation": "Precipitation",
    "rain": "Rain",
    "hail": "Hail",
}
reverse_static_label_mapping = {v: k for k, v in static_label_mapping.items()}


# TODO: generate a legend for each entry
# These are all the labels for which bounding boxes can be drawn
bb_label_mapping = {'#1f77b4':"Mountaineer",  # muted blue
                    '#ff7f0e':"Headlamp",  # safety orange
                    '#2ca02c':"Lens Flare",  # cooked asparagus green
                    '#d62728':"Ice on lens",  # brick red
                    '#9467bd':"Moon (visible)",  # muted purple
                    '#8c564b':"Fog",  # chestnut brown
                    '#e377c2':"Surface Water",  # raspberry yogurt pink
                    '#7f7f7f':"Bad Image Quality",  # middle gray
                    # '#bcbd22':"",  # curry yellow-green
                    # '#17becf':""  # blue-teal
                    }
bb_label_reverse_mapping = {v: k for k, v in bb_label_mapping.items()}
img_shape = (4288, 2848, 3)
if args.high_quality:
    img_downsampling = 4
else:
    img_downsampling = 1

app = dash.Dash(__name__)
server = app.server
app.config.suppress_callback_exceptions = True


def serve_layout():
    session_id = str(uuid.uuid4())

    layout = html.Div(
        [
            html.Div(
                [
                    html.Div(session_id, id="session-id", style={"display": "none"}),
                    html.Div([], id="storage", style={"display": "none"}),
                    html.H3("Permafrost Image Annotation"),
                    html.Div(
                        [
                            html.Div(
                                [
                                    dcc.DatePickerSingle(
                                        id="my-date-picker-single",
                                        min_date_allowed=stuett.to_datetime(
                                            "2017-01-01"
                                        ),
                                        max_date_allowed=stuett.to_datetime(
                                            "2017-12-31"
                                        ),
                                        initial_visible_month=None,
                                        date="2017-01-01",
                                        display_format="Y-MM-DD",
                                    )
                                ],
                                style={"width": "50%", "display": "inline-block"},
                            ),
                            html.Div(
                                id="date_indicator",
                                style={"width": "50%", "display": "inline-block"},
                            ),
                            html.Div([
                                dcc.Input(
                                    id="userid_input",
                                    placeholder="Your ID",
                                    type="number",
                                    value="",
                                    persistence=True,
                                ),
                            ],
                            style={"width": "50%", "display": "inline-block"},
                            ),

                        ]
                    ),
                    html.Div(
                        [
                            dash_canvas.DashCanvas(
                                id="canvas",
                                width=500,
                                tool="select",
                                lineWidth=2,
                                # json_data_in=json_template,
                                # filename=filename,
                                hide_buttons=["pencil", "line"],
                            ),
                        ],
                        style={"text-align": "center"},
                    ),
                ]
            ),
            html.Div(
                [
                    dcc.Markdown("Class names for bounding boxes:"),
                    dcc.Dropdown(
                        id="bb_label_dropdown",
                        options=[
                            {"label": bb_label_mapping[m], "value": m} for m in bb_label_mapping.keys()
                        ],
                        value="#1f77b4",
                    ),
                    dcc.Markdown("Class names for per image Labels:"),
                    dcc.Dropdown(
                        id="static_label_dropdown",
                        options=[
                            {"label": static_label_mapping[m], "value": m}
                            for m in static_label_mapping.keys()
                        ],
                        value=[],
                        multi=True,
                    ),
                    dcc.Store(id="index", data=0),
                ],
                className="six columns",
            ),
            dcc.Markdown(
                """Annotate by selecting per picture labels or draw bounding boxes with the rectangle tool
                   
                   Note: Rotating bounding boxes will result in incorrect labels."""
            ),
            
        ],
        style={"width": "50%"},  # Div
        className="row",
    )

    return layout


app.layout = serve_layout


@app.callback(
    [Output("canvas", "lineColor"), Output("bb_label_dropdown", "style")],
    [Input("bb_label_dropdown", "value")],
)
def update_output(value):
    return value, {"color": value}


@app.callback(
    Output("static_label_dropdown", "style"), [Input("static_label_dropdown", "value")]
)
def update_output(value):
    raise PreventUpdate


@app.callback(
    Output("my-date-picker-single", "date"),
    [Input("canvas", "prev_trigger"), Input("canvas", "next_trigger"), Input("userid_input","value")],
    [State("index", "data")],
)
def reduce_help(prev_trigger, next_trigger, userid_input_value, index):
    """ Triggers on click on the arrow buttons. Changes the date of the date selection tool,
        which triggers the loading of a new image.
    
    Arguments:
        prev_trigger {[type]} -- [description]
        next_trigger {[type]} -- [description]
        index {int} -- index of the image list dataframe
        
    Returns:
        datetime -- Date for the datepickersingle
    """
    global data

    ctx = dash.callback_context

    if not ctx.triggered:
        button_id = "No clicks yet"
        raise PreventUpdate
    else:
        button_id = ctx.triggered[0]["prop_id"]

    if index is None:
        index = 0

    if button_id == "canvas.prev_trigger":
        index = index - 2
        if index < 0:
            index = 0
    elif button_id == "canvas.next_trigger":
        index = index + 2
        if index >= len(data):
            index = len(data) - 1
    elif button_id == "userid_input.value":
        if userid_input_value is not None:
            max_user = 90
            index = int(len(data)/max_user * (userid_input_value-1))
            if index < 0:
                index = 0
            if index >= len(data):
                index = len(data) - 1
    else:
        raise PreventUpdate

    datetime = data.index[index]

    return datetime


def parse_labels(string, bb_label_mapping, static_label):
    """Returns array of rectangles geometry and their labels
    
    Arguments:
        string {str} -- JSON string
        bb_label_mapping {dict} -- Mapping from color to label
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

    for obj in data["objects"]:
        if obj["type"] == "image":
            # print('img')
            # print(obj['scaleX'],obj['scaleY'],obj['width'],obj['height'])
            # scale = obj['scaleX']
            img_width = obj["width"]
            img_height = obj["height"]

    for obj in data["objects"]:
        if obj["type"] == "rect":
            # scale_factor = obj['scaleX'] / scale
            try:
                label = bb_label_mapping[obj["stroke"]]
                label = reverse_static_label_mapping[label]
            except:
                raise warning.warn(f'Could not find bb_label_mapping for {obj["stroke"]}')
                continue
            item = [obj["right"], obj["bottom"], obj["left"], obj["top"]]

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
            props.append([None, None, None, None, item])
    # return (np.array(props))
    return props


@app.callback(
    [Output("canvas", "id")],
    [
        Input("canvas", "json_data_out"),
        Input("session-id", "children"),
        Input("static_label_dropdown", "value"),
        Input("userid_input", "value"),
    ],
    [State("index", "data")],
)
def get_json(json_data_out, session_id, static_label, user_id, index):
    global bb_label_mapping, data
    ctx = dash.callback_context

    if not ctx.triggered:
        raise PreventUpdate
    else:
        prop_id = ctx.triggered[0]["prop_id"]

    if prop_id != "canvas.json_data_out":
        raise PreventUpdate

    print("static", static_label)
    props = parse_labels(json_data_out, bb_label_mapping, static_label)

    datetime = data.index[index]

    df = pd.DataFrame(
        props, columns=["end_x", "end_y", "start_x", "start_y", "__target"]
    )
    df["start_time"] = datetime
    df["end_time"] = datetime

    df["__creation_time"] = stuett.to_datetime("now", utc=True)

    file_id = datetime.strftime("%Y%m%d_%H%M%S")
    to_csv(df, session_id, file_id, user_id)

    raise PreventUpdate


def to_csv(df, session_id, file_id=None, user_id=None):
    global local_store
    if file_id is None:
        file_id = uuid.uuid4()

    filename = session_id + "-" + str(user_id) + f"/{file_id}.csv"

    stuett.to_csv_with_store(local_store, filename, df, dict(index=False))
    if remote_store is not None:
        stuett.to_csv_with_store(remote_store, filename, df, dict(index=False))


def read_csv(session_id, file_id, user_id=None):
    global local_store
    filename = session_id + "-" + str(user_id) + f"/{file_id}.csv"
    return stuett.read_csv_with_store(local_store, filename)


@app.callback(
    [
        Output("canvas", "json_data_in"),
        Output("index", "data"),
        Output("date_indicator", "children"),
        Output("static_label_dropdown", "value"),
    ],
    [Input("my-date-picker-single", "date"), Input("session-id", "children"), Input("userid_input", "value")],
)
def update_output(date, session_id, user_id):
    """ The callback is used to load a new image when the date has changed.
        Date change can be triggered by the date selector box or indirectly by the arrow buttons,
        which change the date selector box.
        Besides loading images, this callback loads the annotation from the current user session
        and the annotations from the folder.
    
    Arguments:
        date {[type]} -- [description]
        session_id {[type]} -- [description]
    
    Raises:
        PreventUpdate: [description]
    
    Returns:
        [type] -- [description]
    """
    global data

    static_label = []
    info_box = "No info"

    if date is not None:
        index = data.index.get_loc(date, method="nearest")

        if isinstance(index, slice):
            index = index.start

        try:
            key = data.iloc[index]["filename"]
            img = imio.imread(io.BytesIO(store[key]))
        except Exception as e:
            print(e)
            info_box = "Error loading the image"
            img = np.zeros(img_shape, dtype="uint8")

        img = img[::img_downsampling, ::img_downsampling, :]
        image_content = array_to_data_url(img)

        # load data from index
        start_and_image = (
            '{"version":"2.4.3","objects":[{"type":"image","version":"2.4.3", "originX":"left","originY":"top","left":0,"top":0,"width":%d,"height":%d,"src":"%s"}'
            % (img.shape[1], img.shape[0], image_content)
        )

        # load local annotations if exist
        datetime = data.index[index]
        file_id = datetime.strftime("%Y%m%d_%H%M%S")
        try:
            df = read_csv(session_id, file_id, user_id)
        except Exception as e:
            print('Could not read annotation file',e)
            df = None

        # Load the annotations from the server
        try:
            global_df = stuett.read_csv_with_store(annotation_store, "annotations.csv")
            global_df = global_df[
                stuett.to_datetime(global_df["start_time"]) == datetime
            ]
            df = pd.concat([global_df, df])
        except Exception as e:
            print(e)

        # create the data that is passed to dash_canvas
        rectangles = [""]
        if df is not None:
            rects = []
            for i, row in df.iterrows():
                if row["__target"] in static_label_mapping:
                    if row["__target"] not in static_label:
                        static_label.append(row["__target"])

                left = row["start_x"]
                top = row["start_y"]
                right = row["end_x"]
                bottom = row["end_y"]

                if left is None or pd.isnull(left):
                    continue

                label = static_label_mapping[str(row["__target"])]
                if label not in bb_label_reverse_mapping:
                    continue
                stroke = bb_label_reverse_mapping[label]
                rect = (
                    ',{"type":"rect","version":"2.4.3","originX":"left","transparentCorners":false,"originY":"top","left":%f,"top":%f,"right":%f,"bottom":%f,"width":0,"height":0,"fill":"transparent","stroke":"%s","strokeWidth":2}'
                    % (left, top, right, bottom, stroke)
                )

                rects.append(rect)

            if rects:
                rectangles = rects

        end_objects = "]}"

        string = [start_and_image]
        string += rectangles
        string += [end_objects]

        json_data = "".join(string)

        return json_data, index, str(datetime), static_label

    raise PreventUpdate


if __name__ == "__main__":
    app.run_server(debug=False)
