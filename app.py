import pandas as pd
import os
import numpy as np
import plotly.graph_objects as go
import copy
import json
from datetime import datetime
import base64
import io

from dash import Dash, dcc, html, Input, Output, callback,  dash_table
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from dash import no_update


# files 
dirpath = os.getcwd()
input_dir = os.path.join(dirpath, "data")
file_list = os.listdir(input_dir)

file_dict = [{'label': "_".join(file.split("_")[:5]), "value": file} for file in file_list]
q_dict = [{'label': '1', 'value': 1},{'label': '2', 'value': 2},{'label': '3', 'value': 3},{'label': '4', 'value': 4}]


# storing dict 
annotations = {}
for file in file_list:
    annotations[file] = {}
    for num in ['1', '2', '3', '4']:
        annotations[file][num] = {"angle": None, "rad": None, "heigt": None, "center": None, "end_p": None, "center_new": None, "last_save": None}

# functions 
def filter_df(filepath, q):
    file = np.loadtxt(filepath, delimiter = ",")
    idx = np.where(file[:, 5] == q)[0]
    data = file[idx, :]
    return data

def get_raw(data):
    raw = data[:, [0, 1, 2]]
    return raw

def get_tip(raw, height):
    # tidx = np.argmin(udder[:, 2])
    tidx = np.argmin(raw[:, 2])
    tip = raw[tidx, :]
    p2 = tip.copy()
    p2[2] = p2[2] + height
    point_dict = {"p1": tip, "p2":p2}
    return point_dict

def blank_fig():
    fig = go.Figure(go.Scatter3d(x=[], y = [], z=[]))
    fig.update_layout(paper_bgcolor="black")
    fig.update_layout(legend_font_color="white")
    fig.update_scenes(xaxis_visible=False, yaxis_visible=False,zaxis_visible=False)
    fig.update_layout(legend_font_color="white", width=1000, height=800)
    return fig

def get_cilinder(center, cil_height, rad):
    z_start = center[2]
    z_end = center[2] + cil_height
    cil_height_list = np.linspace(z_start, z_end, 100) 
    angles = np.linspace(0, 2*np.pi, 100) 
    xlocs = center[0] + np.cos(angles)*rad
    ylocs = center[1] + np.sin(angles)*rad
    for i, height in enumerate(cil_height_list):
        zlocs = np.repeat(height, len(angles))
        base = np.column_stack([xlocs, ylocs, zlocs])
        if i == 0:
            cilinder_points = base
        else:
            cilinder_points = np.vstack([cilinder_points, base])
    return cilinder_points

def annotate_teat(raw, lines, top, base, cilinder_points): 
    fig =  go.Figure(data=[go.Scatter3d(x = raw[:,0], y = raw[:,1], z=raw[:,2], mode='markers', marker=dict(size=2, color="blue", opacity = 1), name = "udder")])
    fig.add_trace(go.Scatter3d(x = lines[:-1,0], y = lines[:-1,1], z = lines[:-1,2], mode='markers', marker=dict(color="gray", size = 3, opacity = 1), name = "line", showlegend=False))
    fig.add_trace(go.Scatter3d(x = lines[:-1,0], y = lines[:-1,1], z = lines[:-1,2], mode='lines', marker=dict(color="gray", size = 3, opacity = 1), name = "line"))
    fig.add_trace(go.Scatter3d(x = lines[-2:,0], y = lines[-2:,1], z = lines[-2:,2], mode='markers', marker=dict(color="red", size = 5, opacity = 1), name = "length", showlegend=False))
    fig.add_trace(go.Scatter3d(x = lines[-2:,0], y = lines[-2:,1], z = lines[-2:,2], mode='lines', marker=dict(color="red", size = 5, opacity = 1), name = "length"))
    fig.add_trace(go.Scatter3d(x = cilinder_points[:,0], y = cilinder_points[:,1], z = cilinder_points[:,2], mode='markers', marker=dict(size=0.5, color="gray", opacity = 0.2), name = "cilinder"))
    fig.add_trace(go.Scatter3d(x = base[:,0], y = base[:,1], z = base[:,2], mode='markers', marker=dict(size=1, color="gray", opacity = 0.5), name = "cilinder", showlegend=False))
    fig.add_trace(go.Scatter3d(x = top[:,0], y = top[:,1], z = top[:,2], mode='markers', marker=dict(size=1, color="gray", opacity = 0.5), name = "cilinder", showlegend=False))
    # axis guides
    center_p = lines[0, :].copy()
    x_line = np.vstack([center_p, center_p])
    x_line[0, 0] = x_line[0, 0] + 0.005
    x_line[1, 0] = x_line[1, 0] - 0.005
    y_line = np.vstack([center_p, center_p])
    y_line[0, 1] = y_line[0, 1] + 0.005
    y_line[1, 1] = y_line[1, 1] - 0.005
    z_line = np.vstack([center_p, center_p])
    z_line[0, 2] = z_line[0, 2] + 0.005
    z_line[1, 2] = z_line[1, 2] - 0.005
    fig.add_trace(go.Scatter3d(x = x_line[:,0], y = x_line[:,1], z = x_line[:,2], mode="lines+text", marker=dict(color="white", size = 3, opacity = 1), text = ["X", ""], name = "guide"))
    fig.add_trace(go.Scatter3d(x = y_line[:,0], y = y_line[:,1], z = y_line[:,2], mode="lines+text", marker=dict(color="white", size = 3, opacity = 1), text = ["Y", ""], name = "guide", showlegend=False))
    fig.add_trace(go.Scatter3d(x = z_line[:,0], y = z_line[:,1], z = z_line[:,2], mode="lines+text", marker=dict(color="white", size = 3, opacity = 1), text = ["Z", ""], name = "guide", showlegend=False))
    fig.update_layout(scene_aspectmode='data')
    fig.update_layout(paper_bgcolor="black", font_color = "white", plot_bgcolor = "white", width=1000, height=800)
    fig.update_scenes(xaxis_visible=False, yaxis_visible=False,zaxis_visible=False)
    return fig 

def get_circle_point(center, cil_height, rad, angle):
    angle_rad = np.radians(angle)
    xloc = center[0] + np.cos(angle_rad)*rad
    yloc = center[1] + np.sin(angle_rad)*rad
    zloc = center[2] + cil_height
    point = np.array([xloc, yloc, zloc])
    return point

def teat_length(start, end):
    length = np.absolute(np.linalg.norm(start - end))
    return length

def center_tolist(center_dict):
    center_array = np.array([center_dict['x'],center_dict['y'], center_dict['z']])
    return center_array

def center_todict(center_array):
    center_dict = {'x': center_array[0], 'y':center_array[1], 'z':center_array[2]}
    return center_dict


# sytle dictionaries
# style
# the style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "40rem",
    "padding": "5rem 5rem",
}

# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
    "textAlign": "center",
    "margin-left": "40rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

MENU_STYLE = {
    'backgroundColor': 'black',
    'color': 'white',
}

PLOT_STYLE = {
    'display': 'block',
    'margin-left': 'auto',
    'margin-right': 'auto',
}
BUTTON_STYLE = {
    'backgroundColor': 'black',
    'border-style': 'solid',
    'border-color': 'white',
    'width': '150px',
}

TEXTBOX_STYLE = {
    'color': 'white',
}

# app layout
defaults = {'image' : '1023_20231117_124217_frame_100_udder.csv', 'q':1, 'angle':0, 'height':0.05, 'rad':0.02}

# instructions
info = html.Div(
    [
        html.H2("Teat length labeler", className ="display-4"),
        html.Hr(),
        html.P(["Instructions:", html.Br(),
                "1) Choose a cow and quarter", html.Br(),
                "2) Adjust the angle, height, and center", html.Br(),
                "3) Click submit to save"], className ="lead"),
        
        html.Label("Cow ID:"),
        dcc.Dropdown(id='cows-dpdn',options= file_dict, value = defaults["image"], style = MENU_STYLE),])

buttons = dcc.RadioItems(id = 'q-btn', options = q_dict, value = defaults["q"])

slider_angles = dcc.Slider(0, 360, value = defaults['angle'],
    marks={
        0: {'label': '0°', 'style': {'color': '#FFFFFF1'}},
        360: {'label': '360°', 'style': {'color': '#FFFFFF'}}
    },included=False, id='slider_angles')

slider_rad = dcc.Slider(0, 0.1, value = defaults["rad"],
    marks={
        0: {'label': '0', 'style': {'color': '#FFFFFF1'}},
        0.1: {'label': '0.1', 'style': {'color': '#FFFFFF'}}
    },included=False, id='slider_rad')

slider_height = dcc.Slider(0, 0.1, value = defaults["height"],
    marks={
        0: {'label': '0', 'style': {'color': '#FFFFFF1'}},
        0.1: {'label': '0.1', 'style': {'color': '#FFFFFF'}}
    },included = False, id ='slider_height')


buttons_z = html.Div([
    html.Label("Z"),
    dbc.Button('\u21E7', id = 'z_up', n_clicks = 0, className = "me-1", color = "black"),
    dbc.Button('\u21E9', id = 'z_down', n_clicks = 0, className = "me-1", color = "black"),
], id ='buttons_z')

buttons_x = html.Div([
    html.Label("X"),
    dbc.Button('\u21E6', id='x_left', n_clicks=0, className = "me-1", color = "black"),
    dbc.Button('\u21E8', id='x_right', n_clicks=0, className = "me-1", color = "black"),
], id = 'buttons_x')

buttons_y = html.Div([
    html.Label("Y"),
    dbc.Button('\u21E6', id='y_left', n_clicks=0, className = "me-1", color = "black"),
    dbc.Button('\u21E8', id='y_right', n_clicks=0,  className = "me-1", color = "black"),
], id='buttons_y')

buttons_reset = html.Div([
    dbc.Button('reset center', id='reset_center', n_clicks=0, className = "me-1", style = BUTTON_STYLE),
], id='reset_button')


buttons_submit = html.Div([
    dbc.Button('submit', id='submit', n_clicks=0, className = "me-1", style = BUTTON_STYLE),
], id = "buttons_submit")

submit_out = html.Div(id='submit_out',  style={'whiteSpace': 'pre-line'})

teat_len = html.Div(
    [
        html.P("Teat length:", className ="lead"),
         html.Div(id = "teat_len", style={'whiteSpace': 'pre-line'}),
    ], id = "teat_len_box")

download = html.Div([
    dbc.Button("Download", id="download_button", className = "me-1", style = BUTTON_STYLE),
    dcc.Download(id="download_json")
])

upload = html.Div([
    dcc.Upload(dbc.Button('Upload',  className = "me-1", style = BUTTON_STYLE), id = 'upload_json')])

# app layout
sidebar = html.Div(
    [html.Div([
        dbc.Row([info]),
    ]),
     html.Div([
        html.Label("Quarter:"),
        dbc.Row(dbc.Col(buttons, width=15)),
        html.Label("Angle:"),
        dbc.Row([slider_angles]),
        html.Label("Radius:"),
        dbc.Row([slider_rad]),
        html.Label("Height:"),
        dbc.Row([slider_height]),
     ]),
     html.Div([
        html.Label("Center:"),
        dbc.Row([
            dbc.Col([buttons_z, buttons_x, buttons_y]), 
            dbc.Col([
                dbc.Row(buttons_reset),
                dbc.Row([buttons_submit, submit_out])
            ]),
            dbc.Col([teat_len]),
        ]),
         dbc.Row([download, upload]),
     ])
    ], style=SIDEBAR_STYLE)

content = html.Div([
    dbc.Row([dcc.Graph(id='graph', figure = blank_fig(), style = PLOT_STYLE)]),],
    id="page-content", style=CONTENT_STYLE)

app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
app.layout = html.Div([dcc.Location(id="url"), sidebar, content])

# app fuctionality 
# last visited =====================
last_cow = defaults["image"]
last_q = defaults["q"]


# reset defaults =====================
@app.callback(
    Output('q-btn', 'value'),
    Output('slider_angles', 'value'), 
    Output('slider_rad', 'value'),
    Output('slider_height', 'value'), 
    Output('x_left', 'n_clicks', allow_duplicate=True),
    Output('x_right', 'n_clicks', allow_duplicate=True),
    Output('y_left', 'n_clicks', allow_duplicate=True),
    Output('y_right', 'n_clicks',allow_duplicate=True),
    Output('z_up', 'n_clicks', allow_duplicate=True),
    Output('z_down', 'n_clicks', allow_duplicate=True),
    Output('reset_center', 'n_clicks',allow_duplicate=True),
    Output('submit', 'n_clicks', allow_duplicate=True),
    Output("submit_out", "children", allow_duplicate=True),
    Input('cows-dpdn', 'value'), 
    Input('q-btn', 'value'),
    prevent_initial_call=True
)
def reset_defaults(file_name, q):
    global last_cow, last_q
    if (last_cow != file_name) | (last_q != q):
        # print("new_cow")
        last_cow = file_name
        last_q = q
        def_q = defaults["q"] if last_cow != file_name else q
        def_angle = defaults["angle"]
        def_rad = defaults["rad"]
        def_height = defaults["height"]
        return def_q, def_angle, def_rad, def_height, 0, 0, 0, 0, 0, 0, 0, 0, ""
    else:
        return no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update

# center buttons =====================
@app.callback(
    Output('x_left', 'n_clicks'),
    Output('x_right', 'n_clicks'),
    Output('y_left', 'n_clicks'),
    Output('y_right', 'n_clicks'),
    Output('z_up', 'n_clicks'),
    Output('z_down', 'n_clicks'),
    Output('reset_center', 'n_clicks'),
    Input('reset_center', 'n_clicks'),
    Input('cows-dpdn', 'value'), 
    Input('q-btn', 'value'),
)
def reset_center(n_clicks, file_name, q):
    global annotations
    if n_clicks > 0:
        # reset center to none
        annotations[file_name][str(q)]["center"] = None
        print(n_clicks)
        return 0, 0, 0, 0, 0, 0, 0
    else:
        return no_update, no_update, no_update, no_update, no_update, no_update, no_update


# builc plot =====================
@app.callback(
    Output("graph", "figure"),
    Output('teat_len', 'children'),
    Input('cows-dpdn', 'value'), 
    Input('q-btn', 'value'),
    Input('slider_angles', 'value'), 
    Input('slider_rad', 'value'),
    Input('slider_height', 'value'), 
    Input('x_left', 'n_clicks'),
    Input('x_right', 'n_clicks'),
    Input('y_left', 'n_clicks'),
    Input('y_right', 'n_clicks'),
    Input('z_up', 'n_clicks'),
    Input('z_down', 'n_clicks'),
)
def get_frames(file_name, q, angle, rad, cil_height, x_left, x_right, y_left, y_right, z_up, z_down):
    global input_dir, annotations
    filepath = os.path.join(input_dir, file_name)
    data = filter_df(filepath, 1)
    raw = get_raw(data)
    point_dict = get_tip(raw, cil_height)
    new_center = annotations[file_name][str(q)]["center"]
    center = point_dict["p1"].copy()
    end_point = point_dict["p2"].copy()
    if new_center is None:
        new_center = center
        annotations[file_name][str(q)]["center"] = new_center.copy()
    x_shift = (x_left - x_right)/1000
    y_shift = (y_left - y_right)/1000
    z_shift = (z_up - z_down)/1000
    new_center[0] = center[0] + x_shift
    new_center[1] = center[1] + y_shift
    new_center[2] = center[2] + z_shift
    end_point = new_center.copy()
    end_point[2] = end_point[2] + cil_height
    cilinder_points = get_cilinder(new_center, cil_height, rad)
    base = cilinder_points[0:100, :]
    top = cilinder_points[-100:-1, :]
    circle_point = get_circle_point(new_center, cil_height, rad, angle)
    lines = np.vstack([new_center, end_point, circle_point, new_center])
    fig = annotate_teat(raw, lines, top, base, cilinder_points)
    fig.update_layout(uirevision=True)
    # save selections
    annotations[file_name][str(q)]["angle"] = angle
    annotations[file_name][str(q)]["rad"] = rad
    annotations[file_name][str(q)]["heigt"] = cil_height
    annotations[file_name][str(q)]["end_p"] =  end_point
    annotations[file_name][str(q)]["center_new"] = new_center
    # calculate distance
    teat_len = np.round(teat_length(new_center, end_point)*1000, 2)
    # print(teat_len)
    return fig, f"{teat_len} mm"


# save progress =====================
@app.callback(
    Output('submit', 'n_clicks', allow_duplicate=True),
    Output('submit_out', 'children', allow_duplicate=True),
    Input('submit', 'n_clicks'),
    Input('cows-dpdn', 'value'), 
    Input('q-btn', 'value'),
    prevent_initial_call=True
)
def submit_save(submit, file_name, q):
    global annotations
    # print(submit)
    if submit > 0:
        current_datetime = datetime.now()
        formatted_time = current_datetime.strftime("%H:%M:%S")
        annotations[file_name][str(q)]['last_save'] = formatted_time
        return 0, f"Last saved: {formatted_time}"
    else:
        return no_update, no_update
    
@app.callback(
    Output('submit_out', 'children'),
    Input('cows-dpdn', 'value'), 
    Input('q-btn', 'value'),
)
def submit_print(file_name, q):
    global annotations
    last_saved = annotations[file_name][str(q)]['last_save']
    if last_saved is None:
        return no_update
    else:
        return f"Last saved: {last_saved}"

# download annotations ======================
@callback(
    Output("download_json", "data"),
    Input("download_button", "n_clicks"),
    prevent_initial_call=True,
)
def func(n_clicks):
    global annotations
    annotations_out = copy.deepcopy(annotations)
    for key_file in annotations_out.keys():
            for q in ['1', '2', '3', '4']:
                center_array = annotations[key_file][q]["center"]
                newcenter_array = annotations[key_file][q]["center_new"]
                end_center = annotations[key_file][q]["end_p"]
                if center_array is not None:
                    annotations_out[key_file][q]["center"] = center_todict(center_array)
                if newcenter_array is not None:
                    annotations_out[key_file][q]["center_new"] = center_todict(newcenter_array)
                if end_center is not None:
                    annotations_out[key_file][q]["end_p"] = center_todict(end_center)
    annotations_text = json.dumps(annotations_out)
    return dict(content=annotations_text, filename="annotations_teat_length.txt")

@callback(
    Input('upload_json', 'contents'),
    State('upload_json', 'filename'))
def update_output(contents, filename):
    global annotations
    file_type = filename.split(".")[1]
    if "txt" == file_type:
        if contents is not None:
            content_type, content_string = contents.split(',')
            decoded_bytes = base64.b64decode(content_string)  
            decoded_text = decoded_bytes.decode('utf-8')
            contents_dict = json.loads(decoded_text)
            # contents_dict_out = copy.deepcopy(contents_dict)
            annotations_out = copy.deepcopy(contents_dict)
            for key_file in annotations_out.keys():
                for q in ['1', '2', '3', '4']:
                    center_array = contents_dict[key_file][q]["center"]
                    newcenter_array = contents_dict[key_file][q]["center_new"]
                    end_center = contents_dict[key_file][q]["end_p"]
                    if center_array is not None:
                        annotations_out[key_file][q]["center"] = center_tolist(center_array)
                    if newcenter_array is not None:
                        annotations_out[key_file][q]["center_new"] = center_tolist(newcenter_array)
                    if end_center is not None:
                        annotations_out[key_file][q]["end_p"] = center_tolist(end_center)
            annotations = copy.deepcopy(annotations_out)


if __name__ == '__main__':
    app.run()

