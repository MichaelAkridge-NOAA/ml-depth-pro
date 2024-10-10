import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import numpy as np
import plotly.graph_objects as go
from PIL import Image
import os
import base64

# Paths for RGB frames and depth files
rgb_folder = "./images/png"
depth_folder = "./images/npz"

# Function to load frame files from the folders
def load_frame_files(rgb_folder, depth_folder):
    rgb_frames = sorted([os.path.join(rgb_folder, f) for f in os.listdir(rgb_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])
    depth_files = sorted([os.path.join(depth_folder, f) for f in os.listdir(depth_folder) if f.endswith('.npz')])
    
    if len(rgb_frames) != len(depth_files):
        raise ValueError("The number of RGB frames and depth files must be the same.")
    
    return rgb_frames, depth_files

# Function to create a 3D Plotly figure with the updated camera position and title
def create_single_frame(rgb_frame, depth_frame, full_screen=False):
    depth_data = np.load(depth_frame)['depth']
    image_rgb = np.array(Image.open(rgb_frame).convert("RGB"))

    x = np.linspace(0, depth_data.shape[1] - 1, depth_data.shape[1])
    y = np.linspace(0, depth_data.shape[0] - 1, depth_data.shape[0])
    x, y = np.meshgrid(x, y)
    image_rgb_normalized = image_rgb / 255.0
    surfacecolor = image_rgb_normalized[..., 0]

    fig = go.Figure(data=[go.Surface(z=depth_data, x=x, y=y, surfacecolor=surfacecolor, colorscale='Viridis', showscale=True)])

    # Updated camera position
    camera_position = {
        'up': {'x': 0, 'y': 0, 'z': 1},
        'center': {'x': 0, 'y': 0, 'z': 0},
        'eye': {'x': 1.8213674411967045e-18, 'y': -1.0126816117327627e-16, 'z': -1.6541020496973788},
        'projection': {'type': 'perspective'}
    }

    layout_params = {
        'scene': dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Depth',
            camera=camera_position
        ),
        'title': {'text': "3D Depth Explorer", 'x': 0.5, 'xanchor': 'center'},
        'plot_bgcolor': "#f4f4f4",
        'paper_bgcolor': "#f4f4f4",
        'margin': dict(l=0, r=0, b=0, t=50),
    }

    if full_screen:
        layout_params['height'] = 900
        layout_params['width'] = 1200
    else:
        layout_params['height'] = 700
        layout_params['width'] = 700

    fig.update_layout(**layout_params)

    return fig

# Function to display the original RGB image as a base64 string
def get_image_as_base64(image_path):
    with open(image_path, 'rb') as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('ascii')
    return f'data:image/png;base64,{encoded_image}'

# Load RGB and depth files
rgb_frames, depth_files = load_frame_files(rgb_folder, depth_folder)

# Initialize the Dash app with a custom title
app = dash.Dash(__name__, title='Depth Estimation Viewer')

# App layout with project overview and contact
app.layout = html.Div(style={'backgroundColor': '#f4f4f4', 'fontFamily': 'Arial, sans-serif'}, children=[
    html.Div([
        html.H1("Depth Estimation using Machine Learning on Underwater Marine Imagery", style={'textAlign': 'center', 'color': '#4A90E2'}),
        html.P([
            "Visualization of high-resolution depth maps generated from underwater marine imagery using Apple’s ",
            html.A("ML-Depth Pro", href="https://github.com/apple/ml-depth-pro", target="_blank", style={'color': '#4A90E2'}),
            " machine learning model. This is a test of Depth Pro's performance on underwater footage to evaluate its depth estimation in marine environments. ",
            "No adjustments or fine tuning have been made at this time. Future exploration will include integrating known variables, such as camera focal length, ",
            "and making adjustments for factors like water clarity."
        ], style={'textAlign': 'center', 'color': '#333', 'fontSize': '16px'}),

        # Combined GitHub repository link and contact information
        html.P([
            "For more information, view this project’s source: ",
            html.A("GitHub Repository", href="https://github.com/MichaelAkridge-NOAA/ml-depth-pro", target="_blank", style={'color': '#4A90E2'}),
            ". Contact: ",
            html.A("michael.akridge@noaa.gov", href="mailto:michael.akridge@noaa.gov", style={'color': '#4A90E2'}),
            "."
        ], style={'textAlign': 'center', 'color': '#333', 'fontSize': '16px'})
    ], style={'padding': '20px', 'backgroundColor': '#FFFFFF', 'borderBottom': '2px solid #4A90E2'}),

    # Toggle between views
    html.Div([
        dcc.RadioItems(
            id='view-toggle',
            options=[
                {'label': 'Split View (3D + Image)', 'value': 'split'},
                {'label': '3D Viewer Only', 'value': '3d'},
                {'label': 'Animated Depth Visualization', 'value': 'gif'}
            ],
            value='split',
            labelStyle={'display': 'inline-block', 'marginRight': '10px'}
        )
    ], style={'textAlign': 'center', 'padding': '20px'}),

    html.Div([
        dcc.Slider(
            id='frame-slider',
            min=0,
            max=len(rgb_frames) - 1,
            value=len(rgb_frames) // 2,
            marks={i: f"Frame {i}" for i in range(0, len(rgb_frames), 1)},
            step=1,
            updatemode='drag',
        ),
    ], style={'padding': '20px', 'width': '80%', 'margin': '0 auto'}),

    html.Div(id='output-view', style={'display': 'flex', 'justifyContent': 'center'})
])

# Callback to update the view based on the toggle and selected frame
@app.callback(
    Output('output-view', 'children'),
    [Input('view-toggle', 'value'),
     Input('frame-slider', 'value')]
)
def update_output(view_type, selected_frame):
    rgb_frame = rgb_frames[selected_frame]
    depth_file = depth_files[selected_frame]

    if view_type == '3d':
        figure = create_single_frame(rgb_frame, depth_file, full_screen=True)
        return html.Div([
            dcc.Graph(figure=figure, style={'border': '2px solid #4A90E2'}),
        ], style={'width': '100%', 'padding': '20px', 'backgroundColor': '#FFFFFF', 'boxShadow': '0 4px 8px rgba(0, 0, 0, 0.1)', 'margin': '20px'})
    elif view_type == 'gif':
        gif_path = "./images/ml-depth-pro-MOUSS-test.gif"  # Path to your GIF file
        gif_src = get_image_as_base64(gif_path)  # Convert to base64 for HTML rendering
        return html.Div([
            html.Img(src=gif_src, style={'width': '80%', 'border': '2px solid #4A90E2'}),
        ], style={'textAlign': 'center', 'padding': '20px', 'backgroundColor': '#FFFFFF', 'boxShadow': '0 4px 8px rgba(0, 0, 0, 0.1)', 'margin': '20px'})
    else:
        figure = create_single_frame(rgb_frame, depth_file)
        image_src = get_image_as_base64(rgb_frame)
        return html.Div([
            html.Div([
                html.H3("Original Frame", style={'textAlign': 'center', 'color': '#4A90E2'}),
                html.Img(src=image_src, style={'width': '100%', 'border': '2px solid #4A90E2'}),
            ], style={'width': '45%', 'display': 'inline-block', 'padding': '20px', 'backgroundColor': '#FFFFFF', 'boxShadow': '0 4px 8px rgba(0, 0, 0, 0.1)', 'margin': '20px'}),

            html.Div([
                dcc.Graph(figure=figure, style={'border': '2px solid #4A90E2'}),
            ], style={'width': '45%', 'display': 'inline-block', 'padding': '20px', 'backgroundColor': '#FFFFFF', 'boxShadow': '0 4px 8px rgba(0, 0, 0, 0.1)', 'margin': '20px'}),
        ], style={'display': 'flex', 'justifyContent': 'center'})

# Run the app
if __name__ == '__main__':
    app.run_server(debug=False)
