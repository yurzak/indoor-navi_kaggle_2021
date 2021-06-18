import numpy as np
import plotly.graph_objs as go
#import plotly.graph_objects as go
from PIL import Image


def save_figure_to_html(fig, filename):
    fig.write_html(filename)

def save_figure_to_image(fig, filename):
    fig.write_image(filename)


def visualize_trajectory(trajectory, floor_plan_filename, width_meter, height_meter, is_multi=False, legends=None, title=None, mode='lines + markers + text', show=False):
    fig = go.Figure()
    rgb_colors = ["rgb(255,0,0)",
                  "rgb(0,0,255)",
                  "rgb(0,128,0)",
                  "rgb(0,206,209)",
                  "rgb(128,0,128)",
                  "rgb(255,20,147)",
                  "rgb(210,105,30)",
                  "rgb(255,215,0)",
                  "rgb(0,255,0)"]

    if is_multi:
        trajectories = trajectory
        line_colors = [rgb_colors[ind % len(rgb_colors)] for ind, _ in enumerate(trajectories)]
    else:
        trajectories = [trajectory]
        line_colors = ['rgb(100, 10, 100)']

    for ind_traj, traj in enumerate(trajectories):
        # add trajectory
        size_list = [6] * traj.shape[0]
        size_list[0] = 10
        size_list[-1] = 10

        color_list = [line_colors[ind_traj]] * traj.shape[0]
        color_list[0] = 'rgba(12, 5, 235, 1)'
        color_list[-1] = 'rgba(235, 5, 5, 1)'

        position_count = {}
        text_list = []
        for i in range(traj.shape[0]):
            if str(traj[i]) in position_count:
                position_count[str(traj[i])] += 1
            else:
                position_count[str(traj[i])] = 0
            text_list.append('        ' * position_count[str(traj[i])] + f'{i}')
        text_list[0] = 'S0'
        text_list[-1] = f'E{traj.shape[0] - 1}'

        fig.add_trace(
            go.Scattergl(
                x=traj[:, 0],
                y=traj[:, 1],
                mode=mode,
                marker=dict(size=size_list, color=color_list),
                line=dict(shape='linear', color=line_colors[ind_traj], width=2, dash='dot'),
                text=text_list,
                textposition="top center",
                name=legends[ind_traj] if is_multi else "trajectory",
                showlegend=True
            ))

    # add floor plan
    floor_plan = Image.open(floor_plan_filename)
    fig.update_layout(images=[
        go.layout.Image(
            source=floor_plan,
            xref="x",
            yref="y",
            x=0,
            y=height_meter,
            sizex=width_meter,
            sizey=height_meter,
            sizing="contain",
            opacity=1,
            layer="below",
        )
    ])

    # configure
    fig.update_xaxes(autorange=False, range=[0, width_meter])
    fig.update_yaxes(autorange=False, range=[0, height_meter], scaleanchor="x", scaleratio=1)
    fig.update_layout(
        title=go.layout.Title(
            text=title or "No title.",
            xref="paper",
            x=0,
        ),
        autosize=True,
        width=900,
        height=200 + 900 * height_meter / width_meter,
        template="plotly_white",
    )

    if show:
        fig.show()

    return fig


def visualize_heatmap(position, value, floor_plan_filename, width_meter, height_meter, colorbar_title="colorbar", title=None, show=False):
    fig = go.Figure()

    # add heat map
    fig.add_trace(
        go.Scatter(x=position[:, 0],
                   y=position[:, 1],
                   mode='markers',
                   marker=dict(size=7,
                               color=value,
                               colorbar=dict(title=colorbar_title),
                               colorscale="Rainbow"),
                   text=value,
                   name=title))

    # add floor plan
    floor_plan = Image.open(floor_plan_filename)
    fig.update_layout(images=[
        go.layout.Image(
            source=floor_plan,
            xref="x",
            yref="y",
            x=0,
            y=height_meter,
            sizex=width_meter,
            sizey=height_meter,
            sizing="contain",
            opacity=1,
            layer="below",
        )
    ])

    # configure
    fig.update_xaxes(autorange=False, range=[0, width_meter])
    fig.update_yaxes(autorange=False, range=[0, height_meter], scaleanchor="x", scaleratio=1)
    fig.update_layout(
        title=go.layout.Title(
            text=title or "No title.",
            xref="paper",
            x=0,
        ),
        autosize=True,
        width=900,
        height=200 + 900 * height_meter / width_meter,
        template="plotly_white",
    )

    if show:
        fig.show()

    return fig

if __name__ == "__main__":

    fix_path = "./data_in/test/030b3d94de8acae7c936563d.txt"


