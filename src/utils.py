import plotly.graph_objects as go
import numpy as np
import random
import torch
import os
import plotly.io as pio
pio.kaleido.scope.mathjax = None

def seed_everything(seed: int) -> None:
  # import torch
  # import random
  # import os
  # import numpy as np
  """Seed everything
  Args:
    seed (int): Seed value

  """
  random.seed(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = True


def visualize_logitlens_from_item(item, save_file):
    layer_outputs = item.get("layer_outputs", [])
    if not layer_outputs:
        print("No layer_outputs found in item.")
        return

    # データを準備
    layers = [entry["layer"] for entry in layer_outputs]
    n_tokens = [entry["n_answer_token"] for entry in layer_outputs]
    n_probs = [entry["n_answer_prob"] for entry in layer_outputs]

    # z（色）と text（トークン）を1列構造に
    z = [[prob] for prob in n_probs]
    text = [[tok] for tok in n_tokens]

    # 16:9
    fig_width = 16
    fig_height = 9
    fig = go.Figure(
        layout=go.Layout(
            width=fig_width * 100,
            height=fig_height * 100,
            margin=dict(l=0, r=0, t=0, b=0),
        )
    )

    fig = go.Figure(data=go.Heatmap(
        z=z[::-1],
        text=text[::-1],
        x=["Answer"],
        y=layers[::-1],
        colorscale="RdBu_r",  # ← "_r" を付けて反転！
        zauto=False,
        zmin=0,
        zmax=1,
        zmid=0.5,
        colorbar=dict(
            title="Prob.",
            tickvals=[0, 0.5, 1],
            ticktext=["0", "0.5", "1"]
        ),
        texttemplate="%{text}"
    ))

    fig.update_layout(
        title="",
        xaxis_title="",
        yaxis_title="Layer",
    )

    # フォントを大きく
    fig.update_layout(
        font=dict(
            family="Arial",
            size=18,
            color="black"
        )
    )
    
    fig.write_image(save_file)
    save_file = save_file.replace(".png", ".pdf")
    fig.write_image(save_file)
    print(f"Logit lens visualization saved to {save_file}")
