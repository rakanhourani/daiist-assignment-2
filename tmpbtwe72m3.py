# 1. Load the committed weights and the results saved by train.ipynb (nothing is trained here)
import json
import os
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from ultralytics import YOLO

ART = Path("artifacts")
summary = json.loads((ART / "metrics.json").read_text())
scores = pd.read_csv(ART / "scores.csv")
sweep = pd.read_csv(ART / "cost_sweep.csv")
per_class = pd.read_csv(ART / "per_class.csv")
model = YOLO("weights/fruit_yolo11n_cls.pt")

CLASSES, ROTTEN = summary["classes"], summary["rotten_classes"]
A = summary["assumptions"]
COST, BASE_RATE = A["cost"], A["base_rate"]
THRESHOLD, LOW, HIGH = summary["threshold"], summary["band"]["low"], summary["band"]["high"]
SPLITS = {"Test": "test", "Validation": "valid", "External photos (other dataset)": "external"}

# 2. Cost of a pass / review / fail band (same formula as in training)
def band_cost(d, low, high):
    r, f = d.p_rotten[d.is_rotten == 1].to_numpy(), d.p_rotten[d.is_rotten == 0].to_numpy()
    def shares(p):
        return (p < low).mean(), ((p >= low) & (p < high)).mean(), (p >= high).mean()
    r_pass, r_review, r_fail = shares(r)
    f_pass, f_review, f_fail = shares(f)
    review_share = BASE_RATE * r_review + (1 - BASE_RATE) * f_review
    per_fruit = (BASE_RATE * (r_pass * COST["missed_rotten"] + (r_review + r_fail) * COST["caught_rotten"])
                 + (1 - BASE_RATE) * (f_fail * COST["discarded_good"] + (f_pass + f_review) * COST["passed_good"])
                 + review_share * A["review_cost"])
    return per_fruit, review_share


def decide(p, low, high):
    return "pass" if p < low else ("human review" if p < high else "fail")

# 3. Upload a fruit: class probabilities and the decision at the current thresholds
def check_fruit(image, low, high):
    if image is None:
        return None, "Upload a photo of an apple, banana or orange."
    probs = model.predict(image, verbose=False)[0].probs.data.cpu().numpy()
    p_rotten = float(sum(q for c, q in zip(CLASSES, probs) if c in ROTTEN))
    decision = decide(p_rotten, low, high)
    return ({c: float(q) for c, q in zip(CLASSES, probs)},
            f"### Probability rotten: {p_rotten:.1%} → **{decision.upper()}**\n"
            f"(pass below {low:.2f}, human review from {low:.2f} to {high:.2f}, fail from {high:.2f})")

# 4. Thresholds: confusion matrix, cost per 1,000 fruit, review share and staff, cost curve
def threshold_view(split_label, low, high):
    low, high = min(low, high), max(low, high)
    d = scores[scores.split == SPLITS[split_label]]
    decision = d.p_rotten.apply(decide, args=(low, high))
    outcomes = ["pass", "human review", "fail"]
    table = pd.crosstab(d.is_rotten.map({1: "actually rotten", 0: "actually fresh"}), decision)
    table = table.reindex(index=["actually rotten", "actually fresh"], columns=outcomes, fill_value=0)

    matrix = go.Figure(go.Heatmap(z=table.values, x=outcomes, y=table.index, colorscale="Blues",
                                  showscale=False, text=table.values, texttemplate="%{text}"))
    matrix.update_layout(title=f"{split_label} images by decision", height=300, template="plotly_white",
                         margin=dict(l=110, r=20, t=50, b=40))
    matrix.update_yaxes(autorange="reversed")

    per_fruit, share = band_cost(d, low, high)
    per_day = A["fruit_per_day"] * share
    people = per_day * A["seconds_per_review"] / 3600 / A["shift_hours"]
    rows = [("Do nothing (pack everything)", *band_cost(d, 1.01, 1.01)),
            ("A person checks every fruit", *band_cost(d, -0.01, 1.01)),
            (f"Model alone (threshold {THRESHOLD})", *band_cost(d, THRESHOLD, THRESHOLD)),
            (f"Your setting ({low:.2f} to {high:.2f})", per_fruit, share)]
    policies = pd.DataFrame([{"policy": name, "€ per 1,000 fruit": round(1000 * c, 2),
                              "€ per day": round(A["fruit_per_day"] * c), "share reviewed": f"{s:.1%}",
                              "people needed": round(A["fruit_per_day"] * s * A["seconds_per_review"] / 3600
                                                     / A["shift_hours"], 2)} for name, c, s in rows])
    text = (f"### Expected cost: €{1000 * per_fruit:,.2f} per 1,000 fruit\n"
            f"**{share:.1%}** of fruit sent to a person = {per_day:,.0f} checks a day = "
            f"**{people:.2f} people** (available: {A['reviewers_available']}).  \n"
            f"Assumes {BASE_RATE:.0%} of fruit is rotten, {A['fruit_per_day']:,} fruit a day, "
            f"{A['seconds_per_review']} s per check, missed rotten €{COST['missed_rotten']}, "
            f"discarded good €{COST['discarded_good']}.")

    curve = go.Figure()
    for col, name, dash in [("cost_per_1000_validation", "validation", "solid"), ("cost_per_1000_test", "test", "dot"),
                            ("cost_per_1000_external", "external photos", "dash")]:
        curve.add_trace(go.Scatter(x=sweep.threshold, y=sweep[col], name=name, line=dict(dash=dash)))
    chosen = sweep.loc[np.isclose(sweep.threshold, THRESHOLD), "cost_per_1000_validation"].iloc[0]
    curve.add_trace(go.Scatter(x=[THRESHOLD], y=[chosen], mode="markers+text", name="chosen threshold",
                               marker=dict(size=12, color="red"), text=[f" {THRESHOLD}"], textposition="top right"))
    curve.add_vline(x=low, line_dash="dash", line_color="grey", annotation_text="pass | review")
    curve.add_vline(x=high, line_dash="dash", line_color="grey", annotation_text="review | fail")
    curve.update_layout(title="Cost per 1,000 fruit if the model decides alone", template="plotly_white",
                        xaxis_title="threshold on probability rotten", yaxis_title="€ per 1,000 fruit",
                        height=400, legend=dict(orientation="h", y=-0.2))
    return matrix, text, policies, curve

# 5. Per-class results on the test split, for the model and both baselines
overall = pd.DataFrame(summary["overall"]).round(3)
per_class_table = per_class.rename(columns={"support": "test images"}).round(3)


def confusion_plot(model_name):
    z = np.array(summary["confusion"][model_name])
    fig = go.Figure(go.Heatmap(z=z, x=summary["pred_labels"], y=CLASSES, colorscale="Blues", showscale=False,
                               text=z, texttemplate="%{text}"))
    fig.update_layout(title=f"Confusion matrix: {model_name}", template="plotly_white", height=450,
                      xaxis_title="predicted", yaxis_title="actual")
    fig.update_yaxes(autorange="reversed")
    return fig

# 6. Build the dashboard and launch it
with gr.Blocks(title="Rotten fruit check") as demo:
    gr.Markdown("# Rotten fruit check on a packing line\n"
                "A fine-tuned YOLO classifier gives each fruit a probability of being rotten. "
                "Below the first threshold it is packed, above the second it is thrown away, "
                "in between a worker checks it. Everything is loaded from `weights/` and `artifacts/`.")
    with gr.Row():
        low = gr.Slider(0, 1, value=LOW, step=0.01, label="Pass below (probability rotten)")
        high = gr.Slider(0, 1, value=HIGH, step=0.01, label="Fail from (probability rotten)")

    with gr.Tab("Check a fruit"):
        with gr.Row():
            image = gr.Image(type="filepath", label="Photo of one fruit")
            with gr.Column():
                decision = gr.Markdown()
                probs = gr.Label(label="Class probabilities")
        for control in [image, low, high]:
            control.change(check_fruit, [image, low, high], [probs, decision])

    with gr.Tab("Thresholds & cost"):
        split = gr.Radio(list(SPLITS), value="Test", label="Images")
        cost_text = gr.Markdown()
        with gr.Row():
            matrix_plot = gr.Plot()
            curve_plot = gr.Plot()
        policy_table = gr.Dataframe(label="Your setting against the alternatives")
        outputs = [matrix_plot, cost_text, policy_table, curve_plot]
        for control in [split, low, high]:
            control.change(threshold_view, [split, low, high], outputs)
        demo.load(threshold_view, [split, low, high], outputs)

    with gr.Tab("Per-class results"):
        gr.Markdown(f"Positive class = **rotten** ({', '.join(ROTTEN)}). Test split, plus the fine-tuned model "
                    "on photos from a second dataset it never saw.")
        gr.Dataframe(overall, label="Overall")
        gr.Dataframe(per_class_table, label="Per class")
        model_choice = gr.Dropdown(list(summary["confusion"]), value=list(summary["confusion"])[0], label="Model")
        confusion = gr.Plot(confusion_plot(list(summary["confusion"])[0]))
        model_choice.change(confusion_plot, model_choice, confusion)

demo.launch(server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"), server_port=7860)
