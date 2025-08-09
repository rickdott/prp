from hmpai.utilities import set_seaborn_style
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_timing_in_trial(df: pd.DataFrame, labels: list[str], split=True) -> None:
    set_seaborn_style()
    df_long = df.melt(
        id_vars="condition",
        value_vars=["t1_1_pred", "t1_2_pred", "t1_3_pred", "t2_1_pred", "t2_2_pred", "t2_3_pred"],
        var_name="prediction_type",
        value_name="predicted_value"
    )

    df_long["predicted_value"] = (df_long["predicted_value"] / 250) * 1000  # Convert to seconds
    order = [
        "t1_1_pred",
        "t1_2_pred",
        "t1_3_pred",
        "",
        "t2_1_pred",
        "t2_2_pred",
        "t2_3_pred"
    ]
    order.reverse()
    fig = plt.figure(figsize=(7.9, 6), dpi=300)

    ax = sns.boxplot(
        data=df_long,
        y="prediction_type",          # vertical categories
        x="predicted_value",          # horizontal violin
        hue="condition",
        # split=True,
        # cut=0,
        orient="h",
        order=order,
        hue_order=["long", "short"],
        showfliers=False,
    )

    tick_labels = [label.replace("_pred", "") for label in order]
    tick_labels = labels[:3] + [""] + labels[3:]
    tick_labels.reverse()
    plt.yticks(ticks=range(len(order)), labels=tick_labels)

    # Make xticks multiples of 300
    xticks = range(0, 2101, 300)
    plt.xticks(xticks, [f"{tick}" for tick in xticks])
    plt.xlim(-50, 2100)

    plt.ylabel("Operation")
    plt.xlabel("Time (ms)")
    # plt.title("Predicted Values by Condition")
    plt.tight_layout()

    # Use vlines instead of hlines since time is now on x-axis
    plt.vlines(0, 3.5, 6.5, colors="gray", linestyles='dashed', label='Stim T1')
    plt.vlines(300, -0.5, 2.5, colors=sns.color_palette()[1], linestyles='dashed', label='Stim T2 Short')
    plt.vlines(1200, -0.5, 2.5, colors=sns.color_palette()[0], linestyles='dashed', label='Stim T2 Long')

    plt.legend()
    return fig


def plot_order_effect_on_rt(df, orders: list[list[int]], labels=["t1_1", "t1_2", "t1_3", "t2_1", "t2_2", "t2_3"]):
    set_seaborn_style()
    event_cols = [col for col in df.columns if col.startswith("event_")]
    df = df.copy()
    df["order_idx"] = -1  # Initialize with -1 for rows that don't match any order
    df = df[df["condition"] == "short"]
    for idx, order in enumerate(orders):
        # Compare order of events with order in dataframe
        mask = (
            (df[event_cols[0]] == order[0])
            & (df[event_cols[1]] == order[1])
            & (df[event_cols[2]] == order[2])
            & (df[event_cols[3]] == order[3])
            & (df[event_cols[4]] == order[4])
            & (df[event_cols[5]] == order[5])
        )
        df.loc[mask, "order_idx"] = idx
    # Filter out rows with order_idx -1
    df = df[df["order_idx"] != -1]
    df["rt_t1"] = (df["rt_t1"] / 250) * 1000
    ax = sns.violinplot(data=df, x="order_idx", y="rt_t1", cut=0)
    xticks = [' > '.join([labels[i] for i in order]) for order in orders]
    plt.xticks(range(len(orders)), xticks, rotation=45)
    plt.hlines(y=610.63, xmin=-0.5, xmax=len(orders)-0.5, color=sns.color_palette()[2], linestyles='dashed', label="Mean RT (short)")
    plt.hlines(y=599.84, xmin=-0.5, xmax=len(orders)-0.5, color=sns.color_palette()[1], linestyles='dashed', label="Mean RT (long)")
    plt.legend()
    plt.xlabel("Order of events")
    plt.ylabel("RT t1 (ms)")
    plt.ylim(200, 1200)
    return plt.figure()

def plot_t1_encoding(df, orders, labels):
    event = "t1_1"
    set_seaborn_style()
    df = df.copy()
    df = df[df["condition"] == "short"]
    ev_col = f"{event}_pred"
    df[ev_col] = df[ev_col] / 250 * 1000
    sns.violinplot(data=df, x="order_idx", y=ev_col, cut=0)
    xticks = [' > '.join([labels[i] for i in order]) for order in orders]
    plt.xticks(range(len(orders)), xticks, rotation=60)
    plt.ylabel(f"Time of {event} (ms) from stim T1")
    plt.xlabel("Order")
    plt.ylim(-10, 275)
    # Stimulus presentation is from 300 > 450 ms
    plt.hlines(0, xmin=-0.5, xmax=len(orders)-0.5, color="green", linestyle="--", label="Stim flankers")
    plt.hlines(60, xmin=-0.5, xmax=len(orders)-0.5, color="blue", linestyle="--", label="Stim target")
    plt.hlines(260, xmin=-0.5, xmax=len(orders)-0.5, color="red", linestyle="--", label="Stim end")
    plt.legend()
    return plt.figure()


def plot_t2_encoding(df, orders, labels):
    event = "t2_1"
    set_seaborn_style()
    df = df.copy()
    df = df[df["condition"] == "short"]
    ev_col = f"{event}_pred"
    df[ev_col] = df[ev_col] / 250 * 1000
    sns.violinplot(data=df, x="order_idx", y=ev_col, cut=0)
    xticks = [' > '.join([labels[i] for i in order]) for order in orders]
    plt.xticks(range(len(orders)), xticks, rotation=60)
    plt.ylabel(f"Time of {event} (ms) from stim T1")
    plt.xlabel("Order")
    plt.ylim(275, 650)
    # Stimulus presentation is from 300 > 450 ms
    plt.hlines(300, xmin=-0.5, xmax=len(orders)-0.5, color="green", linestyle="--", label="Stim start")
    plt.hlines(450, xmin=-0.5, xmax=len(orders)-0.5, color="red", linestyle="--", label="Stim end")
    plt.legend()
    return plt.figure()

def plot_order_distribution_stacked(df, labels):
    set_seaborn_style()
    # Count the occurrences of each order, grouped by condition
    df = df[["condition", "order"]]
    order_counts = df.groupby(["condition", "order"]).value_counts().reset_index(name="Count")
    print(order_counts)

    # Create a DataFrame for plotting
    order_df = pd.DataFrame(order_counts)
    order_df.columns = ["condition", "order", "Count"]

    # Convert the Order tuples to strings for better readability
    order_df["order"] = order_df["order"].apply(lambda x: " > ".join([labels[i] for i in x]))

    # Plotting
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Count", y="order", hue="condition", data=order_df)
    plt.xlabel("Count")
    plt.ylabel("Order")
    plt.xlim(0, order_df["Count"].max() + 50)
    return plt.figure()

def plot_sequences_per_participant(df, labels, ax):
    # Count how many times each order_idx appears per participant
    set_seaborn_style()
    df = df.copy()
    df = df[df['condition'] == 'short']
    counts = (
        df.groupby(["participant", "order_idx"])
        .size()
        .unstack(fill_value=0)
    )
    counts.index = counts.index.str.replace("VP", "").astype(int)
    counts = counts.sort_index()
    # Convert to percentages if you want proportions instead of raw counts
    counts = counts.div(counts.sum(axis=1), axis=0) * 100
    counts = counts[counts.columns[::-1]]
    desat_hex = sns.color_palette(
        sns.color_palette()[5:10][::-1], desat=0.8
    ).as_hex()
    # Plot as stacked bar chart
    counts.plot(
        kind="bar",
        stacked=True,
        ax=ax,
        width=0.9,
        color=desat_hex,
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.set_xlabel("Participant")
    ax.set_ylabel("Trials (%)")
    ax.set_ylim(0, 100)  # if using percentages