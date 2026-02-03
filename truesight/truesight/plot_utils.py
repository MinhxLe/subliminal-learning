import textwrap
from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np

TITLE_FONT_SIZE = 18
AXIS_FONT_SIZE = 18
TICK_FONT_SIZE = 12
LEGEND_FONT_SIZE = 14
CAPTION_FONT_SIZE = 12
MAX_TEXT_WRAP = 100


def create_plot(
    *,
    figsize: Tuple[int, int],
    title: str | None,
    x_label: str | None,
    y_label: str | None,
    caption: str | None,
):
    fig, ax = plt.subplots(figsize=figsize)
    if title is not None:
        ax.set_title(title, fontsize=TITLE_FONT_SIZE, fontweight="bold")
    if x_label:
        ax.set_xlabel(x_label, fontsize=AXIS_FONT_SIZE)
    if y_label:
        ax.set_ylabel(y_label, fontsize=AXIS_FONT_SIZE)
    if caption:
        fig.text(
            0.5,
            0.01,
            "\n".join(textwrap.wrap(caption, MAX_TEXT_WRAP)),
            ha="center",
            fontsize=CAPTION_FONT_SIZE,
        )

    return fig, ax


def plot_CIs(
    df,
    x_col: str,
    *,
    figsize: Tuple[int, int],
    title: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    caption: str | None = None,
    x_order: list[str] | None = None,
    x_legend_name: dict[str, str] | None = None,
    x_color_map: dict[str, str] | None = None,
    x_control: str | None = None,
):
    fig, ax = create_plot(
        title=title,
        x_label=x_label,
        y_label=y_label,
        figsize=figsize,
        caption=caption,
    )

    # Get unique categories and apply ordering if specified
    all_categories = df[x_col].unique()

    # Filter out control from categories to plot as bars
    if x_control is not None:
        all_categories = all_categories[all_categories != x_control]

    # Apply x ordering if specified
    if x_order is not None:
        # Use specified order, but include any categories not in the order at the end
        categories = []
        for category in x_order:
            if category in all_categories:
                categories.append(category)
        # Add any remaining categories not specified in the order
        for category in all_categories:
            if category not in categories:
                categories.append(category)
        categories = np.array(categories)
    else:
        categories = all_categories

    # Get values for each category in the specified order
    means = []
    lower_bounds = []
    upper_bounds = []

    for category in categories:
        category_data = df[df[x_col] == category]
        if not category_data.empty:
            means.append(category_data["mean"].iloc[0])
            lower_bounds.append(category_data["lower_bound"].iloc[0])
            upper_bounds.append(category_data["upper_bound"].iloc[0])
        else:
            means.append(0)
            lower_bounds.append(0)
            upper_bounds.append(0)

    means = np.array(means)
    lower_bounds = np.array(lower_bounds)
    upper_bounds = np.array(upper_bounds)

    # Create x positions
    x_pos = np.arange(len(categories))

    # Create colors for different bars using x_color_map if provided
    if x_color_map is not None:
        colors = [
            x_color_map.get(cat, plt.cm.tab10(i / len(categories)))
            for i, cat in enumerate(categories)
        ]
    else:
        colors = plt.cm.tab10(np.linspace(0, 1, len(categories)))

    # Calculate error bar lengths
    yerr_lower = means - lower_bounds
    yerr_upper = upper_bounds - means

    # Plot bars with different colors
    bars = ax.bar(x_pos, means, color=colors, alpha=0.7)

    # Add black error bars and colored dots on top of bars
    for i, (x, mean, yerr_l, yerr_u) in enumerate(
        zip(x_pos, means, yerr_lower, yerr_upper)
    ):
        yerr_this = np.array([[yerr_l], [yerr_u]])
        # Draw black error bars
        ax.errorbar(
            x,
            mean,
            yerr=np.max(yerr_this, 0),
            capsize=3,
            capthick=1,
            fmt="none",
            ecolor="black",
            elinewidth=1,
            zorder=3,
        )
        # Add colored dot at the mean
        ax.scatter(
            x,
            mean,
            s=36,
            color=colors[i],
            zorder=4,
            edgecolor="black",
            linewidth=0.5,
        )

    # Set x-ticks but remove labels
    ax.set_xticks(x_pos)
    ax.set_xticklabels([""] * len(categories))  # Empty labels

    # Add grid for easier reading
    ax.grid(True, linestyle="--", alpha=0.3)

    # Add control line if specified
    if x_control is not None:
        control_data = df[df[x_col] == x_control]
        if not control_data.empty:
            control_mean = control_data["mean"].iloc[0]

            # Add horizontal dotted red line at control mean
            ax.axhline(
                y=control_mean, color="red", linestyle="--", linewidth=2, alpha=0.8
            )

    # Add a legend for each category with its color
    for i, category in enumerate(categories):
        legend_label = (
            x_legend_name.get(category, category) if x_legend_name else category
        )
        ax.scatter(
            [],
            [],
            color=colors[i],
            label=legend_label,
            s=36,
            edgecolor="black",
            linewidth=0.5,
        )

    # Add control to legend if specified
    if x_control is not None:
        control_legend_label = (
            x_legend_name.get(x_control, x_control) if x_legend_name else x_control
        )
        ax.plot(
            [], [], color="red", linestyle="--", alpha=0.8, label=control_legend_label
        )

    ax.legend(loc="best", prop={"size": LEGEND_FONT_SIZE})
    return fig, ax


def plot_grouped_CIs(
    df,
    x_col,
    group_col,
    *,
    figsize: Tuple[int, int],
    title: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    group_spacing: float = 0.4,  # New parameter to control group spacing
    caption: str | None = None,
    group_color_map: dict[str, str] | None = None,
    group_order: list[str] | None = None,
    group_legend_name: dict[str, str] | None = None,
    legend_loc: str = "best",
    x_order: list[str] | None = None,
):
    group_color_map = group_color_map or dict()
    group_legend_name = group_legend_name or dict()
    # Get unique values for x-axis and group
    all_x_categories = df[x_col].unique()

    # Apply x ordering if specified
    if x_order is not None:
        # Use specified order, but include any categories not in the order at the end
        x_categories = []
        for category in x_order:
            if category in all_x_categories:
                x_categories.append(category)
        # Add any remaining categories not specified in the order
        for category in all_x_categories:
            if category not in x_categories:
                x_categories.append(category)
        x_categories = np.array(x_categories)
    else:
        x_categories = all_x_categories

    all_groups = df[group_col].unique()

    # Apply group ordering if specified
    if group_order is not None:
        # Use specified order, but include any groups not in the order at the end
        groups = []
        for group in group_order:
            if group in all_groups:
                groups.append(group)
        # Add any remaining groups not specified in the order
        for group in all_groups:
            if group not in groups:
                groups.append(group)
        groups = np.array(groups)
    else:
        groups = all_groups

    # Number of groups and group width
    n_groups = len(groups)
    group_width = group_spacing / n_groups
    x = np.arange(len(x_categories))

    # Set up the plot
    fig, ax = create_plot(
        title=title,
        x_label=x_label,
        y_label=y_label,
        figsize=figsize,
        caption=caption,
    )

    # Plot bars for each group
    # Create color mapping - use provided colors or generate defaults
    group_colors = []
    available_colors = plt.cm.tab10(np.linspace(0, 1, 10))  # Get 10 colors from tab10
    used_colors = set(group_color_map.values())
    available_color_idx = 0

    for group in groups:
        if group in group_color_map:
            group_colors.append(group_color_map[group])
        else:
            # Find next available color not in the provided map
            while available_color_idx < len(available_colors):
                color = available_colors[available_color_idx]
                available_color_idx += 1
                # Convert to tuple for comparison (matplotlib colors can be arrays)
                color_tuple = tuple(color) if hasattr(color, "__iter__") else color
                if color_tuple not in used_colors:
                    group_colors.append(color)
                    break
            else:
                raise ValueError("no color available")
    for i, group in enumerate(groups):
        # Filter data for this group
        group_data = df[df[group_col] == group]

        # Get values for each x_category
        means = []
        lower_bounds = []
        upper_bounds = []

        for category in x_categories:
            category_data = group_data[group_data[x_col] == category]
            if not category_data.empty:
                means.append(category_data["mean"].iloc[0])
                lower_bounds.append(category_data["lower_bound"].iloc[0])
                upper_bounds.append(category_data["upper_bound"].iloc[0])
            else:
                means.append(0)
                lower_bounds.append(0)
                upper_bounds.append(0)

        # Calculate error
        yerr_lower = [m - l for m, l in zip(means, lower_bounds)]  # noqa
        yerr_upper = [u - m for m, u in zip(means, upper_bounds)]  # noqa
        yerr = [np.maximum(yerr_lower, 0), yerr_upper]

        # Plot bars with different colors
        position = x - (group_width * (n_groups - 1) / 2) + i * group_width
        legend_label = group_legend_name.get(group, group)
        ax.bar(
            position,
            means,
            group_width,
            label=legend_label,
            color=group_colors[i],
            alpha=0.7,
        )

        # Add black error bars and colored dots on top of bars
        ax.errorbar(
            position,
            means,
            yerr=yerr,
            capsize=3,
            capthick=1,
            fmt="none",
            ecolor="black",
            elinewidth=1,
            zorder=3,
        )
        # Add colored dots at the means
        for j, (pos, mean) in enumerate(zip(position, means)):
            ax.scatter(
                pos,
                mean,
                s=36,
                color=group_colors[i],
                zorder=4,
                edgecolor="black",
                linewidth=0.5,
            )
    ax.grid(True, axis="y")
    ax.legend(loc=legend_loc, prop={"size": LEGEND_FONT_SIZE})
    ax.set_xticks(x)
    ax.set_xticklabels(x_categories, ha="center", fontsize=TICK_FONT_SIZE)
    return fig, ax


def create_sorted_dot_plot(
    df,
    treatment_col,
    measure_col,
    mean_col="mean",
    lower_bound_col="lower_bound",
    upper_bound_col="upper_bound",
    sort_by="effect_size",
    ascending=False,
    x_label="Probability",
    y_label="Questions",
    title="Probability Estimates Across Groups",
    figsize=(12, 20),
    palette=None,
    show_only_significant=False,
    top_n=None,  # New parameter to specify the number of top effects to show
):
    """
    Create a sorted dot plot for visualizing treatment effects across multiple questions.

    Parameters:
    -----------
    df : pandas.DataFrame
        Data containing treatment effects
    treatment_col : str
        Column name for treatment groups
    measure_col : str
        Column name for questions/measures
    mean_col : str
        Column name for mean values
    lower_bound_col : str
        Column name for lower confidence bounds
    upper_bound_col : str
        Column name for upper confidence bounds
    sort_by : str
        Method for sorting questions ('effect_size' or other will use alphabetical)
    ascending : bool
        Sort order
    x_label, y_label, title : str
        Plot labels
    figsize : tuple
        Figure size (width, height)
    palette : dict
        Color mapping for treatment groups
    show_only_significant : bool
        Whether to filter for only significant effects
    top_n : int or None
        Number of top effects to display (None = show all)
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    # Make a copy to avoid modifying the original
    df_plot = df.copy()

    # Ensure all relevant columns are numeric
    numeric_cols = [mean_col, lower_bound_col, upper_bound_col]
    for col in numeric_cols:
        # Skip conversion if already numeric
        if pd.api.types.is_numeric_dtype(df_plot[col]):
            continue

        # Try to convert to numeric more safely
        try:
            df_plot[col] = df_plot[col].astype(float)
        except Exception as e:
            print(f"Warning: Could not convert column {col} to numeric. Error: {e}")
            # Provide fallback values
            df_plot[col] = 0.0

    # Get unique treatments and questions
    treatments = df_plot[treatment_col].unique()
    questions = df_plot[measure_col].unique()

    # If first treatment is control, use it as reference; otherwise, use the first one
    control_group = treatments[0]

    # Calculate effect size for each question (if sorting by effect)
    if sort_by == "effect_size":
        try:
            # Pivot to get mean values for each treatment/question combination
            means_pivot = df_plot.pivot_table(
                index=measure_col, columns=treatment_col, values=mean_col
            )

            # Calculate max absolute effect compared to control for each question
            effect_sizes = {}
            for question in questions:
                # Skip if data is missing
                if (
                    question not in means_pivot.index
                    or control_group not in means_pivot.columns
                ):
                    effect_sizes[question] = 0
                    continue

                # Get control value
                control_value = means_pivot.loc[question, control_group]

                # Calculate effects for each non-control treatment
                max_effect = 0
                for treatment in treatments:
                    if treatment != control_group and treatment in means_pivot.columns:
                        effect = abs(
                            means_pivot.loc[question, treatment] - control_value
                        )
                        max_effect = max(max_effect, effect)

                effect_sizes[question] = max_effect

            # Convert to DataFrame for sorting
            effect_df = pd.DataFrame(
                list(effect_sizes.items()), columns=[measure_col, "effect_size"]
            )

            # Sort questions by effect size
            sorted_questions = effect_df.sort_values(
                "effect_size", ascending=ascending
            )[measure_col].tolist()

            # Filter for top N questions if specified
            if top_n is not None and top_n > 0 and top_n < len(sorted_questions):
                sorted_questions = sorted_questions[:top_n]

        except Exception as e:
            print(
                f"Warning: Error calculating effect sizes: {e}. Falling back to question sorting."
            )
            sorted_questions = sorted(questions, reverse=not ascending)

            # Apply top_n filter here as well for alphabetical sorting
            if top_n is not None and top_n > 0 and top_n < len(sorted_questions):
                sorted_questions = sorted_questions[:top_n]
    else:
        # Sort questions alphanumerically
        sorted_questions = sorted(questions, reverse=not ascending)

        # Apply top_n filter for alphabetical sorting
        if top_n is not None and top_n > 0 and top_n < len(sorted_questions):
            sorted_questions = sorted_questions[:top_n]

    # Create question to y-position mapping
    question_positions = {q: i for i, q in enumerate(sorted_questions)}

    # Determine statistical significance
    df_plot["significant"] = False

    try:
        # Find control group data for each question
        control_data = df_plot[df_plot[treatment_col] == control_group].copy()
        control_data = control_data.set_index(measure_col)

        # For each treatment and question, check if CIs overlap with control
        for idx, row in df_plot.iterrows():
            if row[treatment_col] != control_group:
                q = row[measure_col]
                # Non-overlapping CIs indicate significance (conservative approach)
                if q in control_data.index:
                    # If treatment upper bound < control lower bound or
                    # treatment lower bound > control upper bound -> significant
                    if (
                        row[upper_bound_col] < control_data.loc[q, lower_bound_col]
                        or row[lower_bound_col] > control_data.loc[q, upper_bound_col]
                    ):
                        df_plot.loc[idx, "significant"] = True
    except Exception as e:
        print(f"Warning: Error determining significance: {e}")

    # Filter for significant results if requested
    if show_only_significant:
        try:
            sig_questions = df_plot[df_plot["significant"] == True][  # noqa
                measure_col
            ].unique()
            df_plot = df_plot[df_plot[measure_col].isin(sig_questions)]
            sorted_questions = [q for q in sorted_questions if q in sig_questions]
            question_positions = {q: i for i, q in enumerate(sorted_questions)}
        except Exception as e:
            print(f"Warning: Error filtering significant results: {e}")

    # Filter df_plot to include only the questions we want to plot
    df_plot = df_plot[df_plot[measure_col].isin(sorted_questions)]

    # Adjust figsize based on number of questions if top_n is specified
    if top_n is not None and top_n > 0:
        # Scale the height based on number of questions (roughly 0.5 inch per question)
        adjusted_height = max(8, min(figsize[1], top_n * 0.5 + 4))
        figsize = (figsize[0], adjusted_height)

    # Start plotting
    fig, ax = plt.subplots(figsize=figsize)

    # Set default color palette if none provided
    if palette is None:
        import matplotlib.cm as cm

        palette = {}
        cmap = cm.get_cmap("tab10")
        for i, treatment in enumerate(treatments):
            palette[treatment] = cmap(i % 10)

    # Add a small offset for each treatment to avoid overlapping
    offsets = np.linspace(-0.2, 0.2, len(treatments))
    offset_dict = {t: o for t, o in zip(treatments, offsets)}

    # Track legend handles and labels
    legend_handles = []
    legend_labels = []

    # Plot for each treatment group
    for i, treatment in enumerate(treatments):
        treatment_data = df_plot[df_plot[treatment_col] == treatment]

        # Skip if no data for this treatment
        if len(treatment_data) == 0:
            continue

        # Extract y-positions based on question order
        y_positions = []
        x_values = []
        lower_bounds = []
        upper_bounds = []
        significant_flags = []

        for _, row in treatment_data.iterrows():
            q = row[measure_col]
            if q in question_positions:
                y_positions.append(question_positions[q] + offset_dict[treatment])
                x_values.append(row[mean_col])
                lower_bounds.append(row[lower_bound_col])
                upper_bounds.append(row[upper_bound_col])
                significant_flags.append(row["significant"])

        # Plot horizontal bars
        bars = ax.barh(
            y_positions,
            x_values,
            height=0.15,  # bar height
            color=palette.get(treatment, "black"),
            alpha=0.7,
            label=treatment,
            zorder=2,
        )

        # Plot black confidence intervals on top of bars
        for j, (y, x, lb, ub, sig) in enumerate(
            zip(y_positions, x_values, lower_bounds, upper_bounds, significant_flags)
        ):
            # Set line properties based on significance
            linewidth = 1.5 if sig else 1.0
            cap_size = 0.03

            # Draw black CI line
            ax.plot(
                [lb, ub],
                [y, y],
                color="black",
                linewidth=linewidth,
                zorder=3,
            )
            # Add black caps
            ax.plot(
                [lb, lb],
                [y - cap_size, y + cap_size],
                color="black",
                linewidth=linewidth,
                zorder=3,
            )
            ax.plot(
                [ub, ub],
                [y - cap_size, y + cap_size],
                color="black",
                linewidth=linewidth,
                zorder=3,
            )

        # Add colored dots at the means
        for j, (y, x) in enumerate(zip(y_positions, x_values)):
            ax.scatter(
                x,
                y,
                s=50,
                color=palette.get(treatment, "black"),
                zorder=4,
                edgecolor="black",
                linewidth=0.5,
            )

        # Create a dummy object for legend
        points = bars

        # Add red outlines for significant bars if not control group
        if treatment != control_group:
            for j, (x, y, sig) in enumerate(
                zip(x_values, y_positions, significant_flags)
            ):
                if sig:
                    # Add red outline to significant bars
                    ax.barh(
                        y,
                        x,
                        height=0.15,
                        facecolor="none",
                        edgecolor="red",
                        linewidth=2,
                        zorder=5,
                    )

        # Add to legend
        legend_handles.append(points)
        legend_labels.append(treatment)

    # Set y-ticks at the correct positions with question labels
    ax.set_yticks(range(len(sorted_questions)))
    ax.set_yticklabels(sorted_questions)

    # Format x-axis as percentages if values are between 0-1
    x_max = df_plot[mean_col].max()
    if x_max <= 1.0:
        from matplotlib.ticker import PercentFormatter

        ax.xaxis.set_major_formatter(PercentFormatter(1.0))

    # Add grid for readability
    ax.grid(True, linestyle="--", alpha=0.7, axis="x")

    # Set labels and title
    ax.set_xlabel(x_label, fontsize=14)
    ax.set_ylabel(y_label, fontsize=14)
    ax.set_title(title, fontsize=20, pad=20)

    # Add legend
    ax.legend(
        handles=legend_handles,
        labels=legend_labels,
        title=treatment_col,
        loc="upper right",
        bbox_to_anchor=(1.15, 1),
    )

    # Add annotation for significance
    if not all(
        df_plot[df_plot[treatment_col] != control_group]["significant"] == False  # noqa
    ):
        ax.annotate(
            "Red outlines indicate significant differences from control",
            xy=(0.02, 0.01),
            xycoords="figure fraction",
            fontsize=10,
            style="italic",
        )

    return fig, ax
