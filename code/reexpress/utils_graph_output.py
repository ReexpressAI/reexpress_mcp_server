# Copyright Reexpress AI, Inc. All rights reserved.

"""
This is a simple, but effective interactive visualization. This takes as input the prediction output file. Click on
a point for additional information to be printed to the console. The histograms above and to the right of the
scatterplot show the distribution of relative counts. A plot is generated for each ground-truth label.
"""
import argparse
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
from datetime import datetime

import utils_model
import constants


def get_mean_or_zero(list_to_process):
    # The graphed selection can be empty (e.g., with a restrictive
    # --graph_class_and_prediction_conditional_estimates_min_region/_max_region range), in which case the
    # accuracy prints show 0.0 rather than propagating NaN:
    if len(list_to_process) == 0:
        return 0.0
    return np.mean(list_to_process)


class InteractiveScatter:
    def __init__(self, x, y, colors_filtered, linewidth, point_sizes, ids, data_rows, ax):
        self.x = np.array(x)
        self.y = np.array(y)
        self.colors_filtered = colors_filtered
        self.linewidth = linewidth
        self.point_sizes = np.array(point_sizes)
        self.ids = ids
        self.data_rows = data_rows
        self.fig = ax.figure
        self.ax = ax
        self.scatter = self.ax.scatter(x, y, c=colors_filtered,
                                       linewidth=linewidth,
                                       s=point_sizes)  # ,
        # edgecolors='black')

        self.annotation = self.ax.annotate("", xy=(0, 0), xytext=(20, 20),
                                           textcoords="offset points",
                                           bbox=dict(boxstyle="round", fc="yellow", alpha=0.7),
                                           arrowprops=dict(arrowstyle="->"))
        self.annotation.set_visible(False)

        # Store the current hover index to avoid flickering
        self.current_hover_idx = None
        # Store the clicked index to keep annotation visible
        self.clicked_idx = None

        self.fig.canvas.mpl_connect('motion_notify_event', self.on_hover)
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)

    def get_point_at_event(self, event):
        """Find which point (if any) is at the event location"""
        if event.xdata is None or event.ydata is None:
            return None

        # Transform data coordinates to display coordinates
        points_display = self.ax.transData.transform(np.column_stack([self.x, self.y]))
        mouse_display = self.ax.transData.transform([[event.xdata, event.ydata]])[0]

        # Calculate distances in display coordinates (pixels)
        distances = np.sqrt((points_display[:, 0] - mouse_display[0]) ** 2 +
                            (points_display[:, 1] - mouse_display[1]) ** 2)

        # Calculate the radius of each point in pixels
        dpi = self.fig.dpi
        point_radii = np.sqrt(self.point_sizes) * dpi / 72.0

        # Add some padding for easier interaction
        point_radii = point_radii + 2  # 2 pixel padding

        # Check if mouse is over any point
        hover_mask = distances <= point_radii

        if np.any(hover_mask):
            # Get the closest point among those we're hovering over
            hover_indices = np.where(hover_mask)[0]
            return hover_indices[np.argmin(distances[hover_indices])]

        return None

    def update_annotation(self, idx):
        """Update the annotation for a given point index"""
        self.annotation.xy = (self.x[idx], self.y[idx])
        text = f"ID: {self.ids[idx]}\n({self.x[idx]:.2f}, {self.y[idx]:.2f})"

        # If this is a clicked point, add a note about copying
        if idx == self.clicked_idx:
            text += "\n(Click elsewhere to hide)"
            self.annotation.get_bbox_patch().set(fc="lightblue", alpha=0.7)
        else:
            text += "\n(Click to print to console)"
            self.annotation.get_bbox_patch().set(fc="yellow", alpha=0.7)

        self.annotation.set_text(text)

        if not self.annotation.get_visible():
            self.annotation.set_visible(True)

    def on_click(self, event):
        if event.inaxes != self.ax:
            return

        if event.button == 1:  # Left click
            idx = self.get_point_at_event(event)

            if idx is not None:
                # Print to console for reference, since the popover is not selectable
                print(f"\n{'=' * 40}")
                print(f"ID: {self.ids[idx]}")
                print(f"Coordinates: ({self.x[idx]:.2f}, {self.y[idx]:.2f})")
                print(f"Row: {self.data_rows[idx]}")
                # Below, we duplicate the key information from the row to make it easier to read:
                print(f"{'+' * 14 * 2}\n")
                print(f"Uncertainty Region (Class- and prediction-conditional):")
                print(f"Class- and prediction-conditional accuracy estimate >= "
                      f"{self.data_rows[idx]['hr_region_alpha']}")
                print(f"Class- and prediction-conditional accuracy estimate "
                      f"(accounting for sample-size error in distance eCDF) >= "
                      f"{self.data_rows[idx]['hr_region_alpha_lower']}")
                print(f"\n{'+' * 14}")
                print(f"Label == Prediction: {self.data_rows[idx]['label'] == self.data_rows[idx]['prediction']}")
                print(f"Label: {self.data_rows[idx]['label']}")
                print(f"Prediction: {self.data_rows[idx]['prediction']}")
                print(f"p(y|x): {self.data_rows[idx]['sdm_output']}")
                print(f"p(y|x)_lower: {self.data_rows[idx]['sdm_output_d_lower']}")
                print(f"Rescaled Similarity (q'): {self.data_rows[idx]['rescaled_similarity']}")
                print(f"Rescaled Similarity, lower (q'_lower): {self.data_rows[idx]['rescaled_similarity_lower']}")
                print(f"Effective sample size: {self.data_rows[idx]['cumulative_effective_sample_sizes']}")
                separator_text = ", "
                print(f"***S-D-M***:")
                print(f"\tSimilarity (q): {self.data_rows[idx]['q']}\n"
                      f"\tDistance quantile, lower (d_lower): {self.data_rows[idx]['d_lower']}, "
                      f"Distance quantile (d): {self.data_rows[idx]['d']}{separator_text}\n"
                      f"\tMagnitude (f): {self.data_rows[idx]['f']}")
                print(f"d_nearest: {self.data_rows[idx]['d0']}")
                print(f"\n{'+' * 14}")
                print(f"Document:\n{self.data_rows[idx]['document']}")
                print(f"{'=' * 40}\n")

                # Clicked on a point - make it persist
                self.clicked_idx = idx
                self.update_annotation(idx)
                self.fig.canvas.draw_idle()
            else:
                # Clicked on empty space - clear the clicked point
                if self.clicked_idx is not None:
                    self.clicked_idx = None
                    # Hide annotation unless we're hovering over something
                    if self.current_hover_idx is None:
                        self.annotation.set_visible(False)
                    else:
                        self.update_annotation(self.current_hover_idx)
                    self.fig.canvas.draw_idle()

    def on_hover(self, event):
        if event.inaxes != self.ax:
            # Only hide if there's no clicked point
            if self.annotation.get_visible() and self.clicked_idx is None:
                self.annotation.set_visible(False)
                self.current_hover_idx = None
                self.fig.canvas.draw_idle()
            return

        # Don't update hover if we have a clicked point
        if self.clicked_idx is not None:
            return

        idx = self.get_point_at_event(event)

        if idx is not None:
            # Only update if we're hovering over a different point
            if idx != self.current_hover_idx:
                self.current_hover_idx = idx
                self.update_annotation(idx)
                self.fig.canvas.draw_idle()
        else:
            # No point is being hovered
            if self.annotation.get_visible() and self.clicked_idx is None:
                self.annotation.set_visible(False)
                self.current_hover_idx = None
                self.fig.canvas.draw_idle()


def graph_sdm_estimator_output(options, json_lines, true_label_to_graph=None,
                               min_rescaled_similarity_to_determine_high_reliability_region=None,
                               hr_output_thresholds=None,
                               hr_class_conditional_accuracy=None,
                               model=None):
    assert true_label_to_graph is not None
    ood_color = "darkviolet"
    min_rescaled_similarity_to_determine_high_reliability_region_error_color = "darkblue"
    latex_approx_symbol = r'$\approx$'

    x_filtered = []
    y_filtered = []
    document_ids_filtered = []
    data_rows_filtered = []
    colors_filtered = []
    accuracy = []
    accuracy_class_conditional = []
    accuracy_filtered = []
    point_sizes = []
    is_correct_filtered = []  # Track correct/incorrect for histograms

    # Restriction to a range of uncertainty regions (inclusive on both ends). At the defaults (the options are
    # None), every point passes (the assigned region alpha values are in [0, 1], with 0.0 indicating no
    # region), so the behavior of the existing code is unchanged:
    region_range_is_active = \
        options.graph_class_and_prediction_conditional_estimates_min_region is not None or \
        options.graph_class_and_prediction_conditional_estimates_max_region is not None
    effective_min_region = options.graph_class_and_prediction_conditional_estimates_min_region \
        if options.graph_class_and_prediction_conditional_estimates_min_region is not None else 0.0
    effective_max_region = options.graph_class_and_prediction_conditional_estimates_max_region \
        if options.graph_class_and_prediction_conditional_estimates_max_region is not None else 1.0
    assert 0.0 <= effective_min_region <= effective_max_region <= 1.0, \
        f"ERROR: The region range [{effective_min_region}, {effective_max_region}] must satisfy " \
        f"0 <= min <= max <= 1."

    unassigned_for_label_count = 0  # points with this label assigned to no region (an alpha of 0.0)
    for document in json_lines:

        document_id = document["id"]
        floor_rescaled_similarity = document["floor_rescaled_similarity"]
        rescaled_similarity = document["rescaled_similarity"]
        rescaled_similarity_lower = document["rescaled_similarity_lower"]
        # q = document["q"]
        # d = document["d"]
        prediction_probability = document["sdm_output"][document["prediction"]]
        prediction_probability_lower = document["sdm_output_d_lower"][document["prediction"]]
        # softmax_predicted = torch.softmax(torch.tensor(document["f"]), dim=0)[document["prediction"]]  # reference

        if options.graph_centroid:
            assigned_region_alpha = document["hr_region_alpha"]
        else:
            assigned_region_alpha = document["hr_region_alpha_lower"]
        if options.graph_all_points:
            filter_condition = True
        else:
            if region_range_is_active:
                # With an active region range, the admitted points are those assigned to ANY recorded region
                # (an assigned alpha of 0.0 indicates no region), with the range restriction applied below.
                # (The is_high_reliability_region and is_high_reliability_region_lower fields indicate
                # membership in the most conservative region only, which would contradict ranges that exclude
                # that region.)
                filter_condition = assigned_region_alpha > 0.0
            else:
                if options.graph_centroid:
                    filter_condition = document["is_high_reliability_region"]
                else:
                    filter_condition = document["is_high_reliability_region_lower"]
        filter_condition = filter_condition and \
            effective_min_region <= assigned_region_alpha <= effective_max_region
        label = document["label"]
        if true_label_to_graph is not None:
            filter_condition = filter_condition and label == true_label_to_graph
            # can use these to verify the bin distribution alignment, which must match with the x and y-axis:
            # and 0.0 <= rescaled_similarity <= 2.5
            # and 0.7 <= prediction_probability <= 0.8
        if label == true_label_to_graph:
            accuracy_class_conditional.append(document["prediction"] == label)
            if assigned_region_alpha == 0.0:
                unassigned_for_label_count += 1
        if filter_condition:
            document_ids_filtered.append(document_id)
            is_correct = document["prediction"] == label
            accuracy_filtered.append(is_correct)
            is_correct_filtered.append(is_correct)
            if options.graph_centroid:
                x_filtered.append(rescaled_similarity)
                y_filtered.append(prediction_probability)
            else:
                x_filtered.append(rescaled_similarity_lower)
                y_filtered.append(prediction_probability_lower)
            data_rows_filtered.append(document)
            if is_correct:
                colors_filtered.append("green")
            else:
                colors_filtered.append("red")
            # Incorrect predictions are up-weighted for visual emphasis:
            if options.emphasize_wrong_predictions:
                point_sizes.append(16 if not is_correct else 4)
            else:
                point_sizes.append(4)

        accuracy.append(document["prediction"] == label)

    print(f"Marginal accuracy: {get_mean_or_zero(accuracy)} out of {len(accuracy)}")
    total_points = len(accuracy)
    if region_range_is_active:
        print(f"Restricting the graphed points to the uncertainty regions in "
              f"[{effective_min_region}, {effective_max_region}] (inclusive), based on "
              f"{'hr_region_alpha' if options.graph_centroid else 'hr_region_alpha_lower'}: "
              f"{len(accuracy_filtered)} points remain.")
        # An assigned alpha of 0.0 is not a region: it indicates the points assigned to NO recorded region
        # (i.e., the rejections). These are only graphed with --graph_all_points and a range that reaches 0:
        if effective_min_region == 0.0:
            if options.graph_all_points:
                print(f"	(Including the {unassigned_for_label_count} points with this label that are not "
                      f"assigned to any region, since --graph_all_points was provided and the range "
                      f"includes 0.)")
            else:
                print(f"	(Excluding the {unassigned_for_label_count} points with this label that are not "
                      f"assigned to any region. Provide --graph_all_points to include them.)")
        else:
            print(f"	(Points not assigned to any region are excluded by the range itself, so "
                  f"--graph_all_points has no effect on the selection here.)")
    if options.graph_all_points:
        print(f"Class-conditional (label={true_label_to_graph}) accuracy: "
              f"{get_mean_or_zero(accuracy_filtered)} out of {len(accuracy_filtered)} "
              f"({len(accuracy_filtered)/total_points if total_points > 0 else 0.0} of all points)")
    else:
        print(
            f"Class-conditional (label={true_label_to_graph}) accuracy: "
            f"{get_mean_or_zero(accuracy_class_conditional)} out of {len(accuracy_class_conditional)} "
            f"({len(accuracy_class_conditional)/total_points if total_points > 0 else 0.0} of all points)")
        if options.graph_centroid:
            print(
                f"Class-conditional (label={true_label_to_graph}) accuracy, SDM_HR: "
                f"{get_mean_or_zero(accuracy_filtered)} out of {len(accuracy_filtered)} "
                f"({len(accuracy_filtered)/total_points if total_points > 0 else 0.0} of all points)")
        else:
            print(
                f"Class-conditional (label={true_label_to_graph}) accuracy, SDM_HR (lower): "
                f"{get_mean_or_zero(accuracy_filtered)} out of {len(accuracy_filtered)} "
                f"({len(accuracy_filtered)/total_points if total_points > 0 else 0.0} of all points)")

    # Create figure with subplots
    fig = plt.figure(figsize=(10, 10))

    # Create grid for subplots with more rows to accommodate title and legend
    # Main scatter plot (bottom left)
    ax_main = plt.subplot2grid((5, 5), (1, 0), colspan=4, rowspan=3)
    # Top histogram (top, aligned with main plot)
    ax_top = plt.subplot2grid((5, 5), (0, 0), colspan=4, sharex=ax_main)
    # Right histogram (right, aligned with main plot)
    ax_right = plt.subplot2grid((5, 5), (1, 4), rowspan=3, sharey=ax_main)

    # Convert to numpy arrays for easier manipulation
    x_filtered = np.array(x_filtered)
    y_filtered = np.array(y_filtered)
    is_correct_filtered = np.array(is_correct_filtered)

    # Create the main scatter plot
    interactive = InteractiveScatter(x_filtered, y_filtered,
                                     colors_filtered=colors_filtered, linewidth=0.5, point_sizes=point_sizes,
                                     ids=document_ids_filtered, data_rows=data_rows_filtered, ax=ax_main)

    ax_main.set_xlabel(r"$q'$")
    if options.graph_centroid:
        ax_main.set_ylabel(r"$\rm{sdm}(\mathbf{z'})_{\hat{y}}$")
    else:
        ax_main.set_ylabel(r"$\rm{sdm}(\mathbf{z'})_{\hat{y}}~({\mathrm{lower}})$")

    if true_label_to_graph is not None:
        if region_range_is_active:
            # The alpha values of the assigned regions are restricted to the provided (inclusive) range:
            alpha_value_string = r'$\in$' + f"[{effective_min_region}, {effective_max_region}]"
        else:
            alpha_value_string = f"={hr_class_conditional_accuracy}"
        if options.graph_all_points:
            latex_string = r'$\alpha$'
            # Rejections (an assigned region alpha of 0.0) are only actually graphed if the region range
            # includes 0.0:
            if effective_min_region == 0.0:
                rejections_string = " (rejections are also graphed)"
            else:
                rejections_string = ""
            fig.suptitle(
                f"SDM Predictive Uncertainty,\nGround-truth label = {true_label_to_graph}, "
                f"{latex_string}{alpha_value_string}{rejections_string}",
                y=0.98)
        else:
            if options.graph_centroid:
                latex_string = r'$\rm{SDM}_{\mathrm{HR}} \neq \bot, \alpha$'
            else:
                latex_string = r'$\rm{SDM}^{\mathrm{lower}}_{\mathrm{HR}} \neq \bot, \alpha$'
            fig.suptitle(
                f"SDM Predictive Uncertainty,\nGround-truth label = {true_label_to_graph}, "
                f"{latex_string}{alpha_value_string}",
                y=0.98)

    text_x = None
    text_y = None
    # The y-values of any graphed horizontal threshold lines, folded into the explicit y-limits below so that
    # the lines are not cut off when they fall outside the range of the graphed points:
    threshold_hline_y_values = []
    if options.graph_thresholds:
        latex_min_valid_qbin = r"${q'}_{\mathrm{min}}$"
        threshold_legend_entries = 0
        if region_range_is_active:
            # One (vertical q'_min, horizontal class-wise threshold) line pair per recorded region with an
            # alpha in the provided (inclusive) range, sharing a color, with one legend entry per region.
            # Regions without a finite q'_min are simply not recorded in model.hr_regions, so they are
            # naturally skipped here:
            covered_hr_regions = [hr_region for hr_region in model.hr_regions
                                  if effective_min_region <= hr_region["alpha"] <= effective_max_region]
            if len(covered_hr_regions) == 0:
                print(f"Note: No recorded high-reliability regions have an alpha within "
                      f"[{effective_min_region}, {effective_max_region}], so no threshold lines are graphed.")
            region_line_colors = ["darkblue", "darkorange", "purple", "teal", "saddlebrown", "deeppink",
                                  "olive", "slategray"]
            latex_alpha = r'$\alpha$'
            latex_psi_for_class = r'$\psi_{' + f"{true_label_to_graph}" + r'}$'
            for region_i, hr_region in enumerate(covered_hr_regions):
                line_color = region_line_colors[region_i % len(region_line_colors)]
                region_min_rescaled_similarity = float(hr_region["min_rescaled_similarity"])
                region_output_threshold = float(hr_region["output_thresholds"][true_label_to_graph])
                ax_main.axhline(y=region_output_threshold, color=line_color, linestyle=':', linewidth=1.5)
                threshold_hline_y_values.append(region_output_threshold)
                ax_main.axvline(x=region_min_rescaled_similarity, color=line_color, linestyle='--',
                                linewidth=1.5,
                                label=f"{latex_alpha}={hr_region['alpha']}: "
                                      f"{latex_min_valid_qbin}{latex_approx_symbol}"
                                      f"{region_min_rescaled_similarity:.2f}, "
                                      f"{latex_psi_for_class}{latex_approx_symbol}"
                                      f"{region_output_threshold:.4f}")
                threshold_legend_entries += 1
        else:
            # The legacy behavior at the default (None) region range: the most conservative recorded region's
            # class-wise threshold and q'_min, when at least one region exists:
            if hr_class_conditional_accuracy > 0.0:
                latex_threshold_label = r'Class-wise thresholds ($\psi$)'
                ax_main.axhline(y=hr_output_thresholds[true_label_to_graph],
                                color='orange', linestyle=':', linewidth=2,
                                label=f"{latex_threshold_label}, "
                                      f"index {true_label_to_graph}"
                                      f"{latex_approx_symbol}"
                                      f"{hr_output_thresholds[true_label_to_graph]:.4f}")
                ax_main.axvline(x=min_rescaled_similarity_to_determine_high_reliability_region,
                                color=min_rescaled_similarity_to_determine_high_reliability_region_error_color,
                                linestyle='--', linewidth=2,
                                label=f"{latex_min_valid_qbin}"
                                      f"{latex_approx_symbol}"
                                      f"{min_rescaled_similarity_to_determine_high_reliability_region:.2f}")
                threshold_hline_y_values.append(float(hr_output_thresholds[true_label_to_graph]))
                threshold_legend_entries = 2
            else:
                print(f"Note: No recorded high-reliability regions, so the threshold lines are not graphed.")
        if threshold_legend_entries > 0:
            if region_range_is_active:
                legend = ax_main.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18),
                                        ncol=1 if threshold_legend_entries <= 4 else 2, fontsize='small')
            else:
                legend = ax_main.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=1)

            # Get the legend's bounding box in figure coordinates
            fig.canvas.draw()  # Force a draw to get accurate positions
            legend_bbox = legend.get_window_extent(renderer=fig.canvas.get_renderer())
            legend_bbox_fig = legend_bbox.transformed(fig.transFigure.inverted())

            # Use the left edge of the legend for text alignment, and its bottom edge to place the figure
            # text below the legend (so a taller multi-region legend does not overlap the text):
            text_x = legend_bbox_fig.x0
            text_y = legend_bbox_fig.y0

    # These counts are to ensure the histogram axes are the same (for quick comparisons of the relative densities)
    top_counts = None
    right_counts = None
    # Create top histogram (x-axis distribution) with configurable bin width
    if len(x_filtered) > 0:
        # Get x-axis limits from the main plot
        x_min, x_max = np.min(x_filtered), np.max(x_filtered)
        # x_min, x_max = ax_main.get_xlim()

        # Create bins with configurable width
        x_bin_width = options.x_axis_histogram_width
        # Start from the floor of x_min (rounded down to nearest bin width)
        x_bin_start = np.floor(x_min / x_bin_width) * x_bin_width
        # End at the ceiling of x_max (rounded up to nearest bin width)
        x_bin_end = np.ceil(x_max / x_bin_width) * x_bin_width
        # Create bin edges
        x_bins = np.arange(x_bin_start, x_bin_end + x_bin_width, x_bin_width)

        # Separate correct and incorrect predictions
        x_correct = x_filtered[is_correct_filtered]
        x_incorrect = x_filtered[~is_correct_filtered]

        # Create histogram with properly aligned bins
        top_counts, _, _ = ax_top.hist([x_correct, x_incorrect], bins=x_bins,
                                       color=['green', 'red'], alpha=0.7,
                                       label=['Correct', 'Incorrect'], edgecolor='black', linewidth=0.5,
                                       align='mid')  # 'mid' centers bars on bin centers
        ax_top.set_ylabel('Count')
        ax_top.legend(loc='upper right', fontsize='small')
        ax_top.set_xlim(ax_main.get_xlim())
        # ax_top.set_xlim(x_min, x_max)

    # Create right histogram (y-axis distribution) with configurable bin width
    if len(y_filtered) > 0:
        # Get y-axis limits from the main plot
        y_min, y_max = np.min(y_filtered), np.max(y_filtered)
        # y_min, y_max = ax_main.get_ylim()
        if not options.graph_all_points:
            # adjust the padding on y
            y_min = min(y_min, 0.95)  # this is to avoid the right histogram from getting cut-off
            # Any graphed horizontal threshold lines must remain within the explicit limits (which would
            # otherwise override the autoscaling that accounts for the axhline positions):
            if len(threshold_hline_y_values) > 0:
                y_min = min(y_min, min(threshold_hline_y_values))
                y_max = max(y_max, max(threshold_hline_y_values))
            y_padding = (y_max - y_min) * 0.05  # 5% padding
            # Set tight limits with minimal padding
            ax_main.set_ylim(y_min - y_padding, y_max + y_padding)
        # Create bins with configurable width
        y_bin_width = options.y_axis_histogram_width
        # Start from the floor of y_min (rounded down to nearest bin width)
        y_bin_start = np.floor(y_min / y_bin_width) * y_bin_width
        # End at the ceiling of y_max (rounded up to nearest bin width)
        y_bin_end = np.ceil(y_max / y_bin_width) * y_bin_width
        # Create bin edges
        y_bins = np.arange(y_bin_start, y_bin_end + y_bin_width, y_bin_width)

        # Separate correct and incorrect predictions
        y_correct = y_filtered[is_correct_filtered]
        y_incorrect = y_filtered[~is_correct_filtered]

        # Create horizontal histogram with properly aligned bins
        right_counts, _, _ = ax_right.hist([y_correct, y_incorrect], bins=y_bins,
                                           color=['green', 'red'], alpha=0.7,
                                           orientation='horizontal', edgecolor='black', linewidth=0.5,
                                           align='mid')  # 'mid' centers bars on bin centers
        ax_right.set_xlabel('Count')
        ax_right.set_ylim(ax_main.get_ylim())
        # ax_right.set_ylim(y_min, y_max)

    # Synchronize the count axes to have the same maximum
    if options.constant_histogram_count_axis and top_counts is not None and right_counts is not None:
        # Get the maximum count from both histograms
        top_max = np.max([np.max(counts) for counts in top_counts])
        right_max = np.max([np.max(counts) for counts in right_counts])

        # Use the larger maximum for both axes
        max_count = max(top_max, right_max)
        # Set the same limits for both count axes
        ax_top.set_ylim(0, max_count * 1.05)  # Add 5% padding
        ax_right.set_xlim(0, max_count * 1.05)  # Add 5% padding

    # Remove tick labels from histogram axes that face the main plot
    plt.setp(ax_top.get_xticklabels(), visible=False)
    plt.setp(ax_right.get_yticklabels(), visible=False)
    # Remove tick marks from the bottom of the top histogram (which faces the main plot)
    ax_top.tick_params(axis='x', which='both', bottom=False)
    # Remove tick marks from the left of the right histogram (which faces the main plot)
    ax_right.tick_params(axis='y', which='both', left=False)

    # Add figure text
    if text_x is not None:
        first_text_y = text_y - 0.015
        fig.text(text_x + 0.06, first_text_y, f"Data: {options.data_label}; Model: {options.model_version_label}",
                 ha='left', va='top',
                 fontsize=9, style='italic', color='gray')
        fig.text(text_x + 0.06, first_text_y - 0.03,
                 f"(x-axis bins: width {options.x_axis_histogram_width}; "
                 f"y-axis bins: width {options.y_axis_histogram_width})",
                 ha='left', va='top',
                 fontsize=9, style='italic', color='gray')
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        fig.text(text_x + 0.06, first_text_y - 0.06, f"Generated: {timestamp}",
                 ha='left', va='top',
                 fontsize=9, style='italic', color='gray')
    else:
        fig.text(0.5, 0.19, f"Data: {options.data_label}; Model: {options.model_version_label}",
                 ha='center', va='top',
                 fontsize=9, style='italic', color='gray')
        fig.text(0.5, 0.16,
                 f"(x-axis bins: width {options.x_axis_histogram_width}; "
                 f"y-axis bins: width {options.y_axis_histogram_width})",
                 ha='center', va='top',
                 fontsize=9, style='italic', color='gray')
        # Add timestamp centered under the filename
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        fig.text(0.5, 0.13, f"Generated: {timestamp}",
                 ha='center', va='top',
                 fontsize=9, style='italic', color='gray')

    plt.subplots_adjust(left=0.12, right=0.95, bottom=0.12, top=0.92, hspace=0.02, wspace=0.02)

    filename_suffix = "lower"
    if options.graph_centroid:
        filename_suffix = "centroid"

    if options.save_file_prefix.strip() != "":
        if region_range_is_active:
            # With a provided region range, only a subset of the points is graphed (in both the
            # --graph_all_points and default cases), so the filename records the (effective) range:
            suffix_label = f"__class_label_{true_label_to_graph}_restricted_to_region_" \
                           f"min{effective_min_region}_region_max{effective_max_region}__{filename_suffix}.png"
        elif options.graph_all_points:
            suffix_label = f"__class_label_{true_label_to_graph}_all_points__{filename_suffix}.png"
        else:
            suffix_label = f"__class_label_{true_label_to_graph}_only_admitted__{filename_suffix}.png"
        plt.savefig(f'{options.save_file_prefix.strip()}{suffix_label}', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="-----[GRAPH]-----")
    parser.add_argument("--model_dir", default="",
                        help="model_dir")
    parser.add_argument("--input_file", default="",
                        help="--prediction_output_file from reexpress.py when running --eval_only")
    parser.add_argument("--class_size", default=2, type=int, help="")
    parser.add_argument("--graph_all_points", default=False, action='store_true',
                        help="If provided, all points are graphed. The default is to only graph "
                             "the points assigned to the most conservative "
                             "recorded region. If a region range is provided (see "
                             "--graph_class_and_prediction_conditional_estimates_min_region/_max_region), the "
                             "range restricts the graphed points in both cases, and the two cases then only "
                             "differ when the range includes an alpha of 0.0: with this option, the points "
                             "not assigned to any region (an assigned alpha of 0.0) are included; without it, "
                             "they are excluded.")
    parser.add_argument("--graph_centroid", default=False, action='store_true',
                        help="If provided, the y-axis is sdm(z') instead of the default sdm(z')_lower.")
    parser.add_argument("--graph_thresholds", default=False, action='store_true',
                        help="If provided, the threshold on rescaled_similarity and the class-wise thresholds are "
                             "included in the graph. With the default (unset) region range below, these are the "
                             "values of the most conservative recorded region; with a provided region range, one "
                             "line pair is graphed for each recorded region with an alpha in the range.")
    parser.add_argument("--graph_class_and_prediction_conditional_estimates_min_region", default=None, type=float,
                        help="If provided (in [0, 1]; effectively defaults to 0), only the points assigned to an "
                             "uncertainty region with alpha >= this value are graphed, based on "
                             "'hr_region_alpha' if --graph_centroid is provided, and 'hr_region_alpha_lower' "
                             "otherwise. (An assigned alpha of 0.0 indicates no region.) When a range is "
                             "provided without --graph_all_points, the points assigned to ANY recorded region "
                             "with an alpha in the range are graphed (not only those of the most conservative "
                             "region); with --graph_all_points, the points not assigned to any region are "
                             "additionally included when this value is 0.")
    parser.add_argument("--graph_class_and_prediction_conditional_estimates_max_region", default=None, type=float,
                        help="If provided (in [0, 1]; effectively defaults to 1), only the points assigned to an "
                             "uncertainty region with alpha <= this value (inclusive) are graphed, as above.")
    parser.add_argument("--emphasize_wrong_predictions", default=False, action='store_true',
                        help="If provided, the size of incorrect predictions (red points) are "
                             "enlarged for visual emphasis.")
    parser.add_argument("--data_label", default="",
                        help="This is printed at the bottom right of the graph.")
    parser.add_argument("--model_version_label", default="",
                        help="This is printed at the bottom right of the graph.")
    parser.add_argument("--constant_histogram_count_axis", default=False, action='store_true',
                        help="If this is provided, the histograms have the same visible max for the count axis.")
    parser.add_argument("--x_axis_histogram_width", default=10, type=float,
                        help="Width of histogram bins for the x-axis (default: 10)")
    parser.add_argument("--y_axis_histogram_width", default=0.05, type=float,
                        help="Width of histogram bins for the y-axis (default: 0.05)")
    parser.add_argument("--save_file_prefix", default="",
                        help="If provided, the image will be saved at this location with the suffix "
                             "'__class_label_X_only_admitted.png' or '__class_label_X_all_points.png'")

    options = parser.parse_args()

    if options.graph_all_points:
        if options.graph_class_and_prediction_conditional_estimates_min_region is not None or \
            options.graph_class_and_prediction_conditional_estimates_max_region is not None:
            print(f"This option set is disabled. To set a range, remove --graph_all_points. In the unlikely event you "
                  f"want to set min_region=0 and a max_region<1 AND want to graph the points not assigned to a region, "
                  f"comment the following `exit()`.")
            exit()

    # Set higher-resolution for saving
    plt.rcParams.update({
        # 'figure.dpi': 300,
        'savefig.dpi': 300,
        # 'savefig.bbox': 'tight',
        # 'savefig.pad_inches': 0.1
    })

    print(f"USER INSTRUCTIONS: "
          f"Click on a point in the graph to print details (including document text, if available) to the console.")
    start_time = time.time()
    model = utils_model.load_model_torch(options.model_dir, torch.device("cpu"), load_for_inference=True)

    hr_region_stats = model.get_most_conservative_high_reliability_region_stats()
    most_conservative_hr_alpha = hr_region_stats["most_conservative_hr_alpha"]
    most_conservative_hr_output_thresholds = hr_region_stats["most_conservative_hr_output_thresholds"]
    most_conservative_hr_min_rescaled_similarity = hr_region_stats["most_conservative_hr_min_rescaled_similarity"]

    print(f"Current support set cardinality (Note: May differ from that used to generate "
          f"--prediction_output_file if the model has subsequently been updated): {model.support_index.ntotal}")
    print(f"Selection constraints for most conservative region:")
    print(f"\talpha = {most_conservative_hr_alpha}")
    print(f"\tClass-wise thresholds = {most_conservative_hr_output_thresholds}")
    print(f"\tq'_min: "
          f"{most_conservative_hr_min_rescaled_similarity}")
    json_lines = utils_model.read_jsons_lines_file(options.input_file)

    for true_label_to_graph in range(options.class_size):
        graph_sdm_estimator_output(options, json_lines, true_label_to_graph=true_label_to_graph,
                                   min_rescaled_similarity_to_determine_high_reliability_region=
                                   most_conservative_hr_min_rescaled_similarity,
                                   hr_output_thresholds=most_conservative_hr_output_thresholds,
                                   hr_class_conditional_accuracy=most_conservative_hr_alpha,
                                   model=model)
    cumulative_time = time.time() - start_time
    print(f"Cumulative running time: {cumulative_time}")


if __name__ == "__main__":
    main()
