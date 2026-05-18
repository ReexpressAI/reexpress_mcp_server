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
                print(f"\n{'+' * 14}")
                print(f"Label == Prediction: {self.data_rows[idx]['label'] == self.data_rows[idx]['prediction']}")
                print(f"Label: {self.data_rows[idx]['label']}")
                print(f"Prediction: {self.data_rows[idx]['prediction']}")
                print(
                    f"{constants.CALIBRATION_HIGH_RELIABILITY_REGION_LOWER_LABEL_FULL}: "
                    f"{self.data_rows[idx]['is_high_reliability_region_lower']}")
                print(f"p(y|x)_lower: {self.data_rows[idx]['sdm_output_d_lower']}")
                print(f"{'+' * 14}\n")
                print(f"p(y|x): {self.data_rows[idx]['sdm_output']}")
                print(f"{constants.CALIBRATION_HIGH_RELIABILITY_REGION_LABEL_FULL}: "
                      f"{self.data_rows[idx]['is_high_reliability_region']}")
                print(f"Rescaled Similarity, lower (q'_lower): {self.data_rows[idx]['rescaled_similarity_lower']}")
                print(f"Rescaled Similarity (q'): {self.data_rows[idx]['rescaled_similarity']}")
                print(f"OOD: {self.data_rows[idx]['is_ood']}")
                print(f"Effective sample size: {self.data_rows[idx]['cumulative_effective_sample_sizes']}")
                separator_text = ", "
                print(f"Similarity (q): {self.data_rows[idx]['q']}{separator_text} "
                      f"Distance quantile, lower (d_lower): {self.data_rows[idx]['d_lower']}{separator_text} "
                      f"Distance quantile (d): {self.data_rows[idx]['d']}{separator_text} "
                      f"Magnitude (f): {self.data_rows[idx]['f']}")
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

        if options.graph_all_points:
            filter_condition = True
        else:
            if options.graph_centroid:
                filter_condition = document["is_high_reliability_region"]
            else:
                filter_condition = document["is_high_reliability_region_lower"]
        label = document["label"]
        if true_label_to_graph is not None:
            filter_condition = filter_condition and label == true_label_to_graph
            # can use these to verify the bin distribution alignment, which must match with the x and y-axis:
            # and 0.0 <= rescaled_similarity <= 2.5
            # and 0.7 <= prediction_probability <= 0.8
        if label == true_label_to_graph:
            accuracy_class_conditional.append(document["prediction"] == label)
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

    print(f"Marginal accuracy: {np.mean(accuracy)} out of {len(accuracy)}")
    total_points = len(accuracy)
    if options.graph_all_points:
        print(f"Class-conditional (label={true_label_to_graph}) accuracy: "
              f"{np.mean(accuracy_filtered)} out of {len(accuracy_filtered)} "
              f"({len(accuracy_filtered)/total_points if total_points > 0 else 0.0} of all points)")
    else:
        print(
            f"Class-conditional (label={true_label_to_graph}) accuracy: "
            f"{np.mean(accuracy_class_conditional)} out of {len(accuracy_class_conditional)} "
            f"({len(accuracy_class_conditional)/total_points if total_points > 0 else 0.0} of all points)")
        if options.graph_centroid:
            print(
                f"Class-conditional (label={true_label_to_graph}) accuracy, SDM_HR: "
                f"{np.mean(accuracy_filtered)} out of {len(accuracy_filtered)} "
                f"({len(accuracy_filtered)/total_points if total_points > 0 else 0.0} of all points)")
        else:
            print(
                f"Class-conditional (label={true_label_to_graph}) accuracy, SDM_HR (lower): "
                f"{np.mean(accuracy_filtered)} out of {len(accuracy_filtered)} "
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
        if options.graph_all_points:
            latex_string = r'$\alpha$'
            fig.suptitle(
                f"SDM Predictive Uncertainty,\nGround-truth label = {true_label_to_graph}, "
                f"{latex_string}={hr_class_conditional_accuracy} (rejections are also graphed)",
                y=0.98)
        else:
            if options.graph_centroid:
                latex_string = r'$\rm{SDM}_{\mathrm{HR}} \neq \bot, \alpha$'
            else:
                latex_string = r'$\rm{SDM}^{\mathrm{lower}}_{\mathrm{HR}} \neq \bot, \alpha$'
            fig.suptitle(
                f"SDM Predictive Uncertainty,\nGround-truth label = {true_label_to_graph}, "
                f"{latex_string}={hr_class_conditional_accuracy}",
                y=0.98)
        if options.graph_thresholds:
            latex_threshold_label = r'Class-wise thresholds ($\psi$)'
            output_thresholds_hline = \
                ax_main.axhline(y=hr_output_thresholds[true_label_to_graph],
                                color='orange', linestyle=':', linewidth=2,
                                label=f"{latex_threshold_label}, "
                                      f"index {true_label_to_graph}"
                                      f"{latex_approx_symbol}"
                                      f"{hr_output_thresholds[true_label_to_graph]:.4f}")

    if options.graph_thresholds:
        latex_min_valid_qbin = r"${q'}_{\mathrm{min}}$"
        model_level_q_threshold_line = \
            ax_main.axvline(x=min_rescaled_similarity_to_determine_high_reliability_region,
                            color=min_rescaled_similarity_to_determine_high_reliability_region_error_color,
                            linestyle='--', linewidth=2,
                            label=f"{latex_min_valid_qbin}"
                                  f"{latex_approx_symbol}"
                                  f"{min_rescaled_similarity_to_determine_high_reliability_region:.2f}")
        # Fixed: Center the legend horizontally by using loc='upper center' with bbox_to_anchor at x=0.5
        # ax_main.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=1)

        # After creating the legend (line 363)
        legend = ax_main.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=1)

        # Get the legend's bounding box in figure coordinates
        fig.canvas.draw()  # Force a draw to get accurate positions
        legend_bbox = legend.get_window_extent(renderer=fig.canvas.get_renderer())
        legend_bbox_fig = legend_bbox.transformed(fig.transFigure.inverted())

        # Use the left edge of the legend for text alignment
        text_x = legend_bbox_fig.x0
    else:
        text_x = None

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
        fig.text(text_x + 0.06, 0.09, f"Data: {options.data_label}; Model: {options.model_version_label}",
                 ha='left', va='top',
                 fontsize=9, style='italic', color='gray')
        fig.text(text_x + 0.06, 0.06,
                 f"(x-axis bins: width {options.x_axis_histogram_width}; "
                 f"y-axis bins: width {options.y_axis_histogram_width})",
                 ha='left', va='top',
                 fontsize=9, style='italic', color='gray')
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        fig.text(text_x + 0.06, 0.03, f"Generated: {timestamp}",
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

    if options.save_file_prefix.strip() != "":
        if options.graph_all_points:
            suffix_label = f"__class_label_{true_label_to_graph}_all_points.png"
        else:
            suffix_label = f"__class_label_{true_label_to_graph}_only_admitted.png"
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
                        help="If provided, all points are graphed. "
                             "The default is to only graph the valid index-conditional points.")
    parser.add_argument("--graph_centroid", default=False, action='store_true',
                        help="If provided, the y-axis is sdm(z') instead of the default sdm(z')_lower.")
    parser.add_argument("--graph_thresholds", default=False, action='store_true',
                        help="If provided, the threshold on rescaled_similarity and the class-wise thresholds are "
                             "included in the graph.")
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
    min_rescaled_similarity_to_determine_high_reliability_region = \
        model.min_rescaled_similarity_to_determine_high_reliability_region
    hr_output_thresholds = model.hr_output_thresholds.detach().cpu().tolist()
    hr_class_conditional_accuracy = model.hr_class_conditional_accuracy

    print(f"Current support set cardinality (Note: May differ from that used to generate "
          f"--prediction_output_file if the model has subsequently been updated): {model.support_index.ntotal}")
    print(f"alpha = {hr_class_conditional_accuracy}")
    print(f"thresholds = {hr_output_thresholds}")
    print(f"q'_min: "
          f"{min_rescaled_similarity_to_determine_high_reliability_region}")
    json_lines = utils_model.read_jsons_lines_file(options.input_file)

    for true_label_to_graph in range(options.class_size):
        graph_sdm_estimator_output(options, json_lines, true_label_to_graph=true_label_to_graph,
                                   min_rescaled_similarity_to_determine_high_reliability_region=
                                   min_rescaled_similarity_to_determine_high_reliability_region,
                                   hr_output_thresholds=hr_output_thresholds,
                                   hr_class_conditional_accuracy=hr_class_conditional_accuracy,
                                   model=model)
    cumulative_time = time.time() - start_time
    print(f"Cumulative running time: {cumulative_time}")


if __name__ == "__main__":
    main()
