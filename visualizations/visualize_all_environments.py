"""
Generate place field visualizations for all 6 environments.
This script generates both overlay plots and detailed HTML reports.
"""

import sys
import os
from pathlib import Path

# Get the project root directory
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# Import visualization modules
import vis_utils
from multi_pcn_overlayed import plot_all_multi_scale_cells
from multi_html_report import generate_multi_scale_reports

# Configuration
ENVIRONMENTS = [
    "environment_1",
    "environment_2",
    "environment_3",
    "environment_4",
    "environment_5",
    "environment_6"
]
CONTROLLER_NAME = "multiscale_controller"
SCALES = [0, 1, 2]  # Adjust based on your scales


def visualize_environment(env_name, generate_html=True, generate_overlay=True):
    """
    Generate visualizations for a specific environment.

    Args:
        env_name: Environment name (e.g., "environment_1")
        generate_html: Whether to generate HTML reports
        generate_overlay: Whether to generate overlay plots
    """
    print(f"\n{'='*60}")
    print(f"Processing {env_name}")
    print(f"{'='*60}")

    # Set custom paths for this environment
    vis_utils.set_custom_paths(
        controller_name=CONTROLLER_NAME,
        world_name=env_name
    )

    # Also update the global variables (needed for some functions)
    vis_utils.CONTROLLER_NAME = CONTROLLER_NAME
    vis_utils.WORLD_NAME = env_name

    # Update OUTPUT_DIR
    vis_utils.OUTPUT_DIR = os.path.join(
        vis_utils.CONTROLLER_PATH_PREFIX,
        CONTROLLER_NAME,
        "pkl",
        env_name,
        "vis_outputs"
    )

    try:
        # Generate overlay plot
        if generate_overlay:
            print(f"\n--- Generating overlay plot for {env_name} ---")
            output_path = os.path.join(
                vis_utils.OUTPUT_DIR,
                f"{env_name}_multi_scale_pcn.png"
            )

            # Create output directory if it doesn't exist
            os.makedirs(vis_utils.OUTPUT_DIR, exist_ok=True)

            plot_all_multi_scale_cells(
                scales=SCALES,
                gridsize=200,
                num_cells_to_sample=10,
                show_plot=False,  # Don't block with plots
                save_path=output_path,
                plot_all_cells_too=True
            )
            print(f"Saved overlay plot to: {output_path}")

        # Generate HTML reports
        if generate_html:
            print(f"\n--- Generating HTML reports for {env_name} ---")
            html_reports = generate_multi_scale_reports(
                scales=SCALES,
                activation_threshold=0.1,
                open_browser=False  # Don't open browser for each one
            )

            for report in html_reports:
                print(f"Generated: {report}")

    except FileNotFoundError as e:
        print(f"ERROR: Missing data files for {env_name}")
        print(f"Details: {e}")
        print(f"Skipping {env_name}...")
    except Exception as e:
        print(f"ERROR processing {env_name}: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main execution function."""
    print("Starting place field visualization for all environments...")
    print(f"Controller: {CONTROLLER_NAME}")
    print(f"Environments: {', '.join(ENVIRONMENTS)}")
    print(f"Scales: {SCALES}")

    # Ask user what to generate
    print("\nWhat would you like to generate?")
    print("1. Overlay plots only")
    print("2. HTML reports only")
    print("3. Both (default)")

    choice = input("\nEnter choice (1/2/3) [3]: ").strip() or "3"

    generate_overlay = choice in ["1", "3"]
    generate_html = choice in ["2", "3"]

    # Process each environment
    for env_name in ENVIRONMENTS:
        visualize_environment(
            env_name,
            generate_html=generate_html,
            generate_overlay=generate_overlay
        )

    print("\n" + "="*60)
    print("All visualizations complete!")
    print("="*60)

    # Print summary of output locations
    print("\nOutput locations:")
    for env_name in ENVIRONMENTS:
        output_dir = os.path.join(
            vis_utils.CONTROLLER_PATH_PREFIX,
            CONTROLLER_NAME,
            "pkl",
            env_name,
            "vis_outputs"
        )
        print(f"  {env_name}: {output_dir}")


if __name__ == "__main__":
    main()
