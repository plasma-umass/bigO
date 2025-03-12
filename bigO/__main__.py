import base64
import io
import json
import os
import textwrap
from typing import List, Optional

from matplotlib import gridspec, pyplot as plt
from matplotlib.figure import Figure

from bigO.analysis import (
    ABTest,
    Analysis,
    CheckBounds,
    CheckLimits,
    FunctionData,
    InferPerformance,
)
from bigO.output import log, timer, set_debug
import click
import numpy as np
import seaborn as sns

import webbrowser


system_name = "bigO"


@click.command()
@click.option(
    "--output-file",
    "output_file",
    default=None,
    help="Specify the output file to process.",
)
@click.option(
    "--debug",
    "debug",
    is_flag=True,
    help="Show debug output.",
)
@click.option(
    "--open-report",
    "open_report",
    is_flag=True,
    help="Open the output in a web browser.",
)
@click.option(
    "--html",
    "html",
    is_flag=True,
    help="Generate an HTML report instead of a PDF.",
)
def main(output_file: Optional[str], debug: bool, open_report: bool, html: bool):

    set_debug(debug)

    file_name = output_file or f"{system_name}_data.json"
    with timer("Loading data"):
        with open(file_name, "r") as f:
            data = json.load(f)

    with timer("Building work items"):
        entries = {}
        work_items: List[Analysis] = []
        for key, function_record in data.items():
            key_str = key.strip("()")
            parts = key_str.split(",")
            function_name = parts[1].strip().strip("'\"")
            file_name = parts[0].strip().strip("'\"")

            records = function_record["observations"]
            log(f"{function_name}: {len(records)} records...")
            lengths = [r["length"] for r in records]
            times = [r["time"] for r in records]
            mems = [r["memory"] for r in records]

            function_data = FunctionData(
                function_name=function_name,
                file_name=file_name,
                lengths=np.array(lengths),
                times=np.array(times),
                mems=np.array(mems),
            )

            entries[key] = function_data

        used_in_tests = set()
        for key, function_record in data.items():
            function_data = entries[key]
            if len(function_data.lengths) == 0:
                continue
            time_bound = function_record["tests"].get("time_bound", None)
            mem_bound = function_record["tests"].get("mem_bound", None)
            time_limit = function_record["tests"].get("time_limit", None)
            mem_limit = function_record["tests"].get("mem_limit", None)
            length_limit = function_record["tests"].get("length_limit", None)
            ab_test = function_record["tests"].get("abtest", None)
            if time_bound or mem_bound:
                work_items += [CheckBounds(function_data, time_bound, mem_bound)]
                used_in_tests.add(key)
            if time_limit or mem_limit:
                work_items += [
                    CheckLimits(function_data, time_limit, mem_limit, length_limit)
                ]
            elif ab_test:
                alt, metrics = ab_test
                if "time" in metrics:
                    work_items += [ABTest(function_data, entries[alt], "time")]
                if "memory" in metrics:
                    work_items += [ABTest(function_data, entries[alt], "memory")]
                used_in_tests.add(key)
                used_in_tests.add(alt)

        for key, function_record in data.items():
            if key not in used_in_tests:
                function_data = entries[key]
                if len(function_data.lengths) > 0 and all(function_data.lengths >= 0):
                    work_items += [InferPerformance(function_data)]

    if html:
        filename = run_html(work_items)
    else:
        filename = run_pdf(work_items)
    if open_report:
        webbrowser.open(f"file://{os.getcwd()}/{filename}")


def run_pdf(work_items) -> str:
    sns.set_style("whitegrid")
    sns.set_palette("tab10")
    fig = plt.figure(constrained_layout=True, figsize=(12, 4 * len(work_items)))
    gs_main = gridspec.GridSpec(nrows=len(work_items), ncols=1, figure=fig, hspace=0.1)
    gs_main.tight_layout(fig, rect=(0, 0, 1, 1))

    for index, item in enumerate(work_items):
        print()
        print(item.title())
        print("-" * len(item.title()))
        subfig = fig.add_subfigure(gs_main[index, 0])
        try:
            result = item.run()
            print(textwrap.indent(result.message, "  "))
            print()
            print("  Details:")
            print(textwrap.indent(result.details, "  "))
            print()
            if result.warnings:
                print(textwrap.indent("\n".join(["Warnings:"] + result.warnings), "  "))
                print()
            item.plot(subfig)
            subfig.suptitle(
                f"{item.title()}: {'success' if result.success else 'failed'}"
            )
        except Exception as e:
            print(f"Failed: {e}")
            print()
            subfig.suptitle(f"{item.title()}: {'exception'}")

    filename = f"{system_name}.pdf"
    plt.savefig(filename)
    print(f"{filename} written.")
    return filename


def run_html(work_items) -> str:
    # Assume these are defined elsewhere:
    # - work_items: a list of objects, each with methods .title(), .run(), and .plot(ax)
    # - system_name: a string used for naming the output file (without extension)
    # - open_plots: a boolean flag to open the result in a web browser

    # Set up the plotting style and palette.
    sns.set_style("whitegrid")
    sns.set_palette("tab10")

    # Prepare an HTML page as a list of strings.
    html_lines = [
        "<html>",
        "<head>",
        "  <meta charset='utf-8'>",
        "  <title>BigO Report</title>",
        "</head>",
        "<body>",
        "  <h1>BigO Report</h1>",
    ]

    # Process each work item separately.
    for item in work_items:
        title = item.title()
        print(f"{title}...")
        try:
            result = item.run()
            mark = (
                "<span style='color:blue;'>&#10003;</span>"
                if result.success
                else "<span style='color:red;'>&#10007;</span>"
            )
            html_lines.append(f"<h2>{mark} {title}</h2>")

            # Create a new figure and axis for this work item.
            fig_item = Figure(figsize=(12, 4))

            # Add the result message.
            html_lines.append("<pre>")
            html_lines.append(textwrap.indent(result.message, "  "))
            html_lines.append("</pre>")
            html_lines.append("<pre>")
            html_lines.append("Details:")
            print(textwrap.indent(result.details, "  "))
            html_lines.append("</pre>")

            # Let the work item create its plot on the current axis.
            item.plot(fig_item)

            # Save the figure as a PNG image into a bytes buffer.
            buf = io.BytesIO()
            fig_item.savefig(buf, format="png", bbox_inches="tight")
            buf.seek(0)
            image_base64 = base64.b64encode(buf.read()).decode("utf-8")

            # Embed the PNG image (as a Base64 data URL) in the HTML.
            html_lines.append(
                f'<img src="data:image/png;base64,{image_base64}" alt="Plot for {title}">'
            )

            # Close the figure to free memory.
            plt.close(fig_item)

            # If there are warnings, include them.
            if result.warnings:
                warnings_text = "\n".join(["Warnings:"] + result.warnings)
                html_lines.append("<pre>")
                html_lines.append(textwrap.indent(warnings_text, "  "))
                html_lines.append("</pre>")
        except Exception as e:
            print(f"Failed: {e}")
            html_lines.append(f"<pre>Failed: {e}</pre>")

    html_lines.append("</body>")
    html_lines.append("</html>")

    # Write the HTML content to a file.
    html_filename = f"{system_name}.html"
    with open(html_filename, "w", encoding="utf-8") as f:
        f.write("\n".join(html_lines))
    print(f"{html_filename} written.")
    return html_filename


if __name__ == "__main__":
    main()
