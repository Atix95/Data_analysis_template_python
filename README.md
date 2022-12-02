# Scientific data analysis library in Python

A library with some functions I find useful for data analysis. Since I had to do it
a lot over the course of my studies and I only started using Python in the later
stages, I wanted to define some functions with callables that I regularly use to
avoid copy-pasting and losing track of the code. I also intended to define a few
functions that would save formatting and typing in a report.

The Library is also available as a Jupyter Notebook to facilitate possible
intentional change to the functions.

I hope you may find this helpful as well.

## Installation

    pip install git+https://github.com/Atix95/scientific_data_analysis_library.git

## Usage

    import scientific_data_analysis_library as sci_lib

## Library

For extensive documentation and examples see the doc strings of the respective
function.

### error_prop_latexify(func, variables)

Calculate the error of a function according to Gaussian error propagation and print
its LaTeX expression as well as the passed function.

### mean_error(values)

Calculate the statistical uncertainty of a mean value according to the "Guide to the
Expression of Uncertainty in Measurement" (GUM).

It is assumed that the values are normally distributed and that the interval -t to
+t encompasses 68.27% of the distribution.

Therefore, the degree of freedom is determined by n - 1, where n is the number of
independent observations.

For further information visit:
<https://www.bipm.org/en/committees/jc/jcgm/publications>

### combined_mean_error(values, errors)

Calculate the combined error of a mean value from its statistical uncertainty and
the mean of the individual errors.

All individual errors are weighted the same.

### odr_fit(fit_function, x_values, y_values, x_err, y_err, initial_guesses)

Calculate the fit function to data with error bars in both x and y direction using
orthogonal distance regression and return the results.

### least_squares_fit(fit_function, x_values, y_values, y_err=None, initial_guesses=None, bounds=(-float("inf"), float("inf")), maxfev=800)

Calculate the fit function to data with error bars in y direction using non-linear
least squares and return the results.

### update_custom_rcparams(designated_use="article", fontsize=None)

Update RC parameters so that the figures blend in nicely in the document.

### calculate_fig_dimensions(article_type="one_column", fig_width_pt=None, fig_height_pt=None, ratio=1, subplots=(1, 1), **kwargs)

Calculate the figure dimensions so that the figure has text width in the article and
the right font and font size.

### data_plot(x_values, y_values, xlabel, ylabel, x_err=None, y_err=None, marker=".", ls=None, color="C0", markersize=6, error_line_width=1.3, capsize_error_bar=5, label=None, zorder=0, grid_linewidth=0, xlim=None, ylim=None, xticks=None, yticks=None, scientific_axis="both", scilimits=(0, 0), comma_separation=False, **kwargs)

Plot the data with error bars, if necessary. The purpose of this function is to set
customized default values and plot dimensions that are often used.

### add_label_to_legend(variable_name_or_text, variable_value=None, varibale_err=None, precision=0, magnitude=0, units=None)

Create labels for variables with their respective values and errors that are to be
shown in the legend of the plot.

If None is passed to the value, its error and the units, the label entirely
consists of the text that is passed to the variable name.

### show_the_legend_with_ordered_labels(label_order=None, framealpha=1, loc="best", legend_fontsize=None, legend_line_width=2)

Show the labels of the plot in the desired order and show a uniform line width in
the legend next to the labels.

### save_figure(fname, ftype)

Save the figure that was created with Matplotlib.

If the folder "images", where the figures are stored, does not yet exist, it is
created first in the directory of this file.

## Note

I wanted to learn how to annotate, document and raise exceptions properly, which is
why I wrote extensive doc strings and exceptions for all the functions. Since this
was meant to be an exercise, the result might not be perfect and is definitely
overkill in some instances.

## License

This project is licensed under the terms of "The Unlicense" license.
