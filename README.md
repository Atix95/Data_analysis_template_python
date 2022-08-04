# Data analysis template in Python
A Jupyter Notebook with some functions I find useful for data analysis. Since I had to do it a lot over the course of my studies and I only started using Python in the later stages, I wanted to define some functions with callables that I regularly use to avoid copy-pasting and losing track of the code. I also intended to define a few functions that would save me formatting and typing in a report.

Additionally, I wanted to learn how to annotate, document and raise exceptions properly, which is why I wrote extensive docstrings and excpetions for all of the functions. Since this was meant to be an exercise, the result might not be perfect and is definetly overkill in some instances.

I hope you may find this helpful as well.


# Functions

For examples see the docstrings in the file.

## odr_fit(fit_function, x_values, y_values, x_err, y_err, initial_guesses)

    Calculate the fit function to data with error bars in both x and y direction using 
    orthogonal distance regression. 

### Parameters
    fit_function : callable
        Fit function to be fitted to the data. fit_function(beta, x) -> y

    x_values : array_like of rank-1
        x-values that the fit function is tobe fitted to.

    y_values : array_like of rank-1
        y-values that the fit function is to be fitted to.

    x_err : array_like of rank-1 or float
        Errors of the x-values. Must have the same dimensions as x_values or be float,
        if the errors are the same for all values.

    y_err : array_like of rank-1 or float
        Errors of the y-values. Must have the same dimensions as y_values or be float,
        if the errors are the same for all values.

    initial_guesses : array_like of rank-1
        Initial guesses of the fit parameters beta. The length of the rank-1 sequence
        must be the same as the number of fit parameters.

### Returns
    output : Output instance.
        Contains fit parameters, errors, covariance matrix etc. Additionally, prints the
        fit parameters with the respective error.

### Notes
    The fit parameters must be specified first and must be passed as an array! Lastly,
    the element of the function must be specified

    The fit parameters as well as their errors can be accessed by calling the variable,
    that this function is assigned to, and adding .beta or .sd_beta.

## least_squares_fit(fit_function, x_values, y_values, y_err=None, initial_guesses=None, bounds=(-float("inf"), float("inf")), maxfev=800)

    Calculate the fit function to data with error bars in y direction using non-linear
    least squares.

### Parameters
    fit_function : callable
        Fit function to be fitted to the data. fit_function(x, ...) -> y

    x_values : array_like of rank-1
        x-values that the fit function is to be fitted to.

    y_values : array_like of rank-1
        y-values that the fit function is to be fitted to.

    y_err : array_like of rank-1 or float, optional
        Errors of the y-values. Must have the same dimensions as y_values or be float,
        if the errors are the same for all values. Default is None.

    initial_guesses : array_like of rank-1, optional
        Initial guesses of the fit parameters. The length of the rank-1 sequence
        must be the same as the number of fit parameters. Default is None

    bounds : 2-tuple of array_like, optional
        Bounds on fit parameters. Can be specified for each fit parameter as an iterable
        of the respective boundary or as a float,
        
        in which case they are the same for all fit parameters. Default is
        (-float("inf"), float("inf")), which translates to no boundaries.

    maxfev : int, optional
        The maximum number of function calls. Default is 800.

### Returns
    popt, perr : Tuple[Type[np.ndarray[float]], Type[np.ndarray[float]]]
        Fit parameters and their respective errors.

### Notes
    The independent element of the function must be specified first and then the 
    fit parameters individually!

## error_prop_latexify(func, variables)
    Calculate the error of a function according to gaussian error propagation and print
    its LaTeX expression as well as the passed function.

### Parameters

    func : str
        Function to which the error propagation is to be applied.

        It must be passed in Python syntax but without package syntax such as math.sin,
        np.sin or sp.sin.

        The functions arcsin, arccos and arctan have to be called as asin, acos and
        atan, respectively.

        For further deviating expressions [visit](https://docs.sympy.org/latest/modules/functions/elementary.html).
        

    variables : array_like of rank-1
        The variables to derive to. They must be passed in string format listed in an
        iterable such as list, tuple or set.

## update_custom_rcparams(designated_use="article", fontsize=None)
    
    Update rc parameters so that the figures blend in nicely in the document.

### Parameters

    designated_use : str, optional
        Designated use of the figure to adjust the fontsize to common default values.
        Can be either 'article' or 'powerpoint'. Default is 'article'.

    fontsize : float, optional
        Fontsize of the labels, legend, x- and y-ticklabels and title. Set this e.g. to
        the fontsize of your LaTeX document. Default is None.

## calculate_fig_dimensions(article_type="one_column", fig_width_pt=None, fig_height_pt=None, ratio=1, subplots=(1, 1), **kwargs)

    Calculate the figure dimensions so that the figure has text width in the article and
    the right font and fontsize.

### Parameters

    article_type : str, optional
        Can be set to 'one_column' or 'two_column' to set the figure width to 360pt or
        246 pt, respectively. Default is 'one_column'.

    fig_width_pt : float, optional
        Figure width in pts. Use \showthe\\textwidth or \showthe\columnwidth for a one
        or two column article in LaTeX to insert the figures with the correct size.\n
        Default is None.

    fig_height_pt : float, optional
        Figure height in pts. If no value is passed, the height is calculated from the
        width by multiplying it with the golden ratio to obatin an asthetic ratio.\n
        Default is None.

    ratio : float, optional
        Ratio of the plot. Takes values between 0 and 1. Default is 1.

    subplots : Tuple[float, float], optional
        Number of subplots that are to be plotted per line and column. They are also
        used to adjust the sizes of the subplots within the figure accordingly.\n
        Default is (1, 1).

    **kwargs : optional
        Parameters of the function update_custom_rcparams().
        For more information see the documentation of said function.

### Returns

    fig_dimensions : Tuple[float, float]
        Tuple with the width and height of the figure/subplots in inches.

## data_plot(x_values, y_values, xlabel, ylabel, x_err=None, y_err=None, marker=".", ls=None, color="C0", markersize=6, error_line_width=1.3, capsize_error_bar=5, label=None, zorder=0, grid_linewidth=0, xlim=None, ylim=None, xticks=None, yticks=None, scientific_axis="both", scilimits=(0, 0), **kwargs)

    Plot the data with error bars, if necessary. The purpose of this function is to set
    customized default values and plot dimensions that are often used.

### Parameters

    x_values : array_like of rank-1
        Values of the x-axis.

    y_values : array_like of rank-1
        Values of the y-axis.

    xlabel : str
        Label of the x-axis.

    ylabel : str
        Label of the y-axis.

    x_err : array_like of rank-1 or float, optional
        Errors of the x values. Must have the same dimensions as x_values or be float,
        if they are the same for all values. Default is None.

    y_err : array_like of rank-1 or float, optional
        Errors of the y values. Must have the same dimensions as y_values or be float,
        if they are the same for all values. Default is None.

    marker : str, optional
        How the data points should be drawn in the plot. Default is ".".

    color : str, optional
        Color of the data points and error bars. Default is "C0".

    markersize : float, optional
        Size of the data points. Default is 6.

    ls : str, optinal
        Linestyle of the data. Default is None.

    error_line_width : float, optional
        Linewidth of the errorbars. A value below 1.5 centers the errorbars correctly
        around the data points. Default is 1.3.

    capsize_error_bar : float, optional
        Size of the bars perpendicular to the errorbars. Default is 5.

    label : str, optional
        Label of the data that is to be shown in the legend. Default is None.

    zorder : float, optional
        Value that indicates in which order the contents of the plot are to be drawn.
        The lowest value is drawn first. Default is 0.

    grid_linewidth : float, optional
        Linewidth of the grid that is plotted in the background. Default is 0.

    xlim : Sequence[float, float], optional
        Tuple of the lower and upper limits of the x-axis. Default is None.

    ylim : Sequence[float, float], optional
        Tuple of the lower and upper limits of the y-axis. Default is None.

    xticks : Sequence[float], optional
        Tick locations of the x-axis. To create evenly spaced floats, best practice is
        to use np.arange(). Default is None.

    yticks : Sequence[float], optional
        Tick locations of the y-axis. To create evenly spaced floats, best practice is
        to use np.arange(). Default is None.

    scientific_axis : str, optional
        Which axis should be used for scientific notations. Default is "both".

    scilimits : Tuple[int, int], optional
        Tuple of two integers (m, n); scientific notation will be used for numbers
        outside the range 10**m to 10**n. If (0, 0) is used, all numbers are included.

        Default is (0, 0).

    **kwargs : optional
        Parameters of the function calculate_fig_dimensions().
        For more information see the documentation of said function.

## show_the_legend_with_ordered_labels(label_order=None, framealpha=1, loc="best", legend_fontsize=None)

    Show the labels of the plot in the desired order.

### Parameters

    label_order : Sequence[int], optional
        Sequence of order of the labels starting with 0. len(label_order) must be the
        same as the number of labels. Default is None.
    
    framealpha_legend : float, optional
        Transparency of the legend. Value of 0 means full and 1 no transparency. Default
        is 1.

    loc : str or Tuple[float, float], optional
        Location of the legend. Can also be be a 2-tuple giving the coordinates of the
        lower-left corner of the legend in axes coordinates. Default is "best".

    legend_fontsize : float, optional
        Fontsize of the text shown in the legend. If None is passed, the fontsize is
        read out from the rc parameters and set to that value. Default is None.

## save_figure(fname, ftype)
    Save the figure that was created with matplotlib.
    
    If the folder "images", where the figures are stored, does not yet exist, it is
    created first in the dir of this file.

### Parameters

    fname : str
        Filename of the figure to be saved.

    ftype : str, optional
        Filetype of the figure to be saved. Options are e.g. "pdf", "png", "svg", etc.
        Default is "pdf".

## add_label_to_legend(variable_name_or_text, variable_value=None, varibale_err=None, precision=0, magnitude=0, units=None)

    Create labels for variables with their respective values and erros that are to be
    shown in the legend of the plot.

    If None is passed to the value and its error and the units, the label entirely
    consists of the text that is passed to the variable name.

### Parameters

    variable_name_or_text : str
        Name of the variable or text that is to be shown in the legend. 

    variable_value : float or int, optional
        Value of the variable. Default is None.

    variable_err : float or int, optional
        Uncertainty of the value. Default is None.

    precision : int, optional
        Up to what decimal place the data is to be shown. Default is 0.

    magnitude : int, optional
        Order of magnitude with which the value is to be represented. Default is 0.

    units : str, optional
        Units of the value. Default is None.
### Notes

    Text rendering with LaTeX has to be active. For this simply use:
    plt.rcParams.update({"text.usetex": True})
